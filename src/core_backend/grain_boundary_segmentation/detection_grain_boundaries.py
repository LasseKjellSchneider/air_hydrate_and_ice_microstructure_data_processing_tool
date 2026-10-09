import cv2
import numpy as np
from pathlib import Path
from skimage.filters import meijering
from skimage.morphology import skeletonize
import scipy.ndimage as ndi
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed
import os

# Enable GPU acceleration when CuPy and a compatible device are available.
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndi
    HAS_GPU = cp.cuda.is_available()
except Exception:
    HAS_GPU = False

from .constants_manager import load_constants
from .logger_config import DebugProfiler

def _cupy_meijering(image_gpu, sigmas, black_ridges=True, alpha=1.0/3.0):
    """
    Pure CuPy CUDA implementation of the Meijering neuriteness / ridge filter for 2D images.
    Matches skimage.filters.meijering behavior 1:1 on GPU.
    """
    filtered_max = cp.zeros_like(image_gpu, dtype=cp.float32)
    
    for sigma in sigmas:
        hxx = cp_ndi.gaussian_filter(image_gpu, sigma=sigma, order=(2, 0), mode='reflect') * (sigma ** 2)
        hyy = cp_ndi.gaussian_filter(image_gpu, sigma=sigma, order=(0, 2), mode='reflect') * (sigma ** 2)
        hxy = cp_ndi.gaussian_filter(image_gpu, sigma=sigma, order=(1, 1), mode='reflect') * (sigma ** 2)
        
        trace = hxx + hyy
        hxx_mod = hxx - alpha * trace
        hyy_mod = hyy - alpha * trace
        
        trace_mod = hxx_mod + hyy_mod
        diff_sq = (hxx_mod - hyy_mod) ** 2
        det_part = cp.sqrt(diff_sq + 4.0 * (hxy ** 2))
        
        lambda1 = (trace_mod + det_part) / 2.0
        lambda2 = (trace_mod - det_part) / 2.0
        
        if black_ridges:
            response = cp.maximum(lambda1, lambda2)
        else:
            response = -cp.minimum(lambda1, lambda2)
            
        response = cp.maximum(response, 0.0)
        filtered_max = cp.maximum(filtered_max, response)
        
    return filtered_max


def _save_debug_img(img, filename, intermediates_dir):
    """
    Safely saves intermediate pipeline steps.
    Dynamically downscales images larger than 25 Megapixels to prevent RAM spikes.
    """
    if intermediates_dir is None:
        return
        
    out_path = Path(intermediates_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = img.shape[:2]
    max_pixels = 25_000_000
    
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        
        is_mask = len(np.unique(img)) <= 2
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        
        img_scaled = cv2.resize(img.astype(np.uint8), (new_w, new_h), interpolation=interp)
        cv2.imwrite(str(out_path), img_scaled)
    else:
        cv2.imwrite(str(out_path), img.astype(np.uint8))


def detect_grain_boundaries(image, show, border_mask, intermediates_dir=None, debug_dir=None, parameters=None, return_intermediates=False):
    if parameters is None:
        parameters = load_constants()
    
    profiler = DebugProfiler(None)
    intermediates = {}  
    
    # 1. Denoising
    t, r = profiler.log_step("1.1 Denoising")
    image_denoised = denoising(
        image, 
        d=parameters["denoise_d"], 
        sigma_color=parameters["denoise_sigma_color"], 
        sigma_space=parameters["denoise_sigma_space"]
    )
    _save_debug_img(image_denoised, "02_denoised.png", intermediates_dir)
    if return_intermediates: intermediates["1. Denoised"] = image_denoised
    profiler.log_step("Finished 1.1 Denoising", t, r)

    # 2. Ridge Detection
    t, r = profiler.log_step("1.2 Ridge Detection")
    binary = ridge_detection(
        image_denoised, 
        sigmas=parameters["ridge_sigmas"], 
        black_ridges=parameters["ridge_black_ridges"], 
        threshold=parameters["ridge_threshold"],
        border_mask=border_mask,
        profiler=profiler
    )
    _save_debug_img(binary, "03_ridge_detection.png", intermediates_dir)
    if return_intermediates: intermediates["2. Ridge Filter"] = binary
    del image_denoised
    profiler.log_step("Finished 1.2 Ridge Detection", t, r)

    # 3. Border Masking
    t, r = profiler.log_step("1.3 Border Masking")
    binary_borders_off = np.where(border_mask > 0, binary, 0).astype(np.uint8)
    del binary
    profiler.log_step("Finished 1.3 Border Masking", t, r)

    # 4. Joining Lines
    t, r = profiler.log_step("1.4 Joining Lines")
    binary_joined = joining_lines(
        binary_borders_off, 
        kernel_size=parameters["ridge_dilation_kernel_size_bevor_filtering"],
        iterations=parameters["ridge_dilation_iterations"]
    )
    _save_debug_img(binary_joined, "05_joined_dilated.png", intermediates_dir)
    if return_intermediates: intermediates["3. Dilated (Joined)"] = binary_joined
    del binary_borders_off
    profiler.log_step("Finished 1.4 Joining Lines", t, r)

    # 5. Filling Holes
    t, r = profiler.log_step("1.5 Filling Holes")
    filled_bulked_lines = fill_small_holes(
        binary_joined, 
        max_area=parameters["max_hole_area_px"],
        profiler=profiler
    )
    if return_intermediates: intermediates["4. Holes Filled"] = filled_bulked_lines
    del binary_joined
    profiler.log_step("Finished 1.5 Filling Holes", t, r)

    # 6. Skeletonization
    t, r = profiler.log_step("1.6 Skeletonization")
    binary_skeletonized = chunked_skeletonize_parallel(filled_bulked_lines, profiler=profiler)
    binary_skeletonized = (binary_skeletonized > 0).astype(np.uint8)
    if return_intermediates: intermediates["5. First Skeleton"] = binary_skeletonized * 255
    del filled_bulked_lines
    profiler.log_step("Finished 1.6 Skeletonization", t, r)

    # 7. Pre-Filter Junction Splitting
    t, r = profiler.log_step("1.7 Pre-Filter Junction Splitting")
    binary_splitted, _ = split_junctions(
        binary_skeletonized, 
        min_neighbors=parameters["min_junction_neighbors"]
    )
    profiler.log_step("Finished 1.7 Pre-Filter Junction Splitting", t, r)

    # 8. Filtering Lines
    t, r = profiler.log_step("1.8 Filtering Lines")
    binary_skeletonized_filtered = filter_lines(
        binary_splitted, 
        binary_skeletonized, 
        min_length=parameters["min_length"],
        profiler=profiler
    )
    if return_intermediates: intermediates["6. Filtered Skeleton"] = binary_skeletonized_filtered * 255
    del binary_splitted
    profiler.log_step("Finished 1.8 Filtering Lines", t, r)

    # 9. Post-Filter Junction Splitting
    t, r = profiler.log_step("1.9 Post-Filter Junction Splitting")
    splitted, junctions = split_junctions(
        binary_skeletonized_filtered,
        min_neighbors=parameters["min_junction_neighbors"]
    )
    if return_intermediates: intermediates["7. Final Objects (Split)"] = splitted * 255
    profiler.log_step("Finished 1.9 Post-Filter Junction Splitting", t, r)

    # 10. Extracting Objects
    t, r = profiler.log_step("1.10 Extracting Objects")
    objects = extract_objects(
        splitted, 
        min_object_size=parameters["min_object_size_px"],
        profiler=profiler
    )
    profiler.log_step("Finished 1.10 Extracting Objects", t, r)
    
    if profiler.log_file:
        profiler.log_file.close()

    if return_intermediates:
        return objects, junctions, intermediates
    return objects, junctions


def joining_lines(binary_img, kernel_size, iterations):
    # Normalize the parameters and return the input mask if they are invalid.
    try:
        k_size = int(float(kernel_size)) if kernel_size is not None else 0
        iters = int(float(iterations)) if iterations is not None else 0
    except (ValueError, TypeError):
        return (binary_img > 0).astype(np.uint8) * 255

    # Skip dilation when either parameter disables it.
    if k_size <= 0 or iters <= 0:
        return (binary_img > 0).astype(np.uint8) * 255

    # Ensure the structuring element has a valid positive size.
    safe_k = max(1, k_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_k, safe_k))

    h, w = binary_img.shape
    tile_size = 4000
    margin = safe_k * iters
    out = np.zeros_like(binary_img)

    # Process overlapping tiles to bound memory use on large images.
    def _dilate_tile(coords):
        y, x = coords
        y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
        x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
        
        tile = binary_img[y0:y1, x0:x1]
        
        # Avoid running the dilation kernel on empty tiles.
        if not np.any(tile):
            return (slice(y, min(h, y + tile_size)), slice(x, min(w, x + tile_size))), None

        processed_tile = cv2.dilate(tile, kernel, iterations=iters)
        
        sy0, sy1 = y - y0, y - y0 + min(h - y, tile_size)
        sx0, sx1 = x - x0, x - x0 + min(w - x, tile_size)
        return (slice(y, min(h, y + tile_size)), slice(x, min(w, x + tile_size))), processed_tile[sy0:sy1, sx0:sx1]

    tiles = [(y, x) for y in range(0, h, tile_size) for x in range(0, w, tile_size)]
    optimal_workers = max(4, os.cpu_count() - 2)
    
    results = Parallel(n_jobs=optimal_workers, backend="threading")(
        delayed(_dilate_tile)(coords) for coords in tiles
    )

    for out_slice, inner_tile in results:
        if inner_tile is not None:
            out[out_slice] = inner_tile

    return (out > 0).astype(np.uint8) * 255


def denoising(image, d, sigma_color, sigma_space):
    h, w = image.shape[:2]
    
    tile_size = 4000
    margin = d * 3 
    out = np.zeros_like(image)
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
            
            tile = image[y0:y1, x0:x1]
            filtered_tile = cv2.bilateralFilter(tile, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            
            out_y0, out_y1 = y, min(h, y + tile_size)
            out_x0, out_x1 = x, min(w, x + tile_size)
            sy0, sy1 = out_y0 - y0, out_y0 - y0 + (out_y1 - out_y0)
            sx0, sx1 = out_x0 - x0, out_x0 - x0 + (out_x1 - out_x0)
            
            out[out_y0:out_y1, out_x0:out_x1] = filtered_tile[sy0:sy1, sx0:sx1]
            
    return out


def _process_meijering_tile(args):
    y, x, y0, y1, x0, x1, image, sigmas, black_ridges, r_min, r_max, threshold = args
    
    tile = image[y0:y1, x0:x1].astype(np.float32) / 255.0
    tile_ridges = meijering(tile, sigmas=sigmas, black_ridges=black_ridges)
    
    if r_max > r_min:
        tile_norm = np.clip((tile_ridges - r_min) / (r_max - r_min) * 255.0, 0, 255)
    else:
        tile_norm = np.zeros_like(tile_ridges)
        
    # Accept thresholds expressed either in normalized units or on an 8-bit scale.
    actual_thresh = threshold * 255.0 if threshold <= 1.0 else threshold
    tile_binary = (tile_norm > actual_thresh).astype(np.uint8) * 255
    return y, x, tile_binary, y0, y1, x0, x1


def ridge_detection(image, sigmas, black_ridges, threshold, border_mask, profiler=None):
    h, w = image.shape
    
    if profiler: profiler.log_msg("      -> Calculating robust percentiles via subsampling...")
    step = max(1, h // 1000)
    sub_img = image[::step, ::step].astype(np.float32) / 255.0
    sub_mask = border_mask[::step, ::step]
    
    sub_ridges = meijering(sub_img, sigmas=sigmas, black_ridges=black_ridges)
    valid_ridges = sub_ridges[sub_mask > 0]
    
    if len(valid_ridges) > 0:
        r_min = np.percentile(valid_ridges, 0.1)
        r_max = np.percentile(valid_ridges, 99.9)
    else:
        r_min, r_max = 0.0, 1.0

    tile_size = 4000
    margin = int(max(sigmas) * 4) + 5
    out_binary = np.zeros_like(image, dtype=np.uint8)
    
    # Apply the same threshold convention as the CPU path.
    actual_thresh = threshold * 255.0 if threshold <= 1.0 else threshold
    
    if HAS_GPU:
        if profiler: profiler.log_msg("      -> Applying PARALLEL Ridge Detection (NVIDIA GPU via CuPy CUDA)...")
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
                x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
                
                cp_tile = cp.asarray(image[y0:y1, x0:x1].astype(np.float32) / 255.0)
                cp_ridges = _cupy_meijering(cp_tile, sigmas=sigmas, black_ridges=black_ridges)
                
                if r_max > r_min:
                    cp_norm = cp.clip((cp_ridges - r_min) / (r_max - r_min) * 255.0, 0, 255)
                else:
                    cp_norm = cp.zeros_like(cp_ridges)
                    
                cp_binary = (cp_norm > actual_thresh).astype(cp.uint8) * 255
                tile_binary = cp.asnumpy(cp_binary)
                
                out_y0, out_y1 = y, min(h, y + tile_size)
                out_x0, out_x1 = x, min(w, x + tile_size)
                sy0, sy1 = out_y0 - y0, out_y0 - y0 + (out_y1 - out_y0)
                sx0, sx1 = out_x0 - x0, out_x0 - x0 + (out_x1 - out_x0)
                out_binary[out_y0:out_y1, out_x0:out_x1] = tile_binary[sy0:sy1, sx0:sx1]
                
        return out_binary

    tasks = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
            tasks.append((y, x, y0, y1, x0, x1, image, sigmas, black_ridges, r_min, r_max, threshold))

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    if profiler: profiler.log_msg(f"      -> Applying PARALLEL Ridge Detection ({max_workers} threads, {len(tasks)} tiles)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_meijering_tile, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            y, x, tile_binary, y0, y1, x0, x1 = future.result()
            
            out_y0, out_y1 = y, min(h, y + tile_size)
            out_x0, out_x1 = x, min(w, x + tile_size)
            sy0, sy1 = out_y0 - y0, out_y0 - y0 + (out_y1 - out_y0)
            sx0, sx1 = out_x0 - x0, out_x0 - x0 + (out_x1 - out_x0)
            
            out_binary[out_y0:out_y1, out_x0:out_x1] = tile_binary[sy0:sy1, sx0:sx1]
            
    return out_binary


# Parallel tiled skeletonization
def chunked_skeletonize_parallel(binary_img, profiler=None):
    if profiler: profiler.log_msg("      -> Running multi-core skeletonization (Chunked 22-Core)...")
    
    h, w = binary_img.shape
    tile_size = 4000
    margin = 300  # Overlap reduces discontinuities at tile boundaries.
    
    out = np.zeros_like(binary_img, dtype=np.uint8)
    tiles = [(y, x) for y in range(0, h, tile_size) for x in range(0, w, tile_size)]

    def _process_tile(coords):
        y, x = coords
        y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
        x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
        
        tile = binary_img[y0:y1, x0:x1] > 0
        if not np.any(tile):
            return (slice(y, min(h, y + tile_size)), slice(x, min(w, x + tile_size))), None
            
        skel_tile = skeletonize(tile).astype(np.uint8)
        
        sy0, sy1 = y - y0, y - y0 + min(h - y, tile_size)
        sx0, sx1 = x - x0, x - x0 + min(w - x, tile_size)
        
        return (slice(y, min(h, y + tile_size)), slice(x, min(w, x + tile_size))), skel_tile[sy0:sy1, sx0:sx1]

    optimal_workers = max(4, os.cpu_count() - 2)
    results = Parallel(n_jobs=optimal_workers, backend="loky")(
        delayed(_process_tile)(coords) for coords in tiles
    )

    for out_slice, inner_tile in results:
        if inner_tile is not None:
            out[out_slice] = inner_tile

    return out


# Tiled hole filling
def fill_small_holes(binary_img, max_area, profiler=None):
    if max_area <= 0: return binary_img
    if profiler: profiler.log_msg("      -> Filling small isolated holes (Chunked & Limit-Safe)...")
    
    h, w = binary_img.shape
    tile_size = 4000
    margin = 250  # Overlap keeps holes near tile edges within the processing window.
    out = np.zeros_like(binary_img)
    tiles = [(y, x) for y in range(0, h, tile_size) for x in range(0, w, tile_size)]

    def _process_hole_tile(coords):
        y, x = coords
        y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
        x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
        
        tile = (binary_img[y0:y1, x0:x1] > 0).astype(np.uint8)
        if not np.any(tile):
            return (slice(y, min(h, y + tile_size)), slice(x, min(w, x + tile_size))), None
            
        inverted = 1 - tile
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)
        
        valid_labels = (stats[:, cv2.CC_STAT_AREA] <= max_area)
        
        th, tw = tile.shape
        border_labels = set(np.unique(labels[0, :]))
        border_labels.update(np.unique(labels[th - 1, :]))
        border_labels.update(np.unique(labels[:, 0]))
        border_labels.update(np.unique(labels[:, tw - 1]))

        for bl in border_labels: valid_labels[bl] = False
        valid_labels[0] = False 
        
        tile[valid_labels[labels]] = 1
        
        sy0, sy1 = y - y0, y - y0 + min(h - y, tile_size)
        sx0, sx1 = x - x0, x - x0 + min(w - x, tile_size)
        return (slice(y, min(h, y + tile_size)), slice(x, min(w, x + tile_size))), tile[sy0:sy1, sx0:sx1] * 255

    optimal_workers = max(4, os.cpu_count() - 2)
    results = Parallel(n_jobs=optimal_workers, backend="threading")(
        delayed(_process_hole_tile)(coords) for coords in tiles
    )

    for out_slice, inner_tile in results:
        if inner_tile is not None:
            out[out_slice] = inner_tile

    return out


# Filter short or isolated skeleton components
def filter_lines(binary_splitted, binary_skeletonized, min_length, absolute_min_length=10, profiler=None):
    if profiler: profiler.log_msg("      -> Filtering out isolated noise (64-Bit SciPy + Vectorized)...")
    
    labels = np.zeros_like(binary_splitted, dtype=np.uint32)
    structure = np.ones((3,3), dtype=bool)
    num_labels = ndi.label(binary_splitted, output=labels, structure=structure)

    areas = np.bincount(labels.ravel())
    slices = ndi.find_objects(labels, max_label=num_labels)

    valid_indices = [
        i for i, slc in enumerate(slices) 
        if slc is not None and areas[i + 1] >= absolute_min_length
    ]

    clean_lines_mask = np.zeros_like(binary_splitted, dtype=np.uint8)

    def _process_slice_batch(idx_batch):
        updates = []
        kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        
        for i in idx_batch:
            label_id = i + 1
            area = areas[label_id]
            slc = slices[i]
            y_slice, x_slice = slc

            local_mask = (labels[y_slice, x_slice] == label_id).astype(np.uint8)

            if area >= min_length:
                updates.append((y_slice, x_slice, local_mask))
                continue

            neighbor_count = cv2.filter2D(local_mask, -1, kernel_neighbors, borderType=cv2.BORDER_CONSTANT)
            endpoints_mask = (local_mask == 1) & (neighbor_count <= 1)

            if not np.any(endpoints_mask): 
                continue
                
            ey_idx, ex_idx = np.where(endpoints_mask)
            is_dead_end = False
            
            local_skel = binary_skeletonized[y_slice, x_slice]
            other_white_pixels = (local_skel > 0) & (local_mask == 0)
            bh, bw = local_mask.shape

            for ey, ex in zip(ey_idx, ex_idx):
                ymin, ymax = max(0, ey - 3), min(bh, ey + 4)
                xmin, xmax = max(0, ex - 3), min(bw, ex + 4)
                
                window = other_white_pixels[ymin:ymax, xmin:xmax]
                if np.any(window):
                    wy, wx = np.where(window)
                    dists_sq = (wy + ymin - ey)**2 + (wx + xmin - ex)**2
                    if np.any(dists_sq <= 9.0):
                        continue
                
                is_dead_end = True
                break

            if not is_dead_end:
                updates.append((y_slice, x_slice, local_mask))

        return updates

    optimal_workers = max(4, os.cpu_count() - 2)
    batch_size = max(1, len(valid_indices) // (optimal_workers * 4))
    batches = [valid_indices[i:i + batch_size] for i in range(0, len(valid_indices), batch_size)]

    results = Parallel(n_jobs=optimal_workers, backend="threading")(
        delayed(_process_slice_batch)(b) for b in batches
    )

    for batch_updates in results:
        for y_slc, x_slc, local_mask in batch_updates:
            clean_lines_mask[y_slc, x_slc][local_mask == 1] = 1

    return clean_lines_mask


def split_junctions(binary_skeletonized, min_neighbors):
    h, w = binary_skeletonized.shape
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    
    out_split = np.zeros_like(binary_skeletonized)
    out_junc = np.zeros_like(binary_skeletonized, dtype=bool)
    tile_size = 4000
    margin = 2
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
            
            tile = binary_skeletonized[y0:y1, x0:x1]
            
            if HAS_GPU:
                cp_tile = cp.asarray(tile)
                cp_kernel = cp.asarray(kernel)
                nc_gpu = cp_ndi.convolve(cp_tile.astype(cp.uint8), cp_kernel, mode='constant', cval=0)
                neighbor_count = cp.asnumpy(nc_gpu)
            else:
                neighbor_count = cv2.filter2D(tile, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            
            junc_tile = (tile == 1) & (neighbor_count >= min_neighbors)
            split_tile = tile.copy()
            split_tile[junc_tile] = 0
            
            out_y0, out_y1 = y, min(h, y + tile_size)
            out_x0, out_x1 = x, min(w, x + tile_size)
            sy0, sy1 = out_y0 - y0, out_y0 - y0 + (out_y1 - out_y0)
            sx0, sx1 = out_x0 - x0, out_x0 - x0 + (out_x1 - out_x0)
            
            out_split[out_y0:out_y1, out_x0:out_x1] = split_tile[sy0:sy1, sx0:sx1]
            out_junc[out_y0:out_y1, out_x0:out_x1] = junc_tile[sy0:sy1, sx0:sx1]
            
    return out_split, out_junc

# Extract connected skeleton objects
def extract_objects(binary_splitted, min_object_size, profiler=None):
    if profiler: profiler.log_msg("      -> Extracting coordinate arrays (64-Bit Safe Sort)...")
    
    labels = np.zeros_like(binary_splitted, dtype=np.uint32)
    structure = np.ones((3,3), dtype=bool)
    num_labels = ndi.label(binary_splitted, output=labels, structure=structure)

    areas = np.bincount(labels.ravel())
    
    y_coords, x_coords = np.where(labels > 0)
    valid_pixels = labels[y_coords, x_coords]

    sort_idx = np.argsort(valid_pixels)
    sorted_labels = valid_pixels[sort_idx]
    sorted_y = y_coords[sort_idx]
    sorted_x = x_coords[sort_idx]

    _, split_idx = np.unique(sorted_labels, return_index=True)
    
    label_groups_y = np.split(sorted_y, split_idx[1:])
    label_groups_x = np.split(sorted_x, split_idx[1:])
    unique_labels = sorted_labels[split_idx]

    objects = []
    object_id = 0

    for lbl, grp_y, grp_x in zip(unique_labels, label_groups_y, label_groups_x):
        area = areas[lbl]
        if area >= min_object_size:
            objects.append({
                "id": object_id,
                "coordinates": np.column_stack((grp_x, grp_y)),
                "length_px": int(area),
            })
            object_id += 1

    if profiler: profiler.log_msg(f"      -> Extraction done! {len(objects)} valid logical objects found.")
    return objects
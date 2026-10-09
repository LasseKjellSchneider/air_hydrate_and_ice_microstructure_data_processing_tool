import os
import numpy as np
import math
import pandas as pd
import time
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.morphology import skeletonize, disk
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops
from skimage.feature import peak_local_max
import scipy.ndimage as ndi
from collections import deque, defaultdict
from pathlib import Path
import cv2
import concurrent.futures
from joblib import Parallel, delayed

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndi
    HAS_GPU = cp.cuda.is_available()
except Exception:
    HAS_GPU = False

from .logger_config import DebugProfiler
from .calculating_path_values import calculate_path_features
from .constants_manager import load_constants

CLOSING_PARAMS = {
    "max_radius": 4000,             
    "base_max_cost": 80000,         
    "repulsion_radius": 18,         
    "repulsion_penalty": 10000000,  
    "min_line_length": 200,         
    "min_t_junc_length": 3000,      
    "min_final_object_area": 1500,  
    "dilation_kernel": 150          
}

def _draw_legend(img, color_map_bgr, labels):
    box_width = 300
    box_height = 30 + len(labels) * 35
    legend_overlay = img.copy()
    cv2.rectangle(legend_overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(legend_overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, "Dijkstra Paths (ML Routed)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset = 80
    for cls_id, label in labels.items():
        color = color_map_bgr.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (20, y_offset - 15), (40, y_offset + 5), color, -1)
        cv2.putText(img, label, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35

# Parallel skeletonization
def chunked_skeletonize_parallel(binary_img, profiler=None):
    if profiler: profiler.log_msg("      -> Running multi-core skeletonization (Chunked 22-Core)...")
    
    h, w = binary_img.shape
    tile_size = 4000
    margin = 300  
    
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
    results = Parallel(n_jobs=optimal_workers, backend="loky")(delayed(_process_tile)(coords) for coords in tiles)

    for out_slice, inner_tile in results:
        if inner_tile is not None:
            out[out_slice] = inner_tile
    return out


def chunked_black_tophat(image, kernel_size):
    import cv2
    from joblib import Parallel, delayed
    import os
    
    h, w = image.shape
    k_dim = kernel_size * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dim, k_dim))
    
    out = np.zeros(image.shape, dtype=np.float32)
    tile_size = 4000
    margin = kernel_size * 2
    
    def _process_tile(coords):
        y, x = coords
        y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
        x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
        tile = image[y0:y1, x0:x1]
        filtered_tile = cv2.morphologyEx(tile, cv2.MORPH_BLACKHAT, kernel)
        sy0, sy1 = y - y0, y - y0 + min(h - y, tile_size)
        sx0, sx1 = x - x0, x - x0 + min(w - x, tile_size)
        return (slice(y, y + min(h - y, tile_size)), slice(x, x + min(w - x, tile_size))), filtered_tile[sy0:sy1, sx0:sx1]

    tiles = [(y, x) for y in range(0, h, tile_size) for x in range(0, w, tile_size)]
    optimal_workers = max(4, os.cpu_count() - 2)
    
    results = Parallel(n_jobs=optimal_workers, backend="threading")(delayed(_process_tile)(coords) for coords in tiles)
    for out_slice, inner_tile in results:
        out[out_slice] = inner_tile.astype(np.float32)
    return out

def chunked_edt_mask(binary_img, radius, is_erosion=False):
    h, w = binary_img.shape; img_8u = (binary_img > 0).astype(np.uint8)
    if h * w < 500_000_000:
        dist = cv2.distanceTransform(img_8u if is_erosion else 1 - img_8u, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return (dist > radius).astype(np.uint8) if is_erosion else (dist <= radius).astype(np.uint8)
    tile_size = 4000; margin = int(radius + 2); out = np.zeros_like(img_8u)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
            tile = img_8u[y0:y1, x0:x1]
            if HAS_GPU:
                cp_tile = cp.asarray(tile)
                dist_gpu = cp_ndi.distance_transform_edt(cp_tile if is_erosion else 1 - cp_tile)
                mask_tile = cp.asnumpy(dist_gpu > radius if is_erosion else dist_gpu <= radius).astype(np.uint8)
            else:
                dist = cv2.distanceTransform(tile if is_erosion else 1 - tile, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                mask_tile = (dist > radius if is_erosion else dist <= radius).astype(np.uint8)
            sy0, sy1 = y - y0, y - y0 + min(h, y + tile_size) - y
            sx0, sx1 = x - x0, x - x0 + min(w, x + tile_size) - x
            out[y:min(h, y + tile_size), x:min(w, x + tile_size)] = mask_tile[sy0:sy1, sx0:sx1]
    return out

def chunked_neighbor_count(binary_img):
    h, w = binary_img.shape; kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    if h * w < 500_000_000: return cv2.filter2D(binary_img.astype(np.uint8), cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    tile_size = 4000; margin = 1; out = np.zeros_like(binary_img, dtype=np.uint8)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0, y1 = max(0, y - margin), min(h, y + tile_size + margin)
            x0, x1 = max(0, x - margin), min(w, x + tile_size + margin)
            tile = binary_img[y0:y1, x0:x1].astype(np.uint8)
            if HAS_GPU:
                cp_tile = cp.asarray(tile); cp_kernel = cp.asarray(kernel)
                nc_tile = cp.asnumpy(cp_ndi.convolve(cp_tile, cp_kernel, mode='constant', cval=0.0)).astype(np.uint8)
            else:
                nc_tile = cv2.filter2D(tile, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
            sy0, sy1 = y - y0, y - y0 + min(h, y + tile_size) - y
            sx0, sx1 = x - x0, x - x0 + min(w, x + tile_size) - x
            out[y:min(h, y + tile_size), x:min(w, x + tile_size)] = nc_tile[sy0:sy1, sx0:sx1]
    return out

def _local_route_through_array_ts(cost_map, start, end, margin=120):
    h, w = cost_map.shape
    min_y, max_y = min(start[0], end[0]), max(start[0], end[0])
    min_x, max_x = min(start[1], end[1]), max(start[1], end[1])
    
    y0, y1 = max(0, min_y - margin), min(h, max_y + margin)
    x0, x1 = max(0, min_x - margin), min(w, max_x + margin)
    
    local_cost = cost_map[y0:y1, x0:x1].copy()
    sy, sx = start[0]-y0, start[1]-x0
    ey, ex = end[0]-y0, end[1]-x0
    
    local_cost[sy, sx] = 1.0
    local_cost[ey, ex] = 1.0
    
    local_path, cost = route_through_array(local_cost, (sy, sx), (ey, ex), fully_connected=True)
    return [(y + y0, x + x0) for (y, x) in local_path], cost

def path_crosses_skeleton(path, skeleton):
    if len(path) <= 10: return False
    for y, x in path[5:-5]:
        if skeleton[y, x] == 1: return True
    return False

def _path_crosses_existing(path, drawn_mask):
    if len(path) <= 4: return False
    for y, x in path[2:-2]:
        if drawn_mask[y, x] == 1: return True
    return False

def _trace_line_length(start_y, start_x, skeleton, required_length):
    h, w = skeleton.shape; visited = {(start_y, start_x)}; queue = deque([(start_y, start_x)])
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]; path_coords = [(start_y, start_x)]
    while queue:
        y, x = queue.popleft()
        if len(path_coords) >= required_length: return True, f"line_{start_y}_{start_x}", path_coords
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] == 1 and (ny, nx) not in visited:
                visited.add((ny, nx)); queue.append((ny, nx)); path_coords.append((ny, nx)); break
    return False, f"line_{start_y}_{start_x}", path_coords

def _find_t_junc_targets_cone(ey, ex, skeleton, max_radius, vector_length=40, cone_angle_deg=90):
    is_valid, _, path_coords = _trace_line_length(ey, ex, skeleton, required_length=vector_length)
    if len(path_coords) < 5: return [] 
    ty, tx = path_coords[-1] 
    dy, dx = ey - ty, ex - tx
    norm = math.hypot(dx, dy)
    if norm == 0: return []
    dx, dy = dx / norm, dy / norm
    
    angle = math.radians(cone_angle_deg / 2.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    lx, ly = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
    rx, ry = dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a
    pt_l = (int(ex + lx * max_radius), int(ey + ly * max_radius))
    pt_r = (int(ex + rx * max_radius), int(ey + ry * max_radius))
    
    h, w = skeleton.shape
    y0, y1 = max(0, ey - max_radius), min(h, ey + max_radius)
    x0, x1 = max(0, ex - max_radius), min(w, ex + max_radius)
    
    cone_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cone_pts = np.array([[(ex)-x0, (ey)-y0], [pt_l[0]-x0, pt_l[1]-y0], [pt_r[0]-x0, pt_r[1]-y0]], np.int32)
    cv2.fillPoly(cone_mask, [cone_pts], 1)
    local_skel = skeleton[y0:y1, x0:x1]
    valid_targets = (cone_mask == 1) & (local_skel == 1)
    
    for py, px in path_coords:
        if y0 <= py < y1 and x0 <= px < x1: valid_targets[py - y0, px - x0] = False
            
    target_coords = np.argwhere(valid_targets)
    if len(target_coords) == 0: return []
    
    ley, lex = ey - y0, ex - x0
    dists = (target_coords[:, 0] - ley)**2 + (target_coords[:, 1] - lex)**2
    sorted_idx = np.argsort(dists)
    
    best_targets = []
    for idx in sorted_idx:
        ty, tx = target_coords[idx, 0] + y0, target_coords[idx, 1] + x0
        is_far_enough = True
        for (by, bx) in best_targets:
            if math.hypot(ty - by, tx - bx) < 5:
                is_far_enough = False
                break
        if is_far_enough:
            best_targets.append((ty, tx))
            if len(best_targets) >= 3: break
            
    return best_targets

def _process_e2e_task(pair_data, cost_ref, skel_ref, img_ref, tophat_ref):
    i, j, ep1, ep2 = pair_data
    try:
        t_start = time.perf_counter()
        path, cost = _local_route_through_array_ts(cost_ref, ep1['coord'], ep2['coord'], margin=120)
        t_route = time.perf_counter() - t_start
        if not path_crosses_skeleton(path, skel_ref):
            t_start_feat = time.perf_counter()
            feat = calculate_path_features(path, cost, img_ref, skel_ref, tophat_ref)
            t_feat = time.perf_counter() - t_start_feat
            if feat: return {"ep1_idx": i, "ep2_idx": j, "path": path, "features": feat, "t_route": t_route, "t_feat": t_feat}
    except ValueError: pass
    return None

def _process_tj_task(pair_data, cost_ref, skel_ref, img_ref, tophat_ref):
    i, ep, target_pixel = pair_data
    try:
        t_start = time.perf_counter()
        path, cost = _local_route_through_array_ts(cost_ref, ep['coord'], target_pixel, margin=120)
        t_route = time.perf_counter() - t_start
        if not path_crosses_skeleton(path, skel_ref):
            t_start_feat = time.perf_counter()
            feat = calculate_path_features(path, cost, img_ref, skel_ref, tophat_ref)
            t_feat = time.perf_counter() - t_start_feat
            if feat: return {'ep_idx': i, 'path': path, 'features': feat, "t_route": t_route, "t_feat": t_feat}
    except ValueError: pass
    return None

def close_grain_boundaries(filtered_objects, image, border_mask=None, debug_dir=None, image_callback=None, e2e_rf=None, tjunc_rf=None, prob_threshold=0.5, parameters=None, return_intermediates=False):
    if parameters is None: parameters = load_constants()
    bm = parameters.get("border_margin_px", 50)
    
    dilation_kernel = parameters.get("closing_dilation_kernel", CLOSING_PARAMS["dilation_kernel"])
    min_line_length = parameters.get("closing_min_line_length", CLOSING_PARAMS["min_line_length"])
    repulsion_radius = parameters.get("closing_repulsion_radius", CLOSING_PARAMS["repulsion_radius"])
    repulsion_penalty = parameters.get("closing_repulsion_penalty", CLOSING_PARAMS["repulsion_penalty"])
    max_radius = parameters.get("max_radius", CLOSING_PARAMS["max_radius"])
    vec_len = parameters.get("t_junc_vector_length_px", 40)
    cone_angle = parameters.get("t_junc_cone_angle_deg", 90)
    d_mult = parameters.get("dijkstra_cost_multiplier", 300.0) 
    clip_perc = parameters.get("dijkstra_tophat_clip_percentile", 80.0)
    dark_factor = float(parameters.get("dijkstra_darkness_factor", 0.55))
    
    profiler = DebugProfiler(debug_dir)
    profiler.log_step("Init ML close_grain_boundaries")
    
    profiler.log_msg("=== SYSTEM & GPU DIAGNOSTICS ===")
    if HAS_GPU:
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_gb = mem_info[0] / (1024**3)
            total_gb = mem_info[1] / (1024**3)
            profiler.log_msg(f"GPU Detected: CuPy is active. VRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
        except Exception as e:
            profiler.log_msg(f"GPU Detected but memGetInfo failed: {e}")
    else:
        profiler.log_msg("No GPU detected for CuPy. Running strictly on CPU.")
    profiler.log_msg("================================")
    
    intermediates = {}
    h, w = image.shape[:2]
    new_lines_info = [] 
    diagnostic_masks = {
        "kept_base_raw": np.zeros((h, w), dtype=np.uint8),
        "removed_initial": np.zeros((h, w), dtype=np.uint8),
        "removed_duplicate": np.zeros((h, w), dtype=np.uint8),
        "removed_final_size": np.zeros((h, w), dtype=np.uint8)
    }
    
    if len(filtered_objects) == 0: 
        if return_intermediates: return np.zeros((h,w), dtype=np.uint8), new_lines_info, diagnostic_masks, [], intermediates
        return np.zeros((h,w), dtype=np.uint8), new_lines_info, diagnostic_masks, []

    t_s1, r_s1 = profiler.log_step("Step 1: Base skeleton")
    base_mask = np.zeros((h, w), dtype=np.uint8)
    for obj in filtered_objects:
        coords = obj["coordinates"]
        if len(coords) > 0: base_mask[coords[:, 1], coords[:, 0]] = 1
        
    edt_mask = chunked_edt_mask(base_mask, radius=dilation_kernel // 2)
    # Process tiles in parallel to keep skeletonization practical for large images.
    skeleton = chunked_skeletonize_parallel(edt_mask, profiler)
    del edt_mask
    if return_intermediates: intermediates["8. Raw Base Skeleton"] = skeleton.copy() * 255
    profiler.log_step("Finished Step 1", t_s1, r_s1)

    t_s2, r_s2 = profiler.log_step("Step 2: Size Filter")
    labels = np.zeros_like(skeleton, dtype=np.uint32)
    num_labels = ndi.label(skeleton, output=labels, structure=np.ones((3,3), dtype=bool))
    areas = np.bincount(labels.ravel())
    labels_to_remove = set(i for i in range(1, num_labels) if areas[i] < min_line_length)
    remove_mask = np.isin(labels, list(labels_to_remove))
    diagnostic_masks["removed_initial"][remove_mask] = 1
    skeleton[remove_mask] = 0
    
    diagnostic_masks["kept_base_raw"] = skeleton.copy()
    if return_intermediates: intermediates["9. Size Filtered Skeleton"] = skeleton.copy() * 255
    profiler.log_step("Finished Step 2", t_s2, r_s2)

    t_s3, r_s3 = profiler.log_step("Step 3: Cost Map")
    cost_map_tophat = chunked_black_tophat(image, kernel_size=12)
    clip_val = np.float32(np.percentile(cost_map_tophat[::10, ::10], clip_perc))
    np.clip(cost_map_tophat, 0.0, clip_val, out=cost_map_tophat)
    cost_map_tophat /= (np.max(cost_map_tophat) + np.float32(1e-5))
    
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    valid_ice = blurred[border_mask > 0] if (border_mask is not None and np.any(border_mask > 0)) else blurred.flatten()[::100]
    if len(valid_ice) == 0: valid_ice = blurred.flatten()[::100]
    
    ice_median = np.float32(np.median(valid_ice))
    dynamic_cutoff = ice_median * np.float32(dark_factor)
    
    darkness_norm = np.clip(blurred, 0, dynamic_cutoff).astype(np.float32)
    del blurred 
    darkness_norm /= (dynamic_cutoff + np.float32(1e-5))
    np.subtract(np.float32(1.0), darkness_norm, out=darkness_norm) 
    
    combined_score = np.maximum(cost_map_tophat, darkness_norm)
    del cost_map_tophat, darkness_norm 
    
    if return_intermediates:
        vis_cost = cv2.normalize(combined_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        intermediates["10. Hybrid Cost Score"] = vis_cost
        
    max_comb = np.float32(np.max(combined_score))
    mult_factor = np.float32(-(d_mult / (max_comb + 1e-5)))
    add_factor = np.float32(1.0 + d_mult)
    
    cost_map = combined_score * mult_factor + add_factor
    del combined_score 

    cost_map[skeleton == 1] = 99999.0 
    cost_map[(chunked_edt_mask(skeleton, radius=repulsion_radius) == 1) & (skeleton != 1)] += repulsion_penalty
    
    border_zone = np.zeros((h, w), dtype=bool)
    border_zone[:bm, :] = True; border_zone[-bm:, :] = True; border_zone[:, :bm] = True; border_zone[:, -bm:] = True
    if border_mask is not None: border_zone[chunked_edt_mask(border_mask, radius=bm, is_erosion=True) == 0] = True
    profiler.log_step("Finished Step 3", t_s3, r_s3)

    t_s4, r_s4 = profiler.log_step("Step 4: Endpoints")
    ep_coords = np.argwhere((skeleton == 1) & (chunked_neighbor_count(skeleton) == 1))
    endpoints = [{'coord': (y, x), 'label': labels[y, x], 'near_border': border_zone[y, x], 'connected': False} for y, x in ep_coords if labels[y, x] not in labels_to_remove]
    profiler.log_step("Finished Step 4", t_s4, r_s4)

    t_s45, r_s45 = profiler.log_step("Step 4.5: Precomputing Feature Tophat Map")
    tophat_map_features = chunked_black_tophat(image, kernel_size=6)
    profiler.log_step("Finished Step 4.5", t_s45, r_s45)

    drawn_paths_mask = np.zeros((h, w), dtype=np.uint8)
    optimal_workers = max(4, os.cpu_count() - 2)
    
    t_s5, r_s5 = profiler.log_step("Step 5: E2E Routing")
    potential_e2e = []
    if len(endpoints) >= 2 and e2e_rf is not None:
        tree = cKDTree([ep['coord'] for ep in endpoints])
        pairs = []
        for i, ep1 in enumerate(endpoints):
            if ep1['near_border']: continue 
            dists, indices = tree.query(ep1['coord'], k=10, distance_upper_bound=max_radius)
            for j in np.atleast_1d(indices):
                if j == len(endpoints): continue
                if i >= j: continue
                ep2 = endpoints[j]
                if not ep2['near_border']: pairs.append((i, j, ep1, ep2))
        
        profiler.log_msg(f"Found {len(pairs)} potential E2E pairs. Routing via LOKY Multiprocessing...")
        t_route_start = time.time()
        res_e2e = Parallel(n_jobs=optimal_workers, backend="loky")(
            delayed(_process_e2e_task)(p, cost_map, skeleton, image, tophat_map_features) for p in pairs
        )
        route_time = time.time() - t_route_start
        
        valid_res = [r for r in res_e2e if r is not None]
        if valid_res:
            avg_route = np.mean([r["t_route"] for r in valid_res])
            avg_feat = np.mean([r["t_feat"] for r in valid_res])
            profiler.log_msg(f"   -> E2E Multi-Threading completed in {route_time:.2f}s (Throughput: {len(pairs)/max(0.001, route_time):.2f} pairs/s)")
            profiler.log_msg(f"   -> Avg pure Dijkstra routing time per pair: {avg_route*1000:.2f} ms")
            profiler.log_msg(f"   -> Avg pure Feature extraction time per pair: {avg_feat*1000:.2f} ms")
            
        potential_e2e = valid_res

        if potential_e2e:
            df_e2e = pd.DataFrame([pt["features"] for pt in potential_e2e])[e2e_rf.feature_names_in_]
            probs = e2e_rf.predict_proba(df_e2e)[:, 1] 
            for k, pt in enumerate(potential_e2e): pt["prob"] = probs[k]
            potential_e2e.sort(key=lambda x: x["prob"], reverse=True)
            
    connected_eps = set()
    for conn in potential_e2e:
        i, j, path, prob = conn["ep1_idx"], conn["ep2_idx"], conn["path"], conn["prob"]
        if i in connected_eps or j in connected_eps or _path_crosses_existing(path, drawn_paths_mask): continue
        if prob >= prob_threshold:
            for (y, x) in path: skeleton[y, x] = 1; drawn_paths_mask[y, x] = 1; cost_map[y, x] = 99999.0 
            connected_eps.update([i, j]); endpoints[i]['connected'] = True; endpoints[j]['connected'] = True
            new_lines_info.append({"path": path, "cost": conn["features"]["cost"], "prob": prob, "phase": "E2E"})
        else: new_lines_info.append({"path": path, "cost": conn["features"]["cost"], "prob": prob, "phase": "Rej E2E"})
    profiler.log_step("Finished Step 5", t_s5, r_s5)

    t_s6, r_s6 = profiler.log_step("Step 6: T-Junc Routing")
    potential_tjuncs = []
    if tjunc_rf is not None:
        tj_pairs = []
        for i, ep in enumerate(endpoints):
            if ep['connected'] or ep['near_border']: continue
            targets = _find_t_junc_targets_cone(ep['coord'][0], ep['coord'][1], skeleton, max_radius, vec_len, cone_angle)
            for target_pixel in targets: tj_pairs.append((i, ep, target_pixel))
            
        profiler.log_msg(f"Found {len(tj_pairs)} potential T-Junc pairs. Routing via LOKY Multiprocessing...")
        t_route_start = time.time()
        res_tj = Parallel(n_jobs=optimal_workers, backend="loky")(
            delayed(_process_tj_task)(p, cost_map, skeleton, image, tophat_map_features) for p in tj_pairs
        )
        route_time = time.time() - t_route_start
        
        valid_res_tj = [r for r in res_tj if r is not None]
        if valid_res_tj:
            avg_route = np.mean([r["t_route"] for r in valid_res_tj])
            avg_feat = np.mean([r["t_feat"] for r in valid_res_tj])
            profiler.log_msg(f"   -> T-Junc Multi-Threading completed in {route_time:.2f}s (Throughput: {len(tj_pairs)/max(0.001, route_time):.2f} pairs/s)")
            profiler.log_msg(f"   -> Avg pure Dijkstra routing time per pair: {avg_route*1000:.2f} ms")
            profiler.log_msg(f"   -> Avg pure Feature extraction time per pair: {avg_feat*1000:.2f} ms")
            
        potential_tjuncs = valid_res_tj

        if potential_tjuncs:
            df_tj = pd.DataFrame([pt["features"] for pt in potential_tjuncs])[tjunc_rf.feature_names_in_]
            probs = tjunc_rf.predict_proba(df_tj)[:, 1]
            for k, pt in enumerate(potential_tjuncs): pt["prob"] = probs[k]
            potential_tjuncs.sort(key=lambda x: x["prob"], reverse=True)
        
    for tj in potential_tjuncs:
        i, path, prob = tj['ep_idx'], tj["path"], tj["prob"]
        if endpoints[i]['connected'] or _path_crosses_existing(path, drawn_paths_mask): continue
        if prob >= prob_threshold:
            for (y, x) in path: skeleton[y, x] = 1; drawn_paths_mask[y, x] = 1
            endpoints[i]['connected'] = True
            new_lines_info.append({"path": path, "cost": tj["features"]["cost"], "prob": prob, "phase": "T-Junc"})
        else: new_lines_info.append({"path": path, "cost": tj["features"]["cost"], "prob": prob, "phase": "Rej T"})
    
    del tophat_map_features
    profiler.log_step("Finished Step 6", t_s6, r_s6)

    # Watershed segmentation and region merging
    t_s7, r_s7 = profiler.log_step("Step 7: Watershed & Region Merging")
    
    ws_scale = int(parameters.get("ws_scale_factor", 12))
    ws_blur = int(parameters.get("ws_blur_kernel", 5)) | 1
    ws_min_depth = float(parameters.get("ws_min_seed_depth", 3.0))
    ws_y_int = float(parameters.get("ws_y_intercept", 5.0))
    ws_slope = float(parameters.get("ws_slope", 0.5))
    ws_cutoff = float(parameters.get("ws_cutoff", 50.0))
    
    ws_base = skeleton.copy()
    if border_mask is not None: ws_base[border_mask == 0] = 1
    ws_base[:bm, :] = 1; ws_base[-bm:, :] = 1; ws_base[:, :bm] = 1; ws_base[:, -bm:] = 1
    
    if debug_dir is not None:
        out_path_ws = Path(debug_dir).parent / "ws_base_mask.tif"
        cv2.imwrite(str(out_path_ws), ws_base * 255, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        
    # Dilate each tile before downscaling so thin boundaries remain represented.
    if ws_scale > 1:
        profiler.log_msg("      -> Downscaling skeleton safely (Chunked Dilation)...")
        small_h, small_w = h // ws_scale, w // ws_scale
        small_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        
        tile_size = 4000
        # Match the dilation kernel to the scale factor to retain narrow lines.
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ws_scale + 1, ws_scale + 1))
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
                tile = ws_base[y:y1, x:x1]
                
                if not np.any(tile): 
                    continue
                
                # Expand thin lines before reducing the tile resolution.
                dilated_tile = cv2.dilate(tile, kernel_dilate)
                
                # Map the source tile bounds to the downscaled image.
                sy, ey = y // ws_scale, y1 // ws_scale
                sx, ex = x // ws_scale, x1 // ws_scale
                
                # Resize the tile and place it in the corresponding output region.
                if ey > sy and ex > sx:
                    small_tile = cv2.resize(dilated_tile, (ex - sx, ey - sy), interpolation=cv2.INTER_NEAREST)
                    small_mask[sy:ey, sx:ex] = small_tile
    else:
        small_mask = ws_base.copy()
        
    dist_map = cv2.distanceTransform(1 - small_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_map = cv2.GaussianBlur(dist_map, (ws_blur, ws_blur), 0)
    
    coords = peak_local_max(dist_map, min_distance=20, threshold_abs=ws_min_depth)
    local_max = np.zeros_like(dist_map, dtype=bool)
    if len(coords) > 0: local_max[tuple(coords.T)] = True
    markers, num_markers = ndi.label(local_max, structure=np.ones((3,3)))
    
    if num_markers > 0:
        labels = watershed(-dist_map, markers, mask=(small_mask == 0), watershed_line=True)
        dam_coords = np.argwhere((labels == 0) & (small_mask == 0))
        padded_labels = np.pad(labels, 1, mode='constant', constant_values=0)
        y_p, x_p = dam_coords[:, 0] + 1, dam_coords[:, 1] + 1
        
        neighbors = np.column_stack((
            padded_labels[y_p-1, x_p-1], padded_labels[y_p-1, x_p], padded_labels[y_p-1, x_p+1],
            padded_labels[y_p,   x_p-1],                               padded_labels[y_p,   x_p+1],
            padded_labels[y_p+1, x_p-1], padded_labels[y_p+1, x_p], padded_labels[y_p+1, x_p+1]
        ))
        
        dam_segments = defaultdict(list)
        for idx, (y, x) in enumerate(dam_coords):
            n_labels = neighbors[idx]
            u_labels = np.unique(n_labels[n_labels > 0])
            if len(u_labels) == 1: dam_segments[(u_labels[0], u_labels[0])].append((y, x))
            elif len(u_labels) == 2:
                pair = tuple(sorted(u_labels))
                dam_segments[pair].append((y, x))
            elif len(u_labels) > 2:
                for i in range(len(u_labels)):
                    for j in range(i+1, len(u_labels)):
                        pair = tuple(sorted((u_labels[i], u_labels[j])))
                        dam_segments[pair].append((y, x))
        
        props = regionprops(markers)
        seed_coords = {p.label: p.centroid for p in props}
        merge_graph = []
        
        for (l1, l2), pixels in dam_segments.items():
            dam_length = len(pixels)
            if dam_length == 0: continue
            if l1 == l2:
                merge_graph.append((l1, l1))
                continue
                
            py = sum(p[0] for p in pixels) / dam_length
            px = sum(p[1] for p in pixels) / dam_length
            dam_center = (py, px)
            
            c1 = seed_coords.get(l1, dam_center)
            c2 = seed_coords.get(l2, dam_center)
            
            dist1 = math.hypot(c1[0] - dam_center[0], c1[1] - dam_center[1])
            dist2 = math.hypot(c2[0] - dam_center[0], c2[1] - dam_center[1])
            min_dist = min(dist1, dist2)
            
            thresh = ws_y_int + ws_slope * min_dist
            thresh = min(thresh, ws_cutoff)
            if dam_length > thresh: merge_graph.append((l1, l2))
                
        parent = {i: i for i in range(1, num_markers + 1)}
        def find_parent(i):
            if parent[i] == i: return i
            parent[i] = find_parent(parent[i])
            return parent[i]
            
        for l1, l2 in merge_graph:
            root_l1 = find_parent(l1)
            root_l2 = find_parent(l2)
            if root_l1 != root_l2: parent[root_l1] = root_l2
                
        new_labels = np.zeros_like(labels)
        for i in range(1, num_markers + 1): new_labels[labels == i] = find_parent(i)
            
        for (l1, l2), pixels in dam_segments.items():
            root_l1 = find_parent(l1)
            root_l2 = find_parent(l2)
            if root_l1 == root_l2:
                py = [p[0] for p in pixels]; px = [p[1] for p in pixels]
                new_labels[py, px] = root_l1
            
        if new_labels.max() < 65535: new_labels = new_labels.astype(np.uint16)
        else: new_labels = new_labels.astype(np.float32)
            
        # Use 64-bit index mapping to avoid OpenCV limits on large arrays.
        if ws_scale > 1:
            profiler.log_msg("      -> Upscaling watershed labels (64-Bit Vectorized)...")
            y_idx = np.clip((np.arange(h, dtype=np.int64) * small_h) // h, 0, small_h - 1)
            x_idx = np.clip((np.arange(w, dtype=np.int64) * small_w) // w, 0, small_w - 1)
            big_labels = new_labels[y_idx[:, None], x_idx]
        else:
            big_labels = new_labels
            
        if big_labels.dtype == np.float32: big_labels = big_labels.astype(np.int32)
            
        bounds = find_boundaries(big_labels, mode='inner')
        # Skeletonize the final boundaries in parallel for large images.
        final_skeleton = chunked_skeletonize_parallel(bounds, profiler)
        
        if border_mask is not None: final_skeleton[border_mask == 0] = 1
        final_skeleton[:bm, :] = 1; final_skeleton[-bm:, :] = 1; final_skeleton[:, :bm] = 1; final_skeleton[:, -bm:] = 1
        skeleton = final_skeleton

    profiler.log_step("Finished Step 7", t_s7, r_s7)

    if debug_dir is not None:
        profiler.log_msg("Generating Transparent Debug Overlays...")
        step = 4
        small_img = image[::step, ::step].copy()
        if small_img.dtype != np.uint8: small_img = cv2.normalize(small_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bg_preview = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)
        overlay_mask = np.zeros_like(bg_preview)
        
        skel_dilated = cv2.dilate(skeleton[::step, ::step], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        overlay_mask[skel_dilated == 1] = [0, 255, 0] 
        
        color_map = {"E2E": (255, 0, 0), "Rej E2E": (0, 0, 255), "T-Junc": (0, 255, 255), "Rej T": (0, 165, 255)}
        labels_dict = {"E2E": f"E2E (>={int(prob_threshold*100)}%)", "Rej E2E": f"E2E (<{int(prob_threshold*100)}%)", "T-Junc": f"T-Junc (>={int(prob_threshold*100)}%)", "Rej T": f"T-Junc (<{int(prob_threshold*100)}%)"}
        
        for line in new_lines_info:
            if len(line["path"]) > 1:
                color = color_map.get(line["phase"], (255, 255, 255))
                scaled_path = np.array([[(x // step, y // step)] for (y, x) in line["path"]], dtype=np.int32)
                cv2.polylines(overlay_mask, [scaled_path], isClosed=False, color=color, thickness=10)
                
        for ep in endpoints:
            if not ep['connected']:
                cy, cx = ep['coord'][0] // step, ep['coord'][1] // step
                cv2.circle(overlay_mask, (cx, cy), 15, (0, 0, 255), -1); cv2.circle(overlay_mask, (cx, cy), 15, (255, 255, 255), 2)
                
        active_pixels = np.any(overlay_mask > 0, axis=-1)
        bg_preview[active_pixels] = cv2.addWeighted(overlay_mask, 0.6, bg_preview, 0.4, 0)[active_pixels]
        _draw_legend(bg_preview, color_map, labels_dict)
        
        out_path = Path(debug_dir) / "13_dijkstra_paths_overlay.png"
        cv2.imwrite(str(out_path), bg_preview)
        if image_callback: image_callback(str(out_path), f"Phase 3: Pathfinding (Thresh {int(prob_threshold*100)}%)")

    profiler.close()
    
    if return_intermediates:
        return skeleton, new_lines_info, diagnostic_masks, [], intermediates
    return skeleton, new_lines_info, diagnostic_masks, []
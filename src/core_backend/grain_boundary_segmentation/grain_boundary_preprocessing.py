import cv2
import numpy as np
import tifffile
from pathlib import Path
import imageio.v3 as iio

def border_cut_out(image, cutoff_threshold=120, safety_margin=200, keep_two_largest=False):
    """
    Detects the main sample area on a 4x downscaled image for maximum performance.
    """
    h, w = image.shape[:2]
    
    # 1. Downscale by factor 4 (INTER_AREA is best for shrinking)
    small_img = image[::4, ::4].copy()
    # Use the sliced dimensions when resizing to avoid rounding mismatches.
    small_h, small_w = small_img.shape[:2] 
    
    # 2. Process on the small image
    blurred = cv2.GaussianBlur(small_img, (7, 7), 0)
    _, binary = cv2.threshold(blurred, cutoff_threshold, 255, cv2.THRESH_BINARY)
    
    # ---------------------------------------------------------
    # Close narrow gaps, then remove small protrusions from the sample mask.
    # ---------------------------------------------------------
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_mask = np.zeros_like(small_img, dtype=np.uint8)
    
    if contours:
        # Prefer the largest connected sample regions.
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Retain a second region only when explicitly requested.
        num_to_keep = 2 if keep_two_largest and len(sorted_contours) >= 2 else 1
        cv2.drawContours(small_mask, sorted_contours[:num_to_keep], -1, 255, thickness=cv2.FILLED)
    
    # Restore the mask to the exact source dimensions.
    large_mask = small_mask.repeat(4, axis=0).repeat(4, axis=1)
    large_mask = large_mask[:h, :w]
    
    # Erode the mask by the requested safety margin and discard small islands.
    if safety_margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        large_mask = cv2.erode(large_mask, kernel, iterations=safety_margin)
        
        contours_eroded, _ = cv2.findContours(large_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_mask = np.zeros_like(large_mask)  # Rebuild the mask from retained contours.
        if contours_eroded:
            sorted_eroded = sorted(contours_eroded, key=cv2.contourArea, reverse=True)
            num_to_keep_eroded = 2 if keep_two_largest and len(sorted_eroded) >= 2 else 1
            cv2.drawContours(large_mask, sorted_eroded[:num_to_keep_eroded], -1, 255, thickness=cv2.FILLED)

    # Build a red overlay outside the detected sample for visual review.
    if small_img.dtype != np.uint8:
        small_img_8u = cv2.normalize(small_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        small_img_8u = small_img.copy()
        
    base_image = cv2.cvtColor(small_img_8u, cv2.COLOR_GRAY2BGR)
    
    # Match the downscaled image dimensions before composing the preview.
    final_small_mask = cv2.resize(large_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    red_overlay = base_image.copy()
    red_overlay[final_small_mask == 0] = [0, 0, 255]  # OpenCV images use BGR channel order.
    
    alpha = 0.4
    blended_overlay = cv2.addWeighted(red_overlay, alpha, base_image, 1 - alpha, 0)

    return large_mask, blended_overlay


def process_single_image(img_path, cutoff_threshold=120, safety_margin=200, keep_two_largest=False):
    """
    Reads a single .tif file, processes the borders, and overwrites the mask/overlay.
    Used for on-the-fly GUI updates.
    """
    img_path = Path(img_path)
    with tifffile.TiffFile(str(img_path)) as tif:
        image = tif.asarray()
        
    large_mask, blended_overlay = border_cut_out(
        image, 
        cutoff_threshold=cutoff_threshold, 
        safety_margin=safety_margin, 
        keep_two_largest=keep_two_largest
    )
    
    mask_path = img_path.parent / (img_path.stem + "_mask.png")
    overlay_path = img_path.parent / (img_path.stem + "_overlay.png")
    
    iio.imwrite(str(mask_path), large_mask)
    iio.imwrite(str(overlay_path), blended_overlay)


def process_and_save_borders(folder_path, status_callback=None, progress_callback=None):
    """
    Initial batch run: Iterates through all .tif files, calculates the border mask
    with default values, and saves files.
    """
    cal_folder = Path(folder_path)
    image_paths = list(cal_folder.rglob("*.tif"))
    
    # Exclude generated masks and previews from the source-image batch.
    image_paths = [p for p in image_paths if not p.name.endswith("_mask.tif") 
                   and not p.name.endswith("_mask.png") 
                   and not p.name.endswith("_overlay.png")]
    
    total_images = len(image_paths)
    msg = f"Starting Preprocessing for {total_images} images..."
    if status_callback: status_callback(msg)
    print(f"\n--- {msg} ---")
    
    if total_images == 0:
        if status_callback: status_callback("No valid .tif files found!")
        return

    for idx, img_path in enumerate(image_paths, start=1):
        step_msg = f"Processing {idx}/{total_images}: {img_path.name}"
        if status_callback: status_callback(step_msg)
        print(step_msg)
        
        if progress_callback: progress_callback(idx / total_images)
        
        # Use the same processing path as the interactive review workflow.
        process_single_image(img_path)
        
    final_msg = "Preprocessing complete! Launching Review Mode..."
    if status_callback: status_callback(final_msg)
    print(f"\n{final_msg}\n")
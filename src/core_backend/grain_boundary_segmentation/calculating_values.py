import cv2
import numpy as np

def calculate_values(objects, image):
    """
    Calculates advanced geometric and intensity ROI features for each detected object.
    These features serve as the input vector for the Random Forest classifier.
    Optimized to calculate gradients locally to completely avoid 32-bit OpenCV limits
    and save enormous amounts of RAM for multi-gigapixel images.
    """
    print("Calculating object features (Geometry & ROI Band)...")
    
    # Define the neighborhood bands used to compare each boundary with its surroundings.
    roi_radius_px = 10  # Produces a 21-pixel-wide region of interest.
    core_radius_px = 1  # Produces a 3-pixel-wide boundary core.
    
    objects_calculated = []
     
    for i, obj in enumerate(objects):
        coords = obj["coordinates"]  # Coordinates are stored as [x, y].
        
        # Collect the feature values for this object.
        val = {}
        
        # Geometry features
        val["length_px"] = float(len(coords))
        val["euclidean_dist"] = calc_euclidean_distance(coords)
        val["tortuosity"] = val["length_px"] / val["euclidean_dist"]
        
        # Restrict image processing to a padded crop around the object.
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        
        # Include the full region-of-interest band in the crop.
        pad = roi_radius_px + 2
        h, w = image.shape
        x_min_p, x_max_p = max(0, x_min - pad), min(w, x_max + pad + 1)
        y_min_p, y_max_p = max(0, y_min - pad), min(h, y_max + pad + 1)
        
        # Extract the corresponding image and gradient patch.
        local_img = image[y_min_p:y_max_p, x_min_p:x_max_p]
        
        # Compute gradients locally instead of processing the full image.
        grad_x = cv2.Sobel(local_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(local_img, cv2.CV_32F, 0, 1, ksize=3)
        local_grad = cv2.magnitude(grad_x, grad_y)
        
        # Recreate the detected one-pixel line in crop-local coordinates.
        local_mask = np.zeros_like(local_img, dtype=np.uint8)
        local_coords = coords - [x_min_p, y_min_p]
        local_mask[local_coords[:, 1], local_coords[:, 0]] = 1
        
        # Measure line jaggedness on the local mask.
        val["jaggedness"] = calc_jaggedness(local_mask)

        # Build masks for the boundary core, full neighborhood, and background.
        kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_core = cv2.dilate(local_mask, kernel_core)
        
        kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (roi_radius_px*2 + 1, roi_radius_px*2 + 1))
        mask_roi = cv2.dilate(local_mask, kernel_roi)
        
        mask_bg = mask_roi - mask_core 
        
        # Sample intensity and gradient values from each region.
        pixels_core = local_img[mask_core == 1]
        pixels_bg = local_img[mask_bg == 1]
        pixels_roi_full = local_img[mask_roi == 1]
        
        grad_core = local_grad[mask_core == 1]
        grad_bg = local_grad[mask_bg == 1]
        grad_roi_full = local_grad[mask_roi == 1]
        
        # Intensity statistics for the full neighborhood band
        val["roi_mean_intensity"] = float(np.mean(pixels_roi_full))
        val["roi_std_intensity"] = float(np.std(pixels_roi_full))
        val["roi_p10_intensity"] = float(np.percentile(pixels_roi_full, 10))  # Lower intensity percentile.
        val["roi_p90_intensity"] = float(np.percentile(pixels_roi_full, 90))  # Upper intensity percentile.
        val["roi_contrast_range"] = val["roi_p90_intensity"] - val["roi_p10_intensity"]
        
        # Compare the boundary core with the surrounding background.
        val["core_mean_intensity"] = float(np.mean(pixels_core)) if len(pixels_core) > 0 else 0.0
        val["bg_mean_intensity"] = float(np.mean(pixels_bg)) if len(pixels_bg) > 0 else 0.0
        val["core_bg_contrast"] = val["bg_mean_intensity"] - val["core_mean_intensity"]
        val["core_bg_ratio"] = val["core_mean_intensity"] / (val["bg_mean_intensity"] + 1e-5)

        # Gradient statistics describe edge sharpness in the neighborhood.
        val["roi_mean_grad"] = float(np.mean(grad_roi_full))
        val["roi_max_grad"] = float(np.max(grad_roi_full))
        val["roi_std_grad"] = float(np.std(grad_roi_full))
        
        val["core_mean_grad"] = float(np.mean(grad_core)) if len(grad_core) > 0 else 0.0
        val["bg_mean_grad"] = float(np.mean(grad_bg)) if len(grad_bg) > 0 else 0.0
        val["grad_concentration"] = val["core_mean_grad"] / (val["bg_mean_grad"] + 1e-5)
        
        # Store the computed features with their source object.
        obj["values"] = val
        objects_calculated.append(obj)
        
    print(f"Feature calculation finished for {len(objects_calculated)} objects.")
    return objects_calculated


# ==============================================================================
# FEATURE HELPER FUNCTIONS
# ==============================================================================

def calc_euclidean_distance(coords):
    """Calculates the direct air distance between start and end point of the line."""
    if len(coords) < 2:
        return 1.0
    start = coords[0]
    end = coords[-1]
    dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    return max(float(dist), 1.0)


def calc_jaggedness(local_mask):
    """
    Evaluates how jagged/zig-zag a line is using polygon approximation.
    A smooth arc requires few points to trace, a subgrain boundary requires many.
    Returns a ratio >= 1.0 (Higher = more jagged).
    """
    # Trace the detected line as a contour.
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0
        
    # Use the longest contour to represent the line.
    main_contour = max(contours, key=len)
    contour_len = len(main_contour)
    
    if contour_len < 5:
        return 1.0
        
    # Approximate the contour with the Douglas-Peucker algorithm.
    # A smaller epsilon preserves more detail from the original curve.
    epsilon = 0.015 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Compare contour complexity with the simplified point count.
    approx_len = len(approx)
    if approx_len == 0:
        return 1.0
        
    # Normalize the score so that 1.0 represents minimal jaggedness.
    jaggedness = approx_len / max(1.0, (contour_len / 50.0)) 
    return float(jaggedness)
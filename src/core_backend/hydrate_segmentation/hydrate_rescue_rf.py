import cv2
import numpy as np
import math
import pandas as pd

def extract_9_features(contour, roi_gray, rim_kernel=None):
    """
    Berechnet die 7 Form/Kontrast-Features für EIN spezifisches Kontur-Fragment 
    plus 2 globale ROI-Features (für den Umgebungskontext).
    """
    empty_features = {
        "area": 0.0, "aspect_ratio": 0.0, "circularity": 0.0,
        "solidity": 0.0, "fill_factor": 0.0, "bright": 0.0, "contrast": 1.0,
        "roi_std": 0.0, "roi_edge_density": 0.0
    }
    
    if contour is None or len(contour) < 5: return empty_features
        
    area = cv2.contourArea(contour)
    if area < 1.0: return empty_features
        
    perimeter = cv2.arcLength(contour, True)
    
    try:
        _, (ax1, ax2), _ = cv2.fitEllipse(contour)
    except cv2.error:
        return empty_features
        
    major_axis = max(ax1, ax2)
    minor_axis = min(ax1, ax2)
    aspect_ratio = major_axis / minor_axis if minor_axis > 1e-3 else 99.0
    
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    ellipse_area = (np.pi / 4.0) * major_axis * minor_axis
    fill_factor = area / ellipse_area if ellipse_area > 0 else 0.0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    x, y, w, h = cv2.boundingRect(contour)
    h_roi, w_roi = roi_gray.shape
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_roi, x + w), min(h_roi, y + h)
    
    local_img = roi_gray[y1:y2, x1:x2]
    local_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    contour_shifted = (contour - [x1, y1]).astype(np.int32)
    cv2.drawContours(local_mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)
    
    internal_pixels = local_img[local_mask == 255]
    bright = float(np.percentile(internal_pixels, 75)) if internal_pixels.size > 0 else 0.0
    
    if rim_kernel is None:
        rim_kernel = np.ones((3, 3), np.uint8)
        
    rim_mask = cv2.dilate(local_mask, rim_kernel, iterations=2) - local_mask
    rim_pixels = local_img[rim_mask == 255]
    bright_rim = float(np.mean(rim_pixels)) if rim_pixels.size > 0 else 1.0
    contrast = bright / max(bright_rim, 1.0)
    
    # Lokale Umgebung (Context)
    roi_std = float(np.std(roi_gray))
    edges = cv2.Canny(roi_gray, 50, 150)
    roi_edge_density = float(np.sum(edges) / 255.0) / (h_roi * w_roi + 1e-5)
    
    return {
        "area": float(area),
        "aspect_ratio": float(aspect_ratio),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "fill_factor": float(fill_factor),
        "bright": float(bright),
        "contrast": float(contrast),
        "roi_std": roi_std,
        "roi_edge_density": roi_edge_density
    }

def get_valid_contours(binary_img, filter_border, min_area):
    """Sucht Konturen und filtert Randgänger und zu kleine Schnipsel aus."""
    cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    h, w = binary_img.shape
    for c in cnts:
        area = cv2.contourArea(c)
        if len(c) < 5 or area < min_area: continue
        
        if filter_border:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= 1 or by <= 1 or (bx + bw) >= (w - 1) or (by + bh) >= (h - 1):
                continue
        valid.append(c)
    return valid

def run_canny_engine(roi_gray, params):
    """PHASE 2.1: Canny Edge Rescue"""
    cp = params.get("canny_params", {})
    blur_k = cp.get("blur_kernel_size", 5)
    
    if blur_k > 1:
        img_prep = cv2.GaussianBlur(roi_gray, (blur_k, blur_k), 0)
    else:
        img_prep = roi_gray

    edges = cv2.Canny(img_prep, cp.get("canny_min_val", 12), cp.get("canny_max_val", 42))

    morph_k = cp.get("morph_close_kernel", 15)
    if morph_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=cp.get("morph_close_iter", 2))

    return get_valid_contours(edges, params["roi_params"]["filter_roi_border_touching"], cp.get("min_contour_area_px", 700.0))

def run_adaptive_engine(roi_gray, params):
    """PHASE 2.2: Adaptive Threshold Rescue"""
    ap = params.get("adaptive_params", {})
    blur_k = ap.get("blur_kernel_size", 9)
    
    if blur_k > 1:
        img_prep = cv2.GaussianBlur(roi_gray, (blur_k, blur_k), 0)
    else:
        img_prep = roi_gray

    block_size = max(3, ap.get("block_size", 37))
    if block_size % 2 == 0: block_size += 1
    
    thresh_type = cv2.THRESH_BINARY_INV if ap.get("invert_binary", True) else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(img_prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, ap.get("c_constant", 4))

    open_k = ap.get("morph_open_kernel", 1)
    if open_k > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    close_k = ap.get("morph_close_kernel", 15)
    if close_k > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=ap.get("morph_close_iter", 3))

    return get_valid_contours(binary, params["roi_params"]["filter_roi_border_touching"], ap.get("min_contour_area_px", 700.0))

def get_all_rescue_candidates(roi_gray, rescue_params, original_contour_local=None):
    """Führt Canny und Adaptive aus, bindet die EdgeDrawing-Kontur ein und berechnet für ALLE Konturen die 9 Features."""
    all_candidates = []
    rim_kernel = np.ones((3, 3), np.uint8)
    
    # Evaluiere die Originale Kontur mit
    if original_contour_local is not None and len(original_contour_local) >= 5:
        feats = extract_9_features(original_contour_local, roi_gray, rim_kernel)
        if feats["area"] > 0:
            all_candidates.append({"contour": original_contour_local, "features": feats, "source": "edgedrawing"})
    
    canny_cnts = run_canny_engine(roi_gray, rescue_params)
    for c in canny_cnts:
        feats = extract_9_features(c, roi_gray, rim_kernel)
        if feats["area"] > 0:
            all_candidates.append({"contour": c, "features": feats, "source": "canny"})
            
    adapt_cnts = run_adaptive_engine(roi_gray, rescue_params)
    for c in adapt_cnts:
        feats = extract_9_features(c, roi_gray, rim_kernel)
        if feats["area"] > 0:
            all_candidates.append({"contour": c, "features": feats, "source": "adaptive"})
            
    return all_candidates
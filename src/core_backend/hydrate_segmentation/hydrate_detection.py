import pathlib
import numpy as np
import tifffile
import json
import cv2
import math
import random
import logging
import sys
import time
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import pandas as pd
    import joblib
    from core_backend.hydrate_segmentation.hydrate_rescue_rf import get_all_rescue_candidates
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Pandas or Joblib missing. Random Forest rescue disabled.")

from core_backend.hydrate_segmentation.hydrate_focus import find_focus
from core_backend.hydrate_segmentation.diagnostics import plot_and_save_diagnostics

# =========================================================
# CENTRAL PIPELINE CONFIGURATIONS
# =========================================================

def get_constants_config_path():
    if getattr(sys, 'frozen', False):
        base_dir = pathlib.Path(sys.executable).parent
    else:
        base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    return base_dir / "hydrate_constants.json"

GLOBAL_CONSTANTS = {}

def load_global_constants():
    global GLOBAL_CONSTANTS
    config_path = get_constants_config_path()
    
    default_structure = {
        "global_settings": {
            "pixel_scale_fallback": 1.41,
            "default_overlay_amount": 5,
            "xy_threshold_um": 120.0,
            "stitching_max_um": 200.0,
            "z_step_um_override": -500.0,
            "persp_center_x": 1224,
            "persp_center_y": 1024,
            "persp_scale_per_layer": 0.0093,
            "focus_overlap_threshold": 0.2,
            "overlay_match_dist_px": 5.0,
            "overlay_alpha": 0.7,
            "overlay_beta": 0.3
        },
        "phase_1_detection": {
            "clip_limit": 1.0,
            "median_kernel_size": 7,
            "ed_anchor_thresh": 16,
            "ed_grad_thresh": 4,
            "ed_min_path_len": 50,
            "morph_kernel_size": 11
        },
        "phase_1_triage_bounds": {
            "loose_bounds": {
                "area_min_px": 200.0,
                "area_max_px": 100000.0,          
                "ellipse_area_max_px": 200000.0,  
                "aspect_max": 6.0,
                "circ_min": 0.0,
                "fill_min": 0.0,
                "solid_min": 0.0,
                "bright_min": 90.0,
                "contrast_min": 0.1,
                "contrast_max": 2.5
            },
            "strict_bounds": {
                "area_min_px": 500.0,
                "area_max_px": 35000.0,
                "ellipse_area_max_px": 30000.0,
                "aspect_max": 2.6,
                "circ_min": 0.5,
                "fill_min": 0.92,
                "solid_min": 0.95,
                "bright_min": 120.0,
                "contrast_min": 0.35,
                "contrast_max": 1.8
            }
        },
        "phase_2_rescue": {
            "roi_params": {
                "roi_factor": 2.5,
                "filter_roi_border_touching": True,
                "grid_size": 16,
                "min_fill_factor": 0.5,
                "min_area_um2": 3000.0
            },
            "canny_params": {
                "blur_kernel_size": 5,
                "canny_min_val": 12,
                "canny_max_val": 42,
                "morph_close_kernel": 15,
                "morph_close_iter": 2,
                "min_contour_area_px": 700.0
            },
            "adaptive_params": {
                "blur_kernel_size": 9,
                "block_size": 37,
                "c_constant": 4,
                "invert_binary": True,
                "morph_open_kernel": 1,
                "morph_close_kernel": 15,
                "morph_close_iter": 3,
                "min_contour_area_px": 700.0
            }
        },
        "calibration_settings": {
            "match_area_diff_max": 0.6,
            "match_aspect_diff_max": 0.75,
            "match_overlap_min": 0.15,
            "match_fallback_radius_px": 100.0,
            "fn_penalty_score": 1.0,
            "fp_penalty_score": 0.5,
            "fp_blur_radius_px": 60.0,
            "fixed_total_score_threshold": 3.0,
            "dummy_eval_penalty": 999999.0,
            "layer_diff_penalty_score": 50.0,
            "3d_distance_threshold": 80.0,
            "3d_roi_radius_factor": 1.0,
            "bounds_2d": [
                [5000.0, 20000.0], [3000.0, 8000.0], [0.1, 2.0], [0.1, 2.0], [0.1, 4.0],
                [0.3, 0.75], [10.0, 100.0], [0.1, 2.0], [0.5, 0.95], [10.0, 40.0], [0.1, 2.0],
                [0.5, 0.95], [5.0, 50.0], [0.1, 2.0], [100.0, 170.0], [0.1, 0.5], [0.1, 2.0],
                [0.9, 1.4], [0.1, 0.4], [0.1, 2.0]
            ],
            "bounds_3d": [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]
        },
        "ellipse_fitting": {
            "ellipse_search_radius_um": 400.0,
            "ellipse_grid_step_um": 10.0,
            "ellipse_density_radius_um": 30.0,
            "ellipse_p80_multiplier": 3.0,
            "ellipse_morph_closing_steps": 2,
            "ellipse_fit_padding_um": 60.0,
            "ellipse_params": {
                "top_bottom": {"cx": 0.0, "cy": 140.0, "width": 80.0, "height": 140.0, "angle_deg": 0.0},
                "left_right": {"cx": 140.0, "cy": 0.0, "width": 140.0, "height": 80.0, "angle_deg": 0.0},
                "topleft_bottomright": {"cx": 140.0, "cy": 140.0, "width": 140.0, "height": 80.0, "angle_deg": 45.0},
                "bottomleft_topright": {"cx": 140.0, "cy": -140.0, "width": 140.0, "height": 80.0, "angle_deg": -45.0}
            }
        },
        "rf_params": {
            "optimize": True,
            "n_iter": 500,
            "cv": 5,
            "scoring": "f1_macro",
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "extraction_batch_size": 2000
        }
    }

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                GLOBAL_CONSTANTS = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load hydrate_constants.json: {e}")
            GLOBAL_CONSTANTS = default_structure
    else:
        GLOBAL_CONSTANTS = default_structure
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(GLOBAL_CONSTANTS, f, indent=4)
        except Exception: pass

load_global_constants()

def calc_gaussian_score(x, mu, sigma):
    if sigma == 0: return 0.0
    return math.exp(-0.5 * ((x - mu) / sigma)**2)

def calc_sigmoid_score(x, x0, k):
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if -k * (x - x0) > 0 else 1.0

def gui_hydrate_detection(processed_data_dir, param_dir, rf_model_file, save_overlay, save_random_overlay, amount_str, save_global_mask, update_heatmap_cb=None):
    try:
        amount = int(amount_str) if amount_str else int(GLOBAL_CONSTANTS["global_settings"]["default_overlay_amount"])
    except ValueError:
        amount = int(GLOBAL_CONSTANTS["global_settings"]["default_overlay_amount"])

    processed_data_path = pathlib.Path(processed_data_dir)
    all_detected_hydrates = []
    
    with open(param_dir, "r") as f:
        scoring_params = json.load(f)
        
    scoring_params.setdefault("factor_contrast", 1.0)
    scoring_params.setdefault("factor_laplacian", 1.0)
    scoring_params.setdefault("factor_thickness_edge", 1.0)

    # Load the rescue model from the selected path when it is available.
    rf_model = None
    if ML_AVAILABLE and rf_model_file and pathlib.Path(rf_model_file).exists():
        try:
            rf_model = joblib.load(rf_model_file)
            logging.info(f"Loaded Random Forest rescue model from {rf_model_file}")
        except Exception as e:
            logging.error(f"Failed to load RF model: {e}")

    metadata_preprocessed = None
    for file in processed_data_path.iterdir():
        if file.stem == "metadata_preprocessed":
            with open(file, "r") as f:
                metadata_preprocessed = json.load(f)

    if metadata_preprocessed is None:
        logging.error("metadata_preprocessed.json could not be found!")
        return

    if update_heatmap_cb is not None and "Positions" in metadata_preprocessed:
        update_heatmap_cb(0, 0, 0, is_border=False, init_positions=metadata_preprocessed["Positions"])
    
    stacks_for_overlay = []
    if save_overlay or save_random_overlay:
        for pos in metadata_preprocessed["Positions"]:
            stacks_for_overlay.append((pos["gridX"], pos["gridY"]))
            
    if save_random_overlay:
        stacks_for_overlay = random.sample(stacks_for_overlay, min(amount, len(stacks_for_overlay)))

    det_config = GLOBAL_CONSTANTS["phase_1_detection"]
    clahe_obj = cv2.createCLAHE(clipLimit=det_config.get("clip_limit", 1.0), tileGridSize=(8, 8))
    rim_kernel = np.ones((3, 3), np.uint8)

    for image_stack_path in sorted(processed_data_path.iterdir()):
        if image_stack_path.is_file() and image_stack_path.suffix == ".tif":
            stack = load_tiff_stack(image_stack_path)
            objects_in_stack = []
            all_objects_for_overlay = [] 

            parts = image_stack_path.stem.split("_")
            x_current = int(parts[-2])
            y_current = int(parts[-1])

            pos_meta = next((p for p in metadata_preprocessed["Positions"] if p["gridX"] == x_current and p["gridY"] == y_current), None)
            is_border = pos_meta.get("border", False) if pos_meta else False

            for layer in range(stack.shape[0]):
                raw_image = stack[layer]
                
                image_preprocessed = preprocess_image(raw_image, det_config, clahe=clahe_obj)
                image_binary = edge_detection(image_preprocessed, det_config)
                
                extracted_objects = extract_and_filter_objects(
                    image_binary, image_preprocessed, raw_image, metadata_preprocessed, 
                    image_stack_path.stem, layer, scoring_params, rf_model, rim_kernel
                )
                
                valid_objects = [obj for obj in extracted_objects if not obj.get("is_aborted", False)]
                objects_in_stack.extend(valid_objects)
                
                all_objects_for_overlay.extend(extracted_objects) 
                
                if (x_current, y_current) in stacks_for_overlay:
                    all_detected_hydrates.extend(extracted_objects)
                else:
                    all_detected_hydrates.extend(valid_objects)

            find_focus(objects_in_stack, stack, scoring_params)

            in_focus_hydrates_count = len([obj for obj in objects_in_stack if obj.get("focus", False) and obj.get("total_score", 0) >= scoring_params["total_score_threshold"]])
            if update_heatmap_cb is not None:
                update_heatmap_cb(x_current, y_current, in_focus_hydrates_count, is_border=is_border)
            
            if (x_current, y_current) in stacks_for_overlay:
                save_pipeline_overlay(stack, all_objects_for_overlay, scoring_params, image_stack_path, processed_data_path)
            
    export_analysis_results(
        [o for o in all_detected_hydrates if not o.get("is_aborted", False)], 
        scoring_params, 
        processed_data_path, 
        metadata_preprocessed, 
        GLOBAL_CONSTANTS["global_settings"]["xy_threshold_um"], 
        GLOBAL_CONSTANTS["global_settings"]["stitching_max_um"],
        all_objects_with_aborted=all_detected_hydrates
    )

def load_tiff_stack(stack_path: pathlib.Path) -> np.ndarray:
    with tifffile.TiffFile(str(stack_path)) as tif:
        if len(tif.pages) > 1:
            stack = np.stack([page.asarray() for page in tif.pages], axis=0)
        else:
            stack = tif.asarray()
    if stack.ndim == 2: stack = np.expand_dims(stack, axis=0)
    elif stack.ndim == 3:
        min_axis = int(np.argmin(stack.shape))
        if min_axis != 0 and stack.shape[min_axis] < stack.shape[0]:
            stack = np.moveaxis(stack, min_axis, 0)
    return stack 

def preprocess_image(raw_image, det_config, clahe=None):
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=det_config.get("clip_limit", 1.0), tileGridSize=(8, 8))
    clahe_img = clahe.apply(raw_image)
    k_med = det_config.get("median_kernel_size", 7)
    return cv2.medianBlur(clahe_img, k_med) if k_med > 1 else clahe_img

def edge_detection(image_preprocessed, det_config):
    ed = cv2.ximgproc.createEdgeDrawing()
    ed_params = ed.Params()
    ed_params.MinPathLength = int(det_config.get("ed_min_path_len", 50))
    ed_params.AnchorThresholdValue = int(det_config.get("ed_anchor_thresh", 20))
    ed_params.ScanInterval = 1  
    ed_params.GradientThresholdValue = int(det_config.get("ed_grad_thresh", 5))
    ed.setParams(ed_params)
    
    try:
        ed.detectEdges(image_preprocessed)
        binary = ed.getEdgeImage()
        if binary is None: binary = np.zeros_like(image_preprocessed)
    except Exception:
        binary = np.zeros_like(image_preprocessed)
        
    # Fill detected contours and remove narrow artifacts before closing gaps.
    if np.any(binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(binary)
        valid_contours = [c for c in contours if len(c) >= 3]
        cv2.drawContours(filled_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        kernel_open = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_open)
    
    k_morph = det_config.get("morph_kernel_size", 11)
    if k_morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def get_base_metrics(contour, image_preprocessed, rim_kernel):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        major_axis = max(w, h)
        minor_axis = min(w, h)
        aspect_ratio = major_axis / minor_axis if minor_axis > 1e-3 else 99.0
        ellipse_area = (np.pi / 4.0) * major_axis * minor_axis
    except cv2.error:
        bx, by, bw, bh = cv2.boundingRect(contour)
        cx, cy = bx + bw/2.0, by + bh/2.0
        major_axis, minor_axis, angle, ellipse_area = max(bw, bh), min(bw, bh), 0.0, 999999.0
        aspect_ratio = major_axis / minor_axis if minor_axis > 1e-3 else 99.0

    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    ellipse_fill_factor = area / ellipse_area if ellipse_area > 0 else 0.0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    bx, by, bw, bh = cv2.boundingRect(contour)
    h_img, w_img = image_preprocessed.shape
    x1, y1 = max(0, bx), max(0, by)
    x2, y2 = min(w_img, bx+bw), min(h_img, by+bh)
    
    local_img = image_preprocessed[y1:y2, x1:x2]
    local_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    contour_local = (contour - [x1, y1]).astype(np.int32)
    cv2.drawContours(local_mask, [contour_local], -1, 255, thickness=cv2.FILLED)
    
    internal_pixels = local_img[local_mask == 255]
    brightness_core = float(np.percentile(internal_pixels, 75)) if internal_pixels.size > 0 else 0.0
    
    if rim_kernel is None: rim_kernel = np.ones((3, 3), np.uint8)
    rim_mask = cv2.dilate(local_mask, rim_kernel, iterations=2) - local_mask
    rim_pixels = local_img[rim_mask == 255]
    brightness_rim = float(np.mean(rim_pixels)) if rim_pixels.size > 0 else 1.0
    contrast_ratio = brightness_core / max(brightness_rim, 1.0)
    
    return area, aspect_ratio, circularity, ellipse_fill_factor, solidity, brightness_core, contrast_ratio, angle, (major_axis, minor_axis), cx, cy, ellipse_area


def extract_and_filter_objects(image_binary, image_preprocessed, raw_image, metadata, image_stack, layer, scoring_params, rf_model, rim_kernel):
    # Track time spent generating rescue candidates and running inference.
    t_start_extract = time.perf_counter()
    time_rescue_cands = 0.0
    time_rf_inference = 0.0

    g_sets = GLOBAL_CONSTANTS["global_settings"]
    camera_meta = metadata.get("Camera_preprocessed", {}) if metadata else {}
    pixel_scale = float(camera_meta.get("pixelSizeUm", g_sets.get("pixel_scale_fallback", 1.41)))

    x_current = int(image_stack.split("_")[-2])
    y_current = int(image_stack.split("_")[-1])
    pos_meta = next((p for p in metadata["Positions"] if p["gridX"] == x_current and p["gridY"] == y_current), None)
    origin_image_x = pos_meta["positionX"]
    origin_image_y = pos_meta["positionY"]

    contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_objects = []
    hydrate_id = 1
    
    lb = GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]
    sb = GLOBAL_CONSTANTS["phase_1_triage_bounds"]["strict_bounds"]
    rescue_params = GLOBAL_CONSTANTS["phase_2_rescue"]
    
    min_area_um2 = rescue_params.get("roi_params", {}).get("min_area_um2", 3000.0)
    min_fill_factor = rescue_params.get("roi_params", {}).get("min_fill_factor", 0.5)

    for contour in contours:
        raw_area = cv2.contourArea(contour)
        if len(contour) < 5 or raw_area < 20.0:  
            continue

        metrics = get_base_metrics(contour, image_preprocessed, rim_kernel)
        area, aspect, circ, fill, solid, bright, contrast, angle, axes, cx, cy, ellipse_area = metrics
        major_axis, minor_axis = axes
        
        area_um2 = area * (pixel_scale * pixel_scale)

        passes_loose = (
            lb["area_min_px"] <= area <= lb["area_max_px"] and
            ellipse_area <= lb["ellipse_area_max_px"] and
            aspect <= lb["aspect_max"] and
            circ >= lb["circ_min"] and
            fill >= lb["fill_min"] and
            solid >= lb["solid_min"] and
            bright >= lb["bright_min"] and
            lb["contrast_min"] <= contrast <= lb["contrast_max"]
        )

        passes_strict = (
            sb["area_min_px"] <= area <= sb["area_max_px"] and
            ellipse_area <= sb["ellipse_area_max_px"] and
            aspect <= sb["aspect_max"] and
            circ >= sb["circ_min"] and
            fill >= sb["fill_min"] and
            solid >= sb["solid_min"] and
            bright >= sb["bright_min"] and
            sb["contrast_min"] <= contrast <= sb["contrast_max"]
        )

        # Send otherwise strict candidates through rescue when their physical area is too small.
        if passes_strict and area_um2 < min_area_um2:
            passes_strict = False

        final_contour = contour
        rescue_contour_px = None
        is_aborted = False
        abort_reason = "None"
        pipeline_path = "init"
        source_category_str = "1_good"

        if not passes_loose:
            is_aborted = True
            abort_reason = "Triage: Trash (Failed Loose Bounds)"
            pipeline_path += " -> trash"
            source_category_str = "3_trash"
        
        elif passes_strict:
            pipeline_path += " -> good"
            source_category_str = "1_good"
            
        else:
            pipeline_path += " -> moderate"
            source_category_str = "2_moderate"
            
            if rf_model is not None and ML_AVAILABLE:
                roi_factor = rescue_params["roi_params"]["roi_factor"]
                radius = (major_axis * roi_factor) / 2.0
                h_img, w_img = raw_image.shape
                
                rx1 = max(0, int(cx - radius))
                rx2 = min(w_img, int(cx + radius))
                ry1 = max(0, int(cy - radius))
                ry2 = min(h_img, int(cy + radius))

                if (rx2 - rx1) >= 15 and (ry2 - ry1) >= 15:
                    raw_roi = raw_image[ry1:ry2, rx1:rx2].copy()
                    orig_cnt_local = (contour - [rx1, ry1]).astype(np.int32)

                    t_resc_start = time.perf_counter()
                    rescue_cands = get_all_rescue_candidates(raw_roi, rescue_params, orig_cnt_local)
                    time_rescue_cands += (time.perf_counter() - t_resc_start)
                    
                    if rescue_cands:
                        # Apply minimum-area and fill-factor requirements before classification.
                        filtered_cands = []
                        for c in rescue_cands:
                            c_area_um2 = c["features"]["area"] * (pixel_scale * pixel_scale)
                            if c_area_um2 >= min_area_um2 and c["features"]["fill_factor"] >= min_fill_factor:
                                filtered_cands.append(c)

                        if not filtered_cands:
                            is_aborted = True
                            abort_reason = f"RF Bouncer: Trash (Area < {min_area_um2} um2 or Fill < {min_fill_factor})"
                            pipeline_path += " -> trash"
                        else:
                            t_rf_start = time.perf_counter()
                            df = pd.DataFrame([c["features"] for c in filtered_cands])
                            probs = rf_model.predict_proba(df)
                            preds = rf_model.predict(df)
                            time_rf_inference += (time.perf_counter() - t_rf_start)
                            
                            classes = list(rf_model.classes_)
                            if 4 in classes:
                                idx_4 = classes.index(4)
                                score = 1.0 - probs[:, idx_4]
                            else:
                                score = np.ones(len(filtered_cands))
                                
                            best_idx = np.argmax(score)
                            best_cand = filtered_cands[best_idx]
                            best_pred = preds[best_idx]
                            
                            if best_pred == 4:
                                is_aborted = True
                                abort_reason = "RF: Class 4 (Artifact)"
                                pipeline_path += " -> trash"
                            else:
                                rescue_contour_px = best_cand["contour"] + [rx1, ry1]
                                final_contour = rescue_contour_px
                                algo_used = best_cand["source"]
                                pipeline_path += f" -> {algo_used}_rescued"
                                source_category_str = f"2_{algo_used}"
                                
                                metrics = get_base_metrics(final_contour, image_preprocessed, rim_kernel)
                                area, aspect, circ, fill, solid, bright, contrast, angle, axes, cx, cy, ellipse_area = metrics
                                major_axis, minor_axis = axes
                                area_um2 = area * (pixel_scale * pixel_scale) 
                    else:
                        is_aborted = True
                        abort_reason = "Rescue: No valid contours found in ROI"
                        pipeline_path += " -> trash"
                else:
                    is_aborted = True
                    abort_reason = "Rescue: ROI too small or invalid"
                    pipeline_path += " -> trash"
            else:
                is_aborted = True
                abort_reason = "Rescue: No RF Model loaded"
                pipeline_path += " -> trash"

        total_score = 0.0
        if not is_aborted:
            score_area = calc_gaussian_score(area, scoring_params.get("area_mu", 15000), scoring_params.get("area_sigma", 5000))
            score_aspect = calc_gaussian_score(aspect, scoring_params.get("aspect_mu", 1.0), scoring_params.get("aspect_sigma", 0.5))
            score_circ = calc_sigmoid_score(circ, scoring_params.get("circ_x0", 0.7), scoring_params.get("circ_k", 25))
            score_fill = calc_sigmoid_score(fill, scoring_params.get("fill_x0", 0.75), scoring_params.get("fill_k", 25))
            score_solid = calc_sigmoid_score(solid, scoring_params.get("solid_x0", 0.85), scoring_params.get("solid_k", 30))
            score_bright = calc_sigmoid_score(bright, scoring_params.get("bright_x0", 100), scoring_params.get("bright_k", 0.2))
            score_contrast = calc_gaussian_score(contrast, scoring_params.get("contrast_mu", 1.1), scoring_params.get("contrast_sigma", 0.3))

            total_score = (
                score_area * scoring_params.get("weight_area", 1.0) +
                score_aspect * scoring_params.get("weight_aspect", 1.0) +
                score_circ * scoring_params.get("weight_circ", 1.0) +
                score_fill * scoring_params.get("weight_fill", 1.0) +
                score_solid * scoring_params.get("weight_solid", 1.0) +
                score_bright * scoring_params.get("weight_bright", 1.0) +
                score_contrast * scoring_params.get("weight_contrast", 1.0)
            )
            
            if total_score >= scoring_params.get("total_score_threshold", 5.0):
                pipeline_path += " -> hydrate"
            else:
                is_aborted = True
                abort_reason = "Phase 3: Fuzzy Score failed"
                pipeline_path += " -> trash"

        object_id = layer * 1000000 + hydrate_id
        
        corr_x = g_sets["persp_center_x"] + (cx - g_sets["persp_center_x"]) * (1 + g_sets["persp_scale_per_layer"] * layer)
        corr_y = g_sets["persp_center_y"] + (cy - g_sets["persp_center_y"]) * (1 + g_sets["persp_scale_per_layer"] * layer) 

        shifted_contour = np.zeros_like(final_contour, dtype=np.float32)
        shifted_contour[:, 0, 0] = g_sets["persp_center_x"] + (final_contour[:, 0, 0] - g_sets["persp_center_x"]) * (1 + g_sets["persp_scale_per_layer"] * layer)
        shifted_contour[:, 0, 1] = g_sets["persp_center_y"] + (final_contour[:, 0, 1] - g_sets["persp_center_y"]) * (1 + g_sets["persp_scale_per_layer"] * layer)
        shifted_contour = shifted_contour.astype(np.int32)
        shifted_bbox = cv2.boundingRect(shifted_contour)

        extracted_objects.append({
            "id": object_id,
            "source_stack": image_stack,
            "pipeline_path": pipeline_path,
            "source_category_str": source_category_str,
            "total_score": total_score,  
            "is_aborted": is_aborted,     
            "abort_reason": abort_reason,
            "triage_contour_px": contour,           
            "rescue_contour_px": rescue_contour_px, 
            "contour_image_px": final_contour,      
            "area_px": area,
            "area_um2": area_um2, 
            "aspect_ratio": aspect,
            "circularity": circ,
            "solidity": solid,
            "ellipse_fill_factor": fill,
            "core_brightness_75th": bright,
            "core_rim_contrast": contrast,
            "major_axis_um": major_axis * pixel_scale,
            "minor_axis_um": minor_axis * pixel_scale,
            "ellipse_angle_deg": angle,
            "center_x_px_image": cx,            
            "center_y_px_image": cy,            
            "center_x_um_world": corr_x * pixel_scale + origin_image_x, 
            "center_y_um_world": corr_y * pixel_scale + origin_image_y, 
            "center_z_um_world": layer * g_sets["z_step_um_override"],
            "layer": layer,
            "focus_value": 0,
            "focus": False,
            "is_duplicate": False,  
            "shifted_contour": shifted_contour,
            "shifted_bbox": shifted_bbox        
        })
        hydrate_id += 1

    t_total_extract = time.perf_counter() - t_start_extract
    logging.debug(f"    -> Extract Details: Contours: {len(contours)} | Rescue-Cands Gen: {time_rescue_cands:.3f}s | RF-Inference (Pandas+Predict): {time_rf_inference:.3f}s")
    
    return extracted_objects

def save_pipeline_overlay(stack, all_objects_in_stack, parameters, stack_path, save_dir, factor=1):
    g_sets = GLOBAL_CONSTANTS["global_settings"]
    
    # Store overlay images in a separate directory for each stack.
    overlay_dir = pathlib.Path(save_dir) / "overlays" / stack_path.stem
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect the three diagnostic panels for multipage TIFF output.
    overlay_frames_p1 = []
    overlay_frames_p2 = []
    overlay_frames_p3 = []
    
    # Add a two-line legend scaled to the panel dimensions.
    def add_legend_to_image(img, items):
        img_h, img_w = img.shape[:2]
        
        # Reserve enough height for the legend at all image sizes.
        legend_h = max(160, int(img_h * 0.15)) 
        legend_bg = np.ones((legend_h, img_w, 3), dtype=np.uint8) * 255
        
        num_items = len(items)
        section_width = img_w // num_items
        
        # Scale the legend typography with the source image width.
        target_font_scale = max(1.2, img_w / 1000.0) 
        # Cap the font size so labels fit within their panel sections.
        max_allowed_scale = (section_width * 0.85) / 350.0 
        
        font_scale = min(target_font_scale, max_allowed_scale)
        
        # Scale the legend swatches with the text.
        box_size = int(60 * (font_scale / 1.0))
        thickness = max(2, int(2.5 * font_scale))
        
        for i, (text_line1, text_line2, color) in enumerate(items):
            # Add a small inset within each legend section.
            x_offset = (i * section_width) + int(section_width * 0.04) 
            y_center = legend_h // 2
            
            # Draw the color swatch.
            box_y1 = y_center - box_size // 2
            box_y2 = y_center + box_size // 2
            
            cv2.rectangle(legend_bg, (x_offset, box_y1), (x_offset + box_size, box_y2), color, -1)
            # Outline the swatch for contrast against the legend background.
            cv2.rectangle(legend_bg, (x_offset, box_y1), (x_offset + box_size, box_y2), (0, 0, 0), 2)
            
            # Place the two-line label beside the swatch.
            text_x = x_offset + box_size + max(15, int(25 * font_scale))
            
            # Position the two lines around the legend's vertical center.
            y_line1 = y_center - int(10 * font_scale)
            y_line2 = y_center + int(35 * font_scale)
            
            cv2.putText(legend_bg, text_line1, (text_x, y_line1), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(legend_bg, text_line2, (text_x, y_line2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                        
        return np.vstack((img, legend_bg))

    # Generate one set of diagnostic panels for each stack layer.
    for layer in range(stack.shape[0]):
        raw_image = stack[layer].copy() 
        objects_in_layer = [obj for obj in all_objects_in_stack if obj.get("layer") == layer]
        
        # Panel 1: initial triage classification.
        panel1 = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        for hyd in objects_in_layer:
            cat_str = hyd.get("source_category_str", "")
            if "3_trash" in cat_str:
                cv2.drawContours(panel1, [hyd["triage_contour_px"].astype(np.int32)], -1, (0, 0, 255), 2) 
            elif "1_good" in cat_str:
                cv2.drawContours(panel1, [hyd["triage_contour_px"].astype(np.int32)], -1, (0, 255, 0), 2) 
            elif "2_" in cat_str:
                cv2.drawContours(panel1, [hyd["triage_contour_px"].astype(np.int32)], -1, (0, 215, 255), 2) 

        # Panel 2: fuzzy-score classification.
        panel2_bg = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        mask2 = np.zeros_like(panel2_bg)
        
        for hyd in objects_in_layer:
            if hyd.get("is_aborted", False) and "Fuzzy Score" not in hyd.get("abort_reason", ""): 
                continue 
                
            score_ok = hyd.get("total_score", 0) >= parameters.get("total_score_threshold", 5.0)
            color = (0,0,0)
            
            if "1_good" in hyd.get("source_category_str", ""):
                color = (0, 255, 0) if score_ok else (0, 100, 0)
            elif "2_" in hyd.get("source_category_str", ""):
                color = (255, 255, 0) if score_ok else (139, 0, 0)
                
            if color != (0,0,0) and "contour_image_px" in hyd:
                cv2.drawContours(mask2, [hyd["contour_image_px"].astype(np.int32)], -1, color, thickness=cv2.FILLED)
        
        # Blend only the colored mask pixels so the unclassified background remains unchanged.
        panel2 = panel2_bg.copy()
        mask_active_2 = np.any(mask2 > 0, axis=-1)
        panel2[mask_active_2] = cv2.addWeighted(
            panel2_bg, g_sets["overlay_alpha"], mask2, g_sets["overlay_beta"], 0
        )[mask_active_2]

        # Panel 3: focus classification.
        panel3_bg = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        mask3 = np.zeros_like(panel3_bg)
        
        for hyd in objects_in_layer:
            if hyd.get("is_aborted", False) or hyd.get("total_score", 0) < parameters.get("total_score_threshold", 5.0): 
                continue
                
            is_focus = hyd.get("focus", False)
            color = (0, 255, 0) if is_focus else (0, 165, 255) 
            
            if "contour_image_px" in hyd:
                cv2.drawContours(mask3, [hyd["contour_image_px"].astype(np.int32)], -1, color, thickness=cv2.FILLED)
                
        # Blend only the colored mask pixels over the source image.
        panel3 = panel3_bg.copy()
        mask_active_3 = np.any(mask3 > 0, axis=-1)
        panel3[mask_active_3] = cv2.addWeighted(
            panel3_bg, g_sets["overlay_alpha"], mask3, g_sets["overlay_beta"], 0
        )[mask_active_3]

        # Apply the requested output scale to each panel.
        if factor != 1:
            new_width = int(raw_image.shape[1] * factor)
            new_height = int(raw_image.shape[0] * factor)
            panel1 = cv2.resize(panel1, (new_width, new_height), interpolation=cv2.INTER_AREA)
            panel2 = cv2.resize(panel2, (new_width, new_height), interpolation=cv2.INTER_AREA)
            panel3 = cv2.resize(panel3, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Append a legend to each diagnostic panel.
        items_p1 = [
            ("Triage Class I:", "Primary Candidate", (0, 255, 0)), 
            ("Triage Class II:", "Marginal Candidate", (0, 215, 255)), 
            ("Triage Class III:", "Artifact / Rejected", (0, 0, 255))
        ]
        panel1 = add_legend_to_image(panel1, items_p1)
        
        items_p2 = [
            ("Primary Candidate:", "Score >= Threshold", (0, 255, 0)), 
            ("Primary Candidate:", "Score < Threshold", (0, 100, 0)), 
            ("Recovered Cand.:", "Score >= Threshold", (255, 255, 0)), 
            ("Recovered Cand.:", "Score < Threshold", (139, 0, 0))
        ]
        panel2 = add_legend_to_image(panel2, items_p2)
        
        items_p3 = [
            ("Focal Plane:", "Current Layer", (0, 255, 0)), 
            ("Focal Plane:", "Other Layer", (0, 165, 255))
        ]
        panel3 = add_legend_to_image(panel3, items_p3)

        # Convert each panel to RGB before adding it to the output frame list.
        overlay_frames_p1.append(cv2.cvtColor(panel1, cv2.COLOR_BGR2RGB))
        overlay_frames_p2.append(cv2.cvtColor(panel2, cv2.COLOR_BGR2RGB))
        overlay_frames_p3.append(cv2.cvtColor(panel3, cv2.COLOR_BGR2RGB))
        
    # Save each diagnostic panel as a multipage TIFF stack.
    if overlay_frames_p1:
        stack_name = stack_path.stem
        
        arr_p1 = np.stack(overlay_frames_p1, axis=0)
        arr_p2 = np.stack(overlay_frames_p2, axis=0)
        arr_p3 = np.stack(overlay_frames_p3, axis=0)
        
        tifffile.imwrite(overlay_dir / f"{stack_name}_01_triage.tif", arr_p1, photometric='rgb')
        tifffile.imwrite(overlay_dir / f"{stack_name}_02_score.tif", arr_p2, photometric='rgb')
        tifffile.imwrite(overlay_dir / f"{stack_name}_03_focus.tif", arr_p3, photometric='rgb')

def estimate_ellipse_params_auto(vectors: list, fallback_params: dict) -> tuple[dict, dict]:
    e_sets = GLOBAL_CONSTANTS["ellipse_fitting"]
    search_radius = e_sets["ellipse_search_radius_um"]
    grid_step = e_sets["ellipse_grid_step_um"]
    density_radius = e_sets["ellipse_density_radius_um"]
    p80_multiplier = e_sets["ellipse_p80_multiplier"]

    valid_vectors = [v for v in vectors if math.hypot(v[0], v[1]) <= search_radius]
    grid_1d = np.arange(-search_radius, search_radius + grid_step, grid_step)
    gx, gy = np.meshgrid(grid_1d, grid_1d)
    grid_points = np.c_[gx.ravel(), gy.ravel()]
    
    if len(valid_vectors) < 5:
        return fallback_params, {"gx": gx, "gy": gy, "counts": np.zeros(gx.shape), "p80": 0.0}

    tree = cKDTree(np.array(valid_vectors))
    counts = np.array(tree.query_ball_point(grid_points, r=density_radius, return_length=True))

    p80 = np.percentile(counts, 80)
    baseline = max(1.0, float(p80))
    threshold = baseline * p80_multiplier
    grid_debug_data = {"gx": gx, "gy": gy, "counts": counts.reshape(gx.shape), "p80": baseline}

    binary_mask = (counts >= threshold).astype(np.uint8)
    grid_size = int(np.sqrt(len(binary_mask)))
    binary_img = binary_mask.reshape((grid_size, grid_size))

    closing_steps = int(e_sets.get("ellipse_morph_closing_steps", 2))
    if closing_steps > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=closing_steps)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    if num_labels <= 1: return fallback_params, grid_debug_data
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cluster_indices = np.where(labels.ravel() == largest_label)[0]
    cluster_points = np.c_[gx.ravel()[cluster_indices], gy.ravel()[cluster_indices]]
    
    if len(cluster_points) < 5: return fallback_params, grid_debug_data
        
    pts = np.array(cluster_points, dtype=np.float32).reshape(-1, 1, 2)
    (cx, cy), (width, height), angle_deg = cv2.fitEllipse(pts)
    
    padding_um = e_sets["ellipse_fit_padding_um"]
    width = max(width + padding_um, 40.0)
    height = max(height + padding_um, 40.0)

    return {"cx": float(cx), "cy": float(cy), "width": float(width), "height": float(height), "angle_deg": float(angle_deg)}, grid_debug_data

def mark_duplicates_and_fit_ellipses(all_objects: list, parameters: dict, xy_threshold_um: float = 120.0, stitching_max_um: float = 200.0) -> tuple[int, int, int, dict, dict]:
    if not all_objects: return 0, 0, 0, {}, {}

    vectors_dict = {"top_bottom": [], "left_right": [], "topleft_bottomright": [], "bottomleft_topright": []}
    valid_focus_objects = [obj for obj in all_objects if obj.get("focus", False) and obj.get("total_score", 0) >= parameters.get("total_score_threshold", 4.0)]
    
    for i, hyd1 in enumerate(valid_focus_objects):
        for j, hyd2 in enumerate(valid_focus_objects):
            if j <= i: continue
            cand_stack = hyd1.get("source_stack", "")
            kept_stack = hyd2.get("source_stack", "")
            if cand_stack == kept_stack: continue
                
            try:
                c_gx, c_gy = int(cand_stack.split("_")[-2]), int(cand_stack.split("_")[-1].split(".")[0])
                k_gx, k_gy = int(kept_stack.split("_")[-2]), int(kept_stack.split("_")[-1].split(".")[0])
            except (ValueError, IndexError): continue

            dx_grid, dy_grid = c_gx - k_gx, c_gy - k_gy
            cand_x, cand_y = float(hyd1["center_x_um_world"]), float(hyd1["center_y_um_world"])
            kept_x, kept_y = float(hyd2["center_x_um_world"]), float(hyd2["center_y_um_world"])

            if dx_grid == 0 and dy_grid == 1: vectors_dict["top_bottom"].append((cand_x - kept_x, cand_y - kept_y))
            elif dx_grid == 0 and dy_grid == -1: vectors_dict["top_bottom"].append((kept_x - cand_x, kept_y - cand_y))
            elif dx_grid == 1 and dy_grid == 0: vectors_dict["left_right"].append((cand_x - kept_x, cand_y - kept_y))
            elif dx_grid == -1 and dy_grid == 0: vectors_dict["left_right"].append((kept_x - cand_x, kept_y - cand_y))
            elif dx_grid == 1 and dy_grid == 1: vectors_dict["topleft_bottomright"].append((cand_x - kept_x, cand_y - kept_y))
            elif dx_grid == -1 and dy_grid == -1: vectors_dict["topleft_bottomright"].append((kept_x - cand_x, kept_y - cand_y))
            elif dx_grid == 1 and dy_grid == -1: vectors_dict["bottomleft_topright"].append((cand_x - kept_x, cand_y - kept_y))
            elif dx_grid == -1 and dy_grid == 1: vectors_dict["bottomleft_topright"].append((kept_x - cand_x, kept_y - cand_y))

    fallback_ellipses = GLOBAL_CONSTANTS["ellipse_fitting"]["ellipse_params"]
    auto_ellipse_params = {}
    grid_debug_dict = {}
    for key, fallback in fallback_ellipses.items():
        params, grid_data = estimate_ellipse_params_auto(vectors_dict[key], fallback)
        auto_ellipse_params[key] = params
        grid_debug_dict[key] = grid_data

    def is_in_ellipse(vx, vy, params):
        cx, cy = params["cx"], params["cy"]
        a, b = params["width"] / 2.0, params["height"] / 2.0
        theta = math.radians(params["angle_deg"])
        cos_t, sin_t = math.cos(theta), math.sin(theta) 
        tx, ty = vx - cx, vy - cy
        rx, ry = tx * cos_t + ty * sin_t, -tx * sin_t + ty * cos_t
        return (rx / a)**2 + (ry / b)**2 <= 1.0

    for obj in all_objects: obj["is_duplicate"] = False
    sorted_focus = sorted(valid_focus_objects, key=lambda obj: obj.get("focus_value", 0.0), reverse=True)

    unique_kept = []
    removed_count, potential_2_layer, potential_stitching = 0, 0, 0

    for candidate in sorted_focus:
        cand_x, cand_y = float(candidate["center_x_um_world"]), float(candidate["center_y_um_world"])
        cand_layer, cand_stack = int(candidate["layer"]), candidate.get("source_stack", "")
        is_duplicate = False

        for kept in unique_kept:
            kept_stack = kept.get("source_stack", "")
            if cand_stack == kept_stack: continue

            kept_x, kept_y = float(kept["center_x_um_world"]), float(kept["center_y_um_world"])
            kept_layer = int(kept["layer"])
            layer_diff = abs(cand_layer - kept_layer)

            try:
                cand_gx, cand_gy = int(cand_stack.split("_")[-2]), int(cand_stack.split("_")[-1].split(".")[0])
                kept_gx, kept_gy = int(kept_stack.split("_")[-2]), int(kept_stack.split("_")[-1].split(".")[0])
            except (ValueError, IndexError): continue

            dx_grid, dy_grid = cand_gx - kept_gx, cand_gy - kept_gy
            vector, ellipse_type = None, None

            if dx_grid == 0 and dy_grid == 1: vector, ellipse_type = (cand_x - kept_x, cand_y - kept_y), "top_bottom"
            elif dx_grid == 0 and dy_grid == -1: vector, ellipse_type = (kept_x - cand_x, kept_y - cand_y), "top_bottom"
            elif dx_grid == 1 and dy_grid == 0: vector, ellipse_type = (cand_x - kept_x, cand_y - kept_y), "left_right"
            elif dx_grid == -1 and dy_grid == 0: vector, ellipse_type = (kept_x - cand_x, kept_y - cand_y), "left_right"
            elif dx_grid == 1 and dy_grid == 1: vector, ellipse_type = (cand_x - kept_x, cand_y - kept_y), "topleft_bottomright"
            elif dx_grid == -1 and dy_grid == -1: vector, ellipse_type = (kept_x - cand_x, kept_y - cand_y), "topleft_bottomright"
            elif dx_grid == 1 and dy_grid == -1: vector, ellipse_type = (cand_x - kept_x, cand_y - kept_y), "bottomleft_topright"
            elif dx_grid == -1 and dy_grid == 1: vector, ellipse_type = (kept_x - cand_x, kept_y - cand_y), "bottomleft_topright"

            if ellipse_type and vector is not None:
                if is_in_ellipse(vector[0], vector[1], auto_ellipse_params[ellipse_type]):
                    if layer_diff <= 1:
                        is_duplicate = True
                        candidate["is_duplicate"] = True
                        removed_count += 1
                        break  
                    elif layer_diff == 2:
                        potential_2_layer += 1
            
            if not is_duplicate and layer_diff <= 1:
                raw_dist = math.hypot(cand_x - kept_x, cand_y - kept_y)
                if raw_dist <= stitching_max_um:
                    potential_stitching += 1

        if not is_duplicate:
            unique_kept.append(candidate)

    return removed_count, potential_2_layer, potential_stitching, auto_ellipse_params, grid_debug_dict

def save_density_grid_plot(grid_data_dict, processed_data_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    directions = ["top_bottom", "left_right", "topleft_bottomright", "bottomleft_topright"]
    titles = ["Top / Bottom (Y+)", "Left / Right (X+)", "Diag: Top-Left -> Bot-Right", "Diag: Bot-Left -> Top-Right"]
    
    cmap = mcolors.ListedColormap(['#2d3436', '#0984e3', '#00cec9', '#55efc4', '#ffeaa7', '#fab1a0', '#e17055', '#d63031'])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    e_sets = GLOBAL_CONSTANTS["ellipse_fitting"]
    threshold_idx = min(7, max(0, int(math.floor(e_sets["ellipse_p80_multiplier"]))))

    cbar_labels = []
    for i in range(8):
        label = '< 1x' if i == 0 else '> 7x' if i == 7 else f"{i}x-{i+1}x"
        if i == threshold_idx: label += "\n(Threshold)"
        cbar_labels.append(label)
    
    for i, direction in enumerate(directions):
        ax = axes[i]
        if direction not in grid_data_dict or grid_data_dict[direction] is None:
            ax.set_title(f"{titles[i]} - No Data")
            continue
            
        data = grid_data_dict[direction]
        gx, gy, counts, p80 = data["gx"], data["gy"], data["counts"], data["p80"]
        
        ratios = counts / p80 if p80 > 0 else np.zeros_like(counts)
        categories = np.clip(np.floor(ratios), 0, 7)
        
        sc = ax.scatter(gx.ravel(), gy.ravel(), c=categories.ravel(), cmap=cmap, norm=norm, marker='s', s=15)
        ax.set_title(f"{titles[i]}\n(Baseline p80: {p80:.1f} vectors)", fontsize=11, fontweight='bold')
        ax.set_xlim(-400, 400); ax.set_ylim(-400, 400); ax.set_aspect('equal')
        ax.set_xlabel("X Offset (\u03bcm)", fontsize=10); ax.set_ylabel("Y Offset (\u03bcm)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    cbar.ax.set_yticklabels(cbar_labels)
    cbar.set_label('Density vs 80th Percentile Baseline')
    
    plt.suptitle("Grid Density Categorization for Ellipse Fitting", fontsize=14, fontweight='bold')
    plt.savefig(processed_data_path / "density_grid_diagnostics.png", dpi=300)
    plt.close(fig)

def export_analysis_results(all_objects: list, parameters: dict, processed_data_path: pathlib.Path, metadata: dict, xy_threshold_um: float = 120.0, stitching_max_um: float = 200.0, all_objects_with_aborted=None):
    if all_objects_with_aborted is None:
        all_objects_with_aborted = all_objects
        
    raw_count = len(all_objects_with_aborted)

    analytics_meta = metadata.get("Analytics", {})
    sample_mask_area_mm2 = float(analytics_meta.get("total_mask_area_mm2", 0.0))
    camera_meta = metadata.get("Camera_preprocessed", {})
    
    g_sets = GLOBAL_CONSTANTS["global_settings"]
    pixel_scale_um = float(camera_meta.get("pixelSizeUm", g_sets["pixel_scale_fallback"]))
    
    positions_meta = metadata.get("Positions", [])
    if positions_meta:
        num_z_layers = int(positions_meta[0].get("z_layers", 13))
        z_step_um = float(positions_meta[0].get("z_step_um", 3813.7939))
    else:
        num_z_layers, z_step_um = 13, 3813.7939

    duplicates_removed, potential_2_layer, potential_stitching, final_ellipses, grid_debug_dict = mark_duplicates_and_fit_ellipses(
        all_objects, parameters, xy_threshold_um, stitching_max_um
    )

    sample_height_um = (num_z_layers - 1) * z_step_um * 0.131
    sample_height_mm = sample_height_um / 1000.0
    total_sample_volume_mm3 = sample_mask_area_mm2 * sample_height_mm if sample_mask_area_mm2 > 0 else 0.0

    verified_hydrates, out_of_focus_hydrates, duplicate_hydrates, rejected_objects = [], [], [], []
    total_hydrate_volume_um3 = 0.0
    equivalent_diameters = []
    
    total_cat1_passed = 0
    total_cat2_rescued_passed = 0

    for idx, hyd in enumerate(all_objects_with_aborted, start=1):
        area_um2 = float(hyd.get("area_um2", 0.0))
        eq_diameter_um = float(2.0 * np.sqrt(area_um2 / np.pi)) if area_um2 > 0 else 0.0
        radius_um = eq_diameter_um / 2.0
        estimated_vol_um3 = float((4.0 / 3.0) * np.pi * (radius_um ** 3))

        is_focus = bool(hyd.get("focus", False))
        is_duplicate = bool(hyd.get("is_duplicate", False))
        is_above_threshold = (float(hyd.get("total_score", 0.0)) >= parameters.get("total_score_threshold", 4.0))

        hyd_formatted = {
            "global_id": idx,
            "source_stack": hyd.get("source_stack", ""),
            "layer": int(hyd.get("layer", 0)),
            "pipeline_path": hyd.get("pipeline_path", "unknown"),
            "abort_reason": hyd.get("abort_reason", "None"),
            "is_in_focus": is_focus,
            "is_spatial_duplicate": is_duplicate,
            "focus_score": float(hyd.get("focus_value", 0.0)),
            
            "area_um2": round(area_um2, 2),
            "area_px": round(float(hyd.get("area_px", 0.0)), 1),
            "equivalent_diameter_um": round(eq_diameter_um, 2),
            "estimated_volume_um3": round(estimated_vol_um3, 2),
            "aspect_ratio": round(float(hyd.get("aspect_ratio", 1.0)), 3),
            "circularity": round(float(hyd.get("circularity", 0.0)), 3),
            "ellipse_fill_factor": round(float(hyd.get("ellipse_fill_factor", 0.0)), 3),
            "core_brightness_75th": round(float(hyd.get("core_brightness_75th", 0.0)), 1),
            "total_score": round(float(hyd.get("total_score", 0.0)), 2),
            
            "fitted_ellipse": {
                "major_axis_um": round(float(hyd.get("major_axis_um", 0.0)), 2),
                "minor_axis_um": round(float(hyd.get("minor_axis_um", 0.0)), 2),
                "orientation_angle_deg": round(float(hyd.get("ellipse_angle_deg", 0.0)), 1),
                "angle_reference_note": "0 deg = parallel to image Y-axis. 90 deg = parallel to image X-axis. Since image X-axis points vertically to the sky in the real world, 90 deg indicates a vertical real-world orientation, and 0 deg indicates a horizontal one."
            },
            "world_coordinates_um": {
                "x": round(float(hyd.get("center_x_um_world", 0.0)), 2),
                "y": round(float(hyd.get("center_y_um_world", 0.0)), 2),
                "z": round(float(hyd.get("center_z_um_world", 0.0)), 2)
            }
        }

        if not is_above_threshold or hyd.get("is_aborted", False):
            rejected_objects.append(hyd_formatted)
        elif not is_focus:
            out_of_focus_hydrates.append(hyd_formatted)
        elif is_duplicate:
            duplicate_hydrates.append(hyd_formatted)
        else:
            verified_hydrates.append(hyd_formatted)
            if area_um2 > 0:
                equivalent_diameters.append(eq_diameter_um)
                total_hydrate_volume_um3 += estimated_vol_um3
            
            if "init -> good" in hyd.get("pipeline_path", ""): total_cat1_passed += 1
            if "rescued" in hyd.get("pipeline_path", ""): total_cat2_rescued_passed += 1

    unique_count = len(verified_hydrates)
    number_density_mm3 = (unique_count / total_sample_volume_mm3) if total_sample_volume_mm3 > 0 else 0.0
    sample_vol_um3 = total_sample_volume_mm3 * 1e9
    hydrate_volume_fraction = (total_hydrate_volume_um3 / sample_vol_um3) if sample_vol_um3 > 0 else 0.0

    summary = {
        "sample_metadata": {
            "processed_directory": str(processed_data_path.resolve()),
            "spatial_resolution_um_per_px": pixel_scale_um,
            "z_step_um": z_step_um,
            "total_z_layers": num_z_layers,
            "total_sample_area_mm2": round(sample_mask_area_mm2, 3),
            "total_sample_height_um": round(sample_height_um, 1),
            "total_sample_volume_mm3": round(total_sample_volume_mm3, 4)
        },
        "macro_statistics": {
            "total_contours_raw": raw_count,
            "total_verified_hydrates": unique_count,
            "of_which_phase_1_perfect": total_cat1_passed,
            "of_which_phase_2_rescued": total_cat2_rescued_passed,
            
            "total_out_of_focus_hydrates": len(out_of_focus_hydrates),
            "total_spatial_duplicates": len(duplicate_hydrates),
            "total_rejected_noise_objects": len(rejected_objects),
            
            "hydrate_number_density_per_mm3": round(number_density_mm3, 4),
            "hydrate_volume_fraction": round(hydrate_volume_fraction, 6),
            "mean_equivalent_diameter_um": round(float(np.mean(equivalent_diameters)), 2) if equivalent_diameters else 0.0,
            "median_equivalent_diameter_um": round(float(np.median(equivalent_diameters)), 2) if equivalent_diameters else 0.0,
            "std_equivalent_diameter_um": round(float(np.std(equivalent_diameters)), 2) if equivalent_diameters else 0.0
        },
        "diagnostics": {
            f"potential_2_layer_matches_within_{int(xy_threshold_um)}um": potential_2_layer,
            f"potential_stitching_matches_{int(xy_threshold_um)}_to_{int(stitching_max_um)}um": potential_stitching,
            "fitted_stitching_ellipses": final_ellipses
        },
        "fuzzy_parameters_used": parameters,
        "verified_hydrates": verified_hydrates,
        "out_of_focus_hydrates": out_of_focus_hydrates,
        "duplicate_hydrates": duplicate_hydrates,
        "rejected_objects": rejected_objects
    }

    output_json_path = processed_data_path / "analysis_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    save_density_grid_plot(grid_debug_dict, processed_data_path)
    
    valid_objects_for_diagnostics = [obj for obj in all_objects if obj.get("total_score", 0) >= parameters.get("total_score_threshold", 5.0) and not obj.get("is_aborted", False)]
    plot_and_save_diagnostics(valid_objects_for_diagnostics, processed_data_path, final_ellipses)

    return output_json_path
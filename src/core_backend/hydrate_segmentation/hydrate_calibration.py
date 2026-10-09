import ast
import json
import pathlib
import datetime
import time
import math
import numpy as np
import tifffile
import scipy.optimize
import cv2
import logging
import random
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import threading
import queue
import multiprocessing
from collections import defaultdict

try:
    import joblib
except ImportError:
    pass

from core_backend.hydrate_segmentation.hydrate_detection import (
    preprocess_image,
    edge_detection,
    extract_and_filter_objects
)

try:
    from core_backend.hydrate_segmentation.save_cal_results import save_generation_results
except ImportError:
    save_generation_results = None

logging.basicConfig(
    filename="debug_main.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Load the shared hydrate-segmentation configuration.
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
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                GLOBAL_CONSTANTS = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load hydrate_constants.json: {e}")
            raise
    else:
        logging.error(f"hydrate_constants.json not found at {config_path}!")
        raise FileNotFoundError(f"Missing {config_path}")

load_global_constants()

# Worker and stack caches
_WORKER_IMAGE_CACHE = {}
_WORKER_GT_CACHE = {}
_PRECOMPUTED_STACKS = {} 

def _parse_mutation(value):
    mutation = ast.literal_eval(value)
    if isinstance(mutation, bool):
        raise ValueError("Mutation must be a number or a pair of numbers.")

    if isinstance(mutation, (int, float)):
        values = (float(mutation),)
        parsed_mutation = values[0]
    elif isinstance(mutation, tuple) and len(mutation) == 2:
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in mutation):
            raise ValueError("Mutation range values must be numbers.")
        values = tuple(float(item) for item in mutation)
        parsed_mutation = values
    else:
        raise ValueError("Mutation must be a number or a pair of numbers.")

    if any(not math.isfinite(item) or not 0 <= item < 2 for item in values):
        raise ValueError("Mutation values must be finite and in the range [0, 2).")
    if len(values) == 2 and values[0] > values[1]:
        raise ValueError("Mutation range minimum must not exceed its maximum.")

    return parsed_mutation


def load_tiff_stack(stack_path: pathlib.Path) -> np.ndarray:
    with tifffile.TiffFile(str(stack_path)) as tif:
        if len(tif.pages) > 1: stack = np.stack([page.asarray() for page in tif.pages], axis=0)
        else: stack = tif.asarray()
    if stack.ndim == 2: stack = np.expand_dims(stack, axis=0)
    elif stack.ndim == 3:
        min_axis = int(np.argmin(stack.shape))
        if min_axis != 0 and stack.shape[min_axis] < stack.shape[0]:
            stack = np.moveaxis(stack, min_axis, 0)
    return stack

def decode_parameters_2d(raw_parameters):
    return {
        "area_mu": float(raw_parameters[0]), "area_sigma": float(raw_parameters[1]), "weight_area": float(raw_parameters[2]),
        "aspect_mu": 1.0, "aspect_sigma": float(raw_parameters[3]), "weight_aspect": float(raw_parameters[4]),
        "circ_x0": float(raw_parameters[5]), "circ_k": float(raw_parameters[6]), "weight_circ": float(raw_parameters[7]),
        "fill_x0": float(raw_parameters[8]), "fill_k": float(raw_parameters[9]), "weight_fill": float(raw_parameters[10]),
        "solid_x0": float(raw_parameters[11]), "solid_k": float(raw_parameters[12]), "weight_solid": float(raw_parameters[13]),
        "bright_x0": float(raw_parameters[14]), "bright_k": float(raw_parameters[15]), "weight_bright": float(raw_parameters[16]),
        "contrast_mu": float(raw_parameters[17]), "contrast_sigma": float(raw_parameters[18]), "weight_contrast": float(raw_parameters[19]),
        "total_score_threshold": float(GLOBAL_CONSTANTS["calibration_settings"].get("fixed_total_score_threshold", 5.0))
    }

# Precompute detection results for calibration stacks.
def precompute_all_stacks(calibrate_data_paths, ground_truth_path, metadata, rf_model=None):
    global _PRECOMPUTED_STACKS, _WORKER_IMAGE_CACHE, _WORKER_GT_CACHE
    logging.info(f"Starting Pre-Computation for {len(calibrate_data_paths)} stacks...")
    
    _PRECOMPUTED_STACKS.clear()
    
    det_config = GLOBAL_CONSTANTS["phase_1_detection"]
    clahe_obj = cv2.createCLAHE(clipLimit=det_config.get("clip_limit", 2.0), tileGridSize=(8, 8))
    rim_kernel = np.ones((3, 3), np.uint8)

    dummy_params = {"total_score_threshold": -1.0} 

    for stack_path in calibrate_data_paths:
        stack_name = stack_path.stem
        
        # Support both shared metadata and metadata grouped by input subfolder.
        if "Positions" in metadata:
            current_metadata = metadata
        else:
            current_metadata = metadata.get(stack_path.parent.name)
            if not current_metadata:
                current_metadata = list(metadata.values())[0] if metadata else {}
        
        if stack_name not in _WORKER_GT_CACHE:
            ground_truth_file = ground_truth_path / f"ground_truth_{stack_name}.json"
            if not ground_truth_file.exists(): continue
            with open(ground_truth_file, "r") as f: _WORKER_GT_CACHE[stack_name] = json.load(f)
            
        if stack_name not in _WORKER_IMAGE_CACHE:
            _WORKER_IMAGE_CACHE[stack_name] = load_tiff_stack(stack_path)
            
        stack = _WORKER_IMAGE_CACHE[stack_name]
        
        all_extracted = []
        aborted_bboxes_per_layer = defaultdict(list)
        
        for layer in range(stack.shape[0]):
            raw_image = stack[layer]
            image_preprocessed = preprocess_image(raw_image, det_config, clahe=clahe_obj)
            image_binary = edge_detection(image_preprocessed, det_config)
            
            extracted = extract_and_filter_objects(
                image_binary, image_preprocessed, raw_image, current_metadata, stack_name, layer, dummy_params, rf_model, rim_kernel
            )
            
            for obj in extracted:
                all_extracted.append(obj)
                if obj.get("is_aborted", False) and obj.get("triage_contour_px") is not None:
                    aborted_bboxes_per_layer[layer].append(cv2.boundingRect(obj["triage_contour_px"]))
                    
        _PRECOMPUTED_STACKS[stack_name] = {
            "extracted_objects": all_extracted,
            "aborted_bboxes": aborted_bboxes_per_layer
        }
    logging.info("Pre-Computation finished successfully.")

# Calibration objective functions
def objective_function_2d(raw_parameters, calibrate_data_paths, ground_truth_path, metadata, progress_queue=None, detailed=False, rf_model=None):
    global _PRECOMPUTED_STACKS, _WORKER_GT_CACHE
    
    timings = {"load_tiff": 0.0, "preprocess": 0.0, "edge": 0.0, "extract": 0.0, "match": 0.0}
    parameters = decode_parameters_2d(raw_parameters)
    
    error_fn, error_fp = 0.0, 0.0    
    evaluated_stacks = 0
    total_gt_hydrates = 0
    matched_gt_hydrates_above_threshold = 0
    
    path_features = {
        "1_good_ok_gt": [], "1_good_ok_no_gt": [],
        "1_good_fail_gt": [], "1_good_fail_no_gt": [],
        "2_resc_ok_gt": [], "2_resc_ok_no_gt": [],
        "2_resc_fail_gt": [], "2_resc_fail_no_gt": [],
        "2_rf_trash_gt": [], "2_rf_trash_no_gt": [],
        "3_trash_gt": [], "3_trash_no_gt": []
    }
    
    cal_settings = GLOBAL_CONSTANTS["calibration_settings"]
    limit_area = cal_settings.get('match_area_diff_max', 0.60)
    limit_aspect = cal_settings.get('match_aspect_diff_max', 0.75)
    limit_overlap = cal_settings.get('match_overlap_min', 0.15)
    fallback_radius = cal_settings.get("match_fallback_radius_px", 100.0)
    
    stats = {
        "paths": {
            "1_good_ok_gt": 0, "1_good_ok_no_gt": 0,
            "1_good_fail_gt": 0, "1_good_fail_no_gt": 0,
            "2_resc_ok_gt": 0, "2_resc_ok_no_gt": 0,
            "2_resc_fail_gt": 0, "2_resc_fail_no_gt": 0,
            "2_rf_trash_gt": 0, "2_rf_trash_no_gt": 0,
            "3_trash_gt": 0, "3_trash_no_gt": 0
        },
        "hyd_missed": 0,
        "abort_reasons": {},
        "missed_reasons": {}
    }
    
    mismatch_metrics = {"area_diff": {"pass": [], "fail": []}, "aspect_diff": {"pass": [], "fail": []}, "overlap": {"pass": [], "fail": []}}
    
    fp_blur_radius_sq = cal_settings.get("fp_blur_radius_px", 60.0) ** 2
    fn_penalty = cal_settings.get("fn_penalty_score", 1.0)
    fp_penalty = cal_settings.get("fp_penalty_score", 0.5)
    threshold = parameters.get("total_score_threshold", 5.0)

    for stack_path in calibrate_data_paths:
        stack_name = stack_path.stem
        
        if stack_name not in _PRECOMPUTED_STACKS or stack_name not in _WORKER_GT_CACHE:
            continue
            
        ground_truth_data = _WORKER_GT_CACHE[stack_name]
        all_gt_labels = ground_truth_data.get("ground_truth_labels", [])
        if not all_gt_labels: continue
            
        cached_data = _PRECOMPUTED_STACKS[stack_name]
        all_extracted = cached_data["extracted_objects"]
        aborted_bboxes_per_layer = cached_data["aborted_bboxes"]
        
        evaluated_stacks += 1
        total_gt_hydrates += len(all_gt_labels)
        
        t0 = time.perf_counter()

        # Apply the candidate fuzzy-scoring parameters to extracted objects.
        for hyd in all_extracted:
            is_aborted_by_phase12 = hyd.get("is_aborted", False) and "Fuzzy Score failed" not in hyd.get("abort_reason", "")
            
            if not is_aborted_by_phase12:
                area = hyd.get("area_px", 0)
                aspect = hyd.get("aspect_ratio", 1.0)
                circ = hyd.get("circularity", 0.0)
                fill = hyd.get("ellipse_fill_factor", 0.0)
                solid = hyd.get("solidity", 0.0)
                bright = hyd.get("core_brightness_75th", 0.0)
                contrast = hyd.get("core_rim_contrast", 1.0)
                
                score_area = math.exp(-0.5 * ((area - parameters.get("area_mu", 15000)) / parameters.get("area_sigma", 5000))**2) if parameters.get("area_sigma", 5000) != 0 else 0
                score_aspect = math.exp(-0.5 * ((aspect - parameters.get("aspect_mu", 1.0)) / parameters.get("aspect_sigma", 0.5))**2) if parameters.get("aspect_sigma", 0.5) != 0 else 0
                
                try: score_circ = 1.0 / (1.0 + math.exp(-parameters.get("circ_k", 25) * (circ - parameters.get("circ_x0", 0.7))))
                except OverflowError: score_circ = 0.0 if -parameters.get("circ_k", 25) * (circ - parameters.get("circ_x0", 0.7)) > 0 else 1.0
                
                try: score_fill = 1.0 / (1.0 + math.exp(-parameters.get("fill_k", 25) * (fill - parameters.get("fill_x0", 0.75))))
                except OverflowError: score_fill = 0.0 if -parameters.get("fill_k", 25) * (fill - parameters.get("fill_x0", 0.75)) > 0 else 1.0
                
                try: score_solid = 1.0 / (1.0 + math.exp(-parameters.get("solid_k", 30) * (solid - parameters.get("solid_x0", 0.85))))
                except OverflowError: score_solid = 0.0 if -parameters.get("solid_k", 30) * (solid - parameters.get("solid_x0", 0.85)) > 0 else 1.0
                
                try: score_bright = 1.0 / (1.0 + math.exp(-parameters.get("bright_k", 0.2) * (bright - parameters.get("bright_x0", 100))))
                except OverflowError: score_bright = 0.0 if -parameters.get("bright_k", 0.2) * (bright - parameters.get("bright_x0", 100)) > 0 else 1.0
                
                score_contrast = math.exp(-0.5 * ((contrast - parameters.get("contrast_mu", 1.1)) / parameters.get("contrast_sigma", 0.3))**2) if parameters.get("contrast_sigma", 0.3) != 0 else 0
                
                total_score = (
                    score_area * parameters.get("weight_area", 1.0) +
                    score_aspect * parameters.get("weight_aspect", 1.0) +
                    score_circ * parameters.get("weight_circ", 1.0) +
                    score_fill * parameters.get("weight_fill", 1.0) +
                    score_solid * parameters.get("weight_solid", 1.0) +
                    score_bright * parameters.get("weight_bright", 1.0) +
                    score_contrast * parameters.get("weight_contrast", 1.0)
                )
                
                hyd["total_score"] = total_score
                if total_score < threshold:
                    hyd["is_aborted"] = True
                    hyd["abort_reason"] = "Phase 3: Fuzzy Score failed"
                else:
                    hyd["is_aborted"] = False 

            cat_str = hyd.get("source_category_str", "")
            is_aborted = hyd.get("is_aborted", False)
            score_ok = hyd.get("total_score", 0) >= threshold
            
            if "3_trash" in cat_str: hyd["current_path"] = "3_trash"
            elif is_aborted and "RF" in hyd.get("abort_reason", ""): hyd["current_path"] = "2_rf_trash"
            elif "1_good" in cat_str: hyd["current_path"] = "1_good_ok" if score_ok and not is_aborted else "1_good_fail"
            elif "2_" in cat_str or "rescued" in cat_str: hyd["current_path"] = "2_resc_ok" if score_ok and not is_aborted else "2_resc_fail"
            else: hyd["current_path"] = "3_trash"

        # Match detections to ground-truth objects using the configured cascade.
        matched_obj_ids = set()
        stack_hyd_missed = 0  
        
        gt_list = sorted(all_gt_labels, key=lambda x: x.get('width',1)*x.get('height',1), reverse=True)
        
        for gt in gt_list:
            gt_z = gt.get("best_z", 0)
            gt_cx, gt_cy = gt.get("cx", 0), gt.get("cy", 0)
            gt_w, gt_h = gt.get("width", 1), gt.get("height", 1)
            gt_angle = gt.get("angle_deg", 0)
            
            gt_area = math.pi * (gt_w / 2.0) * (gt_h / 2.0)
            gt_aspect = max(gt_w, gt_h) / min(gt_w, gt_h) if min(gt_w, gt_h) > 0 else 99.0
            
            best_overlap = 0.0
            best_cand_strict = None
            closest_dist_strict = float('inf')
            
            for hyd in all_extracted:
                if hyd["id"] in matched_obj_ids: continue
                if abs(hyd["layer"] - gt_z) > 2: continue 
                if not hyd["current_path"].endswith("_ok"): continue 
                
                hyd_area = hyd.get("area_px", 0)
                if hyd_area == 0: continue
                
                dist = math.hypot(gt_cx - hyd.get("center_x_px_image", 0), gt_cy - hyd.get("center_y_px_image", 0))
                if dist > max(gt_w, gt_h) + 100: continue
                    
                area_diff = abs(hyd_area - gt_area) / gt_area
                aspect_diff = abs(hyd.get("aspect_ratio", 1.0) - gt_aspect) / gt_aspect
                
                box_r = int(max(gt_w, gt_h) * 1.5 + 20)
                mask_gt = np.zeros((box_r*2, box_r*2), dtype=np.uint8)
                mask_hyd = np.zeros((box_r*2, box_r*2), dtype=np.uint8)
                shift_x, shift_y = gt_cx - box_r, gt_cy - box_r
                
                cv2.ellipse(mask_gt, (box_r, box_r), (int(gt_w / 2), int(gt_h / 2)), int(gt_angle), 0, 360, 255, -1)
                
                contour_px = hyd.get("contour_image_px")
                if contour_px is None: continue
                shifted_contour = contour_px - [shift_x, shift_y]
                cv2.drawContours(mask_hyd, [shifted_contour.astype(np.int32)], -1, 255, -1)
                
                overlap_ratio = cv2.countNonZero(cv2.bitwise_and(mask_gt, mask_hyd)) / gt_area
                
                p_area = area_diff <= limit_area
                p_aspect = aspect_diff <= limit_aspect
                p_overlap = overlap_ratio >= limit_overlap
                
                if dist < closest_dist_strict: closest_dist_strict = dist
                
                if p_area and p_aspect and p_overlap and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cand_strict = hyd
                    
            if best_cand_strict is not None:
                matched_obj_ids.add(best_cand_strict["id"])
                best_cand_strict["is_gt_match"] = True
            else:
                best_cand_fallback = None
                closest_dist_fallback = float('inf')
                
                for hyd in all_extracted:
                    if hyd["id"] in matched_obj_ids: continue
                    if abs(hyd["layer"] - gt_z) > 2: continue
                    if hyd["current_path"].endswith("_ok"): continue 
                    
                    dist = math.hypot(gt_cx - hyd.get("center_x_px_image", 0), gt_cy - hyd.get("center_y_px_image", 0))
                    if dist > fallback_radius: continue
                    
                    if dist < closest_dist_fallback:
                        closest_dist_fallback = dist
                        best_cand_fallback = hyd
                
                if best_cand_fallback is not None:
                    matched_obj_ids.add(best_cand_fallback["id"])
                    best_cand_fallback["is_gt_match"] = True
                    error_fn += fn_penalty 
                else:
                    stack_hyd_missed += 1
                    error_fn += fn_penalty

        # Score candidate routes and collect calibration statistics.
        for hyd in all_extracted:
            path = hyd["current_path"]
            is_gt_match = hyd.get("is_gt_match", False)
            
            feat_dict = {
                "area_px": float(hyd.get("area_px", 0)), "aspect_ratio": float(hyd.get("aspect_ratio", 0)),
                "circularity": float(hyd.get("circularity", 0)), "ellipse_fill_factor": float(hyd.get("ellipse_fill_factor", 0)),
                "solidity": float(hyd.get("solidity", 0)), "core_brightness_75th": float(hyd.get("core_brightness_75th", 0)),
                "core_rim_contrast": float(hyd.get("core_rim_contrast", 0)), "total_score": float(hyd.get("total_score", 0))
            }
            
            if is_gt_match:
                stats["paths"][f"{path}_gt"] += 1
                if detailed: path_features[f"{path}_gt"].append(feat_dict)
                if path in ["1_good_ok", "2_resc_ok"]:
                    matched_gt_hydrates_above_threshold += 1
            else:
                hyd_x, hyd_y = hyd.get("center_x_px_image", 0), hyd.get("center_y_px_image", 0)
                is_blurry = any((hyd_x - gt.get("cx",0))**2 + (hyd_y - gt.get("cy",0))**2 < fp_blur_radius_sq for gt in all_gt_labels)
                in_aborted_box = any(bx <= hyd_x <= bx+bw and by <= hyd_y <= by+bh for (bx, by, bw, bh) in aborted_bboxes_per_layer[hyd["layer"]])
                
                hyd["is_ignored_noise"] = is_blurry or in_aborted_box
                
                if not hyd["is_ignored_noise"]:
                    stats["paths"][f"{path}_no_gt"] += 1
                    if detailed: path_features[f"{path}_no_gt"].append(feat_dict)
                    
                    if path in ["1_good_ok", "2_resc_ok"]:
                        error_fp += fp_penalty
                        
            hyd["is_gt_match"] = False

        stats["hyd_missed"] += stack_hyd_missed
        timings["match"] += time.perf_counter() - t0

    if progress_queue is not None:
        try: progress_queue.put(1, block=False)
        except Exception: pass

    if evaluated_stacks == 0:
        dummy_val = cal_settings.get("dummy_eval_penalty", 999999.0)
        return (dummy_val, 0.0, 0.0, 0, 0, timings, path_features, stats, mismatch_metrics) if detailed else dummy_val
    
    total_error = error_fn + error_fp
    return (total_error, error_fn, error_fp, matched_gt_hydrates_above_threshold, total_gt_hydrates, timings, path_features, stats, mismatch_metrics) if detailed else total_error


def estimate_time_2d_calc(data_folder_2d, popsize_str, maxiter_str):
    if not data_folder_2d: return "Select folder first!", True
    try:
        processed_data_path = pathlib.Path(data_folder_2d)
        gt_dir = processed_data_path / "ground_truth"
        
        found_meta = list(processed_data_path.rglob("metadata_preprocessed.json"))
        if not found_meta: return "No metadata found!", True
        
        metadata_map = {}
        for mf in found_meta:
            with open(mf, "r") as f: metadata_map[mf.parent.name] = json.load(f)
            
        stacks = list(processed_data_path.rglob("*.tif"))
        if not stacks: return "No TIFs found!", True
        if not gt_dir.exists() or not list(gt_dir.glob("ground_truth_*.json")): return "Create Ground Truth first!", True

        popsize, maxiter, ndim = int(popsize_str), int(maxiter_str), 20 
        t0 = time.time()
        
        bounds_2d = GLOBAL_CONSTANTS["calibration_settings"]["bounds_2d"]
        dummy_raw = [b[0] for b in bounds_2d]
        
        precompute_all_stacks(stacks[:1], gt_dir, metadata_map, rf_model=None)
        
        for _ in range(10):
            objective_function_2d(dummy_raw, stacks[:1], gt_dir, metadata_map, rf_model=None)
            
        total_evals = (popsize * ndim) * maxiter + (popsize * ndim)
        avg_eval_sec = (time.time() - t0) / 10.0
        total_sec = avg_eval_sec * total_evals * len(stacks) * 1.15
        
        return f"~ {int(total_sec // 3600)}h {int((total_sec // 60) % 60):02d}m", False
    except Exception:
        return "Error in estimation", True


def run_2d_optimization(selected_stacks, gt_dir, metadata, opt_config, progress_callback, completion_callback, intra_gen_callback=None):
    logging.info("--- 2D TURBO OPTIMIZATION STARTED ---")
    try:
        maxiter, popsize, workers = int(opt_config["maxiter"]), int(opt_config["popsize"]), int(opt_config["workers"])
        mutation = _parse_mutation(opt_config["mutation"])
        recombination = float(opt_config["recombination"])
        rf_model_path = opt_config.get("rf_model_path")
    except Exception as e:
        logging.error(f"Error parsing opt_config: {e}", exc_info=True); return

    if not selected_stacks: return
        
    rf_model = None
    if rf_model_path and pathlib.Path(rf_model_path).exists():
        try: rf_model = joblib.load(rf_model_path); logging.info(f"Loaded RF model from {rf_model_path}")
        except Exception: pass

    bounds_2d = GLOBAL_CONSTANTS["calibration_settings"]["bounds_2d"]
    
    precompute_all_stacks(selected_stacks, gt_dir, metadata, rf_model=rf_model)

    hard_bounds = {
        "area_px": (GLOBAL_CONSTANTS["phase_1_triage_bounds"]["strict_bounds"]["area_min_px"], GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["area_max_px"]),
        "aspect_ratio": (None, GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["aspect_max"]), 
        "circularity": (GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["circ_min"], None),
        "ellipse_fill_factor": (GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["fill_min"], None), 
        "solidity": (GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["solid_min"], None),
        "core_brightness_75th": (GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["bright_min"], None), 
        "core_rim_contrast": (GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["contrast_min"], GLOBAL_CONSTANTS["phase_1_triage_bounds"]["loose_bounds"]["contrast_max"]),
        "total_score": (GLOBAL_CONSTANTS["calibration_settings"]["fixed_total_score_threshold"], None)
    }
    
    generations_data = {"gen": [], "fp": [], "fn": [], "latest_stats": {}}
    accumulated_timings_2d = defaultdict(float)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cal_results_dir = gt_dir.parent / f"calibration_results_{timestamp}"
    cal_results_dir.mkdir(parents=True, exist_ok=True)

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    optimization_done = threading.Event()
    current_evals_count = [0] 
    evals_per_gen = popsize * len(bounds_2d)

    def monitor_progress():
        while not optimization_done.is_set():
            try:
                progress_queue.get(timeout=0.2)
                current_evals_count[0] += 1
                if intra_gen_callback: intra_gen_callback(current_evals_count[0], evals_per_gen)
            except queue.Empty: pass

    threading.Thread(target=monitor_progress, daemon=True).start()

    start_time = time.time()
    generation_times = []

    def scipy_callback(xk, convergence=None):
        nonlocal start_time
        try:
            now = time.time()
            generation_times.append(now - start_time)
            start_time = now 
            
            eta_seconds = (sum(generation_times) / len(generation_times)) * (maxiter - len(generations_data["gen"]))
            eta_str = (datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)).strftime("%d.%m.%Y, %H:%M")

            current_evals_count[0] = 0
            if intra_gen_callback: intra_gen_callback(0, evals_per_gen)

            _, err_fn, err_fp, matched, total_gt, timings, path_features, stats, mismatch_dist = objective_function_2d(
                xk, selected_stacks, gt_dir, metadata, progress_queue=None, detailed=True, rf_model=rf_model
            )
            
            decoded_params = decode_parameters_2d(xk)
            gen = len(generations_data["gen"]) + 1

            generations_data["gen"].append(gen); generations_data["fp"].append(err_fp)
            generations_data["fn"].append(err_fn); generations_data["latest_stats"] = stats
            for k, v in timings.items(): accumulated_timings_2d[k] += v

            logging.info(f"[TURBO CALLBACK] Gen {gen} in {generation_times[-1]:.2f}s. ({matched}/{total_gt} matched). ETA: {eta_str}")

            if save_generation_results is not None:
                save_generation_results(
                    gen=gen, output_dir=cal_results_dir, generations_data=generations_data,
                    params=decoded_params, path_features=path_features,
                    stats_table=stats, hard_bounds=hard_bounds, mismatch_distributions=mismatch_dist
                )

            if progress_callback:
                progress_callback(gen, generations_data, matched, total_gt, decoded_params, path_features, stats, hard_bounds, eta_str)
        except Exception as cb_err:
            logging.error(f"Error within SciPy Callback: {cb_err}", exc_info=True)

    try:
        logging.info("Starting SciPy Optimizer with Cached Data (Single-Threaded for Stability)...")

        result = scipy.optimize.differential_evolution(
            objective_function_2d, bounds_2d, args=(selected_stacks, gt_dir, metadata, progress_queue, False, rf_model),
            maxiter=maxiter, popsize=popsize, tol=-1, atol=-1, polish=False, init="latinhypercube", 
            workers=1, updating="immediate", 
            mutation=mutation, recombination=recombination, strategy="rand1bin", disp=False, callback=scipy_callback,
        )
            
        optimization_done.set()
        best_config_2d = decode_parameters_2d(result.x)
            
        output_path = cal_results_dir / f"optimized_parameters_2D_final.json"
        with open(output_path, "w") as f: json.dump(best_config_2d, f, indent=4)
        if completion_callback: completion_callback(str(output_path))

    except Exception as opt_err:
        optimization_done.set()
        logging.error("Fatal error during SciPy Optimization phase!", exc_info=True)

    # Evaluate the three-dimensional calibration data.
def decode_parameters_3d(raw_parameters, base_parameters):
    parameters = base_parameters.copy()
    parameters["factor_contrast"] = float(raw_parameters[0])
    parameters["factor_laplacian"] = float(raw_parameters[1])
    parameters["factor_thickness_edge"] = float(raw_parameters[2])
    return parameters

def objective_function_3d(raw_parameters, focus_truth_files, base_parameters, detailed=False):
    timings = {"load_json": 0.0, "load_roi": 0.0, "calculate_metrics": 0.0}
    parameters = decode_parameters_3d(raw_parameters, base_parameters)
    total_error, total_evaluated, correctly_evaluated = 0.0, 0, 0

    for truth_file in focus_truth_files:
        t0 = time.perf_counter()
        with open(truth_file, "r") as f: focus_truth = json.load(f)
        timings["load_json"] += time.perf_counter() - t0
        base_dir = truth_file.parent

        for candidate_group in focus_truth.get("focus_candidates", []):
            best_layer_gt = candidate_group.get("best_layer")
            if best_layer_gt is None: continue  
            total_evaluated += 1
            highest_focus_score, predicted_best_layer = -1.0, -1
            
            for layer_data in candidate_group.get("layers", []):
                t0 = time.perf_counter()
                roi = np.load(base_dir / layer_data["roi_file"])
                timings["load_roi"] += time.perf_counter() - t0
                if roi.size == 0: continue

                t0 = time.perf_counter()
                score_contrast = float(np.std(roi))
                score_laplacian = float(cv2.Laplacian(roi, cv2.CV_64F).var())
                gradient = np.sqrt(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)**2)
                score_thickness_edge = float(np.mean((gradient) ** 2))

                total_score = (parameters.get("factor_contrast", 1.0) * score_contrast + parameters.get("factor_laplacian", 1.0) * score_laplacian + parameters.get("factor_thickness_edge", 1.0) * score_thickness_edge)
                timings["calculate_metrics"] += time.perf_counter() - t0

                if total_score > highest_focus_score:
                    highest_focus_score = total_score
                    predicted_best_layer = layer_data.get("layer", -1)

            if predicted_best_layer != -1:
                total_error += abs(best_layer_gt - predicted_best_layer) * GLOBAL_CONSTANTS["calibration_settings"].get("layer_diff_penalty_score", 50.0)
                if best_layer_gt == predicted_best_layer: correctly_evaluated += 1
                    
    return (total_error, correctly_evaluated, total_evaluated, timings) if detailed else total_error

def run_3d_optimization(processed_data_path, selected_stacks, gt_focus_dir, metadata_preprocessed, base_params, opt_config, progress_callback=None, completion_callback=None):
    logging.info("--- 3D OPTIMIZATION STARTED ---")
    base_params["distance_threshold"] = GLOBAL_CONSTANTS["calibration_settings"].get("3d_distance_threshold", 80.0)
    base_params["roi_radius_factor"] = GLOBAL_CONSTANTS["calibration_settings"].get("3d_roi_radius_factor", 1.0)

    try:
        maxiter, popsize, workers = int(opt_config["maxiter"]), int(opt_config["popsize"]), int(opt_config["workers"])
        mutation = _parse_mutation(opt_config["mutation"])
        recombination = float(opt_config["recombination"])
    except Exception as e: logging.error(f"Error parsing config: {e}", exc_info=True); return

    focus_truth_files = list(gt_focus_dir.glob("focus_ground_truth_*.json"))
    if not focus_truth_files: return

    bounds_3d = GLOBAL_CONSTANTS["calibration_settings"].get("bounds_3d", [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)])

    generations_data = {"gen": [], "error": []}

    def scipy_callback(xk, convergence=None):
        try:
            err, correct, total, timings = objective_function_3d(xk, focus_truth_files, base_params, detailed=True)
            gen = len(generations_data["gen"]) + 1
            generations_data["gen"].append(gen); generations_data["error"].append(err)
            if progress_callback: progress_callback(generations_data["gen"], generations_data["error"], correct, total)
        except Exception as cb_err: logging.error(f"Error 3D Callback: {cb_err}", exc_info=True)

    try:
        actual_workers = os.cpu_count() or 4 if workers == -1 else workers
        if actual_workers > 1:
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                result = scipy.optimize.differential_evolution(
                    objective_function_3d, bounds_3d, args=(focus_truth_files, base_params),
                    maxiter=maxiter, popsize=popsize, strategy='rand1bin', mutation=mutation,
                    recombination=recombination, tol=-1, atol=-1, workers=executor.map, updating='deferred', disp=False, callback=scipy_callback
                )
        else:
            result = scipy.optimize.differential_evolution(
                objective_function_3d, bounds_3d, args=(focus_truth_files, base_params),
                maxiter=maxiter, popsize=popsize, strategy='rand1bin', mutation=mutation,
                recombination=recombination, tol=-1, atol=-1, workers=1, updating='immediate', disp=False, callback=scipy_callback
            )

        final_parameters = decode_parameters_3d(result.x, base_params)
        output_path = processed_data_path / f"optimized_parameters_3D_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f: json.dump(final_parameters, f, indent=4)
        if completion_callback: completion_callback(str(output_path))
    except Exception as opt_err: logging.error("Fatal error 3D optimization!", exc_info=True)
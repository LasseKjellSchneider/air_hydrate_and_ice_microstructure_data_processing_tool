import json
import random
from pathlib import Path
import numpy as np
import tifffile
import joblib
import pandas as pd
import gc
import traceback
import concurrent.futures
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, make_scorer, recall_score

from .constants_manager import load_constants
from .detection_grain_boundaries import detect_grain_boundaries
from .calculating_values import calculate_values
from .ground_truth_grain_boundaries import label_objects_manually
from .filtering_objects import filter_objects
from .calculating_path_values import calculate_path_features
from .ground_truth_paths import label_paths_manually
from .logger_config import DebugProfiler
from .closing_grain_boundaries import (
    chunked_black_tophat, chunked_edt_mask, 
    chunked_neighbor_count, _local_route_through_array_ts, path_crosses_skeleton,
    _find_t_junc_targets_cone
)
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
import scipy.ndimage as ndi

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist() 
        if isinstance(obj, np.integer): return int(obj)     
        if isinstance(obj, np.floating): return float(obj)   
        return super(NumpyEncoder, self).default(obj)

def generate_ground_truth(cal_folder, squares_per_image, square_size, status_callback=None, config_path=None):
    folder_path = Path(cal_folder)
    profiler = DebugProfiler(folder_path / "debug_calibration_objects.txt")
    
    square_size = 8080
    constants = load_constants(config_path)
    bm = constants.get("border_margin_px", 400)
    
    def log(msg):
        if status_callback: status_callback(msg)
        profiler.log_msg(msg)

    try:
        image_paths = [p for p in folder_path.rglob("*.tif") if "_mask" not in p.name and "_overlay" not in p.name]
        if not image_paths: raise FileNotFoundError("No valid .tif files found!")

        ground_truth = []
        for idx, image_path in enumerate(image_paths, 1):
            log(f"Img {idx}/{len(image_paths)}: Loading {image_path.name}...")
            profiler.log_step(f"Image {idx}")
            with tifffile.TiffFile(str(image_path)) as tif: image = tif.asarray()
            h, w = image.shape[:2]
            
            mask_path = image_path.parent / f"{image_path.stem}_mask.png"
            border_mask = np.array(Image.open(str(mask_path)).convert('L')) if mask_path.exists() else np.full((h, w), 255, dtype=np.uint8)

            i, max_attempts = 0, 200
            squares = []
            sq_w = min(square_size, w)
            sq_h = min(square_size, h)
            
            while i < squares_per_image and max_attempts > 0:
                start_x, start_y = int((w - sq_w) * random.random()), int((h - sq_h) * random.random())
                if np.mean(border_mask[start_y:start_y + sq_h, start_x:start_x + sq_w]) < 200:  
                    max_attempts -= 1; continue 
                squares.append((start_x, start_x + sq_w, start_y, start_y + sq_h))
                i += 1; max_attempts -= 1

            for sq_idx, (x0, x1, y0, y1) in enumerate(squares, 1):
                log(f"Img {idx} | Sq {sq_idx}: Detecting boundaries...")
                sq_img, sq_mask = image[y0:y1, x0:x1], border_mask[y0:y1, x0:x1]
                objects, _ = detect_grain_boundaries(sq_img, False, sq_mask, parameters=constants)
                ground_truth.append({
                    "image_path": str(image_path), "x_start": x0, "x_end": x1, "y_start": y0, "y_end": y1,
                    "objects": calculate_values(objects, sq_img)
                })
                
            del image, border_mask; gc.collect()

        log("Opening interactive labeling tool...")
        # Fit the annotation window within the available display area.
        ground_truth = label_objects_manually(ground_truth, border_margin=bm, target_display_height=765)
        
        output_json_path = folder_path / "ground_truth_objects.json"
        with open(output_json_path, "w", encoding="utf-8") as f: json.dump(ground_truth, f, indent=4, cls=NumpyEncoder)
        profiler.close()
        return output_json_path

    except Exception as e:
        err_msg = f"FATAL ERROR IN CALIBRATION: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"
        profiler.log_msg(err_msg)
        profiler.close()
        raise Exception(f"Calibration crashed. See debug.txt for details. Error: {str(e)}")

def generate_path_ground_truth(cal_folder, object_rf_path, e2e_rf_path, phase, squares_per_image, square_size, max_paths, status_callback=None, config_path=None):
    folder_path = Path(cal_folder)
    profiler = DebugProfiler(folder_path / f"debug_calibration_{phase}.txt")
    
    square_size = 8080
    constants = load_constants(config_path)
    bm = constants.get("border_margin_px", 400)
    vec_len = constants.get("t_junc_vector_length_px", 40)
    cone_angle = constants.get("t_junc_cone_angle_deg", 90)
    d_mult = constants.get("dijkstra_cost_multiplier", 300.0) 
    clip_perc = constants.get("dijkstra_tophat_clip_percentile", 80.0)
    dark_factor = float(constants.get("dijkstra_darkness_factor", 0.55))
    
    c_dil_kernel = constants.get("closing_dilation_kernel", 150)
    c_rep_radius = constants.get("closing_repulsion_radius", 18)
    c_rep_penalty = constants.get("closing_repulsion_penalty", 10000000)
    c_max_radius = constants.get("max_radius", 4000)
    
    def log(msg):
        if status_callback: status_callback(msg)
        profiler.log_msg(msg)

    try:
        image_paths = [p for p in folder_path.rglob("*.tif") if "_mask" not in p.name and "_overlay" not in p.name]
        object_rf = joblib.load(object_rf_path)
        e2e_rf = joblib.load(e2e_rf_path) if e2e_rf_path else None
        
        ground_truth = []
        for idx, image_path in enumerate(image_paths, 1):
            log(f"Img {idx}/{len(image_paths)}: Extracting...")
            with tifffile.TiffFile(str(image_path)) as tif: image = tif.asarray()
            h, w = image.shape[:2]
            mask_path = image_path.parent / f"{image_path.stem}_mask.png"
            border_mask = np.array(Image.open(str(mask_path)).convert('L')) if mask_path.exists() else np.full((h, w), 255, dtype=np.uint8)

            i, max_attempts = 0, 200
            squares = []
            sq_w = min(square_size, w)
            sq_h = min(square_size, h)
            
            while i < squares_per_image and max_attempts > 0:
                sx, sy = int((w - sq_w) * random.random()), int((h - sq_h) * random.random())
                if np.mean(border_mask[sy:sy + sq_h, sx:sx + sq_w]) < 200:  
                    max_attempts -= 1; continue 
                squares.append((sx, sx + sq_w, sy, sy + sq_h))
                i += 1; max_attempts -= 1

            for sq_idx, (x0, x1, y0, y1) in enumerate(squares, 1):
                log(f"Img {idx} | Sq {sq_idx}: Processing arrays...")
                t_sq, _ = profiler.log_step(f"Square {sq_idx}")
                
                sq_img, sq_mask = image[y0:y1, x0:x1], border_mask[y0:y1, x0:x1]
                
                objs, _ = detect_grain_boundaries(sq_img, False, sq_mask, parameters=constants)
                objs_calc = calculate_values(objs, sq_img)
                filtered_objs = filter_objects(objs_calc, object_rf)
                
                base_mask = np.zeros_like(sq_img, dtype=np.uint8)
                for obj in filtered_objs:
                    c = np.array(obj["coordinates"])
                    if len(c) > 0: base_mask[c[:,1], c[:,0]] = 1
                    
                skeleton = skeletonize(chunked_edt_mask(base_mask, radius=c_dil_kernel//2)).astype(np.uint8)
                labels, num_labels = ndi.label(skeleton, structure=np.ones((3,3), dtype=bool))
                
                # Combine ridge contrast and image darkness into the routing cost map.
                tophat_raw = chunked_black_tophat(sq_img, kernel_size=12)
                clip_val = float(np.percentile(tophat_raw, clip_perc))
                tophat_norm = np.clip(tophat_raw, 0, clip_val)
                tophat_norm = tophat_norm / (np.max(tophat_norm) + 1e-5)
                
                blurred = cv2.GaussianBlur(sq_img, (7, 7), 0)
                valid_ice = blurred[sq_mask > 0] if np.any(sq_mask > 0) else blurred.flatten()
                if len(valid_ice) == 0: valid_ice = blurred.flatten()
                
                ice_median = float(np.median(valid_ice[::10]))
                dynamic_cutoff = ice_median * dark_factor
                
                darkness_norm = 1.0 - (np.clip(blurred, 0, dynamic_cutoff).astype(np.float32) / (dynamic_cutoff + 1e-5))
                
                combined_score = np.maximum(tophat_norm, darkness_norm)
                cost_map = combined_score * -(d_mult / (np.max(combined_score) + 1e-5)) + (1.0 + d_mult)
                
                cost_map[skeleton == 1] = 99999.0
                cost_map[(chunked_edt_mask(skeleton, radius=c_rep_radius) == 1) & (skeleton != 1)] += c_rep_penalty
                
                ep_coords = np.argwhere((skeleton == 1) & (chunked_neighbor_count(skeleton) == 1))
                border_zone = np.zeros_like(sq_img, dtype=bool)
                border_zone[:bm, :] = True; border_zone[-bm:, :] = True; border_zone[:, :bm] = True; border_zone[:, -bm:] = True
                endpoints = [{'coord': (y, x), 'label': labels[y, x], 'near_border': border_zone[y, x], 'connected': False} for y, x in ep_coords]
                
                potential_paths = []
                log(f"Img {idx} | Sq {sq_idx}: Multithreaded Dijkstra...")
                
                if phase == "E2E":
                    pairs = []
                    if len(endpoints) >= 2:
                        tree = cKDTree([ep['coord'] for ep in endpoints])
                        for k, ep1 in enumerate(endpoints):
                            if ep1['near_border']: continue
                            dists, indices = tree.query(ep1['coord'], k=10, distance_upper_bound=c_max_radius)
                            for j in np.atleast_1d(indices):
                                if j == len(endpoints): continue
                                if k >= j: continue
                                ep2 = endpoints[j]
                                if not ep2['near_border']: 
                                    pairs.append((k, j, ep1, ep2))
                    
                    def _process_e2e(pair):
                        k, j, ep1, ep2 = pair
                        try:
                            path, cost = _local_route_through_array_ts(cost_map, ep1['coord'], ep2['coord'], margin=200)
                            if not path_crosses_skeleton(path, skeleton):
                                feat = calculate_path_features(path, cost, sq_img, skeleton)
                                if feat: return {"ep1": k, "ep2": j, "path": path, "values": feat, "phase": "E2E"}
                        except ValueError: pass
                        return None
                        
                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        res = list(executor.map(_process_e2e, pairs))
                    potential_paths = [r for r in res if r is not None]
                
                elif phase == "T-Junc":
                    e2e_pairs = []
                    if len(endpoints) >= 2:
                        tree = cKDTree([ep['coord'] for ep in endpoints])
                        for k, ep1 in enumerate(endpoints):
                            if ep1['near_border']: continue
                            dists, indices = tree.query(ep1['coord'], k=10, distance_upper_bound=c_max_radius)
                            for j in np.atleast_1d(indices):
                                if j == len(endpoints): continue
                                if k >= j: continue
                                ep2 = endpoints[j]
                                if not ep2['near_border']: 
                                    e2e_pairs.append((k, j, ep1, ep2))
                            
                    def _process_e2e_prep(pair):
                        k, j, ep1, ep2 = pair
                        try:
                            path, cost = _local_route_through_array_ts(cost_map, ep1['coord'], ep2['coord'], margin=200)
                            if not path_crosses_skeleton(path, skeleton):
                                feat = calculate_path_features(path, cost, sq_img, skeleton)
                                if feat: return {"ep1": k, "ep2": j, "path": path, "values": feat}
                        except ValueError: pass
                        return None

                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        e2e_paths = [r for r in list(executor.map(_process_e2e_prep, e2e_pairs)) if r is not None]
                    
                    if e2e_paths:
                        df_e2e = pd.DataFrame([pt["values"] for pt in e2e_paths])[e2e_rf.feature_names_in_]
                        probs = e2e_rf.predict_proba(df_e2e)[:, 1]
                        for k in range(len(e2e_paths)): e2e_paths[k]["prob"] = probs[k]
                        e2e_paths.sort(key=lambda x: x["prob"], reverse=True)
                        
                        drawn_mask = np.zeros_like(skeleton)
                        for path_info in e2e_paths:
                            if path_info["prob"] >= 0.5:
                                if endpoints[path_info["ep1"]]['connected'] or endpoints[path_info["ep2"]]['connected']: continue
                                path = path_info["path"]
                                crosses = False
                                for y, x in path[2:-2]:
                                    if drawn_mask[y, x] == 1: crosses = True; break
                                if crosses: continue
                                for y, x in path: skeleton[y, x] = 1; drawn_mask[y, x] = 1
                                endpoints[path_info["ep1"]]['connected'] = True; endpoints[path_info["ep2"]]['connected'] = True

                    tj_pairs = []
                    for k, ep in enumerate(endpoints):
                        if ep['connected'] or ep['near_border']: continue
                        for target_pixel in _find_t_junc_targets_cone(ep['coord'][0], ep['coord'][1], skeleton, c_max_radius, vec_len, cone_angle):
                            tj_pairs.append((k, ep, target_pixel))
                            
                    def _process_tj(pair):
                        k, ep, target_pixel = pair
                        try:
                            path, cost = _local_route_through_array_ts(cost_map, ep['coord'], target_pixel, margin=200)
                            if not path_crosses_skeleton(path, skeleton):
                                feat = calculate_path_features(path, cost, sq_img, skeleton)
                                if feat: return {"ep1": k, "ep2": -1, "path": path, "values": feat, "phase": "T-Junc"}
                        except ValueError: pass
                        return None

                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        res = list(executor.map(_process_tj, tj_pairs))
                    potential_paths = [r for r in res if r is not None]

                if len(potential_paths) > max_paths:
                    potential_paths = random.sample(potential_paths, max_paths)

                if potential_paths:
                    ground_truth.append({
                        "image_path": str(image_path), "x_start": x0, "x_end": x1, "y_start": y0, "y_end": y1,
                        "paths": potential_paths, "skeleton": skeleton 
                    })
                    
                del sq_img, sq_mask, objs, objs_calc, filtered_objs, base_mask, skeleton, labels
                del cost_map, ep_coords, border_zone, endpoints
                gc.collect()
                profiler.end_step(f"Square {sq_idx}")

            del image, border_mask
            gc.collect()

        log("Opening interactive labeling tool...")
        labeled_data = []
        for item in ground_truth:
            with tifffile.TiffFile(item["image_path"]) as tif: full_img = tif.asarray()
            gui_img_chunk = full_img[item["y_start"]:item["y_end"], item["x_start"]:item["x_end"]]
            
            # Fit the annotation window within the available display area.
            item["paths"] = label_paths_manually(item["paths"], gui_img_chunk, item["skeleton"], border_margin=bm, target_display_height=765)
            
            del full_img, gui_img_chunk, item["skeleton"]
            gc.collect()
            labeled_data.append(item)

        output_json_path = folder_path / f"ground_truth_{phase.lower()}.json"
        with open(output_json_path, "w", encoding="utf-8") as f: json.dump(labeled_data, f, indent=4, cls=NumpyEncoder)
        profiler.close()
        return output_json_path

    except Exception as e:
        err_msg = f"FATAL ERROR IN PATH CALIBRATION: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"
        profiler.log_msg(err_msg)
        profiler.close()
        raise Exception(f"Path Calibration crashed. See debug.txt for details. Error: {str(e)}")

def train_ml_model(json_path, target_type="object", auto_tune=True, status_callback=None):
    def log(msg):
        if status_callback: status_callback(msg)
        print(f"[Training] {msg}")

    with open(json_path, "r", encoding="utf-8") as f: ground_truth = json.load(f)
    X_data, y_data = [], []
    key = "objects" if target_type == "object" else "paths"

    for item in ground_truth:
        for obj in item.get(key, []):
            if "label" in obj and "values" in obj:
                features = obj["values"].copy()
                if "phase" in features: del features["phase"]
                X_data.append(features); y_data.append(obj["label"])

    if not X_data: raise ValueError("No labeled data found in JSON!")
    df_X = pd.DataFrame(X_data); y = np.array(y_data)
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42, stratify=y)
    
    if auto_tune:
        log("Auto-Tuning (RandomizedSearchCV - Single Threaded)...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        scorer = make_scorer(recall_score, pos_label=1)
        search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=20, scoring=scorer, cv=3, n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    else:
        log("Training with default parameters...")
        best_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=1)
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    acc_str = f"{accuracy_score(y_test, y_pred) * 100:.2f}%"
    model_save_path = Path(json_path).parent / f"rf_model_{target_type}.joblib"
    joblib.dump(best_model, model_save_path)
    return model_save_path.name, acc_str
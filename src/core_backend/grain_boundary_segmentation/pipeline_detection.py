import logging
import os
import psutil
from pathlib import Path
import pandas as pd
import joblib
import tifffile
import imageio.v3 as iio
import numpy as np
import cv2
from collections import Counter

from core_backend.grain_boundary_segmentation.constants_manager import load_constants
from core_backend.grain_boundary_segmentation.logger_config import setup_logging, DebugProfiler
from core_backend.grain_boundary_segmentation.detection_grain_boundaries import detect_grain_boundaries
from core_backend.grain_boundary_segmentation.calculating_values import calculate_values
from core_backend.grain_boundary_segmentation.closing_grain_boundaries import close_grain_boundaries
from core_backend.grain_boundary_segmentation.grain_topology import reconstruct_grains
from core_backend.grain_boundary_segmentation.exporting_as_h5 import export_as_h5
from core_backend.grain_boundary_segmentation.gis_export import export_as_gpkg

def _draw_legend(img, color_map_bgr, labels):
    box_width = 300
    box_height = 30 + len(labels) * 35
    legend_overlay = img.copy()
    cv2.rectangle(legend_overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(legend_overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, "Legend", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset = 80
    for cls_id, label in labels.items():
        color = color_map_bgr.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (20, y_offset - 15), (40, y_offset + 5), color, -1)
        cv2.putText(img, label, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35

def run_detection_pipeline(image_path, model_path, e2e_model_path=None, tjunc_model_path=None, 
                           path_prob_threshold=0.5, resolution=1.41, save_intermediates=False, 
                           progress_callback=None, status_callback=None, image_callback=None, 
                           crop_size=None, config_path=None, **kwargs):
                           
    def notify_progress(step_val, step_msg):
        if status_callback: status_callback(step_msg)
        if progress_callback:
            try: progress_callback(step_val)
            except TypeError: pass

    image_path = Path(image_path)
    base_dir = image_path.parent
    results_dir = base_dir / f"{image_path.stem}_pipeline_results"
    
    # ---------------------------------------------------------
    # Preserve diagnostic images for each pipeline stage.
    # ---------------------------------------------------------
    hard_debug_dir = results_dir / "HARD_DEBUG_TRACE"
    hard_debug_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(results_dir)
    logging.info("=" * 50)
    logging.info(f"Starting Grain Boundary Detection Pipeline (HARD DEBUG MODE)")
    logging.info(f"All diagnostic images will be forced to: {hard_debug_dir}")
    
    try:
        total_ram = psutil.virtual_memory().total / (1024**3)
        logging.info(f"-> System Specs: {os.cpu_count()} Logical Processors | {total_ram:.1f} GB Total RAM")
    except Exception:
        pass
    
    pipeline_params = load_constants(config_path)
    profiler = DebugProfiler(results_dir)
    
    try:
        profiler.start_step("[MACRO] Loading ML models")
        notify_progress(5, "Loading ML models...")
        rf_model = joblib.load(model_path)
        e2e_rf = joblib.load(e2e_model_path) if e2e_model_path else None
        tjunc_rf = joblib.load(tjunc_model_path) if tjunc_model_path else None
        profiler.end_step("[MACRO] Loading ML models")

        profiler.start_step("[MACRO] Loading image and mask")
        notify_progress(10, "Loading image and mask...")
        with tifffile.TiffFile(str(image_path)) as tif: image = tif.asarray()
        mask_path = image_path.parent / f"{image_path.stem}_mask.png"
        border_mask = iio.imread(str(mask_path)) if mask_path.exists() else np.full(image.shape[:2], 255, dtype=np.uint8)

        # Save the border mask used by the detection stages.
        cv2.imwrite(str(hard_debug_dir / "00_input_border_mask.png"), border_mask)

        start_x, start_y = 0, 0
        if crop_size is not None:
            cs = int(crop_size)
            h, w = image.shape[:2]
            start_x = max(0, w // 2 - cs // 2)
            start_y = max(0, h // 2 - cs // 2)
            end_x = min(w, start_x + cs)
            end_y = min(h, start_y + cs)
            image = image[start_y:end_y, start_x:end_x]
            border_mask = border_mask[start_y:end_y, start_x:end_x]
            
        profiler.end_step("[MACRO] Loading image and mask")

        profiler.start_step("[MACRO] Phase 1: Detecting initial boundaries")
        notify_progress(20, "Detecting initial boundaries...")
        
        initial_objects, junctions = detect_grain_boundaries(
            image=image, show=False, border_mask=border_mask, 
            intermediates_dir=hard_debug_dir, 
            debug_dir=hard_debug_dir,
            parameters=pipeline_params
        )
        profiler.end_step("[MACRO] Phase 1: Detecting initial boundaries")

        # Save the raw boundary detections before classification.
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for obj in initial_objects:
            c = np.array(obj["coordinates"])
            if len(c) > 0: raw_mask[c[:, 1], c[:, 0]] = 255
        cv2.imwrite(str(hard_debug_dir / "01_raw_detected_lines.png"), raw_mask)
        logging.info(f"Total raw objects detected: {len(initial_objects)}")

        profiler.start_step("[MACRO] Phase 2a: Calculating values")
        notify_progress(40, "Calculating features...")
        raw_rf_data = calculate_values(initial_objects, image)
        rf_data_flat = []
        for obj in raw_rf_data:
            row = {"coordinates": obj.get("coordinates", [])}
            row.update(obj.get("values", {}))
            rf_data_flat.append(row)
        profiler.end_step("[MACRO] Phase 2a: Calculating values")

        profiler.start_step("[MACRO] Phase 2b: Classifying objects")
        notify_progress(50, "Classifying objects...")
        features_df = pd.DataFrame(rf_data_flat)
        if features_df.empty: 
            logging.error("Feature DataFrame is completely empty! Pipeline aborted.")
            return None, None
            
        valid_mask = features_df['length_px'] > 0
        features_df = features_df[valid_mask]
        X_pred = features_df.drop(columns=['coordinates'])
        try: X_pred = X_pred[rf_model.feature_names_in_]
        except AttributeError: pass

        features_df['predicted_class'] = rf_model.predict(X_pred)
        profiler.end_step("[MACRO] Phase 2b: Classifying objects")

        # Record the number of objects assigned to each class.
        class_counts = Counter(features_df['predicted_class'])
        logging.info(f"--- RANDOM FOREST PREDICTION COUNTS ---")
        for cls_val, cnt in class_counts.items():
            logging.info(f"  -> Class {cls_val}: {cnt} objects")

        filtered_objects = features_df[features_df['predicted_class'] == 1].to_dict('records')
        
        # Save the class-1 objects passed to the gap-closing stage.
        rf_class1_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for obj in filtered_objects:
            c = np.array(obj["coordinates"])
            if len(c) > 0: rf_class1_mask[c[:, 1], c[:, 0]] = 255
        cv2.imwrite(str(hard_debug_dir / "02_rf_class1_only_input.png"), rf_class1_mask)

        if len(filtered_objects) == 0:
            logging.error("CRITICAL: 0 objects classified as Class 1. Closing phase will receive empty data!")

        profiler.start_step("[MACRO] Phase 3: Closing gaps (ML Dijkstra)")
        notify_progress(60, "Closing network gaps (ML routing)...")
        
        closed_grainboundaries, new_lines_info, diagnostic_masks, final_objects_info = close_grain_boundaries(
            filtered_objects=filtered_objects,
            image=image,
            border_mask=border_mask,
            debug_dir=hard_debug_dir,
            image_callback=image_callback,
            e2e_rf=e2e_rf,
            tjunc_rf=tjunc_rf,
            prob_threshold=path_prob_threshold,
            parameters=pipeline_params
        )
        profiler.end_step("[MACRO] Phase 3: Closing gaps (ML Dijkstra)")
        
        # Save the skeleton produced by gap closing.
        cv2.imwrite(str(hard_debug_dir / "03_closed_skeleton_output.png"), closed_grainboundaries * 255)

        subgrains_objs = features_df[features_df['predicted_class'] == 2].to_dict('records')
        bottom_objs = features_df[features_df['predicted_class'] == 3].to_dict('records')

        profiler.start_step("[MACRO] Phase 4: Reconstructing Grains Topology")
        notify_progress(70, "Reconstructing grain topology...")
        
        # Retain intermediate topology outputs for pipeline diagnostics.
        grain_labels, interior_remap, border_remap = reconstruct_grains(
            skeleton=closed_grainboundaries,
            border_mask=border_mask,
            resolution_um_per_px=resolution,
            debug_dir=hard_debug_dir
        )
        profiler.end_step("[MACRO] Phase 4: Reconstructing Grains Topology")

        profiler.start_step("[MACRO] Phase 5: Exporting HDF5 layers")
        notify_progress(80, "Exporting HDF5 data...")
        h5_path = export_as_h5(
            image_path=image_path, 
            grain_labels=grain_labels,            
            interior_remap=interior_remap,        
            border_remap=border_remap,            
            new_lines_info=new_lines_info,
            diagnostic_masks=diagnostic_masks, 
            final_objects_info=final_objects_info,
            subgrains=subgrains_objs, 
            bottom_grains=bottom_objs, 
            output_dir=results_dir,
            resolution_um_per_px=resolution, 
            crop_offset=(start_x, start_y)
        )
        profiler.end_step("[MACRO] Phase 5: Exporting HDF5 layers")

        profiler.start_step("[MACRO] Phase 6: Exporting GIS Vectors")
        notify_progress(90, "Exporting Vector GIS...")
        gpkg_path = export_as_gpkg(
            image_array=image, 
            image_path=image_path, 
            grain_labels=grain_labels,            
            interior_remap=interior_remap,        
            border_remap=border_remap,            
            new_lines_info=new_lines_info, 
            diagnostic_masks=diagnostic_masks, 
            final_objects_info=final_objects_info, 
            output_dir=results_dir, 
            subgrains=subgrains_objs, 
            bottom_grains=bottom_objs,
            resolution_um_per_px=resolution, 
            crop_offset=(start_x, start_y)   
        )
        profiler.end_step("[MACRO] Phase 6: Exporting GIS Vectors")

        notify_progress(100, "Done!")
        return h5_path, gpkg_path

    except Exception as e:
        logging.error(f"Pipeline crashed: {e}", exc_info=True)
        raise e
    finally:
        profiler.end_session()
import os
import cv2
import json
import numpy as np
import pathlib
import pandas as pd
import random
import joblib
from PIL import Image, ImageTk
import customtkinter as ctk
import logging
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from core_backend.hydrate_segmentation.hydrate_detection import load_global_constants
import core_backend.hydrate_segmentation.hydrate_detection as hd
from core_backend.hydrate_segmentation.hydrate_rescue_rf import get_all_rescue_candidates

load_global_constants()

# GUI for rapidly labeling candidate hydrate contours.

class ContourLabelTool(ctk.CTkToplevel):
    def __init__(self, master, candidates, save_csv_path, on_complete_callback):
        super().__init__(master)
        self.title("Phase 2 - Random Forest Rapid Labeling")
        self.geometry("600x700")
        self.attributes("-topmost", True)
        self.focus_force()

        self.candidates = candidates
        self.save_csv_path = save_csv_path
        self.on_complete_callback = on_complete_callback
        
        self.current_index = 0
        self.labeled_data = []

        self.lbl_progress = ctk.CTkLabel(self, text="", font=("Arial", 16, "bold"))
        self.lbl_progress.pack(pady=10)

        self.lbl_algo = ctk.CTkLabel(self, text="", font=("Arial", 14, "italic"), text_color="#3498db")
        self.lbl_algo.pack(pady=0)

        self.canvas = ctk.CTkCanvas(self, width=400, height=400, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)

        instructions = (
            "Bewerte die GRÜNE KONTUR:\n\n"
            "[1] = Single Hydrate (Sauberes Hydrat)\n"
            "[2] = On Boundary (Hydrat an Korngrenze)\n"
            "[3] = Two Hydrates (Zwei Hydrate / Brücke)\n"
            "[4] = Nonsense (Artefakt / Trash)\n"
        )
        self.lbl_instructions = ctk.CTkLabel(self, text=instructions, font=("Arial", 16), justify="left")
        self.lbl_instructions.pack(pady=20)

        self.bind("<KeyPress-1>", lambda e: self.record_label(1))
        self.bind("<KeyPress-2>", lambda e: self.record_label(2))
        self.bind("<KeyPress-3>", lambda e: self.record_label(3))
        self.bind("<KeyPress-4>", lambda e: self.record_label(4))
        self.bind("<KP_1>", lambda e: self.record_label(1))
        self.bind("<KP_2>", lambda e: self.record_label(2))
        self.bind("<KP_3>", lambda e: self.record_label(3))
        self.bind("<KP_4>", lambda e: self.record_label(4))

        self.show_next_candidate()

    def show_next_candidate(self):
        if self.current_index >= len(self.candidates):
            self.finish_labeling()
            return

        self.lbl_progress.configure(text=f"Kandidat {self.current_index + 1} von {len(self.candidates)}")
        
        cand = self.candidates[self.current_index]
        roi_img = cand["roi_image"].copy()
        new_contour = cand["rescue_contour"]
        old_contour = cand["original_contour"]
        algo = cand["source_algo"]
        folder = cand.get("source_folder", "unknown")
        
        self.lbl_algo.configure(text=f"Ordner: {folder} | Quelle: {algo.upper()}")

        if len(roi_img.shape) == 2:
            roi_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
        else:
            roi_color = roi_img

        cv2.drawContours(roi_color, [old_contour], -1, (0, 215, 255), 1)
        cv2.drawContours(roi_color, [new_contour], -1, (0, 255, 0), 2)

        scale_factor = min(400 / roi_color.shape[1], 400 / roi_color.shape[0])
        new_w = int(roi_color.shape[1] * scale_factor)
        new_h = int(roi_color.shape[0] * scale_factor)
        roi_resized = cv2.resize(roi_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pil_img = Image.fromarray(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(200, 200, image=self.tk_img, anchor="center")

    def record_label(self, label_class):
        cand = self.candidates[self.current_index]
        feature_row = cand["features_9"].copy()
        feature_row["label"] = label_class
        feature_row["source_stack"] = cand["source_stack"]
        feature_row["source_folder"] = cand.get("source_folder", "unknown")
        
        self.labeled_data.append(feature_row)
        self.current_index += 1
        self.show_next_candidate()

    def finish_labeling(self):
        df = pd.DataFrame(self.labeled_data)
        if os.path.exists(self.save_csv_path):
            df.to_csv(self.save_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.save_csv_path, index=False)
            
        print(f"[{len(self.labeled_data)}] neue Konturen gesichert in: {self.save_csv_path}")
        self.destroy()
        if self.on_complete_callback:
            self.on_complete_callback()


# Candidate extraction and model-training data preparation.

def extract_category_2_data(controller, parent_folder_path):
    load_global_constants()
    current_constants = hd.GLOBAL_CONSTANTS

    parent_dir = pathlib.Path(parent_folder_path)
    rf_dir = parent_dir / "random_forest_data"
    rf_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rf_dir / "rf_training_features.csv"

    det = current_constants["phase_1_detection"]
    lb = current_constants["phase_1_triage_bounds"]["loose_bounds"]
    sb = current_constants["phase_1_triage_bounds"]["strict_bounds"]
    rescue_params = current_constants["phase_2_rescue"]
    
    batch_size = current_constants.get("rf_params", {}).get("extraction_batch_size", 250)

    import tifffile
    
    subfolders = [f for f in parent_dir.iterdir() if f.is_dir() and f.name != "random_forest_data"]
    if not subfolders:
        subfolders = [parent_dir] 

    metadata_map = {}
    for mf in parent_dir.rglob("metadata_preprocessed.json"):
        with open(mf, "r") as f:
            metadata_map[mf.parent.name] = json.load(f)

    candidates = []

    clahe = cv2.createCLAHE(clipLimit=det.get("clip_limit", 1.0), tileGridSize=(8, 8))
    ed = cv2.ximgproc.createEdgeDrawing()
    ed_p = ed.Params()
    ed_p.AnchorThresholdValue = det.get("ed_anchor_thresh", 20)
    ed_p.GradientThresholdValue = det.get("ed_grad_thresh", 5)
    ed_p.MinPathLength = det.get("ed_min_path_len", 50)
    ed_p.ScanInterval = 1
    ed.setParams(ed_p)
    k_morph = det.get("morph_kernel_size", 11)
    morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph)) if k_morph > 1 else None

    for subfolder in subfolders:
        stacks = list(subfolder.glob("*.tif"))
        if not stacks: continue
        random.shuffle(stacks)
        
        folder_cand_count = 0
        
        curr_meta = metadata_map.get(subfolder.name)
        if not curr_meta and metadata_map:
            curr_meta = list(metadata_map.values())[0]
        cam_meta = curr_meta.get("Camera_preprocessed", {}) if curr_meta else {}
        pixel_scale = float(cam_meta.get("pixelSizeUm", current_constants["global_settings"].get("pixel_scale_fallback", 1.41)))
        
        min_area_um2 = rescue_params.get("roi_params", {}).get("min_area_um2", 3000.0)
        min_fill_factor = rescue_params.get("roi_params", {}).get("min_fill_factor", 0.5)

        for stack_path in stacks:
            if folder_cand_count >= batch_size: break
            try:
                with tifffile.TiffFile(stack_path) as tif:
                    stack = tif.asarray() if len(tif.pages) == 1 else np.stack([p.asarray() for p in tif.pages], axis=0)
                if stack.ndim == 2: stack = np.expand_dims(stack, axis=0)
                elif stack.ndim == 3 and stack.shape[0] > stack.shape[-1]: stack = np.moveaxis(stack, -1, 0)
            except Exception: continue

            for layer in range(stack.shape[0]):
                if folder_cand_count >= batch_size: break
                raw_img = stack[layer]
                h_img, w_img = raw_img.shape

                img_clahe = clahe.apply(raw_img)
                k_med = det.get("median_kernel_size", 7)
                img_prep = cv2.medianBlur(img_clahe, k_med) if k_med > 1 else img_clahe

                try:
                    ed.detectEdges(img_prep)
                    img_bin = ed.getEdgeImage()
                    if img_bin is None: img_bin = np.zeros_like(img_prep)
                except Exception: img_bin = np.zeros_like(img_prep)

                # Fill detected contours and remove narrow artifacts.
                if np.any(img_bin):
                    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filled_mask = np.zeros_like(img_bin)
                    valid_contours = [c for c in contours if len(c) >= 3]
                    cv2.drawContours(filled_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
                    kernel_open = np.ones((3, 3), np.uint8)
                    img_bin = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_open)

                if morph_k is not None: img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, morph_k)
                contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if folder_cand_count >= batch_size: break
                    area = cv2.contourArea(contour)
                    if len(contour) < 5 or area < 10.0: continue

                    perimeter = cv2.arcLength(contour, True)
                    try: (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(contour)
                    except cv2.error: continue

                    major_axis = max(ax1, ax2)
                    minor_axis = min(ax1, ax2)
                    aspect_ratio = major_axis / minor_axis if minor_axis > 1e-3 else 99.0
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                    ellipse_area = (np.pi / 4.0) * major_axis * minor_axis
                    fill_factor = area / ellipse_area if ellipse_area > 0 else 0.0

                    hull = cv2.convexHull(contour)
                    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0.0
                    
                    area_um2 = area * (pixel_scale * pixel_scale)

                    bx, by, bw, bh = cv2.boundingRect(contour)
                    x1_b, y1_b = max(0, bx), max(0, by)
                    x2_b, y2_b = min(w_img, bx + bw), min(h_img, by + bh)

                    local_img = img_prep[y1_b:y2_b, x1_b:x2_b]
                    local_mask = np.zeros((y2_b - y1_b, x2_b - x1_b), dtype=np.uint8)
                    c_loc = (contour - [x1_b, y1_b]).astype(np.int32)
                    cv2.drawContours(local_mask, [c_loc], -1, 255, thickness=cv2.FILLED)

                    int_pix = local_img[local_mask == 255]
                    b_core = float(np.percentile(int_pix, 75)) if int_pix.size > 0 else 0.0

                    passes_loose = (
                        lb["area_min_px"] <= area <= lb["area_max_px"] and
                        ellipse_area <= lb["ellipse_area_max_px"] and
                        aspect_ratio <= lb["aspect_max"] and
                        circularity >= lb["circ_min"] and
                        fill_factor >= lb["fill_min"] and
                        solidity >= lb["solid_min"] and
                        b_core >= lb["bright_min"]
                    )

                    passes_strict = (
                        sb["area_min_px"] <= area <= sb["area_max_px"] and
                        ellipse_area <= sb["ellipse_area_max_px"] and
                        aspect_ratio <= sb["aspect_max"] and
                        circularity >= sb["circ_min"] and
                        fill_factor >= sb["fill_min"] and
                        solidity >= sb["solid_min"] and
                        b_core >= sb["bright_min"]
                    )

                    # Route small otherwise-valid objects through the rescue stage.
                    if passes_strict and area_um2 < min_area_um2:
                        passes_strict = False

                    # Exclude strict-pass objects from rescue-model training.
                    if passes_strict:
                        continue 
                        
                    # Exclude objects that fail the loose triage bounds.
                    if not passes_loose:
                        continue 

                    roi_factor = rescue_params["roi_params"]["roi_factor"]
                    radius = (major_axis * roi_factor) / 2.0
                    rx1 = max(0, int(cx - radius))
                    rx2 = min(w_img, int(cx + radius))
                    ry1 = max(0, int(cy - radius))
                    ry2 = min(h_img, int(cy + radius))

                    if (rx2 - rx1) < 15 or (ry2 - ry1) < 15: continue

                    raw_roi = raw_img[ry1:ry2, rx1:rx2].copy()
                    orig_cnt_local = (contour - [rx1, ry1]).astype(np.int32)

                    rescue_dicts = get_all_rescue_candidates(raw_roi, rescue_params, orig_cnt_local)
                    
                    # Apply area and fill-factor constraints to rescue candidates.
                    for r_dict in rescue_dicts:
                        c_area_um2 = r_dict["features"]["area"] * (pixel_scale * pixel_scale)
                        
                        if c_area_um2 >= min_area_um2 and r_dict["features"]["fill_factor"] >= min_fill_factor:
                            if folder_cand_count >= batch_size: break
                            candidates.append({
                                "roi_image": raw_roi,
                                "original_contour": orig_cnt_local,
                                "rescue_contour": r_dict["contour"],
                                "features_9": r_dict["features"],
                                "source_algo": r_dict["source"],
                                "source_stack": stack_path.stem,
                                "source_folder": subfolder.name 
                            })
                            folder_cand_count += 1

    if not candidates:
        print("Keine rettbaren Konturen in allen Unterordnern gefunden!")
        return

    controller.after(0, lambda: ContourLabelTool(
        controller, candidates, csv_path, on_complete_callback=lambda: print("Session beendet!")
    ))


def train_random_forest(data_folder, progress_cb=None):
    load_global_constants()
    current_constants = hd.GLOBAL_CONSTANTS

    rf_dir = pathlib.Path(data_folder) / "random_forest_data"
    csv_path = rf_dir / "rf_training_features.csv"
    model_path = rf_dir / "rf_model.joblib"

    if not csv_path.exists():
        logging.error("Keine Trainingsdaten gefunden!")
        return False, "CSV file not found."

    try:
        df = pd.read_csv(csv_path).dropna()
        if len(df) < 10: 
            return False, "Zu wenige Daten für ein sinnvolles Training (< 10)."

        y = df["label"]
        has_folders = False
        folders = None
        
        if "source_stack" in df.columns:
            groups = df["source_stack"]
            X = df.drop(["label", "source_stack"], axis=1)
            has_groups = True
        else:
            X = df.drop("label", axis=1)
            groups = None
            has_groups = False

        if "source_folder" in X.columns:
            folders = X["source_folder"]
            X = X.drop("source_folder", axis=1)
            has_folders = True

        rp = current_constants.get("rf_params", {})
        optimize = rp.get("optimize", False)
        
        report_lines = []
        report_lines.append("="*50)
        report_lines.append("RANDOM FOREST TRAINING REPORT")
        report_lines.append("="*50)
        report_lines.append(f"Total Samples: {len(df)}")
        report_lines.append(f"Features: {X.shape[1]}")
        report_lines.append(f"Classes: {list(np.unique(y))}")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=1)

        if optimize:
            n_iter = rp.get("n_iter", 500)
            cv_splits = rp.get("cv", 5)
            scoring_metric = rp.get("scoring", "f1_macro")
            report_lines.append(f"Mode: OPTIMIZATION ({n_iter} iters, {cv_splits} CV)")
            
            param_distributions = {
                "n_estimators": [100, 200, 300, 500, 800],
                "max_depth": [5, 8, 10, 15, None],
                "min_samples_split": [2, 4, 10],
                "min_samples_leaf": [1, 2, 5],
                "class_weight": [None, "balanced"]
            }

            if has_groups and StratifiedGroupKFold is not None:
                cv_strategy = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            elif has_groups:
                cv_strategy = GroupKFold(n_splits=cv_splits)
            else:
                cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

            search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                        n_iter=n_iter, scoring=scoring_metric, cv=cv_strategy,
                                        n_jobs=1, random_state=42, verbose=0)

            if has_groups: search.fit(X, y, groups=groups)
            else: search.fit(X, y)

            best_params = search.best_params_
            report_lines.append(f"Best CV {scoring_metric}: {search.best_score_:.4f}")
            current_constants["rf_params"].update(best_params)
            
            try:
                with open(hd.get_constants_config_path(), "w", encoding="utf-8") as f:
                    json.dump(current_constants, f, indent=4)
            except Exception: pass

            final_rf = RandomForestClassifier(random_state=42, n_jobs=1, **best_params)
            if has_groups: y_pred_cv = cross_val_predict(final_rf, X, y, cv=cv_strategy, groups=groups, n_jobs=1)
            else: y_pred_cv = cross_val_predict(final_rf, X, y, cv=cv_strategy, n_jobs=1)

        else:
            report_lines.append("Mode: STATIC (No Optimization)")
            rf_kwargs = {
                "n_estimators": rp.get("n_estimators", 200), "max_depth": rp.get("max_depth", 12),
                "min_samples_split": rp.get("min_samples_split", 4), "min_samples_leaf": rp.get("min_samples_leaf", 2),
                "class_weight": rp.get("class_weight", "balanced")
            }
            final_rf = RandomForestClassifier(random_state=42, n_jobs=1, **rf_kwargs)
            cv_strat = GroupKFold(n_splits=5) if has_groups else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            if has_groups: y_pred_cv = cross_val_predict(final_rf, X, y, cv=cv_strat, groups=groups, n_jobs=1)
            else: y_pred_cv = cross_val_predict(final_rf, X, y, cv=cv_strat, n_jobs=1)

        report_lines.append("\n" + "-"*50)
        report_lines.append("GLOBAL MODEL EVALUATION (Out-of-Fold)")
        report_lines.append("-" * 50)
        report_lines.append(f"Global Accuracy:    {accuracy_score(y, y_pred_cv):.4f}")
        report_lines.append(f"Global F1 Macro:    {f1_score(y, y_pred_cv, average='macro'):.4f}")
        report_lines.append("\nGlobal Classification Report:")
        report_lines.append(classification_report(y, y_pred_cv, zero_division=0))

        if has_folders:
            report_lines.append("\n" + "="*50)
            report_lines.append("PER-SAMPLE EVALUATION (Generalization Check)")
            report_lines.append("="*50)
            
            df_eval = pd.DataFrame({"true": y, "pred": y_pred_cv, "folder": folders})
            for folder_name, group in df_eval.groupby("folder"):
                acc = accuracy_score(group["true"], group["pred"])
                f1_mac = f1_score(group["true"], group["pred"], average='macro')
                report_lines.append(f"\n[{folder_name}] -> Samples: {len(group)}")
                report_lines.append(f"   Accuracy: {acc:.4f} | F1 Macro: {f1_mac:.4f}")
                
                labels_in_group = sorted(group["true"].unique())
                report_lines.append(f"   Classes present: {labels_in_group}")

        report_str = "\n".join(report_lines)
        print(report_str)

        final_rf.fit(X, y)
        joblib.dump(final_rf, model_path)

        return True, report_str

    except Exception as e:
        logging.error(f"Fehler beim RF-Training: {e}", exc_info=True)
        return False, str(e)
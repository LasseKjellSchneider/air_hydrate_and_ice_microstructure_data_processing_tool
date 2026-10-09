import sys
import pandas as pd
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog
import threading
import numpy as np
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import gc
import random
import joblib
import matplotlib.pyplot as plt
import colorsys
import tempfile
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import scipy.ndimage as ndi
import concurrent.futures

from core_backend.grain_boundary_segmentation import constants_manager
from core_backend.grain_boundary_segmentation.detection_grain_boundaries import detect_grain_boundaries
from core_backend.grain_boundary_segmentation.calculating_values import calculate_values
from core_backend.grain_boundary_segmentation.filtering_objects import filter_objects
from core_backend.grain_boundary_segmentation.closing_grain_boundaries import (
    chunked_black_tophat, 
    chunked_edt_mask, 
    chunked_neighbor_count, 
    _local_route_through_array_ts, 
    path_crosses_skeleton,
    _find_t_junc_targets_cone,
    CLOSING_PARAMS
)

class GrainBoundaryTunerFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        self.image_paths = []
        self.object_rf_path = ""
        self.cache = []
        self.sliders = {}
        
        self.btn_color = "#D67040"
        self.btn_hover = "#804326"
        self.bg_color = "#1A1311"
        
        self._build_ui()

    def _build_ui(self):
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkButton(top_bar, text="Back", command=lambda: self.controller.show_frame("StartFrame"),
                      corner_radius=20, font=("Arial", 14), width=60, height=30).pack(side="left")
        ctk.CTkLabel(top_bar, text="3-Phase Tuner", font=("Arial", 22, "bold")).pack(side="left", padx=15)

        content_frame = ctk.CTkScrollableFrame(self, fg_color=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        setup_box = ctk.CTkFrame(content_frame, fg_color="#261C18", corner_radius=10)
        setup_box.pack(fill="x", pady=5, padx=5)

        self.lbl_imgs = ctk.CTkLabel(setup_box, text="0 Images selected (1-4)", font=("Arial", 13))
        self.lbl_imgs.pack(pady=(5, 2))
        ctk.CTkButton(setup_box, text="Select TIF Images", fg_color=self.btn_color, hover_color=self.btn_hover,
                      command=self._load_images, height=28).pack(pady=3, padx=15, fill="x")

        self.lbl_rf = ctk.CTkLabel(setup_box, text="No Object RF loaded (Needed for Phase 2/3)", font=("Arial", 12), text_color="gray")
        self.lbl_rf.pack(pady=(5, 2))
        ctk.CTkButton(setup_box, text="Select Object RF Model", fg_color=self.btn_color, hover_color=self.btn_hover,
                      command=self._load_rf, height=28).pack(pady=3, padx=15, fill="x")

        self.lbl_crop = ctk.CTkLabel(setup_box, text="Crop Size: 1500 px", font=("Arial", 13))
        self.lbl_crop.pack(pady=(5, 0))
        self.sl_crop = ctk.CTkSlider(setup_box, from_=500, to=4000, number_of_steps=35, button_color=self.btn_color)
        self.sl_crop.set(1500)
        self.sl_crop.pack(fill="x", padx=15, pady=(0, 5))
        self.sl_crop.configure(command=lambda val: self.lbl_crop.configure(text=f"Crop Size: {int(val)} px"))

        self.btn_extract = ctk.CTkButton(setup_box, text="EXTRACT RANDOM PATCHES", fg_color="#8e44ad", hover_color="#732d91",
                                         font=("Arial", 14, "bold"), command=self._extract_patches, height=35)
        self.btn_extract.pack(pady=8, padx=15, fill="x")

        self.lbl_status = ctk.CTkLabel(content_frame, text="Status: Ready", text_color="orange", font=("Arial", 13, "bold"))
        self.lbl_status.pack(pady=5)

        self.tabview = ctk.CTkTabview(content_frame)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        tab1 = self.tabview.add("1. Detection")
        tab2 = self.tabview.add("2. Morph & RF")
        tab3 = self.tabview.add("3. Routing")

        self._build_phase_1(tab1)
        self._build_phase_2(tab2)
        self._build_phase_3(tab3)

        self.btn_save = ctk.CTkButton(content_frame, text="SAVE PARAMETERS (TIMESTAMP JSON)", font=("Arial", 15, "bold"),
                                      fg_color="#27ae60", hover_color="#2ecc71", command=self._save_parameters, height=45)
        self.btn_save.pack(pady=10, fill="x", padx=10)

    def _add_slider(self, parent, key, name, v_min, v_max, step, default):
        lbl = ctk.CTkLabel(parent, text=f"{name}: {default}", font=("Arial", 13))
        lbl.pack(anchor="w", padx=5)
        sl = ctk.CTkSlider(parent, from_=v_min, to=v_max, number_of_steps=int((v_max - v_min) / step), button_color=self.btn_color)
        sl.set(default)
        sl.pack(fill="x", padx=5, pady=(0, 6))
        sl.configure(command=lambda val, l=lbl, n=name: l.configure(text=f"{n}: {int(val) if val >= 10 else float(val):.1f}"))
        self.sliders[key] = sl

    def _build_phase_1(self, parent):
        lbl = ctk.CTkLabel(parent, text="ridge_sigmas (z.B. 3, 4, 5):", font=("Arial", 13))
        lbl.pack(anchor="w", padx=5)
        self.entry_sigmas = ctk.CTkEntry(parent)
        self.entry_sigmas.insert(0, "3, 4, 5")
        self.entry_sigmas.pack(fill="x", padx=5, pady=(0, 6))

        configs = [
            ("denoise_d", "Denoise D", 1, 15, 1, 7),
            ("denoise_sigma_color", "Sigma Color", 10, 150, 5, 50),
            ("denoise_sigma_space", "Sigma Space", 10, 150, 5, 50),
            ("ridge_threshold", "Ridge Threshold", 5, 100, 1, 40),
            ("ridge_dilation_kernel_size_bevor_filtering", "Dilation Before", 1, 100, 1, 50),
            ("min_length", "Min Line Length", 10, 1000, 10, 200),
            ("ridge_dilation_kernel_size_after_filtering", "Dilation After", 1, 200, 1, 100),
            ("max_hole_area_px", "Max Hole Area", 0, 2000, 50, 500),
            ("min_object_size_px", "Min Object Size", 0, 100, 1, 5),
            ("min_junction_neighbors", "Min Junction Neighbors", 2, 5, 1, 3)
        ]
        for c in configs: self._add_slider(parent, *c)
        ctk.CTkButton(parent, text="UPDATE PHASE 1", command=self._update_phase_1, fg_color="#E67E22", font=("Arial", 14, "bold"), height=35).pack(pady=10, fill="x")

    def _build_phase_2(self, parent):
        configs = [
            ("closing_dilation_kernel", "Base Dilation Kernel", 0, 300, 5, 150),
            ("closing_min_line_length", "Prune Min Line Length", 0, 1000, 10, 200),
            ("rf_agreement_percent", "RF Agreement Threshold (%)", 0, 100, 1, 50)
        ]
        for c in configs: self._add_slider(parent, *c)
        ctk.CTkButton(parent, text="UPDATE PHASE 2", command=self._update_phase_2, fg_color="#E67E22", font=("Arial", 14, "bold"), height=35).pack(pady=10, fill="x")

    def _build_phase_3(self, parent):
        configs = [
            ("dijkstra_cost_multiplier", "Cost Multiplier (Avoid Ice)", 10, 1000, 10, 300),
            ("dijkstra_tophat_clip_percentile", "Tophat Clip Percentile (Sat)", 50, 100, 1, 80),
            ("dijkstra_darkness_factor", "Darkness Factor (Hydrates)", 0.0, 1.0, 0.05, 0.55),
            ("max_radius", "Max Path Radius (px)", 200, 5000, 50, 4000),
            ("border_margin_px", "Border Margin (px)", 0, 400, 10, 50),
            ("t_junc_vector_length_px", "T-Junc Vector Length", 10, 150, 5, 40),
            ("t_junc_cone_angle_deg", "T-Junc Cone Angle", 10, 180, 5, 90),
            ("closing_repulsion_radius", "Repulsion Radius (px)", 1, 50, 1, 18)
        ]
        for c in configs: self._add_slider(parent, *c)
        ctk.CTkButton(parent, text="UPDATE PHASE 3", command=self._update_phase_3, fg_color="#E67E22", font=("Arial", 14, "bold"), height=35).pack(pady=10, fill="x")

    def _load_images(self):
        files = filedialog.askopenfilenames(title="Select 1 to 4 TIF Images", filetypes=[("TIF", "*.tif")])
        if files:
            self.image_paths = list(files[:4])
            self.lbl_imgs.configure(text=f"{len(self.image_paths)} Images selected")

    def _load_rf(self):
        file = filedialog.askopenfilename(title="Select Object RF Model", filetypes=[("Joblib", "*.joblib")])
        if file:
            self.object_rf_path = file
            self.lbl_rf.configure(text=f"RF: {Path(file).name}", text_color="#2ecc71")

    def _get_all_params(self):
        params = constants_manager.load_constants()
        int_keys = {
            "denoise_d", "ridge_threshold", "ridge_dilation_kernel_size_bevor_filtering",
            "ridge_dilation_kernel_size_after_filtering", "min_length", "max_hole_area_px",
            "min_object_size_px", "min_junction_neighbors", "closing_dilation_kernel",
            "closing_min_line_length", "max_radius", "border_margin_px", 
            "t_junc_vector_length_px", "closing_repulsion_radius", "ridge_dilation_iterations",
            "rf_agreement_percent"
        }
        
        for k, v in self.sliders.items():
            val = v.get()
            if k in int_keys:
                params[k] = int(val)
            else:
                params[k] = float(val)
                
        try:
            params["ridge_sigmas"] = [int(x.strip()) for x in self.entry_sigmas.get().split(",")]
        except Exception:
            pass
            
        params["ridge_black_ridges"] = True
        return params

    def _show_intermediates_popup(self, intermediates_dict, title):
        if not intermediates_dict:
            return
        n = len(intermediates_dict)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, (name, img) in enumerate(intermediates_dict.items()):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(name, fontsize=10, color="white")
            axes[i].axis('off')

        for j in range(len(intermediates_dict), len(axes)):
            axes[j].axis('off')

        fig.suptitle(title, fontsize=13, fontweight='bold', color="white")
        fig.patch.set_facecolor('#1A1311')
        for ax in axes:
            ax.set_facecolor('#1A1311')
        fig.tight_layout()
        plt.show(block=False)

    def _display_grid_on_canvas(self, results, title):
        n = len(results)
        if n == 0: return
        elif n == 1: grid_img = results[0]
        elif n == 2: grid_img = np.hstack(results)
        elif n == 3:
            grid_img = np.vstack([np.hstack([results[0], results[1]]), np.hstack([results[2], np.zeros_like(results[0])])])
        else:
            top_row, bot_row = np.hstack(results[:2]), np.hstack(results[2:4])
            if bot_row.shape[1] < top_row.shape[1]:
                bot_row = cv2.copyMakeBorder(bot_row, 0, 0, 0, top_row.shape[1] - bot_row.shape[1], cv2.BORDER_CONSTANT)
            grid_img = np.vstack([top_row, bot_row])

        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / "grain_tuner_preview.png"
        cv2.imwrite(str(temp_path), grid_img)
        self.controller.show_image_plot(str(temp_path), title)

    def _extract_patches(self):
        if not self.image_paths:
            self.lbl_status.configure(text="Select images first!", text_color="red")
            return
        self.btn_extract.configure(state="disabled")
        self.lbl_status.configure(text="Extracting patches...", text_color="orange")
        crop_dim = int(self.sl_crop.get())

        def task():
            self.cache = []
            for p in self.image_paths:
                with tifffile.TiffFile(p) as tif: img = tif.asarray()
                h, w = img.shape[:2]
                actual_crop = min(crop_dim, h, w)
                max_y, max_x = max(0, h - actual_crop), max(0, w - actual_crop)

                mask_path = Path(p).parent / f"{Path(p).stem}_mask.png"
                best_x0, best_y0 = 0, 0
                if mask_path.exists() and max_x > 0 and max_y > 0:
                    with Image.open(str(mask_path)) as m_img:
                        for _ in range(25):
                            rx, ry = random.randint(0, max_x), random.randint(0, max_y)
                            sub_mask = np.array(m_img.crop((rx, ry, rx + actual_crop, ry + actual_crop)).convert('L'))
                            if np.mean(sub_mask) > 180:
                                best_x0, best_y0, bmask = rx, ry, sub_mask
                                break
                        else:
                            best_x0, best_y0 = random.randint(0, max_x), random.randint(0, max_y)
                            bmask = np.array(m_img.crop((best_x0, best_y0, best_x0 + actual_crop, best_y0 + actual_crop)).convert('L'))
                else:
                    best_x0, best_y0 = random.randint(0, max_x) if max_x > 0 else 0, random.randint(0, max_y) if max_y > 0 else 0
                    bmask = np.full((actual_crop, actual_crop), 255, dtype=np.uint8)

                patch = img[best_y0:best_y0 + actual_crop, best_x0:best_x0 + actual_crop].copy()
                self.cache.append({"name": Path(p).name, "img": patch, "bmask": bmask})
                del img; gc.collect()

            self.after(0, lambda: self.lbl_status.configure(text=f"{len(self.cache)} Patches extracted! Start Phase 1.", text_color="#2ecc71"))
            self.after(0, lambda: self.btn_extract.configure(state="normal"))
        threading.Thread(target=task, daemon=True).start()

    def _update_phase_1(self):
        if not self.cache:
            self.lbl_status.configure(text="Extract patches first!", text_color="red")
            return
        self.lbl_status.configure(text="Running Phase 1 (Detection)...", text_color="orange")
        params = self._get_all_params()

        def task():
            results = []
            first_intermediates = None
            for i, item in enumerate(self.cache):
                patch, bmask = item["img"], item["bmask"]
                objs, _, interm = detect_grain_boundaries(patch, False, bmask, parameters=params, return_intermediates=True)
                item["objs"] = objs

                if i == 0: first_intermediates = interm

                img_8u = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if patch.dtype != np.uint8 else patch.copy()
                disp = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
                overlay = np.zeros_like(disp)

                for obj in objs:
                    coords = np.array(obj["coordinates"])
                    if len(coords) < 2: continue
                    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(random.random(), 0.9, 1.0)]
                    overlay[coords[:, 1], coords[:, 0]] = (b, g, r)

                overlay_thick = cv2.dilate(overlay, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                active = np.any(overlay_thick > 0, axis=-1)
                disp[active] = cv2.addWeighted(overlay_thick, 0.6, disp, 0.4, 0)[active]
                cv2.putText(disp, item["name"][:25], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                results.append(disp)

            self.after(0, lambda: self._show_intermediates_popup(first_intermediates, f"Phase 1 Steps: {self.cache[0]['name']}"))
            self.after(0, lambda: self._display_grid_on_canvas(results, "Phase 1: Detected Objects"))
            self.after(0, lambda: self.lbl_status.configure(text="Phase 1 complete!", text_color="#2ecc71"))
        threading.Thread(target=task, daemon=True).start()

    def _update_phase_2(self):
        if not self.cache or "objs" not in self.cache[0]:
            self.lbl_status.configure(text="Run Phase 1 first!", text_color="red")
            return
        if not self.object_rf_path:
            self.lbl_status.configure(text="Select Object RF Model!", text_color="red")
            return
        self.lbl_status.configure(text="Running Phase 2 (Morphology)...", text_color="orange")
        params = self._get_all_params()

        def task():
            object_rf = joblib.load(self.object_rf_path)
            results = []
            first_intermediates = {}
            
            agreement_threshold = params.get("rf_agreement_percent", 50) / 100.0

            for i, item in enumerate(self.cache):
                patch = item["img"]
                
                raw_mask = np.zeros_like(patch, dtype=np.uint8)
                for obj in item["objs"]:
                    c = np.array(obj["coordinates"])
                    if len(c) > 0: raw_mask[c[:, 1], c[:, 0]] = 1
                if i == 0: first_intermediates["0. Raw Detected Objects"] = raw_mask * 255

                objs_calc = calculate_values(item["objs"], patch)
                
                filtered_objs = []
                if object_rf is not None and len(objs_calc) > 0:
                    df_features = pd.DataFrame([obj["values"] for obj in objs_calc])
                    
                    if hasattr(object_rf, "feature_names_in_"):
                        df_features_aligned = df_features.reindex(columns=object_rf.feature_names_in_, fill_value=0.0)
                    else:
                        df_features_aligned = df_features
                    
                    class_1_idx = np.where(object_rf.classes_ == 1)[0][0]
                    probs = object_rf.predict_proba(df_features_aligned.values if not hasattr(object_rf, "feature_names_in_") else df_features_aligned)
                    class_1_probs = probs[:, class_1_idx]
                    
                    for idx, obj in enumerate(objs_calc):
                        if class_1_probs[idx] >= agreement_threshold:
                            filtered_objs.append(obj)
                else:
                    filtered_objs = objs_calc

                base_mask = np.zeros_like(patch, dtype=np.uint8)
                for obj in filtered_objs:
                    c = np.array(obj["coordinates"])
                    if len(c) > 0: base_mask[c[:, 1], c[:, 0]] = 1
                if i == 0: first_intermediates["1. RF Filtered Objects"] = base_mask * 255

                d_kernel = params['closing_dilation_kernel']
                if d_kernel > 0:
                    skeleton = skeletonize(chunked_edt_mask(base_mask, radius=d_kernel // 2)).astype(np.uint8)
                else:
                    skeleton = base_mask.copy()
                if i == 0: first_intermediates["2. Thickened Skeleton"] = skeleton.copy() * 255

                min_len = params['closing_min_line_length']
                labels, num_labels = ndi.label(skeleton, structure=np.ones((3, 3), dtype=bool))
                if min_len > 0 and num_labels > 1:
                    areas = np.bincount(labels.ravel())
                    to_remove = set(j for j in range(1, num_labels) if areas[j] < min_len)
                    skeleton[np.isin(labels, list(to_remove))] = 0
                if i == 0: first_intermediates["3. Size Pruned Skeleton"] = skeleton.copy() * 255

                tophat_raw = chunked_black_tophat(patch, kernel_size=12)
                item["skeleton"], item["labels"], item["tophat_raw"] = skeleton, labels, tophat_raw

                img_8u = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if patch.dtype != np.uint8 else patch.copy()
                disp = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
                skel_thick = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                overlay = np.zeros_like(disp)
                overlay[skel_thick == 1] = [0, 255, 0]
                active = np.any(overlay > 0, axis=-1)
                disp[active] = cv2.addWeighted(overlay, 0.6, disp, 0.4, 0)[active]
                cv2.putText(disp, item["name"][:25], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                results.append(disp)

            self.after(0, lambda: self._show_intermediates_popup(first_intermediates, f"Phase 2 Steps: {self.cache[0]['name']}"))
            self.after(0, lambda: self._display_grid_on_canvas(results, "Phase 2: Filtered Base Skeleton"))
            self.after(0, lambda: self.lbl_status.configure(text="Phase 2 complete!", text_color="#2ecc71"))
        threading.Thread(target=task, daemon=True).start()

    def _update_phase_3(self):
        if not self.cache or "skeleton" not in self.cache[0]:
            self.lbl_status.configure(text="Run Phase 2 first!", text_color="red")
            return
        self.lbl_status.configure(text="Running Phase 3 (Routing)...", text_color="orange")
        params = self._get_all_params()

        def task():
            results = []
            for item in self.cache:
                skeleton, labels = item["skeleton"].copy(), item["labels"]
                h, w = skeleton.shape

                bm = int(params["border_margin_px"])
                border_zone = np.zeros((h, w), dtype=bool)
                if bm > 0:
                    border_zone[:bm, :], border_zone[-bm:, :] = True, True
                    border_zone[:, :bm], border_zone[:, -bm:] = True, True

                ep_coords = np.argwhere((skeleton == 1) & (chunked_neighbor_count(skeleton) == 1))
                endpoints = [{'coord': (int(y), int(x)), 'label': labels[y, x], 'near_border': border_zone[y, x], 'connected': False} for y, x in ep_coords]

                # Build the hybrid routing cost map used by the tuner preview.
                tophat_raw = item["tophat_raw"].copy()
                clip_val = float(np.percentile(tophat_raw, params["dijkstra_tophat_clip_percentile"]))
                tophat_norm = np.clip(tophat_raw, 0.0, clip_val)
                tophat_norm = tophat_norm / (np.max(tophat_norm) + 1e-5)
                
                patch = item["img"]
                blurred = cv2.GaussianBlur(patch, (7, 7), 0)
                bmask = item["bmask"]
                valid_ice = blurred[bmask > 0] if (bmask is not None and np.any(bmask > 0)) else blurred.flatten()
                if len(valid_ice) == 0: valid_ice = blurred.flatten()
                
                ice_median = float(np.median(valid_ice[::10])) 
                darkness_factor = float(params.get("dijkstra_darkness_factor", 0.55))
                dynamic_cutoff = ice_median * darkness_factor
                
                darkness_norm = 1.0 - (np.clip(blurred, 0, dynamic_cutoff).astype(np.float32) / (dynamic_cutoff + 1e-5))
                combined_score = np.maximum(tophat_norm, darkness_norm)

                d_mult = float(params["dijkstra_cost_multiplier"])
                cost_map = combined_score * -(d_mult / (np.max(combined_score) + 1e-5)) + (1.0 + d_mult)

                cost_map[skeleton == 1] = 99999.0
                cost_map[(chunked_edt_mask(skeleton, radius=int(params["closing_repulsion_radius"])) == 1) & (skeleton != 1)] += 10000000

                drawn_mask = np.zeros_like(skeleton)
                e2e_paths, tjunc_paths = [], []

                # Route end-to-end endpoint pairs in parallel.
                if len(endpoints) >= 2:
                    tree = cKDTree([ep['coord'] for ep in endpoints])
                    pairs = []
                    for i, ep1 in enumerate(endpoints):
                        if ep1['near_border']: continue
                        _, indices = tree.query(ep1['coord'], k=10, distance_upper_bound=int(params["max_radius"]))
                        for j in np.atleast_1d(indices):
                            if j < len(endpoints) and i < j and not endpoints[j]['near_border']:
                                pairs.append((i, j, ep1, endpoints[j]))
                                
                    def _process_e2e_tuner(pair):
                        i, j, ep1, ep2 = pair
                        try:
                            path, cost = _local_route_through_array_ts(cost_map, ep1['coord'], ep2['coord'], margin=200)
                            if not path_crosses_skeleton(path, skeleton):
                                return {"i": i, "j": j, "path": path, "cost": cost}
                        except ValueError: pass
                        return None

                    # Evaluate candidate routes concurrently.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        res_e2e = list(executor.map(_process_e2e_tuner, pairs))
                    potential_e2e = [r for r in res_e2e if r is not None]
                    
                    # Consider lower-cost routes first when connecting endpoints.
                    potential_e2e.sort(key=lambda x: x["cost"])
                    
                    # Accept non-crossing routes in ascending cost order.
                    for p in potential_e2e:
                        i, j = p["i"], p["j"]
                        if endpoints[i]['connected'] or endpoints[j]['connected']: continue
                        if not path_crosses_skeleton(p["path"], drawn_mask):
                            for y, x in p["path"]: drawn_mask[y, x] = 1
                            endpoints[i]['connected'] = True; endpoints[j]['connected'] = True
                            e2e_paths.append(p["path"])

                # Route candidate T-junction connections.
                tj_pairs = []
                for i, ep in enumerate(endpoints):
                    if ep['connected'] or ep['near_border']: continue
                    targets = _find_t_junc_targets_cone(ep['coord'][0], ep['coord'][1], skeleton, int(params["max_radius"]), int(params["t_junc_vector_length_px"]), params["t_junc_cone_angle_deg"])
                    for t in targets:
                        tj_pairs.append((i, ep, t))
                        
                def _process_tj_tuner(pair):
                    i, ep, t = pair
                    try:
                        path, cost = _local_route_through_array_ts(cost_map, ep['coord'], t, margin=200)
                        if not path_crosses_skeleton(path, skeleton):
                            return {"i": i, "path": path, "cost": cost}
                    except ValueError: pass
                    return None

                # Evaluate candidate routes concurrently.
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    res_tj = list(executor.map(_process_tj_tuner, tj_pairs))
                potential_tjuncs = [r for r in res_tj if r is not None]
                
                # Consider lower-cost routes first when connecting endpoints.
                potential_tjuncs.sort(key=lambda x: x["cost"])
                
                # Accept non-crossing routes in ascending cost order.
                for p in potential_tjuncs:
                    i = p["i"]
                    if endpoints[i]['connected']: continue
                    if not path_crosses_skeleton(p["path"], drawn_mask):
                        for y, x in p["path"]: drawn_mask[y, x] = 1
                        endpoints[i]['connected'] = True
                        tjunc_paths.append(p["path"])

                # Render the accepted routes over the source image.
                img_8u = cv2.normalize(item["img"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if item["img"].dtype != np.uint8 else item["img"].copy()
                disp = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
                overlay = np.zeros_like(disp)

                skel_thick = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                overlay[skel_thick == 1] = [0, 200, 0]

                for path in e2e_paths:
                    pts = np.flip(np.array(path).reshape((-1, 1, 2)), axis=-1).astype(np.int32)
                    cv2.polylines(overlay, [pts], False, (255, 0, 255), 4)
                for path in tjunc_paths:
                    pts = np.flip(np.array(path).reshape((-1, 1, 2)), axis=-1).astype(np.int32)
                    cv2.polylines(overlay, [pts], False, (0, 165, 255), 4)

                active = np.any(overlay > 0, axis=-1)
                disp[active] = cv2.addWeighted(overlay, 0.7, disp, 0.3, 0)[active]
                cv2.putText(disp, item["name"][:25], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                results.append(disp)

            self.after(0, lambda: self._display_grid_on_canvas(results, "Phase 3: Routed Paths (Pink: E2E, Orange: T-Junc)"))
            self.after(0, lambda: self.lbl_status.configure(text="Phase 3 complete! Ready to save.", text_color="#2ecc71"))
        threading.Thread(target=task, daemon=True).start()

    def _save_parameters(self):
        params = self._get_all_params()
        try:
            output_dir = constants_manager.get_config_dir()
            saved_path = constants_manager.save_constants_with_timestamp(params, output_dir)
            self.lbl_status.configure(text=f"Saved: {Path(saved_path).name}", text_color="#2ecc71")
        except Exception as e:
            self.lbl_status.configure(text=f"Error saving: {e}", text_color="red")
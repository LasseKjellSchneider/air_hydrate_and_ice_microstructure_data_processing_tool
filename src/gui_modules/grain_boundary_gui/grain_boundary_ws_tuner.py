import sys
import pandas as pd
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog
import threading
import numpy as np
import tifffile
from PIL import Image
import cv2
import json
import math
import tempfile
import scipy.ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from collections import defaultdict

from core_backend.grain_boundary_segmentation import constants_manager

class GrainBoundaryWatershedTunerFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        self.ws_base = None
        self.params = constants_manager.load_constants()
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
        ctk.CTkLabel(top_bar, text="Phase 4: Watershed Tuner", font=("Arial", 22, "bold")).pack(side="left", padx=15)

        content_frame = ctk.CTkScrollableFrame(self, fg_color=self.bg_color)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        io_box = ctk.CTkFrame(content_frame, fg_color="#261C18", corner_radius=10)
        io_box.pack(fill="x", pady=5, padx=5)

        self.lbl_json = ctk.CTkLabel(io_box, text="No Parameter JSON loaded (Defaults used)", font=("Arial", 13), text_color="gray")
        self.lbl_json.pack(pady=(10, 2))
        ctk.CTkButton(io_box, text="Select Parameter JSON", fg_color=self.btn_color, hover_color=self.btn_hover,
                      command=self._load_json, height=35).pack(pady=3, padx=15, fill="x")

        self.lbl_img = ctk.CTkLabel(io_box, text="No ws_base_mask.tif loaded", font=("Arial", 13), text_color="gray")
        self.lbl_img.pack(pady=(10, 2))
        ctk.CTkButton(io_box, text="Select ws_base_mask.tif", fg_color="#3498db", hover_color="#2980b9",
                      command=self._load_image, height=35).pack(pady=3, padx=15, fill="x")

        self.lbl_status = ctk.CTkLabel(content_frame, text="Status: Ready", text_color="orange", font=("Arial", 14, "bold"))
        self.lbl_status.pack(pady=10)

        slider_box = ctk.CTkFrame(content_frame, fg_color="#261C18", corner_radius=10)
        slider_box.pack(fill="x", pady=10, padx=5)
        ctk.CTkLabel(slider_box, text="Watershed & Merging Parameters", font=("Arial", 16, "bold"), text_color=self.btn_color).pack(pady=10)

        configs = [
            ("ws_scale_factor", "Scale Factor (Downscaling)", 1, 20, 1, self.params.get("ws_scale_factor", 12)),
            ("ws_blur_kernel", "Blur Kernel (Topography)", 1, 15, 2, self.params.get("ws_blur_kernel", 5)),
            ("ws_min_seed_depth", "Min Seed Depth (px to border)", 1.0, 200.0, 1.0, self.params.get("ws_min_seed_depth", 3.0)),
            ("ws_y_intercept", "Base Allowed Dam Length (y-Int)", 0.0, 200.0, 1.0, self.params.get("ws_y_intercept", 5.0)),
            ("ws_slope", "Dam Length Growth (Slope)", 0.0, 2.0, 0.1, self.params.get("ws_slope", 0.5)),
            ("ws_cutoff", "Max Allowed Dam Length (Cutoff)", 50.0, 500.0, 5.0, max(50.0, self.params.get("ws_cutoff", 50.0)))
        ]

        for c in configs:
            self._add_slider(slider_box, *c)
            
        self.chk_topo_var = ctk.BooleanVar(value=False)
        self.chk_topo = ctk.CTkCheckBox(slider_box, text="Show Topography (Distance Map)", variable=self.chk_topo_var, 
                                        fg_color=self.btn_color, hover_color=self.btn_hover, font=("Arial", 14, "bold"))
        self.chk_topo.pack(pady=15, padx=15, anchor="w")

        ctk.CTkButton(content_frame, text="UPDATE PREVIEW", font=("Arial", 16, "bold"),
                      fg_color="#8e44ad", hover_color="#732d91", command=self._run_preview, height=50).pack(pady=15, fill="x", padx=10)

        ctk.CTkButton(content_frame, text="SAVE UPDATED PARAMETERS (JSON)", font=("Arial", 15, "bold"),
                      fg_color="#27ae60", hover_color="#2ecc71", command=self._save_parameters, height=45).pack(pady=5, fill="x", padx=10)

    def _add_slider(self, parent, key, name, v_min, v_max, step, default):
        lbl = ctk.CTkLabel(parent, text=f"{name}: {default}", font=("Arial", 13))
        lbl.pack(anchor="w", padx=15, pady=(5,0))
        sl = ctk.CTkSlider(parent, from_=v_min, to=v_max, number_of_steps=int((v_max - v_min) / step), button_color=self.btn_color)
        sl.set(default)
        sl.pack(fill="x", padx=15, pady=(0, 10))
        
        def update_lbl(val, l=lbl, n=name):
            if step >= 1.0: val = int(val)
            else: val = round(float(val), 2)
            l.configure(text=f"{n}: {val}")
            
        sl.configure(command=update_lbl)
        self.sliders[key] = sl

    def _load_json(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f:
            self.lbl_json.configure(text=Path(f).name, text_color="#2ecc71")
            with open(f, 'r', encoding='utf-8') as file:
                loaded_params = json.load(file)
                self.params.update(loaded_params)
            for key, slider in self.sliders.items():
                if key in self.params:
                    val = self.params[key]
                    if key == "ws_cutoff" and val < 50.0: val = 50.0 
                    slider.set(val)
                    slider._command(val)

    def _load_image(self):
        f = filedialog.askopenfilename(filetypes=[("TIF", "*.tif")])
        if f:
            self.lbl_img.configure(text=f"Loading {Path(f).name}...", text_color="orange")
            self.update_idletasks()
            try:
                with tifffile.TiffFile(f) as tif:
                    img = tif.asarray()
                if img.max() == 1: img = img * 255
                self.ws_base = img.astype(np.uint8)
                self.lbl_img.configure(text=Path(f).name, text_color="#2ecc71")
            except Exception as e:
                self.lbl_img.configure(text=f"Error loading image: {e}", text_color="red")

    def _run_preview(self):
        if self.ws_base is None:
            self.lbl_status.configure(text="Please load ws_base_mask.tif first!", text_color="red")
            return
            
        self.lbl_status.configure(text="Running Watershed Preview...", text_color="orange")
        
        ws_scale = int(self.sliders["ws_scale_factor"].get())
        ws_blur = int(self.sliders["ws_blur_kernel"].get()) | 1 
        ws_min_depth = float(self.sliders["ws_min_seed_depth"].get())
        ws_y_int = float(self.sliders["ws_y_intercept"].get())
        ws_slope = float(self.sliders["ws_slope"].get())
        ws_cutoff = float(self.sliders["ws_cutoff"].get())
        show_topo = self.chk_topo_var.get()

        def task():
            try:
                h, w = self.ws_base.shape
                
                if ws_scale > 1:
                    dilated_base = cv2.dilate(self.ws_base, disk(ws_scale + 1))
                    small_h, small_w = h // ws_scale, w // ws_scale
                    small_mask = cv2.resize(dilated_base, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                else:
                    small_mask = self.ws_base.copy()
                    small_h, small_w = h, w

                dist_map = cv2.distanceTransform(1 - (small_mask // 255), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                dist_map = cv2.GaussianBlur(dist_map, (ws_blur, ws_blur), 0)

                coords = peak_local_max(dist_map, min_distance=20, threshold_abs=ws_min_depth)
                local_max = np.zeros_like(dist_map, dtype=bool)
                if len(coords) > 0:
                    local_max[tuple(coords.T)] = True
                markers, num_markers = ndi.label(local_max, structure=np.ones((3,3)))

                if num_markers == 0:
                    self.controller.after(0, lambda: self.lbl_status.configure(text="No seeds found! Decrease Min Seed Depth.", text_color="red"))
                    return

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
                    
                    if len(u_labels) == 1:
                        dam_segments[(u_labels[0], u_labels[0])].append((y, x))
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
                    
                    if dam_length > thresh:
                        merge_graph.append((l1, l2))
                        
                parent = {i: i for i in range(1, num_markers + 1)}
                def find_parent(i):
                    if parent[i] == i: return i
                    parent[i] = find_parent(parent[i])
                    return parent[i]
                    
                for l1, l2 in merge_graph:
                    root_l1 = find_parent(l1)
                    root_l2 = find_parent(l2)
                    if root_l1 != root_l2:
                        parent[root_l1] = root_l2
                        
                new_labels = np.zeros_like(labels)
                for i in range(1, num_markers + 1):
                    new_labels[labels == i] = find_parent(i)
                    
                for (l1, l2), pixels in dam_segments.items():
                    root_l1 = find_parent(l1)
                    root_l2 = find_parent(l2)
                    if root_l1 == root_l2:
                        py = [p[0] for p in pixels]
                        px = [p[1] for p in pixels]
                        new_labels[py, px] = root_l1

                bounds = find_boundaries(new_labels, mode='inner')
                
                if show_topo:
                    dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp = cv2.applyColorMap(dist_norm, cv2.COLORMAP_MAGMA)
                    disp[small_mask > 0] = [30, 30, 30] 
                else:
                    disp = np.full((small_h, small_w, 3), (40, 40, 40), dtype=np.uint8)
                    disp[small_mask > 0] = [255, 255, 255]
                
                disp[bounds] = [0, 0, 255]
                
                seed_radius = max(1, 10 // ws_scale)
                for c in coords:
                    cv2.circle(disp, (c[1], c[0]), seed_radius, (0, 255, 0), -1)

                temp_path = str(Path(tempfile.gettempdir()) / "ws_tuner_preview.png")
                cv2.imwrite(temp_path, disp)
                
                topo_text = " (Topography Mode)" if show_topo else ""
                self.controller.after(0, lambda: self.controller.show_image_plot(temp_path, f"Watershed Preview{topo_text} - Red: Dams, Green: Seeds"))
                self.controller.after(0, lambda: self.lbl_status.configure(text="Preview updated!", text_color="#2ecc71"))
                
            except Exception as e:
                self.controller.after(0, lambda msg=str(e): self.lbl_status.configure(text=f"Error: {msg}", text_color="red"))

        threading.Thread(target=task, daemon=True).start()

    def _save_parameters(self):
        self.params["ws_scale_factor"] = int(self.sliders["ws_scale_factor"].get())
        self.params["ws_blur_kernel"] = int(self.sliders["ws_blur_kernel"].get()) | 1
        self.params["ws_min_seed_depth"] = float(self.sliders["ws_min_seed_depth"].get())
        self.params["ws_y_intercept"] = float(self.sliders["ws_y_intercept"].get())
        self.params["ws_slope"] = float(self.sliders["ws_slope"].get())
        self.params["ws_cutoff"] = float(self.sliders["ws_cutoff"].get())
        
        try:
            output_dir = constants_manager.get_config_dir()
            saved_path = constants_manager.save_constants_with_timestamp(self.params, output_dir)
            self.lbl_status.configure(text=f"Saved Master-JSON: {Path(saved_path).name}", text_color="#2ecc71")
        except Exception as e:
            self.lbl_status.configure(text=f"Error saving: {e}", text_color="red")
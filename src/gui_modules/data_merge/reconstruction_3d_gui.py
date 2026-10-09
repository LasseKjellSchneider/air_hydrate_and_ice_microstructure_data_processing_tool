import customtkinter as ctk
import os
import math
import json
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import traceback 
import random

# Import backend logic
from core_backend.data_merge.reconstruction_3d_core import extract_vector_contours, build_3d_meshes, assign_hydrates_and_export, extract_fa_data

def um_to_mm(val, pos):
    return f"{val/1000.0:g}"

# Convert NumPy values so mesh geometry can be serialized as JSON.
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class ModelReconstructionFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master, fg_color="transparent")
        self.controller = controller
        
        self.h5_path = None
        self.extracted_contours = None
        self.extracted_types = None 
        self.extracted_fa_data = None  
        self.generated_meshes = None 
        self.ax = None
        self.canvas_fig = None
        
        self.current_single_grain = None
        self.largest_grain_index = 0
        self.current_grain_rank_info = ""
        self.is_vector_preview_active = False  
        
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header_lbl = ctk.CTkLabel(self, text="3D Reconstruction & Hydrate Assignment", font=("Arial", 24, "bold"))
        header_lbl.grid(row=0, column=0, pady=(0, 10), sticky="w", padx=10)

        self.controls = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.controls.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # File selection
        ctk.CTkLabel(self.controls, text="Select Master HDF5 File", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        file_frame = ctk.CTkFrame(self.controls, fg_color="#1F2E2C")
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_h5_path = ctk.CTkLabel(file_frame, text="No file selected...", text_color="gray", anchor="w")
        self.lbl_h5_path.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        
        btn_browse = ctk.CTkButton(file_frame, text="Browse", width=80, command=self.browse_h5)
        btn_browse.pack(side="right", padx=10)
        
        # 3D meshing parameters
        ctk.CTkLabel(self.controls, text="Skeleton Graph Matching (Area-Based)", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        mesh_frame = ctk.CTkFrame(self.controls, fg_color="#1B2433")
        mesh_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_overlap = ctk.CTkLabel(mesh_frame, text="Min Overlap (1:1 Match): 25%", anchor="w")
        self.lbl_overlap.pack(padx=10, pady=(10, 0), anchor="w")
        self.slider_overlap = ctk.CTkSlider(mesh_frame, from_=0.01, to=1.0, number_of_steps=99, command=lambda v: self.lbl_overlap.configure(text=f"Min Overlap (1:1 Match): {int(v*100)}%"))
        self.slider_overlap.set(0.25)
        self.slider_overlap.pack(fill="x", padx=10, pady=(5, 5))

        self.lbl_split = ctk.CTkLabel(mesh_frame, text="Split Overlap (Y-Connection): 70%", anchor="w")
        self.lbl_split.pack(padx=10, pady=(10, 0), anchor="w")
        self.slider_split = ctk.CTkSlider(mesh_frame, from_=0.01, to=1.0, number_of_steps=99, command=lambda v: self.lbl_split.configure(text=f"Split Overlap (Y-Connection): {int(v*100)}%"))
        self.slider_split.set(0.70)
        self.slider_split.pack(fill="x", padx=10, pady=(5, 5))
        
        self.lbl_apex = ctk.CTkLabel(mesh_frame, text="Apex Z-Ratio (Grain Death): 0.50 (Middle)", anchor="w")
        self.lbl_apex.pack(padx=10, pady=(10, 0), anchor="w")
        self.slider_apex = ctk.CTkSlider(mesh_frame, from_=0.1, to=0.9, number_of_steps=80, command=lambda v: self.lbl_apex.configure(text=f"Apex Z-Ratio: {v:.2f}"))
        self.slider_apex.set(0.5)
        self.slider_apex.pack(fill="x", padx=10, pady=(5, 5))

        self.lbl_res = ctk.CTkLabel(mesh_frame, text="Mesh Resolution (Triangles/Edge): 10", anchor="w")
        self.lbl_res.pack(padx=10, pady=(10, 0), anchor="w")
        self.slider_res = ctk.CTkSlider(mesh_frame, from_=2, to=50, number_of_steps=48, command=lambda v: self.lbl_res.configure(text=f"Mesh Resolution: {int(v)}"))
        self.slider_res.set(10)
        self.slider_res.pack(fill="x", padx=10, pady=(5, 10))

        self.btn_mesh = ctk.CTkButton(mesh_frame, text="Calculate Skeleton Graph", height=35, fg_color="#345B94", hover_color="#20395D", command=self.run_meshing)
        self.btn_mesh.pack(fill="x", padx=10, pady=10)

        # Hydrate assignment
        ctk.CTkLabel(self.controls, text="Hydrate Point-in-Polygon", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        assign_frame = ctk.CTkFrame(self.controls, fg_color="#1B2433")
        assign_frame.pack(fill="x", padx=10, pady=5)

        self.btn_assign = ctk.CTkButton(assign_frame, text="Assign Hydrates & Save JSONs", height=35, fg_color="#345B94", hover_color="#20395D", state="disabled", command=self.run_hydrate_assignment)
        self.btn_assign.pack(fill="x", padx=10, pady=10)

        # Grain inspection controls
        self.lbl_inspection = ctk.CTkLabel(self.controls, text="Grain Inspection", font=("Arial", 16, "bold"))
        self.grain_inspection_frame = ctk.CTkFrame(self.controls, fg_color="#2A2A2A")
        
        self.btn_random_grain = ctk.CTkButton(self.grain_inspection_frame, text="Inspect Random Global Grain", command=self.show_random_grain, fg_color="#F39C12", hover_color="#D68910")
        self.btn_random_grain.pack(fill="x", padx=10, pady=(10, 5))
        
        self.btn_largest_grain = ctk.CTkButton(self.grain_inspection_frame, text="Inspect Largest Grains (Descending)", command=self.show_largest_grain, fg_color="#8E44AD", hover_color="#732D91")
        self.btn_largest_grain.pack(fill="x", padx=10, pady=(0, 5))
        
        self.btn_reset_view = ctk.CTkButton(self.grain_inspection_frame, text="Reset to Full View", command=self.reset_view, fg_color="#7F8C8D", hover_color="#606B6C")
        self.btn_reset_view.pack(fill="x", padx=10, pady=(0, 10))

        # Visualization settings
        self.lbl_vis_settings = ctk.CTkLabel(self.controls, text="Visualization Settings", font=("Arial", 16, "bold"))
        self.vis_frame = ctk.CTkFrame(self.controls, fg_color="#16212B")
        
        # Vector preview settings
        self.vector_vis_frame = ctk.CTkFrame(self.vis_frame, fg_color="transparent")
        
        self.var_show_2d_grains = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.vector_vis_frame, text="Show 2D Grains (Orange Wireframe)", variable=self.var_show_2d_grains).pack(anchor="w", pady=5)
        
        self.var_show_fa_rgb = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.vector_vis_frame, text="Show FA Planes (RGB Azimuth/Dip)", variable=self.var_show_fa_rgb).pack(anchor="w", pady=5)
        
        self.var_show_fa_diff = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.vector_vis_frame, text="Show FA Misorientation Planes", variable=self.var_show_fa_diff).pack(anchor="w", pady=5)
        
        # 3D mesh settings
        self.mesh_vis_frame = ctk.CTkFrame(self.vis_frame, fg_color="transparent")
        
        self.lbl_border_filter = ctk.CTkLabel(self.mesh_vis_frame, text="Max Rand-Ebenen (Border Touches): Alle", anchor="w")
        self.lbl_border_filter.pack(anchor="w", pady=(5, 0))
        self.slider_border = ctk.CTkSlider(self.mesh_vis_frame, from_=0, to=4, number_of_steps=4, command=self.update_border_label_only)
        self.slider_border.set(4)
        self.slider_border.pack(fill="x", pady=(5, 15))
        
        self.var_show_clean = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.mesh_vis_frame, text="Show Normal 1:1 Matches (Blue)", variable=self.var_show_clean, fg_color="#3498DB", hover_color="#2980B9").pack(anchor="w", pady=2)
        
        self.var_show_split = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.mesh_vis_frame, text="Show Y-Splits (Pink)", variable=self.var_show_split, fg_color="#FF69B4", hover_color="#FF1493").pack(anchor="w", pady=2)
        
        self.var_show_unassigned = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.mesh_vis_frame, text="Show Unassigned (Transparent Red)", variable=self.var_show_unassigned, fg_color="#E74C3C", hover_color="#C0392B").pack(anchor="w", pady=2)
        
        # Controls shared by both visualization modes
        self.lbl_zscale = ctk.CTkLabel(self.vis_frame, text="Z-Axis Stretch (Detail Level): 1.0x", anchor="w")
        self.slider_zscale = ctk.CTkSlider(self.vis_frame, from_=1.0, to=40.0, number_of_steps=390, command=self.update_z_scale_label_only)
        self.slider_zscale.set(1.0)
        
        self.btn_update_plot = ctk.CTkButton(self.vis_frame, text="🔄 Update Plot", font=("Arial", 14, "bold"), height=40, fg_color="#27AE60", hover_color="#229954", command=self.apply_visualization_settings)

    # Configure controls for the active visualization mode.

    def _update_ui_layout(self):
        """ Steuert, welche Einstellungsboxen wann sichtbar sind. """
        self.lbl_inspection.pack_forget()
        self.grain_inspection_frame.pack_forget()
        self.lbl_vis_settings.pack_forget()
        self.vis_frame.pack_forget()
        self.vector_vis_frame.pack_forget()
        self.mesh_vis_frame.pack_forget()
        self.lbl_zscale.pack_forget()
        self.slider_zscale.pack_forget()
        self.btn_update_plot.pack_forget()
        
        if self.is_vector_preview_active:
            # Show vector-preview controls.
            self.lbl_vis_settings.pack(anchor="w", padx=10, pady=(20, 5))
            self.vis_frame.pack(fill="x", padx=10, pady=5)
            self.vector_vis_frame.pack(fill="x", padx=10, pady=5)
            
            # Keep the shared scale and update controls below the mode-specific settings.
            self.lbl_zscale.pack(anchor="w", padx=10, pady=(15, 0))
            self.slider_zscale.pack(fill="x", padx=10, pady=(5, 10))
            self.btn_update_plot.pack(fill="x", padx=10, pady=10)
            
        elif self.generated_meshes:
            # Show mesh inspection and visualization controls.
            self.lbl_inspection.pack(anchor="w", padx=10, pady=(20, 5))
            self.grain_inspection_frame.pack(fill="x", padx=10, pady=5)
            
            self.lbl_vis_settings.pack(anchor="w", padx=10, pady=(20, 5))
            self.vis_frame.pack(fill="x", padx=10, pady=5)
            self.mesh_vis_frame.pack(fill="x", padx=10, pady=5)
            
            # Keep the shared scale and update controls below the mode-specific settings.
            self.lbl_zscale.pack(anchor="w", padx=10, pady=(15, 0))
            self.slider_zscale.pack(fill="x", padx=10, pady=(5, 10))
            self.btn_update_plot.pack(fill="x", padx=10, pady=10)

    # UI callbacks

    def update_border_label_only(self, val):
        v = int(val)
        if v == 4:
            self.lbl_border_filter.configure(text="Max Rand-Ebenen (Border Touches): Alle")
        else:
            self.lbl_border_filter.configure(text=f"Max Rand-Ebenen (Border Touches): {v}")

    def update_z_scale_label_only(self, val):
        self.lbl_zscale.configure(text=f"Z-Axis Stretch (Detail Level): {val:.1f}x")

    def apply_visualization_settings(self):
        """ Wird nur aufgerufen, wenn der User den Update-Button klickt """
        if getattr(self, 'is_vector_preview_active', False) and self.extracted_contours:
            self.refresh_vector_plot()
        elif self.generated_meshes:
            self.refresh_mesh_plot()

    def browse_h5(self):
        path = filedialog.askopenfilename(title="Select master.h5", filetypes=[("HDF5 Files", "*.h5")])
        if path:
            self.h5_path = path
            self.lbl_h5_path.configure(text=path, text_color="white")
            # Defer extraction until the file dialog has closed.
            self.after(100, self.run_vector_extraction)

    def _safe_clear_plot_container(self):
        internal_canvas = getattr(self.controller.image_plot_container, "_canvas", None)
        for child in list(self.controller.image_plot_container.winfo_children()):
            if child == getattr(self.controller, "canvas", None) or child == internal_canvas:
                continue
            try:
                child.destroy()
            except Exception:
                pass

    def run_vector_extraction(self):
        if not self.h5_path: return
        self.controller._clear_right_container()
        lbl_loading = ctk.CTkLabel(self.controller.image_plot_container, text=f"Loading vector polygons and FA Data...\nThis is fast!", font=("Arial", 20))
        lbl_loading.pack(expand=True)
        self.controller.update() 
        try:
            self.extracted_contours, self.extracted_types = extract_vector_contours(self.h5_path)
            self.extracted_fa_data = extract_fa_data(self.h5_path)
            self.is_vector_preview_active = True
        except Exception as e:
            traceback.print_exc()
            lbl_loading.configure(text=f"Error loading vectors:\n{e}", text_color="red")
            return
        lbl_loading.destroy()
        
        self._update_ui_layout()
        self.plot_3d_vectors(self.extracted_contours)

    def run_meshing(self):
        if not self.extracted_contours:
            self.lbl_overlap.configure(text="Error: Extract vectors first!", text_color="red")
            return
            
        min_overlap = float(self.slider_overlap.get())
        split_overlap = float(self.slider_split.get())
        mesh_res = int(self.slider_res.get())
        apex_ratio = float(self.slider_apex.get()) 
        
        self.controller._clear_right_container()
        lbl_loading = ctk.CTkLabel(self.controller.image_plot_container, text="Running Skeleton Graph Extraction...\nCalculating Assignments.", font=("Arial", 20))
        lbl_loading.pack(expand=True)
        self.controller.update()
        
        try:
            self.generated_meshes = build_3d_meshes(
                contours_by_z=self.extracted_contours,
                types_by_z=self.extracted_types,
                min_overlap=min_overlap,
                split_overlap=split_overlap,
                mesh_resolution=mesh_res,
                apex_ratio=apex_ratio 
            )
            self.is_vector_preview_active = False
            self.reset_inspection_state()
        except Exception as e:
            traceback.print_exc()  
            lbl_loading.configure(text=f"Error during Meshing:\n{e}", text_color="red")
            return
            
        lbl_loading.destroy()
        self.btn_assign.configure(state="normal")
        
        self._update_ui_layout()
        self.refresh_mesh_plot()

    def run_hydrate_assignment(self):
        if not self.generated_meshes or not self.h5_path:
            return
            
        base_dir = os.path.dirname(self.h5_path)
        base_name = os.path.splitext(os.path.basename(self.h5_path))[0]
        
        out_json = os.path.join(base_dir, f"{base_name}_3d_hydrates.json")
        out_mesh_json = os.path.join(base_dir, "3d_mesh_geometry.json")
        
        self.controller._clear_right_container()
        lbl_loading = ctk.CTkLabel(
            self.controller.image_plot_container, 
            text="Assigning Hydrates and Exporting JSONs...\n(This might take a moment)", 
            font=("Arial", 20)
        )
        lbl_loading.pack(expand=True)
        self.controller.update()
        
        try:
            # Assign hydrates to reconstructed grains and export the assignment data.
            assign_hydrates_and_export(self.generated_meshes, self.h5_path, out_json)
            
            # Export the reconstructed mesh geometry for dashboard inspection.
            with open(out_mesh_json, 'w', encoding='utf-8') as f:
                json.dump(self.generated_meshes, f, cls=NumpyEncoder, indent=2)
                
            success_msg = f"Success! Export Complete.\n\nSaved Hydrates to:\n{os.path.basename(out_json)}\nSaved Mesh to:\n{os.path.basename(out_mesh_json)}"
            lbl_loading.configure(text=success_msg, text_color="#2ECC71", font=("Arial", 18, "bold"))
            
        except Exception as e:
            traceback.print_exc()
            lbl_loading.configure(text=f"Error during Assignment/Export:\n{e}", text_color="red")

    # Grain inspection

    def reset_inspection_state(self):
        self.current_single_grain = None
        self.largest_grain_index = 0
        self.current_grain_rank_info = ""

    def reset_view(self):
        self.reset_inspection_state()
        self.refresh_mesh_plot()

    def show_random_grain(self):
        if not self.generated_meshes: return
        
        max_border = int(self.slider_border.get())
        unique_grains = list(set(
            m['meta']['global_grain_id'] for m in self.generated_meshes 
            if 'meta' in m and 'global_grain_id' in m['meta'] and (max_border == 4 or m['meta'].get('border_touch_count', 0) <= max_border)
        ))
        if not unique_grains: return
        
        self.reset_inspection_state()
        self.current_single_grain = random.choice(unique_grains)
        self.refresh_mesh_plot()

    def show_largest_grain(self):
        if not self.generated_meshes: return
        
        max_border = int(self.slider_border.get())
        grain_volumes = {}
        for m in self.generated_meshes:
            if 'meta' in m and 'global_grain_id' in m['meta']:
                border_count = m['meta'].get('border_touch_count', 0)
                if max_border < 4 and border_count > max_border:
                    continue
                    
                gid = m['meta']['global_grain_id']
                vol = m['meta'].get('volume_um3', 0.0)
                grain_volumes[gid] = grain_volumes.get(gid, 0.0) + vol
                
        if not grain_volumes: return
        
        sorted_grains = sorted(grain_volumes.items(), key=lambda x: x[1], reverse=True)
        
        if self.largest_grain_index >= len(sorted_grains):
            self.largest_grain_index = 0
            
        selected_gid, selected_vol = sorted_grains[self.largest_grain_index]
        self.current_single_grain = selected_gid
        
        rank = self.largest_grain_index + 1
        vol_mm3 = selected_vol / 1e9  
        self.current_grain_rank_info = f" (Rank {rank} - Vol: {vol_mm3:.3f} mm³)"
        
        self.largest_grain_index += 1
        self.refresh_mesh_plot()

    def refresh_vector_plot(self):
        if getattr(self, 'is_vector_preview_active', False) and self.extracted_contours:
            self.plot_3d_vectors(self.extracted_contours)

    def refresh_mesh_plot(self):
        if self.generated_meshes and not self.is_vector_preview_active:
            self.plot_3d_meshes(self.generated_meshes)

    # 3D plotting and scaling

    def apply_aspect_ratio(self):
        if not self.ax: return
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range) if max(x_range, y_range) > 0 else 1.0
        z_scale = float(self.slider_zscale.get())
        
        self.ax.set_box_aspect([x_range / max_range, y_range / max_range, (z_range / max_range) * z_scale])
        if self.canvas_fig: self.canvas_fig.draw_idle()

    def format_axes(self):
        formatter = FuncFormatter(um_to_mm)
        self.ax.xaxis.set_major_formatter(formatter)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.zaxis.set_major_formatter(formatter)
        
        self.ax.set_xlabel("X [mm] -> top", color='white', weight='bold')
        self.ax.set_ylabel("Y [mm]", color='white', weight='bold')
        self.ax.set_zlabel("Z [mm]", color='white', weight='bold')

    def get_title_name(self):
        if not self.h5_path:
            return "3D Preview"
        bname = os.path.basename(self.h5_path)
        bname = bname.replace("_master.h5", "").replace("_master", "").replace(".h5", "")
        return bname

    def plot_3d_vectors(self, contours_by_z):
        if not contours_by_z: return
            
        self._safe_clear_plot_container()
        fig = Figure(figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor('#1A1A1A')
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1A1A1A')
        
        self.ax.set_title(self.get_title_name(), color='white', weight='bold', fontsize=16)
        
        self.ax.tick_params(colors='white')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#333333')
        self.ax.yaxis.pane.set_edgecolor('#333333')
        self.ax.zaxis.pane.set_edgecolor('#333333')
        self.ax.grid(color='#333333')

        all_x, all_y, all_z = [], [], []
        
        if self.var_show_2d_grains.get():
            for z_um, contours in contours_by_z.items():
                for cnt in contours:
                    x = cnt[:, 0]
                    y = cnt[:, 1]
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    z_arr = np.full_like(x, z_um)
                    self.ax.plot(x, y, z_arr, color='#D67040', linewidth=1.0, alpha=0.8)
                    all_x.extend(x)
                    all_y.extend(y)
                    all_z.append(z_um)
                
        if self.extracted_fa_data and self.var_show_fa_rgb.get():
            fa_polygons_3d = []
            fa_colors = []
            
            fa_z_levels = sorted(self.extracted_fa_data.keys())
            pole_figure_data = {z: {'theta': [], 'r': [], 'color': []} for z in fa_z_levels}
            
            for z_um in fa_z_levels:
                fa_list = self.extracted_fa_data[z_um]
                for fa_item in fa_list:
                    poly = fa_item['poly']
                    if poly.is_empty: continue
                    
                    c_vec = fa_item['c_axis']
                    cx, cy, cz = c_vec['x'], c_vec['y'], c_vec['z']
                    norm = math.sqrt(cx**2 + cy**2 + cz**2)
                    
                    if norm > 0:
                        cx, cy, cz = cx/norm, cy/norm, cz/norm
                        
                        if cx < 0:
                            cx, cy, cz = -cx, -cy, -cz
                            
                        theta = math.atan2(cz, cy)
                        angle_zenith = math.degrees(math.acos(min(1.0, cx)))
                        
                        if 0 <= theta < math.pi/2:
                            w = theta / (math.pi/2)
                            r, g, b = 1 - w, 0.0, w
                        elif math.pi/2 <= theta <= math.pi:
                            w = (theta - math.pi/2) / (math.pi/2)
                            r, g, b = 0.0, w, 1 - w
                        elif -math.pi <= theta < -math.pi/2:
                            w = (theta + math.pi) / (math.pi/2)
                            r, g, b = w, 1.0, 0.0
                        else:
                            w = (theta + math.pi/2) / (math.pi/2)
                            r, g, b = 1.0, 1 - w, 0.0
                            
                        if angle_zenith >= 30.0:
                            factor = 0.0
                        else:
                            factor = 1.0 - (angle_zenith / 30.0)
                            
                        r = r * (1 - factor) + factor
                        g = g * (1 - factor) + factor
                        b = b * (1 - factor) + factor
                        
                        face_color = (r, g, b, 0.7)
                        
                        pole_figure_data[z_um]['theta'].append(theta)
                        pole_figure_data[z_um]['r'].append(angle_zenith)
                        pole_figure_data[z_um]['color'].append((r, g, b, 0.9))
                    else:
                        face_color = (0.5, 0.5, 0.5, 0.7)
                        
                    def extract_faces(p):
                        coords = np.array(p.exterior.coords)
                        verts = [[pt[0], pt[1], z_um] for pt in coords]
                        fa_polygons_3d.append(verts)
                        fa_colors.append(face_color)
                        all_x.extend([pt[0] for pt in coords])
                        all_y.extend([pt[1] for pt in coords])
                        all_z.append(z_um)
                        
                    if poly.geom_type == 'Polygon':
                        extract_faces(poly)
                    elif poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                        for geom in getattr(poly, 'geoms', []):
                            if geom.geom_type == 'Polygon':
                                extract_faces(geom)
                                
            if fa_polygons_3d:
                collection = Poly3DCollection(fa_polygons_3d, linewidths=0.5, edgecolors='white')
                collection.set_facecolors(fa_colors)
                self.ax.add_collection3d(collection)
                
            self.ax_inset = fig.add_axes([0.80, 0.75, 0.15, 0.15])
            x_cw = np.linspace(-1, 1, 150)
            y_cw = np.linspace(-1, 1, 150)
            X_cw, Y_cw = np.meshgrid(x_cw, y_cw)
            self.R_cw = np.sqrt(X_cw**2 + Y_cw**2)
            self.Angle_cw_base = np.arctan2(Y_cw, X_cw) 
            
            self.cw_im = self.ax_inset.imshow(np.zeros((150, 150, 3)), origin='lower', extent=[-1, 1, -1, 1])
            self.ax_inset.axis('off')
            self.ax_inset.set_title("Y-Z Projection", color='white', fontsize=10, pad=5, weight='bold')

            if fa_z_levels:
                y_positions = [0.55, 0.35, 0.15]
                for i, z_um in enumerate(fa_z_levels[:3]):
                    ax_pf = fig.add_axes([0.80, y_positions[i], 0.15, 0.15], polar=True)
                    ax_pf.set_facecolor('#1A1A1A')
                    ax_pf.tick_params(colors='white', labelsize=6)
                    ax_pf.grid(color='#444444', linestyle='--', linewidth=0.5)
                    ax_pf.set_title(f"FA Plane {i+1} (Polfigur)", color='white', fontsize=9, pad=5, weight='bold')
                    ax_pf.set_rlim(0, 90)
                    ax_pf.set_rticks([30, 60, 90])
                    ax_pf.set_yticklabels(['30°', '60°', ''])
                    ax_pf.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
                    ax_pf.set_xticklabels(['Y+', 'Z+', 'Y-', 'Z-'])
                    
                    pdata = pole_figure_data[z_um]
                    if pdata['theta']:
                        ax_pf.scatter(pdata['theta'], pdata['r'], c=pdata['color'], s=3, alpha=0.9, edgecolors='none')

            def update_colorwheel(event=None):
                if not hasattr(self, 'ax') or not hasattr(self, 'cw_im'): return
                if getattr(self, '_updating_cw', False): return
                
                current_azim = self.ax.azim
                current_elev = self.ax.elev
                
                if current_azim != getattr(self, 'last_azim', None) or current_elev != getattr(self, 'last_elev', None):
                    self._updating_cw = True
                    self.last_azim = current_azim
                    self.last_elev = current_elev
                    
                    proj_matrix = self.ax.get_proj()
                    x0, y0, _ = proj3d.proj_transform(0, 0, 0, proj_matrix)
                    xY, yY, _ = proj3d.proj_transform(0, 1, 0, proj_matrix)
                    
                    angle_y_screen = np.arctan2(yY - y0, xY - x0)
                    
                    rot_angle = self.Angle_cw_base - angle_y_screen
                    theta = np.arctan2(np.sin(rot_angle), np.cos(rot_angle))
                    
                    rgb_new = np.zeros((*theta.shape, 3))
                    
                    m1 = (theta >= 0) & (theta < np.pi/2)
                    m2 = (theta >= np.pi/2) & (theta <= np.pi)
                    m3 = (theta >= -np.pi) & (theta < -np.pi/2)
                    m4 = (theta >= -np.pi/2) & (theta < 0)
                    
                    rgb_new[m1, 0] = 1 - (theta[m1] / (np.pi/2))
                    rgb_new[m1, 1] = 0.0
                    rgb_new[m1, 2] = (theta[m1] / (np.pi/2))
                    
                    rgb_new[m2, 0] = 0.0
                    rgb_new[m2, 1] = (theta[m2] - np.pi/2) / (np.pi/2)
                    rgb_new[m2, 2] = 1 - rgb_new[m2, 1]
                    
                    rgb_new[m3, 0] = (theta[m3] + np.pi) / (np.pi/2)
                    rgb_new[m3, 1] = 1.0
                    rgb_new[m3, 2] = 0.0
                    
                    rgb_new[m4, 0] = 1.0
                    rgb_new[m4, 1] = 1 - ((theta[m4] + np.pi/2) / (np.pi/2))
                    rgb_new[m4, 2] = 0.0
                    
                    R = np.clip(self.R_cw, 0, 1) 
                    factor = np.zeros_like(R)
                    mask_white = R < (30.0 / 90.0)
                    factor[mask_white] = 1.0 - (R[mask_white] / (30.0 / 90.0))
                    
                    rgb_new[:,:,0] = rgb_new[:,:,0] * R + factor
                    rgb_new[:,:,1] = rgb_new[:,:,1] * R + factor
                    rgb_new[:,:,2] = rgb_new[:,:,2] * R + factor
                    
                    bg_color = mcolors.to_rgb('#1A1A1A')
                    rgb_new[self.R_cw > 1.0] = bg_color
                    
                    self.cw_im.set_data(rgb_new)
                    self._updating_cw = False

            update_colorwheel()
            fig.canvas.mpl_connect('draw_event', update_colorwheel)

        if self.extracted_fa_data and self.var_show_fa_diff.get():
            ax_diff_cb = fig.add_axes([0.05, 0.85, 0.2, 0.02])
            cmap_diff = mcolors.LinearSegmentedColormap.from_list("green_red", [(0, 1, 0), (1, 0, 0)])
            norm_diff = mcolors.Normalize(vmin=0, vmax=90)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_diff, cmap=cmap_diff), cax=ax_diff_cb, orientation='horizontal')
            cb.set_label('Misorientation [°]', color='white', fontsize=10, weight='bold')
            cb.outline.set_edgecolor('white')
            ax_diff_cb.tick_params(colors='white', labelsize=9)
            
            fa_diff_polygons_3d = []
            fa_diff_colors = []
            fa_z_levels = sorted(self.extracted_fa_data.keys())
            
            for i in range(len(fa_z_levels) - 1):
                z1 = fa_z_levels[i]
                z2 = fa_z_levels[i+1]
                z_mid = (z1 + z2) / 2.0
                
                list1 = self.extracted_fa_data[z1]
                list2 = self.extracted_fa_data[z2]
                
                for item1 in list1:
                    p1 = item1['poly']
                    b1 = p1.bounds
                    if not b1: continue
                    
                    c1 = item1['c_axis']
                    v1 = np.array([c1['x'], c1['y'], c1['z']])
                    n1 = np.linalg.norm(v1)
                    if n1 == 0: continue
                    v1 = v1 / n1
                    
                    for item2 in list2:
                        p2 = item2['poly']
                        b2 = p2.bounds
                        if not b2: continue
                        
                        if b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]:
                            continue
                            
                        if p1.intersects(p2):
                            try:
                                ix_poly = p1.intersection(p2)
                                if ix_poly.is_empty: continue
                                
                                c2 = item2['c_axis']
                                v2 = np.array([c2['x'], c2['y'], c2['z']])
                                n2 = np.linalg.norm(v2)
                                if n2 == 0: continue
                                v2 = v2 / n2
                                
                                dot = abs(np.dot(v1, v2))
                                dot = max(0.0, min(1.0, dot)) 
                                angle_deg = math.degrees(math.acos(dot))
                                
                                c_val = min(1.0, angle_deg / 90.0)
                                r_col, g_col, b_col, _ = cmap_diff(c_val)
                                face_color = (r_col, g_col, b_col, 0.85)
                                
                                def add_diff_faces(p_ix):
                                    coords = np.array(p_ix.exterior.coords)
                                    verts = [[pt[0], pt[1], z_mid] for pt in coords]
                                    fa_diff_polygons_3d.append(verts)
                                    fa_diff_colors.append(face_color)
                                    all_x.extend([pt[0] for pt in coords])
                                    all_y.extend([pt[1] for pt in coords])
                                    all_z.append(z_mid)
                                    
                                if ix_poly.geom_type == 'Polygon':
                                    add_diff_faces(ix_poly)
                                elif ix_poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                                    for geom in getattr(ix_poly, 'geoms', []):
                                        if geom.geom_type == 'Polygon':
                                            add_diff_faces(geom)
                            except Exception:
                                pass
                                
            if fa_diff_polygons_3d:
                collection = Poly3DCollection(fa_diff_polygons_3d, linewidths=0.2, edgecolors='#ffffff')
                collection.set_facecolors(fa_diff_colors)
                self.ax.add_collection3d(collection)

        if not all_x:
            all_x, all_y, all_z = [0, 1], [0, 1], [0, 6000]

        if all_x:
            self.ax.set_xlim3d(np.min(all_x), np.max(all_x))
            self.ax.set_ylim3d(np.min(all_y), np.max(all_y))
            self.ax.set_zlim3d(np.min(all_z)-10, np.max(all_z)+10)

        self.ax.invert_zaxis()
        self.apply_aspect_ratio()
        self.ax.view_init(elev=20, azim=-45)
        self.format_axes()

        fig.tight_layout()
        self.canvas_fig = FigureCanvasTkAgg(fig, master=self.controller.image_plot_container)
        self.canvas_fig.draw()
        canvas_widget = self.canvas_fig.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_fig, self.controller.image_plot_container)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.controller.opt_canvas_widget = canvas_widget

    def plot_3d_meshes(self, meshes):
        if not meshes: return
        self._safe_clear_plot_container()
            
        fig = Figure(figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor('#1A1A1A')
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1A1A1A')
        
        base_title = self.get_title_name()
        if self.current_single_grain:
            self.ax.set_title(f"{base_title}\nInspecting: {self.current_single_grain}{self.current_grain_rank_info}", color='#F39C12', weight='bold')
        else:
            self.ax.set_title(base_title, color='white', weight='bold', fontsize=16)
        
        self.ax.tick_params(colors='white')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#333333')
        self.ax.yaxis.pane.set_edgecolor('#333333')
        self.ax.zaxis.pane.set_edgecolor('#333333')

        all_x, all_y, all_z = [], [], []
        if self.extracted_contours:
            for z_um, contours in self.extracted_contours.items():
                for cnt in contours:
                    x = cnt[:, 0]
                    y = cnt[:, 1]
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    z_arr = np.full_like(x, z_um)
                    self.ax.plot(x, y, z_arr, color='#7F8C8D', linewidth=0.5, alpha=0.4)
                    all_x.extend(x)
                    all_y.extend(y)
                    all_z.append(z_um)

        colors = {
            'clean': (0.20, 0.60, 0.86, 0.9),      
            'split': (1.00, 0.41, 0.71, 0.9),      
            'unassigned': (0.91, 0.30, 0.24, 0.2)  
        }

        active_types = []
        if self.var_show_clean.get(): active_types.append('clean')
        if self.var_show_split.get(): active_types.append('split')
        if self.var_show_unassigned.get(): active_types.append('unassigned')

        max_border = int(self.slider_border.get())

        polygons = []
        face_colors = []
        
        for mesh in meshes:
            m_type = mesh['type']
            
            border_count = mesh['meta'].get('border_touch_count', 0)
            if max_border < 4 and border_count > max_border:
                continue 
            
            if self.current_single_grain:
                if 'meta' not in mesh or mesh['meta'].get('global_grain_id') != self.current_single_grain:
                    continue
            else:
                if m_type not in active_types:
                    continue
                
            verts = mesh['vertices']
            faces = mesh['faces']
            color = colors.get(m_type, (1, 1, 1, 1))
            
            if self.current_single_grain:
                color = (color[0], color[1], color[2], 0.95)
            
            for face in faces:
                triangle = [verts[face[0]], verts[face[1]], verts[face[2]]]
                polygons.append(triangle)
                face_colors.append(color)
        
        if polygons:
            collection = Poly3DCollection(polygons, edgecolors='#111111', linewidths=0.2)
            collection.set_facecolors(face_colors)
            self.ax.add_collection3d(collection)
        
        if all_x:
            self.ax.set_xlim3d(np.min(all_x), np.max(all_x))
            self.ax.set_ylim3d(np.min(all_y), np.max(all_y))
            self.ax.set_zlim3d(np.min(all_z)-10, np.max(all_z)+10)

        self.ax.invert_zaxis()
        self.apply_aspect_ratio()
        self.ax.view_init(elev=20, azim=-45)
        self.format_axes()

        fig.tight_layout()
        self.canvas_fig = FigureCanvasTkAgg(fig, master=self.controller.image_plot_container)
        self.canvas_fig.draw()
        canvas_widget = self.canvas_fig.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_fig, self.controller.image_plot_container)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.controller.opt_canvas_widget = canvas_widget
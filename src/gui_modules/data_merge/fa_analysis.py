import customtkinter as ctk
from tkinter import filedialog
import pathlib
import h5py
import numpy as np
import cv2
from PIL import Image, ImageTk
import pandas as pd

from core_backend.data_merge.logger_config import PipelineLogger

class FAAnalysisFrame(ctk.CTkFrame):
    """
    GUI module for Step 2: FA Analysis & ROI Cropping.
    Scans for regions.h5 files, lets the user draw a polygon to mask the valid ice core area,
    matches grains with grains.txt, and exports a unified .h5 file with masks, grain statistics,
    and 3D c-axis orientation vectors.
    """
    def __init__(self, master, controller):
        super().__init__(master, fg_color="transparent")
        self.controller = controller
        self.logger = PipelineLogger()
        
        self.input_dir = None
        self.output_dir = None
        self.file_queue = []
        self.current_file_idx = 0
        
        # State variables for polygon drawing
        self.polygon_pts_canvas = []  
        self.polygon_pts_image = []   
        self.current_ws_image = None
        self.current_grains_df = None
        self.tk_preview_image = None
        
        # Geometry tracking
        self.img_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Canvas binding IDs
        self.bind_click_id = None
        self.bind_motion_id = None
        self.bind_right_id = None

        self._setup_layout()

    def _setup_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # --- Header ---
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header_frame.grid_columnconfigure(1, weight=1)

        btn_back = ctk.CTkButton(
            header_frame, text="◀ Back to Dashboard", font=("Arial", 14),
            width=150, fg_color="#345B94", hover_color="#20395D",
            command=lambda: self.controller.show_frame("StartFrame")
        )
        btn_back.grid(row=0, column=0, sticky="w")

        lbl_title = ctk.CTkLabel(
            header_frame, text="FA Analysis & ROI Cropping", 
            font=("Arial", 24, "bold"), text_color="#6C9CE8"
        )
        lbl_title.grid(row=0, column=1, sticky="e")

        # --- Section 1: Directory Selection ---
        dir_frame = ctk.CTkFrame(self, fg_color="#1B2433", corner_radius=15)
        dir_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        dir_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            dir_frame, text="1. Workspace Configuration", 
            font=("Arial", 18, "bold"), text_color="white"
        ).pack(pady=(15, 10), padx=20, anchor="w")

        self.btn_input = ctk.CTkButton(
            dir_frame, text="Select Input Folder", font=("Arial", 16),
            height=40, fg_color="#345B94", hover_color="#20395D", command=self._select_input
        )
        self.btn_input.pack(pady=5, padx=20, fill="x")
        self.lbl_input = ctk.CTkLabel(dir_frame, text="None", text_color="gray")
        self.lbl_input.pack(pady=(0, 10), padx=20, anchor="w")

        self.btn_output = ctk.CTkButton(
            dir_frame, text="Select Output Folder", font=("Arial", 16),
            height=40, fg_color="#345B94", hover_color="#20395D", command=self._select_output
        )
        self.btn_output.pack(pady=5, padx=20, fill="x")
        self.lbl_output = ctk.CTkLabel(dir_frame, text="None", text_color="gray")
        self.lbl_output.pack(pady=(0, 20), padx=20, anchor="w")

        # --- Section 2: Processing Controls ---
        ctrl_frame = ctk.CTkFrame(self, fg_color="#1B2433", corner_radius=15)
        ctrl_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        ctrl_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            ctrl_frame, text="2. Cropping & Processing", 
            font=("Arial", 18, "bold"), text_color="white"
        ).pack(pady=(15, 10), padx=20, anchor="w")

        self.btn_start = ctk.CTkButton(
            ctrl_frame, text="Start Scanning", font=("Arial", 18, "bold"),
            height=50, fg_color="#28a745", hover_color="#1e7e34", state="disabled",
            command=self._start_processing
        )
        self.btn_start.pack(pady=10, padx=20, fill="x")

        self.btn_next = ctk.CTkButton(
            ctrl_frame, text="Apply Crop & Next Image", font=("Arial", 18, "bold"),
            height=50, fg_color="#d35400", hover_color="#e67e22", state="disabled",
            command=self._apply_crop_and_next
        )
        self.btn_next.pack(pady=10, padx=20, fill="x")

        self.lbl_status = ctk.CTkLabel(ctrl_frame, text="Status: Waiting for directories", font=("Arial", 14, "italic"), text_color="gray")
        self.lbl_status.pack(pady=(5, 20), padx=20)

    def _select_input(self):
        folder = filedialog.askdirectory(title="Select Input Root Folder")
        if folder:
            self.input_dir = pathlib.Path(folder)
            self.lbl_input.configure(text=f".../{self.input_dir.name}", text_color="white")
            self._check_ready()

    def _select_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_dir = pathlib.Path(folder)
            self.lbl_output.configure(text=f".../{self.output_dir.name}", text_color="white")
            self._check_ready()

    def _check_ready(self):
        if self.input_dir and self.output_dir:
            self.btn_start.configure(state="normal")
            self.lbl_status.configure(text="Status: Ready to scan for regions.h5", text_color="#28a745")

    def _start_processing(self):
        self.btn_start.configure(state="disabled")
        self.btn_input.configure(state="disabled")
        self.btn_output.configure(state="disabled")
        
        self.file_queue = list(self.input_dir.rglob("regions.h5"))
        if not self.file_queue:
            self.lbl_status.configure(text="Status: No regions.h5 found!", text_color="#dc3545")
            self.logger.warning("No regions.h5 found in the selected directory.")
            self.btn_start.configure(state="normal")
            return
            
        self.current_file_idx = 0
        self._load_current_image()

    def _load_current_image(self):
        if self.current_file_idx >= len(self.file_queue):
            self._cleanup_canvas()
            self.lbl_status.configure(text="Status: All files processed successfully!", text_color="#28a745")
            self.btn_next.configure(state="disabled")
            self.logger.info("All files processed successfully.")
            return

        current_file = self.file_queue[self.current_file_idx]
        self.lbl_status.configure(text=f"Status: Processing {self.current_file_idx + 1}/{len(self.file_queue)}", text_color="#ffc107")
        self.logger.info(f"Loading file ({self.current_file_idx + 1}/{len(self.file_queue)}): {current_file}")

        # Use the parent sample folder as the label for this analysis file.
        display_name = current_file.parent.parent.name

        if display_name.lower().endswith(".h5"):
            display_name = display_name[:-3]

        self.controller.label_image_plot_info.configure(
            text=f"Draw Polygon for: {display_name} | Left Click: Add Point | Right Click: Undo"
        )

        # Load the watershed label image from the HDF5 file.
        with h5py.File(current_file, "r") as f:
            if "ws_image" not in f:
                self.logger.error(f"ws_image missing in {current_file}")
                self.current_file_idx += 1
                self._load_current_image()
                return
            self.current_ws_image = np.squeeze(f["ws_image"][:])

        # Load the grain table that contains the fabric-analysis measurements.
        grains_txt_path = current_file.parent / "grains.txt"
        if grains_txt_path.exists():
            grains_data = []
            with open(grains_txt_path, "r") as gf:
                for line in gf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    grains_data.append([float(x) for x in line.split()])
            
            cols = [
                "grain_id", "size_pixel", "center_x", "center_y",
                "mean_azimuth", "mean_latitude", "mean_colatitude",
                "regelungsgrad", "phase"
            ]
            self.current_grains_df = pd.DataFrame(grains_data, columns=cols)
            self.current_grains_df["grain_id"] = self.current_grains_df["grain_id"].astype(int)
            self.logger.info(f"Loaded grains.txt with {len(self.current_grains_df)} entries.")
        else:
            self.logger.warning(f"No grains.txt found at {grains_txt_path}. C-axis export will be skipped.")
            self.current_grains_df = None

        # Create a color preview of the labeled watershed regions.
        preview_8bit = (self.current_ws_image % 255).astype(np.uint8)
        preview_bgr = cv2.applyColorMap(preview_8bit, cv2.COLORMAP_JET)
        preview_bgr[self.current_ws_image == 0] = [0, 0, 0]
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)

        self._cleanup_canvas()
        self.controller.label_image_plot_info.pack(pady=5, fill="x")
        self.controller.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.controller.canvas.update()

        canvas_w = self.controller.canvas.winfo_width()
        canvas_h = self.controller.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 900, 700

        img_h, img_w = preview_rgb.shape[:2]
        self.img_scale = min(canvas_w / max(img_w, 1), canvas_h / max(img_h, 1))
        new_w = max(1, int(img_w * self.img_scale))
        new_h = max(1, int(img_h * self.img_scale))

        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        img_pil = Image.fromarray(preview_rgb).resize((new_w, new_h), Image.Resampling.NEAREST)
        self.tk_preview_image = ImageTk.PhotoImage(img_pil)

        self.controller.canvas.delete("all")
        self.controller.canvas.create_image(
            self.offset_x, self.offset_y, image=self.tk_preview_image, anchor="nw", tags="bg_image"
        )
        self.controller.canvas.image = self.tk_preview_image
        self.controller.canvas.configure(cursor="crosshair")

        self.polygon_pts_canvas = []
        self.polygon_pts_image = []

        self.bind_click_id = self.controller.canvas.bind("<Button-1>", self._on_canvas_click)
        self.bind_motion_id = self.controller.canvas.bind("<Motion>", self._on_canvas_motion)
        self.bind_right_id = self.controller.canvas.bind("<Button-3>", self._on_canvas_right_click)

        self.btn_next.configure(state="normal")
        self.logger.info(f"Canvas ready. Size: {canvas_w}x{canvas_h}, Scale: {self.img_scale:.4f}")

    def _on_canvas_click(self, event):
        cx, cy = event.x, event.y
        img_h, img_w = self.current_ws_image.shape[:2]
        scaled_w = img_w * self.img_scale
        scaled_h = img_h * self.img_scale

        clamped_cx = max(self.offset_x, min(cx, self.offset_x + scaled_w))
        clamped_cy = max(self.offset_y, min(cy, self.offset_y + scaled_h))

        img_x = int((clamped_cx - self.offset_x) / self.img_scale)
        img_y = int((clamped_cy - self.offset_y) / self.img_scale)

        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))

        self.polygon_pts_canvas.append((clamped_cx, clamped_cy))
        self.polygon_pts_image.append((img_x, img_y))

        self.logger.info(f"Point {len(self.polygon_pts_canvas)}: Canvas=({clamped_cx}, {clamped_cy}) -> Image=({img_x}, {img_y})")
        self._redraw_polygon()

    def _on_canvas_right_click(self, event):
        if self.polygon_pts_canvas:
            self.polygon_pts_canvas.pop()
            self.polygon_pts_image.pop()
            self.logger.info(f"Removed point. Remaining: {len(self.polygon_pts_canvas)}")
            self._redraw_polygon()

    def _on_canvas_motion(self, event):
        if not self.polygon_pts_canvas:
            return

        self.controller.canvas.delete("rubberband")
        last_pt = self.polygon_pts_canvas[-1]
        self.controller.canvas.create_line(
            last_pt[0], last_pt[1], event.x, event.y,
            fill="#00FFCC", width=2, dash=(4, 4), tags="rubberband"
        )

    def _redraw_polygon(self):
        self.controller.canvas.delete("poly_line", "poly_point", "poly_close_line")

        num_pts = len(self.polygon_pts_canvas)
        for i in range(num_pts):
            pt = self.polygon_pts_canvas[i]
            r = 6
            self.controller.canvas.create_oval(
                pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r,
                fill="#FFFF00", outline="#FFFFFF", width=2, tags="poly_point"
            )

            if i > 0:
                prev_pt = self.polygon_pts_canvas[i - 1]
                self.controller.canvas.create_line(
                    prev_pt[0], prev_pt[1], pt[0], pt[1],
                    fill="#00FFCC", width=2, tags="poly_line"
                )

        if num_pts >= 3:
            first_pt = self.polygon_pts_canvas[0]
            last_pt = self.polygon_pts_canvas[-1]
            self.controller.canvas.create_line(
                last_pt[0], last_pt[1], first_pt[0], first_pt[1],
                fill="#FF0055", width=2, dash=(3, 3), tags="poly_close_line"
            )

        self.controller.canvas.tag_raise("poly_line")
        self.controller.canvas.tag_raise("poly_close_line")
        self.controller.canvas.tag_raise("poly_point")
        self.controller.canvas.tag_raise("rubberband")

    def _apply_crop_and_next(self):
        mask = np.zeros_like(self.current_ws_image, dtype=np.uint8)
        
        if len(self.polygon_pts_image) < 3:
            self.logger.warning("Not enough points for polygon. Keeping entire image area.")
            mask.fill(1)
        else:
            pts_array = np.array(self.polygon_pts_image, dtype=np.int32)
            cv2.fillPoly(mask, [pts_array], 1)

        # Apply the drawn polygon to the watershed label image.
        masked_ws = self.current_ws_image.copy()
        masked_ws[mask == 0] = 0

        # Retain the grain identifiers visible inside the polygon.
        surviving_ids = set(np.unique(masked_ws))
        surviving_ids.discard(0)  # Label 0 denotes the background.
        self.logger.info(f"{len(surviving_ids)} grains inside polygon mask.")

        # Derive a stable output name from the sample folder.
        current_file = self.file_queue[self.current_file_idx]
        
        # Use the sample folder containing the current analysis directory.
        sample_name = current_file.parent.parent.name

        # Remove an HDF5 suffix if the folder name includes one.
        if sample_name.lower().endswith(".h5"):
            sample_name = sample_name[:-3]

        # Use a generic name when no sample-specific folder is available.
        if sample_name.lower() in ["regions", self.input_dir.name.lower()]:
            output_filename = "regions_preprocessed.h5"
        else:
            # Include the sample name to keep outputs from different samples distinct.
            output_filename = f"{sample_name}_regions_preprocessed.h5"

        output_file = self.output_dir / output_filename

        try:
            with h5py.File(output_file, "w") as f_out:
                # Save the masked watershed labels.
                f_out.create_dataset(
                    "ws_image", 
                    data=masked_ws[:, :, np.newaxis], 
                    compression="gzip", 
                    compression_opts=4
                )
                self.logger.info("Saved cropped 'ws_image' dataset.")

                # Filter the grain table and calculate C-axis vectors.
                if self.current_grains_df is not None:
                    # Keep only grains that remain inside the polygon mask.
                    filtered_df = self.current_grains_df[self.current_grains_df["grain_id"].isin(surviving_ids)].copy()

                    # Convert azimuth and colatitude measurements to 3D vectors.
                    az_rad = np.radians(filtered_df["mean_azimuth"].values)
                    colat_rad = np.radians(filtered_df["mean_colatitude"].values)

                    # Map the fabric-analysis zenith axis to the global X axis.
                    vx = np.cos(colat_rad)
                    vy = np.sin(colat_rad) * np.sin(az_rad)
                    vz = np.sin(colat_rad) * np.cos(az_rad)

                    filtered_df["c_axis_x"] = vx
                    filtered_df["c_axis_y"] = vy
                    filtered_df["c_axis_z"] = vz

                    # Store the filtered grain measurements as a structured HDF5 table.
                    records = filtered_df.to_records(index=False)
                    f_out.create_dataset("grain_table", data=records)

                    # Store the C-axis vectors separately as an N-by-3 matrix.
                    vectors_nx3 = np.column_stack([vx, vy, vz])
                    f_out.create_dataset("c_axis_vectors", data=vectors_nx3)

                    # Build a per-pixel C-axis map for the masked watershed image.
                    img_h, img_w = masked_ws.shape
                    c_axis_map = np.zeros((img_h, img_w, 3), dtype=np.float32)
                    
                    # Use grain identifiers as indices into a vector lookup table.
                    max_id = int(np.max(self.current_ws_image)) + 1
                    lookup_lut = np.zeros((max_id, 3), dtype=np.float32)
                    for row in filtered_df.itertuples():
                        lookup_lut[int(row.grain_id)] = [row.c_axis_x, row.c_axis_y, row.c_axis_z]

                    # Assign each labeled pixel its grain's C-axis vector.
                    c_axis_map = lookup_lut[masked_ws]
                    f_out.create_dataset(
                        "c_axis_map", 
                        data=c_axis_map, 
                        compression="gzip", 
                        compression_opts=4
                    )
                    self.logger.info(f"Successfully saved {len(filtered_df)} grains, vectors, and c_axis_map.")

            self.logger.info(f"File successfully created: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to process and save {output_file}: {e}")

        # Continue with the next analysis file.
        self.current_file_idx += 1
        self._load_current_image()

    def _cleanup_canvas(self):
        self.controller._clear_right_container()
        if self.bind_click_id:
            self.controller.canvas.unbind("<Button-1>", self.bind_click_id)
        if self.bind_motion_id:
            self.controller.canvas.unbind("<Motion>", self.bind_motion_id)
        if self.bind_right_id:
            self.controller.canvas.unbind("<Button-3>", self.bind_right_id)
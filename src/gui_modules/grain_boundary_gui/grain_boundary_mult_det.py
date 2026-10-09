import sys
from pathlib import Path

# Add the source directory to the import path when this module is run directly.
file_path = Path(__file__).resolve()
for parent in file_path.parents:
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

import customtkinter as ctk
from tkinter import filedialog
import threading
import logging
import pathlib

# Backend operation used by the batch workflow
from core_backend.grain_boundary_segmentation.pipeline_detection import run_detection_pipeline

class GrainBoundaryMultipleDetectionFrame(ctk.CTkFrame):
    """
    GUI Frame for the Grain Boundary Batch Processing step.
    Iterates through a selected parent folder and runs the detection pipeline 
    for all sub-images that already have a generated mask.
    """
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        self.paths = {
            "parent_folder": None,
            "obj_model": None,
            "e2e_model": None,
            "tjunc_model": None,
            "config_json": None
        }
        
        self.cancel_requested = False
        self.ui_elements = []  # Controls disabled while the batch is running.
        
        # Define module specific colors (Orange/Brown theme for Grain Boundaries)
        self.btn_color = "#D67040"
        self.btn_hover = "#804326"
        self.bg_color = "#2E241F"
        self.field_bg = "#1F1815" 

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=5)
        self.grid_columnconfigure(2, weight=1)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # --- TITLE & BACK BUTTON ---
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, columnspan=3, sticky="we", padx=10, pady=10)
        
        self.btn_back = ctk.CTkButton(top_frame, text="Back", command=lambda: self.controller.show_frame("StartFrame"),
                                      corner_radius=20, font=("Arial", 16), width=70, height=35)
        self.btn_back.pack(side="left")
        ctk.CTkLabel(top_frame, text="Grain Boundary Batch Processing", font=("Arial", 35, "bold")).pack(side="left", padx=20)

        # --- MAIN CONTENT BOX ---
        main_box = ctk.CTkFrame(self, fg_color="#1A1311", corner_radius=15)
        main_box.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
        
        main_box.grid_columnconfigure(0, weight=1)
        main_box.grid_rowconfigure(0, weight=1)
        main_box.grid_rowconfigure(1, weight=1)
        
        # =========================================================
        # 1. TOP SECTION: File & Folder Selections
        # =========================================================
        inputs_section = ctk.CTkFrame(main_box, fg_color="transparent")
        inputs_section.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(inputs_section, text="1. Inputs & Models", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=(0, 10), anchor="w")
        
        def add_picker(parent, title, key, filetypes=None, is_folder=False):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.pack(fill="x", pady=7)
            btn = ctk.CTkButton(frame, text=title, width=220, fg_color=self.btn_color, hover_color=self.btn_hover,
                                command=lambda: self._select_path(key, lbl, filetypes, is_folder))
            btn.pack(side="left")
            lbl = ctk.CTkLabel(frame, text="Not selected", text_color="gray")
            lbl.pack(side="left", padx=15)
            self.ui_elements.append(btn)
            
        add_picker(inputs_section, "Select Parent Folder", "parent_folder", is_folder=True)
        add_picker(inputs_section, "Parameter JSON (.json)", "config_json", [("JSON", "*.json")])
        add_picker(inputs_section, "Object RF Model (.joblib)", "obj_model", [("Joblib", "*.joblib")])
        add_picker(inputs_section, "E2E RF Model (.joblib)", "e2e_model", [("Joblib", "*.joblib")])
        add_picker(inputs_section, "T-Junc RF Model (.joblib)", "tjunc_model", [("Joblib", "*.joblib")])

        # =========================================================
        # 2. BOTTOM SECTION: Parameters
        # =========================================================
        params_section = ctk.CTkFrame(main_box, fg_color="transparent")
        params_section.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        
        ctk.CTkLabel(params_section, text="2. Parameters", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=(0, 10), anchor="w")
        
        # Threshold Slider
        ctk.CTkLabel(params_section, text="Path Prob. Threshold:", font=("Arial", 16)).pack(anchor="w", pady=(0,5))
        self.lbl_thresh = ctk.CTkLabel(params_section, text="50%", font=("Arial", 16, "bold"), text_color=self.btn_color)
        self.lbl_thresh.pack(anchor="w")
        self.slider_thresh = ctk.CTkSlider(params_section, from_=0.0, to=1.0, number_of_steps=100, button_color=self.btn_color, progress_color=self.btn_color, command=self._update_thresh_label)
        self.slider_thresh.set(0.50)
        self.slider_thresh.pack(fill="x", pady=(0, 15))
        self.ui_elements.append(self.slider_thresh)
        
        # Crop
        crop_frame = ctk.CTkFrame(params_section, fg_color="transparent")
        crop_frame.pack(fill="x", pady=5)
        self.cb_crop = ctk.CTkCheckBox(crop_frame, text="Crop Images?", font=("Arial", 16), fg_color=self.btn_color, command=self._toggle_crop)
        self.cb_crop.pack(side="left")
        self.entry_crop = ctk.CTkEntry(crop_frame, placeholder_text="Size (px)", width=90, state="disabled", fg_color=self.field_bg)
        self.entry_crop.pack(side="left", padx=15)
        self.ui_elements.extend([self.cb_crop, self.entry_crop])
        
        # Resolution
        res_frame = ctk.CTkFrame(params_section, fg_color="transparent")
        res_frame.pack(fill="x", pady=10)
        ctk.CTkLabel(res_frame, text="Resolution (µm/px):", font=("Arial", 16)).pack(side="left")
        self.entry_res = ctk.CTkEntry(res_frame, width=70, fg_color=self.field_bg)
        self.entry_res.insert(0, "1.41")
        self.entry_res.pack(side="left", padx=15)
        self.ui_elements.append(self.entry_res)
        
        # Debug Intermediates
        self.cb_debug = ctk.CTkCheckBox(params_section, text="Save Debug Steps & Overlays", font=("Arial", 16), fg_color=self.btn_color)
        self.cb_debug.select()
        self.cb_debug.pack(anchor="w", pady=10)
        self.ui_elements.append(self.cb_debug)

        # =========================================================
        # 3. BOTTOM: Actions & Status
        # =========================================================
        bot_frame = ctk.CTkFrame(self, fg_color="transparent")
        bot_frame.grid(row=2, column=1, pady=20)
        
        btn_frame = ctk.CTkFrame(bot_frame, fg_color="transparent")
        btn_frame.pack(pady=10)
        
        self.btn_start = ctk.CTkButton(btn_frame, text="START BATCH", font=("Arial", 22, "bold"), height=60, width=250, fg_color=self.btn_color, hover_color=self.btn_hover, command=self.start_detection)
        self.btn_start.pack(side="left", padx=10)
        
        self.btn_cancel = ctk.CTkButton(btn_frame, text="CANCEL", font=("Arial", 18, "bold"), height=60, width=120, fg_color="#c0392b", hover_color="#922b21", state="disabled", command=self.cancel_detection)
        self.btn_cancel.pack(side="left", padx=10)
        
        self.lbl_status = ctk.CTkLabel(bot_frame, text="Waiting for inputs...", font=("Arial", 16), text_color="gray")
        self.lbl_status.pack()
        
        self.progress_bar = ctk.CTkProgressBar(bot_frame, width=400, progress_color=self.btn_color)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)
        self.progress_bar.pack_forget()

    def _update_thresh_label(self, val):
        self.lbl_thresh.configure(text=f"{int(val * 100)}%")

    def _toggle_crop(self):
        state = "normal" if self.cb_crop.get() else "disabled"
        color = "#2b2b2b" if self.cb_crop.get() else self.field_bg
        if not self.cb_crop.get(): self.entry_crop.delete(0, 'end')
        self.entry_crop.configure(state=state, fg_color=color)

    def _select_path(self, key, label_widget, filetypes, is_folder):
        if is_folder:
            f = filedialog.askdirectory(title="Select Parent Folder")
        else:
            f = filedialog.askopenfilename(filetypes=filetypes)
            
        if f:
            self.paths[key] = f
            label_widget.configure(text=pathlib.Path(f).name, text_color="white")

    def cancel_detection(self):
        self.cancel_requested = True
        self.lbl_status.configure(text="Cancelling... waiting for current image to finish.", text_color="#e67e22")
        self.btn_cancel.configure(state="disabled")

    def _set_ui_state(self, state):
        self.btn_back.configure(state=state)
        self.btn_start.configure(state=state)
        for element in self.ui_elements:
            try:
                element.configure(state=state)
            except Exception:
                pass
        # Preserve the crop field's disabled state when no crop is selected.
        if state == "normal" and not self.cb_crop.get():
            self.entry_crop.configure(state="disabled")

    def start_detection(self):
        if not self.paths["parent_folder"]:
            self.lbl_status.configure(text="Missing Parent Folder!", text_color="#e74c3c")
            return
            
        crop_val = None
        if self.cb_crop.get():
            try: 
                crop_val = int(self.entry_crop.get())
                if crop_val <= 0: raise ValueError
            except: 
                self.lbl_status.configure(text="Invalid crop size!", text_color="#e74c3c")
                return

        try: res_val = float(self.entry_res.get().replace(',', '.'))
        except: res_val = 1.41
            
        self.cancel_requested = False
        self._set_ui_state("disabled")
        self.btn_cancel.configure(state="normal")
        
        self.lbl_status.configure(text="Scanning for files...", text_color="#f39c12")
        self.progress_bar.pack()
        self.progress_bar.set(0)

        threading.Thread(target=self._run_multiple_detection_thread, args=(crop_val, res_val), daemon=True).start()

    def _run_multiple_detection_thread(self, crop_val, res_val):
        try:
            p = Path(self.paths["parent_folder"])
            all_tifs = list(p.rglob("*.tif"))
            
            # Exclude generated masks, overlays, and scaled copies from the input list.
            raw_tifs = [f for f in all_tifs if not f.name.endswith(('_mask.tif', '_mask.png', '_overlay.png', '_micrometer_scaled.tif'))]
            
            # Process only raw images with a matching border mask.
            valid_tifs = []
            for img_path in raw_tifs:
                mask_png = img_path.parent / f"{img_path.stem}_mask.png"
                mask_tif = img_path.parent / f"{img_path.stem}_mask.tif"
                if mask_png.exists() or mask_tif.exists():
                    valid_tifs.append(img_path)

            total_images = len(valid_tifs)

            if total_images == 0:
                self.controller.after(0, self._on_finish, False, "No valid raw .tif files with matching masks found in the selected folder!")
                return

            save_intermediates = bool(self.cb_debug.get())
            successful_images = 0

            for idx, img_path in enumerate(valid_tifs):
                if self.cancel_requested:
                    break

                def status_cb(msg, i=idx):
                    self.controller.after(0, lambda: self.lbl_status.configure(
                        text=f"[{i+1}/{total_images}] {img_path.name}\n-> {msg}", 
                        text_color="#f39c12"
                    ))

                def progress_cb(val, i=idx):
                    overall_progress = (i + val) / total_images
                    self.controller.after(0, lambda: [self.progress_bar.configure(mode="determinate"), self.progress_bar.set(overall_progress)])

                # Suppress per-image previews during batch processing.
                def img_cb(p, t): pass 

                try:
                    status_cb("Starting Detection Pipeline...")
                    run_detection_pipeline(
                        image_path=str(img_path),
                        model_path=self.paths["obj_model"],
                        e2e_model_path=self.paths["e2e_model"],
                        tjunc_model_path=self.paths["tjunc_model"],
                        config_path=self.paths["config_json"],
                        path_prob_threshold=self.slider_thresh.get(),
                        crop_size=crop_val,
                        resolution=res_val,
                        save_intermediates=save_intermediates,
                        status_callback=status_cb,
                        progress_callback=progress_cb,
                        image_callback=img_cb
                    )
                    successful_images += 1
                except Exception as e:
                    logging.error(f"Failed to process {img_path.name}: {e}", exc_info=True)
                    # Continue processing the remaining images after a failure.

            if self.cancel_requested:
                msg = f"Cancelled! Processed {successful_images}/{total_images} images."
            else:
                msg = f"Batch finished! Successfully processed {successful_images}/{total_images} images."
            
            self.controller.after(0, self._on_finish, True, msg)
            
        except Exception as e:
            logging.error(f"Error during batch detection: {e}", exc_info=True)
            self.controller.after(0, self._on_finish, False, str(e))

    def _on_finish(self, success, msg):
        self.progress_bar.pack_forget()
        self._set_ui_state("normal")
        self.btn_cancel.configure(state="disabled")
        
        color = "#2ecc71" if success else "#e74c3c"
        if self.cancel_requested: color = "#e67e22"
        self.lbl_status.configure(text=msg, text_color=color)
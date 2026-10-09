import sys
from pathlib import Path

file_path = Path(__file__).resolve()
for parent in file_path.parents:
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

import customtkinter as ctk
from tkinter import filedialog
import threading
import logging
import pathlib

from core_backend.grain_boundary_segmentation.pipeline_detection import run_detection_pipeline

class GrainBoundaryDetectionFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        self.paths = {
            "obj_model": None,
            "e2e_model": None,
            "tjunc_model": None,
            "image": None,
            "config_json": None
        }
        
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

        # Title & Back
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, columnspan=3, sticky="we", padx=10, pady=10)
        ctk.CTkButton(top_frame, text="Back", command=lambda: self.controller.show_frame("StartFrame"),
                      corner_radius=20, font=("Arial", 16), width=70, height=35).pack(side="left")
        ctk.CTkLabel(top_frame, text="Grain Boundary Detection", font=("Arial", 35, "bold")).pack(side="left", padx=20)

        # Main Content Box
        main_box = ctk.CTkFrame(self, fg_color="#1A1311", corner_radius=15)
        main_box.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
        
        main_box.grid_columnconfigure(0, weight=1)
        main_box.grid_rowconfigure(0, weight=1)
        main_box.grid_rowconfigure(1, weight=1)
        
        # --- TOP SECTION: File Selections ---
        inputs_section = ctk.CTkFrame(main_box, fg_color="transparent")
        inputs_section.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(inputs_section, text="1. Inputs & Models", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=(0, 20), anchor="w")
        
        def add_file_picker(parent, title, key, filetypes):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.pack(fill="x", pady=10)
            btn = ctk.CTkButton(frame, text=title, width=180, fg_color=self.btn_color, hover_color=self.btn_hover,
                                command=lambda: self._select_file(key, lbl, filetypes))
            btn.pack(side="left")
            lbl = ctk.CTkLabel(frame, text="Not selected", text_color="gray")
            lbl.pack(side="left", padx=15)
            
        add_file_picker(inputs_section, "Select Image (.tif)", "image", [("TIF", "*.tif")])
        add_file_picker(inputs_section, "Parameter JSON (.json)", "config_json", [("JSON", "*.json")])
        add_file_picker(inputs_section, "Object RF Model", "obj_model", [("Joblib", "*.joblib")])
        add_file_picker(inputs_section, "E2E RF Model", "e2e_model", [("Joblib", "*.joblib")])
        add_file_picker(inputs_section, "T-Junc RF Model", "tjunc_model", [("Joblib", "*.joblib")])

        # --- BOTTOM SECTION: Settings ---
        params_section = ctk.CTkFrame(main_box, fg_color="transparent")
        params_section.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        
        ctk.CTkLabel(params_section, text="2. Parameters", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=(0, 20), anchor="w")
        
        # Threshold Slider
        ctk.CTkLabel(params_section, text="Path Prob. Threshold:", font=("Arial", 16)).pack(anchor="w", pady=(0,5))
        self.lbl_thresh = ctk.CTkLabel(params_section, text="50%", font=("Arial", 16, "bold"), text_color=self.btn_color)
        self.lbl_thresh.pack(anchor="w")
        self.slider_thresh = ctk.CTkSlider(params_section, from_=0.0, to=1.0, number_of_steps=100, button_color=self.btn_color, progress_color=self.btn_color, command=self._update_thresh_label)
        self.slider_thresh.set(0.50)
        self.slider_thresh.pack(fill="x", pady=(0, 20))
        
        # Crop & Resolution
        crop_frame = ctk.CTkFrame(params_section, fg_color="transparent")
        crop_frame.pack(fill="x", pady=5)
        self.cb_crop = ctk.CTkCheckBox(crop_frame, text="Crop Image?", font=("Arial", 16), fg_color=self.btn_color, command=self._toggle_crop)
        self.cb_crop.pack(side="left")
        self.entry_crop = ctk.CTkEntry(crop_frame, placeholder_text="Size (px)", width=90, state="disabled", fg_color=self.field_bg)
        self.entry_crop.pack(side="left", padx=15)
        
        res_frame = ctk.CTkFrame(params_section, fg_color="transparent")
        res_frame.pack(fill="x", pady=15)
        ctk.CTkLabel(res_frame, text="Resolution (µm/px):", font=("Arial", 16)).pack(side="left")
        self.entry_res = ctk.CTkEntry(res_frame, width=70, fg_color=self.field_bg)
        self.entry_res.insert(0, "1.41")
        self.entry_res.pack(side="left", padx=15)
        
        self.cb_debug = ctk.CTkCheckBox(params_section, text="Save Debug Steps & Overlays", font=("Arial", 16), fg_color=self.btn_color)
        self.cb_debug.select()
        self.cb_debug.pack(anchor="w", pady=15)

        # --- BOTTOM: Start & Status ---
        bot_frame = ctk.CTkFrame(self, fg_color="transparent")
        bot_frame.grid(row=2, column=1, pady=20)
        
        self.btn_start = ctk.CTkButton(bot_frame, text="START DETECTION", font=("Arial", 22, "bold"), height=60, width=300, fg_color=self.btn_color, hover_color=self.btn_hover, command=self.start_detection)
        self.btn_start.pack(pady=10)
        
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

    def _select_file(self, key, label_widget, filetypes):
        f = filedialog.askopenfilename(filetypes=filetypes)
        if f:
            self.paths[key] = f
            label_widget.configure(text=pathlib.Path(f).name, text_color="white")

    def start_detection(self):
        # Allow missing models if only Phase 1 is desired, but require image
        if not self.paths["image"]:
            self.lbl_status.configure(text="Missing Image Input!", text_color="#e74c3c")
            return
            
        crop_val = None
        if self.cb_crop.get():
            try: crop_val = int(self.entry_crop.get())
            except: self.lbl_status.configure(text="Invalid crop size!", text_color="#e74c3c"); return

        try: res_val = float(self.entry_res.get().replace(',', '.'))
        except: res_val = 1.41
            
        self.btn_start.configure(state="disabled")
        self.lbl_status.configure(text="Running...", text_color="#f39c12")
        self.progress_bar.pack()
        self.progress_bar.start()

        threading.Thread(target=self._run_detection_thread, args=(crop_val, res_val), daemon=True).start()

    def _run_detection_thread(self, crop_val, res_val):
        try:
            def stat_cb(msg): self.controller.after(0, lambda: self.lbl_status.configure(text=msg))
            def prog_cb(v): self.controller.after(0, lambda: [self.progress_bar.stop(), self.progress_bar.configure(mode="determinate"), self.progress_bar.set(v)])
            def img_cb(p, t): self.controller.after(0, lambda: self.controller.show_image_plot(p, t))

            h5_path, gpkg_path = run_detection_pipeline(
                image_path=self.paths["image"],
                model_path=self.paths["obj_model"],
                e2e_model_path=self.paths["e2e_model"],
                tjunc_model_path=self.paths["tjunc_model"],
                config_path=self.paths["config_json"],
                path_prob_threshold=self.slider_thresh.get(),
                resolution=res_val,
                save_intermediates=self.cb_debug.get(),
                status_callback=stat_cb,
                progress_callback=prog_cb,
                image_callback=img_cb,
                crop_size=crop_val
            )
            self.controller.after(0, lambda: self._on_finish(True, "Pipeline finished successfully!"))
        except Exception as e:
            logging.error(f"Detection error: {e}", exc_info=True)
            self.controller.after(0, lambda: self._on_finish(False, str(e)))

    def _on_finish(self, success, msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_start.configure(state="normal")
        color = "#2ecc71" if success else "#e74c3c"
        self.lbl_status.configure(text=msg, text_color=color)
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

from core_backend.grain_boundary_segmentation.pipeline_calibration import generate_ground_truth, generate_path_ground_truth, train_ml_model

class GrainBoundaryCalibrationFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        self.paths = {
            "obj_folder": None, "obj_json": None, "obj_model": None,
            "e2e_folder": None, "e2e_json": None, "e2e_model": None,
            "tj_folder": None, "tj_json": None, "config_json": None
        }
        
        self.btn_color = "#D67040"
        self.btn_hover = "#804326"
        self.bg_color = "#2E241F"
        self.field_bg = "#1F1815"

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title & Back
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="we", padx=10, pady=10)
        
        ctk.CTkButton(top_frame, text="Back", command=lambda: self.controller.show_frame("StartFrame"),
                      corner_radius=20, font=("Arial", 16), width=70, height=35).pack(side="left")
        ctk.CTkLabel(top_frame, text="Grain Boundary Calibration", font=("Arial", 35, "bold")).pack(side="left", padx=20)
        
        self.cb_autotune = ctk.CTkCheckBox(top_frame, text="Auto-Tune Hyperparameters", font=("Arial", 16), fg_color=self.btn_color)
        self.cb_autotune.select()
        self.cb_autotune.pack(side="right", padx=20)

        self.btn_config = ctk.CTkButton(top_frame, text="Select Parameter JSON", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("config_json", self.lbl_config, [("JSON", "*.json")]))
        self.btn_config.pack(side="right", padx=10)
        self.lbl_config = ctk.CTkLabel(top_frame, text="Using Defaults", text_color="gray")
        self.lbl_config.pack(side="right", padx=5)

        self.cols_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.cols_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.cols_frame.grid_columnconfigure((0,1,2), weight=1, uniform="col")
        self.cols_frame.grid_rowconfigure(0, weight=1)

        self._build_object_column(0)
        self._build_e2e_column(1)
        self._build_tjunc_column(2)

    def _build_object_column(self, col_idx):
        frame = ctk.CTkFrame(self.cols_frame, fg_color="#1A1311", corner_radius=15)
        frame.grid(row=0, column=col_idx, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(frame, text="1. Object Model\n(Grain vs. Artifact)", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=20)
        
        ctk.CTkLabel(frame, text="Step 1: Generate Ground Truth", font=("Arial", 16, "bold")).pack(pady=(10,5))
        btn_dir = ctk.CTkButton(frame, text="Select Image Folder", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_dir("obj_folder", lbl_dir))
        btn_dir.pack(pady=5)
        lbl_dir = ctk.CTkLabel(frame, text="No folder selected", text_color="gray"); lbl_dir.pack()
        
        sq_frame = ctk.CTkFrame(frame, fg_color="transparent")
        sq_frame.pack(pady=10)
        ctk.CTkLabel(sq_frame, text="Squares:").grid(row=0, column=0, padx=5, pady=2)
        ent_sq = ctk.CTkEntry(sq_frame, width=40, fg_color=self.field_bg)
        ent_sq.insert(0, "3"); ent_sq.grid(row=0, column=1)
        ctk.CTkLabel(sq_frame, text="Size [px]:").grid(row=0, column=2, padx=5, pady=2)
        ent_size = ctk.CTkEntry(sq_frame, width=50, fg_color=self.field_bg)
        ent_size.insert(0, "8080"); ent_size.grid(row=0, column=3)
        
        btn_start = ctk.CTkButton(frame, text="Start Labeling", fg_color="#E67E22", hover_color="#D35400",
                                  command=lambda: self._start_label_obj(ent_sq.get(), ent_size.get(), lbl_status))
        btn_start.pack(pady=10)

        ctk.CTkLabel(frame, text="-"*40, text_color="gray").pack(pady=15)

        ctk.CTkLabel(frame, text="Step 2: Train Model", font=("Arial", 16, "bold")).pack(pady=(10,5))
        btn_json = ctk.CTkButton(frame, text="Select JSON", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("obj_json", lbl_json, [("JSON", "*.json")]))
        btn_json.pack(pady=5)
        lbl_json = ctk.CTkLabel(frame, text="No JSON selected", text_color="gray"); lbl_json.pack()
        
        btn_train = ctk.CTkButton(frame, text="Train Object RF", font=("Arial", 18, "bold"), height=40, fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._start_train("object", "obj_json", lbl_status))
        btn_train.pack(pady=15)
        
        lbl_status = ctk.CTkLabel(frame, text="Waiting...", text_color="gray", wraplength=350, justify="center")
        lbl_status.pack(side="bottom", pady=20)

    def _build_e2e_column(self, col_idx):
        frame = ctk.CTkFrame(self.cols_frame, fg_color="#1A1311", corner_radius=15)
        frame.grid(row=0, column=col_idx, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(frame, text="2. E2E Routing Model\n(End-to-End Paths)", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=20)
        
        ctk.CTkLabel(frame, text="Step 1: Generate Ground Truth", font=("Arial", 16, "bold")).pack(pady=(10,5))
        btn_dir = ctk.CTkButton(frame, text="Select Image Folder", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_dir("e2e_folder", lbl_dir))
        btn_dir.pack(pady=5); lbl_dir = ctk.CTkLabel(frame, text="No folder selected", text_color="gray"); lbl_dir.pack()
        
        btn_mod = ctk.CTkButton(frame, text="Select Object RF Model", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("obj_model", lbl_mod, [("Joblib", "*.joblib")]))
        btn_mod.pack(pady=5); lbl_mod = ctk.CTkLabel(frame, text="Required: Trained Object Model", text_color="gray"); lbl_mod.pack()
        
        sq_frame = ctk.CTkFrame(frame, fg_color="transparent")
        sq_frame.pack(pady=10)
        ctk.CTkLabel(sq_frame, text="Squares:").grid(row=0, column=0, padx=5, pady=2)
        ent_sq = ctk.CTkEntry(sq_frame, width=40, fg_color=self.field_bg)
        ent_sq.insert(0, "3"); ent_sq.grid(row=0, column=1)
        ctk.CTkLabel(sq_frame, text="Size [px]:").grid(row=0, column=2, padx=5, pady=2)
        ent_size = ctk.CTkEntry(sq_frame, width=50, fg_color=self.field_bg)
        ent_size.insert(0, "8080"); ent_size.grid(row=0, column=3)
        ctk.CTkLabel(sq_frame, text="Max Paths/Sq:").grid(row=1, column=0, columnspan=2, padx=5, pady=(5,0), sticky="e")
        ent_max = ctk.CTkEntry(sq_frame, width=50, fg_color=self.field_bg)
        ent_max.insert(0, "15"); ent_max.grid(row=1, column=2, columnspan=2, pady=(5,0), sticky="w")
        
        btn_start = ctk.CTkButton(frame, text="Start Labeling (E2E)", fg_color="#E67E22", hover_color="#D35400", 
                                  command=lambda: self._start_label_path("E2E", "e2e_folder", ent_sq.get(), ent_size.get(), ent_max.get(), lbl_status))
        btn_start.pack(pady=10)

        ctk.CTkLabel(frame, text="-"*40, text_color="gray").pack(pady=15)

        ctk.CTkLabel(frame, text="Step 2: Train Model", font=("Arial", 16, "bold")).pack(pady=(10,5))
        btn_json = ctk.CTkButton(frame, text="Select JSON", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("e2e_json", lbl_json, [("JSON", "*.json")]))
        btn_json.pack(pady=5); lbl_json = ctk.CTkLabel(frame, text="No JSON selected", text_color="gray"); lbl_json.pack()
        
        btn_train = ctk.CTkButton(frame, text="Train E2E RF", font=("Arial", 18, "bold"), height=40, fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._start_train("path_e2e", "e2e_json", lbl_status))
        btn_train.pack(pady=15)
        
        lbl_status = ctk.CTkLabel(frame, text="Waiting...", text_color="gray", wraplength=350, justify="center")
        lbl_status.pack(side="bottom", pady=20)

    def _build_tjunc_column(self, col_idx):
        frame = ctk.CTkFrame(self.cols_frame, fg_color="#1A1311", corner_radius=15)
        frame.grid(row=0, column=col_idx, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(frame, text="3. T-Junction Model\n(Connecting Branches)", font=("Arial", 22, "bold"), text_color=self.btn_color).pack(pady=20)
        
        ctk.CTkLabel(frame, text="Step 1: Generate Ground Truth", font=("Arial", 16, "bold")).pack(pady=(10,5))
        btn_dir = ctk.CTkButton(frame, text="Select Image Folder", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_dir("tj_folder", lbl_dir))
        btn_dir.pack(pady=5); lbl_dir = ctk.CTkLabel(frame, text="No folder selected", text_color="gray"); lbl_dir.pack()
        
        btn_mod_obj = ctk.CTkButton(frame, text="Select Object RF Model", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("obj_model", lbl_mod_obj, [("Joblib", "*.joblib")]))
        btn_mod_obj.pack(pady=(10, 5)); lbl_mod_obj = ctk.CTkLabel(frame, text="Required: Trained Object Model", text_color="gray"); lbl_mod_obj.pack()

        btn_mod_e2e = ctk.CTkButton(frame, text="Select E2E RF Model", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("e2e_model", lbl_mod_e2e, [("Joblib", "*.joblib")]))
        btn_mod_e2e.pack(pady=(10, 5)); lbl_mod_e2e = ctk.CTkLabel(frame, text="Required: Trained E2E Model", text_color="gray"); lbl_mod_e2e.pack()
        
        sq_frame = ctk.CTkFrame(frame, fg_color="transparent")
        sq_frame.pack(pady=10)
        ctk.CTkLabel(sq_frame, text="Squares:").grid(row=0, column=0, padx=5, pady=2)
        ent_sq = ctk.CTkEntry(sq_frame, width=40, fg_color=self.field_bg)
        ent_sq.insert(0, "3"); ent_sq.grid(row=0, column=1)
        ctk.CTkLabel(sq_frame, text="Size [px]:").grid(row=0, column=2, padx=5, pady=2)
        ent_size = ctk.CTkEntry(sq_frame, width=50, fg_color=self.field_bg)
        ent_size.insert(0, "8080"); ent_size.grid(row=0, column=3)
        ctk.CTkLabel(sq_frame, text="Max Paths/Sq:").grid(row=1, column=0, columnspan=2, padx=5, pady=(5,0), sticky="e")
        ent_max = ctk.CTkEntry(sq_frame, width=50, fg_color=self.field_bg)
        ent_max.insert(0, "15"); ent_max.grid(row=1, column=2, columnspan=2, pady=(5,0), sticky="w")
        
        btn_start = ctk.CTkButton(frame, text="Start Labeling (T-Junc)", fg_color="#E67E22", hover_color="#D35400", 
                                  command=lambda: self._start_label_path("T-Junc", "tj_folder", ent_sq.get(), ent_size.get(), ent_max.get(), lbl_status))
        btn_start.pack(pady=10)

        ctk.CTkLabel(frame, text="-"*40, text_color="gray").pack(pady=15)

        ctk.CTkLabel(frame, text="Step 2: Train Model", font=("Arial", 16, "bold")).pack(pady=(10,5))
        btn_json = ctk.CTkButton(frame, text="Select JSON", fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._select_file("tj_json", lbl_json, [("JSON", "*.json")]))
        btn_json.pack(pady=5); lbl_json = ctk.CTkLabel(frame, text="No JSON selected", text_color="gray"); lbl_json.pack()
        
        btn_train = ctk.CTkButton(frame, text="Train T-Junc RF", font=("Arial", 18, "bold"), height=40, fg_color=self.btn_color, hover_color=self.btn_hover, command=lambda: self._start_train("path_tjunc", "tj_json", lbl_status))
        btn_train.pack(pady=15)
        
        lbl_status = ctk.CTkLabel(frame, text="Waiting...", text_color="gray", wraplength=350, justify="center")
        lbl_status.pack(side="bottom", pady=20)

    # --- Utility Methods ---
    def _select_dir(self, key, label_widget):
        d = filedialog.askdirectory()
        if d:
            self.paths[key] = d
            label_widget.configure(text=pathlib.Path(d).name, text_color="white")

    def _select_file(self, key, label_widget, filetypes):
        f = filedialog.askopenfilename(filetypes=filetypes)
        if f:
            self.paths[key] = f
            label_widget.configure(text=pathlib.Path(f).name, text_color="white")

    # --- Thread Execution ---
    def _start_label_obj(self, sq_str, size_str, status_lbl):
        folder = self.paths["obj_folder"]
        cfg_path = self.paths.get("config_json")
        if not folder:
            status_lbl.configure(text="Error: Select folder!", text_color="red"); return
        try: sq, size = int(sq_str), int(size_str)
        except: status_lbl.configure(text="Error: Invalid ints!", text_color="red"); return
        
        status_lbl.configure(text="Running...", text_color="orange")
        def run():
            try:
                res = generate_ground_truth(
                    cal_folder=folder, 
                    squares_per_image=sq, 
                    square_size=size, 
                    status_callback=lambda msg: self.controller.after(0, lambda: status_lbl.configure(text=msg)),
                    config_path=cfg_path
                )
                self.controller.after(0, lambda: status_lbl.configure(text="Saved: " + Path(res).name, text_color="green"))
                self.paths["obj_json"] = str(res)
            except Exception as e:
                error_msg = str(e)  # Capture the exception before scheduling the GUI callback.
                self.controller.after(0, lambda msg=error_msg: status_lbl.configure(text=f"Error: {msg}", text_color="red"))
        threading.Thread(target=run, daemon=True).start()

    def _start_label_path(self, phase, folder_key, sq_str, size_str, max_str, status_lbl):
        folder = self.paths[folder_key]
        obj_mod = self.paths["obj_model"]
        e2e_mod = self.paths["e2e_model"]
        cfg_path = self.paths.get("config_json")
        
        if not folder or not obj_mod:
            status_lbl.configure(text="Error: Missing Image Folder or Object Model!", text_color="red"); return
        if phase == "T-Junc" and not e2e_mod:
            status_lbl.configure(text="Error: Missing E2E Model!", text_color="red"); return
            
        try: sq, size, mx = int(sq_str), int(size_str), int(max_str)
        except: status_lbl.configure(text="Error: Invalid ints!", text_color="red"); return

        status_lbl.configure(text="Running...", text_color="orange")
        def run():
            try:
                res = generate_path_ground_truth(
                    cal_folder=folder, 
                    object_rf_path=obj_mod, 
                    e2e_rf_path=e2e_mod, 
                    phase=phase, 
                    squares_per_image=sq, 
                    square_size=size, 
                    max_paths=mx, 
                    status_callback=lambda msg: self.controller.after(0, lambda: status_lbl.configure(text=msg)),
                    config_path=cfg_path
                )
                self.controller.after(0, lambda: status_lbl.configure(text="Saved: " + Path(res).name, text_color="green"))
            except Exception as e:
                error_msg = str(e)
                self.controller.after(0, lambda msg=error_msg: status_lbl.configure(text=f"Error: {msg}", text_color="red"))
        threading.Thread(target=run, daemon=True).start()

    def _start_train(self, model_type, json_key, status_lbl):
        json_path = self.paths[json_key]
        if not json_path:
            status_lbl.configure(text="Error: Select JSON!", text_color="red"); return
            
        status_lbl.configure(text="Training...", text_color="orange")
        autotune = self.cb_autotune.get() == 1
        
        def run():
            try:
                name, acc = train_ml_model(json_path, model_type, autotune, lambda msg: self.controller.after(0, lambda: status_lbl.configure(text=msg)))
                self.controller.after(0, lambda: status_lbl.configure(text=f"Done! Acc: {acc}", text_color="green"))
                
                if model_type == "object": self.paths["obj_model"] = str(Path(json_path).parent / name)
                if model_type == "path_e2e": self.paths["e2e_model"] = str(Path(json_path).parent / name)
            except Exception as e:
                self.controller.after(0, lambda: status_lbl.configure(text=f"Error: {e}", text_color="red"))
        threading.Thread(target=run, daemon=True).start()
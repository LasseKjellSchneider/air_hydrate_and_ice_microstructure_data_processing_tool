import customtkinter as ctk
import threading
import pathlib
import json
import random
from tkinter import filedialog
import logging
import traceback

from gui_modules.hydrate_segmentation.label_tool import HydrateLabelTool

from core_backend.hydrate_segmentation.hydrate_calibration import (
    estimate_time_2d_calc,
    run_2d_optimization,
    run_3d_optimization,
    load_tiff_stack,
    GLOBAL_CONSTANTS,
    load_global_constants
)

from core_backend.hydrate_segmentation.hydrate_detection import (
    preprocess_image,
    edge_detection,
    extract_and_filter_objects
)
from core_backend.hydrate_segmentation.hydrate_focus import label_focus_candidates

from core_backend.hydrate_segmentation.hydrate_rf_trainer import extract_category_2_data, train_random_forest


class HydrateCalibrationFrame(ctk.CTkFrame):
    """
    GUI Frame for managing 2D Calibration, 3D Z-Focus Calibration, 
    and the new Random Forest Training Pipeline.
    """
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

        if not GLOBAL_CONSTANTS:
            load_global_constants()

        self.data_folder_rf = None
        self.rf_model_file_2d = None
        self.rf_model_file_3d = None

        self.data_folder_2d = None
        self.data_folder_3d = None
        self.param_file_2d = None

        c_rf_btn = "#9b59b6"       
        c_rf_hover = "#8e44ad"     
        c_rf_field_bg = "#2c1b3d"  

        c_2d_btn = "#539C93"       
        c_2d_hover = "#2D6961"     
        c_2d_field_bg = "#1F2E2C"  

        c_3d_btn = "#DAAD4D"       
        c_3d_hover = "#D67040"     
        c_3d_field_bg = "#2E241F"  

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=4)

        for i in range(25):
            self.grid_rowconfigure(i, weight=1)

        # ---------------------------------------------------------
        # HEADER SECTION
        # ---------------------------------------------------------
        ctk.CTkLabel(self, text="Calibration", font=("Arial", 40, "bold")).grid(row=0, column=1, columnspan=3)

        self.btn_back = ctk.CTkButton(
            self, text="back", command=lambda: self.controller.show_frame("StartFrame"),
            corner_radius=40, font=("Arial", 20), width=70, height=30
        )
        self.btn_back.grid(row=0, column=0, sticky="wn", padx=10, pady=10)
        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=1, column=0, columnspan=5)

        # ---------------------------------------------------------
        # RANDOM FOREST SECTION
        # ---------------------------------------------------------
        ctk.CTkLabel(self, text="RF", font=("Arial", 40, "bold"), text_color=c_rf_btn).grid(row=2, column=0, rowspan=2)

        self.btn_select_data_rf = ctk.CTkButton(
            self, text="select preprocessed\ndata folder (RF)", corner_radius=20, font=("Arial", 15), width=120, height=50,
            command=self.select_data_rf, fg_color=c_rf_btn, hover_color=c_rf_hover
        )
        self.btn_select_data_rf.grid(row=2, column=1, sticky="w")
        
        self.label_data_info_rf = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_data_info_rf.grid(row=2, column=3, sticky="w")

        self.frame_rf_actions = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_rf_actions.grid(row=3, column=1, columnspan=3, sticky="we", pady=5)
        self.frame_rf_actions.grid_columnconfigure(2, weight=1)
        
        self.btn_extract_rf = ctk.CTkButton(
            self.frame_rf_actions, text="Create Ground Truth", corner_radius=20, font=("Arial", 15), 
            command=self.handle_extract_rf, fg_color=c_rf_btn, hover_color=c_rf_hover
        )
        self.btn_extract_rf.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        self.btn_train_rf = ctk.CTkButton(
            self.frame_rf_actions, text="Train RF Model", corner_radius=20, font=("Arial", 15), 
            command=self.handle_train_rf, fg_color=c_rf_btn, hover_color=c_rf_hover
        )
        self.btn_train_rf.grid(row=1, column=0, sticky="w")

        self.label_rf_status = ctk.CTkLabel(self.frame_rf_actions, text="", font=("Arial", 15, "bold"), text_color="#2ecc71")
        self.label_rf_status.grid(row=1, column=1, padx=10, sticky="w")
        
        self.progress_bar_rf = ctk.CTkProgressBar(self.frame_rf_actions, height=10, corner_radius=5, progress_color=c_rf_btn)
        self.progress_bar_rf.set(0)
        self.progress_bar_rf.grid(row=1, column=2, padx=(10, 20), sticky="we")

        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=4, column=0, columnspan=5)

        # ---------------------------------------------------------
        # 2D CALIBRATION SECTION
        # ---------------------------------------------------------
        ctk.CTkLabel(self, text="2D", font=("Arial", 40, "bold"), text_color=c_2d_btn).grid(row=5, column=0, rowspan=5)

        self.btn_select_data_2d = ctk.CTkButton(
            self, text="select preprocessed\ndata folder (2D)", corner_radius=20, font=("Arial", 15), width=120, height=50,
            command=self.select_data_2d, fg_color=c_2d_btn, hover_color=c_2d_hover
        )
        self.btn_select_data_2d.grid(row=5, column=1, sticky="w", pady=5)
        
        self.label_data_info_2d = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_data_info_2d.grid(row=5, column=3, sticky="w")

        self.btn_select_rf_model_2d = ctk.CTkButton(
            self, text="select RF Model\n(.joblib)", corner_radius=20, font=("Arial", 15), width=120, height=50,
            command=self.select_rf_model_2d, fg_color=c_2d_btn, hover_color=c_2d_hover
        )
        self.btn_select_rf_model_2d.grid(row=6, column=1, sticky="w", pady=5)
        
        self.label_rf_model_info_2d = ctk.CTkLabel(self, text="no model selected", font=("Arial", 15), text_color="gray")
        self.label_rf_model_info_2d.grid(row=6, column=3, sticky="w")

        self.frame_gt_2d = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_gt_2d.grid(row=7, column=1, columnspan=3, sticky="w", pady=5)
        
        self.btn_create_gt_2d = ctk.CTkButton(
            self.frame_gt_2d, text="Check / Create Ground Truth", corner_radius=20, font=("Arial", 15), 
            command=self.handle_gt_2d, fg_color=c_2d_btn, hover_color=c_2d_hover
        )
        self.btn_create_gt_2d.pack(side="left", padx=(0,10))
        
        self.entry_num_stacks_2d = ctk.CTkEntry(self.frame_gt_2d, width=60, placeholder_text="Stacks", fg_color=c_2d_field_bg, border_color=c_2d_btn)
        self.entry_num_stacks_2d.pack(side="left", padx=(0,10))
        
        self.label_gt_status_2d = ctk.CTkLabel(self.frame_gt_2d, text="", font=("Arial", 15, "bold"), text_color="#2ecc71")
        self.label_gt_status_2d.pack(side="left")

        self.frame_opt = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=10)
        self.frame_opt.grid(row=8, column=1, columnspan=3, sticky="we", pady=5, ipady=5, padx=(0,20))

        params = ["popsize", "maxiter", "workers", "mutation", "recombination"]
        defaults = ["15", "100", "-1", "0.5, 1.8", "0.8"]
        self.opt_entries = {}

        for p_name, default in zip(params, defaults):
            container = ctk.CTkFrame(self.frame_opt, fg_color="transparent")
            container.pack(side="left", expand=True, fill="x", padx=5)
            ctk.CTkLabel(container, text=p_name+":", font=("Arial", 13)).pack(side="left")
            entry = ctk.CTkEntry(container, width=50, font=("Arial", 13), fg_color=c_2d_field_bg, border_color=c_2d_btn)
            entry.insert(0, default)
            entry.pack(side="right")
            self.opt_entries[p_name] = entry

        self.frame_2d_actions = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_2d_actions.grid(row=9, column=1, columnspan=3, sticky="we")
        
        self.btn_estimate_2d = ctk.CTkButton(
            self.frame_2d_actions, text="Estimate Time", corner_radius=20, font=("Arial", 15), width=120, height=40,
            fg_color=c_2d_btn, hover_color=c_2d_hover, command=self.estimate_time_2d
        )
        self.btn_estimate_2d.pack(side="left", padx=(0, 10))
        
        self.label_estimate = ctk.CTkLabel(self.frame_2d_actions, text="~ 0h 00m", font=("Arial", 15, "italic"), text_color="gray")
        self.label_estimate.pack(side="left", padx=(0, 10))

        self.label_opt_status_2d = ctk.CTkLabel(self.frame_2d_actions, text="", font=("Arial", 15, "italic"), text_color="gray")
        self.label_opt_status_2d.pack(side="left", padx=10, expand=True)

        self.btn_start_2d = ctk.CTkButton(
            self.frame_2d_actions, text="Start 2D Optimization", corner_radius=20, font=("Arial", 18, "bold"), 
            fg_color=c_2d_btn, hover_color=c_2d_hover, width=200, height=40, command=self.start_2d_calibration
        )
        self.btn_start_2d.pack(side="right", padx=20)

        self.frame_2d_progress = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_2d_progress.grid(row=10, column=1, columnspan=3, sticky="we", pady=(5,0))
        
        self.progress_bar_2d = ctk.CTkProgressBar(self.frame_2d_progress, height=10, corner_radius=5, progress_color=c_2d_btn)
        self.progress_bar_2d.set(0)
        self.progress_bar_2d.pack(side="left", fill="x", expand=True, padx=(0, 20))
        
        self.label_eta_2d = ctk.CTkLabel(self.frame_2d_progress, text="", font=("Arial", 13, "bold"), text_color="#3498db")
        self.label_eta_2d.pack(side="right", padx=(0, 20))

        # ---------------------------------------------------------
        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=11, column=0, columnspan=5, pady=10)

        # ---------------------------------------------------------
        # 3D CALIBRATION SECTION
        # ---------------------------------------------------------
        ctk.CTkLabel(self, text="3D", font=("Arial", 40, "bold"), text_color=c_3d_btn).grid(row=12, column=0, rowspan=5)

        self.btn_select_data_3d = ctk.CTkButton(
            self, text="select preprocessed\ndata folder (3D)", corner_radius=20, font=("Arial", 15), width=120, height=50,
            command=self.select_data_3d, fg_color=c_3d_btn, hover_color=c_3d_hover
        )
        self.btn_select_data_3d.grid(row=12, column=1, sticky="w", pady=5)
        
        self.label_data_info_3d = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_data_info_3d.grid(row=12, column=3, sticky="w")

        self.btn_select_param_2d = ctk.CTkButton(
            self, text="select 2D param\nJSON file", corner_radius=20, font=("Arial", 15), width=120, height=50,
            command=self.select_param_2d, fg_color=c_3d_btn, hover_color=c_3d_hover
        )
        self.btn_select_param_2d.grid(row=13, column=1, sticky="w", pady=5)
        
        self.label_param_2d_info = ctk.CTkLabel(self, text="no file selected", font=("Arial", 15), text_color="gray")
        self.label_param_2d_info.grid(row=13, column=3, sticky="w")

        self.btn_select_rf_model_3d = ctk.CTkButton(
            self, text="select RF Model\n(.joblib)", corner_radius=20, font=("Arial", 15), width=120, height=50,
            command=self.select_rf_model_3d, fg_color=c_3d_btn, hover_color=c_3d_hover
        )
        self.btn_select_rf_model_3d.grid(row=14, column=1, sticky="w", pady=5)
        
        self.label_rf_model_info_3d = ctk.CTkLabel(self, text="no model selected", font=("Arial", 15), text_color="gray")
        self.label_rf_model_info_3d.grid(row=14, column=3, sticky="w")

        self.frame_gt_3d = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_gt_3d.grid(row=15, column=1, columnspan=3, sticky="w", pady=5)
        
        self.btn_create_gt_3d = ctk.CTkButton(
            self.frame_gt_3d, text="Check / Create 3D Ground Truth", corner_radius=20, font=("Arial", 15), 
            command=self.handle_gt_3d, fg_color=c_3d_btn, hover_color=c_3d_hover
        )
        self.btn_create_gt_3d.pack(side="left", padx=(0,10))
        
        self.entry_num_stacks_3d = ctk.CTkEntry(self.frame_gt_3d, width=60, placeholder_text="Stacks", fg_color=c_3d_field_bg, border_color=c_3d_btn)
        self.entry_num_stacks_3d.pack(side="left", padx=(0,10))
        
        self.label_gt_status_3d = ctk.CTkLabel(self.frame_gt_3d, text="", font=("Arial", 15, "bold"), text_color="#2ecc71")
        self.label_gt_status_3d.pack(side="left")

        self.frame_opt_3d = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=10)
        self.frame_opt_3d.grid(row=16, column=1, columnspan=3, sticky="we", pady=5, ipady=5, padx=(0,20))

        params_3d = ["popsize", "maxiter", "workers", "mutation", "recombination"]
        defaults_3d = ["10", "50", "1", "0.6, 1.2", "0.9"]
        self.opt_entries_3d = {}

        for p_name, default in zip(params_3d, defaults_3d):
            container = ctk.CTkFrame(self.frame_opt_3d, fg_color="transparent")
            container.pack(side="left", expand=True, fill="x", padx=5)
            ctk.CTkLabel(container, text=p_name+":", font=("Arial", 13)).pack(side="left")
            entry = ctk.CTkEntry(container, width=50, font=("Arial", 13), fg_color=c_3d_field_bg, border_color=c_3d_btn)
            entry.insert(0, default)
            entry.pack(side="right")
            self.opt_entries_3d[p_name] = entry

        self.frame_3d_actions = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_3d_actions.grid(row=17, column=1, columnspan=3, sticky="we", pady=10)
        
        self.label_opt_status_3d = ctk.CTkLabel(self.frame_3d_actions, text="", font=("Arial", 15, "italic"), text_color="gray")
        self.label_opt_status_3d.pack(side="left", padx=20, expand=True)

        self.btn_start_3d = ctk.CTkButton(
            self.frame_3d_actions, text="Start 3D Optimization", corner_radius=20, font=("Arial", 18, "bold"), 
            fg_color=c_3d_btn, hover_color=c_3d_hover, width=200, height=40, command=self.start_3d_calibration
        )
        self.btn_start_3d.pack(side="right", padx=20)


    # =========================================================
    # RF CALLBACKS
    # =========================================================
    def select_data_rf(self):
        folder = filedialog.askdirectory(title="Select Parent Data Folder for RF Extraction")
        if folder:
            self.data_folder_rf = folder
            self.label_data_info_rf.configure(text=folder.split("/")[-1], text_color="white")
            self.label_rf_status.configure(text="")

    def handle_extract_rf(self):
        if not self.data_folder_rf:
            self.label_rf_status.configure(text="Select data folder first!", text_color="#e74c3c")
            return
        
        self.label_rf_status.configure(text="Searching subfolders & extracting...", text_color="#f39c12")
        
        def worker():
            extract_category_2_data(self.controller, self.data_folder_rf)
            self.controller.after(0, lambda: self.label_rf_status.configure(text="Finished.", text_color="#2ecc71"))
            
        threading.Thread(target=worker, daemon=True).start()

    def handle_train_rf(self):
        if not self.data_folder_rf:
            self.label_rf_status.configure(text="Select data folder first!", text_color="#e74c3c")
            return
        
        self.label_rf_status.configure(text="Optimizing RF...", text_color="#f39c12")
        self.progress_bar_rf.set(0)
        
        def progress_cb(current, total):
            if total > 0:
                ratio = current / total
                self.controller.after(0, lambda: self.progress_bar_rf.set(ratio))
        
        def worker():
            success, report_or_err = train_random_forest(self.data_folder_rf, progress_cb=progress_cb)
            
            if success:
                self.controller.after(0, lambda: [
                    self.label_rf_status.configure(text="Model trained & saved!", text_color="#2ecc71"),
                    self.progress_bar_rf.set(1.0),
                    self.show_rf_report_popup(report_or_err) 
                ])
            else:
                self.controller.after(0, lambda: self.label_rf_status.configure(text=f"Error! {report_or_err[:20]}", text_color="#e74c3c"))
                
        threading.Thread(target=worker, daemon=True).start()

    def show_rf_report_popup(self, report_text):
        popup = ctk.CTkToplevel(self)
        popup.title("Random Forest Evaluation Report (Per Sample)")
        popup.geometry("800x650")
        popup.attributes("-topmost", True)
        
        txt_canvas = ctk.CTkTextbox(popup, font=("Consolas", 14), fg_color="#1e1e1e", text_color="#ecf0f1")
        txt_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        txt_canvas.insert("0.0", report_text)
        txt_canvas.configure(state="disabled") 
        
        btn_close = ctk.CTkButton(popup, text="Close Report", command=popup.destroy, fg_color="#9b59b6", hover_color="#8e44ad")
        btn_close.pack(pady=10)


    # =========================================================
    # 2D CALLBACKS
    # =========================================================
    def select_data_2d(self):
        folder = filedialog.askdirectory(title="Select Preprocessed Data Folder for 2D")
        if folder:
            self.data_folder_2d = folder
            self.label_data_info_2d.configure(text=folder.split("/")[-1], text_color="white")
            self.label_gt_status_2d.configure(text="")
            self.btn_create_gt_2d.pack(side="left", padx=(0,10))
            self.entry_num_stacks_2d.pack(side="left", padx=(0,10))

    def select_rf_model_2d(self):
        file = filedialog.askopenfilename(title="Select Trained RF Model (.joblib)", filetypes=[("Joblib Files","*.joblib")])
        if file:
            self.rf_model_file_2d = file
            self.label_rf_model_info_2d.configure(text=file.split("/")[-1], text_color="white")

    def handle_gt_2d(self):
        if not self.data_folder_2d:
            self.label_gt_status_2d.configure(text="Select data folder first!", text_color="#e74c3c")
            return
            
        gt_dir = pathlib.Path(self.data_folder_2d) / "ground_truth"
        existing_files = list(gt_dir.glob("ground_truth_*.json")) if gt_dir.exists() else []
        
        if existing_files:
            self.btn_create_gt_2d.pack_forget()
            self.entry_num_stacks_2d.pack_forget()
            self.label_gt_status_2d.configure(
                text=f"Ground truth exists: {gt_dir.name} ({len(existing_files)} files)", 
                text_color="#2ecc71"
            )
        else:
            try:
                num = int(self.entry_num_stacks_2d.get())
            except ValueError:
                num = 3 
                
            processed_data_path = pathlib.Path(self.data_folder_2d)
            
            subfolders = [f for f in processed_data_path.iterdir() if f.is_dir() and f.name not in ["ground_truth", "random_forest_data"]]
            if not subfolders:
                subfolders = [processed_data_path]
                
            selected_stacks = []
            for subfolder in subfolders:
                stacks_in_sub = list(subfolder.glob("*.tif"))
                if stacks_in_sub:
                    take_num = max(1, min(num, len(stacks_in_sub)))
                    selected_stacks.extend(random.sample(stacks_in_sub, take_num))
            
            if not selected_stacks:
                self.label_gt_status_2d.configure(text="No TIF files found in subfolders!", text_color="#e74c3c")
                return
                
            gt_dir.mkdir(parents=True, exist_ok=True)
            
            def launch_sequence(stacks_to_label):
                if not stacks_to_label:
                    self.handle_gt_2d() 
                    return
                stack = stacks_to_label.pop(0)
                HydrateLabelTool(
                    self.controller, stack, gt_dir, 
                    on_close_callback=lambda: launch_sequence(stacks_to_label)
                )
                
            launch_sequence(list(selected_stacks))

    def estimate_time_2d(self):
        if not self.data_folder_2d:
            self.label_estimate.configure(text="Select folder first!", text_color="#e74c3c")
            return

        self.label_estimate.configure(text="Calculating...", text_color="#f39c12")
        
        def worker():
            popsize_str = self.opt_entries["popsize"].get()
            maxiter_str = self.opt_entries["maxiter"].get()
            
            result_str, is_err = estimate_time_2d_calc(self.data_folder_2d, popsize_str, maxiter_str)
            
            txt_color = "#e74c3c" if is_err else "#f39c12"
            self.controller.after(0, lambda: self.label_estimate.configure(text=result_str, text_color=txt_color))
            
        threading.Thread(target=worker, daemon=True).start()

    def start_2d_calibration(self):
        if not self.data_folder_2d:
            self.label_opt_status_2d.configure(text="Select data folder first!", text_color="#e74c3c")
            return
            
        gt_dir = pathlib.Path(self.data_folder_2d) / "ground_truth"
        if not gt_dir.exists() or not list(gt_dir.glob("ground_truth_*.json")):
            self.label_gt_status_2d.configure(text="Generate Ground Truth first!", text_color="#e74c3c")
            return

        processed_data_path = pathlib.Path(self.data_folder_2d)
        
        found_meta = list(processed_data_path.rglob("metadata_preprocessed.json"))
        if not found_meta:
            self.label_opt_status_2d.configure(text="metadata_preprocessed.json missing!", text_color="#e74c3c")
            return

        metadata_map = {}
        for mf in found_meta:
            with open(mf, "r") as f:
                metadata_map[mf.parent.name] = json.load(f)

        existing_stems = [p.stem.replace("ground_truth_", "") for p in gt_dir.glob("ground_truth_*.json")]
        available_stacks = [p for p in processed_data_path.rglob("*.tif") if p.stem in existing_stems]

        if not available_stacks:
            self.label_opt_status_2d.configure(
                text="No matching TIF stacks found for Ground Truth!", 
                text_color="#e74c3c"
            )
            return

        self.controller.setup_optimization_dashboard()
        
        self.progress_bar_2d.set(0)
        self.label_eta_2d.configure(text="Calculating ETA...")
        
        self.label_opt_status_2d.configure(
            text="Initializing 2D optimization (evaluating population)...", 
            text_color="#f39c12"
        )

        opt_config = {
            "popsize": self.opt_entries["popsize"].get(),
            "maxiter": self.opt_entries["maxiter"].get(),
            "workers": self.opt_entries["workers"].get(),
            "mutation": self.opt_entries["mutation"].get(),
            "recombination": self.opt_entries["recombination"].get(),
            "rf_model_path": self.rf_model_file_2d 
        }

        def progress_cb(gen, data, matched, total_gt, decoded_params, path_features, stats_matrix, hard_bounds, eta_str):
            def safe_update():
                try:
                    self.controller.update_optimization_dashboard(
                        gen=gen, data_dict=data, params=decoded_params, 
                        path_features=path_features, 
                        stats_matrix=stats_matrix, 
                        eta_str=eta_str
                    )
                    self.label_opt_status_2d.configure(
                        text=f"Generation {gen} finished. {matched}/{total_gt} hydrates correctly recognized.", 
                        text_color="#f39c12"
                    )
                    self.label_eta_2d.configure(text=f"ETA: {eta_str}")
                except Exception as e:
                    error_msg = traceback.format_exc()
                    logging.error(f"GUI Update Error in progress_cb: {error_msg}")
                    self.label_opt_status_2d.configure(text="GUI Update Error! Check terminal/logs.", text_color="#e74c3c")
                    print(f"\n[CRITICAL GUI ERROR]:\n{error_msg}\n")
            
            self.controller.after(0, safe_update)

        def intra_gen_cb(current, total):
            progress_ratio = (current / total) if total > 0 else 0.0
            self.controller.after(0, lambda: self.progress_bar_2d.set(progress_ratio))

        def completion_cb(path):
            self.controller.after(0, lambda: [
                self.label_opt_status_2d.configure(
                    text="Optimization process successfully finished!", 
                    text_color="#2ecc71"
                ),
                self.label_eta_2d.configure(text="Done."),
                self.progress_bar_2d.set(1.0)
            ])
            print(f"\n[SUCCESS] 2D Optimization finished! Saved to: {path}")

        def worker():
            try:
                run_2d_optimization(
                    selected_stacks=available_stacks, 
                    gt_dir=gt_dir, 
                    metadata=metadata_map, 
                    opt_config=opt_config, 
                    progress_callback=progress_cb,
                    completion_callback=completion_cb,
                    intra_gen_callback=intra_gen_cb
                )
            except Exception as opt_err:
                logging.error("Fatal error during 2D Optimization Worker thread!", exc_info=True)
                self.controller.after(0, lambda: self.label_opt_status_2d.configure(
                    text=f"Error: {opt_err}", text_color="#e74c3c"
                ))

        threading.Thread(target=worker, daemon=True).start()

    # =========================================================
    # 3D CALLBACKS
    # =========================================================
    def select_data_3d(self):
        folder = filedialog.askdirectory(title="Select Preprocessed Data Folder for 3D")
        if folder:
            self.data_folder_3d = folder
            self.label_data_info_3d.configure(text=folder.split("/")[-1], text_color="white")
            self.label_gt_status_3d.configure(text="")
            self.btn_create_gt_3d.pack(side="left", padx=(0,10))
            self.entry_num_stacks_3d.pack(side="left", padx=(0,10))

    def select_param_2d(self):
        file = filedialog.askopenfilename(title="Select 2D Parameters JSON", filetypes=[("JSON Files","*.json")])
        if file:
            self.param_file_2d = file
            self.label_param_2d_info.configure(text=file.split("/")[-1], text_color="white")

    def select_rf_model_3d(self):
        file = filedialog.askopenfilename(title="Select Trained RF Model (.joblib)", filetypes=[("Joblib Files","*.joblib")])
        if file:
            self.rf_model_file_3d = file
            self.label_rf_model_info_3d.configure(text=file.split("/")[-1], text_color="white")

    def handle_gt_3d(self):
        if not self.data_folder_3d or not self.param_file_2d:
            self.label_gt_status_3d.configure(text="Select folder and param file!", text_color="#e74c3c")
            return
            
        gt_focus_dir = pathlib.Path(self.data_folder_3d) / "ground_truth_focus"
        existing_files = list(gt_focus_dir.glob("focus_ground_truth_*.json")) if gt_focus_dir.exists() else []
        
        if existing_files:
            self.btn_create_gt_3d.pack_forget()
            self.entry_num_stacks_3d.pack_forget()
            self.label_gt_status_3d.configure(
                text=f"Ground truth exists: {gt_focus_dir.name} ({len(existing_files)} files)", 
                text_color="#2ecc71"
            )
        else:
            try:
                num = int(self.entry_num_stacks_3d.get())
            except ValueError:
                num = 2
                
            processed_data_path = pathlib.Path(self.data_folder_3d)
            available_stacks = list(processed_data_path.rglob("*.tif"))
            
            if not available_stacks:
                self.label_gt_status_3d.configure(text="No TIF files found!", text_color="#e74c3c")
                return
                
            num = max(1, min(num, len(available_stacks)))
            selected_stacks = random.sample(available_stacks, num)
            gt_focus_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.param_file_2d, "r") as f:
                base_params = json.load(f)
            base_params["distance_threshold"] = 0.0
            base_params["roi_radius_factor"] = 1.2
            
            found_meta = list(processed_data_path.rglob("metadata_preprocessed.json"))
            if not found_meta:
                self.label_gt_status_3d.configure(text="No metadata found!", text_color="#e74c3c")
                return
                
            metadata_map = {}
            for mf in found_meta:
                with open(mf, "r") as f:
                    metadata_map[mf.parent.name] = json.load(f)
                
            def worker():
                for image_stack_path in selected_stacks:
                    stack = load_tiff_stack(image_stack_path)
                    objects_in_stack = []
                    
                    current_meta = metadata_map.get(image_stack_path.parent.name)
                    if not current_meta:
                        current_meta = list(metadata_map.values())[0]

                    for layer in range(stack.shape[0]):
                        raw_image = stack[layer].copy()
                        img_prep = preprocess_image(raw_image, base_params)
                        img_bin = edge_detection(img_prep, base_params)
                        
                        # Handle the detection result format returned by the current backend.
                        extracted = extract_and_filter_objects(
                            img_bin, img_prep, raw_image, current_meta, image_stack_path.stem, layer, base_params, None, None
                        )
                        objects_in_stack.extend(extracted)
                        
                    label_focus_candidates(objects_in_stack, stack, base_params, image_stack_path, gt_focus_dir)
                
                self.controller.after(0, self.handle_gt_3d)
                
            threading.Thread(target=worker, daemon=True).start()

    def start_3d_calibration(self):
        if not self.data_folder_3d or not self.param_file_2d:
            self.label_gt_status_3d.configure(text="Select folder and 2D param file first!", text_color="#e74c3c")
            return

        processed_data_path = pathlib.Path(self.data_folder_3d)
        gt_focus_dir = processed_data_path / "ground_truth_focus"
        
        focus_files = list(gt_focus_dir.glob("focus_ground_truth_*.json")) if gt_focus_dir.exists() else []
        if not focus_files:
            self.label_gt_status_3d.configure(text="No 3D Ground Truth found!", text_color="#e74c3c")
            return

        found_meta = list(processed_data_path.rglob("metadata_preprocessed.json"))
        if not found_meta:
            self.label_gt_status_3d.configure(text="No metadata found!", text_color="#e74c3c")
            return
            
        metadata_map = {}
        for mf in found_meta:
            with open(mf, "r") as f:
                metadata_map[mf.parent.name] = json.load(f)

        with open(self.param_file_2d, "r") as f:
            base_params = json.load(f)

        self.controller.init_3d_plot()

        opt_config_3d = {
            "popsize": self.opt_entries_3d["popsize"].get(),
            "maxiter": self.opt_entries_3d["maxiter"].get(),
            "workers": self.opt_entries_3d["workers"].get(),
            "mutation": self.opt_entries_3d["mutation"].get(),
            "recombination": self.opt_entries_3d["recombination"].get(),
            "rf_model_path": self.rf_model_file_3d  
        }

        def progress_cb_3d(gen_list, err_list, correct, total):
            self.controller.after(0, lambda: [
                self.controller.update_3d_plot(gen_list, err_list),
                self.label_opt_status_3d.configure(
                    text=f"processing... {correct}/{total} correctly evaluated", 
                    text_color="#f39c12"
                )
            ])

        def completion_cb_3d(path):
            self.controller.after(0, lambda: [
                self.label_opt_status_3d.configure(
                    text=self.label_opt_status_3d.cget("text").replace("processing...", "process finished:"), 
                    text_color="#2ecc71"
                )
            ])
            print(f"\n[SUCCESS] 3D Optimization finished! Saved to: {path}")

        def worker():
            try:
                run_3d_optimization(
                    processed_data_path=processed_data_path,
                    selected_stacks=[], 
                    gt_focus_dir=gt_focus_dir,
                    metadata_preprocessed=metadata_map,
                    base_params=base_params,
                    opt_config=opt_config_3d,
                    progress_callback=progress_cb_3d,
                    completion_callback=completion_cb_3d
                )
            except Exception as e:
                logging.error(f"[ERROR in 3D Optimization Worker]: {e}", exc_info=True)

        threading.Thread(target=worker, daemon=True).start()
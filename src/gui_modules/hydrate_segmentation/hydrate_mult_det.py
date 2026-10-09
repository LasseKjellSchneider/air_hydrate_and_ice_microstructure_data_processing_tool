import customtkinter as ctk
import threading
from tkinter import filedialog
import logging
import pathlib
import time
import datetime
import json

# Import both Preprocessing and Detection backend functions
from core_backend.hydrate_segmentation.hydrate_preprocessing import gui_preprocessing_microscope
from core_backend.hydrate_segmentation.hydrate_detection import gui_hydrate_detection

class HydrateMultipleDetectionFrame(ctk.CTkFrame):
    """
    Batch processing frame for Hydrate Segmentation.
    Executes a complete End-to-End Pipeline (Preprocessing -> Detection) 
    iteratively across single datasets or multiple raw data subfolders.
    """
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        self.parent_raw_folder = None
        self.parent_output_folder = None
        self.parameter_file = None
        self.rf_model_file = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=4)
        self.grid_columnconfigure(4, weight=1)

        for i in range(13):
            self.grid_rowconfigure(i, weight=1)

        # ---------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------
        ctk.CTkLabel(self, text="End-to-End Batch Processing", font=("Arial", 40, "bold")).grid(row=0, column=1, columnspan=3, sticky="s")

        self.btn_back = ctk.CTkButton(
            self, text="back", command=lambda: self.controller.show_frame("StartFrame"),
            corner_radius=40, font=("Arial", 20), width=50, height=30
        )
        self.btn_back.grid(row=0, column=0, sticky="wn", padx=10, pady=10)

        ctk.CTkLabel(self, text=self.controller.VERSION, font=("Arial", 10), text_color="gray").grid(row=0, column=4, sticky="ne", padx=10, pady=10)
        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=1, column=0, columnspan=5)

        # ---------------------------------------------------------
        # FOLDER SELECTION
        # ---------------------------------------------------------
        # 1. Raw Data Input (Single OR Parent Folder)
        self.btn_select_raw = ctk.CTkButton(
            self, text="select RAW folder\n(Single or Parent)", command=self.select_raw_folder,
            corner_radius=20, font=("Arial", 15), width=120, height=60
        )
        self.btn_select_raw.grid(row=2, column=1)

        self.label_raw_info = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_raw_info.grid(row=2, column=3, sticky="nsew")

        # 2. Preprocessed Data Output (Target)
        self.btn_select_output = ctk.CTkButton(
            self, text="select parent\nOUTPUT folder", command=self.select_output_folder,
            corner_radius=20, font=("Arial", 15), width=120, height=60, fg_color="#27ae60", hover_color="#2ecc71"
        )
        self.btn_select_output.grid(row=3, column=1)

        self.label_output_info = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_output_info.grid(row=3, column=3, sticky="nsew")

        # 3. Detection Parameter File
        self.btn_select_param = ctk.CTkButton(
            self, text="select detection\nparameter file", corner_radius=20, font=("Arial", 15),
            width=120, height=60, command=self.select_parameter_file
        )
        self.btn_select_param.grid(row=4, column=1)

        self.label_param_info = ctk.CTkLabel(self, text="no file selected", font=("Arial", 15), text_color="gray")
        self.label_param_info.grid(row=4, column=3, sticky="nsew")

        # 4. Optional random-forest rescue model
        self.btn_select_rf = ctk.CTkButton(
            self, text="select RF Model\n(.joblib)", corner_radius=20, font=("Arial", 15),
            width=120, height=60, command=self.select_rf_file, fg_color="#9b59b6", hover_color="#8e44ad"
        )
        self.btn_select_rf.grid(row=5, column=1)

        self.label_rf_info = ctk.CTkLabel(self, text="no model selected (optional)", font=("Arial", 15), text_color="gray")
        self.label_rf_info.grid(row=5, column=3, sticky="nsew")

        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=6, column=0, columnspan=5)

        # ---------------------------------------------------------
        # SETTINGS & EXECUTION
        # ---------------------------------------------------------
        self.cb_overlay = ctk.CTkCheckBox(self, text="save all overlay images", font=("Arial", 15))
        self.cb_overlay.grid(row=7, column=1, columnspan=3)

        self.cb_amount_overlay = ctk.CTkCheckBox(self, text="save overlay from", font=("Arial", 15))
        self.cb_amount_overlay.grid(row=8, column=1, sticky="e")
        ctk.CTkLabel(self, text="random stacks", font=("Arial", 15)).grid(row=8, column=3, sticky="w")
        self.entry_amount_stacks = ctk.CTkEntry(self, placeholder_text="amount", width=60)
        self.entry_amount_stacks.grid(row=8, column=2, padx=5)

        self.cb_global_mask = ctk.CTkCheckBox(self, text="save global mask", font=("Arial", 15))
        self.cb_global_mask.grid(row=9, column=1, columnspan=3)

        self.btn_start_batch = ctk.CTkButton(
            self, text="START WEEKEND BATCH", corner_radius=20, font=("Arial", 25, "bold"),
            width=300, height=70, command=self.start_batch_processing, fg_color="#c0392b", hover_color="#e74c3c"
        )
        self.btn_start_batch.grid(row=10, column=1, columnspan=3, pady=10)
        
        # --- PROGRESS & ETA LABELS ---
        self.frame_progress = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_progress.grid(row=11, column=1, columnspan=3, pady=10)
        
        self.label_status = ctk.CTkLabel(self.frame_progress, text="Waiting to start...", font=("Arial", 18, "bold"), text_color="gray")
        self.label_status.pack(side="top")
        
        self.label_eta = ctk.CTkLabel(self.frame_progress, text="", font=("Arial", 16, "italic"), text_color="#f39c12")
        self.label_eta.pack(side="top", pady=5)

    # ---------------------------------------------------------
    # UI EVENT HANDLERS
    # ---------------------------------------------------------
    def select_raw_folder(self):
        folder_selected = filedialog.askdirectory(title="Select RAW Folder (Single dataset OR Parent folder)")
        if folder_selected:
            self.label_raw_info.configure(text=folder_selected.split("/")[-1], text_color="white")
            self.parent_raw_folder = folder_selected    

    def select_output_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Parent OUTPUT Folder")
        if folder_selected:
            self.label_output_info.configure(text=folder_selected.split("/")[-1], text_color="white")
            self.parent_output_folder = folder_selected   

    def select_parameter_file(self):
        file_selected = filedialog.askopenfilename(title="Select Parameter File", filetypes=[("JSON Files","*.json")])
        if file_selected:
            self.label_param_info.configure(text=file_selected.split("/")[-1], text_color="white")
            self.parameter_file = file_selected  

    def select_rf_file(self):
        file_selected = filedialog.askopenfilename(title="Select Random Forest Model", filetypes=[("Joblib Files","*.joblib")])
        if file_selected:
            self.label_rf_info.configure(text=file_selected.split("/")[-1], text_color="white")
            self.rf_model_file = file_selected

    # ---------------------------------------------------------
    # CORE BATCH LOGIC
    # ---------------------------------------------------------
    def start_batch_processing(self):
        if self.parent_raw_folder and self.parent_output_folder and self.parameter_file:
            
            # Disable UI to prevent interference during batch run
            self.btn_back.configure(state="disabled", fg_color="gray")
            self.btn_select_raw.configure(state="disabled", fg_color="gray")
            self.btn_select_output.configure(state="disabled", fg_color="gray")
            self.btn_select_param.configure(state="disabled", fg_color="gray")
            self.btn_select_rf.configure(state="disabled", fg_color="gray")
            self.btn_start_batch.configure(state="disabled", fg_color="gray")
            
            self.label_status.configure(text="Analyzing folder structure...", text_color="white")
            self.label_eta.configure(text="")

            def run_batch_thread():
                try:
                    raw_path = pathlib.Path(self.parent_raw_folder)
                    out_parent = pathlib.Path(self.parent_output_folder)
                    
                    # Accept one dataset folder or a parent containing dataset folders.
                    tif_files = list(raw_path.glob("*.tif"))
                    if len(tif_files) > 0:
                        datasets = [raw_path]
                        logging.info("Single Dataset Mode detected.")
                    else:
                        datasets = [f for f in raw_path.iterdir() if f.is_dir() and len(list(f.glob("*.tif"))) > 0]
                        logging.info(f"Parent Folder Mode detected. Found {len(datasets)} subfolders.")
                    
                    if not datasets:
                        self.controller.after(0, lambda: self.label_status.configure(text="Error: No TIF datasets found!", text_color="#e74c3c"))
                        return
                        
                    total_folders = len(datasets)
                    start_time = time.time()
                    successful_datasets = 0
                    failed_datasets = []

                    # Write a batch-specific parameter file that preserves the selected model path.
                    with open(self.parameter_file, "r", encoding="utf-8") as f:
                        batch_params = json.load(f)
                    
                    if self.rf_model_file:
                        batch_params["rf_model_path"] = self.rf_model_file
                        
                    safe_param_path = out_parent / "batch_execution_params.json"
                    with open(safe_param_path, "w", encoding="utf-8") as f:
                        json.dump(batch_params, f, indent=4)

                    # Schedule progress and grid updates on the GUI thread.
                    def cb_update_grid(positions):
                        self.controller.after(0, self.controller.init_progress_grid, positions)

                    def cb_update_stacks(progress, x, y):
                        self.controller.after(0, self.controller.update_cell_color, x, y, "#8A2BE2")
                        self.controller.after(0, lambda: self.label_status.configure(
                            text=f"Creating stacks: {progress:.0%}"
                        ))

                    def cb_update_border(progress, x, y, is_border):
                        color = "#7A0F0F" if is_border else "#C03C3C"
                        self.controller.after(0, self.controller.update_cell_color, x, y, color)
                        self.controller.after(0, lambda: self.label_status.configure(
                            text=f"Generating border masks: {progress:.0%}"
                        ))

                    def cb_update_flatfield(progress):
                        self.controller.after(0, lambda: self.label_status.configure(
                            text=f"Calculating flatfields: {progress:.0%}"
                        ))

                    def cb_update_correction(progress, x, y):
                        self.controller.after(0, self.controller.update_cell_color, x, y, "#238B3D")
                        self.controller.after(0, lambda: self.label_status.configure(
                            text=f"Applying flatfield correction: {progress:.0%}"
                        ))

                    def cb_update_status(message):
                        self.controller.after(0, lambda: self.label_status.configure(text=message))
                        
                    def cb_update_heatmap(gx, gy, amount, is_border=False, init_positions=None):
                        self.controller.after(0, self.controller.init_hydrate_heat_map, gx, gy, amount, is_border, init_positions)

                    # Process each dataset independently.
                    for idx, raw_subfolder in enumerate(datasets, start=1):
                        dataset_name = raw_subfolder.name
                        output_subfolder = out_parent / f"{dataset_name}_preprocessed"
                        output_subfolder.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # Preprocess the raw acquisition.
                            self.controller.after(0, lambda i=idx, n=dataset_name: self.label_status.configure(
                                text=f"[{i}/{total_folders}] Preprocessing: {n} ...", text_color="#f39c12"
                            ))
                            
                            gui_preprocessing_microscope(
                                data_folder=str(raw_subfolder),
                                output_folder=str(output_subfolder),
                                progress_stacks_cb=cb_update_stacks,
                                progress_border_cb=cb_update_border,
                                progress_flatfield_cb=cb_update_flatfield,
                                progress_correction_cb=cb_update_correction,
                                init_progress_grid_cb=cb_update_grid,
                                status_cb=cb_update_status
                            )

                            # Run detection with the batch-specific parameter file.
                            self.controller.after(0, lambda i=idx, n=dataset_name: self.label_status.configure(
                                text=f"[{i}/{total_folders}] Detecting Hydrates: {n} ...", text_color="#3498db"
                            ))

                            gui_hydrate_detection(
                                processed_data_dir=str(output_subfolder),
                                param_dir=str(safe_param_path),
                                rf_model_file=self.rf_model_file,
                                save_overlay=self.cb_overlay.get(),
                                save_random_overlay=self.cb_amount_overlay.get(),
                                amount_str=self.entry_amount_stacks.get(),
                                save_global_mask=self.cb_global_mask.get(),
                                update_heatmap_cb=cb_update_heatmap  
                            )
                            
                        except Exception as e:
                            failed_datasets.append(dataset_name)
                            logging.error(f"Error processing dataset '{dataset_name}'", exc_info=True)
                            print(f"\n[ERROR IN BATCH]: Dataset '{dataset_name}' failed: {e}. Skipping to next.\n")
                        else:
                            successful_datasets += 1

                        # Estimate remaining time from the average dataset duration.
                        elapsed_time = time.time() - start_time
                        avg_time_per_dataset = elapsed_time / idx
                        remaining_datasets = total_folders - idx
                        remaining_time_seconds = int(avg_time_per_dataset * remaining_datasets)
                        
                        eta_str = str(datetime.timedelta(seconds=remaining_time_seconds))
                        
                        self.controller.after(0, lambda eta=eta_str: self.label_eta.configure(
                            text=f"Estimated Time Remaining: {eta}"
                        ))

                    if failed_datasets:
                        status_text = (
                            f"Batch completed with failures: {successful_datasets}/{total_folders} succeeded; "
                            f"{len(failed_datasets)} failed."
                        )
                        status_color = "#e74c3c"
                    else:
                        status_text = f"Batch processing completed successfully ({successful_datasets}/{total_folders})."
                        status_color = "#2ecc71"

                    self.controller.after(0, lambda: [
                        self.label_status.configure(text=status_text, text_color=status_color),
                        self.label_eta.configure(text="Batch complete; review the status for any failures.")
                    ])

                except Exception as e:
                    logging.error("Fatal error in Multiple Detection Thread.", exc_info=True)
                    self.controller.after(0, lambda: self.label_status.configure(text="Fatal Batch Error occurred! Check logs.", text_color="#e74c3c"))
                    print(f"\n[FATAL BATCH ERROR]: {e}\n")
                finally:
                    # Re-enable controls after the batch completes or fails.
                    self.controller.after(0, lambda: [
                        self.btn_back.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_raw.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_output.configure(state="normal", fg_color="#27ae60"),
                        self.btn_select_param.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_rf.configure(state="normal", fg_color="#9b59b6"),
                        self.btn_start_batch.configure(state="normal", fg_color="#c0392b"),
                        self.label_raw_info.configure(text="no folder selected", text_color="gray"),
                        self.label_output_info.configure(text="no folder selected", text_color="gray"),
                        self.label_param_info.configure(text="no file selected", text_color="gray"),
                        self.label_rf_info.configure(text="no model selected (optional)", text_color="gray")
                    ])
                    self.parent_raw_folder = None
                    self.parent_output_folder = None
                    self.parameter_file = None
                    self.rf_model_file = None

            threading.Thread(target=run_batch_thread, daemon=True).start()
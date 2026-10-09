import customtkinter as ctk
import threading
from tkinter import filedialog
import logging

from core_backend.hydrate_segmentation.hydrate_detection import gui_hydrate_detection

class HydrateDetectionFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.preprocessed_data_folder = None
        self.parameter_file = None
        self.rf_model_file = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=4)
        self.grid_columnconfigure(4, weight=1)

        for i in range(11):
            self.grid_rowconfigure(i, weight=1)

        ctk.CTkLabel(self, text="Detection", font=("Arial", 40, "bold")).grid(row=0, column=1, columnspan=3, sticky="s")

        self.btn_back = ctk.CTkButton(
            self, text="back", command=lambda: self.controller.show_frame("StartFrame"),
            corner_radius=40, font=("Arial", 20), width=50, height=30
        )
        self.btn_back.grid(row=0, column=0, sticky="wn", padx=10, pady=10)

        ctk.CTkLabel(self, text=self.controller.VERSION, font=("Arial", 10), text_color="gray").grid(row=0, column=4, sticky="ne", padx=10, pady=10)

        self.btn_select_data = ctk.CTkButton(
            self, text="select folder with\npreprocessed data", command=self.select_preprocessed_folder,
            corner_radius=20, font=("Arial", 15), width=120, height=60
        )
        self.btn_select_data.grid(row=2, column=1)

        self.label_preprocessed_data_info = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_preprocessed_data_info.grid(row=2, column=3, sticky="nsew")

        self.btn_select_rf_model = ctk.CTkButton(
            self, text="select RF Model\n(.joblib)", corner_radius=20, font=("Arial", 15),
            width=120, height=60, command=self.select_rf_model
        )
        self.btn_select_rf_model.grid(row=3, column=1)

        self.label_rf_model_info = ctk.CTkLabel(self, text="no model selected (optional)", font=("Arial", 15), text_color="gray")
        self.label_rf_model_info.grid(row=3, column=3, sticky="nsew")

        self.btn_select_parameter_file = ctk.CTkButton(
            self, text="select\nparameter file", corner_radius=20, font=("Arial", 15),
            width=120, height=60, command=self.select_parameter_file
        )
        self.btn_select_parameter_file.grid(row=4, column=1)

        self.label_parameter_file_info = ctk.CTkLabel(self, text="no file selected", font=("Arial", 15), text_color="gray")
        self.label_parameter_file_info.grid(row=4, column=3, sticky="nsew")

        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=1, column=0, columnspan=5)
        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=5, column=0, columnspan=5)

        self.cb_overlay = ctk.CTkCheckBox(self, text="save all overlay images", font=("Arial", 15))
        self.cb_overlay.grid(row=6, column=1, columnspan=3)

        self.cb_amount_overlay = ctk.CTkCheckBox(self, text="save overlay from", font=("Arial", 15))
        self.cb_amount_overlay.grid(row=7, column=1, sticky="e")
        ctk.CTkLabel(self, text="random stacks", font=("Arial", 15)).grid(row=7, column=3, sticky="w")
        self.entry_amount_stacks = ctk.CTkEntry(self, placeholder_text="amount", width=60)
        self.entry_amount_stacks.grid(row=7, column=2, padx=5)

        self.cb_global_mask = ctk.CTkCheckBox(self, text="save global mask", font=("Arial", 15))
        self.cb_global_mask.grid(row=8, column=1, columnspan=3)

        self.btn_start_detection = ctk.CTkButton(
            self, text="start\ndetection", corner_radius=20, font=("Arial", 25),
            width=120, height=60, command=self.start_detection
        )
        self.btn_start_detection.grid(row=9, column=1, columnspan=3)

    def select_preprocessed_folder(self):
        preprocessed_folder_selected = filedialog.askdirectory(title="select folder with preprocessed data")
        if preprocessed_folder_selected:
            self.label_preprocessed_data_info.configure(text=preprocessed_folder_selected.split("/")[-1], text_color="white")
            self.preprocessed_data_folder = preprocessed_folder_selected    

    def select_rf_model(self):
        rf_model_selected = filedialog.askopenfilename(title="select Trained RF Model", filetypes=[("Joblib Files","*.joblib")])
        if rf_model_selected:
            self.label_rf_model_info.configure(text=rf_model_selected.split("/")[-1], text_color="white")
            self.rf_model_file = rf_model_selected  

    def select_parameter_file(self):
        parameter_file_selected = filedialog.askopenfilename(title="select parameter file", filetypes=[("JSON-Dateien","*.json")])
        if parameter_file_selected:
            self.label_parameter_file_info.configure(text=parameter_file_selected.split("/")[-1], text_color="white")
            self.parameter_file = parameter_file_selected  

    def start_detection(self):
        if self.preprocessed_data_folder and self.parameter_file:
            self.btn_back.configure(state="disabled", fg_color="gray")
            self.btn_select_data.configure(state="disabled", fg_color="gray")
            self.btn_select_rf_model.configure(state="disabled", fg_color="gray")
            self.btn_select_parameter_file.configure(state="disabled", fg_color="gray")
            self.btn_start_detection.configure(state="disabled", fg_color="gray")

            def thread_safe_heatmap_update(gx, gy, hydrate_amount, is_border=False, init_positions=None):
                main_app = self.winfo_toplevel()
                self.after(
                    0, 
                    lambda: main_app.init_hydrate_heat_map(
                        gx, gy, hydrate_amount, is_border=is_border, init_positions=init_positions
                    )
                )

            def run_hydrate_detection():
                try:
                    gui_hydrate_detection(
                        self.preprocessed_data_folder,
                        self.parameter_file,
                        self.rf_model_file,
                        self.cb_overlay.get(),
                        self.cb_amount_overlay.get(),
                        self.entry_amount_stacks.get(),
                        self.cb_global_mask.get(),
                        thread_safe_heatmap_update  
                    )
                except Exception as e:
                    logging.error("Fehler im Detection-Thread aufgetreten.", exc_info=True)
                    print(f"\n[CRITICAL ERROR IN DETECTION THREAD]: {e}\n") 
                
                finally:
                    self.controller.after(0, lambda: [
                        self.btn_back.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_data.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_rf_model.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_parameter_file.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_start_detection.configure(state="normal", fg_color="#1F6AA5"),
                        self.label_preprocessed_data_info.configure(text="no folder selected", text_color="gray"),
                        self.label_rf_model_info.configure(text="no model selected (optional)", text_color="gray"),
                        self.label_parameter_file_info.configure(text="no file selected", text_color="gray")
                    ])
                    self.preprocessed_data_folder = None
                    self.parameter_file = None
                    self.rf_model_file = None

            threading.Thread(target=run_hydrate_detection, daemon=True).start()
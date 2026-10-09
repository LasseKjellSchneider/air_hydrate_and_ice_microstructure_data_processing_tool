import customtkinter as ctk
import threading
from tkinter import filedialog
import logging

# Import preprocessing operations from the backend.
from core_backend.hydrate_segmentation.hydrate_preprocessing import (
    gui_information_about_input_data, 
    gui_preprocessing_microscope
)

class HydratePreprocessingFrame(ctk.CTkFrame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.data_folder = None
        self.output_folder = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=4)
        self.grid_columnconfigure(4, weight=1)

        for i in range(14):
            self.grid_rowconfigure(i, weight=1)

        ctk.CTkLabel(self, text="Preprocessing", font=("Arial", 40, "bold")).grid(row=0, column=1, columnspan=3)

        self.btn_back = ctk.CTkButton(
            self, text="back", command=lambda: self.controller.show_frame("StartFrame"),
            corner_radius=40, font=("Arial", 20), width=50, height=30
        )
        self.btn_back.grid(row=0, column=0, sticky="e")

        ctk.CTkLabel(self, text=self.controller.VERSION, font=("Arial", 10), text_color="gray").grid(row=0, column=4)

        self.btn_select_data = ctk.CTkButton(
            self, text="select\ndata folder", command=self.select_data_folder,
            corner_radius=20, font=("Arial", 30), width=120, height=60
        )
        self.btn_select_data.grid(row=2, column=1)

        self.label_data_info = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_data_info.grid(row=2, column=3, sticky="nsew")

        self.btn_select_output = ctk.CTkButton(
            self, text="select\noutput folder", corner_radius=20, font=("Arial", 30),
            width=120, height=60, command=self.select_output_folder
        )
        self.btn_select_output.grid(row=4, column=1)

        self.label_output_info = ctk.CTkLabel(self, text="no folder selected", font=("Arial", 15), text_color="gray")
        self.label_output_info.grid(row=4, column=3, sticky="nsew")

        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=1, column=0, columnspan=5)
        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=5, column=0, columnspan=5)

        self.btn_start_preprocessing = ctk.CTkButton(
            self, text="start\npreprocessing", corner_radius=20,
            command=self.start_preprocessing, font=("Arial", 30), width=120, height=60
        )
        self.btn_start_preprocessing.grid(row=6, column=1, rowspan=8)

        ctk.CTkLabel(self, text="creating stacks", font=("Arial", 15)).grid(row=6, column=3, sticky="s")
        self.progress_stacks = ctk.CTkProgressBar(self, width=150, height=20, corner_radius=10, progress_color="#8A2BE2")
        self.progress_stacks.set(0)
        self.progress_stacks.grid(row=7, column=3, sticky="n")

        ctk.CTkLabel(self, text="creating border mask", font=("Arial", 15)).grid(row=8, column=3, sticky="s")
        self.progress_border_mask = ctk.CTkProgressBar(self, width=150, height=20, corner_radius=10, progress_color="#C03C3C")
        self.progress_border_mask.set(0)
        self.progress_border_mask.grid(row=9, column=3, sticky="n")

        ctk.CTkLabel(self, text="creating flatfield mask", font=("Arial", 15)).grid(row=10, column=3, sticky="s")
        self.progress_flatfield_mask = ctk.CTkProgressBar(self, width=150, height=20, corner_radius=10, progress_color="#F0E68C")
        self.progress_flatfield_mask.set(0)
        self.progress_flatfield_mask.grid(row=11, column=3, sticky="n")

        ctk.CTkLabel(self, text="applying flatfield correction", font=("Arial", 15)).grid(row=12, column=3, sticky="s")
        self.progress_flatfield_correction = ctk.CTkProgressBar(self, width=150, height=20, corner_radius=10, progress_color="#238B3D")
        self.progress_flatfield_correction.set(0)
        self.progress_flatfield_correction.grid(row=13, column=3, sticky="n")

        self.label_processing_status = ctk.CTkLabel(self, text="ready to start", font=("Arial", 15), text_color="gray")
        self.label_processing_status.grid(row=12, column=1)

    def select_data_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Data Folder")
        if folder_selected:
            information_about_input_data = gui_information_about_input_data(folder_selected)
            self.label_data_info.configure(text=information_about_input_data, text_color="white")
            self.data_folder = folder_selected

    def select_output_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Output Folder")
        if folder_selected:
            self.label_output_info.configure(text=folder_selected.split("/")[-1], text_color="white")
            self.output_folder = folder_selected

    def start_preprocessing(self):
        if self.data_folder and self.output_folder:
            self.btn_back.configure(state="disabled", fg_color="gray")
            self.btn_select_data.configure(state="disabled", fg_color="gray")
            self.btn_select_output.configure(state="disabled", fg_color="gray")
            self.btn_start_preprocessing.configure(state="disabled", fg_color="gray")
            self.progress_stacks.set(0)
            self.progress_border_mask.set(0)
            self.progress_flatfield_mask.set(0)
            self.progress_flatfield_correction.set(0)
            self.label_processing_status.configure(text="Initializing...", text_color="white")

            # Queue status updates on the GUI thread.
            def safe_status(msg):
                self.controller.after(0, lambda m=msg: self.label_processing_status.configure(text=m))

            # Initialize the progress grid only when the controller provides it.
            def safe_init_grid(positions):
                if hasattr(self.controller, "init_progress_grid"):
                    self.controller.after(0, self.controller.init_progress_grid, positions)
                
            def safe_stacks(val, x, y):
                def update():
                    self.progress_stacks.set(val)
                    if hasattr(self.controller, "update_cell_color"):
                        self.controller.update_cell_color(x, y, "#8A2BE2")
                self.controller.after(0, update)
                
            def safe_border(val, x, y, border):
                color = "#7A0F0F" if border else "#C03C3C"
                def update():
                    self.progress_border_mask.set(val)
                    if hasattr(self.controller, "update_cell_color"):
                        self.controller.update_cell_color(x, y, color)
                self.controller.after(0, update)
                
            def safe_flatfield(val):
                self.controller.after(0, lambda: self.progress_flatfield_mask.set(val))
                
            def safe_correction(val, x, y):
                def update():
                    self.progress_flatfield_correction.set(val)
                    if hasattr(self.controller, "update_cell_color"):
                        self.controller.update_cell_color(x, y, "#238B3D")
                self.controller.after(0, update)

            def run_preprocessing():
                try:
                    result_dir = gui_preprocessing_microscope(
                        self.data_folder,
                        self.output_folder,
                        safe_stacks, 
                        safe_border,
                        safe_flatfield,
                        safe_correction,
                        safe_init_grid,
                        safe_status
                    )

                    self.controller.after(0, lambda: [
                        self.btn_back.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_data.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_select_output.configure(state="normal", fg_color="#1F6AA5"),
                        self.btn_start_preprocessing.configure(state="normal", fg_color="#1F6AA5"),
                        self.label_processing_status.configure(text="Processing completed!", text_color="#00FF00"),
                        self.label_data_info.configure(text="no folder selected", text_color="gray"),
                        self.label_output_info.configure(text="no folder selected", text_color="gray"),
                        self.controller.show_image_plot(result_dir, "preprocessed image")
                    ])
                    self.data_folder = None
                    self.output_folder = None
                except Exception as e:
                    logging.error("Fehler im Preprocessing-Thread", exc_info=True)
                    safe_status("Error occurred! Check terminal.")

            threading.Thread(target=run_preprocessing, daemon=True).start()
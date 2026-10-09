import customtkinter as ctk
from tkinter import filedialog
import threading
from pathlib import Path
import logging
import sys
from PIL import Image

file_path = Path(__file__).resolve()
for parent in file_path.parents:
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
        
from core_backend.grain_boundary_segmentation.grain_boundary_preprocessing import process_and_save_borders, process_single_image

class GrainBoundaryPreprocessingFrame(ctk.CTkFrame):
    """
    GUI Frame for the Grain Boundary Preprocessing step.
    Includes initial batch processing and a review tool for manual adjustments.
    """
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        
        # State variables
        self.data_folder = None
        self.image_list = []
        self.current_idx = 0
        
        # Theme colors
        self.btn_color = "#D67040"
        self.btn_hover = "#804326"
        self.bg_color = "#2E241F"

        self.setup_start_ui()

    def clear_ui(self):
        """Removes all current widgets from the frame."""
        for widget in self.winfo_children():
            widget.grid_forget()

    def setup_start_ui(self):
        """Sets up the initial View (Folder selection and Batch processing)."""
        self.clear_ui()
        
        # Reset grid weights for Start UI
        for i in range(5):
            self.grid_columnconfigure(i, weight=1 if i%2==0 else 4)
        for i in range(10):
            self.grid_rowconfigure(i, weight=1)

        ctk.CTkLabel(self, text="Grain Boundary Preprocessing", font=("Arial", 35, "bold")).grid(row=0, column=1, columnspan=3)

        self.btn_back = ctk.CTkButton(
            self, text="Back", command=lambda: self.controller.show_frame("StartFrame"),
            corner_radius=20, font=("Arial", 16), width=70, height=35
        )
        self.btn_back.grid(row=0, column=0, sticky="wn", padx=20, pady=20)

        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=1, column=0, columnspan=5)

        self.btn_select_data = ctk.CTkButton(
            self, text="Select Folder\nwith Images", command=self.select_data_folder,
            corner_radius=20, font=("Arial", 20), width=200, height=60,
            fg_color=self.btn_color, hover_color=self.btn_hover
        )
        self.btn_select_data.grid(row=3, column=1)

        self.label_data_info = ctk.CTkLabel(self, text="No folder selected", font=("Arial", 15), text_color="gray")
        self.label_data_info.grid(row=3, column=3, sticky="w")

        ctk.CTkLabel(self, text="-"*150, font=("Arial", 15), text_color="gray").grid(row=4, column=0, columnspan=5)

        self.btn_start = ctk.CTkButton(
            self, text="Start Batch Processing", corner_radius=20,
            command=self.start_batch_processing, font=("Arial", 25, "bold"), width=300, height=70,
            fg_color=self.btn_color, hover_color=self.btn_hover
        )
        self.btn_start.grid(row=6, column=1, columnspan=3)

        self.label_status = ctk.CTkLabel(self, text="Waiting for input...", font=("Arial", 16, "italic"), text_color="gray")
        self.label_status.grid(row=7, column=1, columnspan=3)
        
        self.progress_bar = ctk.CTkProgressBar(self, width=300, height=15, corner_radius=10, progress_color=self.btn_color)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=8, column=1, columnspan=3)
        self.progress_bar.grid_remove()

    def select_data_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Folder with Raw Images")
        if folder_selected:
            self.data_folder = folder_selected
            short_path = ".../" + "/".join(self.data_folder.split("/")[-2:])
            self.label_data_info.configure(text=short_path, text_color="white")
            self.label_status.configure(text="Ready to process.", text_color="#2ecc71")

    def start_batch_processing(self):
        if not self.data_folder:
            self.label_status.configure(text="Please select a folder first!", text_color="#e74c3c")
            return

        self.btn_back.configure(state="disabled", fg_color="gray")
        self.btn_select_data.configure(state="disabled", fg_color="gray")
        self.btn_start.configure(state="disabled", fg_color="gray")
        
        self.label_status.configure(text="Batch processing images... Please wait.", text_color="#f39c12")
        self.progress_bar.grid()
        self.progress_bar.set(0.0)

        threading.Thread(target=self._run_batch_thread, daemon=True).start()

    def _run_batch_thread(self):
        def status_cb(msg):
            self.controller.after(0, lambda: self.label_status.configure(text=msg, text_color="#f39c12"))
        def progress_cb(val):
            self.controller.after(0, lambda: self.progress_bar.set(val))

        try:
            process_and_save_borders(self.data_folder, status_callback=status_cb, progress_callback=progress_cb)
            self.controller.after(0, self._on_batch_finished, True)
        except Exception as e:
            logging.error(f"Error during batch preprocessing: {e}", exc_info=True)
            self.controller.after(0, self._on_batch_finished, False, str(e))

    def _on_batch_finished(self, success, error_msg=""):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()

        if success:
            self.label_status.configure(text="Batch complete! Loading Review Mode...", text_color="#2ecc71")
            self.controller.after(1000, self.start_review_mode) # Switch to review UI after 1 sec
        else:
            self.btn_back.configure(state="normal", fg_color="#1F6AA5")
            self.btn_select_data.configure(state="normal", fg_color=self.btn_color)
            self.btn_start.configure(state="normal", fg_color=self.btn_color)
            self.label_status.configure(text=f"Failed: {error_msg}", text_color="#e74c3c")

    # ---------------------------------------------------------
    # REVIEW MODE
    # ---------------------------------------------------------
    def start_review_mode(self):
        """Gathers images and sets up the Review GUI."""
        raw_images = list(Path(self.data_folder).rglob("*.tif"))
        self.image_list = [p for p in raw_images if not p.name.endswith(("_mask.tif", "_mask.png", "_overlay.png"))]
        self.image_list.sort()
        
        if not self.image_list:
            self.setup_start_ui()
            self.label_status.configure(text="No valid images found for review.", text_color="#e74c3c")
            return
            
        self.current_idx = 0
        self.setup_review_ui()
        self.load_current_image()

    def setup_review_ui(self):
        """Builds the layout for manual threshold adjusting."""
        self.clear_ui()
        
        # Custom grid layout for review
        self.grid_columnconfigure(0, weight=3) # Image Area
        self.grid_columnconfigure(1, weight=1) # Control Area
        self.grid_rowconfigure(0, weight=1)    # Title
        self.grid_rowconfigure(1, weight=8)    # Content
        self.grid_rowconfigure(2, weight=1)    # Bottom Buttons
        
        # Title Label
        self.lbl_review_title = ctk.CTkLabel(self, text="", font=("Arial", 25, "bold"))
        self.lbl_review_title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Image Display Label
        self.img_label = ctk.CTkLabel(self, text="Loading Image...")
        self.img_label.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Controls Frame (Right side)
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        # Controls setup
        self.lbl_threshold_val = ctk.CTkLabel(self.controls_frame, text="Threshold: 120", font=("Arial", 18))
        self.lbl_threshold_val.pack(pady=(40, 10))
        
        self.slider_threshold = ctk.CTkSlider(self.controls_frame, from_=0, to=255, command=self.on_slider_change)
        self.slider_threshold.set(120)
        self.slider_threshold.pack(pady=10)
        
        self.cb_keep_two = ctk.CTkCheckBox(self.controls_frame, text="Keep 2 largest areas", font=("Arial", 16))
        self.cb_keep_two.pack(pady=30)
        
        self.btn_update = ctk.CTkButton(
            self.controls_frame, text="Update Overlay", command=self.update_current_overlay,
            font=("Arial", 16, "bold"), height=40, fg_color="#1F6AA5", hover_color="#144870"
        )
        self.btn_update.pack(pady=20)
        
        self.lbl_update_status = ctk.CTkLabel(self.controls_frame, text="", text_color="gray", font=("Arial", 14))
        self.lbl_update_status.pack(pady=5)
        
        # Next Button (Bottom Right)
        self.btn_next = ctk.CTkButton(
            self, text="Weiter", command=self.next_image,
            font=("Arial", 22, "bold"), height=50, fg_color=self.btn_color, hover_color=self.btn_hover
        )
        self.btn_next.grid(row=2, column=1, padx=20, pady=20, sticky="ew")

    def on_slider_change(self, value):
        self.lbl_threshold_val.configure(text=f"Threshold: {int(value)}")

    def load_current_image(self):
        """Loads the overlay of the current index and resets UI controls to default."""
        img_path = self.image_list[self.current_idx]
        overlay_path = img_path.parent / (img_path.stem + "_overlay.png")
        
        self.lbl_review_title.configure(text=f"Reviewing {self.current_idx + 1} of {len(self.image_list)}: {img_path.name}")
        
        # Reset controls to batch defaults to reflect current image state
        self.slider_threshold.set(120)
        self.lbl_threshold_val.configure(text="Threshold: 120")
        self.cb_keep_two.deselect()
        self.lbl_update_status.configure(text="")
        
        if overlay_path.exists():
            img = Image.open(overlay_path)
            
            # Dynamic resizing to fit nicely into the UI
            max_size = 650
            w, h = img.size
            if w > h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
                
            self.ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
            self.img_label.configure(image=self.ctk_img, text="")
        else:
            self.img_label.configure(image=None, text="Overlay not found!")

    def update_current_overlay(self):
        """Re-runs the algorithm for the current image in a thread."""
        self.btn_update.configure(state="disabled")
        self.btn_next.configure(state="disabled")
        self.lbl_update_status.configure(text="Updating...", text_color="#f39c12")
        
        thresh = int(self.slider_threshold.get())
        keep_two = bool(self.cb_keep_two.get() == 1)
        img_path = self.image_list[self.current_idx]
        
        def task():
            try:
                process_single_image(img_path, cutoff_threshold=thresh, keep_two_largest=keep_two)
                self.controller.after(0, self.on_update_done, True)
            except Exception as e:
                logging.error(f"Update failed: {e}")
                self.controller.after(0, self.on_update_done, False)
                
        threading.Thread(target=task, daemon=True).start()

    def on_update_done(self, success):
        self.btn_update.configure(state="normal")
        self.btn_next.configure(state="normal")
        
        if success:
            self.lbl_update_status.configure(text="Updated Successfully!", text_color="#2ecc71")
            
            # Reload the newly generated image. We preserve slider values so user can see what they set!
            img_path = self.image_list[self.current_idx]
            overlay_path = img_path.parent / (img_path.stem + "_overlay.png")
            
            if overlay_path.exists():
                img = Image.open(overlay_path)
                max_size = 650
                w, h = img.size
                new_w = max_size if w > h else int(w * (max_size / h))
                new_h = int(h * (max_size / w)) if w > h else max_size
                
                self.ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
                self.img_label.configure(image=self.ctk_img)
        else:
            self.lbl_update_status.configure(text="Error updating image", text_color="#e74c3c")

    def next_image(self):
        """Moves to the next image or finishes the review."""
        self.current_idx += 1
        if self.current_idx < len(self.image_list):
            self.load_current_image()
        else:
            # Done with all images
            self.setup_start_ui()
            self.label_status.configure(text="Review Session Finished! All images processed.", text_color="#2ecc71")
            self.data_folder = None
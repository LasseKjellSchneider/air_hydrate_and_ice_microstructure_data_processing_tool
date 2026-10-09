import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _version import VERSION
except ImportError:
    VERSION = "V 2.0 (Dev)"

import multiprocessing
import pathlib
import json
import logging
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import customtkinter as ctk

# ==============================================================================
# GUI MODULE IMPORTS
# ==============================================================================
from gui_modules.hydrate_segmentation.hydrate_pre import HydratePreprocessingFrame
from gui_modules.hydrate_segmentation.hydrate_det import HydrateDetectionFrame
from gui_modules.hydrate_segmentation.hydrate_cal import HydrateCalibrationFrame
from gui_modules.hydrate_segmentation.hydrate_mult_det import HydrateMultipleDetectionFrame

from gui_modules.grain_boundary_gui.grain_boundary_pre import GrainBoundaryPreprocessingFrame
from gui_modules.grain_boundary_gui.grain_boundary_det import GrainBoundaryDetectionFrame
from gui_modules.grain_boundary_gui.grain_boundary_cal import GrainBoundaryCalibrationFrame
from gui_modules.grain_boundary_gui.grain_boundary_mult_det import GrainBoundaryMultipleDetectionFrame
from gui_modules.grain_boundary_gui.grain_boundary_tuner import GrainBoundaryTunerFrame
from gui_modules.grain_boundary_gui.grain_boundary_ws_tuner import GrainBoundaryWatershedTunerFrame

from gui_modules.data_merge.layer_alignment import LayerAlignmentFrame
from gui_modules.data_merge.fa_analysis import FAAnalysisFrame
from gui_modules.data_merge.reconstruction_3d_gui import ModelReconstructionFrame

from core_backend.hydrate_segmentation.optimization_visualizer import OptimizationVisualizer

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("dark-blue")

class MainApp(ctk.CTk):
    """
    Main application controller.
    Manages top-level window layout, global canvases, and transitions between workflow frames.
    """
    VERSION = VERSION

    def __init__(self):
        super().__init__()

        self.title("Ice Core Analysis Pipeline")
        self.geometry("1700x800")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=7, uniform="group1")
        self.grid_columnconfigure(1, weight=10, uniform="group1")

        # Configure the workflow navigation panel.
        self.menu_container = ctk.CTkFrame(self)
        self.menu_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.menu_container.grid_propagate(False)  
        self.menu_container.grid_rowconfigure(0, weight=1)
        self.menu_container.grid_columnconfigure(0, weight=1)

        # Configure the shared image and plot area.
        self.image_plot_container = ctk.CTkFrame(self)
        self.image_plot_container.grid_columnconfigure(0, weight=1)
        
        self.canvas = ctk.CTkCanvas(self.image_plot_container, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.label_image_plot_info = ctk.CTkLabel(
            self.image_plot_container, text="Image/Plot Info", font=("Arial", 15), text_color="gray"
        )
        self.label_image_plot_info.pack(pady=5, fill="x")

        self.opt_visualizer = None
        self.opt_canvas_widget = None

        # Create each workflow view once so it can be raised on demand.
        self.frames = {}
        
        frame_classes = (
            StartFrame, 
            HydratePreprocessingFrame, 
            HydrateDetectionFrame, 
            HydrateCalibrationFrame, 
            HydrateMultipleDetectionFrame,
            GrainBoundaryPreprocessingFrame, 
            GrainBoundaryDetectionFrame, 
            GrainBoundaryCalibrationFrame, 
            GrainBoundaryMultipleDetectionFrame,
            GrainBoundaryTunerFrame,
            GrainBoundaryWatershedTunerFrame,
            LayerAlignmentFrame,
            FAAnalysisFrame,
            ModelReconstructionFrame
        )

        for FrameClass in frame_classes:
            page_name = FrameClass.__name__
            frame = FrameClass(master=self.menu_container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartFrame")

    def show_frame(self, page_name: str):
        full_width_frames = ["StartFrame", "GrainBoundaryCalibrationFrame", "LayerAlignmentFrame"]
        
        if page_name in full_width_frames:
            self.image_plot_container.grid_remove()
            self.menu_container.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        else:
            self.menu_container.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=10, pady=10)
            self.image_plot_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        frame = self.frames[page_name]
        frame.tkraise()

    def _clear_right_container(self):
        """Clears all custom plots, canvases, and widgets from the right container."""
        # Clear the shared canvas before removing transient plot widgets.
        self.canvas.pack_forget()
        self.canvas.delete("all")
        
        if self.opt_canvas_widget:
            self.opt_canvas_widget.destroy()
            self.opt_canvas_widget = None

        # Preserve the frame's internal canvas and the persistent information label.
        internal_canvas = getattr(self.image_plot_container, "_canvas", None)
        for child in list(self.image_plot_container.winfo_children()):
            if child == internal_canvas or child == self.canvas or child == self.label_image_plot_info:
                continue
            try:
                child.destroy()
            except Exception:
                pass


    # ==============================================================================
    # 2D OPTIMIZATION DASHBOARD (FUZZY LOGIC)
    # ==============================================================================
        # 2D optimization dashboard
    def setup_optimization_dashboard(self):
        """Prepares and clears the right container before mounting the 2D optimization visualizer."""
        self._clear_right_container()
        self.label_image_plot_info.configure(text="2D Live Calibration Dashboard (Fuzzy Logic)")
        self.label_image_plot_info.pack(pady=5, fill="x")
        
        self.opt_visualizer = OptimizationVisualizer(self.image_plot_container)
        self.opt_visualizer.init_plot()
        self.opt_canvas_widget = self.opt_visualizer.widget

    def update_optimization_dashboard(self, gen, data_dict, params, path_features=None, stats_matrix=None, eta_str=""):
        """Dispatches real-time training telemetry to the optimization visualizer widget."""
        if hasattr(self, 'opt_visualizer') and self.opt_visualizer:
            self.opt_visualizer.update_plot(
                gen, data_dict, params, 
                path_features=path_features, 
                stats_matrix=stats_matrix, 
                eta_str=eta_str
            )
            
    # ==============================================================================
    # 3D FOCUS OPTIMIZATION DASHBOARD
    # ==============================================================================
        # 3D focus optimization dashboard
    def init_3d_plot(self):
        """Initializes a matplotlib canvas for 3D focus optimization convergence tracking."""
        self._clear_right_container()
        self.label_image_plot_info.configure(text="3D Z-Focus Optimization Progress")
        self.label_image_plot_info.pack(pady=5, fill="x")
        
        self.live_fig, self.live_ax = plt.subplots(figsize=(6, 4))
        self.live_fig.patch.set_facecolor('#1a1a1a')
        self.live_ax.set_facecolor('#2b2b2b')
        self.live_ax.tick_params(colors='white', labelcolor='white')
        self.live_ax.xaxis.label.set_color('white')
        self.live_ax.yaxis.label.set_color('white')
        self.live_ax.title.set_color('white')
        for spine in self.live_ax.spines.values():
            spine.set_color('#555555')

        self.live_ax.set_title("3D Focus Optimization Error", color='white')
        self.live_ax.set_xlabel("Generation", color='white')
        self.live_ax.set_ylabel("Total Z-Error Penalty", color='white')
        self.live_ax.set_ylim(bottom=0)
        
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=self.image_plot_container)
        self.opt_canvas_widget = self.live_canvas.get_tk_widget()
        self.opt_canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)
        self.live_fig.canvas.draw()

    def update_3d_plot(self, gen_list, error_list):
        """Refreshes the convergence curve during 3D Z-focus optimization."""
        if not hasattr(self, 'live_ax') or not self.live_ax:
            return
        self.live_ax.clear()
        self.live_ax.set_facecolor('#2b2b2b')
        self.live_ax.plot(gen_list, error_list, color='#e74c3c', marker='o', linewidth=2)
        self.live_ax.set_title("3D Focus Optimization Progress", color='white')
        self.live_ax.set_xlabel("Generation", color='white')
        self.live_ax.set_ylabel("Total Z-Error Penalty", color='white')
        self.live_ax.grid(True, linestyle='--', alpha=0.5, color='#555555')
        self.live_canvas.draw_idle()

    # ==============================================================================
    # GRID & HEATMAP RENDERING
    # ==============================================================================
        # Progress grid and heatmap rendering
    def init_progress_grid(self, positions_list):
        """Initializes a tile matrix representing physical sample fields of view."""
        self._clear_right_container()
        self.label_image_plot_info.pack_forget()
        self.canvas.pack(fill="both", expand=True)

        max_x = max([p["gridX"] for p in positions_list]) if positions_list else 1
        max_y = max([p["gridY"] for p in positions_list]) if positions_list else 1
        
        canvas_width = 950  
        cell_size = max(10, min(22, canvas_width // (max_x + 1)))
        padding = 2 

        for pos in positions_list:
            gx, gy = pos["gridX"], pos["gridY"]
            x1 = gx * (cell_size + padding) + 20
            y1 = gy * (cell_size + padding) + 20
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            tag_name = f"cell_{gx}_{gy}"
            
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="#222222", outline="#111111", width=1, tags=tag_name        
            )
            
        self.canvas.update_idletasks()

    def update_cell_color(self, gx: int, gy: int, color: str):
        """Updates the fill color of a specific tile in the progress overview canvas."""
        tag_name = f"cell_{gx}_{gy}"
        self.canvas.itemconfig(tag_name, fill=color)
        self.canvas.update_idletasks()

    def init_hydrate_heat_map(self, gx: int, gy: int, hydrate_amount: int, is_border: bool = False, init_positions: list = None):
        """Renders or updates the spatial density map of detected hydrate inclusions."""
        if init_positions is not None:
            self.init_progress_grid(init_positions)
            return

        tag_name = f"cell_{gx}_{gy}"
        
        if is_border and hydrate_amount == 0:
            hex_color = "#304058"
        else:
            cmap = plt.get_cmap('plasma')
            norm = mcolors.Normalize(vmin=0, vmax=30)
            rgba = cmap(norm(hydrate_amount))
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            
            if is_border:
                blend_factor = 0.6
                r = int(r * (1 - blend_factor) + 30 * blend_factor)
                g = int(g * (1 - blend_factor) + 30 * blend_factor)
                b = int(b * (1 - blend_factor) + 30 * blend_factor)

            hex_color = f"#{r:02x}{g:02x}{b:02x}"

        self.canvas.itemconfig(tag_name, fill=hex_color)
        self.canvas.update_idletasks()

    # ==============================================================================
    # IMAGE VIEWER
    # ==============================================================================
    def show_image_plot(self, image_path: str, title: str = ""):
        """
        Zeigt ein Bild oder einen Plot im rechten image_plot_container an.
        Ignoriert das interne _canvas von CustomTkinter, um 'isn't packed' Fehler zu verhindern.
        """
        # Show the plot container and clear any content from the previous view.
        self.image_plot_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self._clear_right_container()
        
        self.label_image_plot_info.configure(text=title)
        self.label_image_plot_info.pack(pady=5, fill="x")

        # Load the image and prepare it for display with Matplotlib.
        try:
            img = Image.open(image_path)
            
            fig = Figure(figsize=(8, 8), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            fig.patch.set_facecolor("#1A1311")
            ax.set_facecolor("#1A1311")
            fig.tight_layout()

            # Embed the Matplotlib canvas in the plot container.
            canvas = FigureCanvasTkAgg(fig, master=self.image_plot_container)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side="top", fill="both", expand=True)

            # Add navigation controls for zooming and panning.
            toolbar = NavigationToolbar2Tk(canvas, self.image_plot_container)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")

        except Exception as e:
            print(f"[show_image_plot Error] {e}")
            raise e

# ==============================================================================
# DASHBOARD (START FRAME)
# ==============================================================================
class StartFrame(ctk.CTkFrame):
    def __init__(self, master, controller: MainApp):
        super().__init__(master)
        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=4)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1, uniform="col")
        self.grid_columnconfigure(1, weight=1, uniform="col")
        self.grid_columnconfigure(2, weight=1, uniform="col")

        title = ctk.CTkLabel(self, text="Select Analysis Module", font=("Arial", 60, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="s", pady=(0, 30))

        # Hydrate detection workflows
        box_hydrate = ctk.CTkFrame(self, fg_color="#1F2E2C", corner_radius=20)
        box_hydrate.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        ctk.CTkLabel(box_hydrate, text="Hydrate\nDetection", font=("Arial", 35, "bold"), text_color="#539C93").pack(pady=30)
        ctk.CTkButton(box_hydrate, text="Preprocessing", font=("Arial", 25), height=50, fg_color="#2D6961", hover_color="#1b3f3a", command=lambda: self.controller.show_frame("HydratePreprocessingFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_hydrate, text="Detection", font=("Arial", 25), height=50, fg_color="#2D6961", hover_color="#1b3f3a", command=lambda: self.controller.show_frame("HydrateDetectionFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_hydrate, text="Multiple Detection", font=("Arial", 25), height=50, fg_color="#1C4741", hover_color="#112b27", command=lambda: self.controller.show_frame("HydrateMultipleDetectionFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_hydrate, text="Calibration", font=("Arial", 25), height=50, fg_color="#2D6961", hover_color="#1b3f3a", command=lambda: self.controller.show_frame("HydrateCalibrationFrame")).pack(pady=15, padx=30, fill="x")

        # Grain-boundary segmentation workflows
        box_grain = ctk.CTkFrame(self, fg_color="#2E241F", corner_radius=20)
        box_grain.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        ctk.CTkLabel(box_grain, text="Grain Boundary\nSegmentation", font=("Arial", 35, "bold"), text_color="#DAAD4D").pack(pady=30)
        ctk.CTkButton(box_grain, text="Preprocessing", font=("Arial", 25), height=50, fg_color="#D67040", hover_color="#804326", command=lambda: self.controller.show_frame("GrainBoundaryPreprocessingFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_grain, text="Detection", font=("Arial", 25), height=50, fg_color="#D67040", hover_color="#804326", command=lambda: self.controller.show_frame("GrainBoundaryDetectionFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_grain, text="Multiple Detection", font=("Arial", 25), height=50, fg_color="#A3542E", hover_color="#6B371E", command=lambda: self.controller.show_frame("GrainBoundaryMultipleDetectionFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_grain, text="Calibration", font=("Arial", 25), height=50, fg_color="#D67040", hover_color="#804326", command=lambda: self.controller.show_frame("GrainBoundaryCalibrationFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_grain, text="Pipeline Tuner", font=("Arial", 25), height=50, fg_color="#D67040", hover_color="#804326", command=lambda: self.controller.show_frame("GrainBoundaryTunerFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_grain, text="Watershed Tuner", font=("Arial", 25), height=50, fg_color="#D67040", hover_color="#804326", command=lambda: self.controller.show_frame("GrainBoundaryWatershedTunerFrame")).pack(pady=15, padx=30, fill="x")

        # 3D reconstruction and data-merging workflows
        box_merge = ctk.CTkFrame(self, fg_color="#1B2433", corner_radius=20)
        box_merge.grid(row=1, column=2, sticky="nsew", padx=20, pady=20)
        ctk.CTkLabel(box_merge, text="3D Modeling &\nData Merge", font=("Arial", 35, "bold"), text_color="#6C9CE8").pack(pady=30)
        ctk.CTkButton(box_merge, text="Layer Alignment", font=("Arial", 25), height=50, fg_color="#345B94", hover_color="#20395D", command=lambda: self.controller.show_frame("LayerAlignmentFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_merge, text="FA Analysis", font=("Arial", 25), height=50, fg_color="#345B94", hover_color="#20395D", command=lambda: self.controller.show_frame("FAAnalysisFrame")).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(box_merge, text="3D Reconstruction & Assignment", font=("Arial", 25), height=50, fg_color="#25426C", hover_color="#15263E", command=lambda: self.controller.show_frame("ModelReconstructionFrame")).pack(pady=15, padx=30, fill="x")

if __name__ == "__main__":
    import multiprocessing
    if getattr(sys, 'frozen', False):
        multiprocessing.freeze_support()
        multiprocessing.set_executable(sys.executable)
    else:
        multiprocessing.freeze_support()

    if sys.platform == "win32":
        sys.argv = [arg for arg in sys.argv if arg.strip() and "=" in arg or not arg.startswith("--multiprocessing-fork")]
    
    app = MainApp()
    app.mainloop()
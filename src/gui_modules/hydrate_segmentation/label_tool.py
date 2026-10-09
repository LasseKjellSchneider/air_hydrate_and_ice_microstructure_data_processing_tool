import json
import pathlib
import math
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tifffile
import numpy as np


class HydrateLabelTool(ctk.CTkToplevel):
    """
    Interactive GUI tool for generating ground truth data.
    Uses a 3-point click system to define bounding ellipses for hydrates.
    Features depth-based color coding to visualize objects across z-layers.
    """
    def __init__(self, master, stack_path, save_dir, on_close_callback=None, current_idx=1, total_stacks=1):
        super().__init__(master)
        
        self.stack_path = pathlib.Path(stack_path)
        self.save_dir = pathlib.Path(save_dir)
        self.on_close_callback = on_close_callback
        self.current_idx = current_idx
        self.total_stacks = total_stacks
        
        self.title(f"Hydrate Focus Label Tool - Stack {current_idx} of {total_stacks}: {self.stack_path.name}")
        self.geometry("1000x800")
        
        # Ensure window remains on top and captures focus
        self.grab_set()

        # 1. Load the TIFF stack natively
        with tifffile.TiffFile(str(self.stack_path)) as tif:
            num_pages = len(tif.pages)
            if num_pages > 1:
                pages_data = [page.asarray() for page in tif.pages]
                self.stack = np.stack(pages_data, axis=0)
            else:
                self.stack = tif.asarray()

        # 2. Validate dimensions and sanitize structure
        if self.stack.ndim == 2:
            self.stack = np.expand_dims(self.stack, axis=0)
        elif self.stack.ndim == 3:
            min_axis = int(np.argmin(self.stack.shape))
            if min_axis != 0 and self.stack.shape[min_axis] < self.stack.shape[0]:
                self.stack = np.moveaxis(self.stack, min_axis, 0)

        # 3. Initialize state variables
        self.num_layers = self.stack.shape[0]
        self.current_layer = 0
        self.labeled_hydrates = []
        
        # State machine variables for 3-point ellipse drawing
        # State 0: Wait for Point 1 (Major Axis Start)
        # State 1: Wait for Point 2 (Major Axis End)
        # State 2: Wait for Point 3 (Minor Axis / Thickness)
        self.drawing_state = 0
        self.p1 = None
        self.p2 = None
        
        # Temporary patches for live preview
        self.temp_line = None
        self.temp_ellipse = None

        print(f"[INFO] Successfully loaded stack '{self.stack_path.name}': {self.num_layers} Z-layers, Shape: {self.stack.shape}")

        # 4. Build the user interface
        self.setup_gui()

        # 5. Force keyboard focus
        self.focus_set()

    def setup_gui(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Give the canvas the expandable row so the controls remain visible.
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # 1. Instruction Label (Header)
        header_text = f"Stack {self.current_idx}/{self.total_stacks} | Left Click: Set Point 1 (Start of Major Axis)"
        self.instruction_label = ctk.CTkLabel(
            main_frame, text=header_text, 
            font=("Arial", 16, "bold"), text_color="#f39c12"
        )
        self.instruction_label.grid(row=0, column=0, pady=(0, 5), sticky="ew")

        # 2. Setup Matplotlib Figure with Dark Theme
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')

        self.im = self.ax.imshow(self.stack[0], cmap='gray')
        self.ax.set_title(f"Z-Layer: 0 / {max(0, self.num_layers - 1)}", color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        canvas_widget.configure(bg="#2b2b2b", highlightthickness=0)
        
        # Bind mouse events for the state machine
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # 3. Control Frame (Slider & Info)
        control_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=10)
        control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(control_frame, text="Z-Layer (Focus):", font=("Arial", 14)).grid(row=0, column=0, sticky="e", padx=5)
        
        max_slider_to = max(0, self.num_layers - 1)
        self.slider_var = ctk.IntVar(value=0)
        
        self.z_slider = ctk.CTkSlider(
            control_frame, 
            from_=0, 
            to=max_slider_to, 
            number_of_steps=max(1, max_slider_to),
            variable=self.slider_var,
            command=self.on_slider_move
        )
        self.z_slider.grid(row=0, column=1, sticky="ew", padx=10)
        self.z_slider.set(0)
        
        self.layer_label = ctk.CTkLabel(control_frame, text=f"Layer: 0 / {max_slider_to}", font=("Arial", 14))
        self.layer_label.grid(row=0, column=2, sticky="w", padx=5)
        
        # 4. Actions Frame (Buttons)
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        ctk.CTkButton(
            btn_frame, text="Undo (Cancel / Delete)", command=self.undo_action, 
            corner_radius=15, fg_color="#c0392b", hover_color="#e74c3c"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Save & Close", command=self.save_and_close, 
            corner_radius=15, fg_color="#27ae60", hover_color="#2ecc71"
        ).pack(side="right", padx=5)
        
        # Key bindings
        self.bind("<Left>", lambda e: self.change_layer(-1))
        self.bind("<Right>", lambda e: self.change_layer(1))
        self.bind("<Escape>", lambda e: self.cancel_drawing())

    def _get_ellipse_color(self, current_layer, ellipse_layer):
        """
        Calculates the color of an ellipse based on its z-layer distance to the current view.
        0 layers diff: Bright Green
        Ellipse is ABOVE (lower index): Interpolates from Red (1 layer) to Blue (5 layers)
        Ellipse is BELOW (higher index): Interpolates from Red (1 layer) to Yellow (5 layers)
        """
        diff = ellipse_layer - current_layer
        
        if diff == 0:
            return "#00FF00"  # Bright Green
            
        # Limit distance to a maximum of 5 for color interpolation
        abs_diff = min(abs(diff), 5)
        
        # Interpolation factor t goes from 0.0 (distance 1) to 1.0 (distance 5)
        t = (abs_diff - 1) / 4.0
        
        if diff < 0:
            # Ellipse is ABOVE the current layer (lower index): Red -> Blue
            r = int(255 * (1 - t))
            g = 0
            b = int(255 * t)
        else:
            # Ellipse is BELOW the current layer (higher index): Red -> Yellow
            r = 255
            g = int(255 * t)
            b = 0
            
        return f"#{r:02x}{g:02x}{b:02x}"

    def update_image(self):
        """Updates the image and redraws all stored ellipses with depth-based coloring."""
        self.im.set_data(self.stack[self.current_layer])
        self.ax.set_title(f"Z-Layer: {self.current_layer} / {max(0, self.num_layers - 1)}", color='white')
        
        # Clear existing patches and lines
        [p.remove() for p in reversed(self.ax.patches)]
        [l.remove() for l in reversed(self.ax.lines)]
        self.temp_line = None
        self.temp_ellipse = None
        
        # Redraw all finalized ground truth ellipses
        for lbl in self.labeled_hydrates:
            color = self._get_ellipse_color(self.current_layer, lbl["best_z"])
            
            # Slightly transparent if not on the exact same layer
            alpha_val = 1.0 if lbl["best_z"] == self.current_layer else 0.6
            
            ell = Ellipse(
                xy=(lbl["cx"], lbl["cy"]), 
                width=lbl["width"], 
                height=lbl["height"], 
                angle=lbl["angle_deg"],
                edgecolor=color, fc='None', lw=2, alpha=alpha_val
            )
            self.ax.add_patch(ell)
            
            # Add center cross
            self.ax.plot(lbl["cx"], lbl["cy"], marker='+', color=color, markersize=6, alpha=alpha_val)

        self.canvas.draw_idle()

    # --- Math Helpers for 3-Point Ellipse ---
    
    def _calc_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _calc_orthogonal_distance(self, p1, p2, p3):
        """Calculates the shortest distance from point p3 to the line defined by p1 and p2."""
        num = abs((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p1[0] - p3[0]) * (p2[1] - p1[1]))
        den = self._calc_distance(p1, p2)
        return num / den if den != 0 else 0

    # --- Interaction Logic ---

    def on_slider_move(self, value):
        new_layer = int(float(value))
        if new_layer == self.current_layer:
            return
        self.current_layer = new_layer
        self.layer_label.configure(text=f"Layer: {self.current_layer} / {max(0, self.num_layers - 1)}")
        self.update_image()

    def change_layer(self, delta):
        new_layer = max(0, min(self.num_layers - 1, self.current_layer + delta))
        if new_layer == self.current_layer:
            return
        self.current_layer = new_layer
        self.slider_var.set(new_layer)
        self.layer_label.configure(text=f"Layer: {self.current_layer} / {max(0, self.num_layers - 1)}")
        self.update_image()

    def cancel_drawing(self):
        """Resets the state machine without deleting stored objects."""
        if self.drawing_state > 0:
            self.drawing_state = 0
            self.p1 = None
            self.p2 = None
            header_text = f"Stack {self.current_idx}/{self.total_stacks} | Left Click: Set Point 1 (Start of Major Axis)"
            self.instruction_label.configure(text=header_text, text_color="#f39c12")
            self.update_image()

    def on_click(self, event):
        # Ignore clicks outside the image bounds
        if event.inaxes != self.ax:
            return
        
        # Right click to cancel current drawing
        if event.button == 3:
            self.cancel_drawing()
            return
            
        x, y = float(event.xdata), float(event.ydata)
        
        if self.drawing_state == 0:
            self.p1 = (x, y)
            self.drawing_state = 1
            header_text = f"Stack {self.current_idx}/{self.total_stacks} | Left Click: Set Point 2 (End of Major Axis) | Right Click: Cancel"
            self.instruction_label.configure(text=header_text, text_color="#3498db")
            
        elif self.drawing_state == 1:
            self.p2 = (x, y)
            if self._calc_distance(self.p1, self.p2) < 2.0:
                print("[WARNING] Points too close, ignoring.")
                return
            self.drawing_state = 2
            header_text = f"Stack {self.current_idx}/{self.total_stacks} | Left Click: Set Point 3 (Ellipse Thickness) | Right Click: Cancel"
            self.instruction_label.configure(text=header_text, text_color="#e74c3c")
            
        elif self.drawing_state == 2:
            # Finalize Ellipse
            width = self._calc_distance(self.p1, self.p2)
            d = self._calc_orthogonal_distance(self.p1, self.p2, (x, y))
            height = 2 * d
            
            # Prevent degenerate flat lines
            if height < 2.0:
                height = 2.0
                
            cx = (self.p1[0] + self.p2[0]) / 2.0
            cy = (self.p1[1] + self.p2[1]) / 2.0
            angle_deg = math.degrees(math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0]))
            
            self.labeled_hydrates.append({
                "cx": float(cx),
                "cy": float(cy),
                "width": float(width),
                "height": float(height),
                "angle_deg": float(angle_deg),
                "best_z": self.current_layer
            })
            
            self.drawing_state = 0
            self.p1 = None
            self.p2 = None
            header_text = f"Stack {self.current_idx}/{self.total_stacks} | Left Click: Set Point 1 (Start of Major Axis)"
            self.instruction_label.configure(text=header_text, text_color="#f39c12")
            self.update_image()

    def on_mouse_move(self, event):
        """Provides a live preview of the line or ellipse being drawn."""
        if event.inaxes != self.ax:
            return
            
        x, y = float(event.xdata), float(event.ydata)
        
        if self.drawing_state == 1:
            # Draw preview line
            if self.temp_line:
                self.temp_line.remove()
            self.temp_line, = self.ax.plot([self.p1[0], x], [self.p1[1], y], 'r--', lw=1.5)
            self.canvas.draw_idle()
            
        elif self.drawing_state == 2:
            # Draw preview ellipse
            width = self._calc_distance(self.p1, self.p2)
            d = self._calc_orthogonal_distance(self.p1, self.p2, (x, y))
            height = max(2 * d, 2.0)
            
            cx = (self.p1[0] + self.p2[0]) / 2.0
            cy = (self.p1[1] + self.p2[1]) / 2.0
            angle_deg = math.degrees(math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0]))
            
            if self.temp_line:
                self.temp_line.remove()
                self.temp_line = None
            if self.temp_ellipse:
                self.temp_ellipse.remove()
                
            self.temp_ellipse = Ellipse(xy=(cx, cy), width=width, height=height, angle=angle_deg,
                                        edgecolor='r', fc='None', lw=1.5, linestyle='--')
            self.ax.add_patch(self.temp_ellipse)
            self.canvas.draw_idle()

    def undo_action(self):
        """Cancels current drawing or deletes the last finalized ellipse."""
        if self.drawing_state > 0:
            self.cancel_drawing()
        elif self.labeled_hydrates:
            removed = self.labeled_hydrates.pop()
            print(f"[INFO] Removed ellipse from layer {removed['best_z']}")
            self.update_image()

    def save_and_close(self):
        """Exports the labeled ellipses to JSON and safely terminates the window."""
        save_path = self.save_dir / f"ground_truth_{self.stack_path.stem}.json"
        
        output_data = {
            "stack_file": self.stack_path.name,
            "total_labeled_hydrates": len(self.labeled_hydrates),
            "ground_truth_labels": self.labeled_hydrates
        }
        
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=4)

        plt.close(self.fig)
        plt.close('all')
        
        # 1. Release window grab
        self.grab_release()
        
        # 2. Destroy the top-level window
        self.destroy()
        
        # 3. Force main loop update to clear the destroyed widget reference
        self.master.update()
        
        # 4. Trigger the orchestrator callback to load next image (or start optimization)
        if self.on_close_callback is not None:
            self.on_close_callback()
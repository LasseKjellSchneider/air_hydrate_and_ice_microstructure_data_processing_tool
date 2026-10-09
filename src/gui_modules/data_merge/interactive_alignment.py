import cv2
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image
import h5py

Image.MAX_IMAGE_PIXELS = None

class InteractiveAlignmentTool(ctk.CTkFrame):
    """
    Automated Point-Matching GUI with manual rotation helpers and a pen tool.
    Dynamically downscales images based on their source (Microscope vs FA) for extreme speed.
    """
    def __init__(
        self, master, anchor_path: str, target_path: str, allow_scaling: bool, 
        is_anchor_fa: bool, is_target_fa: bool, anchor_mirror: bool, target_mirror: bool, 
        phase_name: str, anchor_name: str, target_name: str, callback
    ):
        super().__init__(master, fg_color="transparent")
        
        self.callback = callback
        self.allow_scaling = allow_scaling
        self.phase_name = phase_name
        self.anchor_name = anchor_name
        self.target_name = target_name
        
        # Retain each source image's scale factor for full-resolution alignment.
        self.anchor_img, self.anchor_ds = self._load_and_downscale_image(anchor_path)
        self.target_img, self.target_ds = self._load_and_downscale_image(target_path)

        if is_anchor_fa:   self.anchor_img = cv2.rotate(self.anchor_img, cv2.ROTATE_90_CLOCKWISE)
        if anchor_mirror:  self.anchor_img = cv2.flip(self.anchor_img, 0)
        if is_target_fa:   self.target_img = cv2.rotate(self.target_img, cv2.ROTATE_90_CLOCKWISE)
        if target_mirror:  self.target_img = cv2.flip(self.target_img, 0)
        
        self.anchor_M_vis = np.eye(3, dtype=np.float64)
        self.target_M_vis = np.eye(3, dtype=np.float64)

        self.anchor_pts, self.target_pts = [], []
        
        self.drawn_lines_ax1, self.drawn_lines_ax2 = [], []
        self.is_drawing = False
        self.current_line_x, self.current_line_y = [], []
        self.current_ax, self.current_line_obj = None, None
        
        self.calculated_matrix_large = None
        
        self.setup_ui()
        self._redraw_all()

    def _load_and_downscale_image(self, path: str) -> tuple[np.ndarray, int]:
        if path.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            ds = 24  # Downscale microscope images to keep interactive rendering responsive.
            img_pil = Image.open(path).convert("RGB")
            new_w = max(1, img_pil.width // ds)
            new_h = max(1, img_pil.height // ds)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
            return np.array(img_pil), ds
            
        elif path.lower().endswith('.h5'):
            ds = 8  # Use a smaller reduction for HDF5 fabric-analysis images.
            with h5py.File(path, 'r') as f:
                if "ws_image" in f:
                    ws = np.squeeze(f["ws_image"][:])
                    ws_display = (ws % 255).astype(np.uint8)
                    colored = cv2.applyColorMap(ws_display, cv2.COLORMAP_TURBO)
                    colored[ws == 0] = [0, 0, 0]
                    img = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
                elif "layers" in f and len(f["layers"]) > 0:
                    first_key = list(f["layers"].keys())[0]
                    data = f["layers"][first_key][:]
                    norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    img = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
                else:
                    raise ValueError(f"No suitable display dataset found in {path}")
                
                new_h = max(1, img.shape[0] // ds)
                new_w = max(1, img.shape[1] // ds)
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST), ds
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def setup_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(title_frame, text=f"{self.phase_name}", font=("Arial", 24, "bold"), text_color="#3498DB").pack()

        control_frame = ctk.CTkFrame(self, fg_color="#1B2433")
        control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        top_row = ctk.CTkFrame(control_frame, fg_color="transparent")
        top_row.pack(fill="x", pady=5)
        ctk.CTkButton(top_row, text="⟳ Rotate Anchor 90°", width=140, fg_color="#8E44AD", hover_color="#732D91", command=self.rotate_anchor).pack(side="left", padx=10)
        ctk.CTkButton(top_row, text="⟳ Rotate Target 90°", width=140, fg_color="#8E44AD", hover_color="#732D91", command=self.rotate_target).pack(side="left", padx=10)
        self.pen_mode_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(top_row, text="Pen Tool (Draw Lines)", variable=self.pen_mode_var, font=("Arial", 14, "bold")).pack(side="left", padx=30)
        mode_text = "Similarity (Scaling allowed)" if self.allow_scaling else "Strict Rigid (No Scaling)"
        ctk.CTkLabel(top_row, text=f"Mode: {mode_text}", font=("Arial", 14, "bold"), text_color="#F39C12").pack(side="right", padx=20)

        bottom_row = ctk.CTkFrame(control_frame, fg_color="transparent")
        bottom_row.pack(fill="x", pady=5)
        ctk.CTkButton(bottom_row, text="Clear Points & Lines", fg_color="#C0392B", hover_color="#922B21", command=self.clear_all).pack(side="left", padx=10)
        ctk.CTkButton(bottom_row, text="Calculate & Show Preview", fg_color="#2980B9", hover_color="#1A5276", command=self.calculate_and_preview).pack(side="left", padx=10)
        self.btn_accept = ctk.CTkButton(bottom_row, text="Accept Alignment & Next ►", fg_color="#27AE60", hover_color="#1E8449", state="disabled", command=self.accept_and_continue)
        self.btn_accept.pack(side="right", padx=20)

        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.patch.set_facecolor('#2b2b2b')
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def rotate_anchor(self):
        h, w = self.anchor_img.shape[:2]
        self.anchor_img = cv2.rotate(self.anchor_img, cv2.ROTATE_90_CLOCKWISE)
        M_rot = np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        self.anchor_M_vis = M_rot @ self.anchor_M_vis
        
        if self.anchor_pts:
            pts_hom = np.column_stack((np.array(self.anchor_pts), np.ones(len(self.anchor_pts))))
            self.anchor_pts = (M_rot @ pts_hom.T).T[:, :2].tolist()
        for line in self.drawn_lines_ax1:
            pts_hom = np.column_stack((line['x'], line['y'], np.ones(len(line['x']))))
            new_pts = (M_rot @ pts_hom.T).T
            line['x'], line['y'] = new_pts[:, 0].tolist(), new_pts[:, 1].tolist()
        self._redraw_all()

    def rotate_target(self):
        h, w = self.target_img.shape[:2]
        self.target_img = cv2.rotate(self.target_img, cv2.ROTATE_90_CLOCKWISE)
        M_rot = np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        self.target_M_vis = M_rot @ self.target_M_vis
        
        if self.target_pts:
            pts_hom = np.column_stack((np.array(self.target_pts), np.ones(len(self.target_pts))))
            self.target_pts = (M_rot @ pts_hom.T).T[:, :2].tolist()
        for line in self.drawn_lines_ax2:
            pts_hom = np.column_stack((line['x'], line['y'], np.ones(len(line['x']))))
            new_pts = (M_rot @ pts_hom.T).T
            line['x'], line['y'] = new_pts[:, 0].tolist(), new_pts[:, 1].tolist()
        self._redraw_all()

    def _redraw_all(self):
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.set_title(f"Anchor: {self.anchor_name}", color="#2ECC71", fontsize=14, weight="bold")
        self.ax1.imshow(self.anchor_img)
        if self.anchor_pts:
            pts = np.array(self.anchor_pts)
            self.ax1.scatter(pts[:,0], pts[:,1], c='red', s=60, edgecolors='white', zorder=5)
            for i, (px, py) in enumerate(pts):
                self.ax1.text(px + 20, py + 20, str(i+1), color="yellow", fontsize=14, weight="bold")
        for line in self.drawn_lines_ax1:
            self.ax1.plot(line['x'], line['y'], color='lime', linewidth=2, zorder=3)

        self.ax2.set_title(f"Target: {self.target_name}", color="#E74C3C", fontsize=14, weight="bold")
        self.ax2.imshow(self.target_img)
        if self.target_pts:
            pts = np.array(self.target_pts)
            self.ax2.scatter(pts[:,0], pts[:,1], c='cyan', s=60, edgecolors='white', zorder=5)
            for i, (px, py) in enumerate(pts):
                self.ax2.text(px + 20, py + 20, str(i+1), color="yellow", fontsize=14, weight="bold")
        for line in self.drawn_lines_ax2:
            self.ax2.plot(line['x'], line['y'], color='lime', linewidth=2, zorder=3)

        self.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes is None or event.button != 1: return
        if self.fig.canvas.toolbar.mode != '': return

        if self.pen_mode_var.get():
            self.is_drawing = True
            self.current_line_x, self.current_line_y = [event.xdata], [event.ydata]
            self.current_ax = event.inaxes
            self.current_line_obj, = self.current_ax.plot(self.current_line_x, self.current_line_y, color='lime', linewidth=2, zorder=4)
        else:
            if event.inaxes == self.ax1:
                self.anchor_pts.append([event.xdata, event.ydata])
            elif event.inaxes == self.ax2:
                self.target_pts.append([event.xdata, event.ydata])
            self._redraw_all()

    def on_motion(self, event):
        if not self.is_drawing or event.inaxes != self.current_ax: return
        self.current_line_x.append(event.xdata)
        self.current_line_y.append(event.ydata)
        self.current_line_obj.set_data(self.current_line_x, self.current_line_y)
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.is_drawing:
            self.is_drawing = False
            if len(self.current_line_x) > 1:
                line_dict = {'x': self.current_line_x, 'y': self.current_line_y}
                if self.current_ax == self.ax1: self.drawn_lines_ax1.append(line_dict)
                elif self.current_ax == self.ax2: self.drawn_lines_ax2.append(line_dict)
            else:
                self.current_line_obj.remove()
                self.canvas.draw_idle()

    def clear_all(self):
        self.anchor_pts.clear()
        self.target_pts.clear()
        self.drawn_lines_ax1.clear()
        self.drawn_lines_ax2.clear()
        self.btn_accept.configure(state="disabled")
        self._redraw_all()

    def calculate_and_preview(self):
        if len(self.anchor_pts) < 3 or len(self.target_pts) < 3:
            print("[ERROR] Please provide at least 3 matching point pairs.")
            return

        min_len = min(len(self.anchor_pts), len(self.target_pts))
        
        # Calculate the export transform in full-resolution source coordinates.
        M_anch_inv = np.linalg.inv(self.anchor_M_vis)
        M_targ_inv = np.linalg.inv(self.target_M_vis)
        
        pts_a = np.array(self.anchor_pts[:min_len])
        pts_a_hom = np.column_stack((pts_a, np.ones(len(pts_a))))
        pts_a_unrotated = (M_anch_inv @ pts_a_hom.T).T[:, :2]
        pts_a_orig = (pts_a_unrotated * self.anchor_ds).astype(np.float32)
        
        pts_t = np.array(self.target_pts[:min_len])
        pts_t_hom = np.column_stack((pts_t, np.ones(len(pts_t))))
        pts_t_unrotated = (M_targ_inv @ pts_t_hom.T).T[:, :2]
        pts_t_orig = (pts_t_unrotated * self.target_ds).astype(np.float32)

        matrix_large, _ = cv2.estimateAffinePartial2D(pts_t_orig, pts_a_orig)

        if not self.allow_scaling:
            sx = np.sqrt(matrix_large[0, 0]**2 + matrix_large[0, 1]**2)
            sy = np.sqrt(matrix_large[1, 0]**2 + matrix_large[1, 1]**2)
            matrix_large[0, 0] /= sx
            matrix_large[0, 1] /= sx
            matrix_large[1, 0] /= sy
            matrix_large[1, 1] /= sy

        self.calculated_matrix_large = matrix_large
        self.btn_accept.configure(state="normal")

        # Calculate a separate transform for the downscaled preview images.
        pts_t_vis = np.array(self.target_pts[:min_len], dtype=np.float32)
        pts_a_vis = np.array(self.anchor_pts[:min_len], dtype=np.float32)
        matrix_vis, _ = cv2.estimateAffinePartial2D(pts_t_vis, pts_a_vis)
        
        if not self.allow_scaling:
            sx = np.sqrt(matrix_vis[0, 0]**2 + matrix_vis[0, 1]**2)
            sy = np.sqrt(matrix_vis[1, 0]**2 + matrix_vis[1, 1]**2)
            matrix_vis[0, 0] /= sx
            matrix_vis[0, 1] /= sx
            matrix_vis[1, 0] /= sy
            matrix_vis[1, 1] /= sy

        h, w = self.anchor_img.shape[:2]
        warped_target = cv2.warpAffine(self.target_img, matrix_vis, (w, h), flags=cv2.INTER_LINEAR)

        overlay = np.zeros_like(self.anchor_img)
        overlay[:, :, 0] = self.anchor_img[:, :, 0] 
        overlay[:, :, 1] = warped_target[:, :, 1]   
        
        self.show_preview_window(overlay)

    def show_preview_window(self, overlay_img):
        preview_win = ctk.CTkToplevel(self)
        preview_win.title("Overlay Preview")
        preview_win.geometry("1000x800")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_title("Yellow = Perfect Match | Red = Anchor only | Green = Target only", color="white")
        ax.imshow(overlay_img)
        ax.axis("off")
        
        canvas = FigureCanvasTkAgg(fig, master=preview_win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, preview_win)
        toolbar.update()
        canvas.draw()

    def accept_and_continue(self):
        if self.calculated_matrix_large is not None:
            self.callback(self.calculated_matrix_large)
            
    def destroy(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        super().destroy()
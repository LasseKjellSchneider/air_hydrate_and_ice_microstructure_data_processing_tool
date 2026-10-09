import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde
import logging
import json
import pathlib
import sys

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)

# Load the shared hydrate-segmentation configuration.
def get_constants_config_path():
    if getattr(sys, 'frozen', False):
        base_dir = pathlib.Path(sys.executable).parent
    else:
        base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    return base_dir / "hydrate_constants.json"

GLOBAL_CONSTANTS = {}
try:
    with open(get_constants_config_path(), "r", encoding="utf-8") as f:
        GLOBAL_CONSTANTS = json.load(f)
except Exception as e:
    logging.error(f"Visualizer: Failed to load hydrate_constants.json: {e}")

class OptimizationVisualizer:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.fig = None
        self.widget = None
        self.axes = {}

        self.bg_color = '#1a1a1a'
        self.plot_bg_color = '#2b2b2b'
        self.text_color = 'white'
        self.grid_color = '#555555'
        
        # Assign a distinct color to each candidate-processing category.
        self.c_paths = {
            "1_good_ok":   "#2ecc71", # Bright Green
            "1_good_fail": "#27ae60", # Dark Green
            "2_resc_ok":   "#f1c40f", # Yellow
            "2_resc_fail": "#e67e22", # Orange
            "2_rf_trash":  "#9b59b6", # Purple
            "3_trash":     "#e74c3c"  # Red
        }

    def init_plot(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor(self.bg_color)
        
        gs = gridspec.GridSpec(4, 4, height_ratios=[1.5, 1, 1, 0.25], hspace=0.6, wspace=0.3)
        
        self.axes["error"] = self.fig.add_subplot(gs[0, 0:3])
        self._format_axis(self.axes["error"], "Error Convergence", "Generation", "Penalty Score")

        self.axes["table"] = self.fig.add_subplot(gs[0, 3])
        self.axes["table"].axis('off')

        self.feature_maps = [
            ("area_px", "Area (px)", False),
            ("aspect_ratio", "Aspect Ratio", True),
            ("circularity", "Circularity", False),
            ("ellipse_fill_factor", "Ellipse Fill", False),
            ("solidity", "Solidity", False),
            ("core_brightness_75th", "Core Brightness", False),
            ("core_rim_contrast", "Contrast Ratio", False),
            ("total_score", "Total Fuzzy Score", False)
        ]

        for idx, (f_key, f_title, is_log) in enumerate(self.feature_maps):
            row = 1 if idx < 4 else 2
            col = idx % 4
            ax = self.fig.add_subplot(gs[row, col])
            self._format_axis(ax, f_title, "", "")
            ax.set_yticks([]) 
            self.axes[f_key] = ax

        self.axes["legend"] = self.fig.add_subplot(gs[3, :])
        self.axes["legend"].axis('off')
        
        # Build the category legend.
        patches = [
            mpatches.Patch(color=self.c_paths["1_good_ok"], label='1: Good (OK)'),
            mpatches.Patch(color=self.c_paths["1_good_fail"], label='1: Good (Score Fail)'),
            mpatches.Patch(color=self.c_paths["2_resc_ok"], label='2: Rescued (OK)'),
            mpatches.Patch(color=self.c_paths["2_resc_fail"], label='2: Rescued (Score Fail)'),
            mpatches.Patch(color=self.c_paths["2_rf_trash"], label='2: RF Aborted'),
            mpatches.Patch(color=self.c_paths["3_trash"], label='3: Triage Trash')
        ]
        self.axes["legend"].legend(
            handles=patches, loc='center', ncol=6, facecolor=self.bg_color, 
            edgecolor=self.grid_color, labelcolor='white', fontsize=9
        )

        canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.widget = canvas.get_tk_widget()
        self.widget.pack(fill="both", expand=True, padx=5, pady=5)
        self.fig.canvas.draw()

    def _format_axis(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(self.plot_bg_color)
        ax.tick_params(colors=self.text_color, labelsize=8)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        ax.title.set_fontsize(10)
        ax.title.set_fontweight('bold')
        ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        for spine in ax.spines.values(): spine.set_color(self.grid_color)
        ax.grid(True, linestyle='--', alpha=0.3, color=self.grid_color)

    def _plot_kde_and_rug(self, ax, data, y_bottom, y_height, color, is_gt=True):
        """Zeichnet Rug-Plot und KDE auf einer bestimmten Spur."""
        if not data: return
        
        # Emphasize ground-truth matches and de-emphasize unmatched objects.
        alpha_rug = 0.8 if is_gt else 0.2
        alpha_fill = 0.7 if is_gt else 0.15
        zorder = 5 if is_gt else 4
        
        # Draw one tick for each observation.
        ax.vlines(x=data, ymin=y_bottom, ymax=y_bottom + y_height * 0.25, color=color, alpha=alpha_rug, linewidth=1.5, zorder=zorder)
        
        # Overlay a kernel-density estimate when enough samples are available.
        if len(data) > 1 and np.var(data) > 1e-6:
            try:
                kde = gaussian_kde(data)
                x_grid = np.linspace(min(data), max(data), 150)
                y_kde = kde(x_grid)
                y_kde_norm = (y_kde / max(y_kde)) * (y_height * 0.90)
                
                if is_gt:
                    # Outline the ground-truth density curve for contrast.
                    ax.plot(x_grid, y_bottom + y_kde_norm, color='white', alpha=0.9, linewidth=1.2, zorder=zorder+1)
                ax.fill_between(x_grid, y_bottom, y_bottom + y_kde_norm, color=color, alpha=alpha_fill, zorder=zorder)
            except Exception: pass 

    def _update_table(self, stats):
        ax = self.axes["table"]
        ax.clear()
        ax.axis('off')
        
        paths = stats.get("paths", {})
        table_data = [
            ["1: Good -> OK", str(paths.get("1_good_ok_gt", 0)), str(paths.get("1_good_ok_no_gt", 0))],
            ["1: Good -> FAIL", str(paths.get("1_good_fail_gt", 0)), str(paths.get("1_good_fail_no_gt", 0))],
            ["2: Rescued -> OK", str(paths.get("2_resc_ok_gt", 0)), str(paths.get("2_resc_ok_no_gt", 0))],
            ["2: Rescued -> FAIL", str(paths.get("2_resc_fail_gt", 0)), str(paths.get("2_resc_fail_no_gt", 0))],
            ["2: RF Aborted", str(paths.get("2_rf_trash_gt", 0)), str(paths.get("2_rf_trash_no_gt", 0))],
            ["3: Phase 1 Trash", str(paths.get("3_trash_gt", 0)), str(paths.get("3_trash_no_gt", 0))]
        ]
        cols = ["Path", "GT Matched", "No GT Match"]
        table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.8)

        # Match each table row's text color to its category.
        color_keys = list(self.c_paths.keys())
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(self.grid_color)
            if row == 0:
                cell.set_facecolor('#444444')
                cell.get_text().set_color('white')
                cell.get_text().set_fontweight('bold')
            else:
                cell.set_facecolor(self.plot_bg_color)
                base_color = self.c_paths[color_keys[row-1]]
                if col == 0: cell.get_text().set_color(base_color)
                elif col == 1: cell.get_text().set_color(base_color)
                elif col == 2: cell.get_text().set_color(base_color)
                cell.get_text().set_alpha(1.0 if col == 1 else 0.5)

    def _draw_scoring_curve(self, ax, f_key, params, xlims):
        if f_key == "total_score": return
        x_min, x_max = xlims
        x_grid = np.linspace(x_min, x_max, 500)
        y_curve = np.zeros_like(x_grid)
        
        try:
            if f_key == "area_px":
                mu, sig = params.get("area_mu", 0), params.get("area_sigma", 1)
                y_curve = np.exp(-0.5 * ((x_grid - mu) / sig)**2) if sig > 0 else y_curve
            elif f_key == "aspect_ratio":
                mu, sig = params.get("aspect_mu", 1.0), params.get("aspect_sigma", 1)
                y_curve = np.exp(-0.5 * ((x_grid - mu) / sig)**2) if sig > 0 else y_curve
            elif f_key == "core_rim_contrast":
                mu, sig = params.get("contrast_mu", 1.0), params.get("contrast_sigma", 1)
                y_curve = np.exp(-0.5 * ((x_grid - mu) / sig)**2) if sig > 0 else y_curve
            else:
                if f_key == "circularity": x0, k = params.get("circ_x0", 0), params.get("circ_k", 1)
                elif f_key == "ellipse_fill_factor": x0, k = params.get("fill_x0", 0), params.get("fill_k", 1)
                elif f_key == "solidity": x0, k = params.get("solid_x0", 0), params.get("solid_k", 1)
                elif f_key == "core_brightness_75th": x0, k = params.get("bright_x0", 0), params.get("bright_k", 1)
                else: return
                exponent = np.clip(-k * (x_grid - x0), -500, 500)
                y_curve = 1.0 / (1.0 + np.exp(exponent))
            # Draw the scoring function across the feature plot.
            ax.plot(x_grid, y_curve, color='#3498db', linestyle='-', linewidth=2.5, alpha=0.6, zorder=10)
        except Exception: pass 

    def _draw_bounds(self, ax, f_key):
        """Zeichnet die Soft und Hard Bounds aus der JSON als vertikale Linien."""
        if f_key == "total_score" or "phase_1_triage_bounds" not in GLOBAL_CONSTANTS: return
        
        bounds_map = {
            "area_px": ("area_min_px", "area_max_px"),
            "aspect_ratio": (None, "aspect_max"),
            "circularity": ("circ_min", None),
            "ellipse_fill_factor": ("fill_min", None),
            "solidity": ("solid_min", None),
            "core_brightness_75th": ("bright_min", None),
            "core_rim_contrast": ("contrast_min", "contrast_max")
        }
        
        if f_key not in bounds_map: return
        key_min, key_max = bounds_map[f_key]
        
        strict_b = GLOBAL_CONSTANTS["phase_1_triage_bounds"].get("strict_bounds", {})
        loose_b = GLOBAL_CONSTANTS["phase_1_triage_bounds"].get("loose_bounds", {})

        # Show loose acceptance bounds as gray dotted lines.
        if key_min and key_min in loose_b:
            ax.axvline(x=loose_b[key_min], color='#aaaaaa', linestyle=':', lw=1.5, alpha=0.8, zorder=1)
        if key_max and key_max in loose_b:
            ax.axvline(x=loose_b[key_max], color='#aaaaaa', linestyle=':', lw=1.5, alpha=0.8, zorder=1)
            
        # Show strict acceptance bounds as white dashed lines.
        if key_min and key_min in strict_b:
            ax.axvline(x=strict_b[key_min], color='white', linestyle='--', lw=2, alpha=0.9, zorder=2)
        if key_max and key_max in strict_b:
            ax.axvline(x=strict_b[key_max], color='white', linestyle='--', lw=2, alpha=0.9, zorder=2)

    def update_plot(self, gen, data_dict, params, path_features=None, stats_matrix=None, eta_str=""):
        if path_features is None: path_features = {}
        if stats_matrix is None: stats_matrix = {}

        # Update the global error-convergence plot.
        ax_err = self.axes["error"]
        ax_err.clear()
        self._format_axis(ax_err, f"Global Error Convergence (ETA: {eta_str})", "Generation", "Penalty Score")
        
        gens = data_dict.get("gen", [])
        fp_vals = data_dict.get("fp", [])
        fn_vals = data_dict.get("fn", [])
        
        if gens:
            ax_err.stackplot(gens, fn_vals, fp_vals, labels=["False Negatives", "False Positives"], colors=[self.c_paths["3_trash"], self.c_paths["1_good_fail"]], alpha=0.8)
            ax_err.set_xlim(left=max(1, min(gens)), right=max(2, max(gens)))
            ax_err.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            legend = ax_err.legend(loc="upper right", facecolor=self.bg_color, edgecolor=self.grid_color, fontsize=9)
            for text in legend.get_texts(): text.set_color(self.text_color)

        # Refresh the category summary table.
        self._update_table(stats_matrix)

        # Arrange the six processing categories as horizontal lanes.
        lanes = [
            "3_trash", 
            "2_rf_trash", 
            "2_resc_fail", 
            "2_resc_ok", 
            "1_good_fail", 
            "1_good_ok"
        ]
        
        lane_height = 1.0 / len(lanes)

        for f_key, f_title, is_log in self.feature_maps:
            ax = self.axes[f_key]
            ax.clear()
            self._format_axis(ax, f_title, "", "")
            ax.set_yticks([]) 
            ax.set_ylim(0, 1)

            all_values = []
            
            # Plot ground-truth and unmatched observations for each category.
            for i, path in enumerate(lanes):
                y_bottom = i * lane_height
                color = self.c_paths[path]
                
                v_gt = [obj[f_key] for obj in path_features.get(f"{path}_gt", []) if f_key in obj and obj[f_key] is not None]
                v_nogt = [obj[f_key] for obj in path_features.get(f"{path}_no_gt", []) if f_key in obj and obj[f_key] is not None]
                
                # Draw unmatched observations first so ground-truth data stays visible.
                self._plot_kde_and_rug(ax, v_nogt, y_bottom, lane_height, color, is_gt=False)
                self._plot_kde_and_rug(ax, v_gt, y_bottom, lane_height, color, is_gt=True)
                
                all_values.extend(v_gt)
                all_values.extend(v_nogt)

            # Derive feature-axis limits from the observed values.
            if all_values:
                # Use percentiles to limit the influence of extreme outliers.
                max_val = np.percentile(all_values, 98)
                min_val = np.percentile(all_values, 2)
                
                if is_log:
                    ax.set_xscale('log')
                    ax.set_xticks([1, 2, 3, 4, 5])
                    ax.set_xticklabels(['1', '2', '3', '4', '5'], color='white')
                    ax.set_xlim(left=1.0, right=max(max_val * 1.5, 2.0))
                else:
                    padding = (max_val - min_val) * 0.1
                    ax.set_xlim(left=min(0, min_val - padding), right=max_val + padding)

            # Overlay configured bounds and the scoring function.
            self._draw_bounds(ax, f_key)
            self._draw_scoring_curve(ax, f_key, params, ax.get_xlim())

            if f_key == "total_score":
                threshold = params.get("total_score_threshold", 5.0)
                ax.axvline(x=threshold, color='white', linestyle='-', linewidth=2.5, zorder=10)

        self.fig.canvas.draw_idle()
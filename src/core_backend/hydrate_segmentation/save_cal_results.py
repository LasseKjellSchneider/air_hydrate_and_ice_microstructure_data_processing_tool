import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting in worker threads.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import logging
import sys

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

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
    logging.error(f"SaveCalResults: Failed to load hydrate_constants.json: {e}")

# =====================================================================
# FARBEN & STYLES
# =====================================================================
C_BG = '#1a1a1a'
C_PLOT_BG = '#2b2b2b'
C_TEXT = 'white'
C_GRID = '#555555'

C_PATHS = {
    "1_good_ok":   "#2ecc71", # Bright Green
    "1_good_fail": "#27ae60", # Dark Green
    "2_resc_ok":   "#f1c40f", # Yellow
    "2_resc_fail": "#e67e22", # Orange
    "2_rf_trash":  "#9b59b6", # Purple
    "3_trash":     "#e74c3c"  # Red
}

def _format_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(C_PLOT_BG)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)
    ax.title.set_fontsize(12)
    ax.title.set_fontweight('bold')
    
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
        
    ax.grid(True, linestyle='--', alpha=0.3, color=C_GRID)


def _plot_kde_and_rug_export(ax, data, y_bottom, y_height, color, is_gt=True):
    if not data: return
        
    alpha_rug = 0.8 if is_gt else 0.2
    alpha_fill = 0.7 if is_gt else 0.15
    zorder = 5 if is_gt else 4
    
    ax.vlines(x=data, ymin=y_bottom, ymax=y_bottom + y_height * 0.25, color=color, alpha=alpha_rug, linewidth=1.5, zorder=zorder)
    
    if len(data) > 1 and np.var(data) > 1e-6:
        try:
            kde = gaussian_kde(data)
            x_grid = np.linspace(min(data), max(data), 200)
            y_kde = kde(x_grid)
            y_kde_norm = (y_kde / max(y_kde)) * (y_height * 0.90)
            
            if is_gt:
                ax.plot(x_grid, y_bottom + y_kde_norm, color='white', alpha=0.9, linewidth=1.2, zorder=zorder+1)
            ax.fill_between(x_grid, y_bottom, y_bottom + y_kde_norm, color=color, alpha=alpha_fill, zorder=zorder)
        except Exception:
            pass

def _draw_bounds_export(ax, f_key):
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

    if key_min and key_min in loose_b:
        ax.axvline(x=loose_b[key_min], color='#aaaaaa', linestyle=':', lw=1.5, alpha=0.8, zorder=1)
    if key_max and key_max in loose_b:
        ax.axvline(x=loose_b[key_max], color='#aaaaaa', linestyle=':', lw=1.5, alpha=0.8, zorder=1)
        
    if key_min and key_min in strict_b:
        ax.axvline(x=strict_b[key_min], color='white', linestyle='--', lw=2, alpha=0.9, zorder=2)
    if key_max and key_max in strict_b:
        ax.axvline(x=strict_b[key_max], color='white', linestyle='--', lw=2, alpha=0.9, zorder=2)

def _draw_scoring_curve_export(ax, f_key, params, xlims):
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
            
        ax.plot(x_grid, y_curve, color='#3498db', linestyle='-', linewidth=2.5, alpha=0.6, zorder=10)
    except Exception:
        pass


def _clean_path_features_for_json(path_features):
    """Bereinigt Numpy-Arrays aus allen 12 Kategorien-Listen für den JSON Export."""
    cleaned_dict = {}
    for path_key, obj_list in path_features.items():
        cleaned_list = []
        for item in obj_list:
            new_item = {}
            for k, v in item.items():
                if isinstance(v, np.ndarray): continue
                new_item[k] = v
            cleaned_list.append(new_item)
        cleaned_dict[path_key] = cleaned_list
    return cleaned_dict


def save_generation_results(gen, output_dir, generations_data, params, path_features, stats_table, hard_bounds, mismatch_distributions):
    gen_dir = output_dir / f"generation_{gen:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON Export
    export_dict = {
        "generation": gen,
        "parameters": params,
        "statistics": stats_table,
        "path_features": _clean_path_features_for_json(path_features),
        "mismatch_distributions": mismatch_distributions
    }
    with open(gen_dir / "generation_data.json", "w") as f:
        json.dump(export_dict, f, indent=4)

    # 2. Global Error & Matrix Export
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(C_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)
    
    ax_err = fig.add_subplot(gs[0, 0])
    _format_axis(ax_err, f"Global Error Convergence (Gen 1 to {gen})", "Generation", "Penalty Score")
    
    gens = generations_data.get("gen", [])
    fp_vals = generations_data.get("fp", [])
    fn_vals = generations_data.get("fn", [])
    
    if gens:
        ax_err.stackplot(gens, fn_vals, fp_vals, labels=["False Negatives", "False Positives"], colors=[C_PATHS["3_trash"], C_PATHS["1_good_fail"]], alpha=0.8)
        ax_err.set_xlim(left=max(1, min(gens)), right=max(2, max(gens)))
        ax_err.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        legend = ax_err.legend(loc="upper right", facecolor=C_BG, edgecolor=C_GRID, fontsize=10)
        for text in legend.get_texts(): text.set_color(C_TEXT)

    ax_tab = fig.add_subplot(gs[0, 1])
    ax_tab.axis('off')
    
    paths = stats_table.get("paths", {})
    table_data = [
        ["1: Good -> OK", str(paths.get("1_good_ok_gt", 0)), str(paths.get("1_good_ok_no_gt", 0))],
        ["1: Good -> FAIL", str(paths.get("1_good_fail_gt", 0)), str(paths.get("1_good_fail_no_gt", 0))],
        ["2: Rescued -> OK", str(paths.get("2_resc_ok_gt", 0)), str(paths.get("2_resc_ok_no_gt", 0))],
        ["2: Rescued -> FAIL", str(paths.get("2_resc_fail_gt", 0)), str(paths.get("2_resc_fail_no_gt", 0))],
        ["2: RF Aborted", str(paths.get("2_rf_trash_gt", 0)), str(paths.get("2_rf_trash_no_gt", 0))],
        ["3: Phase 1 Trash", str(paths.get("3_trash_gt", 0)), str(paths.get("3_trash_no_gt", 0))]
    ]
    cols = ["Path", "GT Matched", "No GT Match"]
    table = ax_tab.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    color_keys = list(C_PATHS.keys())
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C_GRID)
        if row == 0:
            cell.set_facecolor('#444444')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor(C_PLOT_BG)
            base_color = C_PATHS[color_keys[row-1]]
            if col == 0: cell.get_text().set_color(base_color)
            elif col == 1: cell.get_text().set_color(base_color)
            elif col == 2: cell.get_text().set_color(base_color)
            cell.get_text().set_alpha(1.0 if col == 1 else 0.5)

    plt.savefig(gen_dir / "01_global_error_and_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Ridge Plots (Feature Distributions in 6 Lanes)
    feature_maps = [
        ("area_px", "02_area_distribution.png", "Area (px)", False),
        ("aspect_ratio", "03_aspect_ratio_distribution.png", "Aspect Ratio", True),
        ("circularity", "04_circularity_distribution.png", "Circularity", False),
        ("ellipse_fill_factor", "05_ellipse_fill_distribution.png", "Ellipse Fill Factor", False),
        ("solidity", "06_solidity_distribution.png", "Solidity", False),
        ("core_brightness_75th", "07_core_brightness_distribution.png", "Core Brightness (75th Percentile)", False),
        ("core_rim_contrast", "08_contrast_ratio_distribution.png", "Core-to-Rim Contrast Ratio", False),
        ("total_score", "09_total_score_distribution.png", "Total Fuzzy Score", False)
    ]

    lanes = [
        "3_trash", 
        "2_rf_trash", 
        "2_resc_fail", 
        "2_resc_ok", 
        "1_good_fail", 
        "1_good_ok"
    ]
    lane_height = 1.0 / len(lanes)

    for f_key, fname, title, is_log in feature_maps:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(C_BG)
        _format_axis(ax, title, title, "Density / Category")
        ax.set_yticks([]) 
        ax.set_ylim(0, 1)
        
        all_values = []

        for i, path in enumerate(lanes):
            y_bottom = i * lane_height
            color = C_PATHS[path]
            
            v_gt = [obj[f_key] for obj in path_features.get(f"{path}_gt", []) if f_key in obj and obj[f_key] is not None]
            v_nogt = [obj[f_key] for obj in path_features.get(f"{path}_no_gt", []) if f_key in obj and obj[f_key] is not None]
            
            _plot_kde_and_rug_export(ax, v_nogt, y_bottom, lane_height, color, is_gt=False)
            _plot_kde_and_rug_export(ax, v_gt, y_bottom, lane_height, color, is_gt=True)
            
            all_values.extend(v_gt)
            all_values.extend(v_nogt)

        if all_values:
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

        _draw_bounds_export(ax, f_key)
        _draw_scoring_curve_export(ax, f_key, params, ax.get_xlim())

        if f_key == "total_score":
            threshold = params.get("total_score_threshold", 5.0)
            ax.axvline(x=threshold, color='white', linestyle='-', linewidth=2.5, zorder=10)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
        plt.savefig(gen_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
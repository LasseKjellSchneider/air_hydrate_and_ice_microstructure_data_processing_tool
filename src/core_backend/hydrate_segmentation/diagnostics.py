import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
import logging

DIAGNOSTICS_CONSTANTS = {
    "filter_max_dist_xy_um": 1000.0,
    "filter_min_dist_xy_um": 5.0,
    "filter_max_dist_z_layers": 3,
    
    "plot_axis_limit_um": 500.0,
    "plot_kde_threshold": 0.02,
    "plot_kde_levels": 25,
    "plot_kde_bw_adjust": 0.6,
    
    "plot_scatter_size": 4,
    "plot_scatter_alpha": 0.3
}

def plot_and_save_diagnostics(all_objects, processed_data_path, ellipse_params):
    try:
        vectors_above = []
        vectors_side_by_side = []
        vectors_leftup_rightdown = []
        vectors_leftdown_rightup = []
        vectors_rest = []
        distances = []

        for i, hydrate in enumerate(all_objects):
            if not hydrate.get("focus", False):
                continue

            for j, potential_duplicate in enumerate(all_objects):
                if j <= i or not potential_duplicate.get("focus", False):
                    continue

                dx = hydrate["center_x_um_world"] - potential_duplicate["center_x_um_world"]
                dy = hydrate["center_y_um_world"] - potential_duplicate["center_y_um_world"]
                dist_xy = math.hypot(dx, dy)

                if dist_xy > DIAGNOSTICS_CONSTANTS["filter_max_dist_xy_um"] or dist_xy < DIAGNOSTICS_CONSTANTS["filter_min_dist_xy_um"]:
                    continue

                dist_z = abs(hydrate["layer"] - potential_duplicate["layer"])
                if dist_z > DIAGNOSTICS_CONSTANTS["filter_max_dist_z_layers"]:
                    continue

                distances.append(dist_xy)

                try:
                    hyd_parts = hydrate["source_stack"].replace(".tif", "").split("_")
                    dup_parts = potential_duplicate["source_stack"].replace(".tif", "").split("_")

                    hydrate_gridx = int(hyd_parts[-2])
                    hydrate_gridy = int(hyd_parts[-1])
                    duplicate_gridx = int(dup_parts[-2])
                    duplicate_gridy = int(dup_parts[-1])
                except Exception:
                    continue

                dx_grid = hydrate_gridx - duplicate_gridx
                dy_grid = hydrate_gridy - duplicate_gridy
                vector = (dx, dy)

                if dx_grid == 0 and dy_grid == 1:
                    vectors_above.append(vector)
                elif dx_grid == 0 and dy_grid == -1:
                    vectors_above.append((-dx, -dy))
                elif dy_grid == 0 and dx_grid == 1:
                    vectors_side_by_side.append(vector)
                elif dy_grid == 0 and dx_grid == -1:
                    vectors_side_by_side.append((-dx, -dy))
                elif dx_grid == 1 and dy_grid == 1:
                    vectors_leftup_rightdown.append(vector)
                elif dx_grid == -1 and dy_grid == -1:
                    vectors_leftup_rightdown.append((-dx, -dy))
                elif dx_grid == 1 and dy_grid == -1:
                    vectors_leftdown_rightup.append(vector)
                elif dx_grid == -1 and dy_grid == 1:
                    vectors_leftdown_rightup.append((-dx, -dy))
                else:
                    vectors_rest.append(vector)

        diagnostics_data = {
            "vectors_above": vectors_above,
            "vectors_side_by_side": vectors_side_by_side,
            "vectors_leftup_rightdown": vectors_leftup_rightdown,
            "vectors_leftdown_rightup": vectors_leftdown_rightup,
            "vectors_rest": vectors_rest,
            "distances": distances,
        }

        with open(processed_data_path / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics_data, f, indent=4)

        category_config = [
            {"key": "vectors_above", "ell_key": "top_bottom", "title": "Top / Bottom (Y+)", "cmap": "Blues", "color": "blue"},
            {"key": "vectors_side_by_side", "ell_key": "left_right", "title": "Left / Right (X+)", "cmap": "Greens", "color": "green"},
            {"key": "vectors_leftup_rightdown", "ell_key": "topleft_bottomright", "title": "Diag: Top-Left -> Bot-Right", "cmap": "Reds", "color": "red"},
            {"key": "vectors_leftdown_rightup", "ell_key": "bottomleft_topright", "title": "Diag: Bot-Left -> Top-Right", "cmap": "Oranges", "color": "orange"}
        ]

        plot_variants = [
            {"mode": "heatmap", "filename": "vector_heatmaps_refined.png"},
            {"mode": "scatter", "filename": "vector_scatter_refined.png"},
        ]

        for variant in plot_variants:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, cfg in enumerate(category_config):
                ax = axes[i]
                data_points = diagnostics_data.get(cfg["key"], [])
                params = ellipse_params.get(cfg["ell_key"])
                
                if not params:
                    ax.set_title(f"{cfg['title']} - No Param Data")
                    continue

                if data_points:
                    df = pd.DataFrame(data_points, columns=["X", "Y"])

                    if variant["mode"] == "heatmap":
                        if len(df) > 5:
                            try:
                                sns.kdeplot(
                                    data=df, x="X", y="Y", ax=ax, fill=True, cmap=cfg["cmap"],
                                    thresh=DIAGNOSTICS_CONSTANTS["plot_kde_threshold"],
                                    levels=DIAGNOSTICS_CONSTANTS["plot_kde_levels"],
                                    bw_adjust=DIAGNOSTICS_CONSTANTS["plot_kde_bw_adjust"],
                                )
                            except Exception as kde_err:
                                logging.warning(f"KDE Plot failed for {cfg['title']} (Likely zero variance). Falling back to scatter. Error: {kde_err}")
                                ax.scatter(df["X"], df["Y"], c=cfg["color"], alpha=0.7)
                        else:
                            ax.scatter(df["X"], df["Y"], c=cfg["color"], alpha=0.7)
                    else:  
                        ax.scatter(
                            df["X"], df["Y"], c=cfg["color"],
                            s=DIAGNOSTICS_CONSTANTS["plot_scatter_size"], 
                            alpha=DIAGNOSTICS_CONSTANTS["plot_scatter_alpha"], 
                            edgecolors="none", rasterized=True, 
                        )

                cx, cy = params.get("cx", 0), params.get("cy", 0)
                width, height = params.get("width", 0), params.get("height", 0)
                angle_deg = params.get("angle_deg", 0)

                ell = Ellipse(
                    xy=(cx, cy), width=width, height=height, angle=angle_deg,
                    edgecolor="red", fc="None", lw=2, linestyle="-",
                )
                ax.add_patch(ell)
                ax.plot(cx, cy, "rx", markersize=6)

                ax.set_title(f"{cfg['title']} (n={len(data_points)})", fontsize=11, fontweight="bold")
                ax.set_xlabel("X Offset (µm)", fontsize=10)
                ax.set_ylabel("Y Offset (µm)", fontsize=10)

                ax.set_aspect("equal", adjustable="box")
                axis_limit = DIAGNOSTICS_CONSTANTS["plot_axis_limit_um"]
                ax.set_xlim(-axis_limit, axis_limit)
                ax.set_ylim(-axis_limit, axis_limit)

                ax.grid(True, linestyle="--", alpha=0.6)
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
                ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

            title_type = "KDE Density" if variant["mode"] == "heatmap" else "Raw Scatter Points"
            
            plt.suptitle(
                f"Stitching Offset Diagnostics & Filter Ellipses ({title_type})",
                fontsize=14, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(processed_data_path / variant["filename"], dpi=300)
            plt.close(fig)

    except Exception as e:
        logging.error(f"Critical error creating diagnostic plots: {e}", exc_info=True)
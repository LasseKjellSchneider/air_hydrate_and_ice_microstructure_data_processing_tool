import json
import sys
import logging
from pathlib import Path
from datetime import datetime

DEFAULT_PARAMETERS = {
    # Phase 1: image-based boundary detection
    "denoise_d": 7,
    "denoise_sigma_color": 50,
    "denoise_sigma_space": 50,
    "ridge_sigmas": [3, 4, 5],
    "ridge_black_ridges": True,
    "ridge_threshold": 40,
    "min_length": 200,                                     
    "ridge_dilation_kernel_size_bevor_filtering": 50,      
    "ridge_dilation_kernel_size_after_filtering": 100,     
    "extend_lines_px": 3,
    "ridge_dilation_iterations": 1,
    "max_hole_area_px": 500,
    "min_junction_neighbors": 3,
    "min_object_size_px": 5,
    "border_margin_px": 50,
    
    # Phase 2: morphology applied after random-forest classification
    "closing_dilation_kernel": 150,
    "closing_min_line_length": 200,
    "closing_repulsion_radius": 18,
    "closing_repulsion_penalty": 10000000,
    
    # Phase 3: path routing
    "max_radius": 4000,
    "t_junc_vector_length_px": 40, 
    "t_junc_cone_angle_deg": 90,
    "dijkstra_cost_multiplier": 300.0,
    "dijkstra_tophat_clip_percentile": 80.0,
    "dijkstra_darkness_factor": 0.65,
    
    # Phase 4: watershed segmentation and region merging
    "ws_scale_factor": 12,       # Downscaling factor used to limit memory use
    "ws_blur_kernel": 5,         # Smoothing kernel for the distance map
    "ws_min_seed_depth": 3.0,    # Minimum boundary distance for a watershed seed
    "ws_y_intercept": 5.0,       # Base threshold for merging watershed regions
    "ws_slope": 0.5,             # Threshold increase per pixel of seed distance
    "ws_cutoff": 50.0            # Maximum accepted watershed boundary length
}

def get_config_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parents[2]

def load_constants(custom_path=None):
    """
    Lädt entweder eine spezifisch übergebene JSON (custom_path) 
    oder fällt auf eine Standard-JSON zurück.
    """
    if custom_path and Path(custom_path).exists():
        config_file = Path(custom_path)
    else:
        config_file = get_config_dir() / "grain_boundary_constants.json"
    
    if not config_file.exists():
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_PARAMETERS, f, indent=4)
        except Exception:
            return DEFAULT_PARAMETERS.copy()
            
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            user_params = json.load(f)
            
        # Add parameters introduced after this configuration was created.
        updated = False
        for key, default_value in DEFAULT_PARAMETERS.items():
            if key not in user_params:
                user_params[key] = default_value
                updated = True
                
        if updated and not custom_path:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(user_params, f, indent=4)
                
        return user_params
    except Exception:
        return DEFAULT_PARAMETERS.copy()

def save_constants_with_timestamp(new_params, output_dir):
    """Speichert die finalen Tuner-Parameter mit Zeitstempel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"grain_boundary_constants_{timestamp}.json"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_params, f, indent=4)
        
    logging.info(f"Tuner Constants saved to: {file_path}")
    return str(file_path)
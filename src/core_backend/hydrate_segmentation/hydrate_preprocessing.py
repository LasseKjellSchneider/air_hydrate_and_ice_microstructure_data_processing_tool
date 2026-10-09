import json
import pathlib
import os
import cv2
import tifffile
import numpy as np
import logging
import time

logging.basicConfig(
    filename='debug_preprocessing.txt', 
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Shared preprocessing parameters
PREPROCESSING_CONSTANTS = {
    # Border-mask generation
    "mask_scale_factor": 4,
    "mask_cutoff_threshold": 120,
    "mask_safety_margin_um": 1000.0,
    "mask_blur_kernel": (51, 51),
    "mask_erosion_kernel": (3, 3),
    "mask_overlay_alpha": 0.6,
    
    # Flatfield correction
    "ff_percentile": 60,
    "ff_fallback_bg": 130.0,
    "ff_clip_min": 20.0,
    "ff_ice_pixel_min": 100,
    "ff_ice_pixel_max": 200,
    "ff_min_valid_pixels": 1000,
    "ff_target_brightness": 160.0,
    "ff_overlay_alpha": 0.3,
    "ff_overlay_beta": 0.7
}


def gui_information_about_input_data(data_folder):
    """
    Reads the metadata.json file from the specified data folder and extracts key information.
    Returns a formatted string with sample name, number of positions, and start time.
    """
    data_path = pathlib.Path(data_folder)
    metadata_path = None
    existing_tiff_files = 0

    # Use os.scandir for efficient directory traversal on network storage.
    with os.scandir(data_folder) as it:
        for entry in it:
            if entry.is_dir():
                if entry.name.startswith("Pos-"):
                    existing_tiff_files += 1
                
                if metadata_path is None:
                    with os.scandir(entry.path) as sub_it:
                        for sub_entry in sub_it:
                            if sub_entry.is_file() and os.path.splitext(sub_entry.name)[0] == "metadata":
                                metadata_path = pathlib.Path(sub_entry.path)
                                break

    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found in {data_folder}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    sample_name = metadata["Summary"]["Prefix"]
    num_positions = metadata["Summary"]["Positions"]
    start_time = metadata["Summary"]["StartTime"][:-13]

    info_string = f"Sample:\n {sample_name}\n\nPositions: {num_positions}\n\nExisting TIFF Files: {existing_tiff_files}\n\nStart Time: {start_time}"
    return info_string


def gui_preprocessing_microscope(data_folder, output_folder, 
                                 progress_stacks_cb=None, 
                                 progress_border_cb=None, 
                                 progress_flatfield_cb=None, 
                                 progress_correction_cb=None,
                                 init_progress_grid_cb=None,
                                 status_cb=None):
    """
    Main orchestrator for the preprocessing pipeline.
    Executes metadata creation, stacking, border masking, and flatfield correction sequentially.
    Logs execution times for performance profiling and updates the GUI status label.
    """
    try:
        print("\n" + "="*50)
        print(f"STARTING PREPROCESSING PIPELINE")
        print(f"Input:  {data_folder}")
        print(f"Output: {output_folder}")
        print("="*50 + "\n")
        
        logging.info(f"Starting preprocessing pipeline. Input: {data_folder}, Output: {output_folder}")
        timings = {}

        # 1. Metadata Generation
        if status_cb: status_cb("Step 1/4: Mapping coordinates & metadata...")
        print("[Step 1] Creating metadata and mapping TIFF files...")
        t0 = time.perf_counter()
        metadata_preprocessed, image_paths, processed_data_path = create_metadata_preprocessed(data_folder, output_folder)
        timings["1_create_metadata"] = time.perf_counter() - t0
        print(f"[Step 1] Completed in {timings['1_create_metadata']:.2f} seconds.\n")
        
        # Initialize the progress grid when the GUI provides a callback.
        if init_progress_grid_cb is not None:
            try:
                positions_list = [{"gridX": pos["gridX"], "gridY": pos["gridY"]} for pos in metadata_preprocessed["Positions"]]
                init_progress_grid_cb(positions_list)
            except Exception as e:
                print(f" [Warning] Could not initialize GUI progress grid: {e}")
        
        # 2. Stack Creation
        if status_cb: status_cb("Step 2/4: Building 3D TIFF stacks...")
        print("[Step 2] Initiating stack creation (building 3D TIFFs)...")
        t0 = time.perf_counter()
        create_stacks(data_folder, processed_data_path, image_paths, progress_stacks_cb)
        timings["2_create_stacks"] = time.perf_counter() - t0
        print(f"[Step 2] Completed in {timings['2_create_stacks']:.2f} seconds.\n")
        
        # 3. Border Mask Generation
        if status_cb: status_cb("Step 3/4: Generating border masks...")
        print("[Step 3] Initiating border mask generation...")
        t0 = time.perf_counter()
        metadata_preprocessed = create_border_masks(processed_data_path, metadata_preprocessed, progress_border_cb)
        timings["3_create_border_masks"] = time.perf_counter() - t0
        print(f"[Step 3] Completed in {timings['3_create_border_masks']:.2f} seconds.\n")
        
        # 4. Flatfield Correction
        if status_cb: status_cb("Step 4/4: Applying flatfield correction...")
        print("[Step 4] Initiating flatfield correction...")
        t0 = time.perf_counter()
        metadata_preprocessed = flatfield_correction(processed_data_path, metadata_preprocessed, progress_flatfield_cb, progress_correction_cb)
        timings["4_flatfield_correction"] = time.perf_counter() - t0
        print(f"[Step 4] Completed in {timings['4_flatfield_correction']:.2f} seconds.\n")

        if status_cb: status_cb("Saving final data...")
        # Persist the metadata generated by the preprocessing stages.
        path_metadata_preprocessed = processed_data_path / "metadata_preprocessed.json"
        with open(path_metadata_preprocessed, "w") as f:
            json.dump(metadata_preprocessed, f, indent=4)

        # Record the duration of each stage and the complete pipeline.
        logging.info("--- PREPROCESSING TIMING REPORT ---")
        for step, duration in sorted(timings.items()):
            logging.info(f"  -> {step}: {duration:.2f} seconds")
        logging.info(f"  -> TOTAL PIPELINE TIME: {sum(timings.values()):.2f} seconds")

        print("\n=== PREPROCESSING SUCCESSFULLY COMPLETED ===")
        if status_cb: status_cb("Preprocessing successfully completed!")
        return processed_data_path / "mosaic_control_layer0.jpg"

    except Exception as e:
        if status_cb: status_cb("Critical Error occurred! Check terminal.")
        print(f"\n!!! CRITICAL ERROR IN PREPROCESSING: {e} !!!")
        import traceback
        traceback.print_exc()
        logging.error("A critical error occurred during preprocessing!", exc_info=True)
        raise e

def create_metadata_preprocessed(data_folder, output_folder):
    """
    Parses raw MicroManager metadata, extracts real stage coordinates, and applies a 1D-clustering
    algorithm to robustly determine the physical imaging grid (immune to meander naming errors).
    """
    processed_data_path = pathlib.Path(output_folder) / pathlib.Path(data_folder).name
    processed_data_path.mkdir(parents=True, exist_ok=True)
    print(f" -> Output directory verified: {processed_data_path}")

    metadata_path = None
    pos_folders = []

    print(" -> Scanning directory for 'Pos-' folders and 'metadata.txt/json'...")
    with os.scandir(data_folder) as it:
        for entry in it:
            if entry.is_dir():
                if entry.name.startswith("Pos-"):
                    pos_folders.append(entry)
                if metadata_path is None:
                    with os.scandir(entry.path) as sub_it:
                        for sub_entry in sub_it:
                            if sub_entry.is_file() and os.path.splitext(sub_entry.name)[0] == "metadata":
                                metadata_path = pathlib.Path(sub_entry.path)
                                print(f" -> Metadata file found: {metadata_path}")
                                break

    if metadata_path is None:
        raise FileNotFoundError("No metadata file found inside the 'Pos-' directories!")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(" -> Metadata JSON successfully loaded into memory.")
    
    summary_preprocessed = {
        "sample": metadata["Summary"]["Prefix"],
        "positions": metadata["Summary"]["Positions"],
        "startTime": metadata["Summary"]["StartTime"]
    }

    Camera_preprocessed = {}
    for key, value in metadata.items():
        if key.startswith("Metadata-Pos"):
            Camera_preprocessed = {
                "pixelType": value["PixelType"],
                "width": value["Width"],
                "height": value["Height"],
                "pixelSizeUm": value["PixelSizeUm"],
                "bitDepth": value["BitDepth"],
                "exposure-ms": value["Exposure-ms"]
            }
            break

    print(" -> Extracting physical stage coordinates based on folder labels...")
    raw_positions = []
    
    # Collect physical stage coordinates before assigning grid indices.
    for key, value in metadata.items():
        if key.startswith("Summary"):
            for sub_key, sub_value in value.items():
                if sub_key == "StagePositions":
                    for StagePosition in sub_value:
                        try:
                            label = StagePosition["Label"]  # Position labels match the acquisition folder names.
                            positionX = StagePosition["DevicePositions"][0]["Position_um"][0]
                            positionY = StagePosition["DevicePositions"][0]["Position_um"][1]
                            
                            if len(StagePosition["DevicePositions"]) > 1:
                                # Correct the hardware-reported Z coordinate, which is ten times too large.
                                positionZ = StagePosition["DevicePositions"][1]["Position_um"][0] / 10.0
                            else:
                                positionZ = 0.0
                                
                            raw_positions.append({
                                "label": label,
                                "x": positionX,
                                "y": positionY,
                                "z": positionZ
                            })
                        except Exception as e:
                            print(f" [Warning] Could not parse StagePosition {StagePosition.get('Label', 'Unknown')}: {e}")

    print(f" -> Extracted {len(raw_positions)} positions from metadata.")

    print(" -> Initiating 1D-clustering of physical coordinates (fixing serpentine/meander errors)...")
    def cluster_coords(coords_list, tolerance=50.0):
        """Groups physical coordinates that fall within the mechanical stage tolerance."""
        sorted_c = sorted(list(set(coords_list)))
        clusters = []
        for c in sorted_c:
            if not clusters or abs(c - np.mean(clusters[-1])) > tolerance:
                clusters.append([c])
            else:
                clusters[-1].append(c)
        return sorted([np.mean(cl) for cl in clusters])

    all_x = [p["x"] for p in raw_positions]
    all_y = [p["y"] for p in raw_positions]
    
    x_clusters = cluster_coords(all_x)
    y_clusters = cluster_coords(all_y)
    print(f" -> Real axes established: {len(x_clusters)} X-columns and {len(y_clusters)} Y-rows.")

    # Map each position to the nearest physical X/Y grid coordinate.
    grouped_by_xy = {}
    for p in raw_positions:
        gx = min(range(len(x_clusters)), key=lambda i: abs(x_clusters[i] - p["x"]))
        gy = min(range(len(y_clusters)), key=lambda i: abs(y_clusters[i] - p["y"]))
        p["gridX"] = gx
        p["gridY"] = gy
        
        key = (gx, gy)
        if key not in grouped_by_xy:
            grouped_by_xy[key] = []
        grouped_by_xy[key].append(p)

    # Group positions by grid cell and assign their Z-layer indices.
    Positions_all = []
    for (gx, gy), pos_list in grouped_by_xy.items():
        # Layer 0 represents the highest Z coordinate in each grid cell.
        pos_list_sorted = sorted(pos_list, key=lambda item: item["z"], reverse=True)
        
        z_list = []
        for z_idx, p_dict in enumerate(pos_list_sorted):
            p_dict["gridZ"] = z_idx  # Store the position's index within the stack.
            z_list.append(p_dict["z"])
            
        z_layers_count = len(z_list)
        z_step_um = 0.0
        if z_layers_count > 1:
            z_step_um = round(abs(z_list[1] - z_list[0]), 4)
            
        Positions_all.append({
            "gridX": gx,
            "gridY": gy,
            "positionX": pos_list_sorted[0]["x"], 
            "positionY": pos_list_sorted[0]["y"],
            "positionZ_start": z_list[0] if z_list else 0.0,
            "z_layers": z_layers_count,
            "z_step_um": z_step_um,
            "positionZ_list": z_list,
            "border": False
        })

    print(" -> Mapping TIFF images from directories to the correct 3D arrays...")
    image_paths = {}
    mapped_files_count = 0

    # Resolve acquisition folder names to their assigned grid coordinates.
    label_to_pos = {p["label"]: p for p in raw_positions}

    for entry in pos_folders:
        folder_name = entry.name
        if folder_name in label_to_pos:
            mapped_pos = label_to_pos[folder_name]
            x = mapped_pos["gridX"]
            y = mapped_pos["gridY"]
            z_idx = mapped_pos["gridZ"]

            # Use the first TIFF file found in the position directory.
            with os.scandir(entry.path) as sub_it:
                for sub_entry in sub_it:
                    if sub_entry.is_file() and sub_entry.name.endswith(".tif"):
                        if x not in image_paths:
                            image_paths[x] = {}
                        if y not in image_paths[x]:
                            image_paths[x][y] = {}
                            
                        image_paths[x][y][z_idx] = pathlib.Path(sub_entry.path)
                        mapped_files_count += 1
                        break

    print(f" -> Mapping completed! {mapped_files_count} TIFF files were successfully mapped.")

    # Omit grid cells that have no corresponding TIFF files.
    available_positions = [
        pos for pos in Positions_all 
        if pos["gridX"] in image_paths and pos["gridY"] in image_paths[pos["gridX"]]
    ]
    
    available_positions = sorted(available_positions, key=lambda p: (p["gridX"], p["gridY"]))

    metadata_preprocessed = {
        "summary_preprocessed": summary_preprocessed, 
        "Positions": available_positions, 
        "Camera_preprocessed": Camera_preprocessed
    }

    return metadata_preprocessed, image_paths, processed_data_path


def create_stacks(data_folder, processed_data_path, image_paths, progress_stacks_cb=None):
    """
    Merges individual 2D images into multi-layer 3D TIFF stacks.
    Mixes red and green color channels equally to optimize dynamic range for ice structures.
    """
    total_stacks = sum(len(y_dict) for y_dict in image_paths.values())
    current_stack = 0
    
    for x in sorted(image_paths.keys()):
        for y in sorted(image_paths[x].keys()):
            image_stack = []
            for current_z in sorted(image_paths[x][y].keys()):
                img_path = str(image_paths[x][y][current_z])
                
                # Read bytes through Python's file API before decoding with OpenCV.
                with open(img_path, "rb") as f:
                    img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                
                if img_bgr is None:
                    raise ValueError(f"Could not read image: {img_path}")
    
                # Average the red and green channels to retain ice-structure contrast.
                red = img_bgr[:, :, 2].astype(np.float32)
                green = img_bgr[:, :, 1].astype(np.float32)
                img_gray = ((red * 0.5) + (green * 0.5)).astype(np.uint8)
    
                image_stack.append(img_gray)
                
            image_stack_array = np.stack(image_stack, axis=0)
            tifffile.imwrite(processed_data_path / f"{pathlib.Path(data_folder).name}_{x}_{y}.tif", image_stack_array, photometric='minisblack')
            
            current_stack += 1
            if progress_stacks_cb is not None:
                progress_stacks_cb(current_stack / total_stacks, x, y)
                

def create_border_masks(processed_data_path, metadata_preprocessed, progress_border_cb=None):
    """
    Creates a global mask of the sample to identify active ice areas and marks empty border stacks.
    Uses configurable constants for downscaling and erosion to prevent edge artifacts.
    """
    positions = metadata_preprocessed["Positions"]
    cam_meta = metadata_preprocessed["Camera_preprocessed"]
    width = int(cam_meta["width"])
    height = int(cam_meta["height"])
    pixel_size = float(cam_meta["pixelSizeUm"])
   
    scale = PREPROCESSING_CONSTANTS["mask_scale_factor"]
    min_pos_x = min([p["positionX"] for p in positions])
    min_pos_y = min([p["positionY"] for p in positions])
    max_pos_x = max([p["positionX"] for p in positions])
    max_pos_y = max([p["positionY"] for p in positions])
   
    global_w = int((max_pos_x - min_pos_x) / pixel_size / scale) + int(width / scale) + 5
    global_h = int((max_pos_y - min_pos_y) / pixel_size / scale) + int(height / scale) + 5
   
    global_stitch = np.zeros((global_h, global_w), dtype=np.uint8)
    sample_file = next(pathlib.Path(processed_data_path).glob("*.tif"))
    with tifffile.TiffFile(str(sample_file)) as tif:
        z_layers_total = len(tif.pages)
        layer_middle = z_layers_total // 2
   
    for pos in positions:
        x, y = pos["gridX"], pos["gridY"]
        stack_path = processed_data_path / f"{processed_data_path.name}_{x}_{y}.tif"
        
        with tifffile.TiffFile(str(stack_path)) as tif:
            safe_middle = min(layer_middle, len(tif.pages) - 1)
            mid_img = tif.pages[safe_middle].asarray()
            
        small_img = cv2.resize(mid_img, (int(width/scale), int(height/scale)), interpolation=cv2.INTER_AREA)
        px_x = int((pos["positionX"] - min_pos_x) / pixel_size / scale)
        px_y = int((pos["positionY"] - min_pos_y) / pixel_size / scale)
        end_y = px_y + small_img.shape[0]
        end_x = px_x + small_img.shape[1]
        
        global_stitch[px_y:end_y, px_x:end_x] = np.maximum(global_stitch[px_y:end_y, px_x:end_x], small_img)

    cutoff_threshold = PREPROCESSING_CONSTANTS["mask_cutoff_threshold"]
    safety_margin_um = PREPROCESSING_CONSTANTS["mask_safety_margin_um"]

    blurred = cv2.GaussianBlur(global_stitch, PREPROCESSING_CONSTANTS["mask_blur_kernel"], 0)
    _, binary = cv2.threshold(blurred, cutoff_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    global_mask = np.zeros_like(global_stitch, dtype=np.uint8)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(global_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        um_per_pixel_downscaled = pixel_size * scale
        erosion_pixels = int(safety_margin_um / um_per_pixel_downscaled)
        global_mask = cv2.erode(global_mask, np.ones(PREPROCESSING_CONSTANTS["mask_erosion_kernel"], np.uint8), iterations=erosion_pixels)

    stitched_bgr = cv2.cvtColor(global_stitch, cv2.COLOR_GRAY2BGR)
    overlay_red = stitched_bgr.copy()
    overlay_red[global_mask == 0] = [0, 0, 255]
    alpha = PREPROCESSING_CONSTANTS["mask_overlay_alpha"]
    
    preview_image = cv2.addWeighted(stitched_bgr, alpha, overlay_red, 1 - alpha, 0)
    preview_path = processed_data_path / "global_mask_preview.png"
    cv2.imwrite(str(preview_path), preview_image)
    cv2.imwrite(str(processed_data_path / "global_sample_mask.png"), global_mask)

    masks_dir = processed_data_path / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    total_positions = len(positions)
    for idx, pos in enumerate(positions): 
        x, y = pos["gridX"], pos["gridY"]
        
        px_x = int((pos["positionX"] - min_pos_x) / pixel_size / scale)
        px_y = int((pos["positionY"] - min_pos_y) / pixel_size / scale)
       
        small_w, small_h = int(width/scale), int(height/scale)
        end_y = px_y + small_h
        end_x = px_x + small_w
        
        local_small_mask = global_mask[px_y:end_y, px_x:end_x]
        local_mask_full = cv2.resize(local_small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(masks_dir / f"mask_{x}_{y}.png"), local_mask_full)
       
        is_border = (cv2.countNonZero(local_mask_full) < (width * height))
        pos["border"] = bool(is_border)

        if progress_border_cb is not None:
            progress_border_cb((idx + 1) / total_positions, x, y, is_border)

    total_valid_pixels_downscaled = int(cv2.countNonZero(global_mask))
    total_valid_area_px = int(total_valid_pixels_downscaled * (scale ** 2))
    total_mask_area_mm2 = total_valid_area_px * (pixel_size ** 2) / 1_000_000.0

    metadata_preprocessed["Analytics"] = {
        "total_analyzed_ice_area_px": total_valid_area_px,
        "total_mask_area_mm2": total_mask_area_mm2
    }
    
    metadata_preprocessed["MosaicSetup"] = {
        "global_w": int(global_w), "global_h": int(global_h), "scale": int(scale),
        "min_pos_x": float(min_pos_x), "min_pos_y": float(min_pos_y), "pixel_size": float(pixel_size)
    }
    
    return metadata_preprocessed


def flatfield_correction(processed_data_path, metadata_preprocessed, progress_ff_cb=None, progress_corr_cb=None):
    """
    Computes global Z-layer flatfields and applies them to normalize illumination.
    Dynamically scales image intensity based on valid ice pixels to maintain constant brightness.
    """
    positions = metadata_preprocessed["Positions"]
    masks_dir = processed_data_path / "masks"
    
    flatfields_save_dir = processed_data_path / "flatfields"
    flatfields_save_dir.mkdir(exist_ok=True)

    clean_positions = [p for p in positions if not p["border"]]

    sample_file = next(processed_data_path.glob("*.tif"))
    with tifffile.TiffFile(str(sample_file)) as tif:
        z_layers = len(tif.pages)
        height, width = tif.pages[0].shape

    flatfields = []
    target_means = []

    for z in range(z_layers):
        slice_buffer = []
        for p in clean_positions:
            stack_path = processed_data_path / f"{processed_data_path.name}_{p['gridX']}_{p['gridY']}.tif"
            with tifffile.TiffFile(str(stack_path)) as tif:
                if z < len(tif.pages):
                    slice_buffer.append(tif.pages[z].asarray())
        
        if slice_buffer:
            slice_stack = np.stack(slice_buffer, axis=0)
            ff_bg = np.percentile(slice_stack, PREPROCESSING_CONSTANTS["ff_percentile"], axis=0)
        else:
            ff_bg = np.ones((height, width), dtype=np.float64) * PREPROCESSING_CONSTANTS["ff_fallback_bg"]
            
        layer_target_mean = np.mean(ff_bg)
        flatfields.append(ff_bg)
        target_means.append(layer_target_mean)
        
        ff_img = np.clip(ff_bg, 0, 255).astype(np.uint8)
        cv2.imwrite(str(flatfields_save_dir / f"flatfield_z{z}.png"), ff_img)

        if progress_ff_cb is not None:
            progress_ff_cb((z + 1) / z_layers)

    mosaic_setup = metadata_preprocessed["MosaicSetup"]
    mosaic_w, mosaic_h, mosaic_scale = mosaic_setup["global_w"], mosaic_setup["global_h"], mosaic_setup["scale"]
    global_mosaic_bgr = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    total_positions = len(positions)
    for idx, p in enumerate(positions):
        stack_path = processed_data_path / f"{processed_data_path.name}_{p['gridX']}_{p['gridY']}.tif"
        temp_path = processed_data_path / f"{processed_data_path.name}_{p['gridX']}_{p['gridY']}_temp.tif"
        
        mask_path = masks_dir / f"mask_{p['gridX']}_{p['gridY']}.png"
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        with tifffile.TiffFile(str(stack_path)) as tif, tifffile.TiffWriter(str(temp_path)) as writer:
            pages_count = len(tif.pages)
            for z in range(pages_count):
                frame = tif.pages[z].asarray()
                z_ff = min(z, len(flatfields) - 1)
                flatfield_safe = np.clip(flatfields[z_ff], PREPROCESSING_CONSTANTS["ff_clip_min"], 255.0)
                
                # Correct spatial illumination variation using the layer flatfield.
                standard_corrected = (frame.astype(np.float32) / flatfield_safe) * target_means[z_ff]
                
                # Estimate a per-frame scale from pixels within the expected ice range.
                p_min = PREPROCESSING_CONSTANTS["ff_ice_pixel_min"]
                p_max = PREPROCESSING_CONSTANTS["ff_ice_pixel_max"]
                ice_pixels = standard_corrected[(standard_corrected >= p_min) & (standard_corrected <= p_max)]
                
                if ice_pixels.size > PREPROCESSING_CONSTANTS["ff_min_valid_pixels"]:
                    current_ice_median = np.median(ice_pixels)
                else:
                    current_ice_median = target_means[z_ff]
                
                # Normalize the corrected frame to the target ice brightness.
                scaling_factor = PREPROCESSING_CONSTANTS["ff_target_brightness"] / current_ice_median
                
                corrected = standard_corrected * scaling_factor
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                
                if z == 0:
                    corr_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
                    
                    if mask is not None:
                        overlay = corr_bgr.copy()
                        overlay[mask == 0] = [0, 0, 255]
                        alpha = PREPROCESSING_CONSTANTS["ff_overlay_alpha"]
                        beta = PREPROCESSING_CONSTANTS["ff_overlay_beta"]
                        cv2.addWeighted(overlay, alpha, corr_bgr, beta, 0, corr_bgr)
                        
                        contours, _ = cv2.findContours((mask == 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(corr_bgr, contours, -1, (0, 0, 255), 3)
                        
                    small_bgr = cv2.resize(corr_bgr, (int(width/mosaic_scale), int(height/mosaic_scale)), interpolation=cv2.INTER_NEAREST)
                    px_x = int((p["positionX"] - mosaic_setup["min_pos_x"]) / mosaic_setup["pixel_size"] / mosaic_scale)
                    px_y = int((p["positionY"] - mosaic_setup["min_pos_y"]) / mosaic_setup["pixel_size"] / mosaic_scale)
                    
                    end_y = px_y + small_bgr.shape[0]
                    end_x = px_x + small_bgr.shape[1]
                    
                    global_mosaic_bgr[px_y:end_y, px_x:end_x] = small_bgr

                if mask is not None:
                    corrected = cv2.bitwise_and(corrected, corrected, mask=mask)
                
                writer.write(corrected, photometric='minisblack')
                
        # Replace the source stack only after the corrected temporary stack is complete.
        stack_path.unlink()
        temp_path.replace(stack_path)

        if progress_corr_cb is not None:
            progress_corr_cb((idx + 1) / total_positions, p["gridX"], p["gridY"])

    mosaic_save_path = processed_data_path / "mosaic_control_layer0.jpg"
    cv2.imwrite(str(mosaic_save_path), global_mosaic_bgr)
    logging.info(f"Generated visual mask overlay mosaic: {mosaic_save_path}")
        
    return metadata_preprocessed
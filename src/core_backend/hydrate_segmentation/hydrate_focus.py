import cv2
import numpy as np
import pathlib
import json
import logging

def _compute_overlap(obj1, obj2):
    """
    Calculates the exact pixel intersection area of two perspective-corrected contours.
    Returns the overlap ratio based on the area of the smaller object.
    """
    if "shifted_bbox" not in obj1 or "shifted_bbox" not in obj2:
        return 0.0

    x1, y1, w1, h1 = obj1["shifted_bbox"]
    x2, y2, w2, h2 = obj2["shifted_bbox"]

    # Quick bounding box collision check to save processing time
    if x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1:
        return 0.0

    min_x, min_y = min(x1, x2), min(y1, y2)
    max_x, max_y = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)

    mask_w, mask_h = max_x - min_x, max_y - min_y
    if mask_w <= 0 or mask_h <= 0:
        return 0.0

    mask1 = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask2 = np.zeros((mask_h, mask_w), dtype=np.uint8)

    cv2.drawContours(mask1, [obj1["shifted_contour"] - [min_x, min_y]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [obj2["shifted_contour"] - [min_x, min_y]], -1, 255, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(mask1, mask2)
    overlap_area = cv2.countNonZero(intersection)

    min_area = min(obj1["area_px"], obj2["area_px"])
    return overlap_area / min_area if min_area > 0 else 0.0


def calculate_focus_metrics(obj, stack, parameters):
    """
    Evaluates the optical sharpness of a hydrate candidate by analyzing its region of interest (ROI)
    using contrast variance, Laplacian energy, and Sobel edge gradients.
    """
    cx = int(obj["center_x_px_image"])
    cy = int(obj["center_y_px_image"])
    layer = int(obj["layer"])

    r = 2 * ((obj["area_px"] * obj["aspect_ratio"] / 3.14) ** 0.5) * parameters["roi_radius_factor"]

    img_height, img_width = stack.shape[1], stack.shape[2]

    x_start = max(0, cx - int(r))
    x_end = min(img_width, cx + int(r))
    y_start = max(0, cy - int(r))
    y_end = min(img_height, cy + int(r))

    roi = stack[layer, y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return 0.0

    score_contrast = float(np.std(roi))
    score_laplacian = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    score_thickness_edge = float(np.mean(gradient ** 2))

    total_score = (
        parameters["factor_contrast"] * score_contrast
        + parameters["factor_laplacian"] * score_laplacian
        + parameters["factor_thickness_edge"] * score_thickness_edge
    )
    return total_score


def find_focus(objects, stack, parameters, overlap_threshold=0.20):
    """
    Groups vertically overlapping objects across the Z-stack and identifies the single 
    sharpest layer representation for each physical hydrate.
    """
    assigned_ids = set()

    for obj in objects:
        if obj["id"] in assigned_ids:
            continue
        assigned_ids.add(obj["id"])
        hydrate_stack = [obj]

        for other_object in objects:
            if other_object["id"] not in assigned_ids:
                if _compute_overlap(obj, other_object) >= overlap_threshold:
                    hydrate_stack.append(other_object)
                    assigned_ids.add(other_object["id"])

        highest_focus_value = -1.0
        best_focus_object = None
        
        for focus_object in hydrate_stack:
            focus_object["focus_value"] = calculate_focus_metrics(focus_object, stack, parameters)
            if focus_object["focus_value"] > highest_focus_value:
                highest_focus_value = focus_object["focus_value"]
                best_focus_object = focus_object

        for focus_object in hydrate_stack:
            focus_object["focus"] = (focus_object == best_focus_object)


def _get_roi(obj, stack, parameters):
    """Internal helper to extract the exact ROI bounding box and image data."""
    cx = int(obj["center_x_px_image"])
    cy = int(obj["center_y_px_image"])
    layer = int(obj["layer"])
    r = 2 * ((obj["area_px"] * obj["aspect_ratio"] / 3.14) ** 0.5) * parameters["roi_radius_factor"]
    
    img_height, img_width = stack.shape[1], stack.shape[2]
    x_start = max(0, cx - int(r))
    x_end = min(img_width, cx + int(r))
    y_start = max(0, cy - int(r))
    y_end = min(img_height, cy + int(r))
    roi = stack[layer, y_start:y_end, x_start:x_end]
    return x_start, x_end, y_start, y_end, roi


def _build_candidate_panel(hydrate_stack, stack, parameters):
    """Internal helper to build a horizontal visual comparison panel for focus annotation."""
    images = []
    for obj in hydrate_stack:
        _, _, _, _, roi = _get_roi(obj, stack, parameters)
        if roi.size == 0:
            roi = np.zeros((64, 64), dtype=np.uint8)
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if len(roi.shape) == 2 else roi.copy()

        text = f"l: {obj['layer']}  s: {obj.get('focus_value', 0):.1f}"
        cv2.rectangle(roi_color, (0, 0), (roi_color.shape[1] - 1, roi_color.shape[0] - 1), (255, 0, 0), 2)
        cv2.putText(roi_color, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(roi_color)

    max_height = max(img.shape[0] for img in images)
    padded_images = []
    for img in images:
        pad_height = max_height - img.shape[0]
        padded = np.vstack((img, np.zeros((pad_height, img.shape[1], 3), dtype=img.dtype))) if pad_height > 0 else img
        padded_images.append(padded)

    return np.hstack(padded_images)


def label_focus_candidates(objects, stack, parameters, stack_path, save_dir):
    """
    Interactive GUI tool generating a panel of ROIs across layers for the same object 
    using strict area overlap tracking, allowing the user to label the true focal plane.
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    focus_truth = {
        "stack_file": pathlib.Path(stack_path).name,
        "detection_parameters": parameters,
        "focus_candidates": []
    }

    assigned_ids = set()
    group_index = 0

    for obj in objects:
        if obj["id"] in assigned_ids:
            continue

        group = [obj]
        assigned_ids.add(obj["id"])

        for other in objects:
            if other["id"] in assigned_ids:
                continue
            if other["layer"] != obj["layer"]:
                # strict 20% area overlap required to consider it the same hydrate
                if _compute_overlap(obj, other) >= 0.20:
                    group.append(other)
                    assigned_ids.add(other["id"])

        if len(group) <= 1:
            continue

        group_index += 1
        group.sort(key=lambda item: item["layer"])
        
        panel = _build_candidate_panel(group, stack, parameters)
        window_title = f"Focus candidates {pathlib.Path(stack_path).stem} group {group_index}"
        cv2.imshow(window_title, panel)
        cv2.waitKey(1)

        logging.info(f"--- Focus candidate group {group_index} ---")
        for idx, candidate in enumerate(group):
            logging.info(f"[{idx}] layer {candidate['layer']}  center=({candidate['center_x_px_image']:.1f},{candidate['center_y_px_image']:.1f})")
            
        choice = input("Choose best focus index, 's' to skip, 'q' to quit: ").strip().lower()
        cv2.destroyWindow(window_title)

        if choice == "q":
            break
        if choice in ("s", ""):
            continue

        try:
            sel = int(choice)
            if 0 <= sel < len(group):
                best_layer = group[sel]["layer"]
            else:
                continue
        except ValueError:
            continue

        hydrate_roi_dir = save_dir / f"{pathlib.Path(stack_path).stem}_group_{group_index}"
        hydrate_roi_dir.mkdir(parents=True, exist_ok=True)
        
        layer_data_list = []
        for candidate in group:
            _, _, _, _, roi = _get_roi(candidate, stack, parameters)
            roi_filename = f"layer_{candidate['layer']}.npy"
            np.save(hydrate_roi_dir / roi_filename, roi)
            
            layer_data_list.append({
                "id": candidate["id"],
                "layer": candidate["layer"],
                "roi_file": str(hydrate_roi_dir.name + "/" + roi_filename)
            })

        focus_truth["focus_candidates"].append({
            "group_id": group_index,
            "best_layer": best_layer,
            "roi_folder": str(hydrate_roi_dir.name),
            "layers": layer_data_list
        })

    output_path = save_dir / f"focus_ground_truth_{pathlib.Path(stack_path).stem}.json"
    with open(output_path, "w") as f:
        json.dump(focus_truth, f, indent=4)

    return output_path
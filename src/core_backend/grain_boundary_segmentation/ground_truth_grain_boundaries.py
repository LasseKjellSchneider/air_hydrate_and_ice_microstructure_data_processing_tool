import cv2
import numpy as np
import tifffile
from pathlib import Path

def label_objects_manually(ground_truth, border_margin=400, target_display_height=1275):
    """
    Interaktives UI-Tool zum Labeln von gefundenen Kanten-Objekten.
    Erweitert um Alpha-Blending, dickere Konturen und eine Post-Labeling Übersicht.
    """
    window_name = "Interactive Object Labeling"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for item in ground_truth:
        with tifffile.TiffFile(item["image_path"]) as tif:
            full_img = tif.asarray()
        
        # Restrict the annotation view to the object's source-image crop.
        gui_img_chunk = full_img[item["y_start"]:item["y_end"], item["x_start"]:item["x_end"]]
        ch, cw = gui_img_chunk.shape[:2]
        
        if gui_img_chunk.dtype != np.uint8:
            gui_img_chunk = cv2.normalize(gui_img_chunk, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Prepare the source image and mark the excluded border region.
        base_bgr = cv2.cvtColor(gui_img_chunk, cv2.COLOR_GRAY2BGR)
        overlay_margin = base_bgr.copy()
        cv2.rectangle(overlay_margin, (0, 0), (cw, border_margin), (0, 0, 255), -1)
        cv2.rectangle(overlay_margin, (0, ch - border_margin), (cw, ch), (0, 0, 255), -1)
        cv2.rectangle(overlay_margin, (0, border_margin), (border_margin, ch - border_margin), (0, 0, 255), -1)
        cv2.rectangle(overlay_margin, (cw - border_margin, border_margin), (cw, ch - border_margin), (0, 0, 255), -1)
        cv2.rectangle(overlay_margin, (border_margin, border_margin), (cw - border_margin, ch - border_margin), (0, 0, 255), 4)
        
        # Keep the underlying image visible beneath the border overlay.
        base_with_margin = cv2.addWeighted(overlay_margin, 0.3, base_bgr, 0.7, 0)

        labeled_objects = []
        objects_to_label = item.get("objects", [])
        total_objs = len(objects_to_label)
        abort = False

        # Review and label each detected object.
        for idx, obj in enumerate(objects_to_label, 1):
            if abort:
                break
            
            coords = np.array(obj["coordinates"])
            if len(coords) == 0:
                obj["label"] = 4  # Default to noise if empty
                labeled_objects.append(obj)
                continue

            # Add context around the object in the local review crop.
            buffer_px = 600
            y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
            x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            
            y0, y1 = max(0, y_min - buffer_px), min(ch, y_max + buffer_px)
            x0, x1 = max(0, x_min - buffer_px), min(cw, x_max + buffer_px)

            local_base = base_with_margin[y0:y1, x0:x1].copy()
            
            # Thicken the object mask so it remains visible in the review panel.
            local_coords = coords - [x0, y0]
            obj_mask = np.zeros(local_base.shape[:2], dtype=np.uint8)
            obj_mask[local_coords[:, 1], local_coords[:, 0]] = 1
            obj_mask = cv2.dilate(obj_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            
            # Highlight the current object in yellow.
            overlay_obj = local_base.copy()
            overlay_obj[obj_mask == 1] = [0, 255, 255]
            
            # Blend the highlight with the image to retain visible ice structure.
            display_img = cv2.addWeighted(overlay_obj, 0.6, local_base, 0.4, 0)
            
            # Scale the crop to the available review-window dimensions.
            h_loc, w_loc = display_img.shape[:2]
            sf = target_display_height / h_loc
            if w_loc * sf > 1800: sf = 1800 / w_loc
            
            new_w_loc, new_h_loc = int(w_loc * sf), int(h_loc * sf)
            interp = cv2.INTER_AREA if sf < 1.0 else cv2.INTER_CUBIC
            scaled_display = cv2.resize(display_img, (new_w_loc, new_h_loc), interpolation=interp)

            # Add the keyboard-labeling controls beside the image.
            ui_width = 400
            ui_height = max(new_h_loc, 600)
            ui_panel = np.full((ui_height, ui_width, 3), (40, 40, 40), dtype=np.uint8)
            
            text_lines = [
                f"Object {idx} / {total_objs}",
                f"Length: {obj.get('length_px', 0)} px",
                "",
                "[1] GOOD (Grain Boundary)",
                "[2] BAD (Subgrain)",
                "[3] BAD (Bottom Face)",
                "[4] BAD (Noise / Artifact)",
                "",
                "[q] Abort & Save"
            ]

            y_pos = 50
            for i, line in enumerate(text_lines):
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                cv2.putText(ui_panel, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2 if i==0 else 1)
                y_pos += 45
                
            if new_h_loc < ui_height:
                scaled_display = cv2.copyMakeBorder(scaled_display, 0, ui_height - new_h_loc, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                
            final_display = np.hstack((scaled_display, ui_panel))
            cv2.imshow(window_name, final_display)
            
            # Wait for one of the supported label or abort keys.
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('1'):
                    obj["label"] = 1
                    labeled_objects.append(obj)
                    break
                elif key == ord('2'):
                    obj["label"] = 2
                    labeled_objects.append(obj)
                    break
                elif key == ord('3'):
                    obj["label"] = 3
                    labeled_objects.append(obj)
                    break
                elif key == ord('4'):
                    obj["label"] = 4
                    labeled_objects.append(obj)
                    break
                elif key == ord('q') or key == 27:
                    abort = True
                    break
        
        item["objects"] = labeled_objects

        # Show an overview of accepted grain-boundary objects.
        if not abort:
            overview_base = base_with_margin.copy()
            overview_overlay = overview_base.copy()
            
            # Mark every object assigned to the grain-boundary class.
            for obj in labeled_objects:
                if obj.get("label") == 1:
                    coords = np.array(obj["coordinates"])
                    if len(coords) > 0:
                        for x, y in coords:
                            # Use overlapping circles to render a continuous thick line.
                            cv2.circle(overview_overlay, (int(x), int(y)), 5, (0, 255, 0), -1) 
            
            # Blend the accepted-object overlay with the source image.
            overview_blended = cv2.addWeighted(overview_overlay, 0.6, overview_base, 0.4, 0)
            
            # Fit the overview within the review window.
            sf_ov = target_display_height / ch
            if cw * sf_ov > 1800: sf_ov = 1800 / cw
            new_w_ov, new_h_ov = int(cw * sf_ov), int(ch * sf_ov)
            
            outro_img = cv2.resize(overview_blended, (new_w_ov, new_h_ov), interpolation=cv2.INTER_AREA)
            
            # Add completion instructions to the overview.
            cv2.rectangle(outro_img, (10, 10), (780, 110), (0, 0, 0), -1)
            cv2.putText(outro_img, "SQUARE FINISHED! OVERVIEW OF ACCEPTED GRAIN BOUNDARIES", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(outro_img, "Press ANY KEY to continue to the next square...", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, outro_img)
            cv2.waitKey(0)

        if abort:
            break

    cv2.destroyAllWindows()
    return ground_truth
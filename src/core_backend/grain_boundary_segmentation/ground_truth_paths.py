import cv2
import numpy as np

def label_paths_manually(potential_paths, image, skeleton, border_margin=400, buffer_px=400, target_display_height=1275):
    """
    Interaktives Tool zum Labeln der ML-gerouteten Pfade (E2E & T-Junc).
    Bietet ein extrem großes Sichtfeld, dicke Konturen und Alpha-Blending.
    """
    window_name = "Interactive Path Labeling"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    ch, cw = image.shape[:2]
    
    # Fit the introduction and summary screens within the display bounds.
    scale_factor = target_display_height / ch
    if cw * scale_factor > 1800: scale_factor = 1800 / cw
    new_w, new_h = int(cw * scale_factor), int(ch * scale_factor)
    
    # Prepare the introductory view of the image and excluded border.
    if image.dtype != np.uint8:
        img_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8u = image.copy()
        
    disp_base = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    overlay = disp_base.copy()
    
    # Display the existing skeleton in green.
    skel_dilated = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    overlay[skel_dilated == 1] = [0, 200, 0]
    
    # Mark the border region excluded from path review.
    cv2.rectangle(overlay, (0, 0), (cw, border_margin), (0, 0, 255), -1)
    cv2.rectangle(overlay, (0, ch - border_margin), (cw, ch), (0, 0, 255), -1)
    cv2.rectangle(overlay, (0, border_margin), (border_margin, ch - border_margin), (0, 0, 255), -1)
    cv2.rectangle(overlay, (cw - border_margin, border_margin), (cw, ch - border_margin), (0, 0, 255), -1)
    cv2.rectangle(overlay, (border_margin, border_margin), (cw - border_margin, ch - border_margin), (0, 0, 255), 4)
    
    # Blend the annotations with the source image.
    disp_intro = cv2.addWeighted(overlay, 0.4, disp_base, 0.6, 0)
    
    intro_img = cv2.resize(disp_intro, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(intro_img, (10, 10), (800, 150), (0, 0, 0), -1)
    cv2.putText(intro_img, "PATH LABELING STARTING...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(intro_img, "Press ANY KEY to start labeling this square", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(intro_img, f"Red Zone: Ignored Border (Margin: {border_margin}px)", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(window_name, intro_img)
    cv2.waitKey(0)

    labeled_paths = []
    abort = False
    total_paths = len(potential_paths)
    blacklisted_eps = set()

    # Review and label each candidate path.
    for idx, item in enumerate(potential_paths, 1):
        if abort: break
            
        ep1, ep2 = item.get("ep1", -1), item.get("ep2", -1)
        path_coords = item["path"]
        
        if ep1 in blacklisted_eps or ep2 in blacklisted_eps:
            item["label"] = 0
            labeled_paths.append(item)
            continue
            
        if len(path_coords) < 2: continue
        path_array = np.array(path_coords)
        
        y_min, y_max = np.min(path_array[:, 0]), np.max(path_array[:, 0])
        x_min, x_max = np.min(path_array[:, 1]), np.max(path_array[:, 1])
        y0, y1 = max(0, y_min - buffer_px), min(ch, y_max + buffer_px)
        x0, x1 = max(0, x_min - buffer_px), min(cw, x_max + buffer_px)
        
        local_img = img_8u[y0:y1, x0:x1]
        local_skel = skeleton[y0:y1, x0:x1]
        
        local_base = cv2.cvtColor(local_img, cv2.COLOR_GRAY2BGR)
        local_overlay = local_base.copy()
        
        # Show the local skeleton with increased line width.
        local_skel_dilated = cv2.dilate(local_skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        local_overlay[local_skel_dilated == 1] = [0, 200, 0] 
        
        # Highlight the candidate Dijkstra path in magenta.
        local_path = path_array - [y0, x0]
        path_mask = np.zeros_like(local_skel, dtype=np.uint8)
        path_mask[local_path[:, 0], local_path[:, 1]] = 1
        path_dilated = cv2.dilate(path_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        local_overlay[path_dilated == 1] = [255, 0, 255] 
        
        # Blend the path overlay with the source crop.
        display_img = cv2.addWeighted(local_overlay, 0.6, local_base, 0.4, 0)
        
        # Draw the endpoints after blending so they remain fully opaque.
        sy, sx = local_path[0]
        ey, ex = local_path[-1]
        cv2.circle(display_img, (sx, sy), 10, (255, 255, 0), -1) 
        cv2.circle(display_img, (ex, ey), 10, (255, 0, 255), -1) 
        
        h_loc, w_loc = display_img.shape[:2]
        sf = target_display_height / h_loc
        if w_loc * sf > 1800: sf = 1800 / w_loc
            
        new_w_loc, new_h_loc = int(w_loc * sf), int(h_loc * sf)
        interp = cv2.INTER_AREA if sf < 1.0 else cv2.INTER_CUBIC
        scaled_display = cv2.resize(display_img, (new_w_loc, new_h_loc), interpolation=interp)
        
        ui_width = 400
        ui_height = max(new_h_loc, 600)
        ui_panel = np.full((ui_height, ui_width, 3), (40, 40, 40), dtype=np.uint8)

        text_lines = [
            f"Path {idx} / {total_paths}", 
            f"Type: {item.get('phase', 'Unknown')}", 
            "",
            "[1] GOOD Path (Keep)",
            "[2] BAD Path (Discard)",
            "",
            "[7] BAD + Block Start (Cyan)",
            "[9] BAD + Block Both Ends",
            "",
            "[q] Abort & Save"
        ]

        y_pos = 50
        for i, line in enumerate(text_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(ui_panel, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2 if i==0 else 1)
            y_pos += 40

        if new_h_loc < ui_height:
            scaled_display = cv2.copyMakeBorder(scaled_display, 0, ui_height - new_h_loc, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            
        final_display = np.hstack((scaled_display, ui_panel))
        cv2.imshow(window_name, final_display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('1'):
                item["label"] = 1
                labeled_paths.append(item)
                break
            elif key == ord('2'):
                item["label"] = 0
                labeled_paths.append(item)
                break
            elif key == ord('7'):
                blacklisted_eps.add(ep1)
                item["label"] = 0
                labeled_paths.append(item)
                break
            elif key == ord('9'):
                blacklisted_eps.add(ep1)
                blacklisted_eps.add(ep2)
                item["label"] = 0
                labeled_paths.append(item)
                break
            elif key == ord('q') or key == 27:
                abort = True
                break

    # Summarize the paths accepted during review.
    res_base = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    res_overlay = res_base.copy()
    
    # Recreate the skeleton and excluded-border overlays.
    res_overlay[skel_dilated == 1] = [0, 200, 0]
    cv2.rectangle(res_overlay, (0, 0), (cw, border_margin), (0, 0, 255), -1)
    cv2.rectangle(res_overlay, (0, ch - border_margin), (cw, ch), (0, 0, 255), -1)
    cv2.rectangle(res_overlay, (0, border_margin), (border_margin, ch - border_margin), (0, 0, 255), -1)
    cv2.rectangle(res_overlay, (cw - border_margin, border_margin), (cw, ch - border_margin), (0, 0, 255), -1)
    cv2.rectangle(res_overlay, (border_margin, border_margin), (cw - border_margin, ch - border_margin), (0, 0, 255), 4)
    
    # Mark every accepted path in the summary image.
    for p in labeled_paths:
        if p["label"] == 1:
            arr = np.array(p["path"])
            mask = np.zeros_like(skeleton)
            mask[arr[:,0], arr[:,1]] = 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            res_overlay[mask == 1] = [0, 0, 255] 
            
    # Blend the summary overlay with the source image.
    res_blended = cv2.addWeighted(res_overlay, 0.6, res_base, 0.4, 0)
            
    outro_img = cv2.resize(res_blended, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(outro_img, (10, 10), (800, 110), (0, 0, 0), -1)
    cv2.putText(outro_img, "SQUARE FINISHED! OVERVIEW OF ACCEPTED PATHS", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(outro_img, "Press ANY KEY to continue to the next square...", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(window_name, outro_img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return labeled_paths
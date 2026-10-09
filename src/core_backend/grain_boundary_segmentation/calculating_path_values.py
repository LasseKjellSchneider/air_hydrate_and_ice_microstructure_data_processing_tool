import numpy as np
import math
from collections import deque

def _get_branch_vector(skel, start_y, start_x, max_steps=20):
    h, w = skel.shape
    visited = {(start_y, start_x)}
    queue = deque([(start_y, start_x, 0)])
    last_y, last_x = start_y, start_x
    
    while queue:
        y, x, dist = queue.popleft()
        last_y, last_x = y, x
        if dist >= max_steps: break
        
        for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            ny, nx = y+dy, x+dx
            if 0<=ny<h and 0<=nx<w and skel[ny, nx] == 1 and (ny, nx) not in visited:
                visited.add((ny,nx))
                queue.append((ny, nx, dist+1))
                break  # Follow one skeleton branch at a time.
                
    # Return the local skeleton direction from the endpoint into the existing branch.
    return last_y - start_y, last_x - start_x

def _calc_angle(dy1, dx1, dy2, dx2):
    mag1 = math.hypot(dy1, dx1)
    mag2 = math.hypot(dy2, dx2)
    if mag1 == 0 or mag2 == 0: return 0.0
    dot = dx1*dx2 + dy1*dy2
    val = max(-1.0, min(1.0, dot / (mag1*mag2)))
    return math.degrees(math.acos(val))

def calculate_path_features(path_coords, cost, image, skeleton, tophat_map):
    if len(path_coords) < 2: return None
        
    path_array = np.array(path_coords)
    length_px = len(path_array)
    
    # Geometric path features
    sy, sx = path_array[0]
    ey, ex = path_array[-1]
    
    euclidean_dist = np.sqrt((sy - ey)**2 + (sx - ex)**2)
    euclidean_dist = max(euclidean_dist, 1.0)
    
    cost_to_length = cost / length_px
    tortuosity = length_px / euclidean_dist
    
    # Compare the path direction with the local skeleton direction.
    p_sy, p_sx = path_array[min(20, length_px - 1)]
    path_vec_start = (p_sy - sy, p_sx - sx)
    
    p_ey, p_ex = path_array[max(0, length_px - 21)]
    path_vec_end = (p_ey - ey, p_ex - ex)
    
    skel_vec_start = _get_branch_vector(skeleton, sy, sx, 20)
    skel_vec_end = _get_branch_vector(skeleton, ey, ex, 20)
    
    # An angle near 180 degrees indicates continuation; near 0 indicates reversal.
    angle_start = _calc_angle(skel_vec_start[0], skel_vec_start[1], path_vec_start[0], path_vec_start[1])
    angle_end = _calc_angle(skel_vec_end[0], skel_vec_end[1], path_vec_end[0], path_vec_end[1])
    
    # Index the source arrays directly at the path coordinates.
    py = path_array[:, 0]
    px = path_array[:, 1]
    
    # Image-intensity features
    pixels = image[py, px]
    mean_intensity = float(np.mean(pixels))
    p80_intensity = float(np.percentile(pixels, 80))
    p20_intensity = float(np.percentile(pixels, 20))
    
    # Mean ridge response from the precomputed image-wide map.
    tophat_pixels = tophat_map[py, px]
    mean_tophat = float(np.mean(tophat_pixels))
    
    features = {
        "cost": float(cost),
        "length_px": float(length_px),
        "euclidean_dist": float(euclidean_dist),
        "cost_to_length": float(cost_to_length),
        "tortuosity": float(tortuosity),
        "angle_start": float(angle_start),
        "angle_end": float(angle_end),
        "mean_intensity": mean_intensity,
        "p80_intensity": p80_intensity,
        "p20_intensity": p20_intensity,
        "mean_tophat": mean_tophat
    }
    
    return features
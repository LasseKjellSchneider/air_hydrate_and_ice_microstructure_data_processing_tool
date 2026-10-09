import numpy as np
import cv2
import scipy.ndimage as ndi
from skimage.segmentation import expand_labels
from collections import defaultdict
import concurrent.futures
from pathlib import Path

def reconstruct_grains(skeleton, border_mask=None, resolution_um_per_px=1.41, workers=6, debug_dir=None):
    """
    Führt die topologische Bereinigung, Loch-Verschmelzung, Voronoi-Expansion 
    und Trennung von Innen-/Randkörnern zentral und einmalig durch.
    Inklusive Hard-Debug-Bildausgabe für jeden logischen Teilschritt.
    """
    print("-> Reconstructing Grains Topology (Shared Module)...")
    h, w = skeleton.shape
    skel = skeleton.copy()
    
    # Close gaps before labeling connected grain regions.
    if border_mask is not None: skel[border_mask == 0] = 1
    skel[0, :] = 1; skel[-1, :] = 1
    skel[:, 0] = 1; skel[:, -1] = 1

    if debug_dir:
        cv2.imwrite(str(Path(debug_dir) / "04_topo_input_skeleton.png"), skel * 255)

    # Use four-connectivity so diagonal contacts remain separate regions.
    print("-> Identifying preliminary grains (4-Connected Flood Fill)...")
    cross_struct = np.array([[0, 1, 0], 
                             [1, 1, 1], 
                             [0, 1, 0]], dtype=bool)
                             
    labels = np.zeros((h, w), dtype=np.uint32)
    num_labels = ndi.label(skel == 0, structure=cross_struct, output=labels)
    areas = np.bincount(labels.ravel())

    if debug_dir:
        # Normalize labels to an 8-bit range for visualization.
        vis_labels = (labels % 254 + 1).astype(np.uint8)
        vis_labels[labels == 0] = 0
        cv2.imwrite(str(Path(debug_dir) / "05_topo_initial_floodfill.png"), vis_labels)

    def get_out_mask(y0, y1, x0, x1):
        chk = np.zeros((y1-y0, x1-x0), dtype=bool)
        if border_mask is not None:
            chk[:] = (border_mask[y0:y1, x0:x1] == 0)
        if y0 == 0: chk[0, :] = True
        if y1 == h: chk[-1, :] = True
        if x0 == 0: chk[:, 0] = True
        if x1 == w: chk[:, -1] = True
        return chk

    print("-> Building global adjacency graph (Multi-Core)...")
    adjacency = defaultdict(lambda: defaultdict(int))
    touches_outside = np.zeros(num_labels + 1, dtype=bool)
    chunk_size = 4000
    kernel_7 = np.ones((7,7), np.uint8)

    def calc_adj(y, x):
        margin = 5
        y0, y1 = max(0, y - margin), min(h, y + chunk_size + margin)
        x0, x1 = max(0, x - margin), min(w, x + chunk_size + margin)
        
        loc_lbl = labels[y0:y1, x0:x1]
        loc_out = get_out_mask(y0, y1, x0, x1).astype(np.uint8)
        
        dilated_out = cv2.dilate(loc_out, kernel_7)
        out_labels = np.unique(loc_lbl[dilated_out == 1])
        
        loc_lbl_exp = expand_labels(loc_lbl, distance=4)
        sy0, sy1 = y - y0, y - y0 + min(h, y + chunk_size) - y
        sx0, sx1 = x - x0, x - x0 + min(w, x + chunk_size) - x
        valid_exp = loc_lbl_exp[sy0:sy1, sx0:sx1]
        
        left, right = valid_exp[:, :-1], valid_exp[:, 1:]
        h_mask = (left != right) & (left > 0) & (right > 0)
        up, down = valid_exp[:-1, :], valid_exp[1:, :]
        v_mask = (up != down) & (up > 0) & (down > 0)
        
        l_vals, r_vals = left[h_mask], right[h_mask]
        u_vals, d_vals = up[v_mask], down[v_mask]
        
        pairs_h = np.c_[l_vals, r_vals] if len(l_vals) > 0 else np.empty((0,2), dtype=np.uint32)
        pairs_v = np.c_[u_vals, d_vals] if len(u_vals) > 0 else np.empty((0,2), dtype=np.uint32)
        
        all_pairs = np.vstack((pairs_h, pairs_v))
        local_adj = defaultdict(int)
        if len(all_pairs) > 0:
            all_pairs = np.sort(all_pairs, axis=1)
            unq_pairs, counts = np.unique(all_pairs, axis=0, return_counts=True)
            for pair, count in zip(unq_pairs, counts):
                local_adj[tuple(pair)] += count
                
        return out_labels, local_adj

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(calc_adj, y, x) for y in range(0, h, chunk_size) for x in range(0, w, chunk_size)]
        for f in concurrent.futures.as_completed(futures):
            o_lbls, l_adj = f.result()
            touches_outside[o_lbls] = True
            for (p1, p2), count in l_adj.items():
                adjacency[p1][p2] += count
                adjacency[p2][p1] += count

    print("-> Applying topological Hole- & Size-Filters...")
    target_area_px = 200000.0 / (resolution_um_per_px ** 2)
    remap = np.arange(num_labels + 1, dtype=np.uint32)
    
    for i in range(1, num_labels + 1):
        if areas[i] == 0: continue
        neighbors = adjacency.get(i, {})
        is_out = touches_outside[i]
        
        if len(neighbors) == 1 and not is_out:
            remap[i] = list(neighbors.keys())[0]
        elif areas[i] < target_area_px:
            if len(neighbors) > 0:
                dom_n = max(neighbors.items(), key=lambda x: x[1])[0]
                if is_out: remap[i] = 0
                else: remap[i] = dom_n
            else:
                if is_out: remap[i] = 0

    def find_root(n, p_arr):
        while p_arr[n] != n: p_arr[n] = p_arr[p_arr[n]]; n = p_arr[n]
        return n

    for i in range(1, num_labels + 1):
        if remap[i] != i and remap[i] != 0:
            root_i, root_target = find_root(i, remap), find_root(remap[i], remap)
            if root_i != root_target: remap[root_i] = root_target
    for i in range(1, num_labels + 1):
        if remap[i] != 0: remap[i] = find_root(i, remap)

    print("-> Committing merges...")
    def apply_remap(y, x):
        loc = labels[y:y+chunk_size, x:x+chunk_size]
        loc[:] = remap[loc]
        loc_out = get_out_mask(y, min(h, y+chunk_size), x, min(w, x+chunk_size))
        loc[loc_out] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                executor.submit(apply_remap, y, x)
                
    if debug_dir:
        vis_labels = (labels % 254 + 1).astype(np.uint8)
        vis_labels[labels == 0] = 0
        cv2.imwrite(str(Path(debug_dir) / "06_topo_merged_labels.png"), vis_labels)

    print("-> Voronoi Expansion (Chunked, In-Place to save RAM)...")
    exp_margin = 150
    def run_voronoi(y, x):
        y0, y1 = max(0, y - exp_margin), min(h, y + chunk_size + exp_margin)
        x0, x1 = max(0, x - exp_margin), min(w, x + chunk_size + exp_margin)
        tile = labels[y0:y1, x0:x1]
        if np.any(tile):
            expanded_tile = expand_labels(tile, distance=150)
            sy0, sy1 = y - y0, y - y0 + min(h, y + chunk_size) - y
            sx0, sx1 = x - x0, x - x0 + min(w, x + chunk_size) - x
            return y, x, expanded_tile[sy0:sy1, sx0:sx1]
        return y, x, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_voronoi, y, x) for y in range(0, h, chunk_size) for x in range(0, w, chunk_size)]
        for f in concurrent.futures.as_completed(futures):
            y, x, tile_res = f.result()
            if tile_res is not None:
                labels[y:min(h, y + chunk_size), x:min(w, x + chunk_size)] = tile_res

    if debug_dir:
        vis_labels = (labels % 254 + 1).astype(np.uint8)
        vis_labels[labels == 0] = 0
        cv2.imwrite(str(Path(debug_dir) / "07_topo_voronoi_expanded.png"), vis_labels)

    print("-> Categorizing Interior vs Border Grains...")
    touches_thick = np.zeros(labels.max() + 1, dtype=bool)
    
    def check_thick(y, x):
        margin = 100
        kernel_100 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (201, 201))
        y0, y1 = max(0, y - margin), min(h, y + chunk_size + margin)
        x0, x1 = max(0, x - margin), min(w, x + chunk_size + margin)
        loc_out = get_out_mask(y0, y1, x0, x1).astype(np.uint8)
        if np.any(loc_out):
            dilated_tile = cv2.dilate(loc_out, kernel_100)
            sy0, sy1 = y - y0, y - y0 + min(h, y + chunk_size) - y
            sx0, sx1 = x - x0, x - x0 + min(w, x + chunk_size) - x
            loc_thick = dilated_tile[sy0:sy1, sx0:sx1] > 0
            return np.unique(labels[y:min(h, y+chunk_size), x:min(w, x+chunk_size)][loc_thick])
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(check_thick, y, x) for y in range(0, h, chunk_size) for x in range(0, w, chunk_size)]
        for f in concurrent.futures.as_completed(futures):
            touching_labels = f.result()
            if len(touching_labels) > 0:
                touches_thick[touching_labels] = True

    interior_remap = np.zeros(labels.max() + 1, dtype=np.uint32)
    border_remap = np.zeros(labels.max() + 1, dtype=np.uint32)
    int_id, bord_id = 1, 1
    
    for i in np.unique(labels):
        if i == 0: continue
        if touches_thick[i]:
            border_remap[i] = bord_id
            bord_id += 1
        else:
            interior_remap[i] = int_id
            int_id += 1

    if debug_dir:
        int_mask = (interior_remap[labels] > 0).astype(np.uint8) * 255
        bord_mask = (border_remap[labels] > 0).astype(np.uint8) * 255
        cv2.imwrite(str(Path(debug_dir) / "08_topo_interior_grains.png"), int_mask)
        cv2.imwrite(str(Path(debug_dir) / "09_topo_border_grains.png"), bord_mask)

    print("-> Topology reconstruction finished!")
    return labels, interior_remap, border_remap
import h5py
import numpy as np
from pathlib import Path
import json
from skimage.segmentation import find_boundaries

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist() 
        if isinstance(obj, np.integer): return int(obj)     
        if isinstance(obj, np.floating): return float(obj)   
        return super(NumpyEncoder, self).default(obj)

def export_as_h5(image_path, grain_labels, interior_remap, border_remap, new_lines_info, diagnostic_masks, final_objects_info, 
                 subgrains=None, bottom_grains=None, min_grain_area_px=500,
                 output_dir=None, resolution_um_per_px=1.41, crop_offset=(0, 0)):

    print("\n" + "="*50)
    print(f"--- EXPORTING TO HDF5 (ULTRA-LOW RAM) (Res: {resolution_um_per_px} µm/px) ---")
    print("="*50)

    img_p = Path(image_path)
    output_dir = Path(output_dir) if output_dir else img_p.parent
    h5_filename = output_dir / f"{img_p.stem}_grain_boundaries.h5"
    h, w = grain_labels.shape
    chunk_size = 4000

    for line in new_lines_info:
        if "cost" in line: line["cost_um"] = line["cost"] * resolution_um_per_px
    for obj in final_objects_info:
        if "length" in obj: obj["length_um"] = obj["length"] * resolution_um_per_px

    with h5py.File(h5_filename, 'w') as h5f:
        meta_grp = h5f.create_group("metadata")
        meta_grp.attrs["original_image"] = str(img_p.name)
        meta_grp.attrs["image_width"] = w; meta_grp.attrs["image_height"] = h
        meta_grp.attrs["resolution_um_per_px"] = resolution_um_per_px
        meta_grp.create_dataset("new_lines_info", data=json.dumps(new_lines_info, cls=NumpyEncoder))
        
        layer_grp = h5f.create_group("layers")
        
        print("-> Saving layers DIRECTLY to HDF5 (Zero-RAM Overhead)...")
        # Generate and write each layer in sequence to limit peak memory use.
        dset_full_skel = layer_grp.create_dataset("00_final_full_skeleton", shape=(h, w), dtype=np.uint8, compression="gzip", compression_opts=4, chunks=True)
        dset_int_skel = layer_grp.create_dataset("01_interior_skeleton", shape=(h, w), dtype=np.uint8, compression="gzip", compression_opts=4, chunks=True)
        dset_interior = layer_grp.create_dataset("12_interior_grains_map", shape=(h, w), dtype=np.uint32, compression="gzip", compression_opts=4, chunks=True)
        dset_border = layer_grp.create_dataset("13_border_grains_map", shape=(h, w), dtype=np.uint32, compression="gzip", compression_opts=4, chunks=True)

        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                y0, y1 = max(0, y - 2), min(h, y + chunk_size + 2)
                x0, x1 = max(0, x - 2), min(w, x + chunk_size + 2)
                
                lbl_tile = grain_labels[y0:y1, x0:x1]
                int_tile = interior_remap[lbl_tile]
                
                if np.any(lbl_tile):
                    fb_tile = find_boundaries(lbl_tile, mode='inner').astype(np.uint8) * 255
                    sy0, sy1 = y - y0, y - y0 + min(h, y + chunk_size) - y
                    sx0, sx1 = x - x0, x - x0 + min(w, x + chunk_size) - x
                    dset_full_skel[y:min(h, y+chunk_size), x:min(w, x+chunk_size)] = fb_tile[sy0:sy1, sx0:sx1]
                    
                if np.any(int_tile):
                    ib_tile = find_boundaries(int_tile, mode='inner').astype(np.uint8) * 255
                    sy0, sy1 = y - y0, y - y0 + min(h, y + chunk_size) - y
                    sx0, sx1 = x - x0, x - x0 + min(w, x + chunk_size) - x
                    dset_int_skel[y:min(h, y+chunk_size), x:min(w, x+chunk_size)] = ib_tile[sy0:sy1, sx0:sx1]
                    
                loc_lbl = grain_labels[y:min(h, y+chunk_size), x:min(w, x+chunk_size)]
                dset_interior[y:min(h, y+chunk_size), x:min(w, x+chunk_size)] = interior_remap[loc_lbl]
                dset_border[y:min(h, y+chunk_size), x:min(w, x+chunk_size)] = border_remap[loc_lbl]

        scratch_mask = np.zeros((h, w), dtype=np.uint8)
        def _save_diag(dict_key, dataset_name):
            data = diagnostic_masks.get(dict_key)
            if data is not None: scratch_mask[:] = data * 255
            else: scratch_mask.fill(0)
            layer_grp.create_dataset(dataset_name, data=scratch_mask, compression="gzip", compression_opts=4)
        
        _save_diag("kept_base_raw", "05_kept_base_raw")

    print(f"-> Successfully exported layers to: {h5_filename.name}")
    print("="*50 + "\n")
    return h5_filename
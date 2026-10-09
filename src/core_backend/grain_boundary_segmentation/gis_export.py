import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, shape
import rasterio
import rasterio.features
from rasterio.transform import Affine
from pathlib import Path
import scipy.ndimage as ndi
from skimage.segmentation import find_boundaries
import concurrent.futures

def export_as_gpkg(image_array, image_path, grain_labels, interior_remap, border_remap, 
                   new_lines_info, diagnostic_masks, final_objects_info, 
                   output_dir, border_mask=None, subgrains=None, bottom_grains=None, 
                   min_grain_area_px=500, resolution_um_per_px=1.41, crop_offset=(0, 0)):
                   
    print("\n" + "="*50)
    print(f"--- EXPORTING GIS VECTORS (ULTRA-LOW RAM) (Res: {resolution_um_per_px} µm/px) ---")
    print("="*50)

    img_p = Path(image_path)
    out_dir = Path(output_dir) if output_dir else img_p.parent
    gpkg_filename = out_dir / f"{img_p.stem}_vectors.gpkg"
    tif_filename = out_dir / f"{img_p.stem}_micrometer_scaled.tif"
    
    off_x, off_y = crop_offset
    res = resolution_um_per_px
    h, w = grain_labels.shape
    WORKERS = 6

    # Write the raster using physical coordinate units.
    try:
        transform = Affine(res, 0, off_x * res, 0, -res, -off_y * res)
        with rasterio.open(
            tif_filename, 'w', driver='GTiff', height=h, width=w, count=1,
            dtype=image_array.dtype, transform=transform, compress='lzw',
            tiled=True, blockxsize=2048, blockysize=2048
        ) as dst:
            dst.write(image_array, 1)
    except Exception:
        pass

    def px_to_um(y, x):
        return (float((x + off_x) * res), float(-(y + off_y) * res))

    # Extract grain polygons from the precomputed label maps.
    print("   -> Extracting Polygons with Rasterio (Multi-Core)...")
    new_slices = ndi.find_objects(grain_labels)
    
    def extract_poly(i):
        if i == 0 or i > len(new_slices) or new_slices[i-1] is None: 
            return None
            
        is_border = border_remap[i] > 0
        is_interior = interior_remap[i] > 0
        
        if not is_border and not is_interior: 
            return None
            
        slc = new_slices[i-1]
        y_slice, x_slice = slc
        local_mask = (grain_labels[y_slice, x_slice] == i).astype(np.uint8)
        local_transform = Affine(res, 0, (off_x + x_slice.start) * res, 0, -res, -(off_y + y_slice.start) * res)
        
        area_um2 = float(np.sum(local_mask) * (res**2))
        
        p_list = []
        for geom_dict, val in rasterio.features.shapes(local_mask, transform=local_transform):
            if val == 1:
                try:
                    poly = shape(geom_dict)
                    if not poly.is_valid: poly = poly.buffer(0)
                    p_list.append(poly)
                except Exception: pass
        return (is_border, area_um2, p_list)

    interior_geoms, interior_areas = [], []
    border_geoms, border_areas = [], []
    
    unique_labels = np.unique(grain_labels)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(extract_poly, i) for i in unique_labels if i != 0]
        for f in concurrent.futures.as_completed(futures):
            res_data = f.result()
            if res_data is not None:
                is_border, area, p_list = res_data
                for p in p_list:
                    if is_border:
                        border_geoms.append(p)
                        border_areas.append(area)
                    else:
                        interior_geoms.append(p)
                        interior_areas.append(area)

    gdf_int = gpd.GeoDataFrame(geometry=interior_geoms) if interior_geoms else gpd.GeoDataFrame()
    if not gdf_int.empty:
        gdf_int['layer'] = "Interior Grains"
        gdf_int['area_um2'] = interior_areas
        
    gdf_bord = gpd.GeoDataFrame(geometry=border_geoms) if border_geoms else gpd.GeoDataFrame()
    if not gdf_bord.empty:
        gdf_bord['layer'] = "Border Grains"
        gdf_bord['area_um2'] = border_areas

    # Convert grain-boundary skeletons to vector line features.
    print("-> Generating Vector Skeletons on the fly (Zero-RAM Overhead)...")
    def mask_to_gdf_chunked(remap_array, layer_name, chunk_size=4000):
        geoms = []
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                y0, y1 = max(0, y - 2), min(h, y + chunk_size + 2)
                x0, x1 = max(0, x - 2), min(w, x + chunk_size + 2)
                
                lbl_tile = grain_labels[y0:y1, x0:x1]
                if remap_array is not None:
                    lbl_tile = remap_array[lbl_tile]
                    
                if not np.any(lbl_tile): continue
                
                fb_tile = find_boundaries(lbl_tile, mode='inner').astype(np.uint8) * 255
                sy0, sy1 = y - y0, y - y0 + min(h, y + chunk_size) - y
                sx0, sx1 = x - x0, x - x0 + min(w, x + chunk_size) - x
                tile = fb_tile[sy0:sy1, sx0:sx1]
                
                if not np.any(tile): continue
                contours, _ = cv2.findContours(tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 2:
                        pts = [px_to_um(pt[0][1] + y, pt[0][0] + x) for pt in cnt]
                        geoms.append(LineString(pts))
        if not geoms: return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(geometry=geoms)
        gdf['layer'] = layer_name
        return gdf

    layers = {
        "00_Full_Skeleton": mask_to_gdf_chunked(None, "Full Skeleton Boundaries"),
        "01_Interior_Skeleton": mask_to_gdf_chunked(interior_remap, "Interior Skeleton Boundaries"),
        "12_Interior_Grains": gdf_int,
        "13_Border_Grains": gdf_bord
    }

    if gpkg_filename.exists():
        try: gpkg_filename.unlink()
        except Exception: pass

    for layer_name, gdf in layers.items():
        if not gdf.empty:
            try:
                gdf.to_file(gpkg_filename, layer=layer_name, driver="GPKG")
            except Exception: pass
                    
    print("="*60)
    return gpkg_filename
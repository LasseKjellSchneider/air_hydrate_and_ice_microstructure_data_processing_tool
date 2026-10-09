import os
import h5py
import numpy as np
import pandas as pd
import cv2
import gc
from core_backend.data_merge.logger_config import PipelineLogger

class MasterHDF5Exporter:
    """
    Vektorbasierter HDF5-Exporter.
    Konvertiert gigantische Raster-Grain-Maps in speichereffiziente 3D-Polygone (Meshes).
    Speichert KEINE Bilder, sondern ausschließlich IDs, Vektoren und Polygon-Koordinaten (WKT).
    """
    def __init__(self, output_filepath: str, spatial_ref, config_data: dict):
        self.logger = PipelineLogger()
        self.output_filepath = output_filepath
        self.spatial_ref = spatial_ref
        self.config = config_data
        
        # Reduce polygon sampling density to limit mesh size.
        self.ds_factor = 15
        
        self.target_res = self.spatial_ref.resolution_um_per_px * self.ds_factor
        
        self.logger.info(f"Initialized Vector Exporter. Mesh Resolution: {self.target_res:.2f} um/vertex")

    def build_master_file(self):
        self.logger.info(f"Creating Master Vector HDF5 at: {self.output_filepath}")
        try:
            with h5py.File(self.output_filepath, 'w') as h5_master:
                self._write_global_metadata(h5_master)
                self._write_hydrates(h5_master)
                
                planes_group = h5_master.create_group("planes")
                
                for layer_idx, layer_info in enumerate(self.config.get("layers", [])):
                    # Skip image-only layers that do not contain HDF5 analysis data.
                    if not layer_info.get("extra_h5_path") and not layer_info.get("file_path", "").lower().endswith('.h5'):
                        self.logger.info(f"Skipping pure image layer {layer_info['type']} (no HDF5 data attached).")
                        continue
                        
                    self.logger.info(f"--- Processing Layer Sequence {layer_idx + 1}/{len(self.config['layers'])} ---")
                    self._process_and_write_layer(planes_group, layer_info, layer_idx)
                    
            self.logger.info(f"Master HDF5 successfully built.")
        except Exception as e:
            self.logger.error(f"FATAL ERROR during build: {str(e)}")
            raise e

    def _write_global_metadata(self, h5_master):
        grp = h5_master.create_group("global_metadata")
        grp.attrs["global_origin_um"] = self.spatial_ref.global_origin_um
        grp.attrs["global_dimensions_um"] = self.spatial_ref.global_dimensions_um
        grp.attrs["mesh_resolution_um_per_vertex"] = self.target_res

    def _write_hydrates(self, h5_master):
        grp = h5_master.create_group("hydrates")
        hydrates_data = self.spatial_ref.raw_analysis.get("verified_hydrates", [])
        
        for idx, h in enumerate(hydrates_data):
            h_grp = grp.create_group(f"hydrate_{idx}")
            for key, val in h.items():
                if isinstance(val, dict):
                    sub_grp = h_grp.create_group(key)
                    for k, v in val.items():
                        if isinstance(v, (int, float, str, bool)):
                            sub_grp.attrs[k] = v
                elif isinstance(val, (int, float, str, bool)):
                    h_grp.attrs[key] = val

    def _process_and_write_layer(self, planes_group, layer_info: dict, layer_idx: int):
        file_path = layer_info.get("file_path", "")
        extra_h5 = layer_info.get("extra_h5_path", "")
        layer_type = layer_info["type"]
        is_fa = "FA" in layer_type
        mirror = layer_info.get("mirror", False)
        M = layer_info.get("transform_matrix", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

        # Convert the configured layer depth from millimeters to micrometers.
        z_mm = float(layer_info.get("z_distance_mm", 0.0))
        z_um = z_mm * 1000.0

        group_name = f"{layer_type}_{layer_idx}" 
        layer_grp = planes_group.create_group(group_name)
        
        meta_grp = layer_grp.create_group("metadata")
        meta_grp.attrs["z_distance_mm"] = z_mm
        meta_grp.attrs["is_mirrored"] = mirror
        meta_grp.attrs["is_fa"] = is_fa
        
        h5_to_process = extra_h5 if extra_h5 else file_path
        
        if h5_to_process.lower().endswith('.h5'):
            self._vectorize_h5_layer(h5_to_process, layer_grp, mirror, is_fa, M, z_um)

    def _vectorize_h5_layer(self, file_path, layer_grp, mirror: bool, is_fa: bool, M: np.ndarray, z_um: float):
        """Extrahiert Meshes aus Raster-HDF5 und transformiert die Geometrien."""
        self.logger.info(f"Vectorizing data from {file_path} at Z={z_um} um")
        
        mesh_records = []
        
        with h5py.File(file_path, 'r') as h5_source:
            
            orig_shape = None

            # Extract microscope grain boundaries from the interior and border maps.
            if "layers" in h5_source:
                if "12_interior_grains_map" in h5_source["layers"]:
                    self.logger.info("Extracting Interior Polygons...")
                    arr = h5_source["layers"]["12_interior_grains_map"][:]
                    orig_shape = arr.shape
                    arr_small = arr[::self.ds_factor, ::self.ds_factor]
                    del arr
                    gc.collect()
                    
                    polygons = self._extract_polygons(arr_small, is_fa, mirror, M, orig_shape, z_um)
                    for poly in polygons:
                        mesh_records.append((poly["id"], "interior", 0.0, 0.0, 0.0, poly["wkt"]))
                        
                if "13_border_grains_map" in h5_source["layers"]:
                    self.logger.info("Extracting Border Polygons...")
                    arr = h5_source["layers"]["13_border_grains_map"][:]
                    arr_small = arr[::self.ds_factor, ::self.ds_factor]
                    del arr
                    gc.collect()
                    
                    polygons = self._extract_polygons(arr_small, is_fa, mirror, M, orig_shape, z_um)
                    for poly in polygons:
                        mesh_records.append((poly["id"], "border", 0.0, 0.0, 0.0, poly["wkt"]))

            # Extract fabric-analysis regions and associate their C-axis vectors.
            elif "ws_image" in h5_source:
                self.logger.info("Extracting FA Polygons from ws_image...")
                arr = np.squeeze(h5_source["ws_image"][:])
                orig_shape = arr.shape
                arr_small = arr[::self.ds_factor, ::self.ds_factor]
                del arr
                gc.collect()
                
                polygons = self._extract_polygons(arr_small, is_fa, mirror, M, orig_shape, z_um)
                
                # Transform the C-axis vectors into the aligned coordinate system.
                if "grain_table" in h5_source and "c_axis_vectors" in h5_source:
                    df = pd.DataFrame(h5_source["grain_table"][:])
                    vecs = h5_source["c_axis_vectors"][:]
                    new_vecs = self._transform_vector_values(vecs, is_fa, mirror, M)
                    
                    # Transform grain centroids using the original image dimensions.
                    pts = np.column_stack((df["center_x"].values, df["center_y"].values))
                    new_pts = self._transform_coordinates(pts, is_fa, mirror, M, orig_shape[0], orig_shape[1])
                    df["center_x"] = new_pts[:, 0]
                    df["center_y"] = new_pts[:, 1]
                    
                    # Map each grain identifier to its transformed C-axis vector.
                    vec_map = {row["grain_id"]: new_vecs[i] for i, row in df.iterrows()}
                    
                    for poly in polygons:
                        gid = poly["id"]
                        cx, cy, cz = vec_map.get(gid, (0.0, 0.0, 0.0))
                        mesh_records.append((gid, "fa_grain", cx, cy, cz, poly["wkt"]))

        # Store the extracted vector meshes in the output HDF5 layer.
        if mesh_records:
            self.logger.info(f"Saving {len(mesh_records)} vector meshes to HDF5.")
            
            # Use a structured array to store mesh attributes as tabular records.
            dt = np.dtype([
                ('grain_id', np.int32),
                ('mesh_type', 'S15'),    # One of: interior, border, or fa_grain.
                ('c_axis_x', np.float32),
                ('c_axis_y', np.float32),
                ('c_axis_z', np.float32),
                ('polygon_wkt', h5py.string_dtype(encoding='utf-8'))  # 3D polygon geometry in WKT format.
            ])
            
            data_array = np.array(mesh_records, dtype=dt)
            layer_grp.create_dataset("vector_meshes", data=data_array, compression="gzip")
        else:
            self.logger.warning(f"No meshes extracted for {file_path}")

    def _extract_polygons(self, label_img_small: np.ndarray, is_fa: bool, mirror: bool, M: np.ndarray, orig_shape: tuple, z_um: float) -> list:
        """
        Sucht Konturen für alle IDs im verkleinerten Bild, vereinfacht sie und
        transformiert die Eckpunkte mathematisch in den globalen HDF5-Raum.
        """
        polygons = []
        unique_ids = np.unique(label_img_small)
        unique_ids = unique_ids[unique_ids > 0]  # Label 0 denotes the background.
        
        orig_h, orig_w = orig_shape
        
        for grain_id in unique_ids:
            # Isolate the current grain label.
            mask = np.zeros(label_img_small.shape, dtype=np.uint8)
            mask[label_img_small == grain_id] = 255
            
            # Extract the grain boundary as polygon vertices.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            # Keep the largest contour if the label contains small artifacts.
            cnt = max(contours, key=cv2.contourArea)
            
            # Reduce the number of polygon vertices with Douglas-Peucker approximation.
            epsilon = 1.0
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, closed=True)
            
            # Reshape the contour to an (N, 2) coordinate array.
            pts = approx_cnt.reshape(-1, 2).astype(np.float64)
            
            # Restore coordinates to the original image scale.
            pts *= self.ds_factor
            
            # Apply the analysis orientation, optional mirror, and alignment transform.
            transformed_pts = self._transform_coordinates(pts, is_fa, mirror, M, orig_h, orig_w)
            
            # A valid polygon requires at least three vertices.
            if len(transformed_pts) < 3:
                continue
            
            # Close the ring before serializing the polygon as 3D WKT.
            wkt_coords_list = [f"{x:.1f} {y:.1f} {z_um:.1f}" for x, y in transformed_pts]
            if wkt_coords_list[0] != wkt_coords_list[-1]:
                wkt_coords_list.append(wkt_coords_list[0])
                
            wkt_coords = ", ".join(wkt_coords_list)
            wkt_string = f"POLYGON Z (({wkt_coords}))"
            
            polygons.append({
                "id": grain_id,
                "wkt": wkt_string
            })
            
        return polygons

    def _transform_coordinates(self, pts: np.ndarray, is_fa: bool, mirror: bool, M: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Transformiert Eckpunkt-Koordinaten mathematisch (Rotation, Mirror, Warp)."""
        new_pts = pts.copy()
        
        # Rotate fabric-analysis image coordinates 90 degrees clockwise.
        if is_fa: 
            old_x = new_pts[:, 0].copy()
            old_y = new_pts[:, 1].copy()
            new_pts[:, 0] = orig_h - old_y
            new_pts[:, 1] = old_x
            
        # Apply the configured mirror across the horizontal image axis.
        if mirror: 
            # Use the full image height as the mirror-axis extent.
            current_h = orig_w if is_fa else orig_h
            new_pts[:, 1] = current_h - new_pts[:, 1]
            
        # Apply the affine alignment matrix to the homogeneous coordinates.
        new_pts_hom = np.column_stack((new_pts, np.ones(len(new_pts))))
        return new_pts_hom.dot(M.T)[:, :2]

    def _transform_vector_values(self, vecs: np.ndarray, is_fa: bool, mirror: bool, M: np.ndarray) -> np.ndarray:
        """Rotiert physikalische 3D-Richtungsvektoren synchron zu den Polygonen."""
        orig_shape = vecs.shape
        v = vecs.reshape(-1, 3).copy()
        
        # Fabric-analysis vectors are already expressed in the global coordinate system.
            
        # Reflect the vector across the horizontal sample plane when mirroring the image.
        if mirror: 
            v[:, 1] = -v[:, 1]  # Reverse the in-plane Y component.
            v[:, 2] = -v[:, 2]  # Reverse the component normal to the sample plane.
            
        # Extract the in-plane rotation; scaling does not affect vector direction.
        angle_rad = np.arctan2(M[1, 0], M[0, 0])
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Construct a rotation matrix without the affine scale component.
        R_pure = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # Apply the alignment rotation to the in-plane vector components.
        v_xy = v[:, 0:2].dot(R_pure.T)
        v[:, 0:2] = v_xy
        
        # Preserve unit-vector length after the coordinate transforms.
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v_norm = v / norms
        
        return v_norm.reshape(orig_shape)
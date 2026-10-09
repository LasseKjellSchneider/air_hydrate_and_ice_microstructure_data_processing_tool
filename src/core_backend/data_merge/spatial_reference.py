import json
import os

class GlobalSpatialReference:
    """
    Parses metadata_preprocessed.json and analysis_results.json to establish 
    a unified global 3D coordinate system.
    """
    def __init__(self, analysis_json_path: str, metadata_json_path: str):
        self.analysis_path = analysis_json_path
        self.metadata_path = metadata_json_path
        
        self.raw_analysis = {}
        self.raw_metadata = {}
        
        # Global coordinate-system parameters derived from the input metadata.
        self.global_origin_um = [0.0, 0.0, 0.0]
        self.global_dimensions_um = [0.0, 0.0, 0.0]
        self.resolution_um_per_px = 1.41  # Fallback when metadata has no resolution.
        
        self._load_and_parse()

    def _load_and_parse(self):
        if os.path.exists(self.analysis_path):
            with open(self.analysis_path, 'r', encoding='utf-8') as f:
                self.raw_analysis = json.load(f)
                
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.raw_metadata = json.load(f)
            
        self._extract_resolution()
        self._calculate_global_bounds()

    def _extract_resolution(self):
        """Extracts pixel resolution from analysis_results.json metadata."""
        meta = self.raw_analysis.get("sample_metadata", {})
        res = meta.get("spatial_resolution_um_per_px", 1.41)
        self.resolution_um_per_px = float(res)

    def _calculate_global_bounds(self):
        """
        Calculates the global origin at the top-left corner of tile (0,0)
        and dimensions based on tile bounds and hydrate depth.
        """
        # Use tile (0, 0) as the preferred origin reference.
        positions = self.raw_metadata.get("Positions", [])
        pos_0_0 = next((p for p in positions if p.get("gridX") == 0 and p.get("gridY") == 0), None)
        
        # Fall back to the first available position when tile (0, 0) is absent.
        if pos_0_0 is None and positions:
            pos_0_0 = positions[0]

        # Use the expected tile dimensions to convert its center to a corner.
        tile_w_px = 2048
        tile_h_px = 2048
        
        if pos_0_0:
            center_x = pos_0_0.get("positionX", 0.0)
            center_y = pos_0_0.get("positionY", 0.0)
            
            # Convert the tile center to its top-left corner; image Y increases downward.
            origin_x = center_x - (tile_w_px * self.resolution_um_per_px / 2.0)
            origin_y = center_y - (tile_h_px * self.resolution_um_per_px / 2.0)
            origin_z = 0.0  # The sample surface defines the zero-depth plane.
        else:
            origin_x, origin_y, origin_z = 0.0, 0.0, 0.0

        self.global_origin_um = [origin_x, origin_y, origin_z]

        # Compute the XY bounds from all tile centers and half-tile extents.
        if positions:
            max_x = max(p.get("positionX", 0.0) for p in positions) + (tile_w_px * self.resolution_um_per_px / 2.0)
            max_y = max(p.get("positionY", 0.0) for p in positions) + (tile_h_px * self.resolution_um_per_px / 2.0)
            dim_x = max_x - origin_x
            dim_y = max_y - origin_y
        else:
            dim_x, dim_y = 50000.0, 70000.0

        # Read the sample depth from the analysis metadata.
        sample_meta = self.raw_analysis.get("sample_metadata", {})
        dim_z = sample_meta.get("total_sample_height_um", 6000.0)

        self.global_dimensions_um = [dim_x, dim_y, dim_z]
        print(f"[INFO] Global Origin set to Top-Left Corner of Tile (0,0): {self.global_origin_um}")
        print(f"[INFO] Global Dimensions (um): {self.global_dimensions_um}")
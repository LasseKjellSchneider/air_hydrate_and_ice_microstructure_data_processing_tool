import h5py
import numpy as np
import logging
import copy
import json
from typing import Dict, List, Tuple
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Union-find structure for tracking grains across adjacent layers.

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, i):
        if self.parent.setdefault(i, i) == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

# HDF5 extraction for grain contours, fabric analysis, and hydrates.

def extract_vector_contours(h5_path: str) -> Tuple[Dict[float, List[np.ndarray]], Dict[float, List[bool]]]:
    """ 
    Lädt immer das volle Sample.
    Gibt die Konturen und eine Liste von Booleans zurück (True = ist Border, False = ist Interior).
    """
    contours_by_z: Dict[float, List[np.ndarray]] = {}
    types_by_z: Dict[float, List[bool]] = {}
    
    with h5py.File(h5_path, 'r') as f:
        gm_attrs = f['global_metadata'].attrs
        res_um_per_px = gm_attrs.get('resolution_um_per_px', 
                        gm_attrs.get('mesh_resolution_um_per_vertex', 15.0) / 15.0)
            
        planes = []
        for pid in f['planes'].keys():
            meta = f['planes'][pid]['metadata'].attrs
            if not meta.get('is_fa', False):
                planes.append((pid, abs(float(meta.get('z_distance_mm', 0.0)))))
        
        planes.sort(key=lambda x: x[1])
        
        for pid, z_mm in planes:
            z_um = z_mm * 1000.0  
            try:
                dataset = f['planes'][pid]['vector_meshes']
                layer_contours = []
                layer_types = []
                for row in dataset:
                    mesh_type = row['mesh_type'].decode('utf-8') if isinstance(row['mesh_type'], bytes) else row['mesh_type']
                    if mesh_type not in ['interior', 'border']: continue
                        
                    wkt_str = row['polygon_wkt'].decode('utf-8') if isinstance(row['polygon_wkt'], bytes) else row['polygon_wkt']
                    start_idx, end_idx = wkt_str.find('(('), wkt_str.rfind('))')
                    
                    if start_idx != -1 and end_idx != -1:
                        pts_list = [
                            [float(v) for v in p.strip().split()[:2]] 
                            for p in wkt_str[start_idx+2 : end_idx].split(',') if len(p.strip().split()) >= 2
                        ]
                        if len(pts_list) < 3: continue
                        pts_um = np.array(pts_list) * res_um_per_px
                        if not np.array_equal(pts_um[0], pts_um[-1]):
                            pts_um = np.vstack([pts_um, pts_um[0]])
                            
                        layer_contours.append(pts_um)
                        layer_types.append(mesh_type != 'interior')  # True for border grains.
                        
                contours_by_z[z_um] = layer_contours
                types_by_z[z_um] = layer_types
            except KeyError:
                pass
    return contours_by_z, types_by_z

def extract_fa_data(h5_path: str) -> Dict[float, List[Dict]]:
    fa_by_z = {}
    with h5py.File(h5_path, 'r') as f:
        gm_attrs = f['global_metadata'].attrs
        res_um_per_px = gm_attrs.get('resolution_um_per_px', 1.41)
        
        for pid in f['planes'].keys():
            try:
                dataset = f['planes'][pid]['vector_meshes']
                z_mm = abs(float(f['planes'][pid]['metadata'].attrs.get('z_distance_mm', 0.0)))
                z_um = z_mm * 1000.0
                
                fa_polys = []
                for row in dataset:
                    mesh_type = row['mesh_type'].decode('utf-8') if isinstance(row['mesh_type'], bytes) else row['mesh_type']
                    if mesh_type == 'fa_grain':
                        wkt_str = row['polygon_wkt'].decode('utf-8') if isinstance(row['polygon_wkt'], bytes) else row['polygon_wkt']
                        start_idx, end_idx = wkt_str.find('(('), wkt_str.rfind('))')
                        if start_idx != -1 and end_idx != -1:
                            pts_list = [[float(v) for v in p.strip().split()[:2]] for p in wkt_str[start_idx+2 : end_idx].split(',')]
                            if len(pts_list) >= 3:
                                poly = Polygon(np.array(pts_list) * res_um_per_px).buffer(0)
                                
                                fa_polys.append({
                                    'poly': poly, 
                                    'c_axis': {
                                        'x': float(row['c_axis_x']), 
                                        'y': float(row['c_axis_y']), 
                                        'z': float(row['c_axis_z'])
                                    }
                                })
                if fa_polys:
                    fa_by_z[z_um] = fa_polys
            except KeyError:
                pass
    return fa_by_z

def extract_hydrates(h5_path: str) -> List[Dict]:
    hydrates = []
    with h5py.File(h5_path, 'r') as f:
        if 'hydrates' not in f: return hydrates
        grp = f['hydrates']
        for h_name in grp.keys():
            h_grp = grp[h_name]
            h_data = {}
            for k, v in h_grp.attrs.items():
                h_data[k] = v.item() if isinstance(v, np.generic) else v
            for sub_name in h_grp.keys():
                h_data[sub_name] = {}
                for k, v in h_grp[sub_name].attrs.items():
                    h_data[sub_name][k] = v.item() if isinstance(v, np.generic) else v
                    
            if 'world_coordinates_um' in h_data and 'z' in h_data['world_coordinates_um']:
                h_data['world_coordinates_um']['z_abs'] = abs(float(h_data['world_coordinates_um']['z']))
            hydrates.append(h_data)
    return hydrates

# Geometry and mesh construction helpers

def ensure_single_polygon(poly_or_multi):
    if poly_or_multi.is_empty: return None
    if poly_or_multi.geom_type == 'Polygon': return poly_or_multi
    if poly_or_multi.geom_type in ['MultiPolygon', 'GeometryCollection']:
        polys = [g for g in getattr(poly_or_multi, 'geoms', []) if g.geom_type == 'Polygon']
        if polys: return max(polys, key=lambda g: g.area)
    return None

def resample_line(coords: np.ndarray, num_points: int) -> np.ndarray:
    if len(coords) < 2: return np.tile(coords[0], (num_points, 1))
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]
    if total_dist == 0: return np.tile(coords[0], (num_points, 1))
    t_target = np.linspace(0, total_dist, num_points)
    return np.column_stack((interp1d(cum_dists, coords[:, 0])(t_target), interp1d(cum_dists, coords[:, 1])(t_target)))

def align_rings(t_ring: np.ndarray, b_ring: np.ndarray) -> np.ndarray:
    t_base, b_base = t_ring[:-1], b_ring[:-1]
    def signed_area(ring):
        return 0.5 * np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
    if (signed_area(t_base) > 0) != (signed_area(b_base) > 0): b_base = b_base[::-1]
    best_shift = np.argmin([np.sum(np.linalg.norm(t_base - np.roll(b_base, shift, axis=0), axis=1)) for shift in range(len(t_base))])
    b_aligned = np.roll(b_base, best_shift, axis=0)
    return np.vstack((b_aligned, b_aligned[0]))

def merge_and_filter_polys(poly_dict: Dict[int, Polygon], ids: List[int]) -> Tuple[np.ndarray, List[int], List[int]]:
    if len(ids) == 1:
        p = poly_dict[ids[0]]
        if p.geom_type != 'Polygon': p = max(p.geoms, key=lambda g: g.area)
        return np.array(p.exterior.coords), ids, []
        
    polys = [poly_dict[i].buffer(1.0, join_style=2) for i in ids]
    merged = unary_union(polys).buffer(-1.0, join_style=2)
    best_poly = max(merged.geoms, key=lambda g: g.area) if merged.geom_type == 'MultiPolygon' else merged
    
    if best_poly.is_empty or best_poly.geom_type != 'Polygon':
        backup = unary_union(polys)
        best_poly = max(backup.geoms, key=lambda g: g.area) if backup.geom_type == 'MultiPolygon' else backup
            
    kept, kicked = [], []
    for i in ids: (kept if poly_dict[i].intersects(best_poly) else kicked).append(i)
    
    if best_poly.geom_type != 'Polygon': best_poly = max(best_poly.geoms, key=lambda g: g.area)
    return np.array(best_poly.exterior.coords), kept, kicked

def create_tube_mesh(t_ring: np.ndarray, b_ring: np.ndarray, z_top: float, z_bot: float, mesh_res: int, steps: int = 10) -> tuple:
    all_verts, faces = [], []
    for i in range(steps + 1):
        t = i / steps
        ring_3d = np.column_stack(((1.0 - t) * t_ring + t * b_ring, np.full(mesh_res, z_top + t * (z_bot - z_top))))
        all_verts.append(ring_3d)
    for i in range(steps):
        r1, r2 = i * mesh_res, (i + 1) * mesh_res
        for j in range(mesh_res - 1):
            faces.extend([[r1+j, r2+j, r1+j+1], [r1+j+1, r2+j, r2+j+1]])
    return np.vstack(all_verts), faces

def create_apex_mesh(ring: np.ndarray, centroid: np.ndarray, z_base: float, z_apex: float, mesh_res: int, is_top: bool, steps: int = 10) -> tuple:
    apex_ring = np.tile(centroid, (mesh_res, 1))
    all_verts, faces = [], []
    for i in range(steps + 1):
        t = i / steps
        ring_3d = np.column_stack(((1.0 - t) * ring + t * apex_ring, np.full(mesh_res, z_base + t * (z_apex - z_base))))
        all_verts.append(ring_3d)
    for i in range(steps):
        r1, r2 = i * mesh_res, (i + 1) * mesh_res
        for j in range(mesh_res - 1):
            faces.extend([[r1+j, r2+j, r1+j+1], [r1+j+1, r2+j, r2+j+1]])
    if not is_top: all_verts = all_verts[::-1]
    return np.vstack(all_verts), faces

# Match grains across layers and construct 3D meshes.

def build_3d_meshes(contours_by_z: Dict[float, List[np.ndarray]], 
                    types_by_z: Dict[float, List[bool]],
                    min_overlap: float, split_overlap: float, 
                    mesh_resolution: int, apex_ratio: float) -> List[Dict]:
    
    if contours_by_z:
        z_keys = sorted(contours_by_z.keys())
        if z_keys[0] > 0.0: 
            contours_by_z[0.0] = copy.deepcopy(contours_by_z[z_keys[0]])
            types_by_z[0.0] = copy.deepcopy(types_by_z[z_keys[0]])
        if z_keys[-1] < 6000.0: 
            contours_by_z[6000.0] = copy.deepcopy(contours_by_z[z_keys[-1]])
            types_by_z[6000.0] = copy.deepcopy(types_by_z[z_keys[-1]])
            
    meshes = []
    uf = UnionFind()
    z_keys = sorted(contours_by_z.keys())
    
    ring_cache = {}  
    
    for i in range(len(z_keys) - 1):
        z_top, z_bot = z_keys[i], z_keys[i+1]
        z_apex_global = z_top + (z_bot - z_top) * apex_ratio
        
        top_polys = {idx: ensure_single_polygon(Polygon(c).buffer(0)) for idx, c in enumerate(contours_by_z[z_top]) if len(c) >= 3}
        top_polys = {k: v for k, v in top_polys.items() if v is not None and v.is_valid and not v.is_empty}
        bot_polys = {idx: ensure_single_polygon(Polygon(c).buffer(0)) for idx, c in enumerate(contours_by_z[z_bot]) if len(c) >= 3}
        bot_polys = {k: v for k, v in bot_polys.items() if v is not None and v.is_valid and not v.is_empty}

        intersections = []
        for t_idx, t_poly in top_polys.items():
            for b_idx, b_poly in bot_polys.items():
                try:
                    if t_poly.intersects(b_poly):
                        ix = t_poly.intersection(b_poly)
                        max_a = max(t_poly.area, b_poly.area)
                        if max_a > 0 and ix.area > 0:
                            intersections.append((ix.area, ix.area / max_a, t_idx, b_idx, t_poly.area, b_poly.area))
                except Exception: pass
                    
        intersections.sort(key=lambda x: x[0], reverse=True)
        matched_top, matched_bot = set(), set()
        clusters, t_to_hub, b_to_hub = {}, {}, {}
        
        for ix_area, ratio_max, t_idx, b_idx, t_a, b_a in intersections:
            if ratio_max >= min_overlap and t_idx not in matched_top and b_idx not in matched_bot:
                matched_top.add(t_idx); matched_bot.add(b_idx)
                clusters[(t_idx, b_idx)] = {'tops': [t_idx], 'bots': [b_idx]}
                t_to_hub[t_idx] = b_to_hub[b_idx] = (t_idx, b_idx)
                    
        for ix_area, ratio_max, t_idx, b_idx, t_a, b_a in intersections:
            if t_idx not in matched_top and b_idx in matched_bot and (ix_area / t_a if t_a > 0 else 0) >= split_overlap:
                hub_key = b_to_hub[b_idx]
                current_top_area = sum(top_polys[i].area for i in clusters[hub_key]['tops'])
                current_bot_area = sum(bot_polys[i].area for i in clusters[hub_key]['bots'])
                if (current_top_area + t_a) <= 1.5 * current_bot_area:
                    matched_top.add(t_idx)
                    clusters[hub_key]['tops'].append(t_idx)
                    t_to_hub[t_idx] = hub_key
                    
            elif b_idx not in matched_bot and t_idx in matched_top and (ix_area / b_a if b_a > 0 else 0) >= split_overlap:
                hub_key = t_to_hub[t_idx]
                current_top_area = sum(top_polys[i].area for i in clusters[hub_key]['tops'])
                current_bot_area = sum(bot_polys[i].area for i in clusters[hub_key]['bots'])
                if (current_bot_area + b_a) <= 1.5 * current_top_area:
                    matched_bot.add(b_idx)
                    clusters[hub_key]['bots'].append(b_idx)
                    b_to_hub[b_idx] = hub_key
        
        final_unassigned_top = set(top_polys.keys()) - matched_top
        final_unassigned_bot = set(bot_polys.keys()) - matched_bot
        
        for hub_key, cluster in clusters.items():
            t_coords, kept_t, kicked_t = merge_and_filter_polys(top_polys, cluster['tops'])
            b_coords, kept_b, kicked_b = merge_and_filter_polys(bot_polys, cluster['bots'])
            final_unassigned_top.update(kicked_t); final_unassigned_bot.update(kicked_b)
            
            if not kept_t: kept_t = cluster['tops']
            if not kept_b: kept_b = cluster['bots']
            
            for t in kept_t: uf.find(f"{z_top}_{t}")
            for b in kept_b: uf.find(f"{z_bot}_{b}")
            
            dom_t = max(kept_t, key=lambda x: top_polys[x].area)
            dom_b = max(kept_b, key=lambda x: bot_polys[x].area)
            
            root_node = f"{z_top}_{dom_t}"
            uf.union(root_node, f"{z_bot}_{dom_b}")
            
            if len(kept_t) == 1:
                t_idx = kept_t[0]
                cache_key = f"{z_top}_{t_idx}"
                if cache_key in ring_cache:
                    t_ring = ring_cache[cache_key]
                else:
                    t_ring = resample_line(t_coords, mesh_resolution)
            else:
                t_ring = resample_line(t_coords, mesh_resolution)
                
            b_ring = align_rings(t_ring, resample_line(b_coords, mesh_resolution))
            
            if len(kept_b) == 1:
                b_idx = kept_b[0]
                ring_cache[f"{z_bot}_{b_idx}"] = b_ring
            
            vol_um3 = ((Polygon(t_coords).area + Polygon(b_coords).area) / 2.0) * abs(z_bot - z_top)
            
            is_clean_match = (len(kept_t) == 1 and len(kept_b) == 1)
            verts, faces = create_tube_mesh(t_ring, b_ring, z_top, z_bot, mesh_resolution, steps=10)
            meshes.append({
                'vertices': verts, 'faces': faces, 'type': 'split' if not is_clean_match else 'clean',
                'meta': {'node': root_node, 'geom_type': 'tube', 't_ring': t_ring, 'b_ring': b_ring, 'z0': min(z_top, z_bot), 'z1': max(z_top, z_bot), 'volume_um3': vol_um3}
            })
        
        for t_idx in final_unassigned_top:
            c = top_polys[t_idx].centroid
            cache_key = f"{z_top}_{t_idx}"
            if cache_key in ring_cache:
                t_ring = ring_cache[cache_key]
            else:
                t_ring = resample_line(contours_by_z[z_top][t_idx], mesh_resolution)
                
            vol_um3 = (top_polys[t_idx].area / 2.0) * abs(z_apex_global - z_top)
            node = f"{z_top}_{t_idx}"
            uf.find(node) 
            verts, faces = create_apex_mesh(t_ring, np.array([c.x, c.y]), z_top, z_apex_global, mesh_resolution, True, steps=10)
            meshes.append({
                'vertices': verts, 'faces': faces, 'type': 'unassigned',
                'meta': {'node': node, 'geom_type': 'apex', 'ring': t_ring, 'centroid': np.array([c.x, c.y]), 'z0': min(z_top, z_apex_global), 'z1': max(z_top, z_apex_global), 'volume_um3': vol_um3}
            })
            
        for b_idx in final_unassigned_bot:
            c = bot_polys[b_idx].centroid
            b_ring = resample_line(contours_by_z[z_bot][b_idx], mesh_resolution)
            vol_um3 = (bot_polys[b_idx].area / 2.0) * abs(z_bot - z_apex_global)
            node = f"{z_bot}_{b_idx}"
            uf.find(node)
            verts, faces = create_apex_mesh(b_ring, np.array([c.x, c.y]), z_bot, z_apex_global, mesh_resolution, False, steps=10)
            meshes.append({
                'vertices': verts, 'faces': faces, 'type': 'unassigned',
                'meta': {'node': node, 'geom_type': 'apex', 'ring': b_ring, 'centroid': np.array([c.x, c.y]), 'z0': min(z_bot, z_apex_global), 'z1': max(z_bot, z_apex_global), 'volume_um3': vol_um3}
            })

    # Count the interior planes touched by each reconstructed grain.
    z_min_global, z_max_global = z_keys[0], z_keys[-1]
    global_grain_border_planes = {}
    
    for z, is_border_list in types_by_z.items():
        if z == z_min_global or z == z_max_global:
            continue
        for idx, is_border in enumerate(is_border_list):
            if is_border:
                node = f"{z}_{idx}"
                gid = f"Grain_Global_{uf.find(node)}"
                if gid not in global_grain_border_planes:
                    global_grain_border_planes[gid] = set()
                global_grain_border_planes[gid].add(z)
    
    for m in meshes:
        gid = f"Grain_Global_{uf.find(m['meta']['node'])}"
        m['meta']['global_grain_id'] = gid
        m['meta']['border_touch_count'] = len(global_grain_border_planes.get(gid, set()))
        
    logging.info(f"Generated 3D Graph with {len(set(uf.parent.values()))} unique Global Grains.")
    return meshes

# Assign hydrates to reconstructed grains and export the results.

def get_cross_section(mesh_meta: Dict, z_target: float) -> np.ndarray:
    z0, z1 = mesh_meta['z0'], mesh_meta['z1']
    if abs(z1 - z0) < 1e-6: t = 0.0
    else: t = max(0.0, min(1.0, (z_target - z0) / (z1 - z0)))
    
    if mesh_meta['geom_type'] == 'tube':
        return (1 - t) * mesh_meta['t_ring'] + t * mesh_meta['b_ring']
    else:
        return (1 - t) * mesh_meta['ring'] + t * np.tile(mesh_meta['centroid'], (len(mesh_meta['ring']), 1))

def assign_hydrates_and_export(meshes: List[Dict], h5_path: str, out_json_path: str):
    hydrates = extract_hydrates(h5_path)
    fa_by_z = extract_fa_data(h5_path)
    if not hydrates: return
    
    global_grains = {}
    for m in meshes: 
        global_grains.setdefault(m['meta']['global_grain_id'], []).append(m)

    export_grains = {}
    
    for gid, g_meshes in global_grains.items():
        total_vol = sum(m['meta']['volume_um3'] for m in g_meshes)
        c_axes_matches = []
        
        grain_z_min = min(m['meta']['z0'] for m in g_meshes)
        grain_z_max = max(m['meta']['z1'] for m in g_meshes)
        intersects_fa_plane = False
        
        for fa_z, fa_list in fa_by_z.items():
            if not (grain_z_min - 300.0 <= fa_z <= grain_z_max + 300.0):
                continue
            
            intersects_fa_plane = True
            spanning_meshes = [m for m in g_meshes if m['meta']['z0'] - 300.0 <= fa_z <= m['meta']['z1'] + 300.0]
            if not spanning_meshes:
                closest_m = min(g_meshes, key=lambda m: min(abs(m['meta']['z0'] - fa_z), abs(m['meta']['z1'] - fa_z)))
                spanning_meshes = [closest_m]
            
            cross_polys = []
            for m in spanning_meshes:
                cross_coords = get_cross_section(m['meta'], fa_z)
                p = Polygon(cross_coords)
                if not p.is_valid: p = p.buffer(0)
                if not p.is_empty: cross_polys.append(p)
                
            if not cross_polys: continue
            global_cross = unary_union(cross_polys)
            if global_cross.is_empty: continue
            
            for fa_item in fa_list:
                try:
                    if global_cross.intersects(fa_item['poly']):
                        ix_area = global_cross.intersection(fa_item['poly']).area
                        if ix_area > 0:
                            c_axes_matches.append({
                                "fa_z_level_um": fa_z,
                                "intersection_area_um2": float(ix_area),
                                "grain_cross_section_area_um2": float(global_cross.area),
                                "overlap_percentage": float((ix_area / global_cross.area) * 100) if global_cross.area > 0 else 0.0,
                                "c_axis_vector": fa_item['c_axis']
                            })
                except Exception: pass
                        
        export_grains[gid] = {
            "grain_id": str(gid),
            "total_volume_um3": float(total_vol),
            "border_touch_count": g_meshes[0]['meta']['border_touch_count'],
            "c_axis_matches": c_axes_matches,
            "contained_hydrates_ids": [],
            "nearest_fallback_hydrates_ids": [],
            "_intersects_fa_plane": intersects_fa_plane
        }

    for h in hydrates:
        hx, hy, hz = h['world_coordinates_um']['x'], h['world_coordinates_um']['y'], h['world_coordinates_um']['z_abs']
        h_pt = Point(hx, hy)
        assigned_gids = set() 
        
        for m in meshes:
            z0, z1 = m['meta']['z0'], m['meta']['z1']
            if z0 - 50.0 <= hz <= z1 + 50.0:
                cross = get_cross_section(m['meta'], hz)
                cross_poly = Polygon(cross)
                if not cross_poly.is_valid: cross_poly = cross_poly.buffer(0)
                
                if cross_poly.intersects(h_pt):
                    assigned_gids.add(m['meta']['global_grain_id'])
        
        assigned_gids = list(assigned_gids)
        h['assigned_grain_ids'] = assigned_gids 

    fallback_count = 0
    for h in hydrates:
        if len(h['assigned_grain_ids']) == 0:
            hx, hy, hz = h['world_coordinates_um']['x'], h['world_coordinates_um']['y'], h['world_coordinates_um']['z_abs']
            h_pt = Point(hx, hy)
            
            min_dist = float('inf')
            nearest_gid = None
            
            for m in meshes:
                z0, z1 = m['meta']['z0'], m['meta']['z1']
                cross_coords = get_cross_section(m['meta'], hz)
                cross_poly = Polygon(cross_coords)
                if not cross_poly.is_valid: cross_poly = cross_poly.buffer(0)
                
                d2d = cross_poly.distance(h_pt)
                dz = 0.0
                if hz < min(z0, z1): dz = min(z0, z1) - hz
                elif hz > max(z0, z1): dz = hz - max(z0, z1)
                
                dist_3d = np.sqrt(d2d**2 + dz**2)
                
                if dist_3d < min_dist:
                    min_dist = dist_3d
                    nearest_gid = m['meta']['global_grain_id']
                    
            if nearest_gid:
                h['assignment_type'] = "nearest_fallback"
                h['assigned_grain_ids'] = [nearest_gid]
                h['nearest_grain_distance_um'] = float(min_dist)
                h['c_axis_matches'] = export_grains[nearest_gid]['c_axis_matches']
                h['assigned_grains_border_touches'] = [export_grains[nearest_gid]['border_touch_count']]
                export_grains[nearest_gid]['nearest_fallback_hydrates_ids'].append(h['global_id'])
                fallback_count += 1
        else:
            h['assignment_type'] = "interior"
            h['nearest_grain_distance_um'] = 0.0
            
            all_c_axes = []
            h['assigned_grains_border_touches'] = []
            for gid in h['assigned_grain_ids']:
                all_c_axes.extend(export_grains[gid]['c_axis_matches'])
                export_grains[gid]['contained_hydrates_ids'].append(h['global_id'])
                h['assigned_grains_border_touches'].append(export_grains[gid]['border_touch_count'])
            h['c_axis_matches'] = all_c_axes

    total_hydrates = len(hydrates)
    hydrates_with_c_axis = sum(1 for h in hydrates if len(h.get('c_axis_matches', [])) > 0)
    single_count = sum(1 for h in hydrates if len(h['assigned_grain_ids']) == 1 and h.get('assignment_type') == 'interior')
    multiple_count = sum(1 for h in hydrates if len(h['assigned_grain_ids']) > 1 and h.get('assignment_type') == 'interior')

    total_grains = len(export_grains)
    grains_with_c_axis = sum(1 for g in export_grains.values() if len(g['c_axis_matches']) > 0)
    grains_intersecting_but_no_c_axis = sum(1 for g in export_grains.values() if g['_intersects_fa_plane'] and len(g['c_axis_matches']) == 0)
    grains_not_intersecting_fa = sum(1 for g in export_grains.values() if not g['_intersects_fa_plane'])

    for g in export_grains.values():
        g.pop('_intersects_fa_plane', None)

    assignment_statistics = {
        "total_hydrates": total_hydrates,
        "hydrates_with_c_axis_count": hydrates_with_c_axis,
        "singly_assigned_interior_hydrates_count": single_count,
        "multiply_assigned_interior_hydrates_count": multiple_count,
        "nearest_fallback_assigned_hydrates_count": fallback_count,
        "notes": "Hydrates categorized as 'interior' lie geometrically inside the 3D grain boundaries. Hydrates categorized as 'nearest_fallback' lay outside the reconstructed geometry (e.g. at sample borders) and were assigned the C-Axis of the closest physical 3D grain via Euclidean distance calculation."
    }

    grain_statistics = {
        "total_3d_grains_reconstructed": total_grains,
        "grains_with_c_axis_count": grains_with_c_axis,
        "grains_intersecting_fa_but_no_c_axis_count": grains_intersecting_but_no_c_axis,
        "grains_not_intersecting_fa_count": grains_not_intersecting_fa,
        "total_volume_all_grains_um3": sum(g['total_volume_um3'] for g in export_grains.values()),
        "notes": "Global Grains are reconstructed using the area overlap and dominant lineage approach. Coordinates have NOT been altered. C_x=1 points to Zenith."
    }

    export_data = {
        "grain_statistics": grain_statistics,
        "assignment_statistics": assignment_statistics,
        "hydrates": hydrates,
        "grains": list(export_grains.values())
    }
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4)
        
    logging.info(f"Done! Evaluated fallbacks for {fallback_count} outside hydrates.")
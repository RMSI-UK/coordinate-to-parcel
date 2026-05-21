from __future__ import annotations

from dataclasses import dataclass
from typing import List

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.ops import polygonize, unary_union
from tqdm import tqdm


@dataclass(frozen=True)
class _CellCandidate:
    pos: int
    area: float
    cx: float
    cy: float
    contains_point: bool
    dist_to_point: float
    centroid_dist: float


def _subset_table(candidates: list[_CellCandidate], px: float, py: float, max_area: float):
    rows = [(0, 0.0, 0.0, 0.0, 0)]
    for bit, cand in enumerate(candidates):
        add_rows = []
        area_vec_x = cand.area * (cand.cx - px)
        area_vec_y = cand.area * (cand.cy - py)
        mask_bit = 1 << bit
        for mask, area, vx, vy, count in rows:
            new_area = area + cand.area
            if new_area <= max_area:
                add_rows.append((mask | mask_bit, new_area, vx + area_vec_x, vy + area_vec_y, count + 1))
        rows.extend(add_rows)
    return rows


def _polygonized_cells(point_geom, polygon_gdf: gpd.GeoDataFrame, *, search_radius: float, max_area: float):
    sidx = polygon_gdf.sindex
    query_geom = point_geom.buffer(float(search_radius))
    idxs = list(sidx.query(query_geom, predicate="intersects")) if sidx is not None else []
    if not idxs:
        return []

    clip_env = query_geom.envelope.buffer(1.0)
    lines = []
    for pos in idxs:
        geom = polygon_gdf.geometry.iloc[int(pos)]
        if geom is None or geom.is_empty:
            continue
        boundary = geom.boundary.intersection(clip_env)
        if boundary is not None and not boundary.is_empty:
            lines.append(boundary)
    if not lines:
        return []

    linework = unary_union(lines)
    cells = []
    for cell in polygonize(linework):
        if cell is None or cell.is_empty:
            continue
        area = float(cell.area)
        if area <= 0.01 or area > max_area:
            continue
        if not cell.intersects(query_geom):
            continue
        cells.append(cell)
    return cells


def _polygonized_cells_multi(point_geom, polygon_gdfs: list[gpd.GeoDataFrame], *, search_radius: float, max_area: float):
    query_geom = point_geom.buffer(float(search_radius))
    clip_env = query_geom.envelope.buffer(1.0)
    lines = []
    for polygon_gdf in polygon_gdfs:
        if polygon_gdf.empty:
            continue
        sidx = polygon_gdf.sindex
        idxs = list(sidx.query(query_geom, predicate="intersects")) if sidx is not None else []
        for pos in idxs:
            geom = polygon_gdf.geometry.iloc[int(pos)]
            if geom is None or geom.is_empty:
                continue
            boundary = geom.boundary.intersection(clip_env)
            if boundary is not None and not boundary.is_empty:
                lines.append(boundary)
    if not lines:
        return []

    linework = unary_union(lines)
    cells = []
    for cell in polygonize(linework):
        if cell is None or cell.is_empty:
            continue
        area = float(cell.area)
        if area <= 0.01 or area > max_area:
            continue
        if not cell.intersects(query_geom):
            continue
        cells.append(cell)
    return cells


def _best_cell_combo(
    point_geom,
    cells: list[object],
    *,
    tolerance: float,
    max_candidates: int,
    max_area: float,
    nearest_matches: int,
):
    candidates: list[_CellCandidate] = []
    for pos, cell in enumerate(cells):
        area = float(cell.area)
        centroid = cell.centroid
        candidates.append(
            _CellCandidate(
                pos=pos,
                area=area,
                cx=float(centroid.x),
                cy=float(centroid.y),
                contains_point=bool(cell.intersects(point_geom)),
                dist_to_point=float(cell.distance(point_geom)),
                centroid_dist=float(centroid.distance(point_geom)),
            )
        )
    candidates.sort(key=lambda c: (not c.contains_point, c.dist_to_point, c.centroid_dist, c.area, c.pos))
    candidates = candidates[:max_candidates]
    if not candidates:
        return None

    mid = len(candidates) // 2
    left = _subset_table(candidates[:mid], point_geom.x, point_geom.y, max_area)
    right = _subset_table(candidates[mid:], point_geom.x, point_geom.y, max_area)
    if not right:
        return None

    right_vectors = np.array([(row[2], row[3]) for row in right], dtype=float)
    tree = cKDTree(right_vectors)
    best_key = None
    best_masks = None
    k = min(max(1, int(nearest_matches)), len(right))
    for left_mask, left_area, left_vx, left_vy, left_count in left:
        if left_area > max_area:
            continue
        _, right_idxs = tree.query([-left_vx, -left_vy], k=k)
        if np.isscalar(right_idxs):
            right_idxs = [int(right_idxs)]
        for right_idx in right_idxs:
            right_mask, right_area, right_vx, right_vy, right_count = right[int(right_idx)]
            area = left_area + right_area
            if area <= 0.0 or area > max_area:
                continue
            vx = left_vx + right_vx
            vy = left_vy + right_vy
            centroid_dist = ((vx * vx + vy * vy) ** 0.5) / area
            if centroid_dist > tolerance:
                continue
            count = left_count + right_count
            key = (centroid_dist, count, -area)
            if best_key is None or key < best_key:
                best_key = key
                best_masks = (left_mask, right_mask)

    if best_masks is None:
        return None

    left_mask, right_mask = best_masks
    selected = []
    for bit, cand in enumerate(candidates[:mid]):
        if left_mask & (1 << bit):
            selected.append(cells[cand.pos])
    for bit, cand in enumerate(candidates[mid:]):
        if right_mask & (1 << bit):
            selected.append(cells[cand.pos])
    if not selected:
        return None
    combo = selected[0] if len(selected) == 1 else unary_union(selected)
    if combo is None or combo.is_empty:
        return None
    return combo


def pick_polygonized_cell_centroid_combo(
    point_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    *,
    tolerance: float,
    search_radius: float = 40.0,
    max_candidates: int = 24,
    max_area: float = 850.0,
    nearest_matches: int = 128,
    desc: str = "Point polygonized cell combo",
) -> gpd.GeoDataFrame:
    if point_gdf.empty or polygon_gdf.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)

    rows: List[dict] = []
    for _, row in tqdm(point_gdf.iterrows(), total=len(point_gdf), desc=desc):
        point_geom = row.geometry
        if point_geom is None or point_geom.is_empty:
            continue
        cells = _polygonized_cells(point_geom, polygon_gdf, search_radius=search_radius, max_area=max_area)
        geom = _best_cell_combo(
            point_geom,
            cells,
            tolerance=tolerance,
            max_candidates=max_candidates,
            max_area=max_area,
            nearest_matches=nearest_matches,
        )
        if geom is not None:
            rows.append({"capture_src_id": int(row["capture_src_id"]), "geometry": geom})

    if not rows:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=point_gdf.crs)

def pick_polygonized_cell_centroid_combo_multi(
    point_gdf: gpd.GeoDataFrame,
    polygon_gdfs: list[gpd.GeoDataFrame],
    *,
    tolerance: float,
    search_radius: float = 40.0,
    max_candidates: int = 24,
    max_area: float = 850.0,
    nearest_matches: int = 128,
    desc: str = "Point combined polygonized cell combo",
) -> gpd.GeoDataFrame:
    if point_gdf.empty or not polygon_gdfs:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)

    polygon_gdfs = [gdf for gdf in polygon_gdfs if not gdf.empty]
    if not polygon_gdfs:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)

    rows: List[dict] = []
    for _, row in tqdm(point_gdf.iterrows(), total=len(point_gdf), desc=desc):
        point_geom = row.geometry
        if point_geom is None or point_geom.is_empty:
            continue
        cells = _polygonized_cells_multi(point_geom, polygon_gdfs, search_radius=search_radius, max_area=max_area)
        geom = _best_cell_combo(
            point_geom,
            cells,
            tolerance=tolerance,
            max_candidates=max_candidates,
            max_area=max_area,
            nearest_matches=nearest_matches,
        )
        if geom is not None:
            rows.append({"capture_src_id": int(row["capture_src_id"]), "geometry": geom})

    if not rows:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=point_gdf.crs)


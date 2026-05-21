from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm


def _search_window(poly, area_scale: float):
    base_area = float(poly.area)
    if base_area <= 0:
        minx, miny, maxx, maxy = poly.bounds
        return box(minx - 1, miny - 1, maxx + 1, maxy + 1)
    side = (base_area * max(area_scale, 1.0)) ** 0.5
    c = poly.centroid
    half = side / 2.0
    return box(c.x - half, c.y - half, c.x + half, c.y + half)


def _candidate_subset(poly, basemap: gpd.GeoDataFrame, max_candidates: int, area_tolerance: float, area_scale: float):
    if basemap.empty:
        return basemap.iloc[0:0].copy()
    window = _search_window(poly, area_scale=area_scale)
    sidx = basemap.sindex
    if sidx is None:
        subset = basemap[basemap.geometry.intersects(window)].copy()
    else:
        idxs = list(sidx.query(window, predicate="intersects"))
        subset = basemap.iloc[idxs].copy() if idxs else basemap.iloc[0:0].copy()
    if subset.empty:
        return subset

    area = float(poly.area)
    if area > 0:
        low = area * max(0.0, 1.0 - area_tolerance)
        high = area * (1.0 + area_tolerance)
        area_filtered = subset[(subset.geometry.area >= low) & (subset.geometry.area <= high)].copy()
        if not area_filtered.empty:
            subset = area_filtered

    subset["__int_area"] = subset.geometry.intersection(poly).area
    subset = subset.sort_values(["__int_area"], ascending=False).head(max_candidates).drop(columns=["__int_area"])
    return subset


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def _area_similarity(a, b) -> float:
    aa = float(a.area)
    bb = float(b.area)
    if aa <= 0 or bb <= 0:
        return 0.0
    return float(min(aa, bb) / max(aa, bb))


def _boundary_snap_score(src, cand, snap_tolerance: float) -> float:
    src_boundary = src.boundary
    src_len = float(src_boundary.length)
    if src_len <= 0:
        return 0.0
    near_part = src_boundary.intersection(cand.boundary.buffer(max(snap_tolerance, 1e-6)))
    near_len = float(near_part.length)
    return max(0.0, min(1.0, near_len / src_len))


def _best_merge_for_polygon(
    src_geom,
    basemap_subset: gpd.GeoDataFrame,
    min_iou: float,
    max_combo_size: int,
    distance_tolerance: float,
) -> Tuple[Optional[object], float, int]:
    if basemap_subset.empty:
        return None, 0.0, 0

    geoms = [g for g in basemap_subset.geometry.tolist() if g is not None and not g.is_empty]
    if not geoms:
        return None, 0.0, 0

    best_geom = None
    best_snap = -1.0
    best_iou = 0.0
    best_area_sim = 0.0
    best_count = 0

    max_k = max(1, min(max_combo_size, len(geoms)))
    for k in range(1, max_k + 1):
        for combo in combinations(range(len(geoms)), k):
            merged = unary_union([geoms[i] for i in combo])
            if merged.is_empty:
                continue
            if merged.distance(src_geom) > distance_tolerance:
                continue
            snap_score = _boundary_snap_score(src_geom, merged, snap_tolerance=distance_tolerance)
            iou = _iou(src_geom, merged)
            area_sim = _area_similarity(src_geom, merged)
            if (
                (snap_score > best_snap)
                or (
                    snap_score == best_snap
                    and (
                        (iou > best_iou)
                        or (iou == best_iou and area_sim > best_area_sim)
                        or (iou == best_iou and area_sim == best_area_sim and (best_count == 0 or k < best_count))
                    )
                )
            ):
                best_snap = snap_score
                best_iou = iou
                best_area_sim = area_sim
                best_geom = merged
                best_count = k

    if best_geom is None or best_iou < min_iou:
        return None, best_iou, best_count
    return best_geom, best_iou, best_count


def run_inline_merge_batch(
    input_polygons: gpd.GeoDataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    min_iou: float,
    distance_tolerance: float,
    area_tolerance: float,
    max_candidates: int,
    max_combo_size: int,
    window_area_scale: float,
    desc: str,
) -> gpd.GeoDataFrame:
    out_rows: List[Dict[str, object]] = []
    if input_polygons.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs="EPSG:27700")

    for _, row in tqdm(input_polygons.iterrows(), total=len(input_polygons), desc=desc):
        src_id = int(row["capture_src_id"])
        src_geom = row.geometry
        if src_geom is None or src_geom.is_empty:
            continue
        subset = _candidate_subset(
            poly=src_geom,
            basemap=basemap_gdf,
            max_candidates=max_candidates,
            area_tolerance=area_tolerance,
            area_scale=window_area_scale,
        )
        merged, iou, green_count = _best_merge_for_polygon(
            src_geom=src_geom,
            basemap_subset=subset,
            min_iou=min_iou,
            max_combo_size=max_combo_size,
            distance_tolerance=distance_tolerance,
        )
        if merged is None:
            continue
        out_rows.append(
            {
                "capture_src_id": src_id,
                "geometry": merged,
                "iou": float(iou),
                "green_count": int(green_count),
            }
        )

    if not out_rows:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=input_polygons.crs)
    return gpd.GeoDataFrame(out_rows, geometry="geometry", crs=input_polygons.crs)

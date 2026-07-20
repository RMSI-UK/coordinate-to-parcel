from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import geopandas as gpd
import pandas as pd
import shapely
from shapely import set_precision
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

DEFAULT_TARGET_CRS = "EPSG:27700"
GAPFILL_COUNCIL_PREFIX = "gapfill_council_"


@dataclass(frozen=True)
class WfsMergeConfig:
    max_merge_area_m2: float = 2000.0
    min_council_merge_iou: float = 0.75
    min_council_merge_council_coverage: float = 0.80
    min_council_merge_wfs_coverage: float = 0.70
    max_council_merge_wfs_count: int = 20
    notch_fill_max_small_to_large_area_ratio: float = 0.35
    notch_fill_max_small_area_m2: float = 150.0
    notch_fill_min_shared_edge_count: int = 3
    notch_fill_min_shared_small_perimeter_ratio: float = 0.45
    notch_fill_min_hull_gap_improvement_ratio: float = 0.25
    overlap_drop_min_small_coverage: float = 0.98
    overlap_drop_min_large_to_small_ratio: float = 1.02
    overlap_drop_min_duplicate_iou: float = 0.98
    min_polygon_part_area_m2: float = 0.01
    merge_shape_min_mrr_ratio: float = 0.45
    merge_shape_max_hull_gap_ratio: float = 0.55
    merge_shape_min_compactness: float = 0.36
    merge_shape_max_weird_angle_count: int = 8
    merge_shape_min_weird_angle_deg: float = 5.0
    merge_shape_max_weird_angle_deg: float = 170.0
    merge_shape_right_angle_min_deg: float = 75.0
    merge_shape_right_angle_max_deg: float = 105.0
    preprocess_precision_grid: float = 0.0
    explode_multipart_inputs: bool = True
    clean_input_overlaps: bool = True
    fill_output_holes: bool = True
    fill_enclosed_gap_holes: bool = True
    enclosed_gap_max_area_m2: float = 250.0
    enclosed_gap_min_shared_edge_m: float = 0.05


_DEFAULT_CONFIG = WfsMergeConfig()
MAX_MERGE_AREA_M2 = _DEFAULT_CONFIG.max_merge_area_m2
MIN_COUNCIL_MERGE_IOU = _DEFAULT_CONFIG.min_council_merge_iou
MIN_COUNCIL_MERGE_COUNCIL_COVERAGE = _DEFAULT_CONFIG.min_council_merge_council_coverage
MIN_COUNCIL_MERGE_WFS_COVERAGE = _DEFAULT_CONFIG.min_council_merge_wfs_coverage
MAX_COUNCIL_MERGE_WFS_COUNT = _DEFAULT_CONFIG.max_council_merge_wfs_count
NOTCH_FILL_MAX_SMALL_TO_LARGE_AREA_RATIO = _DEFAULT_CONFIG.notch_fill_max_small_to_large_area_ratio
NOTCH_FILL_MAX_SMALL_AREA_M2 = _DEFAULT_CONFIG.notch_fill_max_small_area_m2
NOTCH_FILL_MIN_SHARED_EDGE_COUNT = _DEFAULT_CONFIG.notch_fill_min_shared_edge_count
NOTCH_FILL_MIN_SHARED_SMALL_PERIMETER_RATIO = _DEFAULT_CONFIG.notch_fill_min_shared_small_perimeter_ratio
NOTCH_FILL_MIN_HULL_GAP_IMPROVEMENT_RATIO = _DEFAULT_CONFIG.notch_fill_min_hull_gap_improvement_ratio
OVERLAP_DROP_MIN_SMALL_COVERAGE = _DEFAULT_CONFIG.overlap_drop_min_small_coverage
OVERLAP_DROP_MIN_LARGE_TO_SMALL_RATIO = _DEFAULT_CONFIG.overlap_drop_min_large_to_small_ratio
OVERLAP_DROP_MIN_DUPLICATE_IOU = _DEFAULT_CONFIG.overlap_drop_min_duplicate_iou
MIN_POLYGON_PART_AREA_M2 = _DEFAULT_CONFIG.min_polygon_part_area_m2


def _validate_input(gdf: gpd.GeoDataFrame, theme_field: str) -> None:
    if theme_field not in gdf.columns:
        raise ValueError(f"Theme field not found: {theme_field}")
    if gdf.geometry.name not in gdf.columns:
        raise ValueError("Geometry column is missing.")


def _make_valid_geometry(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        if geom.is_valid:
            return geom
    except Exception:
        pass
    try:
        fixed = make_valid(geom)
    except Exception:
        fixed = geom.buffer(0)
    if fixed is None or fixed.is_empty:
        return fixed
    try:
        if fixed.is_valid:
            return fixed
    except Exception:
        return fixed
    try:
        return fixed.buffer(0)
    except Exception:
        return fixed


def _extract_polygonal_geometry(geom):
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            extracted = _extract_polygonal_geometry(part)
            if extracted is None or extracted.is_empty:
                continue
            if extracted.geom_type == "Polygon":
                parts.append(extracted)
            elif extracted.geom_type == "MultiPolygon":
                parts.extend(list(extracted.geoms))
        if not parts:
            return Polygon()
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)
    return Polygon()


def _polygon_part_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return 1
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return sum(_polygon_part_count(part) for part in geom.geoms)
    return 0


def _drop_tiny_polygon_parts(geom, min_area: float = MIN_POLYGON_PART_AREA_M2):
    if geom is None or geom.is_empty:
        return geom
    geom = _extract_polygonal_geometry(_make_valid_geometry(geom))
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type == "Polygon":
        return geom if float(geom.area) >= min_area else Polygon()
    if geom.geom_type == "MultiPolygon":
        parts = [part for part in geom.geoms if float(part.area) >= min_area]
        if not parts:
            return Polygon()
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)
    return Polygon()


def _clean_tiny_polygon_parts(
    gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    cfg = config or WfsMergeConfig()
    out = gdf.copy()
    out = out.set_geometry(
        out.geometry.apply(lambda geom: _drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2))
    )
    return out[out.geometry.notna() & ~out.geometry.is_empty].copy()


def _polygon_exterior_angles(geom, config: WfsMergeConfig | None = None) -> list[float]:
    cfg = config or WfsMergeConfig()
    geom = _drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2)
    parts = []
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        parts = [geom]
    elif geom.geom_type == "MultiPolygon":
        parts = list(geom.geoms)
    else:
        return []
    if not parts:
        return []

    polygon = max(parts, key=lambda part: float(part.area))
    coords = list(polygon.exterior.coords)
    if len(coords) < 4:
        return []
    points = coords[:-1]
    angles: list[float] = []
    point_count = len(points)
    for idx, point in enumerate(points):
        prev_point = points[(idx - 1) % point_count]
        next_point = points[(idx + 1) % point_count]
        v1 = (prev_point[0] - point[0], prev_point[1] - point[1])
        v2 = (next_point[0] - point[0], next_point[1] - point[1])
        len1 = math.hypot(v1[0], v1[1])
        len2 = math.hypot(v2[0], v2[1])
        if len1 < 0.05 or len2 < 0.05:
            continue
        cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def _is_reasonable_merged_shape(geom, config: WfsMergeConfig | None = None) -> bool:
    cfg = config or WfsMergeConfig()
    geom = _drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2)
    if geom is None or geom.is_empty:
        return False
    if _polygon_part_count(geom) > 1:
        return False

    area = float(geom.area)
    perimeter = float(geom.length)
    if area <= 0.0 or perimeter <= 0.0:
        return False

    convex_hull_area = float(geom.convex_hull.area)
    min_rect_area = float(geom.minimum_rotated_rectangle.area)
    if convex_hull_area <= 0.0 or min_rect_area <= 0.0:
        return False

    hull_gap_ratio = max(0.0, convex_hull_area - area) / area
    min_rect_ratio = area / min_rect_area
    compactness = 4.0 * math.pi * area / (perimeter * perimeter)
    angles = _polygon_exterior_angles(geom, cfg)
    weird_angles = [
        angle
        for angle in angles
        if cfg.merge_shape_min_weird_angle_deg < angle < cfg.merge_shape_max_weird_angle_deg
        and not (cfg.merge_shape_right_angle_min_deg <= angle <= cfg.merge_shape_right_angle_max_deg)
    ]

    if (
        min_rect_ratio < cfg.merge_shape_min_mrr_ratio
        and hull_gap_ratio > cfg.merge_shape_max_hull_gap_ratio
        and compactness < cfg.merge_shape_min_compactness
    ):
        return False
    if (
        len(weird_angles) >= cfg.merge_shape_max_weird_angle_count
        and min_rect_ratio < 0.50
        and hull_gap_ratio > 0.45
        and compactness < 0.40
    ):
        return False
    return True


def _is_allowed_merged_geometry(geom, config: WfsMergeConfig | None = None) -> bool:
    cfg = config or WfsMergeConfig()
    geom = _drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2)
    if geom is None or geom.is_empty:
        return False
    return (
        float(geom.area) <= cfg.max_merge_area_m2
        and _polygon_part_count(geom) <= 1
        and _is_reasonable_merged_shape(geom, cfg)
    )


def _is_allowed_council_merge(merged_geom, council_geom, config: WfsMergeConfig | None = None) -> bool:
    cfg = config or WfsMergeConfig()
    if not _is_allowed_merged_geometry(merged_geom, cfg):
        return False
    if council_geom is None or council_geom.is_empty:
        return False

    merged_area = float(merged_geom.area)
    council_area = float(council_geom.area)
    if merged_area <= 0.0 or council_area <= 0.0:
        return False

    inter_area = float(merged_geom.intersection(council_geom).area)
    union_area = float(merged_geom.union(council_geom).area)
    iou = inter_area / union_area if union_area else 0.0
    council_coverage = inter_area / council_area
    wfs_coverage = inter_area / merged_area
    return (
        iou >= cfg.min_council_merge_iou
        and council_coverage >= cfg.min_council_merge_council_coverage
        and wfs_coverage >= cfg.min_council_merge_wfs_coverage
    )


def _fill_polygon_holes(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(part.exterior) for part in geom.geoms if not part.is_empty])
    if geom.geom_type == "GeometryCollection":
        polygons = []
        for part in geom.geoms:
            filled = _fill_polygon_holes(part)
            if filled is None or filled.is_empty:
                continue
            if filled.geom_type == "Polygon":
                polygons.append(filled)
            elif filled.geom_type == "MultiPolygon":
                polygons.extend(list(filled.geoms))
        return MultiPolygon(polygons) if polygons else Polygon()
    return geom


def _fill_geometry_holes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.copy()
    out = out.set_geometry(out.geometry.apply(_fill_polygon_holes))
    return out


def _safe_intersection_area(left, right) -> float:
    try:
        return float(left.intersection(right).area)
    except Exception:
        return float(_make_valid_geometry(left).intersection(_make_valid_geometry(right)).area)


def _remove_nearly_covered_smaller_polygons(
    gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    if len(gdf) < 2:
        return gdf.copy()

    out = gdf.copy()
    geom_col = out.geometry.name
    polygon_mask = out.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
    candidates = out.loc[polygon_mask, [geom_col] + (["__fid"] if "__fid" in out.columns else [])].copy()
    candidates = candidates[candidates.geometry.notna() & ~candidates.geometry.is_empty].copy()
    if len(candidates) < 2:
        return out

    candidates["__coverage_cleanup_area"] = candidates.geometry.area.astype(float)
    candidates = candidates[candidates["__coverage_cleanup_area"] > 0.0].copy()
    if len(candidates) < 2:
        return out

    if "__fid" in candidates.columns:
        ordered = candidates.sort_values(["__coverage_cleanup_area", "__fid"]).index.tolist()
    else:
        ordered = candidates.sort_values(["__coverage_cleanup_area"]).index.tolist()
    rank_by_idx = {idx: rank for rank, idx in enumerate(ordered)}
    area_by_idx = candidates["__coverage_cleanup_area"].to_dict()
    geom_by_idx = candidates.geometry.to_dict()
    sindex = candidates.sindex

    drop_indices: Set[object] = set()
    for small_idx in ordered:
        if small_idx in drop_indices:
            continue
        small_geom = geom_by_idx[small_idx]
        small_area = float(area_by_idx[small_idx])
        if small_area <= 0.0:
            continue

        try:
            neighbor_positions = list(sindex.query(small_geom, predicate="intersects"))
        except TypeError:
            neighbor_positions = list(sindex.query(small_geom))

        for neighbor_pos in neighbor_positions:
            other_idx = candidates.index[int(neighbor_pos)]
            if other_idx == small_idx or other_idx in drop_indices:
                continue
            other_area = float(area_by_idx[other_idx])
            if other_area + 1e-9 < small_area:
                continue
            if abs(other_area - small_area) <= 1e-9 and rank_by_idx[other_idx] > rank_by_idx[small_idx]:
                continue

            intersection_area = _safe_intersection_area(small_geom, geom_by_idx[other_idx])
            if intersection_area <= 0.0:
                continue
            small_coverage = intersection_area / small_area
            if small_coverage < cfg.overlap_drop_min_small_coverage:
                continue

            union_area = small_area + other_area - intersection_area
            iou = intersection_area / union_area if union_area > 0.0 else 0.0
            large_to_small_ratio = other_area / small_area
            if (
                large_to_small_ratio >= cfg.overlap_drop_min_large_to_small_ratio
                or iou >= cfg.overlap_drop_min_duplicate_iou
            ):
                drop_indices.add(small_idx)
                break

    if not drop_indices:
        return out
    return out.drop(index=list(drop_indices)).copy()


def _normalise_crs(gdf: gpd.GeoDataFrame, target_crs: object | None = None) -> gpd.GeoDataFrame:
    out = gdf.copy()
    use_crs = target_crs or out.crs or DEFAULT_TARGET_CRS
    if out.crs is None:
        return out.set_crs(use_crs)
    if target_crs is not None and out.crs != target_crs:
        return out.to_crs(target_crs)
    return out


def _source_value_from_index(value):
    text = str(value)
    return int(text) if text.lstrip("-").isdigit() else value


def preprocess_polygon_layer(
    gdf: gpd.GeoDataFrame,
    *,
    layer_name: str,
    target_crs: object | None = None,
    source_id_column: str = "source_fid",
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    if gdf.empty:
        return _normalise_crs(gdf.copy(), target_crs)

    out = _normalise_crs(gdf, target_crs)
    if source_id_column and source_id_column not in out.columns:
        out[source_id_column] = [_source_value_from_index(idx) for idx in out.index]
    if layer_name == "wfs" and "source_gmlid" not in out.columns and "GmlID" in out.columns:
        out["source_gmlid"] = out["GmlID"].fillna("").astype(str)

    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.empty:
        return out

    out = out.set_geometry(
        out.geometry.apply(_make_valid_geometry)
        .apply(_extract_polygonal_geometry)
        .apply(lambda geom: _drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2))
    )
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.empty:
        return out

    if cfg.explode_multipart_inputs:
        out = out.explode(index_parts=False, ignore_index=True)

    if cfg.preprocess_precision_grid and cfg.preprocess_precision_grid > 0.0:
        out = out.set_geometry(
            out.geometry.apply(lambda geom: set_precision(geom, cfg.preprocess_precision_grid))
            .apply(lambda geom: _drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2))
        )
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()

    if cfg.clean_input_overlaps:
        out = _remove_nearly_covered_smaller_polygons(out, cfg)
    return out.reset_index(drop=True)


def preprocess_wfs_layer(
    gdf: gpd.GeoDataFrame,
    *,
    target_crs: object | None = None,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    return preprocess_polygon_layer(
        gdf,
        layer_name="wfs",
        target_crs=target_crs,
        source_id_column="source_fid",
        config=config,
    )


def preprocess_council_layer(
    gdf: gpd.GeoDataFrame,
    *,
    target_crs: object | None = None,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    return preprocess_polygon_layer(
        gdf,
        layer_name="council",
        target_crs=target_crs,
        source_id_column="council_source_fid",
        config=config,
    )


def _choose_layer(gpkg_path: str, layer: Optional[str]) -> str:
    if layer:
        return layer
    layers = gpd.list_layers(gpkg_path)
    if layers.empty:
        raise ValueError(f"No layers found in: {gpkg_path}")
    return str(layers.iloc[0]["name"])


def load_layer(gpkg_path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    use_layer = _choose_layer(gpkg_path, layer)
    try:
        gdf = gpd.read_file(gpkg_path, layer=use_layer, engine="pyogrio", fid_as_index=True)
    except TypeError:
        gdf = gpd.read_file(gpkg_path, layer=use_layer)
    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_TARGET_CRS)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def resolve_theme_field(gdf: gpd.GeoDataFrame, preferred: str = "Theme") -> str:
    if preferred in gdf.columns:
        return preferred
    lower_to_original = {str(col).lower(): str(col) for col in gdf.columns}
    for candidate in ("theme", preferred.lower()):
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(f"Theme field not found. Available columns: {list(gdf.columns)}")


def filter_wfs_theme_features(
    gdf: gpd.GeoDataFrame,
    theme_field: str | None = None,
    include_terms: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    use_theme_field = theme_field or resolve_theme_field(gdf)
    terms = tuple(term.strip().lower() for term in (include_terms or ("land", "building")) if term.strip())
    if not terms:
        return gdf.iloc[0:0].copy()
    theme = gdf[use_theme_field].fillna("").astype(str).str.lower()
    keep_mask = theme.eq("__capture_no_theme_match__")
    for term in terms:
        keep_mask = keep_mask | theme.str.contains(term, regex=False)
    return gdf.loc[keep_mask].copy()


def _merge_role_series(gdf: gpd.GeoDataFrame, theme_field: str) -> pd.Series:
    theme = gdf[theme_field].fillna("").astype(str)
    building_mask = theme.str.contains("building", case=False, regex=False)
    land_mask = theme.str.contains("land", case=False, regex=False) & (~building_mask)
    role = pd.Series("", index=gdf.index, dtype="string")
    role.loc[building_mask] = "building"
    role.loc[land_mask] = "land"
    return role


def _coerce_source_parts(value) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).split("|")
    return [str(item).strip() for item in values if str(item).strip()]


def _combined_source_fids(row_by_fid: pd.DataFrame, fids: Sequence[int]) -> str:
    parts: list[str] = []
    seen: Set[str] = set()
    for fid in fids:
        if int(fid) not in row_by_fid.index:
            continue
        row = row_by_fid.loc[int(fid)]
        values = _coerce_source_parts(row.get("merge_source_fids"))
        if not values and "source_fid" in row.index:
            values = _coerce_source_parts(row.get("source_fid"))
        if not values:
            values = [str(fid)]
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            parts.append(value)
    return "|".join(parts)


def _combined_source_count(row_by_fid: pd.DataFrame, fids: Sequence[int]) -> int:
    return len(_coerce_source_parts(_combined_source_fids(row_by_fid, fids)))


def _apply_source_tracking(
    out: gpd.GeoDataFrame,
    target_fid: int,
    merged_fids: Sequence[int],
    *,
    stage: str,
    row_by_fid: pd.DataFrame | None = None,
) -> None:
    if row_by_fid is None:
        row_by_fid = out.set_index("__fid")
    source_fids = _combined_source_fids(row_by_fid, [target_fid, *merged_fids])
    if "merge_source_fids" in out.columns:
        out.loc[out["__fid"].eq(target_fid), "merge_source_fids"] = source_fids
    if "merge_source_count" in out.columns:
        out.loc[out["__fid"].eq(target_fid), "merge_source_count"] = len(_coerce_source_parts(source_fids))
    if "merge_stage" in out.columns:
        out.loc[out["__fid"].eq(target_fid), "merge_stage"] = stage


def _assign_wfs_to_council_buckets(
    candidate_gdf: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if candidate_gdf.empty or council_gdf.empty:
        return pd.DataFrame(columns=["__fid", "__council_fid", "__council_inter_area", "__council_coverage"])

    council = council_gdf
    if candidate_gdf.crs and council.crs and candidate_gdf.crs != council.crs:
        council = council.to_crs(candidate_gdf.crs)
    elif council.crs is None and candidate_gdf.crs is not None:
        council = council.set_crs(candidate_gdf.crs)

    council_ref = council[[council.geometry.name]].copy().reset_index(drop=True)
    council_ref["__council_fid"] = council_ref.index.astype(int)
    sindex = council_ref.sindex
    rows = []

    for _, feature in candidate_gdf.iterrows():
        geom = feature.geometry
        if geom is None or geom.is_empty:
            continue
        area = float(geom.area)
        if area <= 0.0:
            continue
        try:
            positions = list(sindex.query(geom, predicate="intersects"))
        except TypeError:
            positions = list(sindex.query(geom))

        best = None
        for pos in positions:
            pos = int(pos)
            council_geom = council_ref.geometry.iloc[pos]
            inter_area = _safe_intersection_area(geom, council_geom)
            if inter_area <= 0.0:
                continue
            coverage = inter_area / area
            key = (coverage, inter_area, -float(council_geom.area), -pos)
            if best is None or key > best[0]:
                best = (key, pos, inter_area, coverage)

        if best is None:
            point = geom.representative_point()
            try:
                point_positions = list(sindex.query(point, predicate="intersects"))
            except TypeError:
                point_positions = list(sindex.query(point))
            for pos in point_positions:
                pos = int(pos)
                council_geom = council_ref.geometry.iloc[pos]
                if council_geom.intersects(point):
                    best = ((0.0, 0.0, -float(council_geom.area), -pos), pos, 0.0, 0.0)
                    break

        if best is None:
            continue
        _, council_pos, inter_area, coverage = best
        rows.append(
            {
                "__fid": int(feature["__fid"]),
                "__council_fid": int(council_pos),
                "__council_inter_area": float(inter_area),
                "__council_coverage": float(coverage),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["__fid", "__council_fid", "__council_inter_area", "__council_coverage"])
    return pd.DataFrame(rows).sort_values(["__fid", "__council_fid"]).drop_duplicates("__fid", keep="first")


def _select_buildings_and_lands(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    buildings = gdf.loc[gdf["__merge_role"] == "building", ["__fid", "geometry"]].copy()
    lands = gdf.loc[gdf["__merge_role"] == "land", ["__fid", "geometry"]].copy()
    return (
        buildings.rename(columns={"__fid": "building_fid"}),
        lands.rename(columns={"__fid": "land_fid"}),
    )


def _filter_merge_candidates_by_area(
    gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    if gdf.empty:
        return gdf
    is_polygon = gdf.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
    return gdf.loc[is_polygon & (gdf.geometry.area <= cfg.max_merge_area_m2)].copy()


def _edge_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    gtype = geom.geom_type
    if gtype in {"LineString", "LinearRing"}:
        return 1 if geom.length > 0 else 0
    if gtype == "MultiLineString":
        return sum(1 for g in geom.geoms if g.length > 0)
    if gtype == "GeometryCollection":
        return sum(_edge_count(g) for g in geom.geoms)
    return 0


def _convex_hull_gap(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return max(0.0, float(geom.convex_hull.area - geom.area))


def _notch_fill_metrics(small_geom, large_geom) -> dict[str, float | int | object]:
    shared = small_geom.boundary.intersection(large_geom.boundary)
    shared_len = float(shared.length)
    shared_count = int(_edge_count(shared))
    union_geom = small_geom.union(large_geom)
    small_area = float(small_geom.area)
    large_area = float(large_geom.area)
    small_perimeter = float(small_geom.length)
    hull_gap_before = _convex_hull_gap(large_geom)
    hull_gap_after = _convex_hull_gap(union_geom)
    hull_gap_improvement = hull_gap_before - hull_gap_after
    return {
        "union_geom": union_geom,
        "shared_len": shared_len,
        "shared_count": shared_count,
        "shared_ratio": shared_len / small_perimeter if small_perimeter > 0.0 else 0.0,
        "small_large_ratio": small_area / large_area if large_area > 0.0 else 0.0,
        "hull_gap_improvement": hull_gap_improvement,
        "hull_gap_improvement_ratio": hull_gap_improvement / small_area if small_area > 0.0 else 0.0,
    }


def _is_notch_fill_pair(
    small_geom,
    large_geom,
    config: WfsMergeConfig | None = None,
) -> tuple[bool, object | None, float]:
    cfg = config or WfsMergeConfig()
    if small_geom is None or large_geom is None or small_geom.is_empty or large_geom.is_empty:
        return False, None, 0.0
    if not small_geom.intersects(large_geom):
        return False, None, 0.0
    if float(small_geom.intersection(large_geom).area) > 1e-6:
        return False, None, 0.0

    metrics = _notch_fill_metrics(small_geom, large_geom)
    union_geom = metrics["union_geom"]
    if not _is_allowed_merged_geometry(union_geom, cfg):
        return False, None, 0.0
    if metrics["small_large_ratio"] > cfg.notch_fill_max_small_to_large_area_ratio:
        return False, None, 0.0
    if metrics["shared_count"] < cfg.notch_fill_min_shared_edge_count:
        return False, None, 0.0
    if metrics["shared_ratio"] < cfg.notch_fill_min_shared_small_perimeter_ratio:
        return False, None, 0.0
    if metrics["hull_gap_improvement_ratio"] < cfg.notch_fill_min_hull_gap_improvement_ratio:
        return False, None, 0.0

    score = (
        float(metrics["shared_ratio"])
        + 0.2 * float(metrics["shared_count"])
        + float(metrics["hull_gap_improvement_ratio"])
    )
    return True, union_geom, score


def _assign_lands_to_buildings(lands: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if lands.empty or buildings.empty:
        return gpd.GeoDataFrame(
            columns=["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]
        )

    candidates = gpd.sjoin(lands, buildings, how="left", predicate="intersects")
    candidates = candidates.dropna(subset=["building_fid"]).copy()
    if candidates.empty:
        return gpd.GeoDataFrame(
            columns=["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]
        )

    candidates["building_fid"] = candidates["building_fid"].astype(int)
    building_geom = buildings.set_index("building_fid")["geometry"]
    right_geoms = gpd.GeoSeries(
        candidates["building_fid"].map(building_geom).to_list(),
        index=candidates.index,
        crs=buildings.crs,
    )
    intersections = shapely.intersection(
        shapely.boundary(candidates.geometry.array),
        shapely.boundary(right_geoms.array),
    )
    candidates["shared_edge_len"] = shapely.length(intersections)
    candidates["shared_edge_count"] = shapely.get_num_geometries(intersections)
    candidates.loc[candidates["shared_edge_len"] <= 0.0, "shared_edge_count"] = 0
    candidates = candidates[candidates["shared_edge_len"] > 0].copy()
    if candidates.empty:
        return gpd.GeoDataFrame(
            columns=["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]
        )

    candidates["len_max"] = candidates.groupby("land_fid")["shared_edge_len"].transform("max")
    candidates["cnt_max"] = candidates.groupby("land_fid")["shared_edge_count"].transform("max")
    candidates["len_norm"] = candidates["shared_edge_len"] / candidates["len_max"].replace(0, 1)
    candidates["cnt_norm"] = candidates["shared_edge_count"] / candidates["cnt_max"].replace(0, 1)
    candidates["score"] = 0.5 * candidates["len_norm"] + 0.5 * candidates["cnt_norm"]

    return (
        candidates.sort_values(
            ["land_fid", "score", "shared_edge_len", "shared_edge_count", "building_fid"],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates(subset=["land_fid"], keep="first")
        [["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]]
    )


def _merge_geometries(
    gdf: gpd.GeoDataFrame,
    assigned: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> Tuple[gpd.GeoDataFrame, int]:
    cfg = config or WfsMergeConfig()
    if assigned.empty:
        return gdf.copy(), 0

    grouped_land_ids = assigned.groupby("building_fid")["land_fid"].apply(list)
    all_rows = gdf.set_index("__fid")
    all_geoms = all_rows["geometry"]
    merged_building_geoms: Dict[int, object] = {}
    merged_land_ids: Set[int] = set()
    source_by_building: Dict[int, str] = {}
    for bfid, lfids in grouped_land_ids.items():
        all_fids = [int(bfid)] + [int(lid) for lid in lfids]
        if _combined_source_count(all_rows, all_fids) > cfg.max_council_merge_wfs_count:
            continue
        geoms = [all_geoms.loc[fid] for fid in all_fids]
        merged_geom = unary_union(geoms)
        if not _is_allowed_merged_geometry(merged_geom, cfg):
            continue
        merged_building_geoms[int(bfid)] = merged_geom
        merged_land_ids.update(int(lid) for lid in lfids)
        source_by_building[int(bfid)] = _combined_source_fids(all_rows, all_fids)

    out = gdf[~gdf["__fid"].isin(merged_land_ids)].copy()
    update_mask = out["__fid"].isin(merged_building_geoms.keys())
    out.loc[update_mask, "geometry"] = out.loc[update_mask, "__fid"].map(merged_building_geoms)
    if "merge_source_fids" in out.columns:
        out.loc[update_mask, "merge_source_fids"] = out.loc[update_mask, "__fid"].map(source_by_building)
    if "merge_source_count" in out.columns:
        out.loc[update_mask, "merge_source_count"] = out.loc[update_mask, "merge_source_fids"].apply(
            lambda value: len(_coerce_source_parts(value))
        )
    if "merge_stage" in out.columns:
        out.loc[update_mask, "merge_stage"] = "edge_merge"
    return out, len(merged_building_geoms)


def _assign_independent_buildings_to_original_lands(
    lands: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    assigned_land_to_building: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if lands.empty or buildings.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    building_ids_all: Set[int] = set(int(v) for v in buildings["building_fid"].tolist())
    building_ids_updated: Set[int] = set()
    if not assigned_land_to_building.empty:
        building_ids_updated = set(int(v) for v in assigned_land_to_building["building_fid"].tolist())

    independent_building_ids = building_ids_all - building_ids_updated
    if not independent_building_ids:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    independent_buildings = buildings[buildings["building_fid"].isin(independent_building_ids)][
        ["building_fid", "geometry"]
    ].copy()
    if independent_buildings.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    candidates = gpd.sjoin(independent_buildings, lands, how="left", predicate="intersects")
    candidates = candidates.dropna(subset=["land_fid"]).copy()
    if candidates.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    candidates["land_fid"] = candidates["land_fid"].astype(int)
    land_geom = lands.set_index("land_fid")["geometry"]
    right_geoms = gpd.GeoSeries(
        candidates["land_fid"].map(land_geom).to_list(),
        index=candidates.index,
        crs=lands.crs,
    )
    intersections = shapely.intersection(
        shapely.boundary(candidates.geometry.array),
        shapely.boundary(right_geoms.array),
    )
    shared_len = shapely.length(intersections)
    candidates["shared_edge_count"] = shapely.get_num_geometries(intersections)
    candidates.loc[shared_len <= 0.0, "shared_edge_count"] = 0
    candidates = candidates[candidates["shared_edge_count"] > 0].copy()
    if candidates.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    return (
        candidates.sort_values(["building_fid", "shared_edge_count", "land_fid"], ascending=[True, False, True])
        .drop_duplicates(subset=["building_fid"], keep="first")[["building_fid", "land_fid", "shared_edge_count"]]
    )


def _merge_independent_buildings_using_original_land_targets(
    gdf_after_step1: gpd.GeoDataFrame,
    assign_building_to_land_original: gpd.GeoDataFrame,
    assigned_land_to_building_step1: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> Tuple[gpd.GeoDataFrame, int, int]:
    cfg = config or WfsMergeConfig()
    if gdf_after_step1.empty or assign_building_to_land_original.empty:
        return gdf_after_step1, 0, 0

    out = gdf_after_step1.copy()
    out_ids = set(int(v) for v in out["__fid"].tolist())
    land_to_building_step1: Dict[int, int] = {}
    if not assigned_land_to_building_step1.empty:
        land_to_building_step1 = {
            int(row["land_fid"]): int(row["building_fid"])
            for _, row in assigned_land_to_building_step1[["land_fid", "building_fid"]].iterrows()
        }

    mappings = []
    for _, row in assign_building_to_land_original.iterrows():
        building_fid = int(row["building_fid"])
        land_fid = int(row["land_fid"])
        if building_fid not in out_ids:
            continue

        target_fid = None
        if land_fid in out_ids:
            target_fid = land_fid
        elif land_fid in land_to_building_step1 and land_to_building_step1[land_fid] in out_ids:
            target_fid = land_to_building_step1[land_fid]
        if target_fid is None or target_fid == building_fid:
            continue
        mappings.append((building_fid, target_fid))

    if not mappings:
        return out, 0, 0

    map_df = pd.DataFrame(mappings, columns=["building_fid", "target_fid"]).drop_duplicates()
    grouped_buildings = map_df.groupby("target_fid")["building_fid"].apply(list)
    all_rows = out.set_index("__fid")
    all_geoms = all_rows["geometry"]

    new_target_geoms: Dict[int, object] = {}
    source_by_target: Dict[int, str] = {}
    merged_building_ids: Set[int] = set()
    for target_fid, bfids in grouped_buildings.items():
        all_fids = [int(target_fid)] + [int(b) for b in bfids]
        if _combined_source_count(all_rows, all_fids) > cfg.max_council_merge_wfs_count:
            continue
        geoms = [all_geoms.loc[fid] for fid in all_fids]
        merged_geom = unary_union(geoms)
        if not _is_allowed_merged_geometry(merged_geom, cfg):
            continue
        new_target_geoms[int(target_fid)] = merged_geom
        merged_building_ids.update(int(b) for b in bfids)
        source_by_target[int(target_fid)] = _combined_source_fids(all_rows, all_fids)

    out = out[~out["__fid"].isin(merged_building_ids)].copy()
    update_mask = out["__fid"].isin(new_target_geoms.keys())
    out.loc[update_mask, "geometry"] = out.loc[update_mask, "__fid"].map(new_target_geoms)
    if "merge_source_fids" in out.columns:
        out.loc[update_mask, "merge_source_fids"] = out.loc[update_mask, "__fid"].map(source_by_target)
    if "merge_source_count" in out.columns:
        out.loc[update_mask, "merge_source_count"] = out.loc[update_mask, "merge_source_fids"].apply(
            lambda value: len(_coerce_source_parts(value))
        )
    if "merge_stage" in out.columns:
        out.loc[update_mask, "merge_stage"] = "independent_building_merge"
    return out, len(merged_building_ids), len(new_target_geoms)


def _run_legacy_edge_merge(
    gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    if gdf.empty:
        return gdf.copy()
    buildings, lands = _select_buildings_and_lands(gdf)
    buildings = _filter_merge_candidates_by_area(buildings, cfg)
    lands = _filter_merge_candidates_by_area(lands, cfg)
    assigned = _assign_lands_to_buildings(lands, buildings)
    out, _ = _merge_geometries(gdf, assigned, cfg)
    independent = _assign_independent_buildings_to_original_lands(lands, buildings, assigned)
    out, _, _ = _merge_independent_buildings_using_original_land_targets(out, independent, assigned, cfg)
    return out


def _run_notch_fill_merge(
    gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
    max_rounds: int = 3,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    out = gdf.copy()
    geom_col = out.geometry.name
    for _ in range(max_rounds):
        if len(out) < 2:
            break

        candidate_mask = (
            out["__merge_role"].eq("building")
            & out.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
            & (out.geometry.area <= cfg.max_merge_area_m2)
        )
        candidates = out.loc[candidate_mask, ["__merge_role", geom_col]].copy()
        if len(candidates) < 2:
            break
        candidates["__area"] = candidates.geometry.area
        small_candidates = candidates.loc[candidates["__area"] <= cfg.notch_fill_max_small_area_m2].sort_values(
            ["__area"]
        )
        if small_candidates.empty:
            break
        sindex = candidates.sindex

        matches = []
        for small_idx, small_geom, small_area in zip(
            small_candidates.index,
            small_candidates.geometry,
            small_candidates["__area"],
        ):
            small_area = float(small_area)
            try:
                candidate_positions = list(sindex.query(small_geom, predicate="intersects"))
            except TypeError:
                candidate_positions = list(sindex.query(small_geom))

            for large_pos in candidate_positions:
                large_pos = int(large_pos)
                large_idx = candidates.index[large_pos]
                if large_idx == small_idx:
                    continue
                large_row = candidates.iloc[large_pos]
                large_area = float(large_row["__area"])
                if large_area < small_area:
                    continue
                if large_area == small_area and str(large_idx) < str(small_idx):
                    continue

                ok, union_geom, score = _is_notch_fill_pair(small_geom, large_row.geometry, cfg)
                if not ok:
                    continue
                matches.append(
                    {
                        "small_idx": small_idx,
                        "large_idx": large_idx,
                        "union_geom": union_geom,
                        "score": score,
                    }
                )

        if not matches:
            break

        row_by_fid = out.set_index("__fid")
        used = set()
        replacements: Dict[object, object] = {}
        removed = set()
        source_updates: Dict[object, str] = {}
        for match in sorted(matches, key=lambda item: item["score"], reverse=True):
            small_idx = match["small_idx"]
            large_idx = match["large_idx"]
            if small_idx in used or large_idx in used:
                continue
            large_fid = int(out.at[large_idx, "__fid"])
            small_fid = int(out.at[small_idx, "__fid"])
            source_update = _combined_source_fids(row_by_fid, [large_fid, small_fid])
            if len(_coerce_source_parts(source_update)) > cfg.max_council_merge_wfs_count:
                continue
            replacements[large_idx] = match["union_geom"]
            removed.add(small_idx)
            used.add(small_idx)
            used.add(large_idx)
            source_updates[large_idx] = source_update

        if not replacements:
            break

        out = out.drop(index=list(removed)).copy()
        new_geometry = out.geometry.copy()
        for idx, geom in replacements.items():
            if idx in new_geometry.index:
                new_geometry.loc[idx] = geom
                if "merge_source_fids" in out.columns and idx in out.index:
                    out.at[idx, "merge_source_fids"] = source_updates.get(idx, out.at[idx, "merge_source_fids"])
                    out.at[idx, "merge_source_count"] = len(_coerce_source_parts(out.at[idx, "merge_source_fids"]))
                    out.at[idx, "merge_stage"] = "notch_fill_merge"
        out = out.set_geometry(new_geometry)

    return out


def _council_geom_for_point(point, council_gdf: gpd.GeoDataFrame):
    try:
        positions = list(council_gdf.sindex.query(point, predicate="intersects"))
    except TypeError:
        positions = list(council_gdf.sindex.query(point))
    for pos in positions:
        geom = council_gdf.geometry.iloc[int(pos)]
        if geom.intersects(point):
            return int(pos), geom
    return None, None


def _merge_gapfill_council_features(
    gdf: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    if gdf.empty or council_gdf.empty or "GmlID" not in gdf.columns:
        return gdf

    out = _clean_tiny_polygon_parts(gdf, cfg)
    if out.empty:
        return out

    council = council_gdf
    if out.crs and council.crs and out.crs != council.crs:
        council = council.to_crs(out.crs)
    elif council.crs is None and out.crs is not None:
        council = council.set_crs(out.crs)

    gapfill_mask = out["GmlID"].fillna("").astype(str).str.startswith(GAPFILL_COUNCIL_PREFIX)
    if not bool(gapfill_mask.any()):
        return out

    gapfill_indices = out.loc[gapfill_mask].geometry.area.sort_values().index.tolist()
    target_indices = out.loc[~gapfill_mask].index.tolist()
    if not target_indices:
        return out

    target_ref = out.loc[target_indices, [out.geometry.name]].copy()
    target_sindex = target_ref.sindex
    target_rep_points = {idx: out.at[idx, out.geometry.name].representative_point() for idx in target_indices}
    removed: Set[object] = set()
    geom_col = out.geometry.name

    for gap_idx in gapfill_indices:
        if gap_idx in removed or gap_idx not in out.index:
            continue
        gap_geom = out.at[gap_idx, geom_col]
        if gap_geom is None or gap_geom.is_empty:
            removed.add(gap_idx)
            continue
        _, council_geom = _council_geom_for_point(gap_geom.representative_point(), council)
        if council_geom is None:
            continue

        try:
            positions = list(target_sindex.query(gap_geom, predicate="intersects"))
        except TypeError:
            positions = list(target_sindex.query(gap_geom))

        best_target = None
        best_geom = None
        best_score = None
        for pos in positions:
            target_idx = target_ref.index[int(pos)]
            if target_idx in removed or target_idx not in out.index:
                continue
            target_geom = out.at[target_idx, geom_col]
            if target_geom is None or target_geom.is_empty:
                continue
            target_point = target_rep_points.get(target_idx) or target_geom.representative_point()
            if not council_geom.intersects(target_point):
                continue

            shared_len = float(gap_geom.boundary.intersection(target_geom.boundary).length)
            inter_area = float(gap_geom.intersection(target_geom).area)
            if shared_len <= 0.0 and inter_area <= 1e-8:
                continue

            merged_geom = _drop_tiny_polygon_parts(
                unary_union([target_geom, gap_geom]),
                cfg.min_polygon_part_area_m2,
            )
            if not _is_allowed_merged_geometry(merged_geom, cfg):
                continue
            merged_area = float(merged_geom.area)
            if merged_area <= 0.0:
                continue
            council_coverage = float(merged_geom.intersection(council_geom).area) / merged_area
            if council_coverage < cfg.min_council_merge_wfs_coverage:
                continue
            score = (shared_len, council_coverage, -float(target_geom.area), str(target_idx))
            if best_score is None or score > best_score:
                best_score = score
                best_target = target_idx
                best_geom = merged_geom

        if best_target is None:
            continue
        row_by_fid = out.set_index("__fid")
        target_fid = int(out.at[best_target, "__fid"])
        gap_fid = int(out.at[gap_idx, "__fid"])
        if _combined_source_count(row_by_fid, [target_fid, gap_fid]) > cfg.max_council_merge_wfs_count:
            continue
        out.at[best_target, geom_col] = best_geom
        target_ref.at[best_target, geom_col] = best_geom
        target_rep_points[best_target] = best_geom.representative_point()
        _apply_source_tracking(out, target_fid, [gap_fid], stage="gapfill_council_merge", row_by_fid=row_by_fid)
        removed.add(gap_idx)

    if not removed:
        return out
    return out.drop(index=list(removed)).copy()


def _collect_interior_hole_polygons(geom, config: WfsMergeConfig) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        holes = []
        for ring in geom.interiors:
            hole = Polygon(ring)
            if hole.is_empty:
                continue
            hole = _drop_tiny_polygon_parts(hole, config.min_polygon_part_area_m2)
            if hole is None or hole.is_empty or hole.geom_type != "Polygon":
                continue
            area = float(hole.area)
            if area < config.min_polygon_part_area_m2:
                continue
            if area > config.enclosed_gap_max_area_m2:
                continue
            holes.append(hole)
        return holes
    if geom.geom_type == "MultiPolygon":
        holes = []
        for part in geom.geoms:
            holes.extend(_collect_interior_hole_polygons(part, config))
        return holes
    if geom.geom_type == "GeometryCollection":
        holes = []
        for part in geom.geoms:
            holes.extend(_collect_interior_hole_polygons(part, config))
        return holes
    return []


def _merge_enclosed_gap_holes(
    gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    if not cfg.fill_enclosed_gap_holes or len(gdf) < 2:
        return gdf

    out = gdf.copy()
    geom_col = out.geometry.name
    valid_mask = out.geometry.notna() & ~out.geometry.is_empty
    polygon_mask = valid_mask & out.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
    target_ref = out.loc[polygon_mask, [geom_col]].copy()
    if len(target_ref) < 2:
        return out

    coverage = _make_valid_geometry(unary_union(list(target_ref.geometry)))
    holes = _collect_interior_hole_polygons(coverage, cfg)
    if not holes:
        return out

    sindex = target_ref.sindex
    target_area = target_ref.geometry.area.astype(float).to_dict()
    replacements: Dict[object, object] = {}

    for hole in sorted(holes, key=lambda item: float(item.area)):
        try:
            positions = list(sindex.query(hole, predicate="intersects"))
        except TypeError:
            positions = list(sindex.query(hole))
        if not positions:
            continue

        best_idx = None
        best_score = None
        best_geom = None
        for pos in positions:
            idx = target_ref.index[int(pos)]
            target_geom = replacements.get(idx, out.at[idx, geom_col])
            if target_geom is None or target_geom.is_empty:
                continue

            shared_len = float(hole.boundary.intersection(target_geom.boundary).length)
            if shared_len < cfg.enclosed_gap_min_shared_edge_m:
                continue

            merged_geom = _drop_tiny_polygon_parts(
                unary_union([target_geom, hole]),
                cfg.min_polygon_part_area_m2,
            )
            if merged_geom is None or merged_geom.is_empty:
                continue
            if _polygon_part_count(merged_geom) > _polygon_part_count(target_geom):
                continue

            score = (shared_len, float(target_area.get(idx, 0.0)), -float(hole.area), str(idx))
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
                best_geom = merged_geom

        if best_idx is None or best_geom is None:
            continue
        replacements[best_idx] = best_geom

    if not replacements:
        return out

    new_geometry = out.geometry.copy()
    for idx, geom in replacements.items():
        if idx not in new_geometry.index:
            continue
        new_geometry.loc[idx] = geom
        if "merge_stage" in out.columns:
            out.at[idx, "merge_stage"] = "enclosed_gap_fill"
    out = out.set_geometry(new_geometry)
    return out


def _initialise_merge_metadata(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "source_fid" not in out.columns:
        out["source_fid"] = [_source_value_from_index(idx) for idx in out.index]
    out["merge_source_fids"] = out["source_fid"].apply(lambda value: "|".join(_coerce_source_parts(value)))
    out.loc[out["merge_source_fids"].eq(""), "merge_source_fids"] = out.loc[
        out["merge_source_fids"].eq(""), "source_fid"
    ].astype(str)
    out["merge_source_count"] = out["merge_source_fids"].apply(lambda value: len(_coerce_source_parts(value)))
    out["merge_stage"] = "raw"
    return out


def _run_council_bucket_merge(
    gdf: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
    config: WfsMergeConfig,
) -> tuple[gpd.GeoDataFrame, Set[int]]:
    candidate_mask = (
        gdf["__merge_role"].isin(["building", "land"])
        & gdf.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
        & (gdf.geometry.area <= config.max_merge_area_m2)
    )
    candidates = gdf.loc[candidate_mask, ["__fid", "__merge_role", "geometry"]].copy()
    assignments = _assign_wfs_to_council_buckets(candidates, council_gdf)
    if assignments.empty:
        return gdf.iloc[0:0].copy(), set()

    council = council_gdf
    if gdf.crs and council.crs and gdf.crs != council.crs:
        council = council.to_crs(gdf.crs)
    elif council.crs is None and gdf.crs is not None:
        council = council.set_crs(gdf.crs)
    council_ref = council[[council.geometry.name]].copy().reset_index(drop=True)
    council_geoms = council_ref.geometry.to_dict()

    all_rows = gdf.set_index("__fid")
    all_geoms = all_rows["geometry"]
    role_by_fid = all_rows["__merge_role"].to_dict()
    replacement_geoms: Dict[int, object] = {}
    replacement_sources: Dict[int, str] = {}
    council_merged_fids: Set[int] = set()

    for _, group in assignments.groupby("__council_fid"):
        fids = sorted(set(int(fid) for fid in group["__fid"].tolist()))
        if len(fids) <= 1:
            continue
        if len(fids) > config.max_council_merge_wfs_count:
            continue
        merged_geom = _drop_tiny_polygon_parts(
            unary_union([all_geoms.loc[fid] for fid in fids]),
            config.min_polygon_part_area_m2,
        )
        council_geom = council_geoms.get(int(group.iloc[0]["__council_fid"]))
        if not _is_allowed_council_merge(merged_geom, council_geom, config):
            continue

        building_fids = [fid for fid in fids if role_by_fid.get(fid) == "building"]
        keep_fid = min(building_fids or fids)
        replacement_geoms[keep_fid] = merged_geom
        replacement_sources[keep_fid] = _combined_source_fids(all_rows, fids)
        council_merged_fids.update(fids)

    council_out = gdf[gdf["__fid"].isin(replacement_geoms.keys())].copy()
    if not council_out.empty:
        council_out = council_out.set_geometry(list(council_out["__fid"].map(replacement_geoms)))
        council_out["merge_source_fids"] = council_out["__fid"].map(replacement_sources)
        council_out["merge_source_count"] = council_out["merge_source_fids"].apply(
            lambda value: len(_coerce_source_parts(value))
        )
        council_out["merge_stage"] = "council_bucket_merge"
    return council_out, council_merged_fids


def build_wfs_merge_gdf(
    os_wfs_gdf: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame | None = None,
    theme_field: str = "Theme",
    include_terms: Sequence[str] | None = None,
    config: WfsMergeConfig | None = None,
) -> gpd.GeoDataFrame:
    cfg = config or WfsMergeConfig()
    raw_wfs = os_wfs_gdf.copy()
    use_theme_field = resolve_theme_field(raw_wfs, preferred=theme_field)
    _validate_input(raw_wfs, use_theme_field)

    target_crs = raw_wfs.crs or (council_gdf.crs if council_gdf is not None else DEFAULT_TARGET_CRS)
    gdf = preprocess_wfs_layer(raw_wfs, target_crs=target_crs, config=cfg)
    if gdf.empty:
        return gdf

    use_theme_field = resolve_theme_field(gdf, preferred=use_theme_field)
    gdf = filter_wfs_theme_features(gdf, use_theme_field, include_terms=include_terms)
    if gdf.empty:
        return gdf

    council = None
    if council_gdf is not None and not council_gdf.empty:
        council = preprocess_council_layer(council_gdf, target_crs=gdf.crs, config=cfg)

    gdf = gdf.reset_index(drop=True)
    gdf["__fid"] = gdf.index.astype(int)
    gdf = _initialise_merge_metadata(gdf)
    gdf["__merge_role"] = _merge_role_series(gdf, use_theme_field)

    council_merged_fids: Set[int] = set()
    out_parts = []
    if council is not None and not council.empty:
        council_out, council_merged_fids = _run_council_bucket_merge(gdf, council, cfg)
        if not council_out.empty:
            out_parts.append(council_out)

    fallback_input = gdf[~gdf["__fid"].isin(council_merged_fids)].copy()
    fallback_out = _run_legacy_edge_merge(fallback_input, cfg)
    if not fallback_out.empty:
        out_parts.append(fallback_out)

    if not out_parts:
        out = gdf.iloc[0:0].copy()
    else:
        out = gpd.GeoDataFrame(pd.concat(out_parts, ignore_index=True), crs=gdf.crs)

    out = _run_notch_fill_merge(out, cfg)
    if cfg.fill_output_holes:
        out = _fill_geometry_holes(out)
    out = _clean_tiny_polygon_parts(out, cfg)
    if council is not None and not council.empty:
        out = _merge_gapfill_council_features(out, council, cfg)
    out = _merge_enclosed_gap_holes(out, cfg)
    out = _remove_nearly_covered_smaller_polygons(out, cfg)
    return out.drop(columns=["__fid", "__merge_role"], errors="ignore").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an OS WFS merge layer constrained by a council polygon layer.",
    )
    parser.add_argument("--wfs-gpkg", required=True, help="Input OS WFS polygons GPKG.")
    parser.add_argument("--wfs-layer", help="Layer name inside --wfs-gpkg. Defaults to the first layer.")
    parser.add_argument("--council-gpkg", required=True, help="Input council polygons GPKG.")
    parser.add_argument("--council-layer", help="Layer name inside --council-gpkg. Defaults to the first layer.")
    parser.add_argument(
        "--output-gpkg",
        help="Output GPKG path. Defaults to <wfs-gpkg-stem>_merge.gpkg next to the WFS input.",
    )
    parser.add_argument("--output-layer", default="os_wfs_merge", help="Output layer name.")
    parser.add_argument("--theme-field", default="Theme", help="WFS theme field name.")
    parser.add_argument(
        "--include-terms",
        default="building,land",
        help="Comma-separated Theme terms to keep before merging.",
    )
    parser.add_argument(
        "--preprocess-precision-grid",
        type=float,
        default=0.0,
        help="Optional precision grid for input topology cleanup. Use 0 to disable.",
    )
    parser.add_argument(
        "--keep-output-holes",
        action="store_true",
        help="Do not fill polygon holes in the merged output.",
    )
    parser.add_argument(
        "--disable-enclosed-gap-fill",
        action="store_true",
        help="Disable fallback that fills small enclosed gaps between multiple polygons.",
    )
    parser.add_argument(
        "--enclosed-gap-max-area",
        type=float,
        default=WfsMergeConfig.enclosed_gap_max_area_m2,
        help="Maximum area in m2 for enclosed inter-polygon gaps to fill.",
    )
    return parser.parse_args()


def _parse_include_terms(value: str) -> Sequence[str]:
    return tuple(term.strip() for term in str(value or "").split(",") if term.strip())


def main() -> None:
    args = parse_args()
    output_path = args.output_gpkg
    if not output_path:
        wfs_path = Path(args.wfs_gpkg)
        output_path = str(wfs_path.with_name(f"{wfs_path.stem}_merge.gpkg"))

    wfs_layer = _choose_layer(args.wfs_gpkg, args.wfs_layer)
    council_layer = _choose_layer(args.council_gpkg, args.council_layer)
    print(f"[INFO] Reading WFS: {args.wfs_gpkg} (layer={wfs_layer})")
    wfs_gdf = load_layer(args.wfs_gpkg, wfs_layer)
    print(f"[INFO] Reading council: {args.council_gpkg} (layer={council_layer})")
    council_gdf = load_layer(args.council_gpkg, council_layer)

    config = WfsMergeConfig(
        preprocess_precision_grid=float(args.preprocess_precision_grid or 0.0),
        fill_output_holes=not bool(args.keep_output_holes),
        fill_enclosed_gap_holes=not bool(args.disable_enclosed_gap_fill),
        enclosed_gap_max_area_m2=float(args.enclosed_gap_max_area or 0.0),
    )
    merged = build_wfs_merge_gdf(
        wfs_gdf,
        council_gdf,
        theme_field=args.theme_field,
        include_terms=_parse_include_terms(args.include_terms),
        config=config,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    merged.to_file(output, layer=args.output_layer, driver="GPKG")
    print(f"[DONE] Wrote {len(merged)} features to: {output} (layer={args.output_layer})")


if __name__ == "__main__":
    main()

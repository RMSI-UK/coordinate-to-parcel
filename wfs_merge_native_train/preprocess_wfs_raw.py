#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon

try:
    from shapely.validation import make_valid
except ImportError:  # pragma: no cover - Shapely 2 also exposes shapely.make_valid.
    make_valid = shapely.make_valid


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg"
DEFAULT_WFS_LAYER = "polygons_in_buffers"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_OUTPUT_LAYER = "wfs_raw_clean"
DEFAULT_TARGET_CRS = "EPSG:27700"
DEFAULT_INCLUDE_THEME_TERMS = "building,land,roads tracks and paths"
DEFAULT_MIN_AREA_M2 = 0.25
DEFAULT_SMALL_PATH_MAX_WIDTH_M = 4.5
DEFAULT_SMALL_PATH_MAX_AREA_M2 = 5000.0
DEFAULT_SMALL_TRACK_MAX_WIDTH_M = 6.0
DEFAULT_SMALL_TRACK_MAX_AREA_M2 = 5000.0
DEFAULT_SMALL_ROAD_MAX_WIDTH_M = 5.0
DEFAULT_SMALL_ROAD_MAX_MRR_WIDTH_M = 8.0
DEFAULT_SMALL_ROAD_MAX_AREA_M2 = 1200.0
DEFAULT_SMALL_ROADSIDE_MAX_WIDTH_M = 3.0
DEFAULT_SMALL_ROADSIDE_MAX_AREA_M2 = 800.0
DEFAULT_ENCLOSED_GAP_MAX_AREA_M2 = 250.0
DEFAULT_ENCLOSED_GAP_MIN_SURROUNDING_POLYGONS = 2
DEFAULT_ENCLOSED_GAP_MIN_SHARED_EDGE_M = 0.05
DEFAULT_POLYGON_HOLE_MAX_AREA_M2 = 250.0


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _parse_terms(value: str) -> set[str]:
    return {part.strip().lower() for part in str(value or "").split(",") if part.strip()}


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be minx,miny,maxx,maxy")
    minx, miny, maxx, maxy = parts
    if minx >= maxx or miny >= maxy:
        raise ValueError("--bbox values must satisfy minx < maxx and miny < maxy")
    return minx, miny, maxx, maxy


def _role(theme: object) -> str:
    text = str(theme or "").lower()
    if "building" in text:
        return "building"
    if "road" in text or "track" in text or "path" in text:
        return "road"
    if "land" in text:
        return "land"
    return "other"


def _role_rank(role: object) -> int:
    text = str(role or "").lower()
    if text == "building":
        return 0
    if text == "road":
        return 1
    if text == "land":
        return 2
    return 3


def _source_value_from_index(value: object) -> object:
    text = str(value)
    return int(text) if text.lstrip("-").isdigit() else value


def _safe_area(geom: object) -> float:
    if geom is None:
        return 0.0
    try:
        return float(shapely.area(geom))
    except Exception:
        try:
            return float(make_valid(geom).area)
        except Exception:
            return 0.0


def _mrr_width(geom: object) -> float:
    if geom is None:
        return 0.0
    try:
        if geom.is_empty:
            return 0.0
        mrr = geom.minimum_rotated_rectangle
        if getattr(mrr, "geom_type", "") != "Polygon":
            return 0.0
        coords = list(mrr.exterior.coords)
        if len(coords) < 5:
            return 0.0
        lengths = [
            math.hypot(coords[idx + 1][0] - coords[idx][0], coords[idx + 1][1] - coords[idx][1])
            for idx in range(4)
        ]
        return float(min(lengths))
    except Exception:
        return 0.0


def _make_valid_geometry(geom: object) -> object:
    if geom is None:
        return Polygon()
    try:
        if geom.is_empty:
            return Polygon()
        if geom.is_valid:
            return geom
    except Exception:
        pass
    try:
        fixed = make_valid(geom)
    except Exception:
        try:
            fixed = geom.buffer(0)
        except Exception:
            return Polygon()
    if fixed is None:
        return Polygon()
    try:
        if fixed.is_empty:
            return Polygon()
        if fixed.is_valid:
            return fixed
    except Exception:
        return fixed
    try:
        return fixed.buffer(0)
    except Exception:
        return fixed


def _extract_polygonal_geometry(geom: object) -> object:
    if geom is None:
        return Polygon()
    try:
        if geom.is_empty:
            return Polygon()
    except Exception:
        return Polygon()
    geom_type = getattr(geom, "geom_type", "")
    if geom_type in {"Polygon", "MultiPolygon"}:
        return geom
    if geom_type == "GeometryCollection":
        parts: list[Polygon] = []
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


def _polygon_parts(geom: object, min_area: float) -> list[Polygon]:
    geom = _extract_polygonal_geometry(_make_valid_geometry(geom))
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom] if float(geom.area) >= min_area else []
    if geom.geom_type == "MultiPolygon":
        return [part for part in geom.geoms if float(part.area) >= min_area]
    return []


def _clean_single_geometry(geom: object, min_area: float, precision_grid: float) -> object:
    cleaned = _extract_polygonal_geometry(_make_valid_geometry(geom))
    if cleaned is None or cleaned.is_empty:
        return Polygon()
    if precision_grid and precision_grid > 0.0:
        try:
            cleaned = shapely.set_precision(cleaned, float(precision_grid))
        except Exception:
            pass
    parts = _polygon_parts(cleaned, min_area=min_area)
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def _shape_metrics(geoms: Iterable[object]) -> pd.DataFrame:
    geom_array = np.asarray(list(geoms), dtype=object)
    area = pd.Series(shapely.area(geom_array), dtype="float64")
    perimeter = pd.Series(shapely.length(geom_array), dtype="float64")
    hull_area = pd.Series(shapely.area(shapely.convex_hull(geom_array)), dtype="float64")
    mrr_area = pd.Series(shapely.area(shapely.minimum_rotated_rectangle(geom_array)), dtype="float64")
    compactness = 4.0 * math.pi * area / (perimeter * perimeter).replace(0.0, 1.0)
    return pd.DataFrame(
        {
            "clean_area": area,
            "clean_perimeter": perimeter,
            "clean_mrr_ratio": area / mrr_area.replace(0.0, 1.0),
            "clean_hull_gap_ratio": (hull_area - area).clip(lower=0.0) / area.replace(0.0, 1.0),
            "clean_compactness": compactness,
        }
    )


def _refresh_clean_metrics(clean: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if clean.empty:
        return clean
    out = clean.copy().reset_index(drop=True)
    out["clean_fid"] = np.arange(len(out), dtype="int64")
    metrics = _shape_metrics(out.geometry.array)
    for column in metrics.columns:
        out[column] = metrics[column].to_numpy()
    if "shape_warning" not in out.columns:
        out["shape_warning"] = ""
    out["shape_warning"] = out["shape_warning"].fillna("").astype(str)
    return out


def _lower_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.lower()


def _road_like_mask(frame: pd.DataFrame) -> pd.Series:
    theme = _lower_column(frame, "Theme")
    group = _lower_column(frame, "DescriptiveGroup")
    term = _lower_column(frame, "DescriptiveTerm")
    text = theme + " " + group + " " + term
    return (
        theme.str.contains("roads tracks and paths", regex=False)
        | text.str.contains("road|track|path", regex=True)
    )


def _small_road_keep_mask(
    frame: pd.DataFrame,
    *,
    path_max_width: float,
    path_max_area: float,
    track_max_width: float,
    track_max_area: float,
    road_max_width: float,
    road_max_mrr_width: float,
    road_max_area: float,
    roadside_max_width: float,
    roadside_max_area: float,
) -> pd.Series:
    group = _lower_column(frame, "DescriptiveGroup")
    term = _lower_column(frame, "DescriptiveTerm")
    width = pd.to_numeric(frame.get("raw_width_proxy", 0.0), errors="coerce").fillna(0.0)
    mrr_width = pd.to_numeric(frame.get("raw_mrr_width", 0.0), errors="coerce").fillna(0.0)
    area = pd.to_numeric(frame.get("raw_part_area", frame.get("raw_area", 0.0)), errors="coerce").fillna(0.0)

    is_path = group.str.contains("path", regex=False) | term.str.contains("step|footbridge", regex=True)
    is_track = term.str.contains("track", regex=False)
    is_road_or_track = group.str.contains("road or track", regex=False)
    is_roadside = group.str.contains("roadside", regex=False)

    keep_path = is_path & width.le(float(path_max_width)) & area.le(float(path_max_area))
    keep_track = is_track & width.le(float(track_max_width)) & area.le(float(track_max_area))
    keep_small_road = (
        is_road_or_track
        & width.le(float(road_max_width))
        & mrr_width.le(float(road_max_mrr_width))
        & area.le(float(road_max_area))
    )
    keep_roadside = is_roadside & width.le(float(roadside_max_width)) & area.le(float(roadside_max_area))
    return keep_path | keep_track | keep_small_road | keep_roadside


def _coverage_union(geoms: Iterable[object]) -> object:
    geom_array = np.asarray(list(geoms), dtype=object)
    if len(geom_array) == 0:
        return Polygon()
    try:
        return shapely.coverage_union_all(geom_array)
    except Exception:
        try:
            return shapely.union_all(geom_array)
        except Exception:
            return shapely.unary_union(list(geom_array))


def _iter_polygon_geometries(geom: object) -> Iterable[Polygon]:
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        yield geom
    elif geom_type == "MultiPolygon":
        for part in geom.geoms:
            if part is not None and not part.is_empty:
                yield part
    elif geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_polygon_geometries(part)


def _enclosed_gap_candidates(
    clean: gpd.GeoDataFrame,
    *,
    min_area: float,
    max_area: float,
) -> list[Polygon]:
    if clean.empty:
        return []
    coverage = _coverage_union(clean.geometry.array)
    candidates: list[Polygon] = []
    for polygon in _iter_polygon_geometries(coverage):
        for ring in polygon.interiors:
            gap = Polygon(ring)
            if gap.is_empty:
                continue
            area = float(gap.area)
            if area < float(min_area):
                continue
            if max_area and max_area > 0.0 and area > float(max_area):
                continue
            candidates.append(gap)
    return candidates


def _gap_neighbor_stats(
    gap: Polygon,
    *,
    geoms: np.ndarray,
    tree: shapely.STRtree,
    min_shared_edge: float,
) -> tuple[int, float]:
    try:
        candidate_positions = tree.query(gap, predicate="intersects")
    except TypeError:
        candidate_positions = tree.query(gap)
    gap_boundary = shapely.boundary(gap)
    neighbor_count = 0
    shared_edge = 0.0
    for candidate_pos in candidate_positions:
        candidate = geoms[int(candidate_pos)]
        if candidate is None or candidate.is_empty:
            continue
        try:
            length = float(shapely.length(shapely.intersection(gap_boundary, shapely.boundary(candidate))))
        except Exception:
            length = float(_make_valid_geometry(gap_boundary).intersection(_make_valid_geometry(candidate).boundary).length)
        if length >= float(min_shared_edge):
            neighbor_count += 1
            shared_edge += length
    return neighbor_count, shared_edge


def _trim_gap_against_clean(
    gap: Polygon,
    *,
    geoms: np.ndarray,
    tree: shapely.STRtree,
    min_area: float,
    overlap_tolerance: float,
) -> list[Polygon]:
    try:
        candidate_positions = tree.query(gap, predicate="intersects")
    except TypeError:
        candidate_positions = tree.query(gap)

    overlap_geoms: list[object] = []
    for candidate_pos in candidate_positions:
        candidate = geoms[int(candidate_pos)]
        if candidate is None or candidate.is_empty:
            continue
        try:
            area = float(shapely.area(shapely.intersection(gap, candidate)))
        except Exception:
            area = float(_make_valid_geometry(gap).intersection(_make_valid_geometry(candidate)).area)
        if area > float(overlap_tolerance):
            overlap_geoms.append(candidate)

    if not overlap_geoms:
        return _polygon_parts(gap, min_area=float(min_area))
    trimmed = _subtract_overlap(gap, overlap_geoms, min_area=float(min_area))
    return _polygon_parts(trimmed, min_area=float(min_area))


def _polygon_hole_overlaps_other(
    hole: Polygon,
    *,
    geoms: np.ndarray,
    tree: shapely.STRtree,
    parent_pos: int,
    overlap_tolerance: float,
) -> bool:
    try:
        candidate_positions = tree.query(hole, predicate="intersects")
    except TypeError:
        candidate_positions = tree.query(hole)
    for candidate_pos in candidate_positions:
        candidate_pos = int(candidate_pos)
        if candidate_pos == int(parent_pos):
            continue
        candidate = geoms[candidate_pos]
        if candidate is None or candidate.is_empty:
            continue
        try:
            area = float(shapely.area(shapely.intersection(hole, candidate)))
        except Exception:
            area = float(_make_valid_geometry(hole).intersection(_make_valid_geometry(candidate)).area)
        if area > float(overlap_tolerance):
            return True
    return False


def _fill_polygon_internal_holes(
    clean: gpd.GeoDataFrame,
    *,
    min_area: float,
    max_area: float,
    overlap_tolerance: float,
    verbose: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    if clean.empty:
        return clean, _empty_gdf(clean.crs), {
            "polygon_hole_fill_enabled": True,
            "polygon_hole_candidate_rows": 0,
            "polygon_hole_fill_rows": 0,
            "polygon_hole_fill_area_m2": 0.0,
            "polygon_hole_already_filled_rows": 0,
        }

    geoms = np.asarray(clean.geometry.array, dtype=object)
    tree = shapely.STRtree(geoms)
    source_start = -1
    if "source_fid" in clean.columns:
        numeric_source = pd.to_numeric(clean["source_fid"], errors="coerce")
        existing_negative = numeric_source[numeric_source < 0]
        if not existing_negative.empty:
            source_start = int(existing_negative.min()) - 1

    records: list[dict[str, Any]] = []
    candidate_count = 0
    already_filled_count = 0
    geom_col = clean.geometry.name
    _log(verbose, "[INFO] Scanning polygon interior holes")
    for parent_pos, row in enumerate(clean.itertuples(index=False)):
        values = row._asdict()
        geom = values.get(geom_col)
        if geom is None or geom.is_empty:
            continue
        polygons = list(_iter_polygon_geometries(geom))
        if not polygons:
            continue
        for polygon_idx, polygon in enumerate(polygons):
            if not polygon.interiors:
                continue
            for hole_idx, ring in enumerate(polygon.interiors):
                hole = Polygon(ring)
                if hole.is_empty:
                    continue
                area = float(hole.area)
                if area < float(min_area):
                    continue
                if max_area and max_area > 0.0 and area > float(max_area):
                    continue
                candidate_count += 1
                if _polygon_hole_overlaps_other(
                    hole,
                    geoms=geoms,
                    tree=tree,
                    parent_pos=int(parent_pos),
                    overlap_tolerance=float(overlap_tolerance),
                ):
                    already_filled_count += 1
                    continue

                source_fid = int(source_start - len(records))
                rec = {column: None for column in clean.columns if column != clean.geometry.name}
                parent_source_fid = values.get("source_fid")
                parent_clean_fid = values.get("clean_fid")
                rec.update(
                    {
                        "GmlID": f"gapfill_polygon_hole_{parent_pos}_{polygon_idx}_{hole_idx}",
                        "TOID": "",
                        "Theme": "building_or_land",
                        "ThemeCount": 1,
                        "DescriptiveGroup": "building_or_land",
                        "DescriptiveGroupCount": 1,
                        "DescriptiveTerm": "polygon_hole_fill",
                        "DescriptiveTermCount": 1,
                        "Make": "Derived",
                        "source_fid": source_fid,
                        "raw_role": "building_or_land",
                        "raw_role_rank": 2,
                        "raw_area": area,
                        "source_part_id": 0,
                        "raw_part_area": area,
                        "raw_width_proxy": float(2.0 * area / (float(hole.length) or 1.0)),
                        "source_fid_sort": source_fid,
                        "preprocess_rank": None,
                        "clean_part_id": 0,
                        "clean_area": area,
                        "overlap_removed_area": 0.0,
                        "overlap_removed_ratio": 0.0,
                        "overlap_candidate_count": 0,
                        "shape_warning": "",
                        "is_polygon_hole_fill": 1,
                        "polygon_hole_fill_area": area,
                        "polygon_hole_parent_source_fid": parent_source_fid,
                        "polygon_hole_parent_clean_fid": parent_clean_fid,
                    }
                )
                rec["geometry"] = hole
                records.append(rec)

    if "is_polygon_hole_fill" not in clean.columns:
        clean = clean.copy()
        clean["is_polygon_hole_fill"] = 0
    for column in ["polygon_hole_fill_area", "polygon_hole_parent_source_fid", "polygon_hole_parent_clean_fid"]:
        if column not in clean.columns:
            clean[column] = 0

    hole_gdf = (
        gpd.GeoDataFrame(records, geometry="geometry", crs=clean.crs)
        if records
        else _empty_gdf(clean.crs)
    )
    if hole_gdf.empty:
        return clean, hole_gdf, {
            "polygon_hole_fill_enabled": True,
            "polygon_hole_candidate_rows": int(candidate_count),
            "polygon_hole_fill_rows": 0,
            "polygon_hole_fill_area_m2": 0.0,
            "polygon_hole_already_filled_rows": int(already_filled_count),
            "polygon_hole_max_area": float(max_area),
        }

    combined = gpd.GeoDataFrame(pd.concat([clean, hole_gdf], ignore_index=True), geometry="geometry", crs=clean.crs)
    combined = _refresh_clean_metrics(combined)
    return combined, hole_gdf, {
        "polygon_hole_fill_enabled": True,
        "polygon_hole_candidate_rows": int(candidate_count),
        "polygon_hole_fill_rows": int(len(hole_gdf)),
        "polygon_hole_fill_area_m2": float(hole_gdf.geometry.area.sum()),
        "polygon_hole_already_filled_rows": int(already_filled_count),
        "polygon_hole_max_area": float(max_area),
    }


def _fill_enclosed_gaps(
    clean: gpd.GeoDataFrame,
    *,
    min_area: float,
    max_area: float,
    min_surrounding_polygons: int,
    min_shared_edge: float,
    overlap_tolerance: float,
    verbose: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    if clean.empty:
        return clean, _empty_gdf(clean.crs), {
            "enclosed_gap_fill_enabled": True,
            "enclosed_gap_candidate_rows": 0,
            "enclosed_gap_fill_rows": 0,
            "enclosed_gap_fill_area_m2": 0.0,
        }

    _log(verbose, "[INFO] Building coverage union for enclosed gap fill")
    candidates = _enclosed_gap_candidates(clean, min_area=float(min_area), max_area=float(max_area))
    _log(verbose, f"[INFO] Enclosed gap candidates after area filter: {len(candidates):,}")
    if not candidates:
        return clean, _empty_gdf(clean.crs), {
            "enclosed_gap_fill_enabled": True,
            "enclosed_gap_candidate_rows": 0,
            "enclosed_gap_fill_rows": 0,
            "enclosed_gap_fill_area_m2": 0.0,
        }

    geoms = np.asarray(clean.geometry.array, dtype=object)
    tree = shapely.STRtree(geoms)
    source_start = -1
    if "source_fid" in clean.columns:
        numeric_source = pd.to_numeric(clean["source_fid"], errors="coerce")
        existing_negative = numeric_source[numeric_source < 0]
        if not existing_negative.empty:
            source_start = int(existing_negative.min()) - 1

    records: list[dict[str, Any]] = []
    trimmed_candidate_parts = 0
    for gap_idx, candidate_gap in enumerate(candidates):
        trimmed_gaps = _trim_gap_against_clean(
            candidate_gap,
            geoms=geoms,
            tree=tree,
            min_area=float(min_area),
            overlap_tolerance=float(overlap_tolerance),
        )
        trimmed_candidate_parts += len(trimmed_gaps)
        for trimmed_idx, gap in enumerate(trimmed_gaps):
            neighbor_count, shared_edge = _gap_neighbor_stats(
                gap,
                geoms=geoms,
                tree=tree,
                min_shared_edge=float(min_shared_edge),
            )
            if neighbor_count < int(min_surrounding_polygons):
                continue

            area = float(gap.area)
            rec = {column: None for column in clean.columns if column != clean.geometry.name}
            rec.update(
                {
                    "GmlID": f"gapfill_enclosed_{gap_idx}_{trimmed_idx}",
                    "TOID": "",
                    "Theme": "building_or_land",
                    "ThemeCount": 1,
                    "DescriptiveGroup": "building_or_land",
                    "DescriptiveGroupCount": 1,
                    "DescriptiveTerm": "enclosed_gap_fill",
                    "DescriptiveTermCount": 1,
                    "Make": "Derived",
                    "source_fid": int(source_start - len(records)),
                    "raw_role": "building_or_land",
                    "raw_role_rank": 2,
                    "raw_area": area,
                    "source_part_id": 0,
                    "raw_part_area": area,
                    "raw_width_proxy": float(2.0 * area / (float(gap.length) or 1.0)),
                    "source_fid_sort": int(source_start - len(records)),
                    "preprocess_rank": None,
                    "clean_part_id": 0,
                    "clean_area": area,
                    "overlap_removed_area": 0.0,
                    "overlap_removed_ratio": 0.0,
                    "overlap_candidate_count": 0,
                    "shape_warning": "",
                    "is_enclosed_gap_fill": 1,
                    "gap_fill_area": area,
                    "gap_fill_neighbor_count": int(neighbor_count),
                    "gap_fill_shared_edge_m": float(shared_edge),
                }
            )
            rec["geometry"] = gap
            records.append(rec)

    if "is_enclosed_gap_fill" not in clean.columns:
        clean = clean.copy()
        clean["is_enclosed_gap_fill"] = 0
    for column in ["gap_fill_area", "gap_fill_neighbor_count", "gap_fill_shared_edge_m"]:
        if column not in clean.columns:
            clean[column] = 0.0

    gap_gdf = (
        gpd.GeoDataFrame(records, geometry="geometry", crs=clean.crs)
        if records
        else _empty_gdf(clean.crs)
    )
    if gap_gdf.empty:
        return clean, gap_gdf, {
            "enclosed_gap_fill_enabled": True,
            "enclosed_gap_candidate_rows": int(len(candidates)),
            "enclosed_gap_fill_rows": 0,
            "enclosed_gap_fill_area_m2": 0.0,
            "enclosed_gap_trimmed_candidate_parts": int(trimmed_candidate_parts),
        }

    combined = gpd.GeoDataFrame(pd.concat([clean, gap_gdf], ignore_index=True), geometry="geometry", crs=clean.crs)
    combined = _refresh_clean_metrics(combined)
    return combined, gap_gdf, {
        "enclosed_gap_fill_enabled": True,
        "enclosed_gap_candidate_rows": int(len(candidates)),
        "enclosed_gap_fill_rows": int(len(gap_gdf)),
        "enclosed_gap_fill_area_m2": float(gap_gdf.geometry.area.sum()),
        "enclosed_gap_trimmed_candidate_parts": int(trimmed_candidate_parts),
        "enclosed_gap_max_area": float(max_area),
        "enclosed_gap_min_surrounding_polygons": int(min_surrounding_polygons),
        "enclosed_gap_min_shared_edge": float(min_shared_edge),
    }


def _write_layer(path: Path, layer: str, gdf: gpd.GeoDataFrame) -> None:
    clean = gdf.copy().reset_index(drop=True)
    for column in clean.columns:
        if column == clean.geometry.name:
            continue
        if clean[column].apply(lambda value: isinstance(value, (list, tuple, set, dict))).any():
            clean[column] = clean[column].map(
                lambda value: json.dumps(value, default=_json_default)
                if isinstance(value, (list, tuple, set, dict))
                else value
            )
    clean.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")


def _empty_gdf(crs: object | None, columns: list[str] | None = None) -> gpd.GeoDataFrame:
    cols = list(columns or [])
    if "geometry" not in cols:
        cols.append("geometry")
    return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)


def _read_raw_wfs(
    path: str,
    layer: str,
    *,
    bbox: tuple[float, float, float, float] | None,
    max_features: int,
    verbose: bool,
) -> gpd.GeoDataFrame:
    kwargs: dict[str, Any] = {"layer": layer, "engine": "pyogrio", "fid_as_index": True}
    if bbox is not None:
        kwargs["bbox"] = bbox
    if max_features and max_features > 0:
        kwargs["rows"] = int(max_features)
    _log(verbose, f"[INFO] Reading raw WFS: {path} ({layer})")
    if bbox is not None:
        _log(verbose, f"[INFO] bbox={bbox}")
    if max_features and max_features > 0:
        _log(verbose, f"[INFO] max_features={max_features:,}")
    gdf = gpd.read_file(path, **kwargs)
    gdf.index.name = "source_fid"
    return gdf


def _normalise_input(
    gdf: gpd.GeoDataFrame,
    *,
    include_terms: set[str],
    target_crs: str,
    min_area: float,
    precision_grid: float,
    drop_large_roads: bool,
    small_path_max_width: float,
    small_path_max_area: float,
    small_track_max_width: float,
    small_track_max_area: float,
    small_road_max_width: float,
    small_road_max_mrr_width: float,
    small_road_max_area: float,
    small_roadside_max_width: float,
    small_roadside_max_area: float,
    verbose: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "raw_rows": int(len(gdf)),
        "raw_crs": str(gdf.crs),
        "theme_filter_terms": sorted(include_terms),
    }
    if gdf.empty:
        return gdf.copy(), _empty_gdf(gdf.crs), summary

    work = gdf.copy()
    if work.crs is None:
        work = work.set_crs(target_crs)
    elif str(work.crs) != str(target_crs):
        work = work.to_crs(target_crs)

    if "source_fid" not in work.columns:
        work["source_fid"] = [_source_value_from_index(idx) for idx in work.index]
    numeric_source_fid = pd.to_numeric(work["source_fid"], errors="coerce")
    if numeric_source_fid.notna().all():
        work["source_fid"] = numeric_source_fid.astype("int64")

    if "Theme" not in work.columns:
        work["Theme"] = ""
    work["raw_role"] = work["Theme"].map(_role)

    if include_terms:
        theme = work["Theme"].fillna("").astype(str).str.lower()
        mask = pd.Series(False, index=work.index)
        for term in include_terms:
            mask |= theme.str.contains(term, regex=False)
        excluded_theme = work.loc[~mask].copy()
        work = work.loc[mask].copy()
    else:
        excluded_theme = work.iloc[0:0].copy()

    excluded_records: list[dict[str, Any]] = []
    for _, row in excluded_theme.iterrows():
        rec = row.drop(labels=[excluded_theme.geometry.name]).to_dict()
        rec["exclude_reason"] = "theme_not_included"
        rec["raw_area"] = _safe_area(row.geometry)
        rec["geometry"] = row.geometry
        excluded_records.append(rec)

    _log(verbose, f"[INFO] Rows after theme filter: {len(work):,}")
    if work.empty:
        excluded = gpd.GeoDataFrame(excluded_records, geometry="geometry", crs=gdf.crs)
        return work, excluded, summary

    cleaned_geoms = []
    for geom in work.geometry.array:
        cleaned_geoms.append(
            _clean_single_geometry(geom, min_area=float(min_area), precision_grid=float(precision_grid))
        )
    work = work.set_geometry(gpd.GeoSeries(cleaned_geoms, index=work.index, crs=work.crs))
    work["raw_area"] = work.geometry.area.astype(float)

    non_empty_mask = pd.Series(
        [geom is not None and not geom.is_empty for geom in work.geometry.array],
        index=work.index,
    )
    valid_mask = non_empty_mask & work["raw_area"].gt(float(min_area))
    invalid = work.loc[~valid_mask].copy()
    for _, row in invalid.iterrows():
        rec = row.drop(labels=[invalid.geometry.name]).to_dict()
        rec["exclude_reason"] = "empty_or_tiny_after_make_valid"
        rec["geometry"] = row.geometry
        excluded_records.append(rec)
    work = work.loc[valid_mask].copy()
    summary["after_geometry_clean_rows"] = int(len(work))
    summary["excluded_empty_or_tiny_rows"] = int(len(invalid))

    if work.empty:
        excluded = gpd.GeoDataFrame(excluded_records, geometry="geometry", crs=gdf.crs)
        return work, excluded, summary

    exploded_records: list[dict[str, Any]] = []
    geom_col = work.geometry.name
    for _, row in work.iterrows():
        attrs = row.drop(labels=[geom_col]).to_dict()
        for part_idx, part in enumerate(_polygon_parts(row.geometry, min_area=float(min_area))):
            rec = dict(attrs)
            rec["source_part_id"] = int(part_idx)
            rec["raw_part_area"] = float(part.area)
            rec["geometry"] = part
            exploded_records.append(rec)

    exploded = gpd.GeoDataFrame(exploded_records, geometry="geometry", crs=work.crs)
    exploded["raw_role"] = exploded["Theme"].map(_role)
    exploded["raw_width_proxy"] = (
        2.0
        * exploded.geometry.area.astype(float)
        / exploded.geometry.length.astype(float).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exploded["raw_mrr_width"] = 0.0

    summary["after_explode_rows_before_road_filter"] = int(len(exploded))
    road_like = _road_like_mask(exploded)
    if road_like.any():
        exploded.loc[road_like, "raw_mrr_width"] = [
            _mrr_width(geom) for geom in exploded.loc[road_like].geometry.array
        ]
    summary["road_like_part_rows_before_filter"] = int(road_like.sum())
    if drop_large_roads and road_like.any():
        small_road = _small_road_keep_mask(
            exploded,
            path_max_width=float(small_path_max_width),
            path_max_area=float(small_path_max_area),
            track_max_width=float(small_track_max_width),
            track_max_area=float(small_track_max_area),
            road_max_width=float(small_road_max_width),
            road_max_mrr_width=float(small_road_max_mrr_width),
            road_max_area=float(small_road_max_area),
            roadside_max_width=float(small_roadside_max_width),
            roadside_max_area=float(small_roadside_max_area),
        )
        large_road = road_like & ~small_road
        for _, row in exploded.loc[large_road].iterrows():
            rec = row.drop(labels=[exploded.geometry.name]).to_dict()
            rec["exclude_reason"] = "large_road_removed"
            rec["geometry"] = row.geometry
            excluded_records.append(rec)
        exploded = exploded.loc[~large_road].copy()
        summary["small_road_part_rows_kept"] = int((road_like & small_road).sum())
        summary["large_road_part_rows_excluded"] = int(large_road.sum())
    else:
        summary["small_road_part_rows_kept"] = int(road_like.sum())
        summary["large_road_part_rows_excluded"] = 0
    summary["drop_large_roads"] = bool(drop_large_roads)

    if exploded.empty:
        excluded = (
            gpd.GeoDataFrame(excluded_records, geometry="geometry", crs=work.crs)
            if excluded_records
            else _empty_gdf(work.crs)
        )
        summary["after_explode_rows"] = 0
        return exploded, excluded, summary

    exploded["raw_role_rank"] = exploded["raw_role"].map(_role_rank).astype(int)
    exploded["source_fid_sort"] = pd.to_numeric(exploded["source_fid"], errors="coerce").fillna(0).astype("int64")
    exploded = exploded.sort_values(
        ["raw_role_rank", "raw_part_area", "source_fid_sort", "source_part_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    exploded["preprocess_rank"] = np.arange(len(exploded), dtype="int64")
    summary["after_explode_rows"] = int(len(exploded))

    excluded = (
        gpd.GeoDataFrame(excluded_records, geometry="geometry", crs=work.crs)
        if excluded_records
        else _empty_gdf(work.crs)
    )
    summary["excluded_theme_rows"] = int(len(excluded_theme))
    return exploded, excluded, summary


def _subtract_overlap(
    geom: object,
    cover_parts: list[object],
    *,
    min_area: float,
) -> object:
    if not cover_parts:
        return geom
    try:
        cover = shapely.union_all(np.asarray(cover_parts, dtype=object))
    except Exception:
        cover = shapely.unary_union(cover_parts)
    if cover is None or cover.is_empty:
        return geom
    try:
        diff = shapely.difference(geom, cover)
    except Exception:
        diff = _make_valid_geometry(geom).difference(_make_valid_geometry(cover))
    parts = _polygon_parts(diff, min_area=min_area)
    if not parts:
        return Polygon()
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def _deoverlap(
    work: gpd.GeoDataFrame,
    *,
    min_area: float,
    overlap_tolerance: float,
    verbose: bool,
    progress_interval: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    if work.empty:
        empty = _empty_gdf(work.crs)
        return work.copy(), empty, empty, {"deoverlap_input_rows": 0, "deoverlap_output_rows": 0}

    geom_col = work.geometry.name
    input_geoms = np.asarray(work.geometry.array, dtype=object)
    tree = shapely.STRtree(input_geoms)
    accepted_by_pos: list[object | None] = [None] * len(work)
    clean_records: list[dict[str, Any]] = []
    overlap_debug_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    total_removed_area = 0.0
    changed_rows = 0
    fully_removed_rows = 0

    rows = list(work.itertuples(index=False))
    geom_pos = work.columns.get_loc(geom_col)

    for pos, row_tuple in enumerate(rows):
        if progress_interval and pos > 0 and pos % progress_interval == 0:
            _log(verbose, f"[INFO] De-overlap progress: {pos:,}/{len(work):,}")

        values = row_tuple._asdict()
        geom = values.pop(geom_col)
        raw_area = float(values.get("raw_part_area") or _safe_area(geom))

        try:
            candidate_positions = tree.query(geom, predicate="intersects")
        except TypeError:
            candidate_positions = tree.query(geom)

        cover_parts: list[object] = []
        for candidate_pos in candidate_positions:
            candidate_pos = int(candidate_pos)
            if candidate_pos >= pos:
                continue
            accepted = accepted_by_pos[candidate_pos]
            if accepted is None or accepted.is_empty:
                continue
            try:
                intersection_area = float(shapely.area(shapely.intersection(geom, accepted)))
            except Exception:
                intersection_area = float(_make_valid_geometry(geom).intersection(_make_valid_geometry(accepted)).area)
            if intersection_area > float(overlap_tolerance):
                cover_parts.append(accepted)

        remaining = _subtract_overlap(geom, cover_parts, min_area=float(min_area)) if cover_parts else geom
        remaining_area = _safe_area(remaining)
        removed_area = max(raw_area - remaining_area, 0.0)
        if removed_area > float(overlap_tolerance):
            changed_rows += 1
            total_removed_area += removed_area
            debug = dict(values)
            debug["overlap_removed_area"] = float(removed_area)
            debug["overlap_removed_ratio"] = float(removed_area / raw_area) if raw_area > 0.0 else 0.0
            debug["overlap_candidate_count"] = int(len(cover_parts))
            debug["geometry"] = geom
            overlap_debug_records.append(debug)

        parts = _polygon_parts(remaining, min_area=float(min_area))
        if not parts:
            fully_removed_rows += 1
            excluded = dict(values)
            excluded["exclude_reason"] = "fully_covered_by_higher_priority"
            excluded["raw_area"] = raw_area
            excluded["overlap_removed_area"] = float(removed_area)
            excluded["geometry"] = geom
            excluded_records.append(excluded)
            accepted_by_pos[pos] = None
            continue

        accepted_geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        accepted_by_pos[pos] = accepted_geom

        for clean_part_id, part in enumerate(parts):
            rec = dict(values)
            rec["clean_part_id"] = int(clean_part_id)
            rec["clean_area"] = float(part.area)
            rec["overlap_removed_area"] = float(removed_area)
            rec["overlap_removed_ratio"] = float(removed_area / raw_area) if raw_area > 0.0 else 0.0
            rec["overlap_candidate_count"] = int(len(cover_parts))
            rec["geometry"] = part
            clean_records.append(rec)

    clean = (
        gpd.GeoDataFrame(clean_records, geometry="geometry", crs=work.crs)
        if clean_records
        else _empty_gdf(work.crs)
    )
    if not clean.empty:
        clean = _refresh_clean_metrics(clean)

    overlap_debug = (
        gpd.GeoDataFrame(overlap_debug_records, geometry="geometry", crs=work.crs)
        if overlap_debug_records
        else _empty_gdf(work.crs)
    )
    excluded = (
        gpd.GeoDataFrame(excluded_records, geometry="geometry", crs=work.crs)
        if excluded_records
        else _empty_gdf(work.crs)
    )
    summary = {
        "deoverlap_input_rows": int(len(work)),
        "deoverlap_output_rows": int(len(clean)),
        "deoverlap_changed_rows": int(changed_rows),
        "deoverlap_fully_removed_rows": int(fully_removed_rows),
        "deoverlap_removed_area_m2": float(total_removed_area),
    }
    return clean, excluded, overlap_debug, summary


def _apply_abnormal_thresholds(
    clean: gpd.GeoDataFrame,
    *,
    max_area: float,
    min_mrr_ratio: float,
    max_hull_gap_ratio: float,
    min_compactness: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    if clean.empty:
        return clean, _empty_gdf(clean.crs), {"abnormal_excluded_rows": 0}

    warning_parts: list[pd.Series] = []
    hard_mask = pd.Series(False, index=clean.index)
    reason = pd.Series("", index=clean.index, dtype="object")

    if max_area and max_area > 0.0:
        mask = clean["clean_area"].astype(float).gt(float(max_area))
        hard_mask |= mask
        reason.loc[mask] = reason.loc[mask].map(lambda old: f"{old}|max_area" if old else "max_area")
    if min_mrr_ratio and min_mrr_ratio > 0.0:
        mask = clean["clean_mrr_ratio"].astype(float).lt(float(min_mrr_ratio))
        warning_parts.append(mask.map(lambda value: "low_mrr_ratio" if value else ""))
    if max_hull_gap_ratio and max_hull_gap_ratio > 0.0:
        mask = clean["clean_hull_gap_ratio"].astype(float).gt(float(max_hull_gap_ratio))
        warning_parts.append(mask.map(lambda value: "high_hull_gap_ratio" if value else ""))
    if min_compactness and min_compactness > 0.0:
        mask = clean["clean_compactness"].astype(float).lt(float(min_compactness))
        warning_parts.append(mask.map(lambda value: "low_compactness" if value else ""))

    if warning_parts:
        warnings = []
        for idx in clean.index:
            parts = [series.loc[idx] for series in warning_parts if series.loc[idx]]
            warnings.append("|".join(parts))
        clean = clean.copy()
        clean["shape_warning"] = warnings

    abnormal = clean.loc[hard_mask].copy()
    if not abnormal.empty:
        abnormal["exclude_reason"] = reason.loc[hard_mask].to_numpy()
    retained = clean.loc[~hard_mask].copy()
    return retained, abnormal, {"abnormal_excluded_rows": int(len(abnormal))}


def _validate_overlaps(
    clean: gpd.GeoDataFrame,
    *,
    overlap_tolerance: float,
    max_pairs: int,
    verbose: bool,
) -> dict[str, Any]:
    if clean.empty or len(clean) < 2:
        return {"validated": True, "overlap_pair_count": 0, "overlap_area_m2": 0.0, "validation_capped": False}

    geoms = np.asarray(clean.geometry.array, dtype=object)
    tree = shapely.STRtree(geoms)
    overlap_pairs = 0
    overlap_area = 0.0
    candidate_pairs = 0

    for pos, geom in enumerate(geoms):
        try:
            candidate_positions = tree.query(geom, predicate="intersects")
        except TypeError:
            candidate_positions = tree.query(geom)
        for candidate_pos in candidate_positions:
            candidate_pos = int(candidate_pos)
            if candidate_pos <= pos:
                continue
            candidate_pairs += 1
            if max_pairs and candidate_pairs > max_pairs:
                _log(verbose, f"[WARN] Overlap validation capped at {max_pairs:,} candidate pairs")
                return {
                    "validated": False,
                    "overlap_pair_count": int(overlap_pairs),
                    "overlap_area_m2": float(overlap_area),
                    "validation_capped": True,
                    "validation_candidate_pairs": int(candidate_pairs),
                }
            try:
                area = float(shapely.area(shapely.intersection(geom, geoms[candidate_pos])))
            except Exception:
                area = float(_make_valid_geometry(geom).intersection(_make_valid_geometry(geoms[candidate_pos])).area)
            if area > float(overlap_tolerance):
                overlap_pairs += 1
                overlap_area += area

    return {
        "validated": True,
        "overlap_pair_count": int(overlap_pairs),
        "overlap_area_m2": float(overlap_area),
        "validation_capped": False,
        "validation_candidate_pairs": int(candidate_pairs),
    }


def preprocess_wfs_raw(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output_gpkg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace it: {output_path}")
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    raw = _read_raw_wfs(
        args.wfs_gpkg,
        args.wfs_layer,
        bbox=_parse_bbox(args.bbox),
        max_features=int(args.max_features),
        verbose=bool(args.verbose),
    )
    include_terms = _parse_terms(args.include_theme_terms)
    work, excluded_initial, summary = _normalise_input(
        raw,
        include_terms=include_terms,
        target_crs=str(args.target_crs),
        min_area=float(args.min_area),
        precision_grid=float(args.precision_grid),
        drop_large_roads=bool(args.drop_large_roads),
        small_path_max_width=float(args.small_path_max_width),
        small_path_max_area=float(args.small_path_max_area),
        small_track_max_width=float(args.small_track_max_width),
        small_track_max_area=float(args.small_track_max_area),
        small_road_max_width=float(args.small_road_max_width),
        small_road_max_mrr_width=float(args.small_road_max_mrr_width),
        small_road_max_area=float(args.small_road_max_area),
        small_roadside_max_width=float(args.small_roadside_max_width),
        small_roadside_max_area=float(args.small_roadside_max_area),
        verbose=bool(args.verbose),
    )

    if args.skip_deoverlap:
        clean = work.copy()
        if not clean.empty:
            clean["clean_part_id"] = 0
            clean["clean_fid"] = np.arange(len(clean), dtype="int64")
            metrics = _shape_metrics(clean.geometry.array)
            for column in metrics.columns:
                clean[column] = metrics[column].to_numpy()
            clean["overlap_removed_area"] = 0.0
            clean["overlap_removed_ratio"] = 0.0
            clean["overlap_candidate_count"] = 0
            clean["shape_warning"] = ""
        excluded_deoverlap = _empty_gdf(work.crs)
        overlap_debug = _empty_gdf(work.crs)
        summary.update({"deoverlap_skipped": True})
    else:
        clean, excluded_deoverlap, overlap_debug, deoverlap_summary = _deoverlap(
            work,
            min_area=float(args.min_area),
            overlap_tolerance=float(args.overlap_tolerance),
            verbose=bool(args.verbose),
            progress_interval=int(args.progress_interval),
        )
        summary.update(deoverlap_summary)
        summary["deoverlap_skipped"] = False

    clean, abnormal_excluded, abnormal_summary = _apply_abnormal_thresholds(
        clean,
        max_area=float(args.max_area),
        min_mrr_ratio=float(args.min_mrr_ratio),
        max_hull_gap_ratio=float(args.max_hull_gap_ratio),
        min_compactness=float(args.min_compactness),
    )
    summary.update(abnormal_summary)

    if args.skip_polygon_hole_fill:
        polygon_hole_fill = _empty_gdf(clean.crs)
        summary.update(
            {
                "polygon_hole_fill_enabled": False,
                "polygon_hole_candidate_rows": 0,
                "polygon_hole_fill_rows": 0,
                "polygon_hole_fill_area_m2": 0.0,
                "polygon_hole_already_filled_rows": 0,
            }
        )
    else:
        clean, polygon_hole_fill, polygon_hole_summary = _fill_polygon_internal_holes(
            clean,
            min_area=float(args.polygon_hole_min_area),
            max_area=float(args.polygon_hole_max_area),
            overlap_tolerance=float(args.overlap_tolerance),
            verbose=bool(args.verbose),
        )
        summary.update(polygon_hole_summary)

    if args.skip_enclosed_gap_fill:
        gap_fill = _empty_gdf(clean.crs)
        summary.update(
            {
                "enclosed_gap_fill_enabled": False,
                "enclosed_gap_candidate_rows": 0,
                "enclosed_gap_fill_rows": 0,
                "enclosed_gap_fill_area_m2": 0.0,
            }
        )
    else:
        clean, gap_fill, gap_summary = _fill_enclosed_gaps(
            clean,
            min_area=float(args.enclosed_gap_min_area),
            max_area=float(args.enclosed_gap_max_area),
            min_surrounding_polygons=int(args.enclosed_gap_min_surrounding_polygons),
            min_shared_edge=float(args.enclosed_gap_min_shared_edge),
            overlap_tolerance=float(args.overlap_tolerance),
            verbose=bool(args.verbose),
        )
        summary.update(gap_summary)

    excluded_layers = [gdf for gdf in [excluded_initial, excluded_deoverlap, abnormal_excluded] if not gdf.empty]
    excluded = (
        gpd.GeoDataFrame(pd.concat(excluded_layers, ignore_index=True), geometry="geometry", crs=clean.crs)
        if excluded_layers
        else _empty_gdf(clean.crs)
    )

    if args.validate_overlaps:
        summary.update(
            _validate_overlaps(
                clean,
                overlap_tolerance=float(args.overlap_tolerance),
                max_pairs=int(args.max_validation_pairs),
                verbose=bool(args.verbose),
            )
        )
    else:
        summary.update({"validated": False})

    summary.update(
        {
            "wfs_gpkg": str(args.wfs_gpkg),
            "wfs_layer": str(args.wfs_layer),
            "output_gpkg": str(output_path),
            "output_layer": str(args.output_layer),
            "clean_rows": int(len(clean)),
            "excluded_rows": int(len(excluded)),
            "overlap_debug_rows": int(len(overlap_debug)),
            "min_area": float(args.min_area),
            "overlap_tolerance": float(args.overlap_tolerance),
            "precision_grid": float(args.precision_grid),
            "skip_deoverlap": bool(args.skip_deoverlap),
            "debug_layers_written": bool(args.write_debug_layers),
            "small_path_max_width": float(args.small_path_max_width),
            "small_path_max_area": float(args.small_path_max_area),
            "small_track_max_width": float(args.small_track_max_width),
            "small_track_max_area": float(args.small_track_max_area),
            "small_road_max_width": float(args.small_road_max_width),
            "small_road_max_mrr_width": float(args.small_road_max_mrr_width),
            "small_road_max_area": float(args.small_road_max_area),
            "small_roadside_max_width": float(args.small_roadside_max_width),
            "small_roadside_max_area": float(args.small_roadside_max_area),
            "skip_polygon_hole_fill": bool(args.skip_polygon_hole_fill),
            "polygon_hole_min_area": float(args.polygon_hole_min_area),
            "polygon_hole_max_area": float(args.polygon_hole_max_area),
            "skip_enclosed_gap_fill": bool(args.skip_enclosed_gap_fill),
            "enclosed_gap_min_area": float(args.enclosed_gap_min_area),
            "enclosed_gap_max_area": float(args.enclosed_gap_max_area),
            "enclosed_gap_min_surrounding_polygons": int(args.enclosed_gap_min_surrounding_polygons),
            "enclosed_gap_min_shared_edge": float(args.enclosed_gap_min_shared_edge),
        }
    )

    _log(args.verbose, f"[INFO] Writing clean layer: {output_path} ({args.output_layer}) rows={len(clean):,}")
    _write_layer(output_path, str(args.output_layer), clean)
    if args.write_debug_layers:
        _log(args.verbose, f"[INFO] Writing excluded layer: rows={len(excluded):,}")
        _write_layer(output_path, "wfs_raw_preprocess_excluded", excluded)
        _log(args.verbose, f"[INFO] Writing overlap debug layer: rows={len(overlap_debug):,}")
        _write_layer(output_path, "wfs_raw_preprocess_overlap_debug", overlap_debug)
        _log(args.verbose, f"[INFO] Writing polygon hole fill layer: rows={len(polygon_hole_fill):,}")
        _write_layer(output_path, "wfs_raw_preprocess_polygon_hole_fill", polygon_hole_fill)
        _log(args.verbose, f"[INFO] Writing enclosed gap fill layer: rows={len(gap_fill):,}")
        _write_layer(output_path, "wfs_raw_preprocess_enclosed_gap_fill", gap_fill)
    else:
        _log(args.verbose, "[INFO] Debug layers not written; production output is one clean layer.")

    summary_path = output_path.with_suffix(".preprocess_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _log(args.verbose, f"[DONE] Summary: {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw WFS polygons into a clean non-overlapping base map.")
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument("--target-crs", default=DEFAULT_TARGET_CRS)
    parser.add_argument("--include-theme-terms", default=DEFAULT_INCLUDE_THEME_TERMS)
    parser.add_argument("--bbox", default="", help="Optional minx,miny,maxx,maxy read filter.")
    parser.add_argument("--max-features", type=int, default=0, help="Optional row cap for smoke tests; 0 means all.")
    parser.add_argument(
        "--min-area",
        type=float,
        default=DEFAULT_MIN_AREA_M2,
        help="Drop polygon parts below this area in square metres.",
    )
    parser.add_argument("--precision-grid", type=float, default=0.0, help="Optional Shapely precision grid; 0 disables.")
    parser.add_argument(
        "--keep-large-roads",
        dest="drop_large_roads",
        action="store_false",
        help="Keep all road-themed polygons instead of filtering to small roads.",
    )
    parser.set_defaults(drop_large_roads=True)
    parser.add_argument("--small-path-max-width", type=float, default=DEFAULT_SMALL_PATH_MAX_WIDTH_M)
    parser.add_argument("--small-path-max-area", type=float, default=DEFAULT_SMALL_PATH_MAX_AREA_M2)
    parser.add_argument("--small-track-max-width", type=float, default=DEFAULT_SMALL_TRACK_MAX_WIDTH_M)
    parser.add_argument("--small-track-max-area", type=float, default=DEFAULT_SMALL_TRACK_MAX_AREA_M2)
    parser.add_argument("--small-road-max-width", type=float, default=DEFAULT_SMALL_ROAD_MAX_WIDTH_M)
    parser.add_argument("--small-road-max-mrr-width", type=float, default=DEFAULT_SMALL_ROAD_MAX_MRR_WIDTH_M)
    parser.add_argument("--small-road-max-area", type=float, default=DEFAULT_SMALL_ROAD_MAX_AREA_M2)
    parser.add_argument("--small-roadside-max-width", type=float, default=DEFAULT_SMALL_ROADSIDE_MAX_WIDTH_M)
    parser.add_argument("--small-roadside-max-area", type=float, default=DEFAULT_SMALL_ROADSIDE_MAX_AREA_M2)
    parser.add_argument("--skip-polygon-hole-fill", action="store_true")
    parser.add_argument("--polygon-hole-min-area", type=float, default=DEFAULT_MIN_AREA_M2)
    parser.add_argument("--polygon-hole-max-area", type=float, default=DEFAULT_POLYGON_HOLE_MAX_AREA_M2)
    parser.add_argument("--skip-enclosed-gap-fill", action="store_true")
    parser.add_argument("--enclosed-gap-min-area", type=float, default=DEFAULT_MIN_AREA_M2)
    parser.add_argument("--enclosed-gap-max-area", type=float, default=DEFAULT_ENCLOSED_GAP_MAX_AREA_M2)
    parser.add_argument(
        "--enclosed-gap-min-surrounding-polygons",
        type=int,
        default=DEFAULT_ENCLOSED_GAP_MIN_SURROUNDING_POLYGONS,
    )
    parser.add_argument("--enclosed-gap-min-shared-edge", type=float, default=DEFAULT_ENCLOSED_GAP_MIN_SHARED_EDGE_M)
    parser.add_argument("--overlap-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-deoverlap", action="store_true")
    parser.add_argument("--max-area", type=float, default=0.0, help="Optional hard abnormal area cutoff; 0 disables.")
    parser.add_argument("--min-mrr-ratio", type=float, default=0.0, help="Optional shape warning threshold; 0 disables.")
    parser.add_argument("--max-hull-gap-ratio", type=float, default=0.0, help="Optional shape warning threshold; 0 disables.")
    parser.add_argument("--min-compactness", type=float, default=0.0, help="Optional shape warning threshold; 0 disables.")
    parser.add_argument("--validate-overlaps", action="store_true")
    parser.add_argument("--max-validation-pairs", type=int, default=2_000_000)
    parser.add_argument("--write-debug-layers", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = preprocess_wfs_raw(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

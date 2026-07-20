#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid


TARGET_CRS = "EPSG:27700"

DEFAULT_INPUT_GPKG = ""
DEFAULT_GEOCODING_GPKG = "/data/sheffield/spatial/sheffield-wp3/sheffield_wp3_edit_geocoding.gpkg"
DEFAULT_GEOCODING_LAYER = "sheffield_wp3_edit_geocoding"
DEFAULT_NATIVE_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_merge_native.gpkg"
DEFAULT_NATIVE_LAYER = "predicted_parcels_with_uprn"
DEFAULT_POLYGON_LAYER = "geocoding_point_wfs_polygons"
DEFAULT_POINT_LAYER = "geocoding_point_wfs_match_points"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/sheffield-wp3/sheffield_wp3_oachargeid_single_wfs_polygons.gpkg"
DEFAULT_FALLBACK_SQUARE_SIZE = 8.0
DEFAULT_PROFILE_JOB_ROOT = "/data/file-browser-data/spatial-jobs"

SELECTED_LAYER = "oachargeid_single_polygons"
CANDIDATE_LAYER = "oachargeid_candidate_polygons"
ASSIGNMENT_LAYER = "oachargeid_point_candidate_assignments"
POINT_POLYGON_OUTPUT_LAYER = "geocoding_point_wfs_polygons"
POINT_MATCH_OUTPUT_LAYER = "geocoding_point_wfs_match_points"

ORIGINAL_SUMMARY_FIELDS = [
    "oachargeid_sub",
    "charge_geographic_description",
    "supplementary_information",
    "address_plotsize",
    "address_clarity",
    "address_premise",
    "address_road",
    "address_locality",
    "address_number",
    "postcode",
    "geocoding_confidence",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _clean_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _csv_unique(values: pd.Series | list[Any], *, limit: int | None = None) -> str:
    out: list[str] = []
    iterable = values.tolist() if isinstance(values, pd.Series) else values
    for value in iterable:
        text = _text(value)
        if not text:
            continue
        if text not in out:
            out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return ",".join(out)


def _pipe_unique(values: pd.Series | list[Any], *, limit: int | None = None) -> str:
    out: list[str] = []
    iterable = values.tolist() if isinstance(values, pd.Series) else values
    for value in iterable:
        text = _text(value)
        if not text:
            continue
        if text not in out:
            out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return " | ".join(out)


def _read_polygons(path: str, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    if "oachargeid_key" not in gdf.columns:
        gdf["oachargeid_key"] = gdf["oachargeid"].map(_clean_key)
    if "oachargeid_sub_key" not in gdf.columns:
        gdf["oachargeid_sub_key"] = gdf["oachargeid_sub"].map(_clean_key) if "oachargeid_sub" in gdf.columns else ""
    if "point_fid" not in gdf.columns:
        raise ValueError("Input polygon layer must contain point_fid")
    gdf["point_fid"] = pd.to_numeric(gdf["point_fid"], errors="coerce").astype("Int64")
    return gdf


def _child_fallback_square_mask(gdf: gpd.GeoDataFrame) -> pd.Series:
    parent_key = gdf["oachargeid_key"].fillna("").astype(str).str.strip()
    child_key = gdf["oachargeid_sub_key"].fillna("").astype(str).str.strip()
    source = gdf.get("polygon_source", pd.Series("", index=gdf.index)).fillna("").astype(str)
    return child_key.ne("") & parent_key.ne("") & child_key.ne(parent_key) & source.eq("fallback_8m_square")


def _read_points(path: str, layer: str) -> gpd.GeoDataFrame:
    points = gpd.read_file(path, layer=layer, engine="pyogrio")
    if points.crs is None:
        points = points.set_crs(TARGET_CRS)
    elif str(points.crs).upper() != TARGET_CRS:
        points = points.to_crs(TARGET_CRS)
    if "point_fid" not in points.columns:
        raise ValueError("Input point layer must contain point_fid")
    points["point_fid"] = pd.to_numeric(points["point_fid"], errors="coerce").astype("Int64")
    return points


def _read_geocoding_points(path: str, layer: str) -> gpd.GeoDataFrame:
    points = gpd.read_file(path, layer=layer, engine="pyogrio", fid_as_index=True)
    if points.crs is None:
        points = points.set_crs(TARGET_CRS)
    elif str(points.crs).upper() != TARGET_CRS:
        points = points.to_crs(TARGET_CRS)
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if "point_fid" not in points.columns:
        points["point_fid"] = points.index.astype(int)
    points["point_fid"] = pd.to_numeric(points["point_fid"], errors="coerce").astype("Int64")
    if "oachargeid_key" not in points.columns:
        points["oachargeid_key"] = points["oachargeid"].map(_clean_key)
    if "oachargeid_sub_key" not in points.columns:
        points["oachargeid_sub_key"] = (
            points["oachargeid_sub"].map(_clean_key) if "oachargeid_sub" in points.columns else ""
        )
    return points


def _expanded_bounds(gdf: gpd.GeoDataFrame, expand: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = [float(value) for value in gdf.total_bounds]
    return (minx - float(expand), miny - float(expand), maxx + float(expand), maxy + float(expand))


def _read_native_parcels(path: str, layer: str, points: gpd.GeoDataFrame, *, fallback_square_size: float) -> gpd.GeoDataFrame:
    bbox = _expanded_bounds(points, max(float(fallback_square_size), 1.0))
    native = gpd.read_file(path, layer=layer, engine="pyogrio", bbox=bbox, fid_as_index=True)
    if native.crs is None:
        native = native.set_crs(TARGET_CRS)
    elif str(native.crs).upper() != TARGET_CRS:
        native = native.to_crs(TARGET_CRS)
    native = native[native.geometry.notna() & ~native.geometry.is_empty].copy()
    native["native_fid"] = native.index.astype(int)
    return native


def _fallback_square(point_geom, size: float):
    half = float(size) / 2.0
    return box(float(point_geom.x) - half, float(point_geom.y) - half, float(point_geom.x) + half, float(point_geom.y) + half)


def _native_sort_columns(joined: gpd.GeoDataFrame) -> tuple[list[str], list[bool]]:
    joined["_has_uprn_sort"] = pd.to_numeric(joined.get("pred_uprn_count", 0), errors="coerce").fillna(0).gt(0).astype(int)
    joined["_one_uprn_sort"] = pd.to_numeric(joined.get("pred_uprn_count", 0), errors="coerce").fillna(0).eq(1).astype(int)
    joined["_regularity_sort"] = pd.to_numeric(joined.get("pred_regularity_score", 0), errors="coerce").fillna(0.0)
    joined["_area_sort"] = pd.to_numeric(joined.get("pred_area", joined.geometry.area), errors="coerce").fillna(joined.geometry.area)
    return (
        ["point_fid", "_one_uprn_sort", "_has_uprn_sort", "_regularity_sort", "_area_sort", "native_fid"],
        [True, False, False, False, True, True],
    )


def _build_point_level_from_geocoding_native(
    geocoding_points: gpd.GeoDataFrame,
    native: gpd.GeoDataFrame,
    *,
    fallback_square_size: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    point_attrs = geocoding_points.copy()
    point_geoms = point_attrs[["point_fid", "geometry"]].copy()
    native_attrs = native.copy()
    native_attrs = native_attrs.rename_geometry("native_geometry")

    joined = gpd.sjoin(
        point_geoms,
        native_attrs,
        how="left",
        predicate="intersects",
    )
    raw_match_count = (
        joined[joined["native_fid"].notna()]
        .groupby("point_fid")
        .size()
        .rename("intersect_match_count")
    )
    sort_cols, ascending = _native_sort_columns(joined)
    joined = joined.sort_values(sort_cols, ascending=ascending)
    best = joined.drop_duplicates("point_fid", keep="first").copy()
    best = best.set_index(best["point_fid"].astype(int), drop=False)
    native_by_fid = native.set_index(native["native_fid"].astype(int), drop=False)

    records: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    native_metric_defaults = {
        "source_count": pd.NA,
        "pred_uprn_count": pd.NA,
        "area": pd.NA,
        "perimeter": pd.NA,
        "mrr_ratio": pd.NA,
        "hull_gap_ratio": pd.NA,
        "regularity_score": pd.NA,
    }

    for _, point in point_attrs.iterrows():
        point_fid = int(point["point_fid"])
        point_geom = point.geometry
        matched = point_fid in best.index and pd.notna(best.loc[point_fid].get("native_fid"))
        rec = {column: point.get(column) for column in point_attrs.columns if column != point_attrs.geometry.name}
        rec["point_fid"] = point_fid
        rec["oachargeid_key"] = _clean_key(rec.get("oachargeid"))
        rec["oachargeid_sub_key"] = _clean_key(rec.get("oachargeid_sub"))
        rec["center_source"] = "point_geometry"
        rec["has_center"] = True
        rec["intersect_match_count"] = int(raw_match_count.get(point_fid, 0))

        if matched:
            row = best.loc[point_fid]
            native_fid = int(row["native_fid"])
            native_row = native_by_fid.loc[native_fid]
            rec["land_road"] = "no"
            rec["wfs_fid"] = str(native_fid)
            rec["hybrid_id"] = str(native_fid)
            rec["hybrid_source"] = "native"
            rec["source_native_fids"] = str(native_fid)
            rec["source_council_fid"] = -1
            rec["patch_reason"] = ""
            rec["native_count"] = 1
            rec["source_count"] = _num(native_row.get("source_count"), 0.0)
            rec["pred_uprn_count"] = _num(native_row.get("pred_uprn_count"), 0.0)
            rec["area"] = _num(native_row.get("pred_area"), float(native_row.geometry.area))
            rec["perimeter"] = _num(native_row.get("pred_perimeter"), float(native_row.geometry.length))
            rec["mrr_ratio"] = _num(native_row.get("pred_mrr_ratio"), 0.0)
            rec["hull_gap_ratio"] = _num(native_row.get("pred_hull_gap_ratio"), 0.0)
            rec["regularity_score"] = _num(native_row.get("pred_regularity_score"), 0.0)
            rec["polygon_source"] = "wfs_merge_native"
            rec["geometry"] = native_row.geometry
        else:
            rec["land_road"] = "yes"
            rec["wfs_fid"] = pd.NA
            rec["hybrid_id"] = ""
            rec["hybrid_source"] = ""
            rec["source_native_fids"] = ""
            rec["source_council_fid"] = pd.NA
            rec["patch_reason"] = "fallback_no_native_intersection"
            rec["native_count"] = pd.NA
            rec.update(native_metric_defaults)
            rec["area"] = float(fallback_square_size) ** 2
            rec["regularity_score"] = 1.0
            rec["polygon_source"] = "fallback_8m_square"
            rec["geometry"] = _fallback_square(point_geom, float(fallback_square_size))

        records.append(rec)
        point_row = dict(rec)
        point_row["geometry"] = point_geom
        point_records.append(point_row)

    polygons = gpd.GeoDataFrame(records, geometry="geometry", crs=TARGET_CRS)
    points = gpd.GeoDataFrame(point_records, geometry="geometry", crs=TARGET_CRS)
    summary = {
        "direct_input_points": int(len(point_attrs)),
        "native_rows_in_point_bounds": int(len(native)),
        "matched_points": int(polygons["polygon_source"].astype(str).eq("wfs_merge_native").sum()),
        "fallback_square_points": int(polygons["polygon_source"].astype(str).eq("fallback_8m_square").sum()),
        "fallback_square_size_m": float(fallback_square_size),
    }
    return polygons, points, summary


def _make_valid_polygonal(geom):
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return None
    try:
        if not geom.is_valid:
            geom = make_valid(geom)
    except Exception:
        geom = geom.buffer(0)
    return geom


def _polygon_parts(geom) -> list[Any]:
    geom = _make_valid_polygonal(geom)
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return []
    geom_type = str(getattr(geom, "geom_type", ""))
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return [part for part in geom.geoms if not part.is_empty]
    if geom_type == "GeometryCollection":
        parts = []
        for child in geom.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _safe_union(geoms: list[Any]):
    valid = []
    for geom in geoms:
        valid.extend(_polygon_parts(geom))
    if not valid:
        return None
    try:
        merged = unary_union(valid)
    except Exception:
        merged = unary_union([_make_valid_polygonal(geom) for geom in valid])
    return _make_valid_polygonal(merged)


def _point_in_part(point_geom, part, *, eps: float) -> bool:
    if point_geom is None or bool(getattr(point_geom, "is_empty", True)):
        return False
    try:
        return bool(part.covers(point_geom) or part.distance(point_geom) <= eps)
    except Exception:
        valid_part = _make_valid_polygonal(part)
        return bool(valid_part is not None and (valid_part.covers(point_geom) or valid_part.distance(point_geom) <= eps))


def _row_geom_intersects_part(row_geom, part, *, eps: float) -> bool:
    if row_geom is None or bool(getattr(row_geom, "is_empty", True)):
        return False
    try:
        if row_geom.intersects(part):
            inter = row_geom.intersection(part)
            if float(getattr(inter, "area", 0.0) or 0.0) > eps:
                return True
            return bool(row_geom.representative_point().distance(part) <= eps)
        return False
    except Exception:
        return False


def _candidate_metrics(
    oachargeid_key: str,
    candidate_id: int,
    candidate_count: int,
    part,
    group: gpd.GeoDataFrame,
    point_by_fid: dict[int, Any],
    *,
    eps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assigned_indexes = []
    for idx, row in group.iterrows():
        point_fid = int(row["point_fid"])
        point_geom = point_by_fid.get(point_fid)
        if _point_in_part(point_geom, part, eps=eps):
            assigned_indexes.append(idx)

    if not assigned_indexes:
        for idx, row in group.iterrows():
            if _row_geom_intersects_part(row.geometry, part, eps=eps):
                assigned_indexes.append(idx)

    assigned = group.loc[assigned_indexes].drop_duplicates("point_fid").copy()
    point_count = int(len(assigned))
    parity = assigned.get("address_range_parity", pd.Series("", index=assigned.index)).fillna("").astype(str).str.strip().str.lower()
    same_count = int(parity.eq("same").sum())
    opposite_count = int(parity.eq("opposite").sum())
    known_count = int(parity.isin(["same", "opposite"]).sum())
    blank_count = int(point_count - known_count)
    same_ratio = float(same_count / max(point_count, 1))
    same_ratio_known = float(same_count / max(known_count, 1)) if known_count else 0.0
    similarity = pd.to_numeric(assigned.get("api_address_similarity", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    similarity_mean = float(similarity.mean()) if point_count else 0.0
    similarity_max = float(similarity.max()) if point_count else 0.0
    land_road_yes = int(assigned.get("land_road", pd.Series("", index=assigned.index)).fillna("").astype(str).eq("yes").sum())

    record = {
        "oachargeid": oachargeid_key,
        "oachargeid_key": oachargeid_key,
        "candidate_id": int(candidate_id),
        "candidate_count": int(candidate_count),
        "selected": 0,
        "point_count": point_count,
        "same_count": same_count,
        "opposite_count": opposite_count,
        "blank_parity_count": blank_count,
        "same_ratio": same_ratio,
        "same_ratio_known": same_ratio_known,
        "address_similarity_mean": similarity_mean,
        "address_similarity_max": similarity_max,
        "land_road_yes_count": land_road_yes,
        "land_road": "yes" if point_count > 0 and land_road_yes == point_count else "no",
        "polygon_sources": _csv_unique(assigned.get("polygon_source", pd.Series([], dtype=object))),
        "wfs_fids": _csv_unique(assigned.get("wfs_fid", pd.Series([], dtype=object)), limit=100),
        "hybrid_sources": _csv_unique(assigned.get("hybrid_source", pd.Series([], dtype=object))),
        "center_sources": _csv_unique(assigned.get("center_source", pd.Series([], dtype=object))),
        "point_fids": _csv_unique(assigned.get("point_fid", pd.Series([], dtype=object)), limit=200),
        "oachargeid_subs": _pipe_unique(assigned.get("oachargeid_sub", pd.Series([], dtype=object)), limit=30),
        "area_m2": float(part.area),
        "geometry": part,
    }
    for field in ORIGINAL_SUMMARY_FIELDS:
        if field in assigned.columns:
            record[field] = _pipe_unique(assigned[field], limit=10)
        else:
            record[field] = ""

    assignments = []
    for _, row in assigned.iterrows():
        point_fid = int(row["point_fid"])
        point_geom = point_by_fid.get(point_fid)
        assignments.append(
            {
                "oachargeid": oachargeid_key,
                "candidate_id": int(candidate_id),
                "selected": 0,
                "point_fid": point_fid,
                "oachargeid_sub": _text(row.get("oachargeid_sub")),
                "address_range_parity": _text(row.get("address_range_parity")).lower(),
                "api_address_similarity": _num(row.get("api_address_similarity")),
                "land_road": _text(row.get("land_road")),
                "polygon_source": _text(row.get("polygon_source")),
                "geometry": point_geom,
            }
        )
    return record, assignments


def _build_parent_candidates(
    group: gpd.GeoDataFrame,
    point_by_fid: dict[int, Any],
    *,
    eps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    geoms = [geom for geom in group.geometry if geom is not None and not bool(getattr(geom, "is_empty", True))]
    merged = _safe_union(geoms)
    parts = sorted(_polygon_parts(merged), key=lambda geom: float(geom.area), reverse=True)
    oachargeid_key = str(group["oachargeid_key"].iloc[0])
    if not parts:
        return [], []
    candidate_count = len(parts)
    candidates = []
    assignments = []
    for candidate_id, part in enumerate(parts, start=1):
        record, rows = _candidate_metrics(
            oachargeid_key,
            candidate_id,
            candidate_count,
            part,
            group,
            point_by_fid,
            eps=eps,
        )
        candidates.append(record)
        assignments.extend(rows)
    return candidates, assignments


def _select_candidates(candidates: gpd.GeoDataFrame, assignments: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if candidates.empty:
        return candidates, assignments
    ranked = candidates.copy()
    ranked = ranked.sort_values(
        [
            "oachargeid_key",
            "same_ratio",
            "address_similarity_mean",
            "point_count",
            "area_m2",
            "candidate_id",
        ],
        ascending=[True, False, False, False, False, True],
    )
    ranked["selection_rank"] = ranked.groupby("oachargeid_key", sort=False).cumcount() + 1
    selected_keys = set(
        zip(
            ranked.loc[ranked["selection_rank"].eq(1), "oachargeid_key"],
            ranked.loc[ranked["selection_rank"].eq(1), "candidate_id"].astype(int),
        )
    )
    ranked["selected"] = [
        1 if (row.oachargeid_key, int(row.candidate_id)) in selected_keys else 0
        for row in ranked.itertuples(index=False)
    ]
    selected = ranked[ranked["selected"].eq(1)].copy()
    selected["selection_rule"] = "same_ratio_desc,address_similarity_mean_desc,point_count_desc,area_desc"

    if not assignments.empty:
        assignments = assignments.copy()
        assignments["selected"] = [
            1 if (row.oachargeid, int(row.candidate_id)) in selected_keys else 0
            for row in assignments.itertuples(index=False)
        ]
    return selected, ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one selected polygon per oachargeid directly from geocoding points "
            "and WFS merge native parcels, with a legacy point-level input mode."
        )
    )
    parser.add_argument(
        "--input-gpkg",
        default=DEFAULT_INPUT_GPKG,
        help="Legacy point-level polygon GPKG. If omitted, direct geocoding + native mode is used.",
    )
    parser.add_argument("--area-profile", default="", help="Optional AreaProfile JSON used to fill native defaults.")
    parser.add_argument("--polygon-layer", default=DEFAULT_POLYGON_LAYER)
    parser.add_argument("--point-layer", default=DEFAULT_POINT_LAYER)
    parser.add_argument("--geocoding-gpkg", default=DEFAULT_GEOCODING_GPKG)
    parser.add_argument("--geocoding-layer", default=DEFAULT_GEOCODING_LAYER)
    parser.add_argument("--native-gpkg", default=DEFAULT_NATIVE_GPKG)
    parser.add_argument("--native-layer", default=DEFAULT_NATIVE_LAYER)
    parser.add_argument("--fallback-square-size", type=float, default=DEFAULT_FALLBACK_SQUARE_SIZE)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--selected-layer", default=SELECTED_LAYER)
    parser.add_argument("--candidate-layer", default=CANDIDATE_LAYER)
    parser.add_argument("--assignment-layer", default=ASSIGNMENT_LAYER)
    parser.add_argument("--point-polygon-output-layer", default=POINT_POLYGON_OUTPUT_LAYER)
    parser.add_argument("--point-match-output-layer", default=POINT_MATCH_OUTPUT_LAYER)
    parser.add_argument("--skip-point-level-output", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--drop-child-fallback-squares", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_area_profile(path: str) -> Any:
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from spatial_pipeline.area_profile import load_area_profile

    return load_area_profile(path)


def _default_profile_output_root(area_key: str) -> Path:
    root = Path(os.environ.get("SPATIAL_PIPELINE_JOB_ROOT", DEFAULT_PROFILE_JOB_ROOT))
    return root / "manual" / str(area_key or "area")


def _apply_area_profile_defaults(args: argparse.Namespace) -> None:
    if not str(args.area_profile or "").strip():
        return
    profile = _load_area_profile(str(args.area_profile))
    root = _default_profile_output_root(profile.area_key)
    if args.native_gpkg == DEFAULT_NATIVE_GPKG:
        args.native_gpkg = str(root / "wfs_merge_native" / "wfs_raw_merged_native.gpkg")
    if args.native_layer == DEFAULT_NATIVE_LAYER:
        args.native_layer = profile.native_merge_layer
    if args.output_gpkg == DEFAULT_OUTPUT_GPKG:
        args.output_gpkg = str(root / "auto_polygon" / "oachargeid_single_polygons.gpkg")


def main() -> None:
    args = parse_args()
    _apply_area_profile_defaults(args)
    output_path = Path(args.output_gpkg)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direct_summary: dict[str, Any] = {}
    if str(args.input_gpkg or "").strip():
        input_mode = "legacy_point_level_gpkg"
        _log(f"[INFO] Reading point-level polygons: {args.input_gpkg}::{args.polygon_layer}")
        polygons = _read_polygons(args.input_gpkg, args.polygon_layer)
        _log(f"[INFO] Reading point centres: {args.input_gpkg}::{args.point_layer}")
        points = _read_points(args.input_gpkg, args.point_layer)
    else:
        input_mode = "direct_geocoding_native"
        _log(f"[INFO] Reading geocoding points: {args.geocoding_gpkg}::{args.geocoding_layer}")
        geocoding_points = _read_geocoding_points(args.geocoding_gpkg, args.geocoding_layer)
        _log(f"[INFO] geocoding points={len(geocoding_points)}")
        _log(f"[INFO] Reading native parcels in point bounds: {args.native_gpkg}::{args.native_layer}")
        native = _read_native_parcels(
            args.native_gpkg,
            args.native_layer,
            geocoding_points,
            fallback_square_size=float(args.fallback_square_size),
        )
        _log(f"[INFO] native parcel rows in bounds={len(native)}")
        polygons, points, direct_summary = _build_point_level_from_geocoding_native(
            geocoding_points,
            native,
            fallback_square_size=float(args.fallback_square_size),
        )
        _log(
            "[INFO] direct point-level matches="
            f"{direct_summary['matched_points']} fallback={direct_summary['fallback_square_points']}"
        )

    polygons = polygons[polygons["oachargeid_key"].fillna("").astype(str).str.strip().ne("")].copy()
    polygons = polygons[polygons.geometry.notna()].copy()
    polygons = polygons[~polygons.geometry.is_empty].copy()
    source_input_polygon_rows = int(len(polygons))
    source_input_oachargeids = int(polygons["oachargeid_key"].nunique())
    dropped_child_fallback_squares = 0
    dropped_child_fallback_oachargeids = 0
    if args.drop_child_fallback_squares:
        drop_mask = _child_fallback_square_mask(polygons)
        dropped_child_fallback_squares = int(drop_mask.sum())
        dropped_child_fallback_oachargeids = int(polygons.loc[drop_mask, "oachargeid_key"].nunique())
        if dropped_child_fallback_squares:
            polygons = polygons.loc[~drop_mask].copy()
        _log(
            "[INFO] dropped child fallback 8m squares="
            f"{dropped_child_fallback_squares} affected_oachargeids={dropped_child_fallback_oachargeids}"
        )
    _log(f"[INFO] polygon rows={len(polygons)} oachargeids={polygons['oachargeid_key'].nunique()}")

    point_by_fid = {
        int(row["point_fid"]): row.geometry
        for _, row in points.iterrows()
        if row.geometry is not None and not bool(getattr(row.geometry, "is_empty", True)) and not pd.isna(row["point_fid"])
    }
    _log(f"[INFO] point centres={len(point_by_fid)}")

    candidate_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for _, group in polygons.groupby("oachargeid_key", sort=False):
        candidates, assignments = _build_parent_candidates(group.copy(), point_by_fid, eps=float(args.eps))
        candidate_rows.extend(candidates)
        assignment_rows.extend(assignments)

    candidates = (
        gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs=TARGET_CRS)
        if candidate_rows
        else gpd.GeoDataFrame({"oachargeid_key": pd.Series(dtype=object)}, geometry=gpd.GeoSeries([], crs=TARGET_CRS))
    )
    assignments = (
        gpd.GeoDataFrame(assignment_rows, geometry="geometry", crs=TARGET_CRS)
        if assignment_rows
        else gpd.GeoDataFrame({"oachargeid": pd.Series(dtype=object)}, geometry=gpd.GeoSeries([], crs=TARGET_CRS))
    )
    selected, candidates_ranked = _select_candidates(candidates, assignments)

    if not assignments.empty and not candidates_ranked.empty:
        selected_keys = set(zip(selected["oachargeid_key"], selected["candidate_id"].astype(int)))
        assignments["selected"] = [
            1 if (row.oachargeid, int(row.candidate_id)) in selected_keys else 0
            for row in assignments.itertuples(index=False)
        ]

    _log(
        "[INFO] candidates="
        f"{len(candidates_ranked)} selected={len(selected)} "
        f"multi_candidate_oachargeids={int(candidates_ranked.groupby('oachargeid_key').size().gt(1).sum()) if not candidates_ranked.empty else 0}"
    )
    _log(f"[INFO] selected unique oachargeids={selected['oachargeid_key'].nunique() if not selected.empty else 0}")
    if not selected.empty:
        _log("[INFO] selected candidate_count distribution:")
        _log(selected["candidate_count"].value_counts(dropna=False).sort_index().to_string())

    _log(f"[INFO] Writing {output_path}")
    selected.to_file(output_path, layer=args.selected_layer, driver="GPKG", engine="pyogrio")
    candidates_ranked.to_file(output_path, layer=args.candidate_layer, driver="GPKG", engine="pyogrio")
    assignments.to_file(output_path, layer=args.assignment_layer, driver="GPKG", engine="pyogrio")
    if input_mode == "direct_geocoding_native" and not bool(args.skip_point_level_output):
        polygons.to_file(output_path, layer=args.point_polygon_output_layer, driver="GPKG", engine="pyogrio")
        points.to_file(output_path, layer=args.point_match_output_layer, driver="GPKG", engine="pyogrio")

    summary = {
        "input_mode": input_mode,
        "input_gpkg": str(args.input_gpkg or "") or None,
        "geocoding_gpkg": str(args.geocoding_gpkg) if input_mode == "direct_geocoding_native" else None,
        "geocoding_layer": str(args.geocoding_layer) if input_mode == "direct_geocoding_native" else None,
        "native_gpkg": str(args.native_gpkg) if input_mode == "direct_geocoding_native" else None,
        "native_layer": str(args.native_layer) if input_mode == "direct_geocoding_native" else None,
        "source_input_polygon_rows": source_input_polygon_rows,
        "source_input_oachargeids": source_input_oachargeids,
        "filtered_input_polygon_rows": int(len(polygons)),
        "filtered_input_oachargeids": int(polygons["oachargeid_key"].nunique()),
        "dropped_child_fallback_squares": dropped_child_fallback_squares,
        "dropped_child_fallback_oachargeids": dropped_child_fallback_oachargeids,
        "candidate_rows": int(len(candidates_ranked)),
        "selected_rows": int(len(selected)),
        "selected_unique_oachargeids": int(selected["oachargeid_key"].nunique()) if not selected.empty else 0,
        "multi_candidate_oachargeids": int(candidates_ranked.groupby("oachargeid_key").size().gt(1).sum()) if not candidates_ranked.empty else 0,
        "all_oachargeids_single_selected": bool(len(selected) == polygons["oachargeid_key"].nunique() == selected["oachargeid_key"].nunique()),
        "selection_rule": "same_ratio_desc,address_similarity_mean_desc,point_count_desc,area_desc",
        "selected_layer": args.selected_layer,
        "candidate_layer": args.candidate_layer,
        "assignment_layer": args.assignment_layer,
        "point_polygon_output_layer": None
        if input_mode != "direct_geocoding_native" or bool(args.skip_point_level_output)
        else args.point_polygon_output_layer,
        "point_match_output_layer": None
        if input_mode != "direct_geocoding_native" or bool(args.skip_point_level_output)
        else args.point_match_output_layer,
        **direct_summary,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log("[DONE] Aggregated to one polygon per oachargeid.")


if __name__ == "__main__":
    main()

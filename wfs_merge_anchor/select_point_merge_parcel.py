#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import MultiPolygon, Polygon


DEFAULT_POINTS_LAYER = "stable_geocoding_points"
DEFAULT_CLEAN_WFS_LAYER = "wfs_raw_clean"
DEFAULT_ANCHOR_LAYER = "wfs_raw_clean_anchor"
DEFAULT_COUNCIL_LAYER = "council_polygons_single_anchor_area05"
DEFAULT_FALLBACK_LAYER = "council_polygons_single_anchor_fallback"
DEFAULT_OUTPUT_LAYER = "oachargeid_single_polygons"
TARGET_CRS = "EPSG:27700"


def _log(message: str) -> None:
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


def _read_points(path: Path, layer: str) -> gpd.GeoDataFrame:
    points = pyogrio.read_dataframe(path, layer=layer, fid_as_index=True)
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    points.index.name = "point_fid"
    points = points.reset_index()
    if points.crs is None:
        points = points.set_crs(TARGET_CRS)
    elif str(points.crs).upper() != TARGET_CRS:
        points = points.to_crs(TARGET_CRS)
    if "oachargeid" not in points.columns:
        raise ValueError(f"{path}:{layer} must contain oachargeid")
    if "oachargeid_sub" not in points.columns:
        points["oachargeid_sub"] = points["oachargeid"]
    return points


def _read_anchors(path: Path, layer: str) -> gpd.GeoDataFrame:
    anchors = pyogrio.read_dataframe(
        path,
        layer=layer,
        columns=["clean_fid", "source_fid", "Theme", "anchor_uprn_count"],
        fid_as_index=True,
    )
    anchors = anchors[anchors.geometry.notna() & ~anchors.geometry.is_empty].copy()
    anchors.index.name = "anchor_fid"
    anchors = anchors.reset_index()
    anchors = anchors.rename(columns={"clean_fid": "anchor_clean_fid", "source_fid": "anchor_source_fid"})
    if anchors.empty:
        return anchors
    anchors["anchor_fid"] = anchors["anchor_fid"].astype("int64")
    anchors["anchor_clean_fid"] = anchors["anchor_clean_fid"].astype("int64")
    anchors["anchor_area"] = anchors.geometry.area.astype(float)
    return anchors


def _read_optional_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=TARGET_CRS), crs=TARGET_CRS)
    try:
        frame = pyogrio.read_dataframe(path, layer=layer)
    except Exception:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=TARGET_CRS), crs=TARGET_CRS)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.crs is None:
        frame = frame.set_crs(TARGET_CRS)
    elif str(frame.crs).upper() != TARGET_CRS:
        frame = frame.to_crs(TARGET_CRS)
    if "anchor_clean_fid" in frame.columns:
        frame["anchor_clean_fid"] = frame["anchor_clean_fid"].astype("int64")
    return frame


def _read_clean_wfs(path_value: str, layer: str) -> gpd.GeoDataFrame:
    if not str(path_value or "").strip():
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=TARGET_CRS), crs=TARGET_CRS)
    path = Path(path_value)
    if not path.exists():
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=TARGET_CRS), crs=TARGET_CRS)
    try:
        frame = pyogrio.read_dataframe(path, layer=layer, fid_as_index=True)
    except Exception:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=TARGET_CRS), crs=TARGET_CRS)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame.index.name = "clean_wfs_fid"
    frame = frame.reset_index()
    if frame.crs is None:
        frame = frame.set_crs(TARGET_CRS)
    elif str(frame.crs).upper() != TARGET_CRS:
        frame = frame.to_crs(TARGET_CRS)
    if "clean_fid" not in frame.columns:
        frame["clean_fid"] = frame["clean_wfs_fid"]
    frame["clean_fid"] = frame["clean_fid"].astype("int64")
    frame["clean_wfs_fid"] = frame["clean_wfs_fid"].astype("int64")
    frame["clean_wfs_area"] = frame.geometry.area.astype(float)
    return frame


def _select_anchor(point: Any, anchors: gpd.GeoDataFrame, max_nearest_distance: float) -> tuple[pd.Series | None, str, float]:
    if anchors.empty:
        return None, "no_anchor", 0.0
    intersects = anchors[anchors.geometry.intersects(point)].copy()
    if not intersects.empty:
        intersects["_distance"] = 0.0
        selected = intersects.sort_values(
            ["_distance", "anchor_area", "anchor_uprn_count", "anchor_clean_fid"],
            ascending=[True, True, False, True],
        ).iloc[0]
        return selected, "intersects", 0.0

    distances = anchors.geometry.distance(point).astype(float)
    nearest_idx = distances.idxmin()
    nearest_distance = float(distances.loc[nearest_idx])
    if max_nearest_distance >= 0.0 and nearest_distance > float(max_nearest_distance):
        return None, "nearest_anchor_too_far", nearest_distance
    selected = anchors.loc[nearest_idx].copy()
    return selected, "nearest", nearest_distance


def _select_clean_wfs_intersection(point: Any, clean_wfs: gpd.GeoDataFrame) -> tuple[pd.Series | None, int]:
    if clean_wfs.empty:
        return None, 0
    hits = clean_wfs[clean_wfs.geometry.intersects(point)].copy()
    if hits.empty:
        return None, 0
    selected = hits.sort_values(["clean_wfs_area", "clean_fid", "clean_wfs_fid"], ascending=[True, True, True]).iloc[0]
    return selected, int(len(hits))


def _fallback_rows_for_clean_fid(fallback: gpd.GeoDataFrame, clean_fid: int) -> gpd.GeoDataFrame:
    if fallback.empty or "anchor_clean_fid" not in fallback.columns:
        return fallback.iloc[0:0].copy()
    rows = fallback[fallback["anchor_clean_fid"].astype("int64").eq(int(clean_fid))].copy()
    sort_columns = [
        column
        for column in ("fallback_area", "area_m2", "anchor_fid")
        if column in rows.columns
    ]
    if sort_columns:
        rows = rows.sort_values(sort_columns)
    return rows


def _anchor_meta_for_clean_fid(anchors: gpd.GeoDataFrame, clean_fid: int) -> tuple[int, str, float]:
    if anchors.empty or "anchor_clean_fid" not in anchors.columns:
        return -1, "point_intersecting_raw_clean", 0.0
    rows = anchors[anchors["anchor_clean_fid"].astype("int64").eq(int(clean_fid))]
    if rows.empty:
        return -1, "point_intersecting_raw_clean", 0.0
    row = rows.iloc[0]
    return int(row.get("anchor_fid", -1)), "point_intersecting_raw_clean", 0.0


def _usable_council_wfs_mask(rows: gpd.GeoDataFrame) -> pd.Series:
    mask = pd.Series(False, index=rows.index)
    if rows.empty:
        return mask
    if "wfs_iou_match_found" in rows.columns:
        values = rows["wfs_iou_match_found"]
        numeric = pd.to_numeric(values, errors="coerce").fillna(0)
        text = values.astype(str).str.strip().str.lower()
        mask = mask | numeric.eq(1) | text.isin({"true", "t", "yes", "y"})
    if "output_geometry_source" in rows.columns:
        mask = mask | rows["output_geometry_source"].astype(str).str.strip().str.lower().eq("wfs_clean_iou_match")
    return mask.fillna(False)


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if part is not None and not part.is_empty]
    if geom_type == "GeometryCollection":
        parts: list[Polygon] = []
        for part in geometry.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def _fill_polygon_holes(geometry: Any) -> Any:
    parts = _polygon_parts(geometry)
    if not parts:
        return geometry
    filled_parts = [Polygon(part.exterior) for part in parts]
    filled = filled_parts[0] if len(filled_parts) == 1 else MultiPolygon(filled_parts)
    if not bool(shapely.is_valid(filled)):
        filled = shapely.make_valid(filled)
        valid_parts = _polygon_parts(filled)
        if valid_parts:
            filled = valid_parts[0] if len(valid_parts) == 1 else MultiPolygon(valid_parts)
    return filled


def _empty_output(crs: object | None = TARGET_CRS) -> gpd.GeoDataFrame:
    columns = {
        "oachargeid": pd.Series(dtype=object),
        "oachargeid_key": pd.Series(dtype=object),
        "candidate_id": pd.Series(dtype="int64"),
        "candidate_count": pd.Series(dtype="int64"),
        "selected": pd.Series(dtype="int64"),
        "point_count": pd.Series(dtype="int64"),
        "polygon_sources": pd.Series(dtype=object),
        "wfs_fids": pd.Series(dtype=object),
        "point_fids": pd.Series(dtype=object),
        "oachargeid_subs": pd.Series(dtype=object),
        "area_m2": pd.Series(dtype="float64"),
        "oachargeid_sub": pd.Series(dtype=object),
        "selection_rank": pd.Series(dtype="int64"),
        "selection_rule": pd.Series(dtype=object),
        "anchor_clean_fid": pd.Series(dtype="int64"),
        "anchor_fid": pd.Series(dtype="int64"),
        "anchor_match_method": pd.Series(dtype=object),
        "anchor_distance_m": pd.Series(dtype="float64"),
        "council_point_hit_count": pd.Series(dtype="int64"),
        "council_usable_wfs_hit_count": pd.Series(dtype="int64"),
        "raw_clean_fid": pd.Series(dtype="int64"),
        "raw_clean_point_hit_count": pd.Series(dtype="int64"),
        "fallback_used": pd.Series(dtype="int64"),
    }
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def select_point_merge_parcels(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    points_path = Path(args.points_gpkg)
    anchors_path = Path(args.anchor_gpkg)
    council_path = Path(args.council_single_anchor_gpkg)
    fallback_path = Path(args.fallback_gpkg)
    clean_wfs_path_value = str(getattr(args, "clean_wfs_gpkg", "") or "").strip()

    _log(f"[INFO] Reading points: {points_path}:{args.points_layer}")
    points = _read_points(points_path, str(args.points_layer))
    _log(f"[INFO] Reading clean WFS fallback: {clean_wfs_path_value}:{args.clean_wfs_layer}")
    clean_wfs = _read_clean_wfs(clean_wfs_path_value, str(args.clean_wfs_layer))
    _log(f"[INFO] Reading anchors: {anchors_path}:{args.anchor_layer}")
    anchors = _read_anchors(anchors_path, str(args.anchor_layer))
    _log(f"[INFO] Reading council candidates: {council_path}:{args.council_single_anchor_layer}")
    council = _read_optional_layer(council_path, str(args.council_single_anchor_layer))
    _log(f"[INFO] Reading fallback candidates: {fallback_path}:{args.fallback_layer}")
    fallback = _read_optional_layer(fallback_path, str(args.fallback_layer))

    records: list[dict[str, Any]] = []
    no_anchor = 0
    no_council_hit = 0
    multiple_council_hits = 0
    no_fallback = 0
    council_selected = 0
    fallback_selected = 0
    raw_clean_selected = 0
    no_raw_clean_hit = 0
    no_usable_council_wfs_match = 0
    output_miss_replaced_by_point_fallback = 0
    output_miss_replaced_by_point_raw_clean = 0
    output_miss_no_raw_clean_hit = 0

    for _, point_row in points.iterrows():
        point = point_row.geometry
        oachargeid = str(point_row.get("oachargeid") or "").strip()
        oachargeid_sub = str(point_row.get("oachargeid_sub") or oachargeid).strip()
        anchor, anchor_method, anchor_distance = _select_anchor(
            point,
            anchors,
            max_nearest_distance=float(args.max_nearest_anchor_distance),
        )
        if anchor is None:
            no_anchor += 1
            selected_clean, clean_hit_count = _select_clean_wfs_intersection(point, clean_wfs)
            if selected_clean is None:
                no_raw_clean_hit += 1
                continue
            geometry = selected_clean.geometry
            if geometry is None or getattr(geometry, "is_empty", False):
                no_raw_clean_hit += 1
                continue
            geometry = _fill_polygon_holes(geometry)
            clean_fid = int(selected_clean.clean_fid)
            raw_clean_selected += 1
            records.append(
                {
                    "oachargeid": oachargeid,
                    "oachargeid_key": oachargeid,
                    "candidate_id": 1,
                    "candidate_count": 1,
                    "selected": 1,
                    "point_count": 1,
                    "polygon_sources": "wfs_raw_clean_no_anchor",
                    "wfs_fids": str(clean_fid),
                    "point_fids": str(int(point_row.get("point_fid", 0))),
                    "oachargeid_subs": oachargeid_sub,
                    "area_m2": float(geometry.area),
                    "oachargeid_sub": oachargeid_sub,
                    "selection_rank": 1,
                    "selection_rule": "raw_clean_polygon_intersecting_point_no_anchor",
                    "anchor_clean_fid": -1,
                    "anchor_fid": -1,
                    "anchor_match_method": anchor_method,
                    "anchor_distance_m": float(anchor_distance),
                    "council_point_hit_count": 0,
                    "council_usable_wfs_hit_count": 0,
                    "raw_clean_fid": clean_fid,
                    "raw_clean_point_hit_count": clean_hit_count,
                    "fallback_used": 0,
                    "geometry": geometry,
                }
            )
            continue

        anchor_clean_fid = int(anchor.anchor_clean_fid)
        anchor_fid = int(anchor.anchor_fid)
        council_hits = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=points.crs), crs=points.crs)
        if "anchor_clean_fid" in council.columns:
            council_rows = council[council["anchor_clean_fid"].astype("int64").eq(anchor_clean_fid)].copy()
            if not council_rows.empty:
                council_hits = council_rows[council_rows.geometry.intersects(point)].copy()

        source = ""
        selection_rule = ""
        geometry = None
        council_hit_count = int(len(council_hits))
        usable_council_hits = council_hits[_usable_council_wfs_mask(council_hits)].copy()
        usable_council_hit_count = int(len(usable_council_hits))
        fallback_used = 0

        if council_hit_count == 1 and usable_council_hit_count == 1:
            selected = usable_council_hits.iloc[0]
            geometry = selected.geometry
            source = "wfs_merge_anchor_council"
            selection_rule = "single_wfs_matched_council_polygon_containing_point"
            council_selected += 1
        else:
            if council_hit_count == 0:
                no_council_hit += 1
                selection_rule = "fallback_no_council_polygon_containing_point"
            elif council_hit_count > 1:
                multiple_council_hits += 1
                selection_rule = "fallback_multiple_council_polygons_containing_point"
            else:
                no_usable_council_wfs_match += 1
                selection_rule = "fallback_no_usable_council_wfs_match"
            fallback_rows = _fallback_rows_for_clean_fid(fallback, anchor_clean_fid)
            if fallback_rows.empty:
                no_fallback += 1
                continue
            selected = fallback_rows.iloc[0]
            geometry = selected.geometry
            source = "wfs_merge_anchor_fallback"
            fallback_used = 1
            fallback_selected += 1

        if geometry is None or getattr(geometry, "is_empty", False):
            no_fallback += 1
            continue
        geometry = _fill_polygon_holes(geometry)
        raw_clean_fid = anchor_clean_fid
        raw_clean_point_hit_count = 0

        try:
            output_intersects_point = bool(geometry.intersects(point))
        except Exception:
            output_intersects_point = False
        if not output_intersects_point:
            selected_clean, clean_hit_count = _select_clean_wfs_intersection(point, clean_wfs)
            if selected_clean is None:
                output_miss_no_raw_clean_hit += 1
            else:
                if source == "wfs_merge_anchor_council":
                    council_selected = max(0, council_selected - 1)
                elif source == "wfs_merge_anchor_fallback":
                    fallback_selected = max(0, fallback_selected - 1)
                clean_fid = int(selected_clean.clean_fid)
                raw_clean_fid = clean_fid
                raw_clean_point_hit_count = clean_hit_count
                anchor_clean_fid = clean_fid
                anchor_fid, anchor_method, anchor_distance = _anchor_meta_for_clean_fid(anchors, clean_fid)
                fallback_rows = _fallback_rows_for_clean_fid(fallback, clean_fid)
                if not fallback_rows.empty:
                    selected = fallback_rows.iloc[0]
                    replacement_geometry = selected.geometry
                    replacement_intersects_point = False
                    if replacement_geometry is not None and not getattr(replacement_geometry, "is_empty", False):
                        replacement_geometry = _fill_polygon_holes(replacement_geometry)
                        try:
                            replacement_intersects_point = bool(replacement_geometry.intersects(point))
                        except Exception:
                            replacement_intersects_point = False
                    if replacement_intersects_point:
                        geometry = replacement_geometry
                        source = "wfs_merge_anchor_fallback"
                        selection_rule = "fallback_point_intersecting_raw_clean_after_output_miss"
                        fallback_used = 1
                        fallback_selected += 1
                        output_miss_replaced_by_point_fallback += 1
                    else:
                        geometry = _fill_polygon_holes(selected_clean.geometry)
                        source = "wfs_raw_clean_output_miss"
                        selection_rule = "raw_clean_polygon_intersecting_point_after_output_miss_bad_fallback"
                        fallback_used = 0
                        raw_clean_selected += 1
                        output_miss_replaced_by_point_raw_clean += 1
                else:
                    geometry = _fill_polygon_holes(selected_clean.geometry)
                    source = "wfs_raw_clean_output_miss"
                    selection_rule = "raw_clean_polygon_intersecting_point_after_output_miss_no_fallback"
                    fallback_used = 0
                    raw_clean_selected += 1
                    output_miss_replaced_by_point_raw_clean += 1

        records.append(
            {
                "oachargeid": oachargeid,
                "oachargeid_key": oachargeid,
                "candidate_id": 1,
                "candidate_count": 1,
                "selected": 1,
                "point_count": 1,
                "polygon_sources": source,
                "wfs_fids": str(anchor_clean_fid),
                "point_fids": str(int(point_row.get("point_fid", 0))),
                "oachargeid_subs": oachargeid_sub,
                "area_m2": float(geometry.area),
                "oachargeid_sub": oachargeid_sub,
                "selection_rank": 1,
                "selection_rule": selection_rule,
                "anchor_clean_fid": anchor_clean_fid,
                "anchor_fid": anchor_fid,
                "anchor_match_method": anchor_method,
                "anchor_distance_m": float(anchor_distance),
                "council_point_hit_count": council_hit_count,
                "council_usable_wfs_hit_count": usable_council_hit_count,
                "raw_clean_fid": raw_clean_fid,
                "raw_clean_point_hit_count": raw_clean_point_hit_count,
                "fallback_used": fallback_used,
                "geometry": geometry,
            }
        )

    if records:
        out = gpd.GeoDataFrame(records, geometry="geometry", crs=points.crs)
        out = out[out["oachargeid"].fillna("").astype(str).str.strip().ne("")].copy()
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
        out = out.sort_values(["oachargeid", "oachargeid_sub", "selection_rank"]).drop_duplicates(
            ["oachargeid", "oachargeid_sub"],
            keep="first",
        )
    else:
        out = _empty_output(points.crs)

    summary = {
        "points_gpkg": str(points_path),
        "points_layer": str(args.points_layer),
        "clean_wfs_gpkg": clean_wfs_path_value,
        "clean_wfs_layer": str(args.clean_wfs_layer),
        "anchor_gpkg": str(anchors_path),
        "anchor_layer": str(args.anchor_layer),
        "council_single_anchor_gpkg": str(council_path),
        "council_single_anchor_layer": str(args.council_single_anchor_layer),
        "fallback_gpkg": str(fallback_path),
        "fallback_layer": str(args.fallback_layer),
        "output_gpkg": str(args.output_gpkg),
        "output_layer": str(args.output_layer),
        "input_point_rows": int(len(points)),
        "clean_wfs_rows": int(len(clean_wfs)),
        "anchor_rows": int(len(anchors)),
        "council_rows": int(len(council)),
        "fallback_rows": int(len(fallback)),
        "selected_rows": int(len(out)),
        "selected_parent_rows": int(out["oachargeid"].astype(str).nunique()) if not out.empty else 0,
        "selected_child_rows": int(len(out)),
        "council_selected_rows": int(council_selected),
        "fallback_selected_rows": int(fallback_selected),
        "raw_clean_selected_rows": int(raw_clean_selected),
        "no_anchor_points": int(no_anchor),
        "no_raw_clean_hit_points": int(no_raw_clean_hit),
        "no_council_hit_points": int(no_council_hit),
        "multiple_council_hit_points": int(multiple_council_hits),
        "no_usable_council_wfs_match_points": int(no_usable_council_wfs_match),
        "output_miss_replaced_by_point_fallback": int(output_miss_replaced_by_point_fallback),
        "output_miss_replaced_by_point_raw_clean": int(output_miss_replaced_by_point_raw_clean),
        "output_miss_no_raw_clean_hit": int(output_miss_no_raw_clean_hit),
        "no_fallback_points": int(no_fallback),
        "max_nearest_anchor_distance": float(args.max_nearest_anchor_distance),
        "selection_rule": (
            "Use one council polygon only when exactly one linked council row contains the point and its geometry "
            "has been replaced by a WFS clean IoU match; otherwise use fallback. If no eligible anchor is found, "
            "use the intersecting wfs_raw_clean polygon. If a selected output polygon does not intersect the "
            "geocoding point, replace it with the fallback merge for the preprocessed wfs_raw_clean polygon "
            "that intersects the point, or that raw clean polygon if no fallback row exists."
        ),
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one point-first merged parcel from wfs_merge_anchor outputs.")
    parser.add_argument("--points-gpkg", required=True)
    parser.add_argument("--points-layer", default=DEFAULT_POINTS_LAYER)
    parser.add_argument("--clean-wfs-gpkg", default="")
    parser.add_argument("--clean-wfs-layer", default=DEFAULT_CLEAN_WFS_LAYER)
    parser.add_argument("--anchor-gpkg", required=True)
    parser.add_argument("--anchor-layer", default=DEFAULT_ANCHOR_LAYER)
    parser.add_argument("--council-single-anchor-gpkg", required=True)
    parser.add_argument("--council-single-anchor-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--fallback-gpkg", required=True)
    parser.add_argument("--fallback-layer", default=DEFAULT_FALLBACK_LAYER)
    parser.add_argument("--output-gpkg", required=True)
    parser.add_argument("--output-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument(
        "--max-nearest-anchor-distance",
        type=float,
        default=25.0,
        help="Maximum distance in metres for using nearest anchor when the point is not inside any anchor; set <0 to disable.",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_gpkg)
    out, summary = select_point_merge_parcels(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Output exists: {output_path}")
        output_path.unlink()
    _log(f"[INFO] Writing selected point parcels: {output_path}:{args.output_layer}")
    write_kwargs = {"geometry_type": "MultiPolygon"} if out.empty else {}
    pyogrio.write_dataframe(out, output_path, layer=str(args.output_layer), driver="GPKG", **write_kwargs)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _log("[DONE] Point parcel selection complete")
    _log(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

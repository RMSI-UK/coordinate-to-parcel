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

try:
    from wfs_merge_anchor.build_single_anchor_fallback import _shape_metrics
except ImportError:  # pragma: no cover
    from build_single_anchor_fallback import _shape_metrics


DEFAULT_COUNCIL_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons.gpkg"
DEFAULT_COUNCIL_LAYER = "council_polygons"
DEFAULT_ANCHOR_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean_anchor.gpkg"
DEFAULT_ANCHOR_LAYER = "wfs_raw_clean_anchor"
DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_LAYER = "wfs_raw_clean"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_area05.gpkg"
DEFAULT_OUTPUT_LAYER = "council_polygons_single_anchor_area05"
OUTPUT_COLUMNS = [
    "council_fid",
    "anchor_intersect_count",
    "anchor_fid",
    "anchor_clean_fid",
    "anchor_source_fid",
    "anchor_theme",
    "anchor_uprn_count",
    "anchor_overlap_area",
    "council_area",
    "anchor_overlap_council_ratio",
    "anchor_min_overlap_area_m2",
    "output_geometry_source",
    "wfs_iou_match_found",
    "wfs_iou",
    "wfs_match_clean_fids",
    "wfs_match_source_fids",
    "wfs_match_count",
    "wfs_match_regularity_score",
    "wfs_match_mrr_ratio",
    "wfs_match_hull_gap_ratio",
    "wfs_match_hole_area_ratio",
    "final_area",
]


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


def _empty_output(crs: object | None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({column: pd.Series(dtype=object) for column in OUTPUT_COLUMNS}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


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
    anchors = anchors.rename(
        columns={
            "clean_fid": "anchor_clean_fid",
            "source_fid": "anchor_source_fid",
            "Theme": "anchor_theme",
        }
    )
    anchors["anchor_fid"] = anchors["anchor_fid"].astype("int64")
    anchors["anchor_clean_fid"] = anchors["anchor_clean_fid"].astype("int64")
    anchors["anchor_source_fid"] = anchors["anchor_source_fid"].astype("int64")
    anchors["anchor_uprn_count"] = anchors["anchor_uprn_count"].astype("int64")
    return anchors


def _ids_text(values: set[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(value)) for value in sorted(int(value) for value in values))


def _plot_eligible(theme: object) -> bool:
    text = str(theme or "").lower()
    has_plot = "building" in text or "land" in text
    roadish = "road" in text or "track" in text or "path" in text
    return bool(has_plot and not roadish)


def _to_multi_polygon(geom: object) -> object:
    if geom is None:
        return geom
    if getattr(geom, "is_empty", False):
        return geom
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "MultiPolygon":
        return geom
    if geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom_type == "GeometryCollection":
        polygons = [part for part in geom.geoms if getattr(part, "geom_type", "") == "Polygon" and not part.is_empty]
        if polygons:
            return MultiPolygon(polygons)
    return geom


def _union_for_clean_ids(ids: set[int], geom_by_clean: dict[int, object]) -> object:
    geoms = [geom_by_clean[int(clean_id)] for clean_id in sorted(ids) if int(clean_id) in geom_by_clean]
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is not None and not geom.is_empty and not bool(shapely.is_valid(geom)):
        geom = shapely.make_valid(geom)
    return _to_multi_polygon(geom)


def _iou(left: object, right: object, *, left_area: float | None = None, right_area: float | None = None) -> float:
    if left is None or right is None or getattr(left, "is_empty", False) or getattr(right, "is_empty", False):
        return 0.0
    inter_area = float(shapely.area(shapely.intersection(left, right)))
    la = float(shapely.area(left)) if left_area is None else float(left_area)
    ra = float(shapely.area(right)) if right_area is None else float(right_area)
    union_area = la + ra - inter_area
    return inter_area / union_area if union_area else 0.0


def _is_regular_match(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        float(metrics.get("regularity_score", 0.0)) >= float(args.wfs_match_min_regularity)
        and float(metrics.get("hull_gap_ratio", 0.0)) <= float(args.wfs_match_max_hull_gap)
        and float(metrics.get("hole_area_ratio", 0.0)) <= float(args.wfs_match_max_hole_area_ratio)
    )


def _candidate_score(
    candidate: dict[str, Any],
) -> tuple[float, float, float, float, int]:
    return (
        float(candidate["iou"]),
        float(candidate["metrics"].get("regularity_score", 0.0)),
        -float(candidate["metrics"].get("hull_gap_ratio", 0.0)),
        float(candidate["metrics"].get("mrr_ratio", 0.0)),
        -int(len(candidate["ids"])),
    )


def _best_wfs_match_for_council(
    council_geom: object,
    council_area: float,
    candidates: pd.DataFrame,
    geom_by_clean: dict[int, object],
    source_by_clean: dict[int, int],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if candidates.empty:
        return None
    candidates = candidates.sort_values(
        ["wfs_intersection_area", "wfs_inside_ratio", "clean_fid"],
        ascending=[False, False, True],
    ).head(int(args.wfs_match_max_candidates))
    clean_ids = [int(value) for value in candidates["clean_fid"].astype("int64").to_numpy()]
    candidate_sets: dict[str, set[int]] = {}

    for clean_id in clean_ids:
        candidate_sets[_ids_text({clean_id})] = {clean_id}

    selected: set[int] = set()
    current_iou = 0.0
    for clean_id in clean_ids:
        if len(selected) >= int(args.wfs_match_max_selected):
            break
        trial = set(selected)
        trial.add(int(clean_id))
        geom = _union_for_clean_ids(trial, geom_by_clean)
        iou = _iou(geom, council_geom, right_area=council_area)
        if iou + float(args.wfs_match_min_iou_gain) >= current_iou:
            selected = trial
            current_iou = max(current_iou, iou)
            candidate_sets[_ids_text(selected)] = set(selected)

    high_inside = {
        int(row.clean_fid)
        for row in candidates.itertuples(index=False)
        if float(row.wfs_inside_ratio) >= float(args.wfs_match_high_inside_ratio)
    }
    if high_inside and len(high_inside) <= int(args.wfs_match_max_selected):
        candidate_sets[_ids_text(high_inside)] = high_inside

    evaluated: list[dict[str, Any]] = []
    for ids in candidate_sets.values():
        geom = _union_for_clean_ids(ids, geom_by_clean)
        if geom is None or getattr(geom, "is_empty", False):
            continue
        metrics = _shape_metrics(geom)
        iou = _iou(geom, council_geom, right_area=council_area)
        if iou < float(args.wfs_match_min_iou):
            continue
        if not _is_regular_match(metrics, args):
            continue
        evaluated.append(
            {
                "ids": set(ids),
                "geometry": geom,
                "metrics": metrics,
                "iou": float(iou),
            }
        )
    if not evaluated:
        return None
    best = max(evaluated, key=_candidate_score)
    ids = set(best["ids"])
    best["source_ids"] = {int(source_by_clean.get(clean_id, clean_id)) for clean_id in ids}
    return best


def _apply_wfs_iou_geometry_matches(out: gpd.GeoDataFrame, args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    if not bool(args.enable_wfs_iou_match) or out.empty:
        out = out.copy()
        out["output_geometry_source"] = "council"
        out["wfs_iou_match_found"] = 0
        out["wfs_iou"] = 0.0
        out["wfs_match_clean_fids"] = ""
        out["wfs_match_source_fids"] = ""
        out["wfs_match_count"] = 0
        out["wfs_match_regularity_score"] = 0.0
        out["wfs_match_mrr_ratio"] = 0.0
        out["wfs_match_hull_gap_ratio"] = 0.0
        out["wfs_match_hole_area_ratio"] = 0.0
        out["final_area"] = np.asarray(shapely.area(out.geometry.to_numpy()), dtype="float64")
        return out, {"enabled": bool(args.enable_wfs_iou_match), "matched_rows": 0}

    wfs_path = Path(args.wfs_gpkg)
    bbox = tuple(float(value) for value in out.total_bounds)
    _log(f"[INFO] Reading WFS clean candidates for council IoU matching: {wfs_path}:{args.wfs_layer}")
    wfs = pyogrio.read_dataframe(
        wfs_path,
        layer=str(args.wfs_layer),
        columns=["clean_fid", "source_fid", "Theme", "clean_area"],
        bbox=bbox,
    )
    wfs = wfs[wfs.geometry.notna() & ~wfs.geometry.is_empty].copy()
    if wfs.crs != out.crs:
        wfs = wfs.to_crs(out.crs)
    wfs["clean_fid"] = wfs["clean_fid"].astype("int64")
    wfs["source_fid"] = wfs["source_fid"].fillna(wfs["clean_fid"]).astype("int64")
    computed_area = pd.Series(shapely.area(wfs.geometry.to_numpy()), index=wfs.index)
    wfs["clean_area"] = wfs["clean_area"].fillna(computed_area).astype("float64")
    wfs = wfs[wfs["Theme"].map(_plot_eligible)].copy().reset_index(drop=True)
    _log(f"[INFO] Plot-eligible WFS candidates in bbox={len(wfs):,}")

    work = out[["council_fid", "council_area", "geometry"]].copy()
    work["match_row_id"] = np.arange(len(work), dtype="int64")
    _log("[INFO] Spatial join council single-anchor polygons to WFS clean candidates")
    joined = gpd.sjoin(
        work[["match_row_id", "council_fid", "council_area", "geometry"]],
        wfs[["clean_fid", "source_fid", "Theme", "clean_area", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="council",
        rsuffix="wfs",
    )
    if joined.empty:
        out = out.copy()
        out["output_geometry_source"] = "council"
        out["wfs_iou_match_found"] = 0
        out["wfs_iou"] = 0.0
        out["wfs_match_clean_fids"] = ""
        out["wfs_match_source_fids"] = ""
        out["wfs_match_count"] = 0
        out["wfs_match_regularity_score"] = 0.0
        out["wfs_match_mrr_ratio"] = 0.0
        out["wfs_match_hull_gap_ratio"] = 0.0
        out["wfs_match_hole_area_ratio"] = 0.0
        out["final_area"] = np.asarray(shapely.area(out.geometry.to_numpy()), dtype="float64")
        return out, {"enabled": True, "matched_rows": 0, "candidate_pairs": 0}

    right_pos = joined["index_wfs"].to_numpy(dtype="int64")
    council_geoms = np.asarray(joined.geometry.to_numpy(), dtype=object)
    wfs_geoms = np.asarray(wfs.geometry.iloc[right_pos].to_numpy(), dtype=object)
    joined["wfs_intersection_area"] = np.asarray(
        shapely.area(shapely.intersection(council_geoms, wfs_geoms)),
        dtype="float64",
    )
    joined = joined[joined["wfs_intersection_area"].ge(float(args.wfs_match_min_intersection_area))].copy()
    joined["wfs_inside_ratio"] = joined["wfs_intersection_area"] / joined["clean_area"].replace(0.0, np.nan)
    joined["wfs_inside_ratio"] = joined["wfs_inside_ratio"].fillna(0.0)
    joined["wfs_council_cover_ratio"] = joined["wfs_intersection_area"] / joined["council_area"].replace(0.0, np.nan)
    joined["wfs_council_cover_ratio"] = joined["wfs_council_cover_ratio"].fillna(0.0)
    joined = joined[
        joined["wfs_inside_ratio"].ge(float(args.wfs_match_min_candidate_inside_ratio))
        | joined["wfs_council_cover_ratio"].ge(float(args.wfs_match_min_candidate_cover_ratio))
    ].copy()
    _log(f"[INFO] WFS IoU candidate pairs after filters={len(joined):,}")

    geom_by_clean = dict(zip(wfs["clean_fid"].astype(int), wfs.geometry.to_numpy()))
    source_by_clean = dict(zip(wfs["clean_fid"].astype(int), wfs["source_fid"].astype(int)))
    output_geoms = np.asarray(out.geometry.to_numpy(), dtype=object).copy()
    out = out.copy()
    out["output_geometry_source"] = "council"
    out["wfs_iou_match_found"] = 0
    out["wfs_iou"] = 0.0
    out["wfs_match_clean_fids"] = ""
    out["wfs_match_source_fids"] = ""
    out["wfs_match_count"] = 0
    out["wfs_match_regularity_score"] = 0.0
    out["wfs_match_mrr_ratio"] = 0.0
    out["wfs_match_hull_gap_ratio"] = 0.0
    out["wfs_match_hole_area_ratio"] = 0.0

    candidate_groups = {int(key): group.copy() for key, group in joined.groupby("match_row_id", sort=False)}
    matched_rows = 0
    evaluated_rows = 0
    for row_id, row in enumerate(out.itertuples(index=False)):
        group = candidate_groups.get(int(row_id))
        if group is None or group.empty:
            continue
        evaluated_rows += 1
        if evaluated_rows == 1 or evaluated_rows % int(args.log_every) == 0:
            _log(f"[INFO] WFS IoU matching rows evaluated={evaluated_rows:,}; matched={matched_rows:,}")
        council_geom = output_geoms[row_id]
        council_area = float(getattr(row, "council_area"))
        match = _best_wfs_match_for_council(
            council_geom,
            council_area,
            group,
            geom_by_clean,
            source_by_clean,
            args,
        )
        if match is None:
            continue
        metrics = match["metrics"]
        output_geoms[row_id] = match["geometry"]
        out.at[row_id, "output_geometry_source"] = "wfs_clean_iou_match"
        out.at[row_id, "wfs_iou_match_found"] = 1
        out.at[row_id, "wfs_iou"] = float(match["iou"])
        out.at[row_id, "wfs_match_clean_fids"] = _ids_text(match["ids"])
        out.at[row_id, "wfs_match_source_fids"] = _ids_text(match["source_ids"])
        out.at[row_id, "wfs_match_count"] = int(len(match["ids"]))
        out.at[row_id, "wfs_match_regularity_score"] = float(metrics.get("regularity_score", 0.0))
        out.at[row_id, "wfs_match_mrr_ratio"] = float(metrics.get("mrr_ratio", 0.0))
        out.at[row_id, "wfs_match_hull_gap_ratio"] = float(metrics.get("hull_gap_ratio", 0.0))
        out.at[row_id, "wfs_match_hole_area_ratio"] = float(metrics.get("hole_area_ratio", 0.0))
        matched_rows += 1

    out = gpd.GeoDataFrame(out, geometry=gpd.GeoSeries(output_geoms, crs=out.crs), crs=out.crs)
    out["final_area"] = np.asarray(shapely.area(out.geometry.to_numpy()), dtype="float64")
    summary = {
        "enabled": True,
        "wfs_gpkg": str(wfs_path),
        "wfs_layer": str(args.wfs_layer),
        "plot_eligible_wfs_candidates": int(len(wfs)),
        "candidate_pairs_after_filters": int(len(joined)),
        "evaluated_rows_with_candidates": int(evaluated_rows),
        "matched_rows": int(matched_rows),
        "min_iou": float(args.wfs_match_min_iou),
        "min_regularity": float(args.wfs_match_min_regularity),
        "max_hull_gap": float(args.wfs_match_max_hull_gap),
        "max_hole_area_ratio": float(args.wfs_match_max_hole_area_ratio),
        "max_candidates_per_council": int(args.wfs_match_max_candidates),
    }
    return out, summary


def build_council_single_anchor(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    council_path = Path(args.council_gpkg)
    anchor_path = Path(args.anchor_gpkg)
    _log(f"[INFO] Reading anchors: {anchor_path}:{args.anchor_layer}")
    anchors = _read_anchors(anchor_path, str(args.anchor_layer))
    if anchors.empty:
        out = _empty_output(anchors.crs)
        return out, {
            "council_gpkg": str(council_path),
            "council_layer": str(args.council_layer),
            "anchor_gpkg": str(anchor_path),
            "anchor_layer": str(args.anchor_layer),
            "output_gpkg": str(args.output_gpkg),
            "output_layer": str(args.output_layer),
            "min_anchor_overlap_area": float(args.min_anchor_overlap_area),
            "council_rows_in_bbox": 0,
            "anchor_rows": 0,
            "qualifying_overlap_pairs": 0,
            "single_anchor_rows": 0,
            "single_anchor_unique_anchors": 0,
            "wfs_iou_geometry_match": {},
            "selection_rule": "No anchors available; empty council single-anchor output.",
        }
    bbox = tuple(float(value) for value in anchors.total_bounds)
    _log(f"[INFO] Anchors={len(anchors):,}; bbox={bbox}")

    _log(f"[INFO] Reading council polygons in anchor bbox: {council_path}:{args.council_layer}")
    council = pyogrio.read_dataframe(council_path, layer=str(args.council_layer), bbox=bbox, fid_as_index=True)
    council = council[council.geometry.notna() & ~council.geometry.is_empty].copy()
    council.index.name = "council_fid"
    council = council.reset_index()
    council["council_fid"] = council["council_fid"].astype("int64")
    if council.crs != anchors.crs:
        council = council.to_crs(anchors.crs)
    _log(f"[INFO] Council rows in bbox={len(council):,}")

    _log("[INFO] Spatial join council polygons to anchors")
    joined = gpd.sjoin(
        council[["council_fid", "geometry"]],
        anchors[["anchor_fid", "anchor_clean_fid", "anchor_source_fid", "anchor_theme", "anchor_uprn_count", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        out = _empty_output(council.crs)
        summary = {
            "council_gpkg": str(council_path),
            "council_layer": str(args.council_layer),
            "anchor_gpkg": str(anchor_path),
            "anchor_layer": str(args.anchor_layer),
            "output_gpkg": str(args.output_gpkg),
            "output_layer": str(args.output_layer),
            "council_rows_in_bbox": int(len(council)),
            "anchor_rows": int(len(anchors)),
            "qualifying_overlap_pairs": 0,
            "single_anchor_rows": 0,
            "single_anchor_unique_anchors": 0,
        }
        return out, summary

    right_pos = joined["index_right"].to_numpy(dtype="int64")
    council_geoms = np.asarray(joined.geometry.to_numpy(), dtype=object)
    anchor_geoms = np.asarray(anchors.geometry.iloc[right_pos].to_numpy(), dtype=object)
    joined["anchor_overlap_area"] = np.asarray(
        shapely.area(shapely.intersection(council_geoms, anchor_geoms)),
        dtype="float64",
    )
    joined = joined[joined["anchor_overlap_area"].ge(float(args.min_anchor_overlap_area))].copy()
    if joined.empty:
        out = _empty_output(council.crs)
        summary = {
            "council_gpkg": str(council_path),
            "council_layer": str(args.council_layer),
            "anchor_gpkg": str(anchor_path),
            "anchor_layer": str(args.anchor_layer),
            "output_gpkg": str(args.output_gpkg),
            "output_layer": str(args.output_layer),
            "council_rows_in_bbox": int(len(council)),
            "anchor_rows": int(len(anchors)),
            "qualifying_overlap_pairs": 0,
            "single_anchor_rows": 0,
            "single_anchor_unique_anchors": 0,
            "min_anchor_overlap_area": float(args.min_anchor_overlap_area),
        }
        return out, summary

    joined["anchor_intersect_count"] = joined.groupby("council_fid")["anchor_fid"].transform("nunique").astype("int64")
    single = joined[joined["anchor_intersect_count"].eq(1)].copy()
    single = single.sort_values(["council_fid", "anchor_overlap_area", "anchor_fid"], ascending=[True, False, True])
    single = single.drop_duplicates("council_fid", keep="first").copy()

    council_attrs = [col for col in council.columns if col != "geometry"]
    out = council.merge(
        single[
            [
                "council_fid",
                "anchor_intersect_count",
                "anchor_fid",
                "anchor_clean_fid",
                "anchor_source_fid",
                "anchor_theme",
                "anchor_uprn_count",
                "anchor_overlap_area",
            ]
        ],
        on="council_fid",
        how="inner",
        validate="one_to_one",
    )
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=council.crs)
    out["council_area"] = np.asarray(shapely.area(out.geometry.to_numpy()), dtype="float64")
    out["anchor_overlap_council_ratio"] = out["anchor_overlap_area"] / out["council_area"].replace(0.0, np.nan)
    out["anchor_overlap_council_ratio"] = out["anchor_overlap_council_ratio"].fillna(0.0)
    out["anchor_min_overlap_area_m2"] = float(args.min_anchor_overlap_area)
    out, wfs_iou_summary = _apply_wfs_iou_geometry_matches(out, args)

    ordered_cols = [
        *council_attrs,
        "anchor_intersect_count",
        "anchor_fid",
        "anchor_clean_fid",
        "anchor_source_fid",
        "anchor_theme",
        "anchor_uprn_count",
        "anchor_overlap_area",
        "council_area",
        "anchor_overlap_council_ratio",
        "anchor_min_overlap_area_m2",
        "output_geometry_source",
        "wfs_iou_match_found",
        "wfs_iou",
        "wfs_match_clean_fids",
        "wfs_match_source_fids",
        "wfs_match_count",
        "wfs_match_regularity_score",
        "wfs_match_mrr_ratio",
        "wfs_match_hull_gap_ratio",
        "wfs_match_hole_area_ratio",
        "final_area",
        "geometry",
    ]
    out = out[ordered_cols].sort_values("council_fid").reset_index(drop=True)
    summary = {
        "council_gpkg": str(council_path),
        "council_layer": str(args.council_layer),
        "anchor_gpkg": str(anchor_path),
        "anchor_layer": str(args.anchor_layer),
        "output_gpkg": str(args.output_gpkg),
        "output_layer": str(args.output_layer),
        "min_anchor_overlap_area": float(args.min_anchor_overlap_area),
        "council_rows_in_bbox": int(len(council)),
        "anchor_rows": int(len(anchors)),
        "qualifying_overlap_pairs": int(len(joined)),
        "single_anchor_rows": int(len(out)),
        "single_anchor_unique_anchors": int(out["anchor_clean_fid"].nunique()) if not out.empty else 0,
        "wfs_iou_geometry_match": wfs_iou_summary,
        "selection_rule": "Council polygon qualifies when exactly one anchor overlaps by at least the area threshold",
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract council polygons with exactly one qualifying WFS anchor.")
    parser.add_argument("--council-gpkg", default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--council-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--anchor-gpkg", default=DEFAULT_ANCHOR_GPKG)
    parser.add_argument("--anchor-layer", default=DEFAULT_ANCHOR_LAYER)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--min-anchor-overlap-area", type=float, default=0.5)
    parser.add_argument("--enable-wfs-iou-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wfs-match-min-iou", type=float, default=0.90)
    parser.add_argument("--wfs-match-min-regularity", type=float, default=0.70)
    parser.add_argument("--wfs-match-max-hull-gap", type=float, default=0.05)
    parser.add_argument("--wfs-match-max-hole-area-ratio", type=float, default=0.02)
    parser.add_argument("--wfs-match-min-intersection-area", type=float, default=0.05)
    parser.add_argument("--wfs-match-min-candidate-inside-ratio", type=float, default=0.50)
    parser.add_argument("--wfs-match-min-candidate-cover-ratio", type=float, default=0.01)
    parser.add_argument("--wfs-match-high-inside-ratio", type=float, default=0.90)
    parser.add_argument("--wfs-match-max-candidates", type=int, default=24)
    parser.add_argument("--wfs-match-max-selected", type=int, default=20)
    parser.add_argument("--wfs-match-min-iou-gain", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_gpkg)
    out, summary = build_council_single_anchor(args)
    if output_path.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Output exists: {output_path}")
        output_path.unlink()
    _log(f"[INFO] Writing council single-anchor polygons: {output_path}:{args.output_layer}")
    write_kwargs = {"geometry_type": "MultiPolygon"} if out.empty else {}
    pyogrio.write_dataframe(out, output_path, layer=str(args.output_layer), driver="GPKG", **write_kwargs)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _log("[DONE] Council single-anchor build complete")
    _log(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

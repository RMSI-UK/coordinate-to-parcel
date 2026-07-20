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


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_LAYER = "wfs_raw_clean"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean_anchor.gpkg"
DEFAULT_OUTPUT_LAYER = "wfs_raw_clean_anchor"
DEFAULT_MAX_ANCHOR_AREA_M2 = 4000.0
DEFAULT_TARGET_CRS = "EPSG:27700"


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


def _is_anchor_theme(theme: object) -> bool:
    text = str(theme or "").lower()
    return "building" in text or "land" in text


def _anchor_rule(max_anchor_area: float) -> str:
    if max_anchor_area > 0.0:
        return (
            "Theme contains building or land, area is at or below max-anchor-area, "
            "and polygon intersects one or more distinct UPRN points"
        )
    return "Theme contains building or land and polygon intersects one or more distinct UPRN points"


def _empty_anchor_gdf(crs: object | None) -> gpd.GeoDataFrame:
    target_crs = crs or DEFAULT_TARGET_CRS
    return gpd.GeoDataFrame(
        {
            "clean_fid": pd.Series(dtype="int64"),
            "source_fid": pd.Series(dtype="int64"),
            "Theme": pd.Series(dtype=object),
            "anchor_uprn_count": pd.Series(dtype="int64"),
        },
        geometry=gpd.GeoSeries([], crs=target_crs),
        crs=target_crs,
    )


def build_anchor_layer(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    wfs_path = Path(args.wfs_gpkg)
    uprn_path = Path(args.uprn_gpkg)
    _log(f"[INFO] Reading clean WFS: {wfs_path}:{args.wfs_layer}")
    wfs = pyogrio.read_dataframe(wfs_path, layer=str(args.wfs_layer))
    wfs = wfs[wfs.geometry.notna() & ~wfs.geometry.is_empty].copy()
    if wfs.empty:
        for column, dtype in [
            ("clean_fid", "int64"),
            ("source_fid", "int64"),
            ("Theme", object),
        ]:
            if column not in wfs.columns:
                wfs[column] = np.array([], dtype=dtype)
    if "clean_fid" not in wfs.columns:
        raise ValueError(f"{wfs_path}:{args.wfs_layer} must contain clean_fid")
    if "Theme" not in wfs.columns:
        raise ValueError(f"{wfs_path}:{args.wfs_layer} must contain Theme")
    if "source_fid" not in wfs.columns:
        raise ValueError(f"{wfs_path}:{args.wfs_layer} must contain source_fid")
    wfs["clean_fid"] = wfs["clean_fid"].astype("int64")
    candidates = wfs[wfs["Theme"].map(_is_anchor_theme)].copy()
    anchor_theme_candidate_rows = int(len(candidates))
    max_anchor_area = float(args.max_anchor_area)
    area_excluded_rows = 0
    if max_anchor_area > 0.0 and not candidates.empty:
        area_mask = candidates.geometry.area.astype(float).le(max_anchor_area)
        area_excluded_rows = int((~area_mask).sum())
        candidates = candidates.loc[area_mask].copy()
    _log(
        f"[INFO] WFS rows={len(wfs):,}; anchor-theme candidates={anchor_theme_candidate_rows:,}; "
        f"after area gate={len(candidates):,}; max_anchor_area={max_anchor_area:g}m2"
    )

    if candidates.empty:
        anchors = _empty_anchor_gdf(wfs.crs)
        summary = {
            "wfs_gpkg": str(wfs_path),
            "wfs_layer": str(args.wfs_layer),
            "uprn_gpkg": str(uprn_path),
            "uprn_layer": str(args.uprn_layer),
            "uprn_id_field": str(args.uprn_id_field),
            "output_gpkg": str(args.output_gpkg),
            "output_layer": str(args.output_layer),
            "wfs_rows": int(len(wfs)),
            "anchor_theme_candidate_rows": anchor_theme_candidate_rows,
            "max_anchor_area_m2": max_anchor_area,
            "anchor_area_excluded_rows": area_excluded_rows,
            "anchor_theme_area_candidate_rows": int(len(candidates)),
            "uprn_points_in_bbox": 0,
            "anchor_rows": 0,
            "anchor_uprn_count_sum": 0,
            "preserved_existing_fid_order": False,
            "anchor_rule": _anchor_rule(max_anchor_area),
        }
        return anchors, summary

    bbox = tuple(float(value) for value in candidates.total_bounds)
    _log(f"[INFO] Reading UPRN points in candidate bbox: {bbox}")
    uprn = pyogrio.read_dataframe(uprn_path, layer=str(args.uprn_layer), columns=[str(args.uprn_id_field)], bbox=bbox)
    uprn = uprn[uprn.geometry.notna() & ~uprn.geometry.is_empty].copy()
    if uprn.crs != candidates.crs:
        uprn = uprn.to_crs(candidates.crs)
    _log(f"[INFO] UPRN points in bbox={len(uprn):,}")

    _log("[INFO] Spatial join UPRN points to anchor-theme polygons")
    joined = gpd.sjoin(
        uprn[[str(args.uprn_id_field), "geometry"]],
        candidates[["clean_fid", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    counts = joined.groupby("clean_fid")[str(args.uprn_id_field)].nunique().astype("int64")
    anchors = candidates[candidates["clean_fid"].isin(counts.index)].copy()
    anchors["anchor_uprn_count"] = anchors["clean_fid"].map(counts).fillna(0).astype("int64")
    anchors = anchors[anchors["anchor_uprn_count"].gt(0)].copy()
    preserved_existing_fid_order = False
    output_path = Path(args.output_gpkg)
    if bool(args.preserve_existing_fid_order) and output_path.exists():
        try:
            existing = pyogrio.read_dataframe(
                output_path,
                layer=str(args.output_layer),
                columns=["clean_fid"],
                read_geometry=False,
                fid_as_index=True,
            )
            existing.index.name = "existing_fid"
            existing = existing.reset_index()
            order = dict(zip(existing["clean_fid"].astype("int64"), existing["existing_fid"].astype("int64")))
            fallback_start = int(existing["existing_fid"].max()) + 1 if not existing.empty else 0
            anchors["_existing_fid_order"] = anchors["clean_fid"].map(order)
            missing_order = anchors["_existing_fid_order"].isna()
            anchors.loc[missing_order, "_existing_fid_order"] = np.arange(
                fallback_start,
                fallback_start + int(missing_order.sum()),
                dtype="int64",
            )
            anchors = anchors.sort_values("_existing_fid_order").drop(columns="_existing_fid_order")
            preserved_existing_fid_order = True
        except Exception as exc:
            _log(f"[WARN] Could not preserve existing anchor FID order from {output_path}: {exc}")
    # Preserve selected order in the GeoPackage FID sequence. Downstream
    # diagnostics often refer to layer FID.
    anchors = anchors.reset_index(drop=True)

    summary = {
        "wfs_gpkg": str(wfs_path),
        "wfs_layer": str(args.wfs_layer),
        "uprn_gpkg": str(uprn_path),
        "uprn_layer": str(args.uprn_layer),
        "uprn_id_field": str(args.uprn_id_field),
        "output_gpkg": str(args.output_gpkg),
        "output_layer": str(args.output_layer),
        "wfs_rows": int(len(wfs)),
        "anchor_theme_candidate_rows": anchor_theme_candidate_rows,
        "max_anchor_area_m2": max_anchor_area,
        "anchor_area_excluded_rows": area_excluded_rows,
        "anchor_theme_area_candidate_rows": int(len(candidates)),
        "uprn_points_in_bbox": int(len(uprn)),
        "anchor_rows": int(len(anchors)),
        "anchor_uprn_count_sum": int(anchors["anchor_uprn_count"].sum()) if not anchors.empty else 0,
        "preserved_existing_fid_order": bool(preserved_existing_fid_order),
        "anchor_rule": _anchor_rule(max_anchor_area),
    }
    return anchors, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract WFS anchor polygons from clean WFS and UPRN points.")
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default="UPRN")
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument(
        "--max-anchor-area",
        type=float,
        default=DEFAULT_MAX_ANCHOR_AREA_M2,
        help="Maximum anchor polygon area in square metres; set <=0 to disable.",
    )
    parser.add_argument("--preserve-existing-fid-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_gpkg)
    anchors, summary = build_anchor_layer(args)
    if output_path.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Output exists: {output_path}")
        output_path.unlink()
    _log(f"[INFO] Writing anchors: {output_path}:{args.output_layer}")
    write_kwargs = {"geometry_type": "MultiPolygon"} if anchors.empty else {}
    pyogrio.write_dataframe(anchors, output_path, layer=str(args.output_layer), driver="GPKG", **write_kwargs)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _log("[DONE] Anchor layer build complete")
    _log(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

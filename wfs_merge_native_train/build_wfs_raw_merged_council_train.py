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
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg"
DEFAULT_WFS_LAYER = "polygons_in_buffers"
DEFAULT_COUNCIL_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons.gpkg"
DEFAULT_COUNCIL_LAYER = "council_polygons"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_merged_council_train.gpkg"

MERGED_ONLY_LAYER = "wfs_raw_merged_council_train_merged_only"


def _log(message: str) -> None:
    print(message, flush=True)


def _ids_text(values: list[int] | tuple[int, ...] | set[int]) -> str:
    return "|".join(str(int(v)) for v in sorted(values))


def _theme_text(rows: pd.DataFrame) -> pd.Series:
    parts = []
    for column in ["Theme", "DescriptiveGroup", "DescriptiveTerm"]:
        if column in rows.columns:
            parts.append(rows[column].fillna("").astype(str))
    if not parts:
        return pd.Series("", index=rows.index)
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep=" ")
    return out.str.lower()


def _as_valid_geometry(values: Any) -> Any:
    valid = shapely.is_valid(values)
    if bool(np.all(valid)):
        return values
    out = np.asarray(values, dtype=object).copy()
    bad = ~np.asarray(valid, dtype=bool)
    out[bad] = shapely.make_valid(out[bad])
    return out


def _polygon_parts(geom: Any) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if not part.is_empty]
    if isinstance(geom, GeometryCollection):
        parts: list[Polygon] = []
        for item in geom.geoms:
            parts.extend(_polygon_parts(item))
        return parts
    return []


def _hole_count(geom: Any) -> int:
    return int(sum(len(part.interiors) for part in _polygon_parts(geom)))


def _as_multipolygon(geom: Any) -> MultiPolygon:
    parts = _polygon_parts(geom)
    if not parts:
        return MultiPolygon()
    return MultiPolygon(parts)


def _fill_holes(geom: Any) -> Any:
    parts = _polygon_parts(geom)
    if not parts:
        return geom
    filled = [Polygon(part.exterior) for part in parts if not part.is_empty]
    if not filled:
        return geom
    out = MultiPolygon(filled) if len(filled) > 1 else filled[0]
    return shapely.make_valid(out) if not bool(shapely.is_valid(out)) else out


def _repair_merged_geometry(geom: Any, *, slit_close_buffer: float) -> tuple[Any, int, float, float]:
    before_area = float(shapely.area(geom))
    before_holes = _hole_count(geom)
    repaired = _fill_holes(geom)
    if float(slit_close_buffer) > 0.0 and repaired is not None and not repaired.is_empty:
        closed = repaired.buffer(float(slit_close_buffer), join_style=2).buffer(
            -float(slit_close_buffer),
            join_style=2,
        )
        if closed is not None and not closed.is_empty:
            repaired = _fill_holes(shapely.make_valid(closed))
    repaired = shapely.make_valid(repaired) if not bool(shapely.is_valid(repaired)) else repaired
    repaired = _as_multipolygon(repaired)
    after_area = float(shapely.area(repaired))
    return repaired, before_holes, float(after_area - before_area), float(before_area)


def _concavity_metrics(geom: Any, *, close_buffer: float) -> dict[str, float]:
    area = float(shapely.area(geom))
    hull = shapely.convex_hull(geom)
    hull_area = float(shapely.area(hull))
    hull_perimeter = float(shapely.length(hull))
    perimeter = float(shapely.length(geom))
    hull_gap_ratio = max(hull_area - area, 0.0) / max(area, 1e-9)
    perimeter_hull_ratio = perimeter / max(hull_perimeter, 1e-9)
    notch_index = hull_gap_ratio * max(perimeter_hull_ratio - 1.0, 0.0)
    close_area_delta_ratio = 0.0
    if float(close_buffer) > 0.0 and geom is not None and not geom.is_empty:
        closed = geom.buffer(float(close_buffer), join_style=2).buffer(
            -float(close_buffer),
            join_style=2,
        )
        if closed is not None and not closed.is_empty:
            close_area_delta_ratio = max(float(shapely.area(closed)) - area, 0.0) / max(area, 1e-9)
    return {
        "shape_hull_gap_ratio": float(hull_gap_ratio),
        "shape_perimeter_hull_ratio": float(perimeter_hull_ratio),
        "shape_notch_index": float(notch_index),
        "shape_concavity_close_area_ratio": float(close_area_delta_ratio),
    }


def _read_raw(path: Path, layer: str) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading raw WFS: {path}:{layer}")
    raw = pyogrio.read_dataframe(path, layer=layer, fid_as_index=True)
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    raw["wfs_fid"] = raw.index.astype("int64")
    raw.geometry = _as_valid_geometry(raw.geometry.to_numpy())
    raw["raw_area"] = raw.geometry.area.astype("float64")
    raw["_is_building"] = _theme_text(raw).str.contains("building", regex=False, na=False)
    raw["_theme_contains_land"] = raw.get("Theme", pd.Series("", index=raw.index)).fillna("").astype(str).str.lower().str.contains(
        "land",
        regex=False,
        na=False,
    )
    raw["_is_anchor_theme"] = raw["_is_building"] | raw["_theme_contains_land"]
    raw["_anchor_kind"] = np.select(
        [
            raw["_is_building"] & raw["_theme_contains_land"],
            raw["_is_building"],
            raw["_theme_contains_land"],
        ],
        ["building_land", "building", "land"],
        default="other",
    )
    raw = raw[raw["raw_area"].gt(0.0)].copy()
    _log(
        "[INFO] Raw WFS polygons="
        f"{len(raw):,}; buildings={int(raw['_is_building'].sum()):,}; "
        f"theme_land={int(raw['_theme_contains_land'].sum()):,}; "
        f"anchor_theme={int(raw['_is_anchor_theme'].sum()):,}; bounds={tuple(raw.total_bounds)}"
    )
    return raw


def _read_uprn(path: Path, layer: str, bbox: tuple[float, float, float, float], crs: Any) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading UPRN bbox: {path}:{layer}")
    uprn = pyogrio.read_dataframe(path, layer=layer, columns=["UPRN"], bbox=bbox)
    uprn = uprn[uprn.geometry.notna() & ~uprn.geometry.is_empty].copy()
    if uprn.crs != crs:
        uprn = uprn.to_crs(crs)
    _log(f"[INFO] UPRN points in raw bbox={len(uprn):,}")
    return uprn


def _anchor_polygons(raw: gpd.GeoDataFrame, uprn: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    anchor_candidates = raw.loc[
        raw["_is_anchor_theme"],
        [
            "wfs_fid",
            "OBJECTID",
            "Theme",
            "DescriptiveGroup",
            "DescriptiveTerm",
            "_is_building",
            "_theme_contains_land",
            "_anchor_kind",
            "geometry",
        ],
    ].copy()
    _log("[INFO] Spatial join UPRN points to raw anchor-theme polygons")
    joined = gpd.sjoin(
        uprn[["UPRN", "geometry"]],
        anchor_candidates[["wfs_fid", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        raise RuntimeError("No UPRN points intersect raw anchor-theme polygons.")
    counts = joined.groupby("wfs_fid")["UPRN"].nunique().astype("int64")
    anchors = anchor_candidates[anchor_candidates["wfs_fid"].isin(counts.index)].copy()
    anchors["anchor_uprn_count"] = anchors["wfs_fid"].map(counts).fillna(0).astype("int64")
    anchors = anchors[anchors["anchor_uprn_count"].gt(0)].copy()
    anchors["anchor_point_geometry"] = shapely.point_on_surface(anchors.geometry.to_numpy())
    _log(
        "[INFO] Anchor polygons="
        f"{len(anchors):,}; kinds={anchors['_anchor_kind'].value_counts().to_dict()}"
    )
    return anchors


def _read_council(
    path: Path,
    layer: str,
    bbox: tuple[float, float, float, float],
    crs: Any,
    source_filter: str,
) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading council polygons in raw bbox: {path}:{layer}")
    council = pyogrio.read_dataframe(path, layer=layer, fid_as_index=True, bbox=bbox)
    council = council[council.geometry.notna() & ~council.geometry.is_empty].copy()
    council["council_fid"] = council.index.astype("int64")
    if council.crs != crs:
        council = council.to_crs(crs)
    if source_filter.strip() and "source_council" in council.columns:
        before = len(council)
        council = council[council["source_council"].astype(str).eq(source_filter.strip())].copy()
        _log(f"[INFO] Council source filter {source_filter!r}: {before:,} -> {len(council):,}")
    council.geometry = _as_valid_geometry(council.geometry.to_numpy())
    council["council_area"] = council.geometry.area.astype("float64")
    council = council[council["council_area"].gt(0.0)].copy()
    _log(f"[INFO] Council polygons after filter={len(council):,}")
    return council


def _council_with_one_anchor(council: gpd.GeoDataFrame, anchors: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    _log("[INFO] Finding council polygons with exactly one anchor polygon")
    anchor_points = gpd.GeoDataFrame(
        anchors[["wfs_fid", "anchor_uprn_count", "_anchor_kind"]].copy(),
        geometry=anchors["anchor_point_geometry"],
        crs=anchors.crs,
    )
    joined = gpd.sjoin(
        anchor_points,
        council[["council_fid", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        raise RuntimeError("No anchor polygon representative points fall inside council polygons.")
    anchor_counts = joined.groupby("council_fid")["wfs_fid"].nunique()
    one_anchor_ids = set(anchor_counts[anchor_counts.eq(1)].index.astype("int64"))
    one_anchor = council[council["council_fid"].isin(one_anchor_ids)].copy()
    first_anchor = (
        joined[joined["council_fid"].isin(one_anchor_ids)]
        .sort_values(["council_fid", "wfs_fid"])
        .drop_duplicates("council_fid")
        .set_index("council_fid")
    )
    one_anchor["anchor_wfs_fid"] = one_anchor["council_fid"].map(first_anchor["wfs_fid"]).astype("int64")
    one_anchor["anchor_uprn_count"] = (
        one_anchor["council_fid"].map(first_anchor["anchor_uprn_count"]).fillna(0).astype("int64")
    )
    one_anchor["anchor_kind"] = one_anchor["council_fid"].map(first_anchor["_anchor_kind"]).fillna("")
    _log(
        "[INFO] Council polygons with anchors="
        f"{int(len(anchor_counts)):,}; exactly_one_anchor={len(one_anchor):,}"
    )
    return one_anchor


def _build_council_candidates(
    raw: gpd.GeoDataFrame,
    council_one_anchor: gpd.GeoDataFrame,
    *,
    min_overlap_ratio: float,
    min_raw_pieces: int,
    min_intersection_area: float,
    raw_piece_min_inside_ratio: float,
    council_piece_min_share: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    _log("[INFO] Building council-to-raw merge candidates")
    raw_geoms = np.asarray(raw.geometry.to_numpy(), dtype=object)
    raw_fids = raw["wfs_fid"].to_numpy(dtype="int64")
    raw_objectids = pd.to_numeric(raw.get("OBJECTID", pd.Series(np.nan, index=raw.index)), errors="coerce").to_numpy()
    raw_areas = raw["raw_area"].to_numpy(dtype="float64")
    raw_is_anchor = (
        raw["anchor_uprn_count"].fillna(0).astype("int64").gt(0)
        & raw["_is_anchor_theme"].fillna(False)
    ).to_numpy()
    raw_pos_by_fid = {int(fid): pos for pos, fid in enumerate(raw_fids)}
    tree = shapely.STRtree(raw_geoms)

    records: list[dict[str, Any]] = []
    raw_position_records: list[dict[str, Any]] = []
    council_rows = list(council_one_anchor.itertuples(index=False))
    for offset, row in enumerate(council_rows, start=1):
        if offset == 1 or offset % 10000 == 0:
            _log(f"[INFO] Coverage scan {offset:,}/{len(council_rows):,}; accepted_candidates={len(records):,}")
        council_geom = row.geometry
        council_area = float(row.council_area)
        if council_area <= 0.0:
            continue
        positions = np.asarray(tree.query(council_geom), dtype="int64")
        if len(positions) == 0:
            continue
        intersections = shapely.area(shapely.intersection(raw_geoms[positions], council_geom))
        intersections = np.asarray(intersections, dtype="float64")
        positive = intersections >= float(min_intersection_area)
        if not bool(np.any(positive)):
            continue
        positions = positions[positive]
        intersections = intersections[positive]
        raw_inside = intersections / np.maximum(raw_areas[positions], 1e-9)
        council_share = intersections / max(council_area, 1e-9)
        selected_mask = (raw_inside >= float(raw_piece_min_inside_ratio)) | (
            council_share >= float(council_piece_min_share)
        )
        if not bool(np.any(selected_mask)):
            continue
        selected_positions = positions[selected_mask]
        selected_intersections = intersections[selected_mask]
        source_count = int(len(selected_positions))
        if source_count < int(min_raw_pieces):
            continue
        selected_raw_area = float(raw_areas[selected_positions].sum())
        intersection_sum = float(selected_intersections.sum())
        council_coverage = intersection_sum / max(council_area, 1e-9)
        wfs_coverage = intersection_sum / max(selected_raw_area, 1e-9)
        if council_coverage < float(min_overlap_ratio) or wfs_coverage < float(min_overlap_ratio):
            continue
        anchor_fid = int(row.anchor_wfs_fid)
        anchor_pos = raw_pos_by_fid.get(anchor_fid)
        if anchor_pos is None or anchor_pos not in set(int(v) for v in selected_positions):
            continue
        selected_anchor_count = int(np.sum(raw_is_anchor[selected_positions]))
        if selected_anchor_count != 1:
            continue
        selected_fids = [int(raw_fids[pos]) for pos in selected_positions]
        selected_objectids = [
            int(raw_objectids[pos])
            for pos in selected_positions
            if not pd.isna(raw_objectids[pos])
        ]
        score = min(council_coverage, wfs_coverage)
        records.append(
            {
                "council_fid": int(row.council_fid),
                "council_label": int(row.LABEL) if hasattr(row, "LABEL") and not pd.isna(row.LABEL) else np.nan,
                "source_council": str(getattr(row, "source_council", "")),
                "anchor_wfs_fid": anchor_fid,
                "anchor_uprn_count": int(row.anchor_uprn_count),
                "anchor_polygon_count": 1,
                "anchor_kind": str(getattr(row, "anchor_kind", "")),
                "raw_source_count": source_count,
                "source_wfs_fids": _ids_text(selected_fids),
                "source_wfs_objectids": _ids_text(selected_objectids),
                "council_area": council_area,
                "selected_raw_area": selected_raw_area,
                "intersection_area": intersection_sum,
                "council_coverage_ratio": float(council_coverage),
                "wfs_coverage_ratio": float(wfs_coverage),
                "selection_score": float(score),
                "geometry": council_geom,
            }
        )
        raw_position_records.append(
            {
                "council_fid": int(row.council_fid),
                "raw_positions": tuple(int(v) for v in selected_positions),
                "raw_fids": tuple(selected_fids),
            }
        )

    if not records:
        raise RuntimeError("No council training candidates passed the overlap and anchor filters.")
    candidates = pd.DataFrame.from_records(records)
    _log(f"[INFO] Council candidates after 90% bidirectional coverage={len(candidates):,}")
    return candidates, raw_position_records


def _resolve_conflicts(
    candidates: pd.DataFrame,
    raw_position_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    _log("[INFO] Resolving overlapping council merge candidates")
    positions_by_council = {int(item["council_fid"]): item["raw_positions"] for item in raw_position_records}
    fids_by_council = {int(item["council_fid"]): item["raw_fids"] for item in raw_position_records}
    order = candidates.sort_values(
        ["selection_score", "council_coverage_ratio", "wfs_coverage_ratio", "raw_source_count"],
        ascending=[False, False, False, False],
    )
    used_positions: set[int] = set()
    accepted_ids: list[int] = []
    conflict_ids: list[int] = []
    for row in order.itertuples(index=False):
        council_fid = int(row.council_fid)
        raw_positions = set(positions_by_council[council_fid])
        if raw_positions & used_positions:
            conflict_ids.append(council_fid)
            continue
        accepted_ids.append(council_fid)
        used_positions |= raw_positions
    accepted = candidates[candidates["council_fid"].isin(accepted_ids)].copy()
    conflicts = candidates[candidates["council_fid"].isin(conflict_ids)].copy()
    accepted["conflict_status"] = "accepted"
    conflicts["conflict_status"] = "rejected_raw_overlap"
    _log(f"[INFO] Accepted council merges={len(accepted):,}; conflict_rejected={len(conflicts):,}")
    return accepted, conflicts, positions_by_council, fids_by_council


def _build_merged_only_layer(
    raw: gpd.GeoDataFrame,
    accepted: pd.DataFrame,
    positions_by_council: dict[int, tuple[int, ...]],
    fids_by_council: dict[int, tuple[int, ...]],
    *,
    slit_close_buffer: float,
    concavity_close_buffer: float,
    max_hull_gap_ratio: float,
    max_notch_index: float,
    max_concavity_close_area_ratio: float,
) -> gpd.GeoDataFrame:
    _log("[INFO] Building merged-only training layer")
    raw_geoms = np.asarray(raw.geometry.to_numpy(), dtype=object)
    accepted = accepted.sort_values("council_fid").copy()

    records: list[dict[str, Any]] = []
    component_id = 1
    rejected_concavity = 0
    for row in accepted.itertuples(index=False):
        council_fid = int(row.council_fid)
        raw_positions = positions_by_council[council_fid]
        raw_fids = fids_by_council[council_fid]
        geom = shapely.union_all(raw_geoms[list(raw_positions)])
        geom = shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom
        geom, holes_filled, fallback_area_delta, before_area = _repair_merged_geometry(
            geom,
            slit_close_buffer=float(slit_close_buffer),
        )
        concavity = _concavity_metrics(geom, close_buffer=float(concavity_close_buffer))
        is_concave = (
            concavity["shape_hull_gap_ratio"] > float(max_hull_gap_ratio)
            or concavity["shape_notch_index"] > float(max_notch_index)
            or concavity["shape_concavity_close_area_ratio"] > float(max_concavity_close_area_ratio)
        )
        if is_concave:
            rejected_concavity += 1
            continue
        records.append(
            {
                "train_component_id": component_id,
                "train_source": "council_merge",
                "source_council_fid": council_fid,
                "source_council_label": row.council_label,
                "source_council": row.source_council,
                "source_wfs_fids": row.source_wfs_fids,
                "source_wfs_objectids": row.source_wfs_objectids,
                "raw_source_count": int(row.raw_source_count),
                "anchor_polygon_count": int(row.anchor_polygon_count),
                "anchor_kind": str(row.anchor_kind),
                "anchor_wfs_fid": int(row.anchor_wfs_fid),
                "uprn_count": int(row.anchor_uprn_count),
                "council_coverage_ratio": float(row.council_coverage_ratio),
                "wfs_coverage_ratio": float(row.wfs_coverage_ratio),
                "selection_score": float(row.selection_score),
                "fallback_holes_filled": int(holes_filled),
                "fallback_slit_close_buffer": float(slit_close_buffer),
                "fallback_area_before": float(before_area),
                "fallback_area_delta": float(fallback_area_delta),
                "fallback_area_delta_ratio": float(fallback_area_delta / max(before_area, 1e-9)),
                **concavity,
                "Theme": pd.NA,
                "DescriptiveGroup": pd.NA,
                "DescriptiveTerm": pd.NA,
                "geometry": geom,
            }
        )
        component_id += 1

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=raw.crs)
    out.attrs["rejected_concavity"] = int(rejected_concavity)
    _log(
        "[INFO] Merged-only training layer rows="
        f"{len(out):,}; holes_filled_rows={int(out['fallback_holes_filled'].gt(0).sum()):,}; "
        f"area_delta_sum={float(out['fallback_area_delta'].sum()):.3f}; "
        f"rejected_concavity={int(rejected_concavity):,}"
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw-WFS council-merged training target.")
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--council-gpkg", default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--council-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--council-source-filter", default="Sheffield_City_Council")
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.90)
    parser.add_argument("--min-raw-pieces", type=int, default=2)
    parser.add_argument("--min-intersection-area", type=float, default=0.01)
    parser.add_argument("--raw-piece-min-inside-ratio", type=float, default=0.50)
    parser.add_argument("--council-piece-min-share", type=float, default=0.02)
    parser.add_argument(
        "--slit-close-buffer",
        type=float,
        default=0.25,
        help="Metres used for buffer/unbuffer closing of narrow knife-cut slits after hole filling.",
    )
    parser.add_argument(
        "--concavity-close-buffer",
        type=float,
        default=2.0,
        help="Metres used only to detect narrow exterior concavities; detected concave polygons are rejected.",
    )
    parser.add_argument("--max-hull-gap-ratio", type=float, default=0.03)
    parser.add_argument("--max-notch-index", type=float, default=0.005)
    parser.add_argument("--max-concavity-close-area-ratio", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_raw(Path(args.wfs_gpkg), str(args.wfs_layer))
    bbox = tuple(float(v) for v in raw.total_bounds)
    uprn = _read_uprn(Path(args.uprn_gpkg), str(args.uprn_layer), bbox, raw.crs)
    anchors = _anchor_polygons(raw, uprn)
    raw = raw.merge(
        anchors[["wfs_fid", "anchor_uprn_count"]],
        on="wfs_fid",
        how="left",
    )
    raw["anchor_uprn_count"] = raw["anchor_uprn_count"].fillna(0).astype("int64")

    council = _read_council(
        Path(args.council_gpkg),
        str(args.council_layer),
        bbox,
        raw.crs,
        str(args.council_source_filter),
    )
    council_one_anchor = _council_with_one_anchor(council, anchors)
    candidates, raw_position_records = _build_council_candidates(
        raw,
        council_one_anchor,
        min_overlap_ratio=float(args.min_overlap_ratio),
        min_raw_pieces=int(args.min_raw_pieces),
        min_intersection_area=float(args.min_intersection_area),
        raw_piece_min_inside_ratio=float(args.raw_piece_min_inside_ratio),
        council_piece_min_share=float(args.council_piece_min_share),
    )
    accepted, conflicts, positions_by_council, fids_by_council = _resolve_conflicts(
        candidates,
        raw_position_records,
    )
    merged_only = _build_merged_only_layer(
        raw,
        accepted,
        positions_by_council,
        fids_by_council,
        slit_close_buffer=float(args.slit_close_buffer),
        concavity_close_buffer=float(args.concavity_close_buffer),
        max_hull_gap_ratio=float(args.max_hull_gap_ratio),
        max_notch_index=float(args.max_notch_index),
        max_concavity_close_area_ratio=float(args.max_concavity_close_area_ratio),
    )

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    pyogrio.write_dataframe(merged_only, output_gpkg, layer=MERGED_ONLY_LAYER, driver="GPKG")

    summary = {
        "wfs_gpkg": str(args.wfs_gpkg),
        "wfs_layer": str(args.wfs_layer),
        "council_gpkg": str(args.council_gpkg),
        "council_layer": str(args.council_layer),
        "council_source_filter": str(args.council_source_filter),
        "uprn_gpkg": str(args.uprn_gpkg),
        "uprn_layer": str(args.uprn_layer),
        "output_gpkg": str(output_gpkg),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "min_raw_pieces": int(args.min_raw_pieces),
        "slit_close_buffer": float(args.slit_close_buffer),
        "concavity_close_buffer": float(args.concavity_close_buffer),
        "max_hull_gap_ratio": float(args.max_hull_gap_ratio),
        "max_notch_index": float(args.max_notch_index),
        "max_concavity_close_area_ratio": float(args.max_concavity_close_area_ratio),
        "raw_polygons": int(len(raw)),
        "anchor_definition": "raw WFS polygon has >=1 UPRN and is building or Theme contains Land",
        "anchor_polygons": int(len(anchors)),
        "anchor_polygon_kind_counts": {
            str(key): int(value)
            for key, value in anchors["_anchor_kind"].value_counts().sort_index().items()
        },
        "council_polygons_after_filter": int(len(council)),
        "council_one_anchor": int(len(council_one_anchor)),
        "council_candidates_90pct": int(len(candidates)),
        "accepted_council_merges": int(len(accepted)),
        "conflict_rejected_council_merges": int(len(conflicts)),
        "concavity_rejected_merges": int(merged_only.attrs.get("rejected_concavity", 0)),
        "output_layer": MERGED_ONLY_LAYER,
        "output_rows": int(len(merged_only)),
        "fallback_hole_rows": int(merged_only["fallback_holes_filled"].gt(0).sum()),
        "fallback_holes_filled": int(merged_only["fallback_holes_filled"].sum()),
        "fallback_area_delta_sum": float(merged_only["fallback_area_delta"].sum()),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Council training merge complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

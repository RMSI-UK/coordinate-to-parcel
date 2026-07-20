#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REPO_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_DIR / "wfs_merge_native_train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import box

from train_wfs_merged_council_anchor_group_model import (  # noqa: E402
    _combo_anchor_groups,
    _distances_from_anchor,
    _enumerate_anchor_groups,
    _ids_text,
    _load_adjacency,
    _parse_int_list,
    _parse_quota_specs,
    _quota_combo_anchor_groups,
    _role_for_node,
    _shape_metrics,
)
from train_wfs_raw_anchor_group_model import _add_uprn_counts  # noqa: E402


BASE_DIR = Path("/data/sheffield/spatial/base-map")
DEFAULT_WFS_CLEAN_GPKG = BASE_DIR / "sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_CLEAN_LAYER = "wfs_raw_clean"
DEFAULT_UPRN_GPKG = Path("/data/base-data/osopenuprn_202602.gpkg")
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_UPRN_FIELD = "UPRN"
DEFAULT_EDGE_CACHE = (
    BASE_DIR
    / "tmp/wfs_raw_anchor_group_model_completeness_v2_context_cache/"
    / "shared_edges_e455305190c051e0db7e7441.joblib"
)
DEFAULT_MODEL = (
    BASE_DIR
    / "tmp/wfs_merged_council_anchor_group_model_v2_full_quota128/"
    / "wfs_merged_council_anchor_group_model_v1.joblib"
)
DEFAULT_OUTPUT = (
    BASE_DIR
    / "tmp/wfs_merged_council_anchor_group_model_v2_full_quota128/"
    / "bbox_429000_384000_434000_389000_anchor_group_inference.gpkg"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in str(text).replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid bbox: {text!r}")
    return xmin, ymin, xmax, ymax


def _safe_int_series(frame: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="int64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("int64")


def _safe_float_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")


def _theme_text(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column in ["Theme", "DescriptiveGroup", "DescriptiveTerm", "raw_role"]:
        if column in frame.columns:
            parts.append(frame[column].fillna("").astype(str))
    if not parts:
        return pd.Series("", index=frame.index)
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep=" ")
    return out.str.lower()


def _make_valid(values: Any) -> Any:
    valid = shapely.is_valid(values)
    if bool(np.all(valid)):
        return values
    out = np.asarray(values, dtype=object).copy()
    bad = ~np.asarray(valid, dtype=bool)
    out[bad] = shapely.make_valid(out[bad])
    return out


def _read_raw_context(path: Path, layer: str, bbox_values: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading raw clean context: {path}:{layer}; bbox={bbox_values}")
    raw = pyogrio.read_dataframe(path, layer=layer, fid_as_index=True, bbox=bbox_values)
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    raw.index = raw.index.astype(int)
    raw["raw_clean_fid"] = raw.index.astype("int64")
    raw["raw_clean_attr_fid"] = _safe_int_series(raw, "clean_fid", -1)
    raw["source_fid"] = _safe_int_series(raw, "source_fid")
    raw.geometry = _make_valid(raw.geometry.to_numpy())
    raw["clean_area"] = _safe_float_series(raw, "clean_area")
    raw.loc[raw["clean_area"].le(0), "clean_area"] = raw.loc[raw["clean_area"].le(0)].geometry.area
    raw["clean_perimeter"] = _safe_float_series(raw, "clean_perimeter")
    raw.loc[raw["clean_perimeter"].le(0), "clean_perimeter"] = raw.loc[raw["clean_perimeter"].le(0)].geometry.length
    for column in ["clean_mrr_ratio", "clean_hull_gap_ratio", "clean_compactness", "raw_width_proxy", "raw_mrr_width", "gap_fill_area"]:
        raw[column] = _safe_float_series(raw, column)
    for column in ["is_polygon_hole_fill", "is_enclosed_gap_fill"]:
        raw[column] = _safe_int_series(raw, column)

    text = _theme_text(raw)
    raw["is_building_theme"] = text.str.contains("building", regex=False, na=False).astype("int64")
    raw["plot_eligible"] = (
        text.str.contains("building", regex=False, na=False)
        | text.str.contains("land", regex=False, na=False)
    ).astype("int64")
    raw["theme_role"] = np.select(
        [
            text.str.contains("building", regex=False, na=False),
            text.str.contains("land", regex=False, na=False),
            text.str.contains("road", regex=False, na=False)
            | text.str.contains("path", regex=False, na=False)
            | text.str.contains("track", regex=False, na=False),
        ],
        ["building", "land", "road"],
        default="other",
    )
    for column in ["raw_role", "Theme", "DescriptiveGroup", "DescriptiveTerm", "GmlID", "TOID"]:
        if column not in raw.columns:
            raw[column] = ""
        raw[column] = raw[column].fillna("").astype(str)
    _log(
        "[INFO] Raw context rows="
        f"{len(raw):,}; plot_eligible={int(raw['plot_eligible'].sum()):,}; "
        f"building={int(raw['is_building_theme'].sum()):,}"
    )
    return raw


def _prepare_input(raw: gpd.GeoDataFrame, *, uprn_gpkg: Path, uprn_layer: str, uprn_field: str) -> gpd.GeoDataFrame:
    wfs = raw.copy()
    if "clean_fid" in wfs.columns:
        wfs = wfs.drop(columns=["clean_fid"])
    wfs = wfs.rename(columns={"raw_clean_fid": "clean_fid"}).copy()
    wfs = _add_uprn_counts(
        wfs,
        uprn_gpkg=uprn_gpkg,
        uprn_layer=uprn_layer,
        uprn_id_field=uprn_field,
    )
    out = wfs.rename(columns={"clean_fid": "raw_clean_fid"}).copy()
    out["uprn_count"] = _safe_int_series(out, "uprn_count")
    out["has_uprn"] = out["uprn_count"].gt(0).astype("int64")
    out["zero_uprn_plot_eligible"] = (
        out["plot_eligible"].astype(int).eq(1) & out["uprn_count"].eq(0)
    ).astype("int64")
    out["is_building_uprn_anchor"] = (
        out["is_building_theme"].astype(int).eq(1) & out["uprn_count"].gt(0)
    ).astype("int64")
    out["is_building_label_anchor"] = 0
    out["is_nonanchor_uprn"] = (
        out["has_uprn"].eq(1) & out["is_building_uprn_anchor"].eq(0)
    ).astype("int64")
    out["objectid"] = _safe_int_series(out, "OBJECTID")
    out["feature_code"] = _safe_int_series(out, "FeatureCode")
    out["gml_id"] = out["GmlID"].fillna("").astype(str)
    out["toid"] = out["TOID"].fillna("").astype(str)
    out["theme"] = out["Theme"].fillna("").astype(str)
    out["desc_group"] = out["DescriptiveGroup"].fillna("").astype(str)
    out["desc_term"] = out["DescriptiveTerm"].fillna("").astype(str)
    return out


def _build_node_indexes(frame: gpd.GeoDataFrame) -> tuple[dict[int, dict[str, Any]], dict[int, Any]]:
    attrs: dict[int, dict[str, Any]] = {}
    geoms: dict[int, Any] = {}
    for row in frame.itertuples(index=False):
        fid = int(row.raw_clean_fid)
        attrs[fid] = {
            "raw_clean_fid": fid,
            "source_fid": int(row.source_fid),
            "clean_area": float(row.clean_area),
            "clean_perimeter": float(row.clean_perimeter),
            "clean_mrr_ratio": float(row.clean_mrr_ratio),
            "clean_hull_gap_ratio": float(row.clean_hull_gap_ratio),
            "clean_compactness": float(row.clean_compactness),
            "raw_width_proxy": float(row.raw_width_proxy),
            "raw_mrr_width": float(row.raw_mrr_width),
            "uprn_count": int(row.uprn_count),
            "has_uprn": int(row.has_uprn),
            "is_building_theme": int(row.is_building_theme),
            "zero_uprn_plot_eligible": int(row.zero_uprn_plot_eligible),
            "plot_eligible": int(row.plot_eligible),
            "is_polygon_hole_fill": int(row.is_polygon_hole_fill),
            "is_enclosed_gap_fill": int(row.is_enclosed_gap_fill),
            "gap_fill_area": float(row.gap_fill_area),
            "is_building_uprn_anchor": int(row.is_building_uprn_anchor),
            "is_nonanchor_uprn": int(row.is_nonanchor_uprn),
            "theme_role": str(row.theme_role or ""),
            "raw_role": str(row.raw_role or ""),
        }
        geoms[fid] = row.geometry
    return attrs, geoms


def _group_union(group: frozenset[int], geoms: dict[int, Any]) -> Any:
    values = [geoms[int(fid)] for fid in sorted(group) if int(fid) in geoms]
    if not values:
        return None
    geom = shapely.union_all(np.asarray(values, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _inference_candidate_features(
    *,
    anchor: int,
    candidate_group: frozenset[int],
    proposal_source: str,
    attrs: dict[int, dict[str, Any]],
    geoms: dict[int, Any],
    adjacency: dict[int, list[tuple[int, float]]],
    distance_cache: dict[int, int],
    shape_cache: dict[frozenset[int], dict[str, float]],
) -> dict[str, Any]:
    group = frozenset(int(v) for v in candidate_group)
    rows = [attrs[int(fid)] for fid in sorted(group) if int(fid) in attrs]
    areas = np.asarray([float(row["clean_area"]) for row in rows], dtype="float64")
    source_ids = {int(row["source_fid"]) for row in rows}
    roles = [_role_for_node(row) for row in rows]
    role_counts = {role: roles.count(role) for role in sorted(set(roles))}
    role_signature = "|".join(f"{role}:{role_counts[role]}" for role in sorted(role_counts))
    if group not in shape_cache:
        shape_cache[group] = _shape_metrics(_group_union(group, geoms))
    shape = shape_cache[group]

    candidate_area_sum = float(areas.sum()) if len(areas) else 0.0
    anchor_area = float(attrs.get(int(anchor), {}).get("clean_area", 0.0))
    internal_edges: list[float] = []
    anchor_shared = 0.0
    frontier_shared = 0.0
    frontier_ids: set[int] = set()
    for left in group:
        for right, shared in adjacency.get(int(left), ()):
            right = int(right)
            shared = float(shared)
            if right in group and int(left) < right:
                internal_edges.append(shared)
                if int(left) == int(anchor) or right == int(anchor):
                    anchor_shared += shared
            elif right not in group:
                frontier_shared += shared
                frontier_ids.add(right)
    frontier_rows = [attrs[int(fid)] for fid in frontier_ids if int(fid) in attrs]
    distances = [int(distance_cache.get(int(fid), 999)) for fid in group]
    building_uprn_anchor_count = int(sum(int(row.get("is_building_uprn_anchor", 0)) for row in rows))
    anchor_is_building_uprn = int(attrs.get(int(anchor), {}).get("is_building_uprn_anchor", 0))
    record: dict[str, Any] = {
        "anchor_raw_clean_fid": int(anchor),
        "proposal_source": str(proposal_source),
        "candidate_clean_fids": _ids_text(group),
        "candidate_source_fids": _ids_text(source_ids),
        "candidate_clean_count": int(len(group)),
        "candidate_source_count": int(len(source_ids)),
        "candidate_area_sum": candidate_area_sum,
        "candidate_area_union_to_sum": (float(shape["candidate_area_union"]) / candidate_area_sum) if candidate_area_sum else 0.0,
        "anchor_area": anchor_area,
        "added_area": max(candidate_area_sum - anchor_area, 0.0),
        "added_area_to_anchor": (max(candidate_area_sum - anchor_area, 0.0) / anchor_area) if anchor_area else 0.0,
        "largest_piece_area_ratio": (float(areas.max()) / candidate_area_sum) if len(areas) and candidate_area_sum else 0.0,
        "mean_piece_area": float(areas.mean()) if len(areas) else 0.0,
        "std_piece_area": float(areas.std()) if len(areas) else 0.0,
        "building_count": int(role_counts.get("building", 0)),
        "land_count": int(role_counts.get("land", 0)),
        "gapfill_count": int(role_counts.get("gapfill", 0)),
        "road_count": int(role_counts.get("road", 0)),
        "other_role_count": int(role_counts.get("other", 0)),
        "role_signature": role_signature,
        "plot_eligible_count": int(sum(int(row.get("plot_eligible", 0)) for row in rows)),
        "zero_uprn_plot_eligible_count": int(sum(int(row.get("zero_uprn_plot_eligible", 0)) for row in rows)),
        "uprn_count_sum": int(sum(int(row.get("uprn_count", 0)) for row in rows)),
        "has_uprn_count": int(sum(int(row.get("has_uprn", 0)) for row in rows)),
        "building_uprn_anchor_count": building_uprn_anchor_count,
        "label_anchor_count": int(int(anchor) in group),
        "nonseed_building_uprn_anchor_count": int(building_uprn_anchor_count - anchor_is_building_uprn),
        "nonanchor_uprn_count": int(sum(int(row.get("is_nonanchor_uprn", 0)) for row in rows)),
        "polygon_hole_fill_count": int(sum(int(row.get("is_polygon_hole_fill", 0)) for row in rows)),
        "enclosed_gap_fill_count": int(sum(int(row.get("is_enclosed_gap_fill", 0)) for row in rows)),
        "gap_fill_area_sum": float(sum(float(row.get("gap_fill_area", 0.0)) for row in rows)),
        "internal_shared_edge_sum": float(sum(internal_edges)),
        "internal_shared_edge_max": float(max(internal_edges)) if internal_edges else 0.0,
        "internal_shared_edge_min": float(min(internal_edges)) if internal_edges else 0.0,
        "internal_shared_edge_count": int(len(internal_edges)),
        "anchor_shared_edge_sum": float(anchor_shared),
        "frontier_shared_edge_sum": float(frontier_shared),
        "frontier_plot_eligible_count": int(sum(int(row.get("plot_eligible", 0)) for row in frontier_rows)),
        "frontier_zero_uprn_plot_eligible_count": int(sum(int(row.get("zero_uprn_plot_eligible", 0)) for row in frontier_rows)),
        "frontier_building_uprn_anchor_count": int(sum(int(row.get("is_building_uprn_anchor", 0)) for row in frontier_rows)),
        "frontier_nonanchor_uprn_count": int(sum(int(row.get("is_nonanchor_uprn", 0)) for row in frontier_rows)),
        "max_graph_distance_from_anchor": int(max(distances)) if distances else 0,
        "mean_graph_distance_from_anchor": float(np.mean(distances)) if distances else 0.0,
    }
    record.update(shape)
    return record


def _cheap_group_score(group: frozenset[int], *, anchor: int, attrs: dict[int, dict[str, Any]], adjacency: dict[int, list[tuple[int, float]]], distances: dict[int, int]) -> tuple[float, int, str]:
    area = sum(float(attrs.get(int(fid), {}).get("clean_area", 0.0)) for fid in group)
    internal = 0.0
    anchor_shared = 0.0
    for left in group:
        for right, shared in adjacency.get(int(left), ()):
            if int(right) in group and int(left) < int(right):
                internal += float(shared)
                if int(left) == int(anchor) or int(right) == int(anchor):
                    anchor_shared += float(shared)
    dist_sum = sum(int(distances.get(int(fid), 99)) for fid in group)
    score = dist_sum * 20.0 - internal * 1.5 - anchor_shared * 0.5 + math.log1p(area) * 0.05
    return float(score), int(len(group)), _ids_text(group)


def _coerce_int_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = [int(v) for v in value]
        if not values:
            raise ValueError("Expected at least one integer value.")
        return values
    return _parse_int_list(value)


def _coerce_quota_specs(value: Any) -> list[dict[int, int]]:
    if isinstance(value, (list, tuple)):
        specs: list[dict[int, int]] = []
        for spec in value:
            if not isinstance(spec, dict):
                return _parse_quota_specs(str(value))
            specs.append({int(key): int(val) for key, val in spec.items()})
        if specs:
            return specs
    return _parse_quota_specs(str(value))


def _candidate_groups_for_anchor(
    *,
    anchor: int,
    params: dict[str, Any],
    attrs: dict[int, dict[str, Any]],
    adjacency: dict[int, list[tuple[int, float]]],
) -> tuple[dict[frozenset[int], str], dict[int, int]]:
    area_by_clean = {int(fid): float(row.get("clean_area", 0.0)) for fid, row in attrs.items()}
    distances = _distances_from_anchor(anchor, adjacency, int(params.get("max_graph_depth", 3)))
    groups: dict[frozenset[int], str] = {frozenset({int(anchor)}): "anchor_only"}
    for group in _enumerate_anchor_groups(
        anchor=anchor,
        adjacency=adjacency,
        area_by_clean=area_by_clean,
        max_group_size=int(params.get("max_group_size", 12)),
        max_group_area=float(params.get("max_group_area", 20000.0)),
        per_label_limit=int(params.get("enum_per_label", 28)),
    ):
        groups.setdefault(group, "enumerated")
    local_ns = _coerce_int_list(params.get("combo_local_n", [12, 14]))
    combo_groups = _combo_anchor_groups(
        anchor=anchor,
        distances=distances,
        adjacency=adjacency,
        area_by_clean=area_by_clean,
        max_group_size=int(params.get("max_group_size", 12)),
        max_group_area=float(params.get("max_group_area", 20000.0)),
        local_n=int(local_ns[0]),
        max_extra=int(params.get("combo_max_extra", 7)),
        per_label_limit=int(params.get("combo_per_label", 500)),
    )
    for group in combo_groups:
        groups.setdefault(group, "combo")
    quota_specs = params.get("combo_extra_quotas")
    if not quota_specs:
        quota_specs = _parse_quota_specs("1:16,2:32,3:48,4:56,5:48,6:16,7:4;1:10,2:16,3:24,4:28,5:20,6:6,7:2")
    else:
        quota_specs = _coerce_quota_specs(quota_specs)
    for group in _quota_combo_anchor_groups(
        anchor=anchor,
        distances=distances,
        adjacency=adjacency,
        area_by_clean=area_by_clean,
        max_group_size=int(params.get("max_group_size", 12)),
        max_group_area=float(params.get("max_group_area", 20000.0)),
        local_ns=local_ns,
        quota_specs=quota_specs,
        max_extra=int(params.get("combo_max_extra", 7)),
    ):
        groups.setdefault(group, "combo_quota")
    return groups, distances


def _select_non_overlapping(top1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_rows = []
    conflict_rows = []
    used: set[int] = set()
    order = top1.sort_values(["proba", "candidate_clean_count", "anchor_raw_clean_fid"], ascending=[False, False, True])
    for row in order.itertuples(index=False):
        group = {int(part) for part in str(row.candidate_clean_fids).split("|") if part}
        if group & used:
            conflict_rows.append(row._asdict())
            continue
        accepted_rows.append(row._asdict())
        used |= group
    return pd.DataFrame(accepted_rows), pd.DataFrame(conflict_rows)


def _records_to_gdf(records: list[dict[str, Any]], geoms: dict[int, Any], crs: Any) -> gpd.GeoDataFrame:
    out_records = []
    for idx, record in enumerate(records, start=1):
        group = frozenset(int(part) for part in str(record["candidate_clean_fids"]).split("|") if part)
        geom = _group_union(group, geoms)
        if geom is None or geom.is_empty:
            continue
        out = dict(record)
        out["pred_id"] = idx
        out["geometry"] = geom
        out_records.append(out)
    return gpd.GeoDataFrame(out_records, geometry="geometry", crs=crs)


def _write_layer(frame: gpd.GeoDataFrame, output: Path, layer: str) -> None:
    frame = frame.copy()
    geometry_column = frame.geometry.name if isinstance(frame, gpd.GeoDataFrame) else "geometry"
    renamed: dict[str, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        if column == geometry_column:
            continue
        base = str(column)
        if base.lower() in {"fid", "ogc_fid"}:
            base = f"{base}_attr"
        candidate = base
        suffix = 2
        while candidate.lower() in seen or candidate == geometry_column:
            candidate = f"{base}_{suffix}"
            suffix += 1
        renamed[str(column)] = candidate
        seen.add(candidate.lower())
    if renamed:
        frame = frame.rename(columns=renamed)
    if frame.empty:
        frame = gpd.GeoDataFrame(frame.copy(), geometry=[], crs=frame.crs)
    pyogrio.write_dataframe(frame, output, layer=layer, driver="GPKG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply anchor-feature WFS group model to a bbox.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--wfs-clean-gpkg", default=str(DEFAULT_WFS_CLEAN_GPKG))
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=str(DEFAULT_UPRN_GPKG))
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-field", default=DEFAULT_UPRN_FIELD)
    parser.add_argument("--edge-cache", default=str(DEFAULT_EDGE_CACHE))
    parser.add_argument("--bbox", default="429000,384000,434000,389000")
    parser.add_argument("--context-buffer", type=float, default=150.0)
    parser.add_argument("--output-gpkg", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-candidates-per-anchor", type=int, default=128)
    parser.add_argument("--top-debug-candidates", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output = Path(args.output_gpkg)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    bbox_values = _parse_bbox(str(args.bbox))
    target_box = box(*bbox_values)
    buffer = float(args.context_buffer)
    context_bbox = (
        bbox_values[0] - buffer,
        bbox_values[1] - buffer,
        bbox_values[2] + buffer,
        bbox_values[3] + buffer,
    )
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = list(payload["feature_columns"])
    params = dict(payload.get("params", {}))
    params["max_candidates_per_label"] = int(args.top_candidates_per_anchor)

    raw = _read_raw_context(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), context_bbox)
    inputs = _prepare_input(
        raw,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_field=str(args.uprn_field),
    )
    nodes = inputs[inputs["plot_eligible"].astype(int).eq(1)].copy()
    attrs, geoms = _build_node_indexes(nodes)
    adjacency, _shared_by_pair = _load_adjacency(Path(args.edge_cache), set(attrs), top_neighbors=int(params.get("top_neighbors", 14)))

    anchor_mask = nodes["is_building_uprn_anchor"].astype(int).eq(1)
    anchor_points = shapely.point_on_surface(nodes.loc[anchor_mask].geometry.to_numpy())
    anchor_in_bbox = shapely.intersects(anchor_points, target_box)
    anchors = nodes.loc[anchor_mask].iloc[np.asarray(anchor_in_bbox, dtype=bool)].copy()
    anchors = anchors.sort_values("raw_clean_fid").reset_index(drop=True)
    _log(f"[INFO] Anchors in target bbox={len(anchors):,}")

    records: list[dict[str, Any]] = []
    top1_records: list[dict[str, Any]] = []
    shape_cache: dict[frozenset[int], dict[str, float]] = {}
    for offset, row in enumerate(anchors.itertuples(index=False), start=1):
        if offset == 1 or offset % 500 == 0:
            _log(f"[INFO] Scoring anchors {offset:,}/{len(anchors):,}; candidate_rows={len(records):,}")
        anchor = int(row.raw_clean_fid)
        groups, distances = _candidate_groups_for_anchor(anchor=anchor, params=params, attrs=attrs, adjacency=adjacency)
        ordered = sorted(
            groups.items(),
            key=lambda item: _cheap_group_score(item[0], anchor=anchor, attrs=attrs, adjacency=adjacency, distances=distances),
        )[: int(args.top_candidates_per_anchor)]
        anchor_group = frozenset({int(anchor)})
        if all(group != anchor_group for group, _source in ordered):
            ordered.append((anchor_group, groups.get(anchor_group, "anchor_only")))
        anchor_records = [
            _inference_candidate_features(
                anchor=anchor,
                candidate_group=group,
                proposal_source=source,
                attrs=attrs,
                geoms=geoms,
                adjacency=adjacency,
                distance_cache=distances,
                shape_cache=shape_cache,
            )
            for group, source in ordered
        ]
        if not anchor_records:
            continue
        frame = pd.DataFrame.from_records(anchor_records)
        for column in feature_cols:
            if column not in frame.columns:
                frame[column] = np.nan
        proba = model.predict_proba(frame[feature_cols])[:, 1]
        frame["proba"] = proba.astype("float64")
        frame["candidate_rank"] = np.arange(1, len(frame) + 1, dtype="int64")
        frame["anchor_source_fid"] = int(row.source_fid)
        frame["anchor_uprn_count"] = int(row.uprn_count)
        frame = frame.sort_values(["proba", "candidate_rank"], ascending=[False, True]).reset_index(drop=True)
        frame["model_rank"] = np.arange(1, len(frame) + 1, dtype="int64")
        model_top = frame.iloc[0].to_dict()
        hole_count = pd.to_numeric(
            frame.get("candidate_interior_hole_count", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        no_hole = frame[hole_count.le(0.0)]
        if not no_hole.empty:
            selected = no_hole.iloc[0].to_dict()
            selector_reason = "model_top_nohole" if int(selected.get("model_rank", 1)) == 1 else "nohole_residual_selector"
        else:
            selected = dict(model_top)
            selector_reason = "no_nohole_candidate"
        selected["top1_candidate_rank"] = int(selected.get("candidate_rank", 1))
        selected["selected_model_rank"] = int(selected.get("model_rank", 1))
        selected["selector_reason"] = selector_reason
        selected["model_top_proba"] = float(model_top.get("proba", np.nan))
        selected["model_top_candidate_clean_fids"] = str(model_top.get("candidate_clean_fids", ""))
        selected["model_top_candidate_rank"] = int(model_top.get("candidate_rank", 1))
        selected["model_top_interior_hole_count"] = float(model_top.get("candidate_interior_hole_count", 0.0) or 0.0)
        selected["model_top_interior_hole_area"] = float(model_top.get("candidate_interior_hole_area", 0.0) or 0.0)
        top1_records.append(selected)
        records.extend(frame.head(3).to_dict("records"))

    top1 = pd.DataFrame.from_records(top1_records)
    if top1.empty:
        raise RuntimeError("No top1 candidates were scored.")
    accepted, conflicts = _select_non_overlapping(top1)
    _log(f"[INFO] Top1 rows={len(top1):,}; accepted_nonoverlap={len(accepted):,}; conflicts={len(conflicts):,}")

    predicted = _records_to_gdf(accepted.to_dict("records"), geoms, nodes.crs)
    top1_gdf = _records_to_gdf(top1.to_dict("records"), geoms, nodes.crs)
    debug = pd.DataFrame.from_records(records).sort_values(["proba"], ascending=False).head(int(args.top_debug_candidates))
    debug_gdf = _records_to_gdf(debug.to_dict("records"), geoms, nodes.crs) if not debug.empty else gpd.GeoDataFrame(geometry=[], crs=nodes.crs)

    anchor_layer = anchors.copy()
    anchor_layer["geometry"] = shapely.point_on_surface(anchor_layer.geometry.to_numpy())
    anchor_layer = gpd.GeoDataFrame(anchor_layer, geometry="geometry", crs=nodes.crs)

    used_ids: set[int] = set()
    for text in accepted.get("candidate_clean_fids", pd.Series(dtype=str)).astype(str):
        used_ids.update(int(part) for part in text.split("|") if part)
    leftover = nodes[~nodes["raw_clean_fid"].astype(int).isin(used_ids)].copy()
    leftover = leftover[shapely.intersects(leftover.geometry.to_numpy(), target_box)].copy()

    _write_layer(predicted, output, "predicted_parcels")
    _write_layer(top1_gdf, output, "top1_candidates_all")
    _write_layer(anchor_layer, output, "anchor_points")
    _write_layer(debug_gdf, output, "top_scored_candidates_debug")
    _write_layer(leftover, output, "leftover_raw_clean")

    summary = {
        "model": str(model_path),
        "output_gpkg": str(output),
        "bbox": list(bbox_values),
        "context_bbox": list(context_bbox),
        "raw_context_rows": int(len(raw)),
        "plot_eligible_context_rows": int(len(nodes)),
        "anchors_in_bbox": int(len(anchors)),
        "top1_rows": int(len(top1)),
        "accepted_nonoverlap": int(len(accepted)),
        "conflict_top1_rows": int(len(conflicts)),
        "predicted_rows": int(len(predicted)),
        "leftover_raw_clean_rows": int(len(leftover)),
        "top1_proba_min": float(top1["proba"].min()),
        "top1_proba_median": float(top1["proba"].median()),
        "top1_proba_mean": float(top1["proba"].mean()),
        "layers": [
            "predicted_parcels",
            "top1_candidates_all",
            "anchor_points",
            "top_scored_candidates_debug",
            "leftover_raw_clean",
        ],
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Bbox anchor-group inference complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

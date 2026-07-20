#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from apply_wfs_merge_completion_model import _read_optional_layer, _source_reference_values, _write_layer
from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/"
    "03_operation_pruned_only.gpkg"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/"
    "03b_local_mode_split.gpkg"
)
PASSTHROUGH_DEBUG_LAYERS = [
    "prune_removed_sources",
    "prune_candidate_debug",
    "zero_uprn_attachment_candidates",
    "neighborhood_overmerge_split_components",
    "neighborhood_overmerge_split_removed_edges",
    "neighborhood_overmerge_split_results",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _role(theme: object) -> str:
    text = str(theme or "").lower()
    if "building" in text:
        return "building"
    if "land" in text:
        return "land"
    return "other"


def _mrr_orientation_deg(geom) -> float:
    if geom is None or shapely.is_empty(geom):
        return 0.0
    mrr = shapely.minimum_rotated_rectangle(geom)
    if not hasattr(mrr, "exterior"):
        return 0.0
    coords = list(mrr.exterior.coords)
    lengths: list[float] = []
    angles: list[float] = []
    for start, end in zip(coords, coords[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        lengths.append(length)
        angles.append(math.degrees(math.atan2(dy, dx)) % 180.0)
    if not lengths:
        return 0.0
    return float(angles[int(np.argmax(lengths))] % 90.0)


def _angle_delta_deg(a: float, b: float) -> float:
    delta = abs((float(a) - float(b)) % 90.0)
    return float(min(delta, 90.0 - delta))


def _area_mode(values: list[float]) -> tuple[float, int]:
    cleaned = [float(v) for v in values if np.isfinite(float(v)) and float(v) > 0.0]
    if not cleaned:
        return float("nan"), 0
    median = float(np.median(cleaned))
    bin_width = max(5.0, median * 0.08)
    bins = [int(round(v / bin_width)) for v in cleaned]
    mode_bin, support = Counter(bins).most_common(1)[0]
    in_bin = [v for v, b in zip(cleaned, bins) if b == mode_bin]
    return float(np.median(in_bin)), int(support)


def _local_mode_features(
    predicted: gpd.GeoDataFrame,
    component_index: dict[int, int],
    spatial_index,
    comp_id: int,
    geom,
    *,
    radius: float,
    orientation_tolerance: float,
) -> dict[str, float]:
    target_orientation = _mrr_orientation_deg(geom)
    idx = spatial_index.query(shapely.buffer(geom, float(radius)), predicate="intersects")
    if len(idx) == 0:
        return _empty_local_features(target_orientation)

    component_pos = component_index.get(int(comp_id), -1)
    idx = [int(i) for i in idx if int(i) != component_pos]
    if not idx:
        return _empty_local_features(target_orientation)

    local = predicted.iloc[idx].copy()
    local_geom = local.geometry.array
    distance = shapely.distance(local_geom, geom)
    local = local[distance <= float(radius)].copy()
    if local.empty:
        return _empty_local_features(target_orientation)

    orientations = np.asarray([_mrr_orientation_deg(g) for g in local.geometry.array], dtype="float64")
    deltas = np.asarray([_angle_delta_deg(o, target_orientation) for o in orientations], dtype="float64")
    uprn = local["pred_uprn_count"].fillna(0).astype(int).to_numpy()
    area = local["pred_area"].fillna(local.geometry.area).astype(float).to_numpy()
    regularity = local["pred_regularity_score"].fillna(0).astype(float).to_numpy()
    mrr_ratio = local["pred_mrr_ratio"].fillna(0).astype(float).to_numpy()
    hull_gap = local["pred_hull_gap_ratio"].fillna(1).astype(float).to_numpy()
    regular_one = (uprn == 1) & (regularity >= 0.90) & (mrr_ratio >= 0.88) & (hull_gap <= 0.12)
    same_orientation = deltas <= float(orientation_tolerance)
    same_pattern = regular_one & same_orientation

    same_areas = area[same_pattern]
    regular_areas = area[regular_one]
    mode_same, mode_same_support = _area_mode(same_areas.tolist())
    mode_regular, mode_regular_support = _area_mode(regular_areas.tolist())
    median_same = float(np.median(same_areas)) if len(same_areas) else float("nan")
    median_regular = float(np.median(regular_areas)) if len(regular_areas) else float("nan")

    if len(same_areas):
        q25, q75 = np.percentile(same_areas, [25, 75])
        iqr_same = float(q75 - q25)
    else:
        iqr_same = float("nan")

    return {
        "local_neighbor_count": float(len(local)),
        "local_regular_one_uprn_count": float(np.sum(regular_one)),
        "local_same_orientation_one_uprn_count": float(np.sum(same_pattern)),
        "local_mode_area": float(mode_same if np.isfinite(mode_same) else mode_regular),
        "local_mode_area_support": float(mode_same_support if mode_same_support else mode_regular_support),
        "local_median_area": float(median_same if np.isfinite(median_same) else median_regular),
        "local_same_pattern_area_iqr": float(iqr_same if np.isfinite(iqr_same) else 0.0),
        "local_target_orientation_deg": float(target_orientation),
        "local_same_pattern_ratio": _safe_ratio(float(np.sum(same_pattern)), float(len(local))),
    }


def _empty_local_features(target_orientation: float) -> dict[str, float]:
    return {
        "local_neighbor_count": 0.0,
        "local_regular_one_uprn_count": 0.0,
        "local_same_orientation_one_uprn_count": 0.0,
        "local_mode_area": float("nan"),
        "local_mode_area_support": 0.0,
        "local_median_area": float("nan"),
        "local_same_pattern_area_iqr": 0.0,
        "local_target_orientation_deg": float(target_orientation),
        "local_same_pattern_ratio": 0.0,
    }


def _build_source_adjacency(group: gpd.GeoDataFrame, *, min_shared_edge: float) -> dict[int, set[int]]:
    source_ids = group["source_fid"].astype(int).tolist()
    geom_by_source = dict(zip(group["source_fid"].astype(int), group.geometry))
    adjacency: dict[int, set[int]] = {int(source_id): set() for source_id in source_ids}
    for left, right in itertools.combinations(source_ids, 2):
        shared = float(
            shapely.length(
                shapely.intersection(
                    shapely.boundary(geom_by_source[int(left)]),
                    shapely.boundary(geom_by_source[int(right)]),
                )
            )
        )
        if shared >= float(min_shared_edge):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    return adjacency


def _is_connected(source_ids: list[int], adjacency: dict[int, set[int]]) -> bool:
    if len(source_ids) <= 1:
        return True
    allowed = {int(v) for v in source_ids}
    seen = {int(source_ids[0])}
    stack = [int(source_ids[0])]
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == allowed


def _group_source_uprn(group: gpd.GeoDataFrame) -> int:
    if "source_uprn_count" in group.columns:
        return int(group["source_uprn_count"].fillna(0).astype(int).sum())
    if "uprn_count" in group.columns:
        return int(group["uprn_count"].fillna(0).astype(int).sum())
    return 0


def _score_partition(
    groups: list[list[int]],
    source_by_fid: gpd.GeoDataFrame,
    adjacency: dict[int, set[int]],
    local_mode_area: float,
    target_orientation: float,
) -> tuple[float, dict[str, float], list[Any]]:
    group_metrics: list[dict[str, float]] = []
    group_geoms: list[Any] = []
    area_errors: list[float] = []
    regularity_values: list[float] = []
    hull_gaps: list[float] = []
    orientation_deltas: list[float] = []
    connected_values: list[int] = []
    uprn_values: list[int] = []

    for source_ids in groups:
        rows = source_by_fid.loc[source_ids]
        geom = shapely.union_all(rows.geometry.array)
        metrics = _shape_metrics(geom)
        orientation_delta = _angle_delta_deg(_mrr_orientation_deg(geom), target_orientation)
        uprn = _group_source_uprn(rows)
        connected = int(_is_connected([int(v) for v in source_ids], adjacency))
        area_ratio = _safe_ratio(float(metrics["area"]), local_mode_area)
        area_error = abs(math.log(max(area_ratio, 1e-6)))
        group_metrics.append(
            {
                "area": float(metrics["area"]),
                "area_to_local_mode": float(area_ratio),
                "regularity_score": float(metrics["regularity_score"]),
                "mrr_ratio": float(metrics["mrr_ratio"]),
                "hull_gap_ratio": float(metrics["hull_gap_ratio"]),
                "orientation_delta_deg": float(orientation_delta),
                "uprn_count": float(uprn),
                "connected": float(connected),
            }
        )
        group_geoms.append(geom)
        area_errors.append(float(area_error))
        regularity_values.append(float(metrics["regularity_score"]))
        hull_gaps.append(float(metrics["hull_gap_ratio"]))
        orientation_deltas.append(float(orientation_delta))
        connected_values.append(int(connected))
        uprn_values.append(int(uprn))

    mean_area_error = float(np.mean(area_errors)) if area_errors else 9.0
    max_area_error = float(np.max(area_errors)) if area_errors else 9.0
    mean_regularity = float(np.mean(regularity_values)) if regularity_values else 0.0
    min_regularity = float(np.min(regularity_values)) if regularity_values else 0.0
    max_hull_gap = float(np.max(hull_gaps)) if hull_gaps else 1.0
    mean_orientation_delta = float(np.mean(orientation_deltas)) if orientation_deltas else 45.0
    all_connected = int(all(connected_values)) if connected_values else 0
    one_uprn_group_ratio = _safe_ratio(float(sum(1 for value in uprn_values if value == 1)), float(len(uprn_values)))
    max_group_uprn_count = float(max(uprn_values or [0]))
    mean_uprn_error_to_one = float(np.mean([abs(value - 1) for value in uprn_values])) if uprn_values else 9.0

    area_fit_score = max(0.0, 1.0 - mean_area_error / 0.55)
    regularity_score = max(0.0, min(mean_regularity, 1.0))
    orientation_score = max(0.0, 1.0 - mean_orientation_delta / 25.0)
    hull_score = max(0.0, 1.0 - max_hull_gap / 0.18)
    uprn_score = max(0.0, 1.0 - mean_uprn_error_to_one / 2.0)
    score = (
        0.35 * area_fit_score
        + 0.25 * regularity_score
        + 0.15 * orientation_score
        + 0.10 * hull_score
        + 0.10 * float(all_connected)
        + 0.05 * uprn_score
    )
    summary = {
        "split_score": float(score),
        "split_group_count": float(len(groups)),
        "split_mean_area_error": mean_area_error,
        "split_max_area_error": max_area_error,
        "split_mean_regularity": mean_regularity,
        "split_min_regularity": min_regularity,
        "split_max_hull_gap": max_hull_gap,
        "split_mean_orientation_delta": mean_orientation_delta,
        "split_all_groups_connected": float(all_connected),
        "split_one_uprn_group_ratio": float(one_uprn_group_ratio),
        "split_max_group_uprn_count": float(max_group_uprn_count),
        "split_mean_uprn_error_to_one": mean_uprn_error_to_one,
        "split_min_area_to_local_mode": float(min([m["area_to_local_mode"] for m in group_metrics] or [0.0])),
        "split_max_area_to_local_mode": float(max([m["area_to_local_mode"] for m in group_metrics] or [0.0])),
    }
    return float(score), summary, group_geoms


def _enumerate_split_candidate(
    component: pd.Series,
    group: gpd.GeoDataFrame,
    local: dict[str, float],
    *,
    max_seed_count: int,
    max_zero_source_count: int,
    min_shared_edge: float,
) -> tuple[dict[str, Any] | None, list[list[int]]]:
    local_mode_area = float(local.get("local_mode_area", float("nan")))
    if not np.isfinite(local_mode_area) or local_mode_area <= 0.0:
        return None, []

    group = group.copy()
    group["source_role_for_split"] = group["Theme"].map(_role)
    source_uprn = group["source_uprn_count"].fillna(group.get("uprn_count", 0)).astype(int)
    seed_rows = group[source_uprn.eq(1)].copy()
    if len(seed_rows) < 2 or len(seed_rows) > int(max_seed_count):
        return None, []
    if int(source_uprn[source_uprn.gt(1)].sum()) > 0:
        return None, []

    zero_rows = group[source_uprn.eq(0)].copy()
    if len(zero_rows) > int(max_zero_source_count):
        return None, []

    seed_ids = seed_rows["source_fid"].astype(int).tolist()
    zero_ids = zero_rows["source_fid"].astype(int).tolist()
    component_area = float(component["pred_area"])
    expected_split_count = int(round(_safe_ratio(component_area, local_mode_area)))
    expected_split_count = max(2, min(expected_split_count, int(max_seed_count), len(seed_ids)))
    source_by_fid = group.set_index(group["source_fid"].astype(int), drop=False)
    adjacency = _build_source_adjacency(group, min_shared_edge=float(min_shared_edge))
    target_orientation = float(local["local_target_orientation_deg"])

    best_record: dict[str, Any] | None = None
    best_groups: list[list[int]] = []
    if expected_split_count == len(seed_ids):
        seed_assignments = [tuple(range(len(seed_ids)))]
    else:
        seed_assignments = []
        for assignment in itertools.product(range(expected_split_count), repeat=len(seed_ids)):
            if int(assignment[0]) != 0:
                continue
            if len(set(assignment)) != expected_split_count:
                continue
            seed_assignments.append(tuple(int(v) for v in assignment))

    for seed_assignment in seed_assignments:
        for zero_assignment in itertools.product(range(expected_split_count), repeat=len(zero_ids)):
            groups = [[] for _ in range(expected_split_count)]
            for seed_id, group_index in zip(seed_ids, seed_assignment):
                groups[int(group_index)].append(int(seed_id))
            for zero_id, group_index in zip(zero_ids, zero_assignment):
                groups[int(group_index)].append(int(zero_id))
            if any(not values for values in groups):
                continue
            groups = [sorted(values) for values in groups]
            score, summary, _group_geoms = _score_partition(
                groups,
                source_by_fid,
                adjacency,
                local_mode_area,
                target_orientation,
            )
            if not int(summary["split_all_groups_connected"]):
                continue
            allowed_max_uprn = max(1.0, math.ceil(float(component["pred_uprn_count"]) / float(expected_split_count)))
            if float(summary["split_max_group_uprn_count"]) > allowed_max_uprn:
                continue
            if float(summary["split_min_area_to_local_mode"]) < 0.35:
                continue
            if float(summary["split_max_area_to_local_mode"]) > 1.85:
                continue
            if float(summary["split_min_regularity"]) < 0.72:
                continue
            record = {
                "component_id": int(component["pred_component_id"]),
                "component_source_count": int(component["source_count"]),
                "component_uprn_count": int(component["pred_uprn_count"]),
                "component_area": float(component["pred_area"]),
                "component_regularity_score": float(component["pred_regularity_score"]),
                "component_mrr_ratio": float(component["pred_mrr_ratio"]),
                "component_hull_gap_ratio": float(component["pred_hull_gap_ratio"]),
                "component_area_to_local_mode": _safe_ratio(float(component["pred_area"]), local_mode_area),
                "expected_split_count_from_area": float(expected_split_count),
                "seed_count": int(len(seed_ids)),
                "zero_source_count": int(len(zero_ids)),
                "seed_source_fids": "|".join(str(v) for v in seed_ids),
                "zero_source_fids": "|".join(str(v) for v in zero_ids),
                "split_groups": json.dumps(groups),
                "geometry": component.geometry,
            }
            record.update(local)
            record.update(summary)
            if best_record is None or float(record["split_score"]) > float(best_record["split_score"]):
                best_record = record
                best_groups = groups
    return best_record, best_groups


def build_local_mode_split_candidates(
    predicted: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    *,
    require_possible_false_positive: bool,
    radius: float,
    orientation_tolerance: float,
    min_local_same_pattern_count: int,
    min_local_mode_support: int,
    min_component_area_to_mode: float,
    max_component_area_to_mode: float,
    max_component_area: float,
    max_component_source_count: int,
    max_seed_count: int,
    max_zero_source_count: int,
    min_shared_edge: float,
) -> tuple[gpd.GeoDataFrame, dict[int, list[list[int]]]]:
    predicted = predicted.copy().reset_index(drop=True)
    component_index = dict(zip(predicted["pred_component_id"].astype(int), predicted.index.astype(int)))
    spatial_index = predicted.sindex
    source_groups = {
        int(comp_id): group.copy()
        for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=False)
    }
    records: list[dict[str, Any]] = []
    groups_by_component: dict[int, list[list[int]]] = {}

    candidate_mask = (
        predicted["pred_uprn_count"].fillna(0).astype(int).ge(2)
        & predicted["source_count"].fillna(0).astype(int).le(int(max_component_source_count))
        & predicted["pred_area"].fillna(0).astype(float).le(float(max_component_area))
    )
    if bool(require_possible_false_positive) and "possible_false_positive_cluster" in predicted.columns:
        candidate_mask = candidate_mask & predicted["possible_false_positive_cluster"].fillna(0).astype(int).eq(1)
    candidate_predicted = predicted[candidate_mask].copy()

    for row in candidate_predicted.itertuples(index=False):
        comp_id = int(row.pred_component_id)
        group = source_groups.get(comp_id)
        if group is None or group.empty:
            continue
        source_uprn = group["source_uprn_count"].fillna(group.get("uprn_count", 0)).astype(int)
        seed_count = int(source_uprn.eq(1).sum())
        zero_count = int(source_uprn.eq(0).sum())
        if seed_count < 2 or seed_count > int(max_seed_count):
            continue
        if zero_count > int(max_zero_source_count):
            continue
        if int(source_uprn[source_uprn.gt(1)].sum()) > 0:
            continue
        local = _local_mode_features(
            predicted,
            component_index,
            spatial_index,
            comp_id,
            row.geometry,
            radius=float(radius),
            orientation_tolerance=float(orientation_tolerance),
        )
        local_mode_area = float(local.get("local_mode_area", float("nan")))
        if not np.isfinite(local_mode_area) or local_mode_area <= 0.0:
            continue
        area_to_mode = _safe_ratio(float(row.pred_area), local_mode_area)
        if float(local["local_same_orientation_one_uprn_count"]) < int(min_local_same_pattern_count):
            continue
        if float(local["local_mode_area_support"]) < int(min_local_mode_support):
            continue
        if area_to_mode < float(min_component_area_to_mode) or area_to_mode > float(max_component_area_to_mode):
            continue

        component = pd.Series(row._asdict())
        best_record, split_groups = _enumerate_split_candidate(
            component,
            group,
            local,
            max_seed_count=int(max_seed_count),
            max_zero_source_count=int(max_zero_source_count),
            min_shared_edge=float(min_shared_edge),
        )
        if best_record is None:
            continue
        records.append(best_record)
        groups_by_component[int(best_record["component_id"])] = split_groups

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=predicted.crs), {}
    out = gpd.GeoDataFrame(records, geometry="geometry", crs=predicted.crs)
    out = out.sort_values(["split_score", "component_area_to_local_mode"], ascending=[False, False]).reset_index(drop=True)
    return out, groups_by_component


def _select_split_candidates(
    candidates: gpd.GeoDataFrame,
    *,
    threshold: float,
    min_mean_regularity: float,
    max_mean_area_error: float,
    max_hull_gap: float,
) -> gpd.GeoDataFrame:
    if candidates.empty:
        return candidates
    selected = candidates[
        candidates["split_score"].astype(float).ge(float(threshold))
        & candidates["split_mean_regularity"].astype(float).ge(float(min_mean_regularity))
        & candidates["split_mean_area_error"].astype(float).le(float(max_mean_area_error))
        & candidates["split_max_hull_gap"].astype(float).le(float(max_hull_gap))
        & candidates["split_all_groups_connected"].astype(int).eq(1)
    ].copy()
    return selected.drop_duplicates("component_id", keep="first").reset_index(drop=True)


def _filter_edges_to_current_components(edges: gpd.GeoDataFrame, sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if edges.empty:
        return edges
    out = edges.copy()
    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    left_component = out["left_source_fid"].astype(int).map(source_to_component)
    right_component = out["right_source_fid"].astype(int).map(source_to_component)
    keep = left_component.notna() & right_component.notna() & left_component.eq(right_component)
    out = out[keep].copy()
    out["pred_component_id"] = left_component[keep].astype(int).to_numpy()
    return out


def _build_predicted_parcels_from_sources(
    sources: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    edge_stats: dict[int, dict[str, float]] = {}
    if not edges.empty:
        edge_values = edges.copy()
        merge_proba = edge_values.get("completion_proba", pd.Series(np.nan, index=edge_values.index)).fillna(
            edge_values.get("model_proba", pd.Series(np.nan, index=edge_values.index))
        )
        edge_values["merge_proba"] = merge_proba.astype(float)
        for comp_id, group in edge_values.groupby(edge_values["pred_component_id"].astype(int)):
            edge_stats[int(comp_id)] = {
                "predicted_edge_count": int(len(group)),
                "proba_min": float(group["merge_proba"].min()),
                "proba_mean": float(group["merge_proba"].mean()),
                "proba_max": float(group["merge_proba"].max()),
            }

    records: list[dict[str, Any]] = []
    reference_by_component: dict[int, list[int]] = {}
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        geom = shapely.union_all(group.geometry.array)
        shape = _shape_metrics(geom)
        refs = _source_reference_values(group)
        reference_by_component[int(comp_id)] = refs
        semantic_count = int(group["is_semantic_source"].fillna(0).astype(int).sum())
        stats = edge_stats.get(
            int(comp_id),
            {"predicted_edge_count": 0, "proba_min": np.nan, "proba_mean": np.nan, "proba_max": np.nan},
        )
        rec = {
            "pred_component_id": int(comp_id),
            "source_count": int(len(group)),
            "semantic_source_count": semantic_count,
            "outside_source_count": int(len(group) - semantic_count),
            "reference_merge_fid_count": int(len(refs)),
            "reference_merge_fids": "|".join(str(v) for v in refs),
            "max_reference_split_count": 1,
            "predicted_edge_count": stats["predicted_edge_count"],
            "proba_min": stats["proba_min"],
            "proba_mean": stats["proba_mean"],
            "proba_max": stats["proba_max"],
            "has_predicted_merge": int(len(group) > 1),
            "possible_false_positive_cluster": int(len(refs) > 1),
            "possible_split_reference": 0,
            "pred_uprn_count": _group_source_uprn(group),
            "geometry": geom,
        }
        for name, value in shape.items():
            rec[f"pred_{name}"] = float(value)
        records.append(rec)

    ref_component_counts: dict[int, int] = {}
    for refs in reference_by_component.values():
        for ref in refs:
            ref_component_counts[ref] = ref_component_counts.get(ref, 0) + 1
    for rec in records:
        refs = reference_by_component[int(rec["pred_component_id"])]
        max_split = max([ref_component_counts.get(ref, 1) for ref in refs] or [1])
        rec["max_reference_split_count"] = int(max_split)
        rec["possible_split_reference"] = int(len(refs) == 1 and max_split > 1)

    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _apply_splits(
    sources: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    groups_by_component: dict[int, list[list[int]]],
) -> gpd.GeoDataFrame:
    if selected.empty:
        return sources.copy()
    out = sources.copy()
    out["local_mode_split_status"] = out.get("local_mode_split_status", "base")
    out["local_mode_split_from_component"] = out.get("local_mode_split_from_component", np.nan)
    out["local_mode_split_group_index"] = out.get("local_mode_split_group_index", np.nan)
    out["local_mode_split_score"] = out.get("local_mode_split_score", np.nan)
    next_component_id = int(out["pred_component_id"].astype(int).max()) + 1

    selected_by_component = selected.set_index(selected["component_id"].astype(int), drop=False)
    for comp_id, split_groups in groups_by_component.items():
        if int(comp_id) not in selected_by_component.index:
            continue
        row = selected_by_component.loc[int(comp_id)]
        for group_index, source_ids in enumerate(split_groups):
            target_component = int(comp_id) if int(group_index) == 0 else int(next_component_id)
            if int(group_index) > 0:
                next_component_id += 1
            mask = out["source_fid"].astype(int).isin([int(v) for v in source_ids])
            out.loc[mask, "pred_component_id"] = target_component
            out.loc[mask, "local_mode_split_status"] = "local_mode_split"
            out.loc[mask, "local_mode_split_from_component"] = int(comp_id)
            out.loc[mask, "local_mode_split_group_index"] = int(group_index)
            out.loc[mask, "local_mode_split_score"] = float(row["split_score"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split local-mode overmerged WFS parcel components.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--allow-unflagged-components", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.76)
    parser.add_argument("--radius", type=float, default=100.0)
    parser.add_argument("--orientation-tolerance", type=float, default=15.0)
    parser.add_argument("--min-local-same-pattern-count", type=int, default=18)
    parser.add_argument("--min-local-mode-support", type=int, default=5)
    parser.add_argument("--min-component-area-to-mode", type=float, default=1.65)
    parser.add_argument("--max-component-area-to-mode", type=float, default=4.25)
    parser.add_argument("--max-component-area", type=float, default=650.0)
    parser.add_argument("--max-component-source-count", type=int, default=12)
    parser.add_argument("--max-seed-count", type=int, default=4)
    parser.add_argument("--max-zero-source-count", type=int, default=9)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--min-mean-regularity", type=float, default=0.84)
    parser.add_argument("--max-mean-area-error", type=float, default=0.50)
    parser.add_argument("--max-split-hull-gap", type=float, default=0.22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    _log(f"[INFO] Reading input: {input_gpkg}")
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
    sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")
    edges = _read_optional_layer(input_gpkg, "predicted_positive_edges", sources.crs)
    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", sources.crs)
    excluded_gapfill_council_sources = _read_optional_layer(
        input_gpkg,
        "excluded_gapfill_council_sources",
        sources.crs,
    )
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", sources.crs)

    _log("[INFO] Building local-mode split candidates")
    candidates, groups_by_component = build_local_mode_split_candidates(
        predicted,
        sources,
        require_possible_false_positive=not bool(args.allow_unflagged_components),
        radius=float(args.radius),
        orientation_tolerance=float(args.orientation_tolerance),
        min_local_same_pattern_count=int(args.min_local_same_pattern_count),
        min_local_mode_support=int(args.min_local_mode_support),
        min_component_area_to_mode=float(args.min_component_area_to_mode),
        max_component_area_to_mode=float(args.max_component_area_to_mode),
        max_component_area=float(args.max_component_area),
        max_component_source_count=int(args.max_component_source_count),
        max_seed_count=int(args.max_seed_count),
        max_zero_source_count=int(args.max_zero_source_count),
        min_shared_edge=float(args.min_shared_edge),
    )
    selected = _select_split_candidates(
        candidates,
        threshold=float(args.threshold),
        min_mean_regularity=float(args.min_mean_regularity),
        max_mean_area_error=float(args.max_mean_area_error),
        max_hull_gap=float(args.max_split_hull_gap),
    )
    _log(f"[INFO] Split candidates={len(candidates):,}; selected={len(selected):,}")

    sources_new = _apply_splits(sources, selected, groups_by_component)
    edges_new = _filter_edges_to_current_components(edges, sources_new)
    predicted_new = _build_predicted_parcels_from_sources(sources_new, edges_new)

    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()

    selected_debug = selected.copy()
    if not selected_debug.empty:
        selected_debug["local_mode_split_selected"] = 1
    candidate_debug = candidates.copy()
    if not candidate_debug.empty:
        selected_ids = set(selected["component_id"].astype(int)) if not selected.empty else set()
        candidate_debug["local_mode_split_selected"] = candidate_debug["component_id"].astype(int).isin(selected_ids).astype(int)

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(merged_only.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels_merged_only")
    _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    _write_layer(sources_new, output_gpkg, "prediction_source_polygons")
    _write_layer(excluded_gapfill_council_sources, output_gpkg, "excluded_gapfill_council_sources")
    _write_layer(excluded_problem_sources, output_gpkg, "excluded_problem_sources")
    _write_layer(edges_new, output_gpkg, "predicted_positive_edges")
    _write_layer(possible_fp.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_false_positive_clusters")
    _write_layer(possible_split.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_split_reference_clusters")
    _write_layer(predicted_new, output_gpkg, "predicted_parcels_with_uprn")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only_with_uprn")
    _write_layer(possible_fp, output_gpkg, "possible_false_positive_clusters_with_uprn")
    _write_layer(possible_split, output_gpkg, "possible_split_reference_clusters_with_uprn")
    _write_layer(merged_only[merged_only["pred_uprn_count"].le(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_le1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].eq(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_eq1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].gt(1)].copy(), output_gpkg, "predicted_parcels_merged_only_multi_uprn")
    for layer in PASSTHROUGH_DEBUG_LAYERS:
        passthrough = _read_optional_layer(input_gpkg, layer, sources.crs)
        if not passthrough.empty:
            _write_layer(passthrough, output_gpkg, layer)
    _write_layer(candidate_debug, output_gpkg, "local_mode_split_candidates")
    _write_layer(selected_debug, output_gpkg, "local_mode_split_selected")

    summary = {
        "input_gpkg": str(input_gpkg),
        "output_gpkg": str(output_gpkg),
        "allow_unflagged_components": bool(args.allow_unflagged_components),
        "threshold": float(args.threshold),
        "radius": float(args.radius),
        "min_local_same_pattern_count": int(args.min_local_same_pattern_count),
        "min_component_area_to_mode": float(args.min_component_area_to_mode),
        "max_component_area_to_mode": float(args.max_component_area_to_mode),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "base_predicted_components": int(len(predicted)),
        "new_predicted_components": int(len(predicted_new)),
        "base_multi_uprn_components": int(predicted["pred_uprn_count"].fillna(0).astype(int).gt(1).sum()),
        "new_multi_uprn_components": int(predicted_new["pred_uprn_count"].fillna(0).astype(int).gt(1).sum()),
        "possible_false_positive_clusters": int(len(possible_fp)),
        "possible_split_reference_clusters": int(len(possible_split)),
    }
    summary_path = output_gpkg.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Local-mode split complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

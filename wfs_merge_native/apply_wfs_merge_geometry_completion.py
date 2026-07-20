#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import LineString

from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_anchor_group_repaired_threshold_085_gate_096.gpkg"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_geometry_completed_trial_v1.gpkg"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _ids_text(values: set[int] | frozenset[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(v)) for v in sorted(values))


def _parse_ids(text: object) -> list[int]:
    out: list[int] = []
    for part in str(text or "").split("|"):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_priority_fids(text: object) -> set[int]:
    out: set[int] = set()
    for part in str(text or "").replace(",", "|").replace(";", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _write_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    clean = gdf.copy().reset_index(drop=True)
    for column in clean.columns:
        if column == clean.geometry.name:
            continue
        if clean[column].apply(lambda value: isinstance(value, (list, tuple, set, dict))).any():
            clean[column] = clean[column].map(
                lambda value: json.dumps(value) if isinstance(value, (list, tuple, set, dict)) else value
            )
    clean.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")


def _read_optional_layer(path: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _component_reference_values(group: gpd.GeoDataFrame) -> list[int]:
    values: list[int] = []
    for value in group["reference_merge_fid"].dropna():
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(values))


def _prepare_predicted(path: Path) -> gpd.GeoDataFrame:
    predicted = gpd.read_file(path, layer="predicted_parcels_with_uprn", engine="pyogrio", fid_as_index=True)
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted.index = predicted.index.astype(int)
    predicted["layer_fid"] = predicted.index.astype(int)
    predicted["pred_component_id"] = predicted["pred_component_id"].astype(int)
    predicted["pred_area"] = predicted.geometry.area.astype(float)
    return predicted


def _build_seed_fids(
    predicted: gpd.GeoDataFrame,
    *,
    max_seed_area: float,
    max_seed_source_count: int,
    max_seed_regularity: float,
    min_seed_hull_gap: float,
    max_seed_mrr: float,
) -> set[int]:
    split = predicted["possible_split_reference"].fillna(0).astype(int).eq(1)
    size = predicted["pred_area"].astype(float).le(float(max_seed_area))
    source_count = predicted["source_count"].fillna(999).astype(int).le(int(max_seed_source_count))
    shape_anomaly = (
        predicted["pred_regularity_score"].fillna(0).astype(float).lt(float(max_seed_regularity))
        | predicted["pred_hull_gap_ratio"].fillna(0).astype(float).gt(float(min_seed_hull_gap))
        | predicted["pred_mrr_ratio"].fillna(0).astype(float).lt(float(max_seed_mrr))
    )
    return set(predicted.loc[split & size & source_count & shape_anomaly, "layer_fid"].astype(int))


def _shared_edges_for_seeds(
    predicted: gpd.GeoDataFrame,
    seed_fids: set[int],
    *,
    min_shared_edge: float,
    max_pair_area: float,
    query_chunk_size: int,
) -> pd.DataFrame:
    if not seed_fids:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])

    all_geoms = predicted.geometry.reset_index(drop=True)
    all_fids = predicted["layer_fid"].astype(int).to_numpy()
    fid_to_pos = {int(fid): pos for pos, fid in enumerate(all_fids)}
    areas = predicted["pred_area"].astype(float).to_numpy()
    sindex = predicted.sindex

    seed_positions = np.array([fid_to_pos[int(fid)] for fid in sorted(seed_fids) if int(fid) in fid_to_pos], dtype=int)
    rows: list[pd.DataFrame] = []
    for start in range(0, len(seed_positions), int(query_chunk_size)):
        positions = seed_positions[start : start + int(query_chunk_size)]
        if len(positions) == 0:
            continue
        query_geoms = all_geoms.iloc[positions]
        left_pos, right_pos = sindex.query(query_geoms.geometry.array, predicate="intersects")
        if len(left_pos) == 0:
            continue
        absolute_left = positions[left_pos]
        absolute_right = right_pos
        left_fids = all_fids[absolute_left]
        right_fids = all_fids[absolute_right]
        keep = left_fids != right_fids
        if not bool(np.any(keep)):
            continue
        left_fids = left_fids[keep]
        right_fids = right_fids[keep]
        left_pos_values = absolute_left[keep]
        right_pos_values = absolute_right[keep]

        canonical_left = np.minimum(left_fids, right_fids)
        canonical_right = np.maximum(left_fids, right_fids)
        pair_area = areas[left_pos_values] + areas[right_pos_values]
        chunk = pd.DataFrame(
            {
                "left_fid": canonical_left.astype(int),
                "right_fid": canonical_right.astype(int),
                "left_pos": left_pos_values.astype(int),
                "right_pos": right_pos_values.astype(int),
                "pair_area": pair_area.astype(float),
            }
        ).drop_duplicates(["left_fid", "right_fid"])
        chunk = chunk[chunk["pair_area"].le(float(max_pair_area))].copy()
        if chunk.empty:
            continue
        shared_values: list[float] = []
        for edge_start in range(0, len(chunk), 50_000):
            edge_chunk = chunk.iloc[edge_start : edge_start + 50_000]
            left_geom = all_geoms.iloc[edge_chunk["left_pos"].astype(int).to_numpy()]
            right_geom = all_geoms.iloc[edge_chunk["right_pos"].astype(int).to_numpy()]
            shared = shapely.length(
                shapely.intersection(shapely.boundary(left_geom.array), shapely.boundary(right_geom.array))
            )
            shared_values.extend(float(v) for v in shared)
        chunk["shared_edge_len"] = shared_values
        chunk = chunk[chunk["shared_edge_len"].ge(float(min_shared_edge))].copy()
        if not chunk.empty:
            rows.append(chunk[["left_fid", "right_fid", "shared_edge_len"]])

    if not rows:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])
    edges = pd.concat(rows, ignore_index=True).drop_duplicates(["left_fid", "right_fid"], keep="first")
    return edges.reset_index(drop=True)


def _adjacency(edges: pd.DataFrame, top_neighbors: int) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    adjacency: dict[int, list[tuple[int, float]]] = {}
    shared: dict[tuple[int, int], float] = {}
    for row in edges.itertuples(index=False):
        left = int(row.left_fid)
        right = int(row.right_fid)
        value = float(row.shared_edge_len)
        adjacency.setdefault(left, []).append((right, value))
        adjacency.setdefault(right, []).append((left, value))
        shared[(min(left, right), max(left, right))] = value
    for fid, values in list(adjacency.items()):
        values = sorted(values, key=lambda item: (-float(item[1]), int(item[0])))[: int(top_neighbors)]
        adjacency[fid] = values
    return adjacency, shared


def _enumerate_connected_groups(
    *,
    seed_fids: set[int],
    seed_order: list[int],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    area_by_fid: dict[int, float],
    perimeter_by_fid: dict[int, float],
    max_group_size: int,
    max_after_area: float,
    max_candidates: int,
) -> set[frozenset[int]]:
    def cheap_pass(group: frozenset[int]) -> bool:
        areas = [float(area_by_fid.get(fid, 0.0)) for fid in group]
        area_sum = sum(areas)
        if not areas or area_sum <= 0.0 or area_sum > float(max_after_area):
            return False
        max_area_ratio = _safe_ratio(max(areas), area_sum)
        small_area_ratio = _safe_ratio(area_sum - max(areas), area_sum)
        if max_area_ratio < 0.45 or small_area_ratio < 0.004:
            return False
        internal_shared = 0.0
        sorted_fids = sorted(group)
        for i, left in enumerate(sorted_fids):
            for right in sorted_fids[i + 1 :]:
                internal_shared += float(shared_by_pair.get((min(left, right), max(left, right)), 0.0))
        if internal_shared < 4.0:
            return False
        perimeter_sum = sum(float(perimeter_by_fid.get(fid, 0.0)) for fid in group)
        boundary_simplification = _safe_ratio(2.0 * internal_shared, perimeter_sum)
        internal_to_sqrt_area = _safe_ratio(internal_shared, math.sqrt(area_sum))
        if boundary_simplification < 0.07:
            return False
        return bool(
            max_area_ratio >= 0.82
            or boundary_simplification >= 0.20
            or internal_to_sqrt_area >= 0.70
        )

    groups: set[frozenset[int]] = set()
    ordered_seeds = [int(fid) for fid in seed_order if int(fid) in seed_fids]
    ordered_seen = set(ordered_seeds)
    ordered_seeds.extend(sorted(int(fid) for fid in seed_fids if int(fid) not in ordered_seen))
    for seed in ordered_seeds:
        if seed not in adjacency:
            continue
        start = frozenset({int(seed)})
        stack = [start]
        seen = {start}
        while stack:
            current = stack.pop()
            if len(current) >= 2 and cheap_pass(current):
                groups.add(current)
                if len(groups) >= int(max_candidates):
                    return groups
            if len(current) >= int(max_group_size):
                continue
            frontier: dict[int, float] = {}
            for fid in current:
                for neighbor, shared_len in adjacency.get(int(fid), []):
                    if neighbor not in current:
                        frontier[neighbor] = max(float(shared_len), frontier.get(neighbor, 0.0))
            for neighbor, _ in sorted(frontier.items(), key=lambda item: (-float(item[1]), int(item[0]))):
                new_group = frozenset(set(current) | {int(neighbor)})
                if new_group in seen:
                    continue
                area = sum(float(area_by_fid.get(fid, 0.0)) for fid in new_group)
                if area > float(max_after_area):
                    continue
                seen.add(new_group)
                stack.append(new_group)
    return groups


def _candidate_metrics(
    fids: frozenset[int],
    *,
    predicted: gpd.GeoDataFrame,
    geom_by_fid: dict[int, Any],
    attrs_by_fid: dict[int, dict[str, Any]],
    shared_by_pair: dict[tuple[int, int], float],
) -> dict[str, Any]:
    geoms = [geom_by_fid[int(fid)] for fid in sorted(fids)]
    union_geom = shapely.union_all(geoms)
    shape = _shape_metrics(union_geom)
    areas = np.asarray([float(shapely.area(geom)) for geom in geoms], dtype="float64")
    perimeters = np.asarray([float(shapely.length(geom)) for geom in geoms], dtype="float64")
    part_shapes = [_shape_metrics(geom) for geom in geoms]
    largest_idx = int(np.argmax(areas))
    largest_shape = part_shapes[largest_idx]
    complete_like = [
        bool(
            part["regularity_score"] >= 0.90
            and part["hull_gap_ratio"] <= 0.06
            and part["mrr_ratio"] >= 0.80
        )
        for part in part_shapes
    ]
    internal_shared = 0.0
    sorted_fids = sorted(fids)
    for i, left in enumerate(sorted_fids):
        for right in sorted_fids[i + 1 :]:
            internal_shared += float(shared_by_pair.get((min(left, right), max(left, right)), 0.0))
    union_perimeter = float(shapely.length(union_geom))
    perimeter_sum = float(perimeters.sum())
    weighted_regularity = float(
        np.average([float(part["regularity_score"]) for part in part_shapes], weights=areas)
    )
    refs = sorted(
        {
            str(attrs_by_fid[int(fid)].get("reference_merge_fids") or "")
            for fid in fids
            if str(attrs_by_fid[int(fid)].get("reference_merge_fids") or "")
        }
    )
    uprn_sum = int(sum(int(attrs_by_fid[int(fid)].get("pred_uprn_count") or 0) for fid in fids))
    source_count_sum = int(sum(int(attrs_by_fid[int(fid)].get("source_count") or 0) for fid in fids))
    record: dict[str, Any] = {
        "candidate_fids": _ids_text(fids),
        "candidate_component_ids": _ids_text(
            {int(attrs_by_fid[int(fid)].get("pred_component_id")) for fid in fids}
        ),
        "group_size": int(len(fids)),
        "group_area": float(shape["area"]),
        "group_uprn_count": int(uprn_sum),
        "group_source_count": int(source_count_sum),
        "reference_fids": "|".join(refs),
        "reference_fid_count": int(len(refs)),
        "max_area_ratio": _safe_ratio(float(areas.max()), float(areas.sum())),
        "small_area_ratio": _safe_ratio(float(areas.sum() - areas.max()), float(areas.sum())),
        "internal_shared_len": float(internal_shared),
        "boundary_simplification": _safe_ratio(perimeter_sum - union_perimeter, perimeter_sum),
        "internal_to_sqrt_area": _safe_ratio(float(internal_shared), math.sqrt(float(areas.sum()))),
        "weighted_part_regularity": float(weighted_regularity),
        "largest_part_regularity": float(largest_shape["regularity_score"]),
        "largest_part_mrr_ratio": float(largest_shape["mrr_ratio"]),
        "largest_part_hull_gap_ratio": float(largest_shape["hull_gap_ratio"]),
        "complete_like_part_count": int(sum(complete_like)),
        "all_parts_complete_like": int(all(complete_like)),
        "geometry": union_geom,
    }
    for name, value in shape.items():
        record[f"group_{name}"] = float(value)
    record["regularity_gain_vs_largest"] = float(shape["regularity_score"] - largest_shape["regularity_score"])
    record["hull_gap_reduction_vs_largest"] = float(largest_shape["hull_gap_ratio"] - shape["hull_gap_ratio"])
    record["regularity_gain_vs_weighted_parts"] = float(shape["regularity_score"] - weighted_regularity)
    return record


def _score_candidate(row: pd.Series) -> tuple[bool, float, bool, bool, bool]:
    good_neighbor_merge = bool(
        int(row["complete_like_part_count"]) >= int(row["group_size"]) - 1
        and float(row["max_area_ratio"]) < 0.82
        and float(row["regularity_gain_vs_largest"]) < 0.04
        and float(row["hull_gap_reduction_vs_largest"]) < 0.04
    )
    strong_base = bool(
        float(row["group_regularity_score"]) >= 0.94
        and float(row["group_mrr_ratio"]) >= 0.88
        and float(row["group_hull_gap_ratio"]) <= 0.06
        and float(row["boundary_simplification"]) >= 0.20
        and float(row["max_area_ratio"]) >= 0.50
        and not good_neighbor_merge
    )
    strong = bool(
        strong_base
        and (
            float(row["max_area_ratio"]) >= 0.84
            or float(row["regularity_gain_vs_largest"]) >= 0.04
            or float(row["hull_gap_reduction_vs_largest"]) >= 0.04
            or float(row["group_regularity_score"]) >= 0.985
        )
    )
    patch = bool(
        float(row["max_area_ratio"]) >= 0.82
        and 0.004 <= float(row["small_area_ratio"]) <= 0.15
        and float(row["internal_shared_len"]) >= 4.0
        and float(row["boundary_simplification"]) >= 0.07
        and float(row["group_hull_gap_ratio"]) <= 0.075
        and float(row["group_regularity_score"]) >= float(row["largest_part_regularity"]) - 0.055
        and not good_neighbor_merge
    )
    passes = strong or patch
    score = (
        1.2 * float(row["group_regularity_score"])
        + 0.9 * float(row["group_mrr_ratio"])
        - 1.0 * float(row["group_hull_gap_ratio"])
        + 1.6 * float(row["boundary_simplification"])
        + 0.25 * min(float(row["internal_to_sqrt_area"]), 2.5)
        + 0.45 * max(float(row["regularity_gain_vs_largest"]), 0.0)
        + 0.55 * max(float(row["hull_gap_reduction_vs_largest"]), 0.0)
        + 0.35 * float(row["small_area_ratio"])
        + 0.10 * int(row["group_size"])
        + (0.65 if patch else 0.0)
        + (0.45 if strong else 0.0)
        - (1.4 if good_neighbor_merge else 0.0)
    )
    return passes, float(score), strong, patch, good_neighbor_merge


def _build_candidates(
    predicted: gpd.GeoDataFrame,
    *,
    seed_fids: set[int],
    priority_fids: set[int],
    min_shared_edge: float,
    max_pair_area: float,
    max_after_area: float,
    top_neighbors: int,
    max_group_size: int,
    query_chunk_size: int,
    max_candidates: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    _log("[INFO] Building shared-edge graph for suspicious split components")
    edges = _shared_edges_for_seeds(
        predicted,
        seed_fids,
        min_shared_edge=min_shared_edge,
        max_pair_area=max_pair_area,
        query_chunk_size=query_chunk_size,
    )
    if edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs=predicted.crs), edges
    _log(f"[INFO] Shared-edge rows={len(edges):,}")
    adjacency, shared_by_pair = _adjacency(edges, top_neighbors=top_neighbors)
    area_by_fid = predicted.set_index("layer_fid")["pred_area"].astype(float).to_dict()
    perimeter_by_fid = predicted.set_index("layer_fid").geometry.length.astype(float).to_dict()
    seed_frame = predicted[predicted["layer_fid"].astype(int).isin(seed_fids)].copy()
    seed_frame["seed_anomaly_score"] = (
        (1.0 - seed_frame["pred_regularity_score"].fillna(0.0).astype(float)).clip(lower=0.0)
        + seed_frame["pred_hull_gap_ratio"].fillna(0.0).astype(float).clip(lower=0.0)
        + (1.0 - seed_frame["pred_mrr_ratio"].fillna(0.0).astype(float)).clip(lower=0.0)
        + 0.08 * np.log1p(seed_frame["pred_area"].fillna(0.0).astype(float).clip(lower=0.0))
    )
    seed_order = (
        seed_frame.sort_values(["seed_anomaly_score", "layer_fid"], ascending=[False, True])["layer_fid"]
        .astype(int)
        .tolist()
    )
    priority_order = [int(fid) for fid in sorted(priority_fids) if int(fid) in seed_fids]
    if priority_order:
        seen_priority = set(priority_order)
        seed_order = priority_order + [int(fid) for fid in seed_order if int(fid) not in seen_priority]
    groups = _enumerate_connected_groups(
        seed_fids=seed_fids,
        seed_order=seed_order,
        adjacency=adjacency,
        shared_by_pair=shared_by_pair,
        area_by_fid=area_by_fid,
        perimeter_by_fid=perimeter_by_fid,
        max_group_size=max_group_size,
        max_after_area=max_after_area,
        max_candidates=max_candidates,
    )
    _log(f"[INFO] Raw connected groups={len(groups):,}")
    if not groups:
        return gpd.GeoDataFrame(geometry=[], crs=predicted.crs), edges

    geom_by_fid = predicted.set_index("layer_fid").geometry.to_dict()
    attrs_by_fid = predicted.set_index("layer_fid").drop(columns="geometry").to_dict("index")

    records: list[dict[str, Any]] = []
    for group in groups:
        areas = [float(area_by_fid.get(fid, 0.0)) for fid in group]
        if not areas or sum(areas) > float(max_after_area):
            continue
        max_area_ratio = _safe_ratio(max(areas), sum(areas))
        small_area_ratio = _safe_ratio(sum(areas) - max(areas), sum(areas))
        if max_area_ratio < 0.45 or small_area_ratio < 0.004:
            continue
        internal_shared = 0.0
        sorted_fids = sorted(group)
        for i, left in enumerate(sorted_fids):
            for right in sorted_fids[i + 1 :]:
                internal_shared += float(shared_by_pair.get((min(left, right), max(left, right)), 0.0))
        if internal_shared < 4.0:
            continue
        perimeter_sum = sum(float(perimeter_by_fid.get(fid, 0.0)) for fid in group)
        boundary_simplification = _safe_ratio(2.0 * internal_shared, perimeter_sum)
        if boundary_simplification < 0.07:
            continue
        rec = _candidate_metrics(
            group,
            predicted=predicted,
            geom_by_fid=geom_by_fid,
            attrs_by_fid=attrs_by_fid,
            shared_by_pair=shared_by_pair,
        )
        passes, score, strong, patch, guarded = _score_candidate(pd.Series(rec))
        rec["geometry_completion_score"] = float(score)
        rec["geometry_completion_pass"] = int(passes)
        rec["geometry_completion_strong"] = int(strong)
        rec["geometry_completion_patch"] = int(patch)
        rec["good_neighbor_merge_guard"] = int(guarded)
        records.append(rec)

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=predicted.crs), edges
    candidates = gpd.GeoDataFrame(records, geometry="geometry", crs=predicted.crs)
    return candidates, edges


def _select_candidates(candidates: gpd.GeoDataFrame, threshold: float) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if candidates.empty:
        return candidates.copy(), candidates.copy()
    eligible = candidates[
        candidates["geometry_completion_pass"].eq(1)
        & candidates["geometry_completion_score"].ge(float(threshold))
    ].copy()
    review = candidates[
        candidates["geometry_completion_pass"].eq(1)
        & candidates["geometry_completion_score"].lt(float(threshold))
    ].copy()
    if eligible.empty:
        return eligible, review.sort_values("geometry_completion_score", ascending=False).reset_index(drop=True)
    eligible = eligible.sort_values(
        [
            "geometry_completion_score",
            "group_size",
            "internal_shared_len",
            "group_regularity_score",
            "group_hull_gap_ratio",
        ],
        ascending=[False, False, False, False, True],
    )
    selected_rows: list[pd.Series] = []
    used_fids: set[int] = set()
    for _, row in eligible.iterrows():
        fids = set(_parse_ids(row["candidate_fids"]))
        if fids & used_fids:
            continue
        selected_rows.append(row)
        used_fids |= fids
    selected = (
        gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=candidates.crs).reset_index(drop=True)
        if selected_rows
        else eligible.iloc[0:0].copy()
    )
    return selected, review.sort_values("geometry_completion_score", ascending=False).reset_index(drop=True)


def _synthetic_repair_edges(
    selected: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    predicted: gpd.GeoDataFrame,
    target_component_by_group: dict[str, int],
) -> gpd.GeoDataFrame:
    if selected.empty:
        return gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    first_source_by_component = {
        int(comp_id): int(group["source_fid"].astype(int).iloc[0])
        for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int))
    }
    source_ref = sources.set_index(sources["source_fid"].astype(int))["reference_merge_fid"].to_dict()
    geom_by_component = predicted.set_index(predicted["pred_component_id"].astype(int)).geometry.to_dict()
    comp_by_fid = predicted.set_index("layer_fid")["pred_component_id"].astype(int).to_dict()
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        candidate_key = str(row.candidate_fids)
        target_component = int(target_component_by_group[candidate_key])
        target_source = int(first_source_by_component.get(target_component, -1))
        target_point = geom_by_component[target_component].representative_point()
        for fid in _parse_ids(candidate_key):
            component = int(comp_by_fid[int(fid)])
            if component == target_component:
                continue
            source = int(first_source_by_component.get(component, -1))
            point = geom_by_component[component].representative_point()
            records.append(
                {
                    "left_source_fid": target_source,
                    "right_source_fid": source,
                    "left_merge_fid": int(source_ref.get(target_source, -1))
                    if pd.notna(source_ref.get(target_source, np.nan))
                    else -1,
                    "right_merge_fid": int(source_ref.get(source, -1))
                    if pd.notna(source_ref.get(source, np.nan))
                    else -1,
                    "label": -1,
                    "model_proba": np.nan,
                    "completion_proba": float(row.geometry_completion_score),
                    "role_pair": "geometry_completion",
                    "pred_component_id": target_component,
                    "model_stage": "geometry_completion",
                    "geometry": LineString(
                        [
                            (shapely.get_x(target_point), shapely.get_y(target_point)),
                            (shapely.get_x(point), shapely.get_y(point)),
                        ]
                    ),
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _ensure_edge_columns(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = edges.copy()
    defaults = {
        "left_source_fid": -1,
        "right_source_fid": -1,
        "left_merge_fid": -1,
        "right_merge_fid": -1,
        "label": -1,
        "model_proba": np.nan,
        "completion_proba": np.nan,
        "role_pair": "",
        "pred_component_id": -1,
        "model_stage": "edge",
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
    return out


def _filter_edges_to_current_components(edges: gpd.GeoDataFrame, sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = _ensure_edge_columns(edges)
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
        edge_values["merge_proba"] = edge_values["completion_proba"].fillna(edge_values["model_proba"]).astype(float)
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
        refs = _component_reference_values(group)
        reference_by_component[int(comp_id)] = refs
        stats = edge_stats.get(
            int(comp_id),
            {"predicted_edge_count": 0, "proba_min": np.nan, "proba_mean": np.nan, "proba_max": np.nan},
        )
        semantic_count = int(group["is_semantic_source"].fillna(0).astype(int).sum())
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
            "pred_uprn_count": int(group["source_uprn_count"].fillna(0).astype(int).sum()),
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


def _apply_repairs(
    *,
    input_gpkg: Path,
    output_gpkg: Path,
    predicted: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    _log("[INFO] Reading source polygons")
    sources = pyogrio.read_dataframe(input_gpkg, layer="prediction_source_polygons")
    sources = sources[sources.geometry.notna() & ~sources.geometry.is_empty].copy()
    sources["source_fid"] = sources["source_fid"].astype(int)
    sources["pred_component_id"] = sources["pred_component_id"].astype(int)
    sources_new = sources.copy()
    sources_new["geometry_completion_status"] = "base"
    sources_new["geometry_completed_to_component"] = np.nan
    sources_new["geometry_completion_score"] = np.nan
    sources_new["geometry_completion_group_fids"] = ""

    comp_by_fid = predicted.set_index("layer_fid")["pred_component_id"].astype(int).to_dict()
    area_by_comp = predicted.set_index("pred_component_id")["pred_area"].astype(float).to_dict()
    target_component_by_group: dict[str, int] = {}
    for row in selected.itertuples(index=False):
        fids = _parse_ids(row.candidate_fids)
        components = sorted({int(comp_by_fid[int(fid)]) for fid in fids})
        target_component = max(components, key=lambda comp: (float(area_by_comp.get(comp, 0.0)), -int(comp)))
        target_component_by_group[str(row.candidate_fids)] = int(target_component)
        mask = sources_new["pred_component_id"].astype(int).isin(components)
        sources_new.loc[mask, "geometry_completion_status"] = "geometry_completed"
        sources_new.loc[mask, "geometry_completed_to_component"] = int(target_component)
        sources_new.loc[mask, "geometry_completion_score"] = float(row.geometry_completion_score)
        sources_new.loc[mask, "geometry_completion_group_fids"] = str(row.candidate_fids)
        sources_new.loc[mask, "pred_component_id"] = int(target_component)

    _log("[INFO] Rebuilding edges and predicted parcels")
    edges = pyogrio.read_dataframe(input_gpkg, layer="predicted_positive_edges")
    edges_current = _filter_edges_to_current_components(edges, sources_new)
    repair_edges = _synthetic_repair_edges(selected, sources, predicted, target_component_by_group)
    if not repair_edges.empty:
        edges_new = pd.concat([edges_current, repair_edges], ignore_index=True)
        edges_new = gpd.GeoDataFrame(edges_new, geometry="geometry", crs=sources.crs)
    else:
        edges_new = edges_current
    predicted_new = _build_predicted_parcels_from_sources(sources_new, edges_new)
    return sources_new, edges_new, predicted_new


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a geometry-only tiling completion trial to WFS merge output.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--threshold", type=float, default=2.65)
    parser.add_argument("--min-shared-edge", type=float, default=0.2)
    parser.add_argument("--max-seed-area", type=float, default=2000.0)
    parser.add_argument("--max-seed-source-count", type=int, default=4)
    parser.add_argument("--max-seed-regularity", type=float, default=0.985)
    parser.add_argument("--min-seed-hull-gap", type=float, default=0.005)
    parser.add_argument("--max-seed-mrr", type=float, default=0.985)
    parser.add_argument("--max-pair-area", type=float, default=2000.0)
    parser.add_argument("--max-after-area", type=float, default=2000.0)
    parser.add_argument("--max-group-size", type=int, default=5)
    parser.add_argument("--top-neighbors", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=5000)
    parser.add_argument("--max-candidates", type=int, default=800000)
    parser.add_argument("--top-candidate-layer-limit", type=int, default=20000)
    parser.add_argument("--priority-fids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    _log("[INFO] Reading predicted parcels")
    predicted = _prepare_predicted(input_gpkg)
    seed_fids = _build_seed_fids(
        predicted,
        max_seed_area=float(args.max_seed_area),
        max_seed_source_count=int(args.max_seed_source_count),
        max_seed_regularity=float(args.max_seed_regularity),
        min_seed_hull_gap=float(args.min_seed_hull_gap),
        max_seed_mrr=float(args.max_seed_mrr),
    )
    priority_fids = _parse_priority_fids(args.priority_fids)
    if priority_fids:
        valid_priority_fids = set(predicted["layer_fid"].astype(int)) & priority_fids
        seed_fids |= valid_priority_fids
        _log(f"[INFO] Priority fids={len(valid_priority_fids):,}")
    _log(f"[INFO] Seed fids={len(seed_fids):,}")

    candidates, shared_edges = _build_candidates(
        predicted,
        seed_fids=seed_fids,
        priority_fids=priority_fids,
        min_shared_edge=float(args.min_shared_edge),
        max_pair_area=float(args.max_pair_area),
        max_after_area=float(args.max_after_area),
        top_neighbors=int(args.top_neighbors),
        max_group_size=int(args.max_group_size),
        query_chunk_size=int(args.query_chunk_size),
        max_candidates=int(args.max_candidates),
    )
    _log(f"[INFO] Candidate rows={len(candidates):,}")
    selected, review = _select_candidates(candidates, threshold=float(args.threshold))
    _log(f"[INFO] Selected groups={len(selected):,}; review groups={len(review):,}")

    sources_new, edges_new, predicted_new = _apply_repairs(
        input_gpkg=input_gpkg,
        output_gpkg=output_gpkg,
        predicted=predicted,
        selected=selected,
    )

    _log(f"[INFO] Writing output: {output_gpkg}")
    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(predicted_new, output_gpkg, "predicted_parcels_with_uprn")
    _write_layer(merged_only.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels_merged_only")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only_with_uprn")
    _write_layer(possible_fp.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_false_positive_clusters")
    _write_layer(possible_split.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_split_reference_clusters")
    _write_layer(sources_new, output_gpkg, "prediction_source_polygons")
    _write_layer(edges_new, output_gpkg, "predicted_positive_edges")
    _write_layer(selected, output_gpkg, "geometry_completion_selected")
    _write_layer(review.head(int(args.top_candidate_layer_limit)), output_gpkg, "geometry_completion_review_candidates")
    if not candidates.empty:
        top = candidates.sort_values("geometry_completion_score", ascending=False).head(
            int(args.top_candidate_layer_limit)
        )
        _write_layer(top, output_gpkg, "geometry_completion_candidates_top")
    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", predicted.crs)
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", predicted.crs)
    if not semantic_reference.empty:
        _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    if not excluded_problem_sources.empty:
        _write_layer(excluded_problem_sources, output_gpkg, "excluded_problem_sources")

    summary = {
        "input_gpkg": str(input_gpkg),
        "output_gpkg": str(output_gpkg),
        "threshold": float(args.threshold),
        "seed_fids": int(len(seed_fids)),
        "shared_edge_rows": int(len(shared_edges)),
        "candidate_rows": int(len(candidates)),
        "candidate_pass_rows": int(candidates["geometry_completion_pass"].sum()) if not candidates.empty else 0,
        "selected_groups": int(len(selected)),
        "selected_source_components": int(sum(len(_parse_ids(v)) for v in selected["candidate_fids"]))
        if not selected.empty
        else 0,
        "old_predicted_rows": int(len(predicted)),
        "new_predicted_rows": int(len(predicted_new)),
        "old_possible_split_rows": int(predicted["possible_split_reference"].fillna(0).astype(int).sum()),
        "new_possible_split_rows": int(predicted_new["possible_split_reference"].fillna(0).astype(int).sum()),
        "old_possible_false_positive_rows": int(
            predicted["possible_false_positive_cluster"].fillna(0).astype(int).sum()
        ),
        "new_possible_false_positive_rows": int(
            predicted_new["possible_false_positive_cluster"].fillna(0).astype(int).sum()
        ),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not selected.empty:
        selected.drop(columns="geometry").to_csv(output_gpkg.with_suffix(".selected.csv"), index=False)
    if not candidates.empty:
        candidates.drop(columns="geometry").sort_values("geometry_completion_score", ascending=False).head(
            int(args.top_candidate_layer_limit)
        ).to_csv(output_gpkg.with_suffix(".candidates_top.csv"), index=False)
    _log(json.dumps(summary, indent=2))
    _log("[DONE] Geometry completion trial complete")


if __name__ == "__main__":
    main()

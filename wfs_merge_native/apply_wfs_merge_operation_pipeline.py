#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely

from apply_wfs_merge_completion_model import _build_predicted_parcels, _write_layer
from train_wfs_merge_completion_model import _shape_metrics
from train_wfs_merge_prune_model import DEFAULT_OUTPUT_DIR as DEFAULT_PRUNE_MODEL_DIR
from train_wfs_merge_prune_model import MODEL_FILE_NAME as PRUNE_MODEL_FILE_NAME
from train_wfs_merge_prune_model import build_prune_candidates


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3/"
    "model_predicted_polygons_completion_v3_threshold_090_strict_regularity_guard.gpkg"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_operation_models_v1/"
    "model_predicted_polygons_operation_v1_pruned_group_completed.gpkg"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _read_optional_layer(path: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


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


def _select_prune_operations(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    max_component_area: float,
    source_role: str,
    max_only_land_prune_source_area_ratio: float,
    min_mrr_gain: float,
    min_hull_gap_reduction: float,
    min_regularity_gain: float,
    min_after_regularity: float,
    max_after_hull_gap: float,
) -> pd.DataFrame:
    only_land_after_prune = (
        candidates["source_role"].astype(str).str.lower().eq("land")
        & candidates["rest_land_count"].fillna(0).astype(int).eq(0)
    )
    allowed_only_land_prune = candidates["source_area_ratio"].fillna(1.0).astype(float).le(
        float(max_only_land_prune_source_area_ratio)
    )
    selected = candidates[
        candidates["prune_proba"].ge(float(threshold))
        & candidates["component_uprn_count"].eq(1)
        & candidates["component_area"].le(float(max_component_area))
        & candidates["source_role"].astype(str).str.lower().eq(str(source_role).lower())
        & (~only_land_after_prune | allowed_only_land_prune)
        & candidates["remove_mrr_gain"].ge(float(min_mrr_gain))
        & candidates["remove_hull_gap_reduction"].ge(float(min_hull_gap_reduction))
        & candidates["remove_regularity_score_gain"].ge(float(min_regularity_gain))
        & candidates["after_remove_regularity_score"].ge(float(min_after_regularity))
        & candidates["after_remove_hull_gap_ratio"].le(float(max_after_hull_gap))
        & candidates["rest_source_count"].ge(1)
    ].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(
        [
            "prune_proba",
            "remove_regularity_score_gain",
            "remove_mrr_gain",
            "remove_hull_gap_reduction",
        ],
        ascending=[False, False, False, False],
    )
    return selected.drop_duplicates("component_id", keep="first").reset_index(drop=True)


def _component_summary(sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    records: list[dict[str, object]] = []
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        geom = shapely.union_all(group.geometry.array)
        records.append(
            {
                "component_id": int(comp_id),
                "component_source_count": int(len(group)),
                "component_uprn_count": int(group["source_uprn_count"].fillna(0).astype(int).sum()),
                "component_area": float(shapely.area(geom)),
                "component_perimeter": float(shapely.length(geom)),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _reference_key(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _component_context_summary(sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    records: list[dict[str, object]] = []
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        geom = shapely.union_all(group.geometry.array)
        metrics = _shape_metrics(geom)
        refs = {
            ref
            for ref in group["reference_merge_fid"].map(_reference_key).to_list()
            if ref is not None
        }
        records.append(
            {
                "component_id": int(comp_id),
                "component_source_count": int(len(group)),
                "component_uprn_count": int(group["source_uprn_count"].fillna(0).astype(int).sum()),
                "component_reference_count": int(len(refs)),
                "component_reference_fids": "|".join(str(v) for v in sorted(refs)),
                "component_area": float(metrics["area"]),
                "component_mrr_ratio": float(metrics["mrr_ratio"]),
                "component_hull_gap_ratio": float(metrics["hull_gap_ratio"]),
                "component_regularity_score": float(metrics["regularity_score"]),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _local_regular_mode_features(
    components: gpd.GeoDataFrame,
    component_id: int,
    *,
    radius: float,
    min_neighbor_area: float,
    max_neighbor_area: float,
) -> dict[str, float]:
    if components.empty:
        return {}
    row_match = components[components["component_id"].astype(int).eq(int(component_id))]
    if row_match.empty:
        return {}
    row = row_match.iloc[0]
    geom = row.geometry
    if geom is None or shapely.is_empty(geom):
        return {}

    minx, miny, maxx, maxy = shapely.bounds(shapely.buffer(geom, float(radius)))
    candidates = components.cx[minx:maxx, miny:maxy].copy()
    if candidates.empty:
        return {}
    candidates = candidates[candidates["component_id"].astype(int).ne(int(component_id))].copy()
    if candidates.empty:
        return {}
    distances = candidates.geometry.distance(geom)
    candidates = candidates[distances.le(float(radius))].copy()
    if candidates.empty:
        return {
            "local_neighbor_count": 0.0,
            "local_regular_one_uprn_count": 0.0,
            "local_regular_one_uprn_area_median": 0.0,
            "local_regular_one_uprn_area_mode10": 0.0,
            "component_area_to_local_median": 0.0,
            "component_area_per_uprn_to_local_median": 0.0,
        }

    regular_one = candidates[
        candidates["component_uprn_count"].fillna(0).astype(int).eq(1)
        & candidates["component_area"].astype(float).between(float(min_neighbor_area), float(max_neighbor_area))
        & candidates["component_mrr_ratio"].astype(float).ge(0.85)
        & candidates["component_hull_gap_ratio"].astype(float).le(0.12)
        & candidates["component_regularity_score"].astype(float).ge(0.90)
    ].copy()
    local_median = float(regular_one["component_area"].median()) if not regular_one.empty else 0.0
    if regular_one.empty:
        local_mode = 0.0
    else:
        buckets = (regular_one["component_area"].astype(float) / 10.0).round().astype(int) * 10
        local_mode = float(buckets.value_counts().index[0])
    uprn_count = max(int(row["component_uprn_count"] or 0), 1)
    return {
        "local_neighbor_count": float(len(candidates)),
        "local_regular_one_uprn_count": float(len(regular_one)),
        "local_regular_one_uprn_area_median": local_median,
        "local_regular_one_uprn_area_mode10": local_mode,
        "component_area_to_local_median": float(row["component_area"]) / (local_median or 1.0),
        "component_area_per_uprn_to_local_median": float(row["component_area"]) / float(uprn_count) / (local_median or 1.0),
    }


def _connected_source_groups(nodes: set[int], edge_pairs: list[tuple[int, int]]) -> list[set[int]]:
    parent = {int(node): int(node) for node in nodes}

    def find(value: int) -> int:
        root = int(value)
        while parent[root] != root:
            root = parent[root]
        while parent[int(value)] != root:
            current = parent[int(value)]
            parent[int(value)] = root
            value = current
        return root

    def union(left: int, right: int) -> None:
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edge_pairs:
        if left in parent and right in parent:
            union(int(left), int(right))

    grouped: dict[int, set[int]] = {}
    for node in nodes:
        grouped.setdefault(find(int(node)), set()).add(int(node))
    return list(grouped.values())


def _source_group_stats(group: gpd.GeoDataFrame, source_ids: set[int]) -> dict[str, object]:
    rows = group[group["source_fid"].astype(int).isin(source_ids)].copy()
    geom = shapely.union_all(rows.geometry.array)
    refs = {
        ref
        for ref in rows["reference_merge_fid"].map(_reference_key).to_list()
        if ref is not None
    }
    return {
        "source_ids": set(int(v) for v in source_ids),
        "source_fids": "|".join(str(v) for v in sorted(source_ids)),
        "source_count": int(len(rows)),
        "uprn_count": int(rows["source_uprn_count"].fillna(0).astype(int).sum()),
        "reference_count": int(len(refs)),
        "reference_fids": "|".join(str(v) for v in sorted(refs)),
        "area": float(shapely.area(geom)),
        "geometry": geom,
    }


def _shared_edge_between_geoms(left_geom, right_geom) -> float:
    if left_geom is None or right_geom is None or shapely.is_empty(left_geom) or shapely.is_empty(right_geom):
        return 0.0
    return float(shapely.length(shapely.intersection(shapely.boundary(left_geom), shapely.boundary(right_geom))))


def _attach_zero_uprn_split_orphans(
    group: gpd.GeoDataFrame,
    source_groups: list[set[int]],
    *,
    min_shared_edge: float,
) -> list[set[int]]:
    if len(source_groups) <= 2:
        return source_groups
    stats = [_source_group_stats(group, ids) for ids in source_groups]
    positive = [item for item in stats if int(item["uprn_count"]) > 0]
    orphans = [item for item in stats if int(item["uprn_count"]) == 0]
    if len(positive) < 2 or not orphans:
        return source_groups

    positive_sets = [set(item["source_ids"]) for item in positive]
    for orphan in sorted(orphans, key=lambda item: float(item["area"])):
        shared = [
            _shared_edge_between_geoms(orphan["geometry"], positive_item["geometry"])
            for positive_item in positive
        ]
        if not shared:
            continue
        best_idx = int(np.argmax(shared))
        if float(shared[best_idx]) < float(min_shared_edge):
            continue
        positive_sets[best_idx] |= set(orphan["source_ids"])
    return positive_sets


def _apply_neighborhood_overmerge_splits(
    sources: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    max_component_area: float,
    radius: float,
    min_local_regular_one_uprn: int,
    min_area_to_local_median: float,
    min_area_per_uprn_to_local_median: float,
    max_area_per_uprn_to_local_median: float,
    min_zero_orphan_attach_shared_edge: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    sources = sources.copy()
    components = _component_context_summary(sources)
    if components.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=sources.crs)
        return sources, empty, empty, empty

    candidate_components = components[
        components["component_uprn_count"].astype(int).ge(2)
        & components["component_reference_count"].astype(int).ge(2)
        & components["component_area"].astype(float).le(float(max_component_area))
    ].copy()
    if candidate_components.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=sources.crs)
        return sources, empty, empty, empty

    source_ref = sources.set_index(sources["source_fid"].astype(int))["reference_merge_fid"].map(_reference_key).to_dict()
    edge_rows = edges.copy()
    if edge_rows.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=sources.crs)
        return sources, empty, empty, empty
    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    edge_rows["left_current_component"] = edge_rows["left_source_fid"].astype(int).map(source_to_component)
    edge_rows["right_current_component"] = edge_rows["right_source_fid"].astype(int).map(source_to_component)
    edge_rows = edge_rows[
        edge_rows["left_current_component"].notna()
        & edge_rows["right_current_component"].notna()
        & edge_rows["left_current_component"].eq(edge_rows["right_current_component"])
    ].copy()

    component_debug_records: list[dict[str, object]] = []
    edge_debug_frames: list[gpd.GeoDataFrame] = []
    result_records: list[dict[str, object]] = []
    max_component_id = int(sources["pred_component_id"].astype(int).max())
    next_component_id = max_component_id + 1

    for row in candidate_components.itertuples(index=False):
        comp_id = int(row.component_id)
        local = _local_regular_mode_features(
            components,
            comp_id,
            radius=float(radius),
            min_neighbor_area=25.0,
            max_neighbor_area=250.0,
        )
        local_count = int(local.get("local_regular_one_uprn_count", 0.0))
        local_median = float(local.get("local_regular_one_uprn_area_median", 0.0))
        area_ratio = float(local.get("component_area_to_local_median", 0.0))
        per_uprn_ratio = float(local.get("component_area_per_uprn_to_local_median", 0.0))
        suspect = (
            local_count >= int(min_local_regular_one_uprn)
            and local_median > 0.0
            and area_ratio >= float(min_area_to_local_median)
            and per_uprn_ratio >= float(min_area_per_uprn_to_local_median)
            and per_uprn_ratio <= float(max_area_per_uprn_to_local_median)
        )
        component_record: dict[str, object] = {
            "component_id": comp_id,
            "split_selected": 0,
            "split_reason": "not_suspect",
            "component_uprn_count": int(row.component_uprn_count),
            "component_reference_count": int(row.component_reference_count),
            "component_source_count": int(row.component_source_count),
            "component_area": float(row.component_area),
            "component_reference_fids": str(row.component_reference_fids),
            **local,
            "geometry": row.geometry,
        }
        if not suspect:
            component_debug_records.append(component_record)
            continue

        group = sources[sources["pred_component_id"].astype(int).eq(comp_id)].copy()
        nodes = set(group["source_fid"].astype(int))
        component_edges = edge_rows[edge_rows["left_current_component"].astype(int).eq(comp_id)].copy()
        if component_edges.empty:
            component_record["split_reason"] = "no_internal_edges"
            component_debug_records.append(component_record)
            continue

        kept_pairs: list[tuple[int, int]] = []
        removed_mask: list[bool] = []
        for edge in component_edges.itertuples(index=False):
            left = int(edge.left_source_fid)
            right = int(edge.right_source_fid)
            left_ref = source_ref.get(left)
            right_ref = source_ref.get(right)
            remove = left_ref is not None and right_ref is not None and left_ref != right_ref
            removed_mask.append(bool(remove))
            if not remove:
                kept_pairs.append((left, right))
        removed_count = int(sum(removed_mask))
        if removed_count == 0:
            component_record["split_reason"] = "no_cross_reference_bridge"
            component_debug_records.append(component_record)
            continue

        source_groups = _connected_source_groups(nodes, kept_pairs)
        source_groups = _attach_zero_uprn_split_orphans(
            group,
            source_groups,
            min_shared_edge=float(min_zero_orphan_attach_shared_edge),
        )
        if len(source_groups) < 2:
            component_record["split_reason"] = "bridge_cut_kept_connected"
            component_debug_records.append(component_record)
            continue

        stats = [_source_group_stats(group, ids) for ids in source_groups]
        positive_stats = [item for item in stats if int(item["uprn_count"]) > 0]
        if len(positive_stats) < 2:
            component_record["split_reason"] = "not_enough_positive_children"
            component_debug_records.append(component_record)
            continue
        max_child_uprn = max(int(item["uprn_count"]) for item in positive_stats)
        if max_child_uprn >= int(row.component_uprn_count):
            component_record["split_reason"] = "no_uprn_split_gain"
            component_debug_records.append(component_record)
            continue
        child_area_ok = all(
            0.35 <= (float(item["area"]) / max(int(item["uprn_count"]), 1) / (local_median or 1.0)) <= 2.10
            for item in positive_stats
        )
        if not child_area_ok:
            component_record["split_reason"] = "child_area_outside_local_mode"
            component_debug_records.append(component_record)
            continue

        sorted_stats = sorted(
            stats,
            key=lambda item: (-int(item["uprn_count"]), -float(item["area"]), str(item["source_fids"])),
        )
        component_record["split_selected"] = 1
        component_record["split_reason"] = "neighborhood_modal_overmerge"
        component_record["removed_cross_reference_edge_count"] = removed_count
        component_record["split_child_count"] = len(sorted_stats)
        component_debug_records.append(component_record)

        new_ids: list[int] = [comp_id]
        new_ids.extend(range(next_component_id, next_component_id + len(sorted_stats) - 1))
        next_component_id += max(len(sorted_stats) - 1, 0)
        for new_component_id, stat in zip(new_ids, sorted_stats):
            mask = sources["source_fid"].astype(int).isin(set(stat["source_ids"]))
            sources.loc[mask, "pred_component_id"] = int(new_component_id)
            sources.loc[mask, "operation_status"] = "neighborhood_overmerge_split"
            sources.loc[mask, "neighborhood_split_from_component"] = comp_id
            result_records.append(
                {
                    "old_component_id": comp_id,
                    "new_component_id": int(new_component_id),
                    "source_fids": str(stat["source_fids"]),
                    "source_count": int(stat["source_count"]),
                    "uprn_count": int(stat["uprn_count"]),
                    "reference_count": int(stat["reference_count"]),
                    "reference_fids": str(stat["reference_fids"]),
                    "area": float(stat["area"]),
                    **local,
                    "geometry": stat["geometry"],
                }
            )

        removed_edges = component_edges[pd.Series(removed_mask, index=component_edges.index)].copy()
        if not removed_edges.empty:
            removed_edges["split_component_id"] = comp_id
            removed_edges["split_reason"] = "cross_reference_bridge_in_neighborhood_modal_overmerge"
            edge_debug_frames.append(removed_edges)

    component_debug = gpd.GeoDataFrame(component_debug_records, geometry="geometry", crs=sources.crs)
    if edge_debug_frames:
        edge_debug = gpd.GeoDataFrame(pd.concat(edge_debug_frames, ignore_index=True), geometry="geometry", crs=edges.crs)
    else:
        edge_debug = gpd.GeoDataFrame(geometry=[], crs=edges.crs)
    result_debug = gpd.GeoDataFrame(result_records, geometry="geometry", crs=sources.crs)
    return sources, component_debug, edge_debug, result_debug


def _component_shared_edges(components: gpd.GeoDataFrame, min_shared_edge: float) -> pd.DataFrame:
    if components.empty:
        return pd.DataFrame(columns=["left_component_id", "right_component_id", "shared_edge_len"])
    ref = components[["component_id", "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(ref, ref, how="inner", predicate="intersects", lsuffix="left", rsuffix="right")
    if joined.empty:
        return pd.DataFrame(columns=["left_component_id", "right_component_id", "shared_edge_len"])
    joined = joined.rename(columns={"component_id_left": "left_component_id", "component_id_right": "right_component_id"})
    joined = joined[joined["left_component_id"].astype(int).lt(joined["right_component_id"].astype(int))].copy()
    if joined.empty:
        return pd.DataFrame(columns=["left_component_id", "right_component_id", "shared_edge_len"])
    geom_by_component = components.set_index("component_id").geometry.to_dict()
    left_geoms = gpd.GeoSeries(
        joined["left_component_id"].astype(int).map(geom_by_component).to_list(),
        index=joined.index,
        crs=components.crs,
    )
    right_geoms = gpd.GeoSeries(
        joined["right_component_id"].astype(int).map(geom_by_component).to_list(),
        index=joined.index,
        crs=components.crs,
    )
    joined["shared_edge_len"] = shapely.length(
        shapely.intersection(shapely.boundary(left_geoms.array), shapely.boundary(right_geoms.array))
    )
    joined = joined[joined["shared_edge_len"].ge(float(min_shared_edge))].copy()
    return joined[["left_component_id", "right_component_id", "shared_edge_len"]].reset_index(drop=True)


def _build_zero_uprn_attachment_candidates(
    components: gpd.GeoDataFrame,
    *,
    min_shared_edge: float,
    max_added_area: float,
    max_target_area: float,
    max_after_area: float,
    max_added_source_count: int,
    min_target_shared_dominance: float,
) -> gpd.GeoDataFrame:
    if components.empty:
        return gpd.GeoDataFrame(geometry=[], crs=components.crs)
    edges = _component_shared_edges(components, min_shared_edge)
    if edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs=components.crs)

    attrs = components.set_index("component_id")
    records: list[dict[str, object]] = []
    for added_id, group in pd.concat(
        [
            edges.rename(columns={"left_component_id": "added_component_id", "right_component_id": "neighbor_component_id"}),
            edges.rename(columns={"right_component_id": "added_component_id", "left_component_id": "neighbor_component_id"}),
        ],
        ignore_index=True,
    ).groupby("added_component_id", sort=True):
        added_id = int(added_id)
        if added_id not in attrs.index:
            continue
        added = attrs.loc[added_id]
        if int(added["component_uprn_count"]) != 0:
            continue
        if float(added["component_area"]) > float(max_added_area):
            continue
        if int(added["component_source_count"]) > int(max_added_source_count):
            continue

        target_rows: list[dict[str, object]] = []
        for row in group.itertuples(index=False):
            target_id = int(row.neighbor_component_id)
            if target_id not in attrs.index:
                continue
            target = attrs.loc[target_id]
            if int(target["component_uprn_count"]) != 1:
                continue
            target_area = float(target["component_area"])
            added_area = float(added["component_area"])
            if target_area > float(max_target_area):
                continue
            if target_area + added_area > float(max_after_area):
                continue
            target_rows.append(
                {
                    "target_component_id": target_id,
                    "added_component_id": added_id,
                    "shared_edge_len": float(row.shared_edge_len),
                    "target_area": target_area,
                    "added_area": added_area,
                    "after_area": target_area + added_area,
                    "target_source_count": int(target["component_source_count"]),
                    "added_source_count": int(added["component_source_count"]),
                    "target_uprn_count": int(target["component_uprn_count"]),
                    "added_uprn_count": int(added["component_uprn_count"]),
                }
            )
        if not target_rows:
            continue
        target_rows = sorted(target_rows, key=lambda item: (-float(item["shared_edge_len"]), int(item["target_component_id"])))
        best = target_rows[0]
        second_shared = float(target_rows[1]["shared_edge_len"]) if len(target_rows) > 1 else 0.0
        dominance = float(best["shared_edge_len"]) / max(second_shared, 1e-9)
        if len(target_rows) > 1 and dominance < float(min_target_shared_dominance):
            continue
        best["target_neighbor_count"] = int(len(target_rows))
        best["second_best_shared_edge_len"] = float(second_shared)
        best["target_shared_dominance"] = float(dominance)
        best["geometry"] = added.geometry
        records.append(best)

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=components.crs)
    out = gpd.GeoDataFrame(records, geometry="geometry", crs=components.crs)
    return out.sort_values(["shared_edge_len", "added_area"], ascending=[False, True]).reset_index(drop=True)


def _apply_zero_uprn_attachments(
    sources: gpd.GeoDataFrame,
    *,
    max_iterations: int,
    min_shared_edge: float,
    max_added_area: float,
    max_target_area: float,
    max_after_area: float,
    max_added_source_count: int,
    min_target_shared_dominance: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    sources = sources.copy()
    selected_records: list[gpd.GeoDataFrame] = []
    used_added: set[int] = set()

    for iteration in range(1, int(max_iterations) + 1):
        components = _component_summary(sources)
        candidates = _build_zero_uprn_attachment_candidates(
            components,
            min_shared_edge=float(min_shared_edge),
            max_added_area=float(max_added_area),
            max_target_area=float(max_target_area),
            max_after_area=float(max_after_area),
            max_added_source_count=int(max_added_source_count),
            min_target_shared_dominance=float(min_target_shared_dominance),
        )
        if candidates.empty:
            break
        candidates = candidates[~candidates["added_component_id"].astype(int).isin(used_added)].copy()
        if candidates.empty:
            break

        kept_rows: list[pd.Series] = []
        claimed_added: set[int] = set()
        for _, row in candidates.iterrows():
            added_id = int(row["added_component_id"])
            if added_id in claimed_added:
                continue
            claimed_added.add(added_id)
            kept_rows.append(row)
        if not kept_rows:
            break
        selected = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=sources.crs).reset_index(drop=True)
        selected["zero_attach_iteration"] = int(iteration)
        selected_records.append(selected)

        for row in selected.itertuples(index=False):
            added_id = int(row.added_component_id)
            target_id = int(row.target_component_id)
            mask = sources["pred_component_id"].astype(int).eq(added_id)
            sources.loc[mask, "pred_component_id"] = target_id
            sources.loc[mask, "operation_status"] = "zero_uprn_attached"
            sources.loc[mask, "zero_attached_to_component"] = target_id
            sources.loc[mask, "zero_attach_iteration"] = int(iteration)
            sources.loc[mask, "zero_attach_shared_edge_len"] = float(row.shared_edge_len)
            used_added.add(added_id)

    if selected_records:
        return sources, pd.concat(selected_records, ignore_index=True)
    return sources, gpd.GeoDataFrame(geometry=[], crs=sources.crs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the operation-based WFS merge pipeline.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--prune-model-dir", default=DEFAULT_PRUNE_MODEL_DIR)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--prune-threshold", type=float, default=0.80)
    parser.add_argument("--max-prune-component-area", type=float, default=2000.0)
    parser.add_argument("--prune-source-role", default="land")
    parser.add_argument("--max-only-land-prune-source-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-prune-mrr-gain", type=float, default=0.10)
    parser.add_argument("--min-prune-hull-gap-reduction", type=float, default=0.08)
    parser.add_argument("--min-prune-regularity-gain", type=float, default=0.08)
    parser.add_argument("--min-after-prune-regularity", type=float, default=0.90)
    parser.add_argument("--max-after-prune-hull-gap", type=float, default=0.10)
    parser.add_argument("--disable-zero-uprn-attachment", action="store_true")
    parser.add_argument("--zero-attach-max-iterations", type=int, default=4)
    parser.add_argument("--zero-attach-min-shared-edge", type=float, default=3.0)
    parser.add_argument("--zero-attach-max-added-area", type=float, default=1000.0)
    parser.add_argument("--zero-attach-max-target-area", type=float, default=2000.0)
    parser.add_argument("--zero-attach-max-after-area", type=float, default=2000.0)
    parser.add_argument("--zero-attach-max-added-source-count", type=int, default=8)
    parser.add_argument("--zero-attach-min-target-shared-dominance", type=float, default=1.25)
    parser.add_argument("--disable-neighborhood-overmerge-split", action="store_true")
    parser.add_argument("--overmerge-split-max-component-area", type=float, default=2000.0)
    parser.add_argument("--overmerge-split-radius", type=float, default=75.0)
    parser.add_argument("--overmerge-split-min-local-regular-one-uprn", type=int, default=8)
    parser.add_argument("--overmerge-split-min-area-to-local-median", type=float, default=1.45)
    parser.add_argument("--overmerge-split-min-area-per-uprn-to-local-median", type=float, default=0.45)
    parser.add_argument("--overmerge-split-max-area-per-uprn-to-local-median", type=float, default=1.65)
    parser.add_argument("--overmerge-split-min-zero-orphan-attach-shared-edge", type=float, default=0.50)
    parser.add_argument("--disable-parcel-completion", action="store_true", default=True)
    parser.add_argument("--disable-group-completion", action="store_true", default=True, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    prune_model_dir = Path(args.prune_model_dir)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    _log(f"[INFO] Reading input prediction: {input_gpkg}")
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
    sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")
    edges = gpd.read_file(input_gpkg, layer="predicted_positive_edges", engine="pyogrio")
    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", sources.crs)
    excluded_gapfill_council_sources = _read_optional_layer(
        input_gpkg,
        "excluded_gapfill_council_sources",
        sources.crs,
    )
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", sources.crs)

    if "original_pred_component_id" not in sources.columns:
        sources = sources.copy()
        sources["original_pred_component_id"] = sources["pred_component_id"].astype(int)

    _log("[INFO] Building prune candidates")
    candidates = build_prune_candidates(predicted, sources)
    if candidates.empty:
        _log("[INFO] No prune candidates; carrying prediction forward")
        candidates = candidates.copy()
        candidates["prune_selected"] = pd.Series(dtype="int64")
        selected = candidates.iloc[0:0].copy()
    else:
        prune_model = joblib.load(prune_model_dir / PRUNE_MODEL_FILE_NAME)
        prune_meta = json.loads((prune_model_dir / "prune_metrics.json").read_text(encoding="utf-8"))
        feature_cols = prune_meta["feature_columns"]
        missing = sorted(set(feature_cols) - set(candidates.columns))
        if missing:
            raise RuntimeError(f"Prune candidates are missing model features: {missing}")
        candidates["prune_proba"] = prune_model.predict_proba(candidates[feature_cols])[:, 1]
        candidates["prune_pred_raw"] = candidates["prune_proba"].ge(float(args.prune_threshold)).astype(int)

        selected = _select_prune_operations(
            candidates,
            threshold=float(args.prune_threshold),
            max_component_area=float(args.max_prune_component_area),
            source_role=str(args.prune_source_role),
            max_only_land_prune_source_area_ratio=float(args.max_only_land_prune_source_area_ratio),
            min_mrr_gain=float(args.min_prune_mrr_gain),
            min_hull_gap_reduction=float(args.min_prune_hull_gap_reduction),
            min_regularity_gain=float(args.min_prune_regularity_gain),
            min_after_regularity=float(args.min_after_prune_regularity),
            max_after_hull_gap=float(args.max_after_prune_hull_gap),
        )
        selected_keys = {
            (int(row.component_id), int(row.source_fid))
            for row in selected.itertuples(index=False)
        }
        candidates["prune_selected"] = [
            int((int(row.component_id), int(row.source_fid)) in selected_keys)
            for row in candidates.itertuples(index=False)
        ]
    _log(f"[INFO] Prune candidates={len(candidates):,}; selected={len(selected):,}")

    sources = sources.copy()
    sources["operation_status"] = "base"
    sources["pruned_from_component"] = np.nan
    sources["prune_proba"] = np.nan

    max_component_id = int(sources["pred_component_id"].astype(int).max())
    if "source_fid" in selected.columns:
        selected_by_source = {int(row.source_fid): row for row in selected.itertuples(index=False)}
    else:
        selected_by_source = {}
    next_component_id = max_component_id + 1
    for source_fid, row in selected_by_source.items():
        mask = sources["source_fid"].astype(int).eq(source_fid)
        sources.loc[mask, "pruned_from_component"] = int(row.component_id)
        sources.loc[mask, "prune_proba"] = float(row.prune_proba)
        sources.loc[mask, "pred_component_id"] = int(next_component_id)
        sources.loc[mask, "operation_status"] = "pruned_to_singleton"
        next_component_id += 1

    zero_attachment_candidates = gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    if not bool(args.disable_zero_uprn_attachment):
        _log("[INFO] Applying zero-UPRN attachment")
        sources, zero_attachment_candidates = _apply_zero_uprn_attachments(
            sources,
            max_iterations=int(args.zero_attach_max_iterations),
            min_shared_edge=float(args.zero_attach_min_shared_edge),
            max_added_area=float(args.zero_attach_max_added_area),
            max_target_area=float(args.zero_attach_max_target_area),
            max_after_area=float(args.zero_attach_max_after_area),
            max_added_source_count=int(args.zero_attach_max_added_source_count),
            min_target_shared_dominance=float(args.zero_attach_min_target_shared_dominance),
        )
        _log(f"[INFO] Zero-UPRN attachments selected={len(zero_attachment_candidates):,}")

    neighborhood_split_components = gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    neighborhood_split_edges = gpd.GeoDataFrame(geometry=[], crs=edges.crs)
    neighborhood_split_results = gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    if not bool(args.disable_neighborhood_overmerge_split):
        _log("[INFO] Applying neighborhood modal overmerge split")
        sources, neighborhood_split_components, neighborhood_split_edges, neighborhood_split_results = (
            _apply_neighborhood_overmerge_splits(
                sources,
                edges,
                max_component_area=float(args.overmerge_split_max_component_area),
                radius=float(args.overmerge_split_radius),
                min_local_regular_one_uprn=int(args.overmerge_split_min_local_regular_one_uprn),
                min_area_to_local_median=float(args.overmerge_split_min_area_to_local_median),
                min_area_per_uprn_to_local_median=float(args.overmerge_split_min_area_per_uprn_to_local_median),
                max_area_per_uprn_to_local_median=float(args.overmerge_split_max_area_per_uprn_to_local_median),
                min_zero_orphan_attach_shared_edge=float(args.overmerge_split_min_zero_orphan_attach_shared_edge),
            )
        )
        selected_split_count = 0
        if not neighborhood_split_components.empty and "split_selected" in neighborhood_split_components.columns:
            selected_split_count = int(neighborhood_split_components["split_selected"].fillna(0).astype(int).sum())
        _log(
            "[INFO] Neighborhood modal overmerge components="
            f"{len(neighborhood_split_components):,}; selected={selected_split_count:,}; "
            f"new_children={len(neighborhood_split_results):,}"
        )

    parcel_completion_disabled = bool(args.disable_parcel_completion or args.disable_group_completion)
    if not parcel_completion_disabled:
        raise RuntimeError(
            "Parcel completion is not part of the production native pipeline. "
            "Use the anchor group repair stage instead, or restore the full parcel-training stack explicitly."
        )

    edges_new = _filter_edges_to_current_components(edges, sources)
    predicted_new = _build_predicted_parcels(sources, edges_new, predicted)
    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    merged_only_no_uprn = merged_only.drop(columns=["pred_uprn_count"])
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()

    prune_debug = candidates.copy()
    if "source_fid" in prune_debug.columns:
        source_geom = sources.set_index(sources["source_fid"].astype(int)).geometry
        prune_debug["geometry"] = prune_debug["source_fid"].astype(int).map(source_geom)
    else:
        prune_debug["geometry"] = gpd.GeoSeries([], crs=sources.crs)
    prune_debug = gpd.GeoDataFrame(prune_debug, geometry="geometry", crs=sources.crs)
    if "prune_selected" in prune_debug.columns:
        prune_removed = prune_debug[prune_debug["prune_selected"].eq(1)].copy()
    else:
        prune_removed = prune_debug.iloc[0:0].copy()

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(merged_only_no_uprn, output_gpkg, "predicted_parcels_merged_only")
    _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    _write_layer(sources, output_gpkg, "prediction_source_polygons")
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
    _write_layer(prune_removed, output_gpkg, "prune_removed_sources")
    _write_layer(prune_debug, output_gpkg, "prune_candidate_debug")
    _write_layer(zero_attachment_candidates, output_gpkg, "zero_uprn_attachment_candidates")
    _write_layer(neighborhood_split_components, output_gpkg, "neighborhood_overmerge_split_components")
    _write_layer(neighborhood_split_edges, output_gpkg, "neighborhood_overmerge_split_removed_edges")
    _write_layer(neighborhood_split_results, output_gpkg, "neighborhood_overmerge_split_results")

    summary = {
        "input_gpkg": str(input_gpkg),
        "output_gpkg": str(output_gpkg),
        "prune_model_dir": str(prune_model_dir),
        "prune_threshold": float(args.prune_threshold),
        "max_prune_component_area": float(args.max_prune_component_area),
        "prune_source_role": str(args.prune_source_role),
        "max_only_land_prune_source_area_ratio": float(args.max_only_land_prune_source_area_ratio),
        "min_prune_mrr_gain": float(args.min_prune_mrr_gain),
        "min_prune_hull_gap_reduction": float(args.min_prune_hull_gap_reduction),
        "min_prune_regularity_gain": float(args.min_prune_regularity_gain),
        "min_after_prune_regularity": float(args.min_after_prune_regularity),
        "max_after_prune_hull_gap": float(args.max_after_prune_hull_gap),
        "prune_candidate_rows": int(len(candidates)),
        "prune_selected_rows": int(len(selected)),
        "excluded_gapfill_council_sources": int(len(excluded_gapfill_council_sources)),
        "excluded_problem_sources": int(len(excluded_problem_sources)),
        "zero_uprn_attachment_disabled": bool(args.disable_zero_uprn_attachment),
        "zero_uprn_attachment_rows": int(len(zero_attachment_candidates)),
        "zero_attach_max_iterations": int(args.zero_attach_max_iterations),
        "zero_attach_min_shared_edge": float(args.zero_attach_min_shared_edge),
        "zero_attach_max_added_area": float(args.zero_attach_max_added_area),
        "zero_attach_max_target_area": float(args.zero_attach_max_target_area),
        "zero_attach_max_after_area": float(args.zero_attach_max_after_area),
        "zero_attach_max_added_source_count": int(args.zero_attach_max_added_source_count),
        "zero_attach_min_target_shared_dominance": float(args.zero_attach_min_target_shared_dominance),
        "neighborhood_overmerge_split_disabled": bool(args.disable_neighborhood_overmerge_split),
        "neighborhood_overmerge_split_component_candidates": int(len(neighborhood_split_components)),
        "neighborhood_overmerge_split_selected_components": int(
            neighborhood_split_components["split_selected"].fillna(0).astype(int).sum()
        )
        if not neighborhood_split_components.empty and "split_selected" in neighborhood_split_components.columns
        else 0,
        "neighborhood_overmerge_split_removed_edges": int(len(neighborhood_split_edges)),
        "neighborhood_overmerge_split_result_children": int(len(neighborhood_split_results)),
        "overmerge_split_max_component_area": float(args.overmerge_split_max_component_area),
        "overmerge_split_radius": float(args.overmerge_split_radius),
        "overmerge_split_min_local_regular_one_uprn": int(args.overmerge_split_min_local_regular_one_uprn),
        "overmerge_split_min_area_to_local_median": float(args.overmerge_split_min_area_to_local_median),
        "overmerge_split_min_area_per_uprn_to_local_median": float(
            args.overmerge_split_min_area_per_uprn_to_local_median
        ),
        "overmerge_split_max_area_per_uprn_to_local_median": float(
            args.overmerge_split_max_area_per_uprn_to_local_median
        ),
        "parcel_completion_disabled": bool(parcel_completion_disabled),
        "predicted_components": int(len(predicted_new)),
        "merged_only_components": int(len(merged_only)),
        "possible_false_positive_clusters": int(len(possible_fp)),
        "possible_split_reference_clusters": int(len(possible_split)),
        "merged_only_uprn_counts": merged_only["pred_uprn_count"].value_counts().sort_index().astype(int).to_dict(),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Operation pipeline complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

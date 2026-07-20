#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import deque
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

from train_wfs_merge_completion_model import _shape_metrics


CATEGORICAL_FEATURES = [
    "role_pair",
    "left_role",
    "right_role",
]

TARGET_COL = "label"

ID_COLS = {
    "component_id",
    "edge_fid",
    "left_source_fid",
    "right_source_fid",
    "left_merge_fid",
    "right_merge_fid",
    "edge_reference_label",
    "component_reference_count",
    "label_source",
    "manual_label",
    "manual_reason",
    "mid_x",
    "mid_y",
}


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _source_role(theme: object) -> str:
    text = str(theme or "").lower()
    if "building" in text:
        return "building"
    if "land" in text:
        return "land"
    return "other"


def _angle_delta(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return 45.0
    delta = abs(float(left) - float(right)) % 90.0
    return float(min(delta, 90.0 - delta))


def _mrr_orientation_deg(geom) -> float:
    if geom is None or shapely.is_empty(geom):
        return 0.0
    mrr = shapely.minimum_rotated_rectangle(geom)
    if not hasattr(mrr, "exterior"):
        return 0.0
    coords = list(mrr.exterior.coords)
    longest = (0.0, 0.0)
    for start, end in zip(coords, coords[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if length > longest[0]:
            longest = (length, math.degrees(math.atan2(dy, dx)) % 180.0)
    angle = float(longest[1])
    return angle - 90.0 if angle >= 90.0 else angle


def _update_prefixed(record: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        record[f"{prefix}_{key}"] = float(value)


def _component_graph(nodes: list[int], edges: pd.DataFrame, skip_edge_fid: int | None = None) -> dict[int, set[int]]:
    graph = {int(node): set() for node in nodes}
    for edge_fid, row in edges.iterrows():
        if skip_edge_fid is not None and int(edge_fid) == int(skip_edge_fid):
            continue
        left = int(row.left_source_fid)
        right = int(row.right_source_fid)
        if left not in graph or right not in graph:
            continue
        graph[left].add(right)
        graph[right].add(left)
    return graph


def _connected_parts(nodes: list[int], edges: pd.DataFrame, skip_edge_fid: int | None = None) -> list[list[int]]:
    graph = _component_graph(nodes, edges, skip_edge_fid=skip_edge_fid)
    seen: set[int] = set()
    parts: list[list[int]] = []
    for node in nodes:
        node = int(node)
        if node in seen:
            continue
        queue: deque[int] = deque([node])
        seen.add(node)
        part: list[int] = []
        while queue:
            current = queue.popleft()
            part.append(current)
            for neighbor in graph[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        parts.append(sorted(part))
    return parts


def _part_metrics(sources_by_fid: gpd.GeoDataFrame, source_fids: list[int]) -> dict[str, Any]:
    group = sources_by_fid.loc[source_fids]
    geom = shapely.union_all(group.geometry.array)
    metrics = _shape_metrics(geom)
    theme = group["Theme"].fillna("").astype(str)
    is_building = theme.str.contains("building", case=False, regex=False)
    is_land = theme.str.contains("land", case=False, regex=False)
    return {
        "source_count": int(len(group)),
        "uprn_count": int(group["source_uprn_count"].fillna(0).astype(int).sum()),
        "building_count": int(is_building.sum()),
        "land_count": int(is_land.sum()),
        "building_area": float(group.geometry[is_building].area.sum()) if bool(is_building.any()) else 0.0,
        "land_area": float(group.geometry[is_land].area.sum()) if bool(is_land.any()) else 0.0,
        "geometry": geom,
        "metrics": metrics,
    }


def _prepare_regular_local_index(predicted: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, shapely.STRtree | None]:
    needed = [
        "pred_uprn_count",
        "pred_area",
        "pred_regularity_score",
        "pred_mrr_ratio",
        "pred_hull_gap_ratio",
    ]
    if predicted.empty or any(column not in predicted.columns for column in needed):
        return gpd.GeoDataFrame(geometry=[], crs=predicted.crs), None
    regular = predicted[
        predicted["pred_uprn_count"].fillna(0).astype(int).eq(1)
        & predicted["pred_area"].astype(float).between(20.0, 2000.0)
        & predicted["pred_regularity_score"].astype(float).ge(0.85)
        & predicted["pred_mrr_ratio"].astype(float).ge(0.75)
        & predicted["pred_hull_gap_ratio"].astype(float).le(0.15)
    ].copy()
    if regular.empty:
        return regular, None
    regular["local_mrr_orientation_deg"] = regular.geometry.map(_mrr_orientation_deg)
    return regular.reset_index(drop=True), shapely.STRtree(regular.geometry.array)


def _local_mode_features(
    geom,
    orientation: float,
    regular: gpd.GeoDataFrame,
    tree: shapely.STRtree | None,
    *,
    radius: float,
    angle_tolerance: float,
) -> dict[str, float]:
    if tree is None or regular.empty or geom is None or shapely.is_empty(geom):
        return {
            "local_regular_count": 0.0,
            "local_aligned_count": 0.0,
            "local_area_median": 0.0,
            "local_area_q1": 0.0,
            "local_area_q3": 0.0,
            "local_area_iqr_ratio": 0.0,
            "local_area_cv": 0.0,
            "local_orientation_median_delta": 45.0,
        }
    buffered = shapely.buffer(geom, float(radius))
    idx = tree.query(buffered, predicate="intersects")
    if len(idx) == 0:
        return {
            "local_regular_count": 0.0,
            "local_aligned_count": 0.0,
            "local_area_median": 0.0,
            "local_area_q1": 0.0,
            "local_area_q3": 0.0,
            "local_area_iqr_ratio": 0.0,
            "local_area_cv": 0.0,
            "local_orientation_median_delta": 45.0,
        }
    local = regular.iloc[np.asarray(idx, dtype=int)].copy()
    distance = local.geometry.distance(geom)
    local = local[distance.le(float(radius))].copy()
    if local.empty:
        return {
            "local_regular_count": 0.0,
            "local_aligned_count": 0.0,
            "local_area_median": 0.0,
            "local_area_q1": 0.0,
            "local_area_q3": 0.0,
            "local_area_iqr_ratio": 0.0,
            "local_area_cv": 0.0,
            "local_orientation_median_delta": 45.0,
        }
    deltas = local["local_mrr_orientation_deg"].astype(float).map(lambda value: _angle_delta(float(value), orientation))
    aligned = local[deltas.le(float(angle_tolerance))].copy()
    use = aligned if not aligned.empty else local
    areas = use["pred_area"].astype(float)
    median = float(areas.median()) if len(areas) else 0.0
    q1 = float(areas.quantile(0.25)) if len(areas) else 0.0
    q3 = float(areas.quantile(0.75)) if len(areas) else 0.0
    mean = float(areas.mean()) if len(areas) else 0.0
    std = float(areas.std(ddof=0)) if len(areas) else 0.0
    return {
        "local_regular_count": float(len(local)),
        "local_aligned_count": float(len(aligned)),
        "local_area_median": median,
        "local_area_q1": q1,
        "local_area_q3": q3,
        "local_area_iqr_ratio": _safe_ratio(q3 - q1, median),
        "local_area_cv": _safe_ratio(std, mean),
        "local_orientation_median_delta": float(deltas.median()) if len(deltas) else 45.0,
    }


def _edge_line(left_geom, right_geom) -> LineString:
    left_point = left_geom.representative_point()
    right_point = right_geom.representative_point()
    return LineString(
        [
            (float(shapely.get_x(left_point)), float(shapely.get_y(left_point))),
            (float(shapely.get_x(right_point)), float(shapely.get_y(right_point))),
        ]
    )


def _shared_edge_len(left_geom, right_geom) -> float:
    return float(shapely.length(shapely.intersection(shapely.boundary(left_geom), shapely.boundary(right_geom))))


def _edge_records_for_component(
    comp_id: int,
    component_sources: gpd.GeoDataFrame,
    component_edges: gpd.GeoDataFrame,
    predicted_row: pd.Series | None,
    regular: gpd.GeoDataFrame,
    tree: shapely.STRtree | None,
    *,
    local_radius: float,
    local_angle_tolerance: float,
) -> list[dict[str, Any]]:
    nodes = sorted(component_sources["source_fid"].astype(int).tolist())
    if len(nodes) < 2 or component_edges.empty:
        return []
    base_parts = _connected_parts(nodes, component_edges)
    if len(base_parts) != 1:
        return []

    sources_by_fid = component_sources.set_index(component_sources["source_fid"].astype(int), drop=False)
    component_geom = (
        predicted_row.geometry
        if predicted_row is not None and getattr(predicted_row, "geometry", None) is not None
        else shapely.union_all(component_sources.geometry.array)
    )
    component_metrics = _shape_metrics(component_geom)
    component_orientation = _mrr_orientation_deg(component_geom)
    local = _local_mode_features(
        component_geom,
        component_orientation,
        regular,
        tree,
        radius=float(local_radius),
        angle_tolerance=float(local_angle_tolerance),
    )
    local_median = float(local.get("local_area_median", 0.0))
    component_uprn = int(component_sources["source_uprn_count"].fillna(0).astype(int).sum())
    component_area = float(component_metrics["area"])
    refs = sorted({int(v) for v in component_sources["reference_merge_fid"].dropna().astype(int)})

    records: list[dict[str, Any]] = []
    for edge_fid, edge in component_edges.iterrows():
        parts = _connected_parts(nodes, component_edges, skip_edge_fid=int(edge_fid))
        if len(parts) != 2:
            continue
        left_fid = int(edge.left_source_fid)
        right_fid = int(edge.right_source_fid)
        if left_fid not in sources_by_fid.index or right_fid not in sources_by_fid.index:
            continue

        part_infos = [_part_metrics(sources_by_fid, part) for part in parts]
        part_infos = sorted(part_infos, key=lambda item: float(item["metrics"]["area"]), reverse=True)
        large = part_infos[0]
        small = part_infos[1]
        side_areas = [float(info["metrics"]["area"]) for info in part_infos]
        side_uprns = [int(info["uprn_count"]) for info in part_infos]
        side_regs = [float(info["metrics"]["regularity_score"]) for info in part_infos]
        side_mrr = [float(info["metrics"]["mrr_ratio"]) for info in part_infos]
        side_hull = [float(info["metrics"]["hull_gap_ratio"]) for info in part_infos]
        area_per_uprn = [
            _safe_ratio(area, max(uprn, 1))
            for area, uprn in zip(side_areas, side_uprns)
        ]
        if local_median > 0.0:
            side_area_log_devs = [abs(math.log(max(area, 1e-9) / local_median)) for area in side_areas]
            side_area_per_uprn_log_devs = [
                abs(math.log(max(value, 1e-9) / local_median))
                for value in area_per_uprn
            ]
        else:
            side_area_log_devs = [0.0, 0.0]
            side_area_per_uprn_log_devs = [0.0, 0.0]

        left_source = sources_by_fid.loc[left_fid]
        right_source = sources_by_fid.loc[right_fid]
        left_geom = left_source.geometry
        right_geom = right_source.geometry
        shared_len = _shared_edge_len(left_geom, right_geom)
        left_role = _source_role(left_source.get("Theme", ""))
        right_role = _source_role(right_source.get("Theme", ""))
        left_perimeter = float(shapely.length(left_geom))
        right_perimeter = float(shapely.length(right_geom))
        left_area = float(shapely.area(left_geom))
        right_area = float(shapely.area(right_geom))
        left_uprn = _safe_int(left_source.get("source_uprn_count", 0))
        right_uprn = _safe_int(right_source.get("source_uprn_count", 0))

        left_merge = int(edge.left_merge_fid) if pd.notna(getattr(edge, "left_merge_fid", np.nan)) else -1
        right_merge = int(edge.right_merge_fid) if pd.notna(getattr(edge, "right_merge_fid", np.nan)) else -1
        edge_label = int(edge.label) if pd.notna(getattr(edge, "label", np.nan)) else -1
        record: dict[str, Any] = {
            "component_id": int(comp_id),
            "edge_fid": int(edge_fid),
            "left_source_fid": left_fid,
            "right_source_fid": right_fid,
            "left_merge_fid": left_merge,
            "right_merge_fid": right_merge,
            "edge_reference_label": edge_label,
            "edge_model_proba": float(getattr(edge, "model_proba", np.nan)),
            "role_pair": f"{left_role}__{right_role}",
            "left_role": left_role,
            "right_role": right_role,
            "left_uprn_count": left_uprn,
            "right_uprn_count": right_uprn,
            "left_area": left_area,
            "right_area": right_area,
            "edge_shared_edge_len": shared_len,
            "edge_shared_ratio_left_perimeter": _safe_ratio(shared_len, left_perimeter),
            "edge_shared_ratio_right_perimeter": _safe_ratio(shared_len, right_perimeter),
            "edge_shared_ratio_min_perimeter": _safe_ratio(shared_len, min(left_perimeter, right_perimeter)),
            "edge_area_balance": _safe_ratio(min(left_area, right_area), max(left_area, right_area)),
            "edge_both_have_uprn": int(left_uprn > 0 and right_uprn > 0),
            "edge_one_has_uprn": int((left_uprn > 0) ^ (right_uprn > 0)),
            "edge_neither_has_uprn": int(left_uprn == 0 and right_uprn == 0),
            "component_source_count": int(len(component_sources)),
            "component_uprn_count": component_uprn,
            "component_reference_count": int(len(refs)),
            "component_area_to_local_median": _safe_ratio(component_area, local_median),
            "component_area_per_uprn_to_local_median": _safe_ratio(_safe_ratio(component_area, max(component_uprn, 1)), local_median),
            "split_large_area": float(large["metrics"]["area"]),
            "split_small_area": float(small["metrics"]["area"]),
            "split_area_balance": _safe_ratio(min(side_areas), max(side_areas)),
            "split_large_uprn_count": int(large["uprn_count"]),
            "split_small_uprn_count": int(small["uprn_count"]),
            "split_min_uprn_count": int(min(side_uprns)),
            "split_max_uprn_count": int(max(side_uprns)),
            "split_both_sides_have_uprn": int(min(side_uprns) > 0),
            "split_min_regularity_score": float(min(side_regs)),
            "split_mean_regularity_score": float(np.mean(side_regs)),
            "split_min_mrr_ratio": float(min(side_mrr)),
            "split_max_hull_gap_ratio": float(max(side_hull)),
            "split_area_to_local_median_mean": _safe_ratio(float(np.mean(side_areas)), local_median),
            "split_area_to_local_median_min": _safe_ratio(float(min(side_areas)), local_median),
            "split_area_to_local_median_max": _safe_ratio(float(max(side_areas)), local_median),
            "split_area_log_dev_mean": float(np.mean(side_area_log_devs)),
            "split_area_log_dev_max": float(max(side_area_log_devs)),
            "split_area_per_uprn_log_dev_mean": float(np.mean(side_area_per_uprn_log_devs)),
            "split_area_per_uprn_log_dev_max": float(max(side_area_per_uprn_log_devs)),
            "split_area_per_uprn_to_local_median_min": _safe_ratio(float(min(area_per_uprn)), local_median),
            "split_area_per_uprn_to_local_median_max": _safe_ratio(float(max(area_per_uprn)), local_median),
            "mid_x": float(shapely.get_x(shapely.centroid(component_geom))),
            "mid_y": float(shapely.get_y(shapely.centroid(component_geom))),
            "geometry": _edge_line(left_geom, right_geom),
        }
        _update_prefixed(record, "component", component_metrics)
        for key, value in local.items():
            record[key] = float(value)
        records.append(record)
    return records


def build_overmerge_split_candidates(
    predicted: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    *,
    max_component_area: float = 2000.0,
    max_component_source_count: int = 30,
    min_component_uprn_count: int = 2,
    local_radius: float = 100.0,
    local_angle_tolerance: float = 15.0,
) -> gpd.GeoDataFrame:
    if predicted.empty or sources.empty or edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs=sources.crs if not sources.empty else predicted.crs)

    sources = sources.copy()
    edges = edges.copy()
    if "source_uprn_count" not in sources.columns:
        sources["source_uprn_count"] = 0
    if "Theme" not in sources.columns:
        sources["Theme"] = ""
    if "reference_merge_fid" not in sources.columns:
        sources["reference_merge_fid"] = np.nan
    if "pred_component_id" not in sources.columns:
        raise ValueError("sources is missing pred_component_id")

    predicted_by_component = predicted.set_index(predicted["pred_component_id"].astype(int), drop=False)
    regular, tree = _prepare_regular_local_index(predicted)
    records: list[dict[str, Any]] = []

    component_sizes = sources.groupby(sources["pred_component_id"].astype(int)).size()
    component_uprn = sources.groupby(sources["pred_component_id"].astype(int))["source_uprn_count"].sum()
    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    if "pred_component_id" not in edges.columns:
        left_comp = edges["left_source_fid"].astype(int).map(source_to_component)
        right_comp = edges["right_source_fid"].astype(int).map(source_to_component)
        edges["pred_component_id"] = np.where(left_comp.eq(right_comp), left_comp, -1)
    edges = edges[edges["pred_component_id"].fillna(-1).astype(int).ge(0)].copy()

    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        comp_id = int(comp_id)
        if int(component_sizes.get(comp_id, 0)) < 2:
            continue
        if int(component_sizes.get(comp_id, 0)) > int(max_component_source_count):
            continue
        if int(component_uprn.get(comp_id, 0)) < int(min_component_uprn_count):
            continue
        predicted_row = predicted_by_component.loc[comp_id] if comp_id in predicted_by_component.index else None
        comp_area = (
            float(predicted_row.get("pred_area", np.nan))
            if predicted_row is not None and pd.notna(predicted_row.get("pred_area", np.nan))
            else float(shapely.area(shapely.union_all(group.geometry.array)))
        )
        if comp_area > float(max_component_area):
            continue
        component_edges = edges[edges["pred_component_id"].fillna(-1).astype(int).eq(comp_id)].copy()
        if component_edges.empty:
            continue
        records.extend(
            _edge_records_for_component(
                comp_id,
                group,
                component_edges,
                predicted_row,
                regular,
                tree,
                local_radius=float(local_radius),
                local_angle_tolerance=float(local_angle_tolerance),
            )
        )

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    out = gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)
    return out.reset_index(drop=True)


def feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLS | {TARGET_COL, "sample_weight", "geometry"}
    feature_cols = [column for column in dataset.columns if column not in excluded]
    categorical_cols = [column for column in CATEGORICAL_FEATURES if column in feature_cols]
    numeric_cols = [
        column
        for column in feature_cols
        if column not in categorical_cols and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    feature_cols = numeric_cols + categorical_cols
    return feature_cols, numeric_cols, categorical_cols

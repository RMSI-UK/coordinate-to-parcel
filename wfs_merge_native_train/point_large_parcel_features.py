from __future__ import annotations

import math
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import Point

from train_wfs_merge_completion_model import _shape_metrics


ID_COLUMNS = {
    "candidate_fids",
    "seed_fid",
    "label",
    "label_source",
    "sample_weight",
}
CATEGORICAL_FEATURES = ["seed_role", "dominant_role"]


def log(message: str) -> None:
    print(message, flush=True)


def safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def ids_text(values: set[int] | frozenset[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(v)) for v in sorted(values))


def parse_fid_groups(text: object) -> list[set[int]]:
    groups: list[set[int]] = []
    for group_text in str(text or "").split(";"):
        ids: set[int] = set()
        for part in group_text.replace(",", "|").split("|"):
            part = part.strip()
            if not part:
                continue
            ids.add(int(part))
        if len(ids) >= 2:
            groups.append(ids)
    return groups


def _clean_text(value: object) -> str:
    return str(value or "").strip().lower()


def classify_wfs_role(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    theme = _clean_text(row.get("Theme", ""))
    group = _clean_text(row.get("DescriptiveGroup", ""))
    term = _clean_text(row.get("DescriptiveTerm", ""))
    make = _clean_text(row.get("Make", ""))
    text = " ".join([theme, group, term, make])

    is_building = "building" in group or "building" in theme
    is_land = "land" in theme
    is_road_theme = "roads tracks and paths" in theme
    is_road_or_track = is_road_theme and ("road or track" in group or "carriageway" in text)
    is_roadside = is_road_theme and "roadside" in group
    is_path = is_road_theme and ("path" in group or "path" in term)
    is_water = "water" in theme or "watercourse" in group or "watercourse" in term
    is_structure = "structure" in theme or "structure" in group

    if is_road_or_track:
        role = "hard_road"
    elif is_building:
        role = "building"
    elif is_roadside:
        role = "roadside"
    elif is_path:
        role = "path"
    elif is_water:
        role = "water"
    elif is_land:
        role = "land"
    elif is_structure:
        role = "structure"
    else:
        role = "other"

    return {
        "wfs_role": role,
        "is_building": int(is_building),
        "is_land": int(is_land),
        "is_roadside": int(is_roadside),
        "is_path": int(is_path),
        "is_water": int(is_water),
        "is_structure": int(is_structure),
        "is_hard_road": int(is_road_or_track),
        "is_boundary_support": int(is_road_or_track or is_water),
        "is_candidate_traversable": int(not is_road_or_track),
    }


@dataclass(frozen=True)
class PointLargeParcelContext:
    seed_fid: int
    seed_geometry: Any
    local: gpd.GeoDataFrame
    edges: pd.DataFrame
    adjacency: dict[int, list[tuple[int, float]]]
    shared_by_pair: dict[tuple[int, int], float]


def _with_wfs_fid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "wfs_fid" not in out.columns:
        out["wfs_fid"] = out.index.astype(int)
    out["wfs_fid"] = out["wfs_fid"].astype(int)
    return out


def _make_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.geometry = shapely.make_valid(out.geometry.array)
    out = out[out.geometry.notna() & ~shapely.is_empty(out.geometry.array)].copy()
    return out


def read_seed_by_point(
    *,
    wfs_gpkg: Path,
    wfs_layer: str,
    point_x: float,
    point_y: float,
    point_buffer: float,
) -> int:
    point = Point(float(point_x), float(point_y))
    bbox = (
        float(point_x) - float(point_buffer),
        float(point_y) - float(point_buffer),
        float(point_x) + float(point_buffer),
        float(point_y) + float(point_buffer),
    )
    local = pyogrio.read_dataframe(wfs_gpkg, layer=wfs_layer, bbox=bbox, fid_as_index=True)
    local = _with_wfs_fid(_make_valid_geometries(local))
    if local.empty:
        raise RuntimeError(f"No WFS polygons found near point ({point_x}, {point_y}).")
    covers = shapely.covers(local.geometry.array, point)
    candidates = local.loc[np.asarray(covers, dtype=bool)].copy()
    if candidates.empty:
        intersects = shapely.intersects(local.geometry.array, point)
        candidates = local.loc[np.asarray(intersects, dtype=bool)].copy()
    if candidates.empty:
        raise RuntimeError(f"No WFS polygon covers point ({point_x}, {point_y}).")
    candidates["area"] = candidates.geometry.area.astype(float)
    return int(candidates.sort_values(["area", "wfs_fid"]).iloc[0]["wfs_fid"])


def read_local_context(
    *,
    wfs_gpkg: Path,
    wfs_layer: str,
    seed_fid: int,
    local_buffer: float,
    min_shared_edge: float,
) -> PointLargeParcelContext:
    seed = pyogrio.read_dataframe(wfs_gpkg, layer=wfs_layer, fids=[int(seed_fid)], fid_as_index=True)
    seed = _with_wfs_fid(_make_valid_geometries(seed))
    if seed.empty:
        raise RuntimeError(f"Seed fid {seed_fid} was not found in {wfs_gpkg}:{wfs_layer}.")
    seed_geometry = seed.geometry.iloc[0]
    bbox_geom = shapely.buffer(seed_geometry, float(local_buffer))
    bbox = tuple(float(v) for v in shapely.bounds(bbox_geom))
    local = pyogrio.read_dataframe(wfs_gpkg, layer=wfs_layer, bbox=bbox, fid_as_index=True)
    local = _with_wfs_fid(_make_valid_geometries(local))
    local = local.loc[np.asarray(shapely.intersects(local.geometry.array, bbox_geom), dtype=bool)].copy()
    role_records = [classify_wfs_role(row) for _, row in local.iterrows()]
    roles = pd.DataFrame(role_records, index=local.index)
    local = pd.concat([local, roles], axis=1)
    local["wfs_area"] = local.geometry.area.astype(float)
    local["wfs_perimeter"] = local.geometry.length.astype(float)

    edges = build_local_shared_edges(local, min_shared_edge=float(min_shared_edge))
    adjacency, shared_by_pair = adjacency_from_edges(edges)
    return PointLargeParcelContext(
        seed_fid=int(seed_fid),
        seed_geometry=seed_geometry,
        local=local,
        edges=edges,
        adjacency=adjacency,
        shared_by_pair=shared_by_pair,
    )


def build_local_shared_edges(local: gpd.GeoDataFrame, *, min_shared_edge: float) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])
    work = local.reset_index(drop=True)
    geoms = work.geometry
    fids = work["wfs_fid"].astype(int).to_numpy()
    left_pos, right_pos = work.sindex.query(geoms.array, predicate="intersects")
    if len(left_pos) == 0:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])
    keep = left_pos < right_pos
    left_pos = left_pos[keep]
    right_pos = right_pos[keep]
    if len(left_pos) == 0:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])
    shared = shapely.length(
        shapely.intersection(
            shapely.boundary(geoms.iloc[left_pos].array),
            shapely.boundary(geoms.iloc[right_pos].array),
        )
    )
    edges = pd.DataFrame(
        {
            "left_fid": fids[left_pos].astype(int),
            "right_fid": fids[right_pos].astype(int),
            "shared_edge_len": np.asarray(shared, dtype="float64"),
        }
    )
    edges = edges[edges["shared_edge_len"].ge(float(min_shared_edge))].copy()
    if edges.empty:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])
    edges["left_fid"], edges["right_fid"] = (
        np.minimum(edges["left_fid"], edges["right_fid"]).astype(int),
        np.maximum(edges["left_fid"], edges["right_fid"]).astype(int),
    )
    return edges.drop_duplicates(["left_fid", "right_fid"]).reset_index(drop=True)


def adjacency_from_edges(edges: pd.DataFrame) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    adjacency: dict[int, list[tuple[int, float]]] = {}
    shared_by_pair: dict[tuple[int, int], float] = {}
    for row in edges.itertuples(index=False):
        left = int(row.left_fid)
        right = int(row.right_fid)
        shared = float(row.shared_edge_len)
        adjacency.setdefault(left, []).append((right, shared))
        adjacency.setdefault(right, []).append((left, shared))
        shared_by_pair[(min(left, right), max(left, right))] = shared
    for fid, values in list(adjacency.items()):
        adjacency[fid] = sorted(values, key=lambda item: (-float(item[1]), int(item[0])))
    return adjacency, shared_by_pair


def _dominant_role(local_group: gpd.GeoDataFrame) -> str:
    if local_group.empty:
        return "unknown"
    areas = local_group.geometry.area.astype(float)
    by_role = areas.groupby(local_group["wfs_role"].astype(str)).sum()
    if by_role.empty:
        return "unknown"
    return str(by_role.sort_values(ascending=False).index[0])


def _edge_internal_shared(fids: frozenset[int], edges: pd.DataFrame) -> float:
    if not fids or edges.empty:
        return 0.0
    in_left = edges["left_fid"].isin(fids)
    in_right = edges["right_fid"].isin(fids)
    return float(edges.loc[in_left & in_right, "shared_edge_len"].sum())


def _external_edge_stats(fids: frozenset[int], context: PointLargeParcelContext) -> dict[str, float]:
    edges = context.edges
    if edges.empty:
        return {
            "external_shared_len": 0.0,
            "hard_boundary_shared_len": 0.0,
            "hard_road_boundary_shared_len": 0.0,
            "water_boundary_shared_len": 0.0,
            "external_neighbor_count": 0.0,
            "hard_boundary_neighbor_count": 0.0,
        }
    left_in = edges["left_fid"].isin(fids)
    right_in = edges["right_fid"].isin(fids)
    external = edges.loc[left_in ^ right_in].copy()
    if external.empty:
        return {
            "external_shared_len": 0.0,
            "hard_boundary_shared_len": 0.0,
            "hard_road_boundary_shared_len": 0.0,
            "water_boundary_shared_len": 0.0,
            "external_neighbor_count": 0.0,
            "hard_boundary_neighbor_count": 0.0,
        }
    external["outside_fid"] = np.where(
        external["left_fid"].isin(fids),
        external["right_fid"],
        external["left_fid"],
    ).astype(int)
    attrs = context.local.set_index("wfs_fid")[
        ["is_boundary_support", "is_hard_road", "is_water"]
    ].to_dict("index")
    external["outside_boundary_support"] = external["outside_fid"].map(
        lambda fid: int(attrs.get(int(fid), {}).get("is_boundary_support", 0))
    )
    external["outside_hard_road"] = external["outside_fid"].map(
        lambda fid: int(attrs.get(int(fid), {}).get("is_hard_road", 0))
    )
    external["outside_water"] = external["outside_fid"].map(
        lambda fid: int(attrs.get(int(fid), {}).get("is_water", 0))
    )
    hard = external["outside_boundary_support"].eq(1)
    hard_road = external["outside_hard_road"].eq(1)
    water = external["outside_water"].eq(1)
    return {
        "external_shared_len": float(external["shared_edge_len"].sum()),
        "hard_boundary_shared_len": float(external.loc[hard, "shared_edge_len"].sum()),
        "hard_road_boundary_shared_len": float(external.loc[hard_road, "shared_edge_len"].sum()),
        "water_boundary_shared_len": float(external.loc[water, "shared_edge_len"].sum()),
        "external_neighbor_count": float(external["outside_fid"].nunique()),
        "hard_boundary_neighbor_count": float(external.loc[hard, "outside_fid"].nunique()),
    }


def candidate_features(
    fids: frozenset[int],
    *,
    context: PointLargeParcelContext,
    label: int | None = None,
    label_source: str = "",
    sample_weight: float = 1.0,
) -> dict[str, Any]:
    ordered = sorted(int(fid) for fid in fids)
    local_by_fid = context.local.set_index("wfs_fid", drop=False)
    missing = [fid for fid in ordered if fid not in local_by_fid.index]
    if missing:
        raise ValueError(f"Candidate includes fids outside local context: {missing[:5]}")

    group = local_by_fid.loc[ordered].copy()
    geoms = [geom for geom in group.geometry]
    union_geom = shapely.union_all(geoms)
    seed_geom = context.seed_geometry
    group_shape = _shape_metrics(union_geom)
    seed_shape = _shape_metrics(seed_geom)
    areas = group.geometry.area.astype(float).to_numpy()
    perimeters = group.geometry.length.astype(float).to_numpy()
    perimeter_sum = float(np.sum(perimeters))
    area_sum = float(np.sum(areas))
    internal_shared = _edge_internal_shared(fids, context.edges)
    external = _external_edge_stats(fids, context)

    role_area = group.groupby("wfs_role").geometry.apply(lambda values: float(values.area.sum()))
    role_count = group["wfs_role"].astype(str).value_counts()
    largest_area = float(np.max(areas)) if len(areas) else 0.0
    hard_road_area = float(group.loc[group["is_hard_road"].eq(1), "wfs_area"].sum())

    hard_boundary_ratio = safe_ratio(external["hard_boundary_shared_len"], group_shape["perimeter"])
    hard_road_boundary_ratio = safe_ratio(external["hard_road_boundary_shared_len"], group_shape["perimeter"])
    water_boundary_ratio = safe_ratio(external["water_boundary_shared_len"], group_shape["perimeter"])
    soft_external_shared_len = max(float(external["external_shared_len"]) - float(external["hard_boundary_shared_len"]), 0.0)
    soft_external_shared_ratio = safe_ratio(soft_external_shared_len, group_shape["perimeter"])
    hard_boundary_to_external_shared_ratio = safe_ratio(
        external["hard_boundary_shared_len"],
        external["external_shared_len"],
    )
    boundary_simplification = safe_ratio(perimeter_sum - group_shape["perimeter"], perimeter_sum)
    regularity_gain = float(group_shape["regularity_score"] - seed_shape["regularity_score"])
    hull_gap_reduction = float(seed_shape["hull_gap_ratio"] - group_shape["hull_gap_ratio"])
    mrr_ratio_gain = float(group_shape["mrr_ratio"] - seed_shape["mrr_ratio"])
    perimeter_reduction_ratio = safe_ratio(seed_shape["perimeter"] - group_shape["perimeter"], seed_shape["perimeter"])
    area_ratio_vs_seed = safe_ratio(group_shape["area"], seed_shape["area"])
    external_neighbor_density = safe_ratio(external["external_neighbor_count"], len(ordered))

    heuristic_score = (
        2.0 * hard_boundary_ratio
        + 1.0 * hard_boundary_to_external_shared_ratio
        + 1.0 * boundary_simplification
        + 1.0 * hull_gap_reduction
        + 0.6 * regularity_gain
        + 0.08 * safe_ratio(internal_shared, math.sqrt(max(group_shape["area"], 1.0)))
        - 0.8 * soft_external_shared_ratio
        - 0.2 * external_neighbor_density
        - 0.30 * max(area_ratio_vs_seed - 2.2, 0.0)
        - 1.2 * safe_ratio(hard_road_area, max(area_sum, 1.0))
        - 0.03 * max(len(ordered) - 24, 0)
    )

    record: dict[str, Any] = {
        "seed_fid": int(context.seed_fid),
        "candidate_fids": ids_text(fids),
        "group_size": int(len(ordered)),
        "seed_role": str(local_by_fid.loc[int(context.seed_fid), "wfs_role"]),
        "dominant_role": _dominant_role(group),
        "area_sum_parts": float(area_sum),
        "area_ratio_vs_seed": float(area_ratio_vs_seed),
        "perimeter_sum_parts": float(perimeter_sum),
        "largest_part_area_ratio": safe_ratio(largest_area, area_sum),
        "small_part_area_ratio": safe_ratio(area_sum - largest_area, area_sum),
        "internal_shared_len": float(internal_shared),
        "internal_to_sqrt_area": safe_ratio(internal_shared, math.sqrt(max(group_shape["area"], 1.0))),
        "external_shared_len": float(external["external_shared_len"]),
        "external_neighbor_count": int(external["external_neighbor_count"]),
        "external_neighbor_density": float(external_neighbor_density),
        "hard_boundary_shared_len": float(external["hard_boundary_shared_len"]),
        "hard_road_boundary_shared_len": float(external["hard_road_boundary_shared_len"]),
        "water_boundary_shared_len": float(external["water_boundary_shared_len"]),
        "hard_boundary_neighbor_count": int(external["hard_boundary_neighbor_count"]),
        "hard_boundary_ratio": float(hard_boundary_ratio),
        "hard_road_boundary_ratio": float(hard_road_boundary_ratio),
        "water_boundary_ratio": float(water_boundary_ratio),
        "soft_external_shared_len": float(soft_external_shared_len),
        "soft_external_shared_ratio": float(soft_external_shared_ratio),
        "hard_boundary_to_external_shared_ratio": float(hard_boundary_to_external_shared_ratio),
        "boundary_simplification": float(boundary_simplification),
        "regularity_gain_vs_seed": float(regularity_gain),
        "hull_gap_reduction_vs_seed": float(hull_gap_reduction),
        "mrr_ratio_gain_vs_seed": float(mrr_ratio_gain),
        "perimeter_reduction_ratio_vs_seed": float(perimeter_reduction_ratio),
        "hard_road_area_ratio": safe_ratio(hard_road_area, area_sum),
        "building_area_ratio": safe_ratio(float(role_area.get("building", 0.0)), area_sum),
        "land_area_ratio": safe_ratio(float(role_area.get("land", 0.0)), area_sum),
        "roadside_path_area_ratio": safe_ratio(float(role_area.get("roadside", 0.0)) + float(role_area.get("path", 0.0)), area_sum),
        "water_area_ratio": safe_ratio(float(role_area.get("water", 0.0)), area_sum),
        "building_count": int(role_count.get("building", 0)),
        "land_count": int(role_count.get("land", 0)),
        "roadside_path_count": int(role_count.get("roadside", 0) + role_count.get("path", 0)),
        "water_count": int(role_count.get("water", 0)),
        "hard_road_count": int(role_count.get("hard_road", 0)),
        "heuristic_score": float(heuristic_score),
        "geometry": union_geom,
    }
    for key, value in seed_shape.items():
        record[f"seed_{key}"] = float(value)
    for key, value in group_shape.items():
        record[f"group_{key}"] = float(value)
    if label is not None:
        record["label"] = int(label)
        record["label_source"] = str(label_source)
        record["sample_weight"] = float(sample_weight)
    return record


def _frontier_fids(
    fids: frozenset[int],
    *,
    context: PointLargeParcelContext,
    allowed_fids: set[int],
    top_frontier: int,
) -> list[int]:
    frontier: dict[int, float] = {}
    for fid in fids:
        for neighbor, shared in context.adjacency.get(int(fid), []):
            neighbor = int(neighbor)
            if neighbor in fids or neighbor not in allowed_fids:
                continue
            frontier[neighbor] = max(float(shared), frontier.get(neighbor, 0.0))
    return [
        int(fid)
        for fid, _ in sorted(frontier.items(), key=lambda item: (-float(item[1]), int(item[0])))[: int(top_frontier)]
    ]


def _is_connected_group(fids: frozenset[int], adjacency: dict[int, list[tuple[int, float]]]) -> bool:
    if len(fids) <= 1:
        return True
    start = next(iter(fids))
    visited = {int(start)}
    stack = [int(start)]
    while stack:
        fid = stack.pop()
        for neighbor, _shared in adjacency.get(fid, []):
            neighbor = int(neighbor)
            if neighbor in fids and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(fids)


def _refine_group_by_full_score(
    group: frozenset[int],
    *,
    context: PointLargeParcelContext,
    allowed_fids: set[int],
    area_by_fid: dict[int, float],
    max_group_size: int,
    max_group_area: float,
    top_frontier: int,
    max_steps: int,
    score_cache: dict[frozenset[int], float],
) -> set[frozenset[int]]:
    def score(candidate: frozenset[int]) -> float:
        if candidate not in score_cache:
            score_cache[candidate] = float(candidate_features(candidate, context=context)["heuristic_score"])
        return score_cache[candidate]

    current = frozenset(group)
    emitted: set[frozenset[int]] = {current}
    current_score = score(current)
    for _ in range(int(max_steps)):
        moves: list[frozenset[int]] = []
        for fid in sorted(current):
            if int(fid) == int(context.seed_fid):
                continue
            removed = frozenset(v for v in current if int(v) != int(fid))
            if len(removed) >= 1 and _is_connected_group(removed, context.adjacency):
                moves.append(removed)
        for neighbor in _frontier_fids(
            current,
            context=context,
            allowed_fids=allowed_fids,
            top_frontier=int(top_frontier),
        ):
            added = frozenset(set(current) | {int(neighbor)})
            if len(added) > int(max_group_size):
                continue
            area = sum(float(area_by_fid.get(fid, 0.0)) for fid in added)
            if area <= float(max_group_area):
                moves.append(added)
        if not moves:
            break
        best = max(moves, key=score)
        best_score = score(best)
        emitted.add(best)
        if best_score <= current_score + 1e-6:
            break
        current = best
        current_score = best_score
    return emitted


def _pocket_completion_groups(
    group: frozenset[int],
    *,
    context: PointLargeParcelContext,
    allowed_fids: set[int],
    area_by_fid: dict[int, float],
    max_group_size: int,
    max_group_area: float,
    max_pocket_area: float,
    min_pocket_shared: float,
    max_pocket_frontier: int,
    pocket_exclusion_depth: int,
    boundary_exclusion_depth: int,
) -> set[frozenset[int]]:
    frontier_shared: dict[int, float] = {}
    for fid in group:
        for neighbor, shared in context.adjacency.get(int(fid), []):
            neighbor = int(neighbor)
            if neighbor in group or neighbor not in allowed_fids:
                continue
            if float(area_by_fid.get(neighbor, 0.0)) > float(max_pocket_area):
                continue
            frontier_shared[neighbor] = frontier_shared.get(neighbor, 0.0) + float(shared)
    pocket = [
        int(fid)
        for fid, shared in sorted(frontier_shared.items(), key=lambda item: (-float(item[1]), int(item[0])))
        if float(shared) >= float(min_pocket_shared)
    ][: int(max_pocket_frontier)]
    if not pocket:
        return set()

    local_by_fid = context.local.set_index("wfs_fid", drop=False)
    removable_boundary = [
        int(fid)
        for fid in group
        if int(fid) != int(context.seed_fid)
        and int(local_by_fid.loc[int(fid), "is_boundary_support"]) == 1
    ][:6]

    out: set[frozenset[int]] = set()
    pocket_exclusions: list[tuple[int, ...]] = [()]
    boundary_exclusions: list[tuple[int, ...]] = [()]
    for depth in range(1, min(int(pocket_exclusion_depth), len(pocket)) + 1):
        pocket_exclusions.extend(combinations(pocket, depth))
    for depth in range(1, min(int(boundary_exclusion_depth), len(removable_boundary)) + 1):
        boundary_exclusions.extend(combinations(removable_boundary, depth))
    for excluded_boundary in boundary_exclusions:
        for excluded_pocket in pocket_exclusions:
            candidate = frozenset((set(group) | set(pocket)) - set(excluded_boundary) - set(excluded_pocket))
            if len(candidate) > int(max_group_size):
                continue
            candidate_area = sum(float(area_by_fid.get(fid, 0.0)) for fid in candidate)
            if candidate_area > float(max_group_area):
                continue
            if _is_connected_group(candidate, context.adjacency):
                out.add(candidate)
    return out


def generate_seed_candidates(
    context: PointLargeParcelContext,
    *,
    max_group_size: int,
    max_group_area: float,
    beam_width: int,
    top_frontier: int,
    max_candidates: int,
    refine_top_groups: int,
    refine_max_steps: int,
    refine_frontier: int,
    pocket_max_area: float,
    pocket_min_shared: float,
    pocket_max_frontier: int,
    pocket_exclusion_depth: int,
    boundary_exclusion_depth: int,
    manual_positive_groups: list[set[int]] | None = None,
) -> set[frozenset[int]]:
    manual_positive_groups = manual_positive_groups or []
    local_by_fid = context.local.set_index("wfs_fid", drop=False)
    allowed = set(
        int(row.wfs_fid)
        for row in context.local.itertuples(index=False)
        if int(row.is_candidate_traversable) == 1
    )
    if int(context.seed_fid) not in allowed:
        allowed.add(int(context.seed_fid))

    area_by_fid = local_by_fid["wfs_area"].astype(float).to_dict()
    perimeter_by_fid = local_by_fid["wfs_perimeter"].astype(float).to_dict()
    boundary_support_fids = set(
        int(row.wfs_fid)
        for row in context.local.itertuples(index=False)
        if int(row.is_boundary_support) == 1
    )
    hard_road_fids = set(
        int(row.wfs_fid)
        for row in context.local.itertuples(index=False)
        if int(row.is_hard_road) == 1
    )
    start = frozenset({int(context.seed_fid)})
    candidates: set[frozenset[int]] = {start}
    beam: list[frozenset[int]] = [start]
    feature_cache: dict[frozenset[int], float] = {}

    def rank(group: frozenset[int]) -> float:
        if group not in feature_cache:
            area = sum(float(area_by_fid.get(fid, 0.0)) for fid in group)
            perimeter = sum(float(perimeter_by_fid.get(fid, 0.0)) for fid in group)
            internal_shared = 0.0
            hard_external = 0.0
            for fid in group:
                for neighbor, shared in context.adjacency.get(int(fid), []):
                    if neighbor in group:
                        if int(fid) < int(neighbor):
                            internal_shared += float(shared)
                    elif int(neighbor) in boundary_support_fids:
                        hard_external += float(shared)
            rough_boundary = max(perimeter - 2.0 * internal_shared, 1.0)
            included_hard_road_area = sum(float(area_by_fid.get(fid, 0.0)) for fid in group if fid in hard_road_fids)
            feature_cache[group] = float(
                1.25 * safe_ratio(2.0 * internal_shared, perimeter)
                + 0.85 * safe_ratio(hard_external, rough_boundary)
                + 0.30 * safe_ratio(internal_shared, math.sqrt(max(area, 1.0)))
                - 1.5 * safe_ratio(included_hard_road_area, max(area, 1.0))
                - 0.018 * max(len(group) - 16, 0)
            )
        return feature_cache[group]

    for _depth in range(1, int(max_group_size)):
        next_groups: set[frozenset[int]] = set()
        for group in beam:
            for neighbor in _frontier_fids(
                group,
                context=context,
                allowed_fids=allowed,
                top_frontier=int(top_frontier),
            ):
                new_group = frozenset(set(group) | {int(neighbor)})
                if new_group in candidates or new_group in next_groups:
                    continue
                group_area = sum(float(area_by_fid.get(fid, 0.0)) for fid in new_group)
                if group_area > float(max_group_area):
                    continue
                next_groups.add(new_group)
        if not next_groups:
            break
        ranked = sorted(next_groups, key=rank, reverse=True)
        beam = ranked[: int(beam_width)]
        candidates.update(beam)
        if len(candidates) >= int(max_candidates):
            break

    refined: set[frozenset[int]] = set()
    if refine_top_groups > 0 and candidates:
        full_score_cache: dict[frozenset[int], float] = {}
        seeds_for_refine = sorted(candidates, key=rank, reverse=True)[: int(refine_top_groups)]
        for group in seeds_for_refine:
            refined.update(
                _refine_group_by_full_score(
                    group,
                    context=context,
                    allowed_fids=allowed,
                    area_by_fid=area_by_fid,
                    max_group_size=int(max_group_size),
                    max_group_area=float(max_group_area),
                    top_frontier=int(refine_frontier),
                    max_steps=int(refine_max_steps),
                    score_cache=full_score_cache,
                )
            )
        candidates.update(refined)

    seed_area = float(area_by_fid.get(int(context.seed_fid), 0.0))
    compact_seed_groups = []
    for group in candidates:
        group_area = sum(float(area_by_fid.get(fid, 0.0)) for fid in group)
        if (
            4 <= len(group) <= 12
            and group_area <= float(max_group_area)
            and group_area >= max(seed_area * 1.05, seed_area + 1.0)
            and group_area <= max(seed_area * 2.6, seed_area + 1.0)
        ):
            compact_seed_groups.append(group)
    compact_seed_groups = sorted(compact_seed_groups, key=rank, reverse=True)[: max(int(refine_top_groups) * 50, 1000)]
    pocket_seed_groups = set(refined) | set(sorted(candidates, key=rank, reverse=True)[: int(refine_top_groups)]) | set(compact_seed_groups)
    pocket_groups: set[frozenset[int]] = set()
    for group in pocket_seed_groups:
        pocket_groups.update(
            _pocket_completion_groups(
                group,
                context=context,
                allowed_fids=allowed,
                area_by_fid=area_by_fid,
                max_group_size=int(max_group_size),
                max_group_area=float(max_group_area),
                max_pocket_area=float(pocket_max_area),
                min_pocket_shared=float(pocket_min_shared),
                max_pocket_frontier=int(pocket_max_frontier),
                pocket_exclusion_depth=int(pocket_exclusion_depth),
                boundary_exclusion_depth=int(boundary_exclusion_depth),
            )
        )
    candidates.update(pocket_groups)

    post_pocket_refined: set[frozenset[int]] = set()
    if pocket_groups:
        full_score_cache = {}
        for group in sorted(pocket_groups, key=rank, reverse=True)[: int(refine_top_groups)]:
            post_pocket_refined.update(
                _refine_group_by_full_score(
                    group,
                    context=context,
                    allowed_fids=allowed,
                    area_by_fid=area_by_fid,
                    max_group_size=int(max_group_size),
                    max_group_area=float(max_group_area),
                    top_frontier=int(refine_frontier),
                    max_steps=min(int(refine_max_steps), 10),
                    score_cache=full_score_cache,
                )
            )
    candidates.update(post_pocket_refined)

    valid = set(context.local["wfs_fid"].astype(int))
    manual_keep: set[frozenset[int]] = set()
    for manual_group in manual_positive_groups:
        group = frozenset(int(fid) for fid in manual_group if int(fid) in valid)
        if len(group) >= 2:
            candidates.add(group)
            manual_keep.add(group)

    if len(candidates) > int(max_candidates):
        structural_keep: set[frozenset[int]] = set()
        for group in candidates:
            group_area = sum(float(area_by_fid.get(fid, 0.0)) for fid in group)
            if (
                16 <= len(group) <= 24
                and group_area >= max(seed_area * 1.2, seed_area + 1.0)
                and group_area <= max(seed_area * 2.0, seed_area + 1.0)
            ):
                structural_keep.add(group)
        preserve = set(post_pocket_refined) | set(manual_keep) | structural_keep
        remaining_limit = max(int(max_candidates) - len(preserve), 0)
        ranked_remaining = sorted(candidates - preserve, key=rank, reverse=True)[:remaining_limit]
        candidates = preserve | set(ranked_remaining)

    return candidates


def build_point_large_parcel_candidates(
    *,
    wfs_gpkg: Path,
    wfs_layer: str,
    seed_fid: int,
    local_buffer: float,
    min_shared_edge: float,
    max_group_size: int,
    max_group_area: float,
    beam_width: int,
    top_frontier: int,
    max_candidates: int,
    refine_top_groups: int,
    refine_max_steps: int,
    refine_frontier: int,
    pocket_max_area: float,
    pocket_min_shared: float,
    pocket_max_frontier: int,
    pocket_exclusion_depth: int,
    boundary_exclusion_depth: int,
    manual_positive_groups: list[set[int]] | None = None,
    include_labels: bool = True,
) -> tuple[gpd.GeoDataFrame, PointLargeParcelContext]:
    manual_positive_groups = manual_positive_groups or []
    context = read_local_context(
        wfs_gpkg=wfs_gpkg,
        wfs_layer=wfs_layer,
        seed_fid=int(seed_fid),
        local_buffer=float(local_buffer),
        min_shared_edge=float(min_shared_edge),
    )
    log(f"[INFO] Local polygons={len(context.local):,}; shared edges={len(context.edges):,}")
    groups = generate_seed_candidates(
        context,
        max_group_size=int(max_group_size),
        max_group_area=float(max_group_area),
        beam_width=int(beam_width),
        top_frontier=int(top_frontier),
        max_candidates=int(max_candidates),
        refine_top_groups=int(refine_top_groups),
        refine_max_steps=int(refine_max_steps),
        refine_frontier=int(refine_frontier),
        pocket_max_area=float(pocket_max_area),
        pocket_min_shared=float(pocket_min_shared),
        pocket_max_frontier=int(pocket_max_frontier),
        pocket_exclusion_depth=int(pocket_exclusion_depth),
        boundary_exclusion_depth=int(boundary_exclusion_depth),
        manual_positive_groups=manual_positive_groups,
    )
    log(f"[INFO] Candidate groups={len(groups):,}")

    records: list[dict[str, Any]] = []
    manual_sets = [set(int(fid) for fid in group) for group in manual_positive_groups]
    for group in groups:
        label: int | None = None
        label_source = ""
        weight = 1.0
        if include_labels:
            if any(set(group) == manual for manual in manual_sets):
                label = 1
                label_source = "manual_complete_positive"
                weight = 80.0
            elif any(set(group) & manual for manual in manual_sets):
                label = 0
                label_source = "manual_partial_or_overmerge_negative"
                weight = 10.0
            else:
                label = 0
                label_source = "local_unknown_negative"
                weight = 1.0
        records.append(
            candidate_features(
                group,
                context=context,
                label=label,
                label_source=label_source,
                sample_weight=weight,
            )
        )
    candidates = gpd.GeoDataFrame(records, geometry="geometry", crs=context.local.crs)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["label", "heuristic_score", "group_size"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return candidates, context


def feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLUMNS | {"geometry"}
    candidates = [column for column in dataset.columns if column not in excluded]
    categorical = [column for column in CATEGORICAL_FEATURES if column in candidates]
    numeric = [
        column
        for column in candidates
        if column not in categorical and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    return numeric + categorical, numeric, categorical


def boundary_neighbors_for_group(
    fids: set[int] | frozenset[int],
    context: PointLargeParcelContext,
) -> gpd.GeoDataFrame:
    fids = frozenset(int(fid) for fid in fids)
    edges = context.edges
    if edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs=context.local.crs)
    left_in = edges["left_fid"].isin(fids)
    right_in = edges["right_fid"].isin(fids)
    external = edges.loc[left_in ^ right_in].copy()
    if external.empty:
        return gpd.GeoDataFrame(geometry=[], crs=context.local.crs)
    external["outside_fid"] = np.where(
        external["left_fid"].isin(fids),
        external["right_fid"],
        external["left_fid"],
    ).astype(int)
    shared = external.groupby("outside_fid", as_index=False)["shared_edge_len"].sum()
    local = context.local.merge(shared, left_on="wfs_fid", right_on="outside_fid", how="inner")
    return gpd.GeoDataFrame(local, geometry="geometry", crs=context.local.crs).sort_values(
        "shared_edge_len", ascending=False
    )

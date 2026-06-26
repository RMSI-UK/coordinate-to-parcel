from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from train_wfs_merge_completion_model import _shape_metrics


ID_COLUMNS = {
    "candidate_fids",
    "candidate_component_ids",
    "candidate_reference_fids",
    "reference_complete_fids",
    "reference_relation",
    "seed_fid",
    "label",
    "label_source",
    "sample_weight",
}
CATEGORICAL_FEATURES: list[str] = []


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


def read_predicted(input_gpkg: Path) -> gpd.GeoDataFrame:
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio", fid_as_index=True)
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted.index = predicted.index.astype(int)
    predicted["layer_fid"] = predicted.index.astype(int)
    predicted["pred_component_id"] = predicted["pred_component_id"].astype(int)
    predicted["pred_area"] = predicted.geometry.area.astype(float)
    predicted["pred_perimeter"] = predicted.geometry.length.astype(float)
    for column in ["pred_regularity_score", "pred_mrr_ratio", "pred_hull_gap_ratio", "pred_compactness"]:
        if column not in predicted.columns:
            predicted[column] = np.nan
    return predicted


def read_sources(input_gpkg: Path) -> gpd.GeoDataFrame:
    sources = pyogrio.read_dataframe(input_gpkg, layer="prediction_source_polygons")
    sources = sources[sources.geometry.notna() & ~sources.geometry.is_empty].copy()
    sources["source_fid"] = sources["source_fid"].astype(int)
    sources["pred_component_id"] = sources["pred_component_id"].astype(int)
    if "source_uprn_count" not in sources.columns:
        sources["source_uprn_count"] = sources.get("uprn_count", 0)
    if "reference_merge_fid" not in sources.columns:
        sources["reference_merge_fid"] = np.nan
    return sources


def build_seed_fids(
    predicted: gpd.GeoDataFrame,
    *,
    max_seed_area: float,
    include_all_under_area: bool,
) -> list[int]:
    under_area = predicted["pred_area"].astype(float).le(float(max_seed_area))
    if include_all_under_area:
        seed = under_area
    else:
        split = predicted["possible_split_reference"].fillna(0).astype(int).eq(1)
        small_source_count = predicted["source_count"].fillna(999).astype(int).le(4)
        shape_anomaly = (
            predicted["pred_regularity_score"].fillna(0).astype(float).lt(0.985)
            | predicted["pred_hull_gap_ratio"].fillna(0).astype(float).gt(0.005)
            | predicted["pred_mrr_ratio"].fillna(0).astype(float).lt(0.985)
        )
        seed = under_area & (split | small_source_count | shape_anomaly)
    work = predicted.loc[seed, ["layer_fid", "pred_area", "pred_regularity_score", "pred_hull_gap_ratio", "pred_mrr_ratio"]].copy()
    work["seed_priority"] = (
        np.log1p(work["pred_area"].fillna(0.0).astype(float).clip(lower=0.0))
        + 2.0 * (1.0 - work["pred_regularity_score"].fillna(0.0).astype(float)).clip(lower=0.0)
        + 2.0 * work["pred_hull_gap_ratio"].fillna(0.0).astype(float).clip(lower=0.0)
        + (1.0 - work["pred_mrr_ratio"].fillna(0.0).astype(float)).clip(lower=0.0)
    )
    return work.sort_values(["seed_priority", "layer_fid"], ascending=[False, True])["layer_fid"].astype(int).tolist()


def build_shared_edges(
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
    seed_positions = np.array([fid_to_pos[fid] for fid in sorted(seed_fids) if fid in fid_to_pos], dtype=int)

    parts: list[pd.DataFrame] = []
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
        left_positions = absolute_left[keep]
        right_positions = absolute_right[keep]
        pair_area = areas[left_positions] + areas[right_positions]
        chunk = pd.DataFrame(
            {
                "left_fid": np.minimum(left_fids, right_fids).astype(int),
                "right_fid": np.maximum(left_fids, right_fids).astype(int),
                "left_pos": left_positions.astype(int),
                "right_pos": right_positions.astype(int),
                "pair_area": pair_area.astype(float),
            }
        ).drop_duplicates(["left_fid", "right_fid"])
        chunk = chunk[chunk["pair_area"].le(float(max_pair_area))].copy()
        if chunk.empty:
            continue
        shared_values: list[float] = []
        for offset in range(0, len(chunk), 50_000):
            edge_chunk = chunk.iloc[offset : offset + 50_000]
            left_geom = all_geoms.iloc[edge_chunk["left_pos"].astype(int).to_numpy()]
            right_geom = all_geoms.iloc[edge_chunk["right_pos"].astype(int).to_numpy()]
            shared = shapely.length(
                shapely.intersection(shapely.boundary(left_geom.array), shapely.boundary(right_geom.array))
            )
            shared_values.extend(float(v) for v in shared)
        chunk["shared_edge_len"] = shared_values
        chunk = chunk[chunk["shared_edge_len"].ge(float(min_shared_edge))].copy()
        if not chunk.empty:
            parts.append(chunk[["left_fid", "right_fid", "shared_edge_len"]])

    if not parts:
        return pd.DataFrame(columns=["left_fid", "right_fid", "shared_edge_len"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["left_fid", "right_fid"]).reset_index(drop=True)


def adjacency_from_edges(edges: pd.DataFrame, top_neighbors: int) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
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
        adjacency[fid] = sorted(values, key=lambda item: (-float(item[1]), int(item[0])))[: int(top_neighbors)]
    return adjacency, shared_by_pair


def enumerate_connected_groups(
    *,
    seed_order: list[int],
    seed_fids: set[int],
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_fid: dict[int, float],
    max_group_size: int,
    max_after_area: float,
    per_seed_limit: int,
    max_candidate_groups: int,
) -> set[frozenset[int]]:
    groups: set[frozenset[int]] = set()
    for seed in seed_order:
        seed = int(seed)
        if seed not in seed_fids or seed not in adjacency:
            continue
        emitted_for_seed = 0
        start = frozenset({seed})
        stack = [start]
        seen = {start}
        while stack and emitted_for_seed < int(per_seed_limit):
            current = stack.pop()
            if len(current) >= 2:
                groups.add(current)
                emitted_for_seed += 1
                if len(groups) >= int(max_candidate_groups):
                    return groups
            if len(current) >= int(max_group_size):
                continue
            frontier: dict[int, float] = {}
            for fid in current:
                for neighbor, shared in adjacency.get(int(fid), []):
                    if neighbor not in current:
                        frontier[int(neighbor)] = max(float(shared), frontier.get(int(neighbor), 0.0))
            for neighbor, _shared in sorted(frontier.items(), key=lambda item: (-float(item[1]), int(item[0]))):
                new_group = frozenset(set(current) | {int(neighbor)})
                if new_group in seen:
                    continue
                area = sum(float(area_by_fid.get(fid, 0.0)) for fid in new_group)
                if area > float(max_after_area):
                    continue
                seen.add(new_group)
                stack.append(new_group)
    return groups


def reference_groups_by_fid(sources: gpd.GeoDataFrame, predicted: gpd.GeoDataFrame) -> dict[int, set[int]]:
    component_to_fid = dict(zip(predicted["pred_component_id"].astype(int), predicted["layer_fid"].astype(int)))
    ref_groups: dict[int, set[int]] = {}
    valid = sources["reference_merge_fid"].notna()
    for ref, group in sources[valid].groupby(sources.loc[valid, "reference_merge_fid"].astype(int)):
        fids = {
            int(component_to_fid[int(comp)])
            for comp in group["pred_component_id"].dropna().astype(int)
            if int(comp) in component_to_fid
        }
        if len(fids) >= 2:
            ref_groups[int(ref)] = fids
    return ref_groups


def _complete_like(shape: dict[str, float]) -> bool:
    return bool(
        shape["regularity_score"] >= 0.90
        and shape["hull_gap_ratio"] <= 0.06
        and shape["mrr_ratio"] >= 0.80
    )


def candidate_features(
    fids: frozenset[int],
    *,
    seed_fid: int,
    predicted: gpd.GeoDataFrame,
    geom_by_fid: dict[int, Any],
    attrs_by_fid: dict[int, dict[str, Any]],
    shared_by_pair: dict[tuple[int, int], float],
    adjacency: dict[int, list[tuple[int, float]]],
    shape_by_fid: dict[int, dict[str, float]],
) -> dict[str, Any]:
    ordered_fids = sorted(int(fid) for fid in fids)
    geoms = [geom_by_fid[fid] for fid in ordered_fids]
    union_geom = shapely.union_all(geoms)
    group_shape = _shape_metrics(union_geom)
    part_shapes = [shape_by_fid[fid] for fid in ordered_fids]
    areas = np.asarray([float(attrs_by_fid[fid]["pred_area"]) for fid in ordered_fids], dtype="float64")
    perimeters = np.asarray([float(attrs_by_fid[fid]["pred_perimeter"]) for fid in ordered_fids], dtype="float64")
    largest_idx = int(np.argmax(areas))
    largest_shape = part_shapes[largest_idx]

    internal_shared = 0.0
    for i, left in enumerate(ordered_fids):
        for right in ordered_fids[i + 1 :]:
            internal_shared += float(shared_by_pair.get((min(left, right), max(left, right)), 0.0))

    neighbor_shared_total = 0.0
    neighbor_complete_shared = 0.0
    neighbor_count: set[int] = set()
    neighbor_complete_count: set[int] = set()
    for fid in ordered_fids:
        for neighbor, shared in adjacency.get(fid, []):
            if neighbor in fids:
                continue
            neighbor_count.add(int(neighbor))
            neighbor_shared_total += float(shared)
            if neighbor in shape_by_fid and _complete_like(shape_by_fid[neighbor]):
                neighbor_complete_count.add(int(neighbor))
                neighbor_complete_shared += float(shared)

    refs: set[str] = set()
    component_ids: set[int] = set()
    uprn_sum = 0
    source_count_sum = 0
    possible_split_count = 0
    for fid in ordered_fids:
        attrs = attrs_by_fid[fid]
        component_ids.add(int(attrs["pred_component_id"]))
        ref_text = str(attrs.get("reference_merge_fids") or "")
        if ref_text:
            refs.update(part for part in ref_text.split("|") if part)
        uprn_sum += int(attrs.get("pred_uprn_count") or 0)
        source_count_sum += int(attrs.get("source_count") or 0)
        possible_split_count += int(attrs.get("possible_split_reference") or 0)

    weighted_part_regularity = float(
        np.average([float(shape["regularity_score"]) for shape in part_shapes], weights=areas)
    )
    complete_like_count = int(sum(1 for shape in part_shapes if _complete_like(shape)))
    perimeter_sum = float(perimeters.sum())
    union_perimeter = float(group_shape["perimeter"])
    record: dict[str, Any] = {
        "seed_fid": int(seed_fid),
        "candidate_fids": ids_text(fids),
        "candidate_component_ids": ids_text(component_ids),
        "candidate_reference_fids": "|".join(sorted(refs)),
        "candidate_reference_fid_count": int(len(refs)),
        "group_size": int(len(fids)),
        "group_area": float(group_shape["area"]),
        "group_source_count": int(source_count_sum),
        "group_uprn_count": int(uprn_sum),
        "possible_split_part_count": int(possible_split_count),
        "max_area_ratio": safe_ratio(float(areas.max()), float(areas.sum())),
        "min_area_ratio": safe_ratio(float(areas.min()), float(areas.sum())),
        "small_area_ratio": safe_ratio(float(areas.sum() - areas.max()), float(areas.sum())),
        "area_balance_min_to_max": safe_ratio(float(areas.min()), float(areas.max())),
        "internal_shared_len": float(internal_shared),
        "internal_shared_log1p": float(np.log1p(max(internal_shared, 0.0))),
        "internal_to_sqrt_area": safe_ratio(float(internal_shared), math.sqrt(float(areas.sum()))),
        "boundary_simplification": safe_ratio(perimeter_sum - union_perimeter, perimeter_sum),
        "shared_to_external_shared_ratio": safe_ratio(float(internal_shared), float(neighbor_shared_total)),
        "neighbor_count": int(len(neighbor_count)),
        "neighbor_complete_like_count": int(len(neighbor_complete_count)),
        "neighbor_complete_like_ratio": safe_ratio(float(len(neighbor_complete_count)), float(len(neighbor_count))),
        "neighbor_shared_total": float(neighbor_shared_total),
        "neighbor_complete_shared_ratio": safe_ratio(float(neighbor_complete_shared), float(neighbor_shared_total)),
        "weighted_part_regularity": weighted_part_regularity,
        "largest_part_regularity": float(largest_shape["regularity_score"]),
        "largest_part_mrr_ratio": float(largest_shape["mrr_ratio"]),
        "largest_part_hull_gap_ratio": float(largest_shape["hull_gap_ratio"]),
        "complete_like_part_count": int(complete_like_count),
        "complete_like_part_ratio": safe_ratio(float(complete_like_count), float(len(fids))),
        "all_parts_complete_like": int(complete_like_count == len(fids)),
        "regularity_gain_vs_largest": float(group_shape["regularity_score"] - largest_shape["regularity_score"]),
        "hull_gap_reduction_vs_largest": float(largest_shape["hull_gap_ratio"] - group_shape["hull_gap_ratio"]),
        "regularity_gain_vs_weighted_parts": float(group_shape["regularity_score"] - weighted_part_regularity),
        "geometry": union_geom,
    }
    for name, value in group_shape.items():
        record[f"group_{name}"] = float(value)
    return record


def label_candidate(
    fids: frozenset[int],
    *,
    manual_positive_groups: list[set[int]],
    reference_groups: dict[int, set[int]],
) -> tuple[int, str, float, str]:
    fid_set = set(int(fid) for fid in fids)
    for manual in manual_positive_groups:
        if fid_set == set(manual):
            return 1, "manual_complete_positive", 80.0, "manual_complete"
        if fid_set & set(manual):
            return 0, "manual_partial_or_overmerge_negative", 15.0, "manual_overlap"

    overlapping_refs = {ref: group for ref, group in reference_groups.items() if fid_set & group}
    if len(overlapping_refs) == 1:
        ref, full_group = next(iter(overlapping_refs.items()))
        if fid_set == full_group:
            return 1, "reference_complete_positive", 8.0, "reference_complete"
        if fid_set < full_group:
            return 0, "reference_partial_negative", 5.0, "reference_partial"
        return 0, "reference_overmerge_negative", 5.0, "reference_overmerge"
    if len(overlapping_refs) > 1:
        union = set().union(*overlapping_refs.values())
        if fid_set == union:
            return 0, "multi_reference_complete_overmerge_negative", 4.0, "multi_reference_complete"
        return 0, "multi_reference_overlap_negative", 3.0, "multi_reference_overlap"
    return 0, "reference_unknown_negative", 1.0, "reference_unknown"


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


def build_parcel_assembly_candidates(
    *,
    input_gpkg: Path,
    max_seed_area: float,
    max_after_area: float,
    max_pair_area: float,
    max_group_size: int,
    top_neighbors: int,
    per_seed_limit: int,
    max_candidate_groups: int,
    min_shared_edge: float,
    query_chunk_size: int,
    include_all_under_area: bool,
    manual_positive_fid_groups: list[set[int]] | None = None,
    include_labels: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    manual_positive_fid_groups = manual_positive_fid_groups or []
    predicted = read_predicted(input_gpkg)
    sources = read_sources(input_gpkg) if include_labels else gpd.GeoDataFrame(geometry=[], crs=predicted.crs)
    seed_order = build_seed_fids(
        predicted,
        max_seed_area=float(max_seed_area),
        include_all_under_area=bool(include_all_under_area),
    )
    priority: list[int] = []
    valid_fids = set(predicted["layer_fid"].astype(int))
    for group in manual_positive_fid_groups:
        priority.extend(sorted(int(fid) for fid in group if int(fid) in valid_fids))
    if priority:
        seen = set(priority)
        seed_order = priority + [fid for fid in seed_order if fid not in seen]
    seed_fids = set(seed_order)
    log(f"[INFO] Assembly seed fids={len(seed_fids):,}")

    edges = build_shared_edges(
        predicted,
        seed_fids,
        min_shared_edge=float(min_shared_edge),
        max_pair_area=float(max_pair_area),
        query_chunk_size=int(query_chunk_size),
    )
    log(f"[INFO] Assembly shared-edge rows={len(edges):,}")
    adjacency, shared_by_pair = adjacency_from_edges(edges, top_neighbors=int(top_neighbors))
    area_by_fid = predicted.set_index("layer_fid")["pred_area"].astype(float).to_dict()
    groups = enumerate_connected_groups(
        seed_order=seed_order,
        seed_fids=seed_fids,
        adjacency=adjacency,
        area_by_fid=area_by_fid,
        max_group_size=int(max_group_size),
        max_after_area=float(max_after_area),
        per_seed_limit=int(per_seed_limit),
        max_candidate_groups=int(max_candidate_groups),
    )
    valid_fids = set(predicted["layer_fid"].astype(int))
    for manual_group in manual_positive_fid_groups:
        group = frozenset(int(fid) for fid in manual_group if int(fid) in valid_fids)
        if len(group) >= 2 and sum(float(area_by_fid.get(fid, 0.0)) for fid in group) <= float(max_after_area):
            groups.add(group)
    log(f"[INFO] Assembly connected groups={len(groups):,}")

    geom_by_fid = predicted.set_index("layer_fid").geometry.to_dict()
    attrs_by_fid = predicted.set_index("layer_fid").drop(columns="geometry").to_dict("index")
    shape_by_fid = {
        int(row.layer_fid): {
            "regularity_score": float(row.pred_regularity_score),
            "hull_gap_ratio": float(row.pred_hull_gap_ratio),
            "mrr_ratio": float(row.pred_mrr_ratio),
        }
        for row in predicted.itertuples(index=False)
    }
    reference_groups = reference_groups_by_fid(sources, predicted) if include_labels else {}

    records: list[dict[str, Any]] = []
    for group in groups:
        seed_candidates = sorted(group, key=lambda fid: (-float(area_by_fid.get(fid, 0.0)), int(fid)))
        seed_fid = int(seed_candidates[0]) if seed_candidates else int(next(iter(group)))
        rec = candidate_features(
            group,
            seed_fid=seed_fid,
            predicted=predicted,
            geom_by_fid=geom_by_fid,
            attrs_by_fid=attrs_by_fid,
            shared_by_pair=shared_by_pair,
            adjacency=adjacency,
            shape_by_fid=shape_by_fid,
        )
        if include_labels:
            label, label_source, weight, relation = label_candidate(
                group,
                manual_positive_groups=manual_positive_fid_groups,
                reference_groups=reference_groups,
            )
            rec["label"] = int(label)
            rec["label_source"] = label_source
            rec["sample_weight"] = float(weight)
            rec["reference_relation"] = relation
            positive_ref_fids = [
                ref for ref, ref_group in reference_groups.items() if set(group) == ref_group
            ]
            rec["reference_complete_fids"] = "|".join(str(v) for v in sorted(positive_ref_fids))
        else:
            rec["reference_relation"] = "unknown"
            rec["reference_complete_fids"] = ""
        records.append(rec)

    if not records:
        empty = gpd.GeoDataFrame(geometry=[], crs=predicted.crs)
        return empty, edges
    candidates = gpd.GeoDataFrame(records, geometry="geometry", crs=predicted.crs)
    return candidates, edges

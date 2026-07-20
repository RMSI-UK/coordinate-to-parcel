#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path("/data/sheffield/spatial/base-map")
DEFAULT_LABEL_GPKG = BASE_DIR / "sheffield_wfs_merged_council_train_parcel.gpkg"
DEFAULT_INPUT_GPKG = BASE_DIR / "sheffield_wfs_merged_council_train_molecular_depth3.gpkg"
DEFAULT_EDGE_CACHE = (
    BASE_DIR
    / "tmp/wfs_raw_anchor_group_model_completeness_v2_context_cache/"
    / "shared_edges_e455305190c051e0db7e7441.joblib"
)
DEFAULT_MANIFEST = Path(__file__).with_name("wfs_merged_council_anchor_feature_manifest.json")
DEFAULT_OUTPUT_DIR = BASE_DIR / "tmp/wfs_merged_council_anchor_group_model_v1"

LABEL_LAYER = "train_parcel_label"
INPUT_LAYER = "train_input_molecular_depth3"
MODEL_FILE_NAME = "wfs_merged_council_anchor_group_model_v1.joblib"
METRICS_FILE_NAME = "wfs_merged_council_anchor_group_model_metrics_v1.json"
CANDIDATES_FILE_NAME = "wfs_merged_council_anchor_group_candidates_v1.parquet"
TARGET_COL = "label"

ID_COLUMNS = {
    "label",
    "sample_weight",
    "label_id",
    "split",
    "spatial_group",
    "candidate_clean_fids",
    "candidate_source_fids",
    "target_clean_fids",
    "target_source_fids",
    "anchor_raw_clean_fid",
    "proposal_source",
    "positive_generated_by_enumeration",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _ids_text(values: set[int] | frozenset[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(v)) for v in sorted(int(x) for x in values))


def _safe_ratio(num: float, den: float) -> float:
    den = float(den)
    return float(num) / den if den else 0.0


def _as_int(frame: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="int64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("int64")


def _as_float(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("float64")


def _parse_ids(value: Any) -> set[int]:
    if value is None or pd.isna(value):
        return set()
    out: set[int] = set()
    for part in str(value).replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(float(part)))
        except ValueError:
            continue
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


def _shape_metrics(geom: Any) -> dict[str, float]:
    if geom is None or geom.is_empty:
        return {
            "candidate_area_union": 0.0,
            "candidate_perimeter_union": 0.0,
            "candidate_compactness": 0.0,
            "candidate_hull_gap_ratio": 0.0,
            "candidate_mrr_ratio": 0.0,
            "candidate_mrr_width": 0.0,
            "candidate_bounds_aspect": 0.0,
            "candidate_interior_hole_count": 0.0,
            "candidate_interior_hole_area": 0.0,
            "candidate_interior_hole_area_ratio": 0.0,
            "candidate_interior_hole_max_area": 0.0,
        }
    area = float(shapely.area(geom))
    perimeter = float(shapely.length(geom))
    compactness = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0.0 else 0.0
    hull_area = float(shapely.area(shapely.convex_hull(geom)))
    hull_gap = max(hull_area - area, 0.0) / max(area, 1e-9)
    hole_areas: list[float] = []
    for part in _polygon_parts(geom):
        for ring in part.interiors:
            hole = Polygon(ring)
            if not hole.is_empty:
                hole_areas.append(float(hole.area))
    hole_area = float(sum(hole_areas))
    try:
        mrr = shapely.minimum_rotated_rectangle(geom)
        mrr_area = float(shapely.area(mrr))
        mrr_ratio = area / max(mrr_area, 1e-9)
        coords = list(mrr.exterior.coords) if hasattr(mrr, "exterior") else []
        if len(coords) >= 4:
            edges = [
                math.dist(coords[i], coords[i + 1])
                for i in range(min(4, len(coords) - 1))
            ]
            positive_edges = [edge for edge in edges if edge > 0.0]
            mrr_width = min(positive_edges) if positive_edges else 0.0
        else:
            mrr_width = 0.0
    except Exception:
        mrr_ratio = 0.0
        mrr_width = 0.0
    minx, miny, maxx, maxy = shapely.bounds(geom)
    width = float(maxx - minx)
    height = float(maxy - miny)
    bounds_aspect = max(width, height) / max(min(width, height), 1e-9)
    return {
        "candidate_area_union": area,
        "candidate_perimeter_union": perimeter,
        "candidate_compactness": float(compactness),
        "candidate_hull_gap_ratio": float(hull_gap),
        "candidate_mrr_ratio": float(mrr_ratio),
        "candidate_mrr_width": float(mrr_width),
        "candidate_bounds_aspect": float(bounds_aspect),
        "candidate_interior_hole_count": float(len(hole_areas)),
        "candidate_interior_hole_area": float(hole_area),
        "candidate_interior_hole_area_ratio": float(hole_area / max(area, 1e-9)),
        "candidate_interior_hole_max_area": float(max(hole_areas) if hole_areas else 0.0),
    }


def _make_valid_geometries(values: Any) -> Any:
    valid = shapely.is_valid(values)
    if bool(np.all(valid)):
        return values
    out = np.asarray(values, dtype=object).copy()
    bad = ~np.asarray(valid, dtype=bool)
    out[bad] = shapely.make_valid(out[bad])
    return out


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"numeric_features", "categorical_features"}
    missing = required - set(manifest)
    if missing:
        raise RuntimeError(f"Feature manifest is missing keys: {sorted(missing)}")
    return manifest


def _read_labels(path: Path, layer: str, *, max_labels: int, random_state: int) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading labels: {path}:{layer}")
    labels = pyogrio.read_dataframe(path, layer=layer)
    labels = labels[labels.geometry.notna() & ~labels.geometry.is_empty].copy()
    labels["label_id"] = _as_int(labels, "label_id")
    labels["anchor_raw_clean_fid"] = _as_int(labels, "anchor_raw_clean_fid")
    labels["anchor_source_fid"] = _as_int(labels, "anchor_source_fid")
    labels["anchor_uprn_count"] = _as_int(labels, "anchor_uprn_count")
    labels = labels[labels["label_id"].gt(0) & labels["anchor_raw_clean_fid"].gt(0)].copy()
    if int(max_labels) > 0 and int(max_labels) < len(labels):
        labels = labels.sample(n=int(max_labels), random_state=int(random_state)).sort_values("label_id").copy()
        _log(f"[INFO] Applied max labels sample: {len(labels):,}")
    labels.geometry = _make_valid_geometries(labels.geometry.to_numpy())
    centroids = labels.geometry.centroid
    labels["centroid_x"] = centroids.x.astype("float64")
    labels["centroid_y"] = centroids.y.astype("float64")
    labels["target_source_set"] = labels["label_source_fids"].map(_parse_ids)
    return labels.reset_index(drop=True)


def _read_inputs(path: Path, layer: str, label_ids: set[int]) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading input molecules: {path}:{layer}")
    inputs = pyogrio.read_dataframe(path, layer=layer)
    inputs = inputs[inputs.geometry.notna() & ~inputs.geometry.is_empty].copy()
    inputs["raw_clean_fid"] = _as_int(inputs, "raw_clean_fid")
    inputs["source_fid"] = _as_int(inputs, "source_fid")
    inputs["label_id"] = _as_int(inputs, "label_id", -1)
    inputs["neighbor_depth"] = _as_int(inputs, "neighbor_depth")
    keep_depth0 = inputs["neighbor_depth"].eq(0) & inputs["label_id"].isin(label_ids)
    keep_neighbors = inputs["neighbor_depth"].gt(0)
    inputs = inputs[keep_depth0 | keep_neighbors].copy()
    inputs.geometry = _make_valid_geometries(inputs.geometry.to_numpy())
    for column in [
        "uprn_count",
        "has_uprn",
        "is_building_theme",
        "zero_uprn_plot_eligible",
        "plot_eligible",
        "is_polygon_hole_fill",
        "is_enclosed_gap_fill",
        "is_building_uprn_anchor",
        "is_building_label_anchor",
        "is_nonanchor_uprn",
    ]:
        inputs[column] = _as_int(inputs, column)
    for column in [
        "clean_area",
        "clean_perimeter",
        "clean_mrr_ratio",
        "clean_hull_gap_ratio",
        "clean_compactness",
        "raw_width_proxy",
        "raw_mrr_width",
        "gap_fill_area",
    ]:
        inputs[column] = _as_float(inputs, column)
    for column in ["theme_role", "raw_role"]:
        if column not in inputs.columns:
            inputs[column] = ""
        inputs[column] = inputs[column].fillna("").astype(str)
    _log(
        "[INFO] Input rows retained for training context="
        f"{len(inputs):,}; depth0={int(inputs['neighbor_depth'].eq(0).sum()):,}; "
        f"neighbors={int(inputs['neighbor_depth'].gt(0).sum()):,}"
    )
    return inputs.reset_index(drop=True)


def _assign_spatial_split(labels: gpd.GeoDataFrame, *, tile_size: float, test_size: float, random_state: int) -> dict[int, tuple[str, str]]:
    tile_x = np.floor(labels["centroid_x"].to_numpy(dtype="float64") / float(tile_size)).astype("int64")
    tile_y = np.floor(labels["centroid_y"].to_numpy(dtype="float64") / float(tile_size)).astype("int64")
    groups = np.asarray([f"{x}_{y}" for x, y in zip(tile_x, tile_y)], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, test_idx = next(splitter.split(labels, groups=groups))
    split = np.full(len(labels), "train", dtype=object)
    split[test_idx] = "test"
    _log(
        "[INFO] Spatial split: "
        f"train_labels={len(train_idx):,}; test_labels={len(test_idx):,}; "
        f"spatial_groups={len(set(groups)):,}"
    )
    return {
        int(label_id): (str(split_value), str(group))
        for label_id, split_value, group in zip(labels["label_id"], split, groups)
    }


def _build_node_indexes(inputs: gpd.GeoDataFrame) -> tuple[dict[int, dict[str, Any]], dict[int, Any]]:
    attrs: dict[int, dict[str, Any]] = {}
    geoms: dict[int, Any] = {}
    for row in inputs.itertuples(index=False):
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
            "is_building_label_anchor": int(row.is_building_label_anchor),
            "is_nonanchor_uprn": int(row.is_nonanchor_uprn),
            "theme_role": str(row.theme_role or ""),
            "raw_role": str(row.raw_role or ""),
        }
        geoms[fid] = row.geometry
    return attrs, geoms


def _load_adjacency(edge_cache: Path, valid_nodes: set[int], *, top_neighbors: int) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    _log(f"[INFO] Loading shared-edge cache: {edge_cache}")
    cache = joblib.load(edge_cache)
    raw_adjacency: dict[int, list[tuple[int, float]]] = cache["adjacency"]
    adjacency: dict[int, list[tuple[int, float]]] = {}
    shared_by_pair: dict[tuple[int, int], float] = {}
    for node in valid_nodes:
        values = [
            (int(neighbor), float(shared))
            for neighbor, shared in raw_adjacency.get(int(node), ())
            if int(neighbor) in valid_nodes
        ]
        values = sorted(values, key=lambda item: (-item[1], item[0]))[: int(top_neighbors)]
        if values:
            adjacency[int(node)] = values
            for neighbor, shared in values:
                pair = (min(int(node), int(neighbor)), max(int(node), int(neighbor)))
                shared_by_pair[pair] = max(float(shared), shared_by_pair.get(pair, 0.0))
    _log(f"[INFO] Restricted adjacency nodes={len(adjacency):,}; pairs={len(shared_by_pair):,}")
    return adjacency, shared_by_pair


def _distances_from_anchor(anchor: int, adjacency: dict[int, list[tuple[int, float]]], max_depth: int) -> dict[int, int]:
    distances = {int(anchor): 0}
    queue: deque[int] = deque([int(anchor)])
    while queue:
        node = queue.popleft()
        depth = distances[node]
        if depth >= int(max_depth):
            continue
        for neighbor, _shared in adjacency.get(node, ()):
            neighbor = int(neighbor)
            if neighbor not in distances:
                distances[neighbor] = depth + 1
                queue.append(neighbor)
    return distances


def _enumerate_anchor_groups(
    *,
    anchor: int,
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_clean: dict[int, float],
    max_group_size: int,
    max_group_area: float,
    per_label_limit: int,
) -> set[frozenset[int]]:
    emitted: set[frozenset[int]] = set()
    start = frozenset({int(anchor)})
    stack = [start]
    seen = {start}
    while stack and len(emitted) < int(per_label_limit):
        current = stack.pop()
        emitted.add(current)
        if len(current) >= int(max_group_size):
            continue
        frontier: dict[int, float] = {}
        for node in current:
            for neighbor, shared in adjacency.get(int(node), ()):
                if int(neighbor) not in current:
                    frontier[int(neighbor)] = max(float(shared), frontier.get(int(neighbor), 0.0))
        ordered = sorted(frontier.items(), key=lambda item: (-item[1], int(item[0])))
        for neighbor, _shared in ordered:
            group = frozenset(set(current) | {int(neighbor)})
            if group in seen:
                continue
            area = sum(float(area_by_clean.get(int(fid), 0.0)) for fid in group)
            if area > float(max_group_area):
                continue
            seen.add(group)
            stack.append(group)
    return emitted


def _combo_anchor_groups(
    *,
    anchor: int,
    distances: dict[int, int],
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_clean: dict[int, float],
    max_group_size: int,
    max_group_area: float,
    local_n: int,
    max_extra: int,
    per_label_limit: int,
) -> set[frozenset[int]]:
    if int(local_n) <= 0 or int(max_extra) <= 0 or int(per_label_limit) <= 0:
        return set()
    best_shared: dict[int, float] = {}
    for node, depth in distances.items():
        if int(node) == int(anchor) or int(depth) <= 0:
            continue
        for neighbor, shared in adjacency.get(int(node), ()):
            if int(neighbor) in distances and int(distances[int(neighbor)]) < int(depth):
                best_shared[int(node)] = max(float(shared), best_shared.get(int(node), 0.0))
        for neighbor, shared in adjacency.get(int(anchor), ()):
            if int(neighbor) == int(node):
                best_shared[int(node)] = max(float(shared), best_shared.get(int(node), 0.0))
    local_nodes = [
        int(node)
        for node in sorted(
            (node for node in distances if int(node) != int(anchor)),
            key=lambda item: (int(distances[int(item)]), -float(best_shared.get(int(item), 0.0)), int(item)),
        )
    ][: int(local_n)]
    groups: set[frozenset[int]] = {frozenset({int(anchor)})}
    max_extra = min(int(max_extra), int(max_group_size) - 1, len(local_nodes))
    for extra_count in range(1, max_extra + 1):
        scored_combos: list[tuple[float, tuple[int, ...]]] = []
        for combo in itertools.combinations(local_nodes, extra_count):
            group_ids = (int(anchor), *combo)
            area = sum(float(area_by_clean.get(int(fid), 0.0)) for fid in group_ids)
            if area > float(max_group_area):
                continue
            score = sum(float(distances.get(int(fid), 99)) for fid in combo)
            score -= 0.02 * sum(float(best_shared.get(int(fid), 0.0)) for fid in combo)
            scored_combos.append((float(score), tuple(int(fid) for fid in combo)))
        for _score, combo in sorted(scored_combos, key=lambda item: (item[0], item[1])):
            groups.add(frozenset({int(anchor), *combo}))
            if len(groups) >= int(per_label_limit):
                return groups
    return groups


def _parse_int_list(text: str | int) -> list[int]:
    if isinstance(text, int):
        return [int(text)]
    values: list[int] = []
    for part in str(text).replace("|", ",").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_quota_specs(text: str) -> list[dict[int, int]]:
    specs: list[dict[int, int]] = []
    for spec_text in str(text).split(";"):
        spec_text = spec_text.strip()
        if not spec_text:
            continue
        spec: dict[int, int] = {}
        for part in spec_text.replace("|", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Invalid quota part {part!r}; expected extra_count:limit")
            key, value = part.split(":", 1)
            spec[int(key.strip())] = int(value.strip())
        specs.append(spec)
    if not specs:
        raise ValueError("Expected at least one quota spec.")
    return specs


def _quota_combo_anchor_groups(
    *,
    anchor: int,
    distances: dict[int, int],
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_clean: dict[int, float],
    max_group_size: int,
    max_group_area: float,
    local_ns: list[int],
    quota_specs: list[dict[int, int]],
    max_extra: int,
) -> set[frozenset[int]]:
    groups: set[frozenset[int]] = {frozenset({int(anchor)})}
    if not local_ns:
        return groups
    if len(quota_specs) == 1 and len(local_ns) > 1:
        quota_specs = quota_specs * len(local_ns)
    if len(quota_specs) != len(local_ns):
        raise ValueError("--combo-extra-quotas must provide either one spec or one spec per --combo-local-n value.")

    best_shared: dict[int, float] = {}
    for node, depth in distances.items():
        node = int(node)
        if node == int(anchor) or int(depth) <= 0:
            continue
        for neighbor, shared in adjacency.get(node, ()):
            neighbor = int(neighbor)
            if neighbor in distances and int(distances[neighbor]) < int(depth):
                best_shared[node] = max(float(shared), best_shared.get(node, 0.0))
        for neighbor, shared in adjacency.get(int(anchor), ()):
            if int(neighbor) == node:
                best_shared[node] = max(float(shared), best_shared.get(node, 0.0))

    local_all = [
        int(node)
        for node in sorted(
            (node for node in distances if int(node) != int(anchor)),
            key=lambda item: (int(distances[int(item)]), -float(best_shared.get(int(item), 0.0)), int(item)),
        )
    ]
    for local_n, quota_by_extra in zip(local_ns, quota_specs):
        local_nodes = local_all[: int(local_n)]
        max_extra_for_pass = min(int(max_extra), int(max_group_size) - 1, len(local_nodes))
        for extra_count in range(1, max_extra_for_pass + 1):
            limit = int(quota_by_extra.get(extra_count, 0))
            if limit <= 0:
                continue
            scored_combos: list[tuple[float, tuple[int, ...]]] = []
            for combo in itertools.combinations(local_nodes, extra_count):
                group_ids = (int(anchor), *combo)
                area = sum(float(area_by_clean.get(int(fid), 0.0)) for fid in group_ids)
                if area > float(max_group_area):
                    continue
                score = sum(float(distances.get(int(fid), 99)) for fid in combo)
                score -= 0.02 * sum(float(best_shared.get(int(fid), 0.0)) for fid in combo)
                scored_combos.append((float(score), tuple(int(fid) for fid in combo)))
            for _score, combo in sorted(scored_combos, key=lambda item: (item[0], item[1]))[:limit]:
                groups.add(frozenset({int(anchor), *combo}))
    return groups


def _group_union(group: frozenset[int], geoms: dict[int, Any]) -> Any:
    values = [geoms[int(fid)] for fid in sorted(group) if int(fid) in geoms]
    if not values:
        return None
    geom = shapely.union_all(np.asarray(values, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _role_for_node(attr: dict[str, Any]) -> str:
    for column in ["theme_role", "raw_role"]:
        value = str(attr.get(column, "") or "").lower()
        if value in {"building", "land", "gapfill", "road", "other"}:
            return value
        if "building" in value:
            return "building"
        if "land" in value:
            return "land"
        if "gap" in value or "hole" in value:
            return "gapfill"
        if "road" in value or "path" in value or "track" in value:
            return "road"
    if int(attr.get("is_building_theme", 0)) == 1:
        return "building"
    return "other"


def _candidate_features(
    *,
    label_id: int,
    candidate_group: frozenset[int],
    target_group: frozenset[int],
    anchor: int,
    proposal_source: str,
    split: str,
    spatial_group: str,
    attrs: dict[int, dict[str, Any]],
    geoms: dict[int, Any],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    distance_cache: dict[int, int],
    shape_cache: dict[frozenset[int], dict[str, float]],
) -> dict[str, Any]:
    group = frozenset(int(v) for v in candidate_group)
    label = int(group == target_group)
    rows = [attrs[int(fid)] for fid in sorted(group) if int(fid) in attrs]
    areas = np.asarray([float(row["clean_area"]) for row in rows], dtype="float64")
    source_ids = {int(row["source_fid"]) for row in rows}
    roles = [_role_for_node(row) for row in rows]
    role_counts = Counter(roles)
    role_signature = "|".join(f"{role}:{role_counts[role]}" for role in sorted(role_counts))

    if group not in shape_cache:
        geom = _group_union(group, geoms)
        shape_cache[group] = _shape_metrics(geom)
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
    target_source_ids = {int(attrs[int(fid)]["source_fid"]) for fid in target_group if int(fid) in attrs}

    record: dict[str, Any] = {
        "label": label,
        "sample_weight": 8.0 if label else (2.0 if proposal_source in {"target_omit", "target_overmerge", "target_replace"} else 1.0),
        "label_id": int(label_id),
        "split": str(split),
        "spatial_group": str(spatial_group),
        "candidate_clean_fids": _ids_text(group),
        "candidate_source_fids": _ids_text(source_ids),
        "target_clean_fids": _ids_text(target_group),
        "target_source_fids": _ids_text(target_source_ids),
        "anchor_raw_clean_fid": int(anchor),
        "proposal_source": str(proposal_source),
        "candidate_clean_count": int(len(group)),
        "candidate_source_count": int(len(source_ids)),
        "candidate_area_sum": candidate_area_sum,
        "candidate_area_union_to_sum": _safe_ratio(shape["candidate_area_union"], candidate_area_sum),
        "anchor_area": anchor_area,
        "added_area": max(candidate_area_sum - anchor_area, 0.0),
        "added_area_to_anchor": _safe_ratio(max(candidate_area_sum - anchor_area, 0.0), anchor_area),
        "largest_piece_area_ratio": _safe_ratio(float(areas.max()) if len(areas) else 0.0, candidate_area_sum),
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
        "building_uprn_anchor_count": int(sum(int(row.get("is_building_uprn_anchor", 0)) for row in rows)),
        "label_anchor_count": int(sum(int(row.get("is_building_label_anchor", 0)) for row in rows)),
        "nonseed_building_uprn_anchor_count": int(
            sum(int(row.get("is_building_uprn_anchor", 0)) for row in rows)
            - sum(int(row.get("is_building_label_anchor", 0)) for row in rows)
        ),
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


def _hard_negative_groups(
    *,
    target_group: frozenset[int],
    anchor: int,
    adjacency: dict[int, list[tuple[int, float]]],
    attrs: dict[int, dict[str, Any]],
    max_overmerge_neighbors: int,
) -> list[tuple[frozenset[int], str]]:
    groups: list[tuple[frozenset[int], str]] = []
    if target_group != frozenset({int(anchor)}):
        groups.append((frozenset({int(anchor)}), "anchor_only"))
    for fid in sorted(target_group - {int(anchor)}):
        omitted = frozenset(set(target_group) - {int(fid)})
        if omitted:
            groups.append((omitted, "target_omit"))

    frontier: dict[int, float] = {}
    for node in target_group:
        for neighbor, shared in adjacency.get(int(node), ()):
            neighbor = int(neighbor)
            if neighbor not in target_group and neighbor in attrs:
                frontier[neighbor] = max(float(shared), frontier.get(neighbor, 0.0))
    ordered = [fid for fid, _shared in sorted(frontier.items(), key=lambda item: (-item[1], item[0]))]
    for neighbor in ordered[: int(max_overmerge_neighbors)]:
        groups.append((frozenset(set(target_group) | {int(neighbor)}), "target_overmerge"))
    if len(target_group) > 2:
        removable = sorted(target_group - {int(anchor)})[:3]
        for remove_id, add_id in zip(removable, ordered[:3]):
            replaced = frozenset((set(target_group) - {int(remove_id)}) | {int(add_id)})
            if replaced and replaced != target_group:
                groups.append((replaced, "target_replace"))
    return groups


def build_candidates(
    *,
    labels: gpd.GeoDataFrame,
    inputs: gpd.GeoDataFrame,
    attrs: dict[int, dict[str, Any]],
    geoms: dict[int, Any],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    split_by_label: dict[int, tuple[str, str]],
    max_graph_depth: int,
    max_group_size: int,
    max_group_area: float,
    enum_per_label: int,
    combo_local_ns: list[int],
    combo_max_extra: int,
    combo_per_label: int,
    combo_extra_quotas: list[dict[int, int]],
    max_overmerge_neighbors: int,
    max_candidates_per_label: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    depth0 = inputs[inputs["neighbor_depth"].astype(int).eq(0)].copy()
    target_by_label = {
        int(label_id): frozenset(int(v) for v in group["raw_clean_fid"])
        for label_id, group in depth0.groupby("label_id", sort=False)
    }
    area_by_clean = {int(fid): float(row["clean_area"]) for fid, row in attrs.items()}
    shape_cache: dict[frozenset[int], dict[str, float]] = {}
    records: list[dict[str, Any]] = []
    generated_positive_labels = 0
    forced_positive_labels = 0
    skipped_labels = 0

    label_rows = list(labels.itertuples(index=False))
    for offset, row in enumerate(label_rows, start=1):
        if offset == 1 or offset % 5000 == 0:
            _log(f"[INFO] Building candidates {offset:,}/{len(label_rows):,}; rows={len(records):,}")
        label_id = int(row.label_id)
        anchor = int(row.anchor_raw_clean_fid)
        target_group = target_by_label.get(label_id)
        if not target_group or anchor not in target_group:
            skipped_labels += 1
            continue
        split, spatial_group = split_by_label[label_id]
        distances = _distances_from_anchor(anchor, adjacency, int(max_graph_depth))
        enumerated = _enumerate_anchor_groups(
            anchor=anchor,
            adjacency=adjacency,
            area_by_clean=area_by_clean,
            max_group_size=int(max_group_size),
            max_group_area=float(max_group_area),
            per_label_limit=int(enum_per_label),
        )
        combo_groups = _combo_anchor_groups(
            anchor=anchor,
            distances=distances,
            adjacency=adjacency,
            area_by_clean=area_by_clean,
            max_group_size=int(max_group_size),
            max_group_area=float(max_group_area),
            local_n=int(combo_local_ns[0]),
            max_extra=int(combo_max_extra),
            per_label_limit=int(combo_per_label),
        )
        quota_combo_groups = _quota_combo_anchor_groups(
            anchor=anchor,
            distances=distances,
            adjacency=adjacency,
            area_by_clean=area_by_clean,
            max_group_size=int(max_group_size),
            max_group_area=float(max_group_area),
            local_ns=list(combo_local_ns),
            quota_specs=list(combo_extra_quotas),
            max_extra=int(combo_max_extra),
        )
        generated_groups = set(enumerated) | set(combo_groups) | set(quota_combo_groups)
        generated_positive = target_group in generated_groups
        if generated_positive:
            generated_positive_labels += 1
        else:
            forced_positive_labels += 1

        group_sources: dict[frozenset[int], str] = {}
        for group in enumerated:
            group_sources.setdefault(group, "enumerated")
        for group in combo_groups:
            group_sources.setdefault(group, "combo")
        for group in quota_combo_groups:
            group_sources.setdefault(group, "combo_quota")
        for group, source in _hard_negative_groups(
            target_group=target_group,
            anchor=anchor,
            adjacency=adjacency,
            attrs=attrs,
            max_overmerge_neighbors=int(max_overmerge_neighbors),
        ):
            group_sources.setdefault(group, source)
        group_sources[target_group] = "positive_enumerated" if generated_positive else "positive_forced"

        ordered_groups = sorted(
            group_sources.items(),
            key=lambda item: (
                0 if item[0] == target_group else 1,
                0 if item[1].startswith("target") else 1,
                abs(len(item[0]) - len(target_group)),
                len(item[0]),
                _ids_text(item[0]),
            ),
        )
        seen_groups: set[frozenset[int]] = set()
        kept = 0
        for group, source in ordered_groups:
            if group in seen_groups:
                continue
            if len(group) > int(max_group_size) and group != target_group:
                continue
            kept += 1
            seen_groups.add(group)
            record = _candidate_features(
                label_id=label_id,
                candidate_group=group,
                target_group=target_group,
                anchor=anchor,
                proposal_source=source,
                split=split,
                spatial_group=spatial_group,
                attrs=attrs,
                geoms=geoms,
                adjacency=adjacency,
                shared_by_pair=shared_by_pair,
                distance_cache=distances,
                shape_cache=shape_cache,
            )
            record["positive_generated_by_enumeration"] = int(generated_positive)
            records.append(record)
            if kept >= int(max_candidates_per_label):
                break

    candidates = pd.DataFrame.from_records(records)
    summary = {
        "candidate_rows": int(len(candidates)),
        "labels_total": int(len(label_rows)),
        "labels_skipped": int(skipped_labels),
        "labels_with_positive_generated_by_enumeration": int(generated_positive_labels),
        "labels_with_positive_forced": int(forced_positive_labels),
        "candidate_label_counts": {
            str(int(key)): int(value)
            for key, value in candidates["label"].value_counts().sort_index().items()
        }
        if not candidates.empty
        else {},
        "proposal_source_counts": {
            str(key): int(value)
            for key, value in candidates["proposal_source"].value_counts().sort_index().items()
        }
        if not candidates.empty
        else {},
    }
    return candidates, summary


def _threshold_at_recall(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> dict[str, Any] | None:
    if len(set(int(v) for v in y_true)) < 2:
        return None
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return None
    eligible = np.where(recall[:-1] >= float(target_recall))[0]
    if len(eligible) == 0:
        return None
    idx = int(eligible[np.argmax(precision[:-1][eligible])])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= float(threshold)).astype("int64")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[1, 0],
        zero_division=0,
    )
    out = {
        "threshold": float(threshold),
        "precision_positive": float(precision[0]),
        "recall_positive": float(recall[0]),
        "f1_positive": float(f1[0]),
        "support_positive": int(support[0]),
        "precision_negative": float(precision[1]),
        "recall_negative": float(recall[1]),
        "f1_negative": float(f1[1]),
        "support_negative": int(support[1]),
    }
    if len(set(int(v) for v in y_true)) >= 2:
        out["average_precision"] = float(average_precision_score(y_true, proba))
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    return out


def _selection_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    work = frame.sort_values(["label_id", "proba"], ascending=[True, False]).copy()
    work["rank"] = work.groupby("label_id", sort=False).cumcount() + 1
    selected = work[work["rank"].eq(1)].copy()
    exact_selected = int(selected["label"].astype(int).sum())
    label_count = int(selected["label_id"].nunique())
    topk: dict[str, float] = {}
    for k in [1, 2, 3, 5, 10]:
        top = work[work["rank"].le(k)]
        exact_labels = int(top[top["label"].astype(int).eq(1)]["label_id"].nunique())
        topk[f"top{k}_exact_recall"] = _safe_ratio(exact_labels, label_count)
    generated_positive = (
        work[work["label"].astype(int).eq(1)]
        .groupby("label_id")["positive_generated_by_enumeration"]
        .max()
    )
    return {
        "labels": label_count,
        "selected_exact": exact_selected,
        "selected_exact_rate": _safe_ratio(exact_selected, label_count),
        "candidate_pool_positive_labels": int(work[work["label"].astype(int).eq(1)]["label_id"].nunique()),
        "candidate_pool_positive_recall": _safe_ratio(
            int(work[work["label"].astype(int).eq(1)]["label_id"].nunique()),
            label_count,
        ),
        "inference_like_positive_recall_ceiling": _safe_ratio(
            int(generated_positive[generated_positive.astype(int).eq(1)].index.nunique()),
            label_count,
        ),
        **topk,
    }


def train_model(
    *,
    candidates: pd.DataFrame,
    manifest: dict[str, Any],
    random_state: int,
    max_iter: int,
    learning_rate: float,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, list[str], list[str]]:
    numeric_features = [col for col in manifest["numeric_features"] if col in candidates.columns]
    categorical_features = [col for col in manifest["categorical_features"] if col in candidates.columns]
    if not numeric_features:
        raise RuntimeError("No numeric features available for training.")

    train = candidates[candidates["split"].eq("train")].copy()
    test = candidates[candidates["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise RuntimeError("Training split produced empty train or test candidates.")
    y_train = train[TARGET_COL].astype(int).to_numpy()
    y_test = test[TARGET_COL].astype(int).to_numpy()
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise RuntimeError("Training and test splits must both contain positive and negative candidates.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=int(random_state),
    )
    model = Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])
    _log(
        "[INFO] Training model: "
        f"train_rows={len(train):,}; test_rows={len(test):,}; "
        f"features={len(numeric_features) + len(categorical_features):,}"
    )
    model.fit(
        train[numeric_features + categorical_features],
        y_train,
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )

    train_proba = model.predict_proba(train[numeric_features + categorical_features])[:, 1]
    test_proba = model.predict_proba(test[numeric_features + categorical_features])[:, 1]
    threshold_95 = _threshold_at_recall(y_train, train_proba, 0.95)
    threshold = float(threshold_95["threshold"]) if threshold_95 else 0.5

    scored = candidates.copy()
    scored.loc[train.index, "proba"] = train_proba
    scored.loc[test.index, "proba"] = test_proba

    metrics = {
        "feature_columns": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rows": int(y_train.sum()),
        "test_positive_rows": int(y_test.sum()),
        "threshold_95_recall_from_train": threshold_95,
        "classification_train_at_0_5": _classification_metrics(y_train, train_proba, 0.5),
        "classification_test_at_0_5": _classification_metrics(y_test, test_proba, 0.5),
        "classification_train_at_95_recall_threshold": _classification_metrics(y_train, train_proba, threshold),
        "classification_test_at_95_recall_threshold": _classification_metrics(y_test, test_proba, threshold),
        "selection_train": _selection_metrics(scored[scored["split"].eq("train")]),
        "selection_test": _selection_metrics(scored[scored["split"].eq("test")]),
    }
    return model, metrics, scored, numeric_features, categorical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an anchor-feature WFS council parcel group scorer.")
    parser.add_argument("--label-gpkg", default=str(DEFAULT_LABEL_GPKG))
    parser.add_argument("--label-layer", default=LABEL_LAYER)
    parser.add_argument("--input-gpkg", default=str(DEFAULT_INPUT_GPKG))
    parser.add_argument("--input-layer", default=INPUT_LAYER)
    parser.add_argument("--edge-cache", default=str(DEFAULT_EDGE_CACHE))
    parser.add_argument("--feature-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-labels", type=int, default=0, help="0 means all labels; positive value makes a reproducible smoke sample.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--tile-size", type=float, default=1000.0)
    parser.add_argument("--top-neighbors", type=int, default=14)
    parser.add_argument("--max-graph-depth", type=int, default=3)
    parser.add_argument("--max-group-size", type=int, default=12)
    parser.add_argument("--max-group-area", type=float, default=20000.0)
    parser.add_argument("--enum-per-label", type=int, default=28)
    parser.add_argument("--combo-local-n", default="12,14")
    parser.add_argument("--combo-max-extra", type=int, default=7)
    parser.add_argument("--combo-per-label", type=int, default=500)
    parser.add_argument(
        "--combo-extra-quotas",
        default="1:16,2:32,3:48,4:56,5:48,6:16,7:4;1:10,2:16,3:24,4:28,5:20,6:6,7:2",
        help="Semicolon-separated per-local-n quotas, formatted as extra_count:limit pairs.",
    )
    parser.add_argument("--max-overmerge-neighbors", type=int, default=5)
    parser.add_argument("--max-candidates-per-label", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--write-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(Path(args.feature_manifest))
    combo_local_ns = _parse_int_list(args.combo_local_n)
    combo_extra_quotas = _parse_quota_specs(str(args.combo_extra_quotas))
    labels = _read_labels(
        Path(args.label_gpkg),
        str(args.label_layer),
        max_labels=int(args.max_labels),
        random_state=int(args.random_state),
    )
    label_ids = set(int(v) for v in labels["label_id"])
    inputs = _read_inputs(Path(args.input_gpkg), str(args.input_layer), label_ids)
    split_by_label = _assign_spatial_split(
        labels,
        tile_size=float(args.tile_size),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )
    attrs, geoms = _build_node_indexes(inputs)
    adjacency, shared_by_pair = _load_adjacency(
        Path(args.edge_cache),
        set(attrs),
        top_neighbors=int(args.top_neighbors),
    )

    candidates, build_summary = build_candidates(
        labels=labels,
        inputs=inputs,
        attrs=attrs,
        geoms=geoms,
        adjacency=adjacency,
        shared_by_pair=shared_by_pair,
        split_by_label=split_by_label,
        max_graph_depth=int(args.max_graph_depth),
        max_group_size=int(args.max_group_size),
        max_group_area=float(args.max_group_area),
        enum_per_label=int(args.enum_per_label),
        combo_local_ns=combo_local_ns,
        combo_max_extra=int(args.combo_max_extra),
        combo_per_label=int(args.combo_per_label),
        combo_extra_quotas=combo_extra_quotas,
        max_overmerge_neighbors=int(args.max_overmerge_neighbors),
        max_candidates_per_label=int(args.max_candidates_per_label),
    )
    if candidates.empty:
        raise RuntimeError("No candidates generated.")
    _log(
        "[INFO] Candidate build complete: "
        f"rows={len(candidates):,}; positives={int(candidates['label'].sum()):,}; "
        f"labels={candidates['label_id'].nunique():,}"
    )

    model, metrics, scored, numeric_features, categorical_features = train_model(
        candidates=candidates,
        manifest=manifest,
        random_state=int(args.random_state),
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
    )
    summary = {
        "label_gpkg": str(args.label_gpkg),
        "label_layer": str(args.label_layer),
        "input_gpkg": str(args.input_gpkg),
        "input_layer": str(args.input_layer),
        "edge_cache": str(args.edge_cache),
        "feature_manifest": str(args.feature_manifest),
        "output_dir": str(output_dir),
        "model_file": str(output_dir / MODEL_FILE_NAME),
        "max_labels": int(args.max_labels),
        "params": {
            "random_state": int(args.random_state),
            "test_size": float(args.test_size),
            "tile_size": float(args.tile_size),
            "top_neighbors": int(args.top_neighbors),
            "max_graph_depth": int(args.max_graph_depth),
            "max_group_size": int(args.max_group_size),
            "max_group_area": float(args.max_group_area),
            "enum_per_label": int(args.enum_per_label),
            "combo_local_n": combo_local_ns,
            "combo_max_extra": int(args.combo_max_extra),
            "combo_per_label": int(args.combo_per_label),
            "combo_extra_quotas": combo_extra_quotas,
            "max_overmerge_neighbors": int(args.max_overmerge_neighbors),
            "max_candidates_per_label": int(args.max_candidates_per_label),
            "max_iter": int(args.max_iter),
            "learning_rate": float(args.learning_rate),
        },
        "candidate_build": build_summary,
        "metrics": metrics,
    }
    payload = {
        "model": model,
        "feature_manifest": manifest,
        "feature_columns": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "params": summary["params"],
        "metrics": metrics,
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if bool(args.write_candidates):
        scored.to_parquet(output_dir / CANDIDATES_FILE_NAME, index=False)
    else:
        preview_cols = [
            "label_id",
            "split",
            "label",
            "proba",
            "proposal_source",
            "candidate_clean_fids",
            "target_clean_fids",
            "positive_generated_by_enumeration",
        ]
        scored.sort_values(["split", "label_id", "proba"], ascending=[True, True, False])[
            [col for col in preview_cols if col in scored.columns]
        ].head(200000).to_csv(output_dir / "wfs_merged_council_anchor_group_scored_preview.csv", index=False)
    _log("[DONE] Anchor-feature group model trained")
    _log(json.dumps(summary["metrics"]["selection_test"], indent=2))


if __name__ == "__main__":
    main()

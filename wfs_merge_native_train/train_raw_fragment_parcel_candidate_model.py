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
import joblib
import numpy as np
import pandas as pd
import pyogrio
import shapely
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_wfs_merge_completion_model import _shape_metrics
from train_council_parcel_quality_model import _goal_gate, _morphology_metrics


DEFAULT_TARGET_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_merged_council_train.gpkg"
DEFAULT_TARGET_LAYER = "wfs_raw_merged_council_train_merged_only"
DEFAULT_WFS_CLEAN_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_CLEAN_LAYER = "wfs_raw_clean"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/raw_fragment_parcel_candidate_model_v1"

MODEL_FILE_NAME = "raw_fragment_parcel_candidate_model_v1.joblib"
METRICS_FILE_NAME = "raw_fragment_parcel_candidate_metrics_v1.json"
CANDIDATES_FILE_NAME = "raw_fragment_parcel_candidate_rows_v1.csv"

TARGET_COL = "label"
CATEGORICAL_FEATURES = ["anchor_kind", "role_signature"]
ID_COLUMNS = {
    "candidate_id",
    "target_train_component_id",
    "candidate_source_fids",
    "target_source_fids",
    "anchor_source_fid",
    "negative_type",
    "split",
    "spatial_group",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _ids_text(values: set[int] | frozenset[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(value)) for value in sorted(int(v) for v in values))


def _parse_id_set(value: object) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _role_text(row: pd.Series | dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(col, "") or "")
        for col in ["Theme", "DescriptiveGroup", "DescriptiveTerm", "raw_role"]
    ).lower()
    if "building" in text:
        return "building"
    if "land" in text:
        return "land"
    if "road" in text or "path" in text or "track" in text:
        return "road"
    if "gap" in text or "hole" in text:
        return "gapfill"
    return "other"


def _as_valid(values: Any) -> Any:
    valid = shapely.is_valid(values)
    if bool(np.all(valid)):
        return values
    out = np.asarray(values, dtype=object).copy()
    out[~np.asarray(valid, dtype=bool)] = shapely.make_valid(out[~np.asarray(valid, dtype=bool)])
    return out


def _union(geoms: list[Any]) -> Any:
    if not geoms:
        return None
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _read_targets(path: Path, layer: str, max_rows: int) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading council-train target parcels: {path}:{layer}")
    target = pyogrio.read_dataframe(path, layer=layer)
    target = target[target.geometry.notna() & ~target.geometry.is_empty].copy()
    if int(max_rows) > 0:
        target = target.head(int(max_rows)).copy()
        _log(f"[INFO] Applied max target rows: {len(target):,}")
    target.geometry = _as_valid(target.geometry.to_numpy())
    target["train_component_id"] = pd.to_numeric(target["train_component_id"], errors="coerce")
    target["anchor_wfs_fid"] = pd.to_numeric(target["anchor_wfs_fid"], errors="coerce")
    target = target[target["train_component_id"].notna() & target["anchor_wfs_fid"].notna()].copy()
    target["train_component_id"] = target["train_component_id"].astype("int64")
    target["anchor_wfs_fid"] = target["anchor_wfs_fid"].astype("int64")
    target["target_source_set"] = target["source_wfs_fids"].map(_parse_id_set)
    target = target[target["target_source_set"].map(len).ge(2)].copy()
    cent = target.geometry.centroid
    target["centroid_x"] = cent.x.astype("float64")
    target["centroid_y"] = cent.y.astype("float64")
    _log(f"[INFO] Target rows after cleanup={len(target):,}")
    return target.reset_index(drop=True)


def _assign_spatial_split(target: gpd.GeoDataFrame, *, tile_size: float, test_size: float, random_state: int) -> None:
    tile_x = np.floor(target["centroid_x"].to_numpy(dtype="float64") / float(tile_size)).astype("int64")
    tile_y = np.floor(target["centroid_y"].to_numpy(dtype="float64") / float(tile_size)).astype("int64")
    groups = pd.Series([f"{x}_{y}" for x, y in zip(tile_x, tile_y)], index=target.index)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    _train_idx, test_idx = next(splitter.split(target, groups=groups))
    target["split"] = "train"
    target.loc[target.index[test_idx], "split"] = "test"
    target["spatial_group"] = groups
    _log(
        "[INFO] Spatial split: "
        f"train={int(target['split'].eq('train').sum()):,}; "
        f"test={int(target['split'].eq('test').sum()):,}; groups={groups.nunique():,}"
    )


def _read_raw_clean_sources(
    *,
    path: Path,
    layer: str,
    source_ids: set[int],
    chunk_size: int,
) -> gpd.GeoDataFrame:
    ids = sorted(int(v) for v in source_ids)
    columns = [
        "source_fid",
        "clean_fid",
        "Theme",
        "DescriptiveGroup",
        "DescriptiveTerm",
        "raw_role",
        "clean_area",
        "clean_perimeter",
        "is_polygon_hole_fill",
        "is_enclosed_gap_fill",
    ]
    frames: list[gpd.GeoDataFrame] = []
    _log(f"[INFO] Reading raw-clean fragments by source_fid chunks: ids={len(ids):,}")
    for start in range(0, len(ids), int(chunk_size)):
        chunk = ids[start : start + int(chunk_size)]
        where = "source_fid IN (%s)" % ",".join(str(int(v)) for v in chunk)
        frame = pyogrio.read_dataframe(path, layer=layer, columns=columns, where=where)
        if not frame.empty:
            frames.append(frame)
        _log(f"[INFO] Read raw-clean chunk {start // int(chunk_size) + 1}: ids={len(chunk):,}; rows={len(frame):,}")
    if not frames:
        raise RuntimeError("No raw-clean fragments matched target source ids.")
    raw = pd.concat(frames, ignore_index=True)
    raw = gpd.GeoDataFrame(raw, geometry="geometry", crs=frames[0].crs)
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    raw.geometry = _as_valid(raw.geometry.to_numpy())
    raw["source_fid"] = pd.to_numeric(raw["source_fid"], errors="coerce")
    raw = raw[raw["source_fid"].notna()].copy()
    raw["source_fid"] = raw["source_fid"].astype("int64")
    raw["source_role"] = raw.apply(_role_text, axis=1)
    raw["area"] = raw.geometry.area.astype("float64")
    raw["perimeter"] = raw.geometry.length.astype("float64")
    _log(f"[INFO] Raw-clean matched rows={len(raw):,}; unique_source={raw['source_fid'].nunique():,}")
    return raw.reset_index(drop=True)


def _build_source_indexes(raw: gpd.GeoDataFrame) -> tuple[dict[int, list[Any]], dict[int, dict[str, Any]]]:
    geom_by_source: dict[int, list[Any]] = {}
    attrs_by_source: dict[int, dict[str, Any]] = {}
    for source_fid, group in raw.groupby("source_fid"):
        source = int(source_fid)
        geom_by_source[source] = list(group.geometry)
        roles = [str(v or "other") for v in group["source_role"]]
        role_counts = {role: roles.count(role) for role in sorted(set(roles))}
        attrs_by_source[source] = {
            "source_fid": source,
            "area": float(group["area"].sum()),
            "perimeter": float(group["perimeter"].sum()),
            "source_role": max(role_counts, key=role_counts.get) if role_counts else "other",
            "role_counts": role_counts,
            "part_count": int(len(group)),
            "hole_fill_count": int(pd.to_numeric(group.get("is_polygon_hole_fill", 0), errors="coerce").fillna(0).sum()),
            "gap_fill_count": int(pd.to_numeric(group.get("is_enclosed_gap_fill", 0), errors="coerce").fillna(0).sum()),
        }
    return geom_by_source, attrs_by_source


def _geom_for_sources(source_ids: set[int] | frozenset[int], geom_by_source: dict[int, list[Any]]) -> Any:
    geoms: list[Any] = []
    for source_fid in sorted(int(v) for v in source_ids):
        geoms.extend(geom_by_source.get(int(source_fid), []))
    return _union(geoms)


def _source_composition(source_ids: set[int] | frozenset[int], attrs_by_source: dict[int, dict[str, Any]]) -> dict[str, Any]:
    roles = [str(attrs_by_source.get(int(fid), {}).get("source_role", "missing")) for fid in source_ids]
    role_counts = {role: roles.count(role) for role in ["building", "land", "gapfill", "road", "other", "missing"]}
    areas = np.asarray([float(attrs_by_source.get(int(fid), {}).get("area", 0.0)) for fid in source_ids], dtype="float64")
    return {
        "candidate_source_count": int(len(source_ids)),
        "source_area_sum": float(areas.sum()) if len(areas) else 0.0,
        "source_area_max": float(areas.max()) if len(areas) else 0.0,
        "source_area_mean": float(areas.mean()) if len(areas) else 0.0,
        "largest_source_area_ratio": _safe_ratio(float(areas.max()) if len(areas) else 0.0, float(areas.sum()) if len(areas) else 0.0),
        "role_signature": "|".join(f"{role}:{count}" for role, count in role_counts.items() if count),
        **{f"role_{role}_count": int(count) for role, count in role_counts.items()},
    }


def _candidate_features(
    *,
    candidate_id: str,
    target_row: pd.Series,
    candidate_sources: set[int],
    label: int,
    negative_type: str,
    geom_by_source: dict[int, list[Any]],
    attrs_by_source: dict[int, dict[str, Any]],
    anchor_source_ids: set[int] | None = None,
    other_anchor_source_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    anchor_source_fid = int(target_row["anchor_wfs_fid"])
    target_sources = set(int(v) for v in target_row["target_source_set"])
    if anchor_source_ids is None:
        anchor_source_ids = {anchor_source_fid}
    candidate_geom = _geom_for_sources(set(candidate_sources), geom_by_source)
    base_geom = _geom_for_sources(set(anchor_source_ids), geom_by_source)
    if candidate_geom is None or candidate_geom.is_empty or base_geom is None or base_geom.is_empty:
        return None
    candidate_shape = _shape_metrics(candidate_geom)
    base_shape = _shape_metrics(base_geom)
    candidate_area = float(candidate_shape["area"])
    base_area = float(base_shape["area"])
    candidate_perimeter = float(candidate_shape["perimeter"])
    base_perimeter = float(base_shape["perimeter"])
    added_sources = set(candidate_sources) - set(anchor_source_ids)
    other_anchor_source_ids = set(other_anchor_source_ids or set())

    rec: dict[str, Any] = {
        "candidate_id": str(candidate_id),
        "target_train_component_id": int(target_row["train_component_id"]),
        "anchor_source_fid": anchor_source_fid,
        "candidate_source_fids": _ids_text(set(candidate_sources)),
        "target_source_fids": _ids_text(target_sources),
        "negative_type": str(negative_type),
        "split": str(target_row["split"]),
        "spatial_group": str(target_row["spatial_group"]),
        TARGET_COL: int(label),
        "anchor_kind": str(target_row.get("anchor_kind", "") or ""),
        "added_source_count": int(len(added_sources)),
        "contains_other_target_anchor_count": int(len(set(candidate_sources) & other_anchor_source_ids)),
        "candidate_area_to_base": _safe_ratio(candidate_area, base_area),
        "candidate_added_area_to_base": _safe_ratio(max(candidate_area - base_area, 0.0), base_area),
        "candidate_perimeter_to_base": _safe_ratio(candidate_perimeter, base_perimeter),
        "regularity_gain_vs_base": float(candidate_shape["regularity_score"] - base_shape["regularity_score"]),
        "compactness_gain_vs_base": float(candidate_shape["compactness"] - base_shape["compactness"]),
        "hull_gap_reduction_vs_base": float(base_shape["hull_gap_ratio"] - candidate_shape["hull_gap_ratio"]),
        "mrr_gap_reduction_vs_base": float(base_shape["mrr_gap_ratio"] - candidate_shape["mrr_gap_ratio"]),
        "notch_index_reduction_vs_base": float(base_shape["notch_index"] - candidate_shape["notch_index"]),
        "perimeter_hull_ratio_reduction_vs_base": float(
            base_shape["perimeter_hull_ratio"] - candidate_shape["perimeter_hull_ratio"]
        ),
    }
    rec.update(_source_composition(set(candidate_sources), attrs_by_source))
    for key, value in candidate_shape.items():
        rec[f"shape_{key}"] = float(value)
    for key, value in base_shape.items():
        rec[f"base_shape_{key}"] = float(value)
    for key, value in _morphology_metrics(candidate_geom).items():
        rec[f"morph_{key}"] = float(value)
    return rec


def _target_adjacency(target: gpd.GeoDataFrame, *, min_shared_edge: float, top_neighbors: int) -> dict[int, list[int]]:
    work = target.reset_index(drop=True)
    geoms = work.geometry.reset_index(drop=True)
    left, right = work.sindex.query(geoms.geometry.array, predicate="intersects")
    left = left.astype("int64")
    right = right.astype("int64")
    keep = left < right
    left = left[keep]
    right = right[keep]
    if len(left) == 0:
        return {}
    shared = shapely.length(shapely.intersection(shapely.boundary(geoms.iloc[left].array), shapely.boundary(geoms.iloc[right].array)))
    edges = pd.DataFrame({"left": left, "right": right, "shared": np.asarray(shared, dtype="float64")})
    edges = edges[edges["shared"].ge(float(min_shared_edge))].copy()
    both = pd.concat(
        [
            edges.rename(columns={"left": "src", "right": "dst"}),
            edges.rename(columns={"right": "src", "left": "dst"}),
        ],
        ignore_index=True,
    )
    both = both.sort_values(["src", "shared", "dst"], ascending=[True, False, True])
    both = both[both.groupby("src").cumcount().lt(int(top_neighbors))].copy()
    adj: dict[int, list[int]] = {}
    for row in both.itertuples(index=False):
        adj.setdefault(int(row.src), []).append(int(row.dst))
    return adj


def _build_candidates(
    target: gpd.GeoDataFrame,
    *,
    geom_by_source: dict[int, list[Any]],
    attrs_by_source: dict[int, dict[str, Any]],
    max_partial_per_target: int,
    max_neighbor_negatives: int,
    min_neighbor_shared_edge: float,
    top_target_neighbors: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    all_anchor_sources = {int(v) for v in target["anchor_wfs_fid"].astype(int)}
    adjacency = _target_adjacency(target, min_shared_edge=float(min_neighbor_shared_edge), top_neighbors=int(top_target_neighbors))
    records: list[dict[str, Any]] = []

    for pos, row in target.iterrows():
        target_sources = set(int(v) for v in row["target_source_set"])
        anchor = int(row["anchor_wfs_fid"])
        if anchor not in target_sources:
            continue
        positive = _candidate_features(
            candidate_id=f"pos:{int(row.train_component_id)}",
            target_row=row,
            candidate_sources=target_sources,
            label=1,
            negative_type="positive",
            geom_by_source=geom_by_source,
            attrs_by_source=attrs_by_source,
            other_anchor_source_ids=all_anchor_sources - {anchor},
        )
        if positive is not None:
            records.append(positive)

        anchor_only = _candidate_features(
            candidate_id=f"neg_anchor_only:{int(row.train_component_id)}",
            target_row=row,
            candidate_sources={anchor},
            label=0,
            negative_type="partial_anchor_only",
            geom_by_source=geom_by_source,
            attrs_by_source=attrs_by_source,
            other_anchor_source_ids=all_anchor_sources - {anchor},
        )
        if anchor_only is not None:
            records.append(anchor_only)

        added = sorted(target_sources - {anchor})
        if added:
            partial_sources: list[set[int]] = []
            partial_sources.append({anchor, added[0]})
            if len(added) > 1:
                partial_sources.append(set(target_sources) - {added[-1]})
            if len(added) > int(max_partial_per_target):
                sampled = list(rng.choice(np.asarray(added, dtype="int64"), size=int(max_partial_per_target), replace=False))
                partial_sources.append({anchor, *[int(v) for v in sampled]})
            for idx, sources in enumerate(partial_sources[: max(int(max_partial_per_target), 1)]):
                if set(sources) == target_sources:
                    continue
                rec = _candidate_features(
                    candidate_id=f"neg_partial:{int(row.train_component_id)}:{idx}",
                    target_row=row,
                    candidate_sources=set(sources),
                    label=0,
                    negative_type="partial",
                    geom_by_source=geom_by_source,
                    attrs_by_source=attrs_by_source,
                    other_anchor_source_ids=all_anchor_sources - {anchor},
                )
                if rec is not None:
                    records.append(rec)

        neighbor_positions = adjacency.get(int(pos), [])[: int(max_neighbor_negatives)]
        for nidx, neighbor_pos in enumerate(neighbor_positions):
            neighbor = target.iloc[int(neighbor_pos)]
            if str(neighbor["split"]) != str(row["split"]):
                continue
            neighbor_sources = set(int(v) for v in neighbor["target_source_set"])
            over_sources = target_sources | neighbor_sources
            rec = _candidate_features(
                candidate_id=f"neg_overmerge:{int(row.train_component_id)}:{int(neighbor.train_component_id)}",
                target_row=row,
                candidate_sources=over_sources,
                label=0,
                negative_type="overmerge_neighbor",
                geom_by_source=geom_by_source,
                attrs_by_source=attrs_by_source,
                other_anchor_source_ids=all_anchor_sources - {anchor},
            )
            if rec is not None:
                records.append(rec)
            non_anchor_neighbor = sorted(neighbor_sources - {int(neighbor["anchor_wfs_fid"])})
            if non_anchor_neighbor:
                sliver_sources = target_sources | {int(non_anchor_neighbor[0])}
                rec = _candidate_features(
                    candidate_id=f"neg_extra_fragment:{int(row.train_component_id)}:{int(neighbor.train_component_id)}:{nidx}",
                    target_row=row,
                    candidate_sources=sliver_sources,
                    label=0,
                    negative_type="extra_neighbor_fragment",
                    geom_by_source=geom_by_source,
                    attrs_by_source=attrs_by_source,
                    other_anchor_source_ids=all_anchor_sources - {anchor},
                )
                if rec is not None:
                    records.append(rec)

    dataset = pd.DataFrame.from_records(records)
    _log(f"[INFO] Built raw-fragment candidate rows={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
    return dataset


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLUMNS | {TARGET_COL, "sample_weight", "raw_fragment_proba"}
    excluded |= {"candidate_kind", "target_source_count", "target_overlap_count", "target_jaccard"}
    feature_cols = [column for column in dataset.columns if column not in excluded]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_cols]
    numeric = [column for column in feature_cols if column not in categorical and pd.api.types.is_numeric_dtype(dataset[column])]
    return numeric + categorical, numeric, categorical


def _add_pool_rank_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    group_key = "anchor_source_fid"
    if group_key not in out.columns:
        return out
    specs = [
        ("candidate_source_count", "max"),
        ("added_source_count", "max"),
        ("source_area_sum", "max"),
        ("shape_area", "max"),
        ("candidate_area_to_base", "max"),
    ]
    for column, reducer in specs:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        if reducer == "max":
            group_max = values.groupby(out[group_key]).transform("max").replace(0.0, np.nan)
            out[f"pool_{column}_to_max"] = (values / group_max).fillna(0.0)
            out[f"pool_{column}_rank_desc"] = values.groupby(out[group_key]).rank(method="average", ascending=False)
    if {"shape_area", "candidate_source_count"}.issubset(out.columns):
        out["pool_area_source_rank_mean"] = (
            out["pool_shape_area_rank_desc"].astype(float)
            + out["pool_candidate_source_count_rank_desc"].astype(float)
        ) / 2.0
    return out


def _threshold_at_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> dict[str, Any] | None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    eligible = np.where(precision[:-1] >= float(target_precision))[0]
    if len(eligible) == 0:
        return None
    idx = int(eligible[np.argmax(recall[:-1][eligible])])
    return {"threshold": float(thresholds[idx]), "precision": float(precision[idx]), "recall": float(recall[idx])}


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= float(threshold)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=[1, 0], zero_division=0)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "precision_positive": float(precision[0]),
        "recall_positive": float(recall[0]),
        "f1_positive": float(f1[0]),
        "precision_negative": float(precision[1]),
        "recall_negative": float(recall[1]),
        "f1_negative": float(f1[1]),
        "support_positive": int(support[0]),
        "support_negative": int(support[1]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).astype(int).tolist(),
        "threshold_at_precision_0.95": _threshold_at_precision(y_true, proba, 0.95),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a production-semantic raw-fragment parcel candidate model.")
    parser.add_argument("--target-gpkg", default=DEFAULT_TARGET_GPKG)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--candidate-output-csv", default="")
    parser.add_argument("--build-candidates-only", action="store_true")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--source-query-chunk-size", type=int, default=5000)
    parser.add_argument("--tile-size", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-partial-per-target", type=int, default=2)
    parser.add_argument("--max-neighbor-negatives", type=int, default=2)
    parser.add_argument("--min-neighbor-shared-edge", type=float, default=0.05)
    parser.add_argument("--top-target-neighbors", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-predictions-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.candidate_input_csv).strip():
        dataset = pd.read_csv(args.candidate_input_csv, low_memory=False)
        dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
        _log(f"[INFO] Loaded cached candidates={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
    else:
        target = _read_targets(Path(args.target_gpkg), str(args.target_layer), int(args.max_target_rows))
        _assign_spatial_split(target, tile_size=float(args.tile_size), test_size=float(args.test_size), random_state=int(args.random_state))
        source_ids: set[int] = set()
        for values in target["target_source_set"]:
            source_ids.update(int(v) for v in values)
        raw = _read_raw_clean_sources(
            path=Path(args.wfs_clean_gpkg),
            layer=str(args.wfs_clean_layer),
            source_ids=source_ids,
            chunk_size=int(args.source_query_chunk_size),
        )
        geom_by_source, attrs_by_source = _build_source_indexes(raw)
        dataset = _build_candidates(
            target,
            geom_by_source=geom_by_source,
            attrs_by_source=attrs_by_source,
            max_partial_per_target=int(args.max_partial_per_target),
            max_neighbor_negatives=int(args.max_neighbor_negatives),
            min_neighbor_shared_edge=float(args.min_neighbor_shared_edge),
            top_target_neighbors=int(args.top_target_neighbors),
            random_state=int(args.random_state),
        )
        candidate_output = Path(args.candidate_output_csv) if str(args.candidate_output_csv).strip() else output_dir / CANDIDATES_FILE_NAME
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(candidate_output, index=False)
        _log(f"[INFO] Wrote candidate CSV: rows={len(dataset):,}; path={candidate_output}")
    if bool(args.build_candidates_only):
        _log("[DONE] Candidate build complete")
        return

    dataset = _add_pool_rank_features(dataset)
    dataset["sample_weight"] = 1.0
    feature_cols, numeric_cols, categorical_cols = _feature_columns(dataset)
    train = dataset[dataset["split"].eq("train")].copy()
    test = dataset[dataset["split"].eq("test")].copy()
    _log(f"[INFO] Dataset rows={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Split/label counts:\n{dataset.groupby(['split', TARGET_COL]).size()}")
    _log(f"[INFO] Features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=4)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2_regularization),
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train[feature_cols], train[TARGET_COL].astype(int), model__sample_weight=train["sample_weight"].to_numpy(dtype="float64"))

    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    train_p95 = _threshold_at_precision(train[TARGET_COL].to_numpy(dtype=int), train_proba, 0.95) or {}
    threshold = float(train_p95.get("threshold", 0.5))
    test_metrics = _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, threshold)
    train_metrics = _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, threshold)
    goal_gate = _goal_gate(test[TARGET_COL].to_numpy(dtype=int), test_proba, min_precision=0.95, min_recall=0.95)

    payload = {
        "model_kind": "raw_fragment_parcel_candidate_scorer",
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "target_gpkg": str(args.target_gpkg),
            "target_layer": str(args.target_layer),
            "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
            "wfs_clean_layer": str(args.wfs_clean_layer),
            "candidate_input_csv": str(args.candidate_input_csv),
            "threshold_95p_from_train": threshold,
            "max_target_rows": int(args.max_target_rows),
            "max_partial_per_target": int(args.max_partial_per_target),
            "max_neighbor_negatives": int(args.max_neighbor_negatives),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    metrics = {
        "model_kind": payload["model_kind"],
        "model": str(output_dir / MODEL_FILE_NAME),
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "negative_type_counts": dataset["negative_type"].value_counts().to_dict(),
        "split_label_counts": {
            f"{split}_{label}": int(value)
            for (split, label), value in dataset.groupby(["split", TARGET_COL]).size().items()
        },
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "threshold_95p_from_train": threshold,
        "goal_gate_95_precision_95_recall": goal_gate,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not bool(args.skip_predictions_output):
        report = test[["candidate_id", "negative_type", "split", TARGET_COL, "candidate_source_fids", "target_source_fids"]].copy()
        report["raw_fragment_proba"] = test_proba
        report.sort_values("raw_fragment_proba", ascending=False).to_csv(output_dir / "raw_fragment_parcel_candidate_test_predictions_v1.csv", index=False)
    _log("[DONE] Raw-fragment parcel candidate training complete")
    _log(json.dumps(goal_gate, indent=2))
    _log(json.dumps(test_metrics, indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

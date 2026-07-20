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


DEFAULT_INPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_merged_council_train.gpkg"
DEFAULT_INPUT_LAYER = "wfs_raw_merged_council_train_merged_only"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/council_parcel_quality_model_v1"

MODEL_FILE_NAME = "council_parcel_quality_model_v1.joblib"
METRICS_FILE_NAME = "council_parcel_quality_metrics_v1.json"
PREDICTIONS_FILE_NAME = "council_parcel_quality_predictions_v1.csv"
CANDIDATES_FILE_NAME = "council_parcel_quality_candidates_v1.csv"

TARGET_COL = "label"
CATEGORICAL_FEATURES = ["anchor_kind_signature"]
ID_COLUMNS = {
    "candidate_id",
    "member_train_component_ids",
    "member_source_wfs_fids",
    "negative_type",
    "split",
    "spatial_group",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _ids_text(values: set[int] | list[int] | tuple[int, ...]) -> str:
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


def _as_valid(values: Any) -> Any:
    valid = shapely.is_valid(values)
    if bool(np.all(valid)):
        return values
    out = np.asarray(values, dtype=object).copy()
    out[~np.asarray(valid, dtype=bool)] = shapely.make_valid(out[~np.asarray(valid, dtype=bool)])
    return out


def _union(geoms: list[Any]) -> Any:
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _read_labels(path: Path, layer: str) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading council-train labels: {path}:{layer}")
    gdf = pyogrio.read_dataframe(path, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf.geometry = _as_valid(gdf.geometry.to_numpy())
    gdf["train_component_id"] = pd.to_numeric(gdf["train_component_id"], errors="coerce")
    gdf = gdf[gdf["train_component_id"].notna()].copy()
    gdf["train_component_id"] = gdf["train_component_id"].astype("int64")
    gdf["source_wfs_set"] = gdf["source_wfs_fids"].map(_parse_id_set)
    gdf["area"] = gdf.geometry.area.astype("float64")
    gdf["perimeter"] = gdf.geometry.length.astype("float64")
    cent = gdf.geometry.centroid
    gdf["centroid_x"] = cent.x.astype("float64")
    gdf["centroid_y"] = cent.y.astype("float64")
    _log(f"[INFO] Council-train rows={len(gdf):,}")
    return gdf.reset_index(drop=True)


def _assign_spatial_split(gdf: gpd.GeoDataFrame, *, tile_size: float, test_size: float, random_state: int) -> pd.Series:
    tile_x = np.floor(gdf["centroid_x"].to_numpy(dtype="float64") / float(tile_size)).astype("int64")
    tile_y = np.floor(gdf["centroid_y"].to_numpy(dtype="float64") / float(tile_size)).astype("int64")
    groups = pd.Series([f"{x}_{y}" for x, y in zip(tile_x, tile_y)], index=gdf.index)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, test_idx = next(splitter.split(gdf, groups=groups))
    split = pd.Series("train", index=gdf.index, dtype="object")
    split.iloc[test_idx] = "test"
    gdf["spatial_group"] = groups
    _log(
        "[INFO] Spatial split: "
        f"train={int(split.eq('train').sum()):,}; test={int(split.eq('test').sum()):,}; "
        f"groups={groups.nunique():,}"
    )
    return split


def _shared_edge_pairs(
    gdf: gpd.GeoDataFrame,
    *,
    min_shared_edge: float,
    top_neighbors: int,
) -> pd.DataFrame:
    if len(gdf) < 2:
        return pd.DataFrame(columns=["left_pos", "right_pos", "shared_edge_len"])
    _log(f"[INFO] Building council-train shared-edge pairs: rows={len(gdf):,}")
    work = gdf.reset_index(drop=True)
    geoms = work.geometry.reset_index(drop=True)
    sindex = work.sindex
    left_local, right_pos = sindex.query(geoms.geometry.array, predicate="intersects")
    if len(left_local) == 0:
        return pd.DataFrame(columns=["left_pos", "right_pos", "shared_edge_len"])
    left_pos = left_local.astype("int64")
    right_pos = right_pos.astype("int64")
    keep = left_pos < right_pos
    left_pos = left_pos[keep]
    right_pos = right_pos[keep]
    if len(left_pos) == 0:
        return pd.DataFrame(columns=["left_pos", "right_pos", "shared_edge_len"])
    shared = shapely.length(
        shapely.intersection(
            shapely.boundary(geoms.iloc[left_pos].array),
            shapely.boundary(geoms.iloc[right_pos].array),
        )
    )
    edges = pd.DataFrame(
        {
            "left_pos": left_pos.astype("int64"),
            "right_pos": right_pos.astype("int64"),
            "shared_edge_len": np.asarray(shared, dtype="float64"),
        }
    )
    edges = edges[edges["shared_edge_len"].ge(float(min_shared_edge))].copy()
    if edges.empty:
        return edges
    both = pd.concat(
        [
            edges.rename(columns={"left_pos": "src", "right_pos": "dst"}),
            edges.rename(columns={"right_pos": "src", "left_pos": "dst"}),
        ],
        ignore_index=True,
    )
    both = both.sort_values(["src", "shared_edge_len", "dst"], ascending=[True, False, True])
    both["rank"] = both.groupby("src").cumcount()
    both = both[both["rank"].lt(int(top_neighbors))].copy()
    keys = {
        (min(int(row.src), int(row.dst)), max(int(row.src), int(row.dst)))
        for row in both.itertuples(index=False)
    }
    out = pd.DataFrame.from_records(
        [
            {
                "left_pos": int(left),
                "right_pos": int(right),
                "shared_edge_len": float(
                    edges.loc[
                        ((edges["left_pos"].eq(left)) & (edges["right_pos"].eq(right)))
                        | ((edges["left_pos"].eq(right)) & (edges["right_pos"].eq(left))),
                        "shared_edge_len",
                    ].max()
                ),
            }
            for left, right in sorted(keys)
        ]
    )
    _log(f"[INFO] Shared-edge pairs={len(out):,}")
    return out


def _component_summary(rows: gpd.GeoDataFrame) -> dict[str, Any]:
    source_sets = [set(v) if isinstance(v, set) else _parse_id_set(v) for v in rows["source_wfs_set"]]
    source_union: set[int] = set()
    for values in source_sets:
        source_union.update(int(v) for v in values)
    anchor_kinds = [str(v or "unknown").lower() for v in rows["anchor_kind"]]
    kind_counts = {kind: anchor_kinds.count(kind) for kind in sorted(set(anchor_kinds))}
    return {
        "raw_source_count_sum": float(pd.to_numeric(rows["raw_source_count"], errors="coerce").fillna(0).sum()),
        "raw_source_count_mean": float(pd.to_numeric(rows["raw_source_count"], errors="coerce").fillna(0).mean()),
        "raw_source_count_max": float(pd.to_numeric(rows["raw_source_count"], errors="coerce").fillna(0).max()),
        "source_wfs_unique_count": int(len(source_union)),
        "uprn_count_sum": float(pd.to_numeric(rows["uprn_count"], errors="coerce").fillna(0).sum()),
        "uprn_count_max": float(pd.to_numeric(rows["uprn_count"], errors="coerce").fillna(0).max()),
        "anchor_building_count": int(kind_counts.get("building", 0)),
        "anchor_land_count": int(kind_counts.get("land", 0)),
        "anchor_other_count": int(sum(v for k, v in kind_counts.items() if k not in {"building", "land"})),
        "anchor_kind_signature": "|".join(f"{kind}:{count}" for kind, count in kind_counts.items()),
        "fallback_holes_filled_sum": float(pd.to_numeric(rows["fallback_holes_filled"], errors="coerce").fillna(0).sum()),
        "fallback_area_delta_ratio_sum": float(
            pd.to_numeric(rows["fallback_area_delta_ratio"], errors="coerce").fillna(0).sum()
        ),
    }


def _half_clip_geometry(geom: Any, *, seed: int) -> Any | None:
    if geom is None or geom.is_empty:
        return None
    minx, miny, maxx, maxy = shapely.bounds(geom)
    width = float(maxx - minx)
    height = float(maxy - miny)
    if width <= 0.0 or height <= 0.0:
        return None
    pad = max(width, height) * 0.05 + 1.0
    midx = (float(minx) + float(maxx)) / 2.0
    midy = (float(miny) + float(maxy)) / 2.0
    boxes = [
        shapely.box(float(minx) - pad, float(miny) - pad, midx, float(maxy) + pad),
        shapely.box(midx, float(miny) - pad, float(maxx) + pad, float(maxy) + pad),
        shapely.box(float(minx) - pad, float(miny) - pad, float(maxx) + pad, midy),
        shapely.box(float(minx) - pad, midy, float(maxx) + pad, float(maxy) + pad),
    ]
    area = float(shapely.area(geom))
    if area <= 0.0:
        return None
    for offset in range(len(boxes)):
        clipped = shapely.intersection(geom, boxes[(int(seed) + offset) % len(boxes)])
        if clipped is None or clipped.is_empty:
            continue
        clipped = shapely.make_valid(clipped) if not bool(shapely.is_valid(clipped)) else clipped
        frac = _safe_ratio(float(shapely.area(clipped)), area)
        if 0.25 <= frac <= 0.88:
            return clipped
    return None


def _attached_neighbor_sliver(anchor_geom: Any, neighbor_geom: Any, *, seed: int) -> Any | None:
    if anchor_geom is None or neighbor_geom is None or anchor_geom.is_empty or neighbor_geom.is_empty:
        return None
    anchor_area = float(shapely.area(anchor_geom))
    neighbor_area = float(shapely.area(neighbor_geom))
    if anchor_area <= 0.0 or neighbor_area <= 0.0:
        return None
    buffer_dist = min(max(anchor_area ** 0.5 * 0.12, 1.0), 12.0)
    for scale in [1.0, 1.8, 3.0]:
        sliver = shapely.intersection(neighbor_geom, shapely.buffer(anchor_geom, buffer_dist * scale))
        if sliver is None or sliver.is_empty:
            continue
        sliver = shapely.make_valid(sliver) if not bool(shapely.is_valid(sliver)) else sliver
        frac = _safe_ratio(float(shapely.area(sliver)), neighbor_area)
        if 0.05 <= frac <= 0.75:
            geom = shapely.union_all(np.asarray([anchor_geom, sliver], dtype=object))
            if geom is None or geom.is_empty:
                continue
            geom = shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom
            if float(shapely.area(geom)) > anchor_area * 1.03:
                return geom
    clipped = _half_clip_geometry(neighbor_geom, seed=seed)
    if clipped is None or clipped.is_empty:
        return None
    geom = shapely.union_all(np.asarray([anchor_geom, clipped], dtype=object))
    if geom is None or geom.is_empty:
        return None
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _buffer_clean(geom: Any, distance: float) -> Any:
    try:
        out = shapely.buffer(geom, float(distance))
    except Exception:
        return None
    if out is None or out.is_empty:
        return out
    return shapely.make_valid(out) if not bool(shapely.is_valid(out)) else out


def _morphology_metrics(geom: Any) -> dict[str, float]:
    area = float(shapely.area(geom))
    perimeter = float(shapely.length(geom))
    if area <= 0.0:
        return {
            "opening_loss_ratio_r1": 0.0,
            "opening_loss_ratio_r2": 0.0,
            "erosion_survival_ratio_r1": 0.0,
            "erosion_survival_ratio_r2": 0.0,
            "boundary_band_ratio_r1": 0.0,
            "boundary_band_ratio_r2": 0.0,
            "slenderness_area_perimeter": 0.0,
        }
    base_radius = min(max(area ** 0.5 * 0.018, 0.35), 3.0)
    metrics: dict[str, float] = {
        "slenderness_area_perimeter": _safe_ratio(4.0 * area, perimeter * perimeter),
    }
    for suffix, radius in (("r1", base_radius), ("r2", base_radius * 2.0)):
        eroded = _buffer_clean(geom, -float(radius))
        eroded_area = float(shapely.area(eroded)) if eroded is not None and not eroded.is_empty else 0.0
        opened = None
        if eroded is not None and not eroded.is_empty:
            opened = _buffer_clean(eroded, float(radius))
        opened_area = float(shapely.area(opened)) if opened is not None and not opened.is_empty else 0.0
        expanded = _buffer_clean(geom, float(radius))
        expanded_area = float(shapely.area(expanded)) if expanded is not None and not expanded.is_empty else area
        metrics[f"opening_loss_ratio_{suffix}"] = _safe_ratio(max(area - opened_area, 0.0), area)
        metrics[f"erosion_survival_ratio_{suffix}"] = _safe_ratio(eroded_area, area)
        metrics[f"boundary_band_ratio_{suffix}"] = _safe_ratio(max(expanded_area - area, 0.0), area)
    return metrics


def _candidate_features(
    *,
    candidate_id: str,
    member_rows: gpd.GeoDataFrame,
    label: int,
    negative_type: str,
    split: str,
    candidate_geom: Any | None = None,
    candidate_geom_is_member_union: bool = False,
) -> dict[str, Any]:
    if candidate_geom is None:
        geoms = [geom for geom in member_rows.geometry if geom is not None and not geom.is_empty]
        geom = _union(geoms)
        member_area_sum = float(member_rows["area"].sum())
        member_perimeter_sum = float(member_rows["perimeter"].sum())
    else:
        geom = candidate_geom
        if bool(candidate_geom_is_member_union):
            member_area_sum = float(shapely.area(geom))
            member_perimeter_sum = float(shapely.length(geom))
        else:
            member_area_sum = float(member_rows["area"].sum())
            member_perimeter_sum = float(member_rows["perimeter"].sum())
    base_geom = _union([base for base in member_rows.geometry if base is not None and not base.is_empty])
    shape = _shape_metrics(geom)
    base_shape = _shape_metrics(base_geom) if base_geom is not None and not base_geom.is_empty else shape
    candidate_area = float(shape["area"])
    base_area = float(base_shape["area"])
    candidate_perimeter = float(shape["perimeter"])
    base_perimeter = float(base_shape["perimeter"])
    train_ids = [int(v) for v in member_rows["train_component_id"]]
    source_texts = [str(v or "") for v in member_rows["source_wfs_fids"]]
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "member_train_component_ids": _ids_text(train_ids),
        "member_source_wfs_fids": "|".join(source_texts),
        "negative_type": str(negative_type),
        "split": str(split),
        "spatial_group": str(member_rows["spatial_group"].iloc[0]),
        TARGET_COL: int(label),
        "member_area_sum": member_area_sum,
        "member_perimeter_sum": member_perimeter_sum,
        "candidate_area_to_member_area": _safe_ratio(candidate_area, member_area_sum),
        "candidate_perimeter_to_member_perimeter": _safe_ratio(candidate_perimeter, member_perimeter_sum),
        "base_area": base_area,
        "base_perimeter": base_perimeter,
        "candidate_area_to_base": _safe_ratio(candidate_area, base_area),
        "candidate_added_area_to_base": _safe_ratio(max(candidate_area - base_area, 0.0), base_area),
        "candidate_removed_area_to_base": _safe_ratio(max(base_area - candidate_area, 0.0), base_area),
        "candidate_perimeter_to_base": _safe_ratio(candidate_perimeter, base_perimeter),
        "regularity_gain_vs_base": float(shape["regularity_score"] - base_shape["regularity_score"]),
        "compactness_gain_vs_base": float(shape["compactness"] - base_shape["compactness"]),
        "hull_gap_reduction_vs_base": float(base_shape["hull_gap_ratio"] - shape["hull_gap_ratio"]),
        "mrr_gap_reduction_vs_base": float(base_shape["mrr_gap_ratio"] - shape["mrr_gap_ratio"]),
        "notch_index_reduction_vs_base": float(base_shape["notch_index"] - shape["notch_index"]),
        "perimeter_hull_ratio_reduction_vs_base": float(
            base_shape["perimeter_hull_ratio"] - shape["perimeter_hull_ratio"]
        ),
    }
    record.update(_component_summary(member_rows))
    for key, value in shape.items():
        record[f"shape_{key}"] = float(value)
    for key, value in _morphology_metrics(geom).items():
        record[f"morph_{key}"] = float(value)
    return record


def _positive_records(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index in range(len(gdf)):
        row = gdf.iloc[[row_index]]
        records.append(
            _candidate_features(
                candidate_id=f"pos:{int(row['train_component_id'].iloc[0])}",
                member_rows=row,
                label=1,
                negative_type="positive",
                split=str(row["split"].iloc[0]),
            )
        )
    return records


def _negative_records(
    gdf: gpd.GeoDataFrame,
    edges: pd.DataFrame,
    *,
    max_negative_rows: int,
    max_partial_negative_rows: int,
    max_attached_negative_rows: int,
    random_state: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    partial_source = gdf
    if int(max_partial_negative_rows) > 0 and len(partial_source) > int(max_partial_negative_rows):
        partial_source = partial_source.sample(n=int(max_partial_negative_rows), random_state=int(random_state))
    for row_index, row in partial_source.iterrows():
        clipped = _half_clip_geometry(row.geometry, seed=int(row.train_component_id))
        if clipped is None or clipped.is_empty:
            continue
        member_rows = gdf.iloc[[int(row_index)]]
        records.append(
            _candidate_features(
                candidate_id=f"neg_partial_cut:{int(row.train_component_id)}",
                member_rows=member_rows,
                label=0,
                negative_type="partial_cut",
                split=str(row.split),
                candidate_geom=clipped,
                candidate_geom_is_member_union=True,
            )
        )
        if len(records) % 2000 == 0:
            _log(f"[INFO] Built hard partial negatives={len(records):,}")
    if edges.empty:
        return records
    pairs = edges.copy()
    pairs["split_left"] = gdf.loc[pairs["left_pos"].astype(int), "split"].to_numpy()
    pairs["split_right"] = gdf.loc[pairs["right_pos"].astype(int), "split"].to_numpy()
    pairs = pairs[pairs["split_left"].eq(pairs["split_right"])].copy()

    attached_pairs = pairs
    if int(max_attached_negative_rows) > 0 and len(attached_pairs) > int(max_attached_negative_rows):
        attached_pairs = attached_pairs.sample(n=int(max_attached_negative_rows), random_state=int(random_state) + 11)
    for idx, row in enumerate(attached_pairs.itertuples(index=False), start=1):
        left = int(row.left_pos)
        right = int(row.right_pos)
        anchor_row = gdf.iloc[[left]]
        geom = _attached_neighbor_sliver(
            gdf.geometry.iloc[left],
            gdf.geometry.iloc[right],
            seed=int(gdf["train_component_id"].iloc[left]),
        )
        if geom is None or geom.is_empty:
            continue
        records.append(
            _candidate_features(
                candidate_id=f"neg_attached_sliver:{int(gdf['train_component_id'].iloc[left])}:{int(gdf['train_component_id'].iloc[right])}",
                member_rows=anchor_row,
                label=0,
                negative_type="attached_neighbor_sliver",
                split=str(anchor_row["split"].iloc[0]),
                candidate_geom=geom,
                candidate_geom_is_member_union=True,
            )
        )
        if idx % 2000 == 0:
            _log(f"[INFO] Built attached-sliver negatives checked={idx:,}; total_negatives={len(records):,}")

    if int(max_negative_rows) > 0 and len(pairs) > int(max_negative_rows):
        pairs = pairs.sample(n=int(max_negative_rows), random_state=int(random_state))
    for row in pairs.itertuples(index=False):
        left = int(row.left_pos)
        right = int(row.right_pos)
        member_rows = gdf.iloc[[left, right]]
        split = str(member_rows["split"].iloc[0])
        records.append(
            _candidate_features(
                candidate_id=f"neg_overmerge_2:{int(member_rows['train_component_id'].iloc[0])}:{int(member_rows['train_component_id'].iloc[1])}",
                member_rows=member_rows,
                label=0,
                negative_type="overmerge_2",
                split=split,
            )
        )
    return records


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLUMNS | {TARGET_COL, "sample_weight", "parcel_quality_proba"}
    feature_cols = [column for column in dataset.columns if column not in excluded]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_cols]
    numeric = [
        column
        for column in feature_cols
        if column not in categorical and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    return numeric + categorical, numeric, categorical


def _read_candidate_dataset(path: Path) -> pd.DataFrame:
    _log(f"[INFO] Reading cached council parcel quality candidates: {path}")
    dataset = pd.read_csv(path, low_memory=False)
    if TARGET_COL not in dataset.columns:
        raise RuntimeError(f"Candidate cache missing required column: {TARGET_COL}")
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
    _log(f"[INFO] Cached candidate rows={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
    return dataset


def _thresholds_at_precision(y_true: np.ndarray, proba: np.ndarray, targets: list[float]) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    out: dict[str, Any] = {}
    for target in targets:
        eligible = np.where(precision[:-1] >= float(target))[0]
        if len(eligible) == 0:
            out[str(target)] = None
            continue
        idx = int(eligible[np.argmax(recall[:-1][eligible])])
        out[str(target)] = {
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
        }
    return out


def _goal_gate(y_true: np.ndarray, proba: np.ndarray, *, min_precision: float, min_recall: float) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    valid = np.where((precision[:-1] >= float(min_precision)) & (recall[:-1] >= float(min_recall)))[0]
    if len(valid) > 0:
        idx = int(valid[np.argmax(recall[:-1][valid])])
        return {
            "pass": True,
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "min_precision": float(min_precision),
            "min_recall": float(min_recall),
        }
    den = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        den,
        out=np.zeros_like(den, dtype="float64"),
        where=den != 0,
    )
    best_idx = int(np.argmax(f1)) if len(f1) else 0
    return {
        "pass": False,
        "threshold": float(thresholds[best_idx]) if len(thresholds) else None,
        "precision": float(precision[best_idx]) if len(precision) else None,
        "recall": float(recall[best_idx]) if len(recall) else None,
        "f1": float(f1[best_idx]) if len(f1) else None,
        "min_precision": float(min_precision),
        "min_recall": float(min_recall),
        "recall_at_precision_0.95": (_thresholds_at_precision(y_true, proba, [float(min_precision)]).get(str(float(min_precision))) or {}),
    }


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= float(threshold)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[1, 0],
        zero_division=0,
    )
    out: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "precision_positive": float(precision[0]),
        "recall_positive": float(recall[0]),
        "f1_positive": float(f1[0]),
        "support_positive": int(support[0]),
        "precision_negative": float(precision[1]),
        "recall_negative": float(recall[1]),
        "f1_negative": float(f1[1]),
        "support_negative": int(support[1]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).astype(int).tolist(),
        "thresholds_at_precision": _thresholds_at_precision(y_true, proba, [0.95, 0.97, 0.99]),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["roc_auc"] = None
    try:
        out["average_precision"] = float(average_precision_score(y_true, proba))
    except ValueError:
        out["average_precision"] = None
    return out


def _false_positive_counts(rows: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict[str, int]:
    work = rows.copy()
    work["_pred"] = (proba >= float(threshold)).astype(int)
    false_positive = work[work[TARGET_COL].astype(int).eq(0) & work["_pred"].eq(1)]
    if false_positive.empty:
        return {}
    return {str(key): int(value) for key, value in false_positive["negative_type"].value_counts().items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dataset-only council parcel quality model.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--input-layer", default=DEFAULT_INPUT_LAYER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--candidate-output-csv", default="")
    parser.add_argument("--build-candidates-only", action="store_true")
    parser.add_argument("--tile-size", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--top-neighbors", type=int, default=6)
    parser.add_argument("--max-negative-rows", type=int, default=90000)
    parser.add_argument("--max-partial-negative-rows", type=int, default=8000)
    parser.add_argument("--max-attached-negative-rows", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=23)
    parser.add_argument("--l2-regularization", type=float, default=0.06)
    parser.add_argument("--skip-predictions-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.candidate_input_csv).strip():
        dataset = _read_candidate_dataset(Path(args.candidate_input_csv))
    else:
        gdf = _read_labels(Path(args.input_gpkg), str(args.input_layer))
        gdf["split"] = _assign_spatial_split(
            gdf,
            tile_size=float(args.tile_size),
            test_size=float(args.test_size),
            random_state=int(args.random_state),
        )
        edges = _shared_edge_pairs(gdf, min_shared_edge=float(args.min_shared_edge), top_neighbors=int(args.top_neighbors))

        records = _positive_records(gdf)
        records.extend(
            _negative_records(
                gdf,
                edges,
                max_negative_rows=int(args.max_negative_rows),
                max_partial_negative_rows=int(args.max_partial_negative_rows),
                max_attached_negative_rows=int(args.max_attached_negative_rows),
                random_state=int(args.random_state),
            )
        )
        dataset = pd.DataFrame.from_records(records)
    if dataset.empty:
        raise RuntimeError("No candidate rows were generated.")
    if str(args.candidate_output_csv).strip():
        candidate_output = Path(args.candidate_output_csv)
    else:
        candidate_output = output_dir / CANDIDATES_FILE_NAME
    if not str(args.candidate_input_csv).strip():
        dataset.to_csv(candidate_output, index=False)
        _log(f"[INFO] Wrote candidate cache: rows={len(dataset):,}; path={candidate_output}")
    if bool(args.build_candidates_only):
        _log("[DONE] Candidate build complete")
        return
    dataset["sample_weight"] = 1.0
    _log(f"[INFO] Candidate dataset rows={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Split/label counts:\n{dataset.groupby(['split', TARGET_COL]).size()}")

    feature_cols, numeric_cols, categorical_cols = _feature_columns(dataset)
    train = dataset[dataset["split"].eq("train")].copy()
    test = dataset[dataset["split"].eq("test")].copy()
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
    _log("[INFO] Training council parcel quality model")
    pipeline.fit(
        train[feature_cols],
        train[TARGET_COL].astype(int),
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )

    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    all_proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]

    train_thresholds = _thresholds_at_precision(train[TARGET_COL].to_numpy(dtype=int), train_proba, [0.95])
    selected = train_thresholds.get("0.95") or {}
    threshold_95p = float(selected.get("threshold", args.threshold))
    _log(f"[INFO] Train-derived threshold for precision>=0.95: {threshold_95p:.8f}")

    dataset = dataset.copy()
    dataset["parcel_quality_proba"] = all_proba
    train_metrics = _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, threshold_95p)
    test_metrics = _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, threshold_95p)
    all_metrics = _metrics(dataset[TARGET_COL].to_numpy(dtype=int), all_proba, threshold_95p)

    payload = {
        "model_kind": "council_parcel_quality_scorer",
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "input_gpkg": str(args.input_gpkg),
            "input_layer": str(args.input_layer),
            "tile_size": float(args.tile_size),
            "test_size": float(args.test_size),
            "min_shared_edge": float(args.min_shared_edge),
            "top_neighbors": int(args.top_neighbors),
            "max_negative_rows": int(args.max_negative_rows),
            "max_partial_negative_rows": int(args.max_partial_negative_rows),
            "max_attached_negative_rows": int(args.max_attached_negative_rows),
            "threshold_95p_from_train": threshold_95p,
            "random_state": int(args.random_state),
            "max_iter": int(args.max_iter),
            "learning_rate": float(args.learning_rate),
            "max_leaf_nodes": int(args.max_leaf_nodes),
            "l2_regularization": float(args.l2_regularization),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)

    metrics = {
        "model_kind": "council_parcel_quality_scorer",
        "input_gpkg": str(args.input_gpkg),
        "input_layer": str(args.input_layer),
        "output_dir": str(output_dir),
        "model": str(output_dir / MODEL_FILE_NAME),
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "negative_type_counts": dataset["negative_type"].value_counts().to_dict(),
        "split_label_counts": {
            f"{split}_{label}": int(value)
            for (split, label), value in dataset.groupby(["split", TARGET_COL]).size().items()
        },
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "threshold_95p_from_train": threshold_95p,
        "test_goal_gate_95_precision_95_recall": _goal_gate(
            test[TARGET_COL].to_numpy(dtype=int),
            test_proba,
            min_precision=0.95,
            min_recall=0.95,
        ),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "test_false_positive_counts_by_negative_type": _false_positive_counts(test, test_proba, threshold_95p),
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if bool(args.skip_predictions_output):
        _log("[INFO] Skipping prediction CSV write")
    else:
        report_cols = [
            "candidate_id",
            "member_train_component_ids",
            "negative_type",
            "split",
            TARGET_COL,
            "parcel_quality_proba",
            "shape_area",
            "shape_regularity_score",
            "shape_hull_gap_ratio",
            "shape_notch_index",
            "raw_source_count_sum",
            "uprn_count_sum",
            "anchor_building_count",
            "anchor_land_count",
            "anchor_kind_signature",
        ]
        report_cols = [column for column in report_cols if column in dataset.columns]
        dataset[report_cols].sort_values("parcel_quality_proba", ascending=False).to_csv(
            output_dir / PREDICTIONS_FILE_NAME,
            index=False,
        )

    _log("[DONE] Council parcel quality training complete")
    _log(json.dumps(test_metrics, indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

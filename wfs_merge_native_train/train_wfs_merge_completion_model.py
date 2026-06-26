#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_wfs_merge_edge_model import _add_derived_features


DEFAULT_BASE_DIR = "/data/sheffield/spatial/base-map"
DEFAULT_EDGE_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1"
DEFAULT_INPUT_PREDICTION_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1/"
    "model_predicted_polygons_threshold_090_shape_guard.gpkg"
)
DEFAULT_EDGE_CSV = (
    "/data/sheffield/spatial/base-map/"
    "sheffield_wp5_wfs_merge_training_small_edges_50k_uprn_rect_semantic_clean.csv"
)
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3"
MODEL_FILE_NAME = "wfs_merge_completion_model_v3.joblib"
CANDIDATES_FILE_NAME = "completion_candidates_v3.csv"
PREDICTIONS_FILE_NAME = "completion_candidate_predictions_v3.csv"


CATEGORICAL_FEATURES = [
    "role_pair",
    "candidate_role",
    "candidate_theme",
    "candidate_descriptive_group",
    "candidate_descriptive_term",
    "candidate_make",
]
ID_COLS = {
    "component_id",
    "candidate_component_id",
    "component_source_fid",
    "candidate_source_fid",
    "component_reference_merge_fid",
    "candidate_reference_merge_fid",
    "edge_label_from_training",
    "spatial_group",
}
TARGET_COL = "label"


def _log(message: str) -> None:
    print(message, flush=True)


def _role(theme: object) -> str:
    text = str(theme or "").lower()
    if "building" in text:
        return "building"
    if "land" in text:
        return "land"
    return "other"


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _iter_polygons(geom):
    if geom is None or shapely.is_empty(geom):
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        yield geom
    elif geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _ring_area(ring) -> float:
    try:
        return float(abs(Polygon(ring).area))
    except Exception:
        return 0.0


def _mrr_dimensions(geom) -> dict[str, float]:
    mrr = shapely.minimum_rotated_rectangle(geom)
    if not hasattr(mrr, "exterior"):
        return {
            "mrr_min_side": 0.0,
            "mrr_max_side": 0.0,
            "mrr_aspect_ratio": 1.0,
            "mrr_perimeter": 0.0,
            "mrr_orientation_deg": 0.0,
        }

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
        return {
            "mrr_min_side": 0.0,
            "mrr_max_side": 0.0,
            "mrr_aspect_ratio": 1.0,
            "mrr_perimeter": 0.0,
            "mrr_orientation_deg": 0.0,
        }

    min_side = min(lengths)
    max_side = max(lengths)
    orientation = angles[int(np.argmax(lengths))]
    return {
        "mrr_min_side": float(min_side),
        "mrr_max_side": float(max_side),
        "mrr_aspect_ratio": _safe_ratio(max_side, min_side),
        "mrr_perimeter": float(sum(lengths)),
        "mrr_orientation_deg": float(orientation),
    }


def _angle_delta_to_mrr_axis(angle: float, reference_angle: float) -> float:
    return abs(((angle - reference_angle + 45.0) % 90.0) - 45.0)


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values or not weights or sum(weights) <= 0.0:
        return 45.0
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype="float64")[order]
    sorted_weights = np.asarray(weights, dtype="float64")[order]
    cutoff = float(quantile) * float(sorted_weights.sum())
    idx = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def _orthogonality_metrics(geom, reference_angle: float) -> dict[str, float]:
    deviations: list[float] = []
    lengths: list[float] = []
    vertex_count = 0
    for poly in _iter_polygons(geom) or []:
        coords = list(poly.exterior.coords)
        vertex_count += max(len(coords) - 1, 0)
        for start, end in zip(coords, coords[1:]):
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            deviations.append(_angle_delta_to_mrr_axis(angle, reference_angle))
            lengths.append(length)
    total = float(sum(lengths))
    if total <= 0.0:
        return {
            "orthogonal_mean_deviation_deg": 45.0,
            "orthogonal_p90_deviation_deg": 45.0,
            "orthogonal_len_ratio_10deg": 0.0,
            "orthogonal_len_ratio_15deg": 0.0,
            "exterior_vertex_count": int(vertex_count),
        }
    weighted_mean = float(np.average(np.asarray(deviations, dtype="float64"), weights=np.asarray(lengths, dtype="float64")))
    len_ratio_10 = float(sum(length for length, dev in zip(lengths, deviations) if dev <= 10.0) / total)
    len_ratio_15 = float(sum(length for length, dev in zip(lengths, deviations) if dev <= 15.0) / total)
    return {
        "orthogonal_mean_deviation_deg": weighted_mean,
        "orthogonal_p90_deviation_deg": _weighted_quantile(deviations, lengths, 0.90),
        "orthogonal_len_ratio_10deg": len_ratio_10,
        "orthogonal_len_ratio_15deg": len_ratio_15,
        "exterior_vertex_count": int(vertex_count),
    }


def _hole_metrics(geom) -> dict[str, float]:
    hole_count = 0
    hole_area = 0.0
    for poly in _iter_polygons(geom) or []:
        hole_count += len(poly.interiors)
        hole_area += sum(_ring_area(ring) for ring in poly.interiors)
    return {
        "hole_count": int(hole_count),
        "hole_area": float(hole_area),
    }


def _shape_metrics(geom) -> dict[str, float]:
    area = float(shapely.area(geom))
    perimeter = float(shapely.length(geom))
    mrr_area = float(shapely.area(shapely.minimum_rotated_rectangle(geom))) or 1.0
    hull_area = float(shapely.area(shapely.convex_hull(geom)))
    hull_perimeter = float(shapely.length(shapely.convex_hull(geom)))
    envelope_area = float(shapely.area(shapely.envelope(geom))) or 1.0
    mrr = _mrr_dimensions(geom)
    holes = _hole_metrics(geom)
    orthogonal = _orthogonality_metrics(geom, mrr["mrr_orientation_deg"])
    mrr_gap_ratio = max(mrr_area - area, 0.0) / (area or 1.0)
    hull_gap_ratio = max(hull_area - area, 0.0) / (area or 1.0)
    perimeter_mrr_ratio = _safe_ratio(perimeter, mrr["mrr_perimeter"])
    perimeter_hull_ratio = _safe_ratio(perimeter, hull_perimeter)
    convexity = _safe_ratio(area, hull_area)
    mrr_ratio = area / mrr_area
    orthogonal_fit = orthogonal["orthogonal_len_ratio_10deg"]
    perimeter_fit = min(_safe_ratio(1.0, perimeter_mrr_ratio), 1.0)
    regularity_score = (
        0.35 * mrr_ratio
        + 0.25 * convexity
        + 0.20 * perimeter_fit
        + 0.20 * orthogonal_fit
    )
    metrics = {
        "area": area,
        "perimeter": perimeter,
        "mrr_ratio": mrr_ratio,
        "mrr_gap_ratio": mrr_gap_ratio,
        "mrr_gap_area": max(mrr_area - area, 0.0),
        "hull_gap_ratio": hull_gap_ratio,
        "hull_gap_area": max(hull_area - area, 0.0),
        "convexity": convexity,
        "bbox_fill_ratio": _safe_ratio(area, envelope_area),
        "compactness": 4.0 * math.pi * area / ((perimeter * perimeter) or 1.0),
        "perimeter_mrr_ratio": perimeter_mrr_ratio,
        "perimeter_hull_ratio": perimeter_hull_ratio,
        "boundary_complexity": perimeter_hull_ratio,
        "notch_index": hull_gap_ratio * max(perimeter_hull_ratio - 1.0, 0.0),
        "hole_count": holes["hole_count"],
        "hole_area_ratio": _safe_ratio(holes["hole_area"], area),
        "regularity_score": float(regularity_score),
    }
    metrics.update({k: v for k, v in mrr.items() if k != "mrr_orientation_deg"})
    metrics.update(orthogonal)
    return metrics

def _source_composition(group: gpd.GeoDataFrame) -> dict[str, float]:
    theme = group["Theme"].fillna("").astype(str)
    is_building = theme.str.contains("building", case=False, regex=False)
    is_land = theme.str.contains("land", case=False, regex=False)
    areas = group.geometry.area.astype(float)
    building_area = float(areas[is_building].sum())
    land_area = float(areas[is_land].sum())
    total = building_area + land_area
    return {
        "component_building_count": int(is_building.sum()),
        "component_land_count": int(is_land.sum()),
        "component_building_area": building_area,
        "component_land_area": land_area,
        "component_building_area_ratio": _safe_ratio(building_area, total),
        "component_land_area_ratio": _safe_ratio(land_area, total),
        "component_largest_land_ratio": _safe_ratio(float(areas[is_land].max()) if bool(is_land.any()) else 0.0, total),
    }


def _update_prefixed_metrics(record: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        record[f"{prefix}_{name}"] = float(value)


def _component_tables(
    predicted: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[int, set[int]], dict[int, Any], dict[int, dict[str, float]]]:
    comp_uprn = dict(zip(predicted["pred_component_id"].astype(int), predicted["pred_uprn_count"].fillna(0).astype(int)))
    comp_sources: dict[int, set[int]] = {}
    comp_geom: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int)):
        source_ids = set(group["source_fid"].astype(int))
        geom = shapely.union_all(group.geometry.array)
        shape = _shape_metrics(geom)
        refs = sorted({int(v) for v in group["reference_merge_fid"].dropna().astype(int)})
        comp_sources[int(comp_id)] = source_ids
        comp_geom[int(comp_id)] = geom
        rec = {
            "component_id": int(comp_id),
            "component_source_count": int(len(group)),
            "component_uprn_count": int(comp_uprn.get(int(comp_id), 0)),
            "component_reference_merge_fid_count": int(len(refs)),
            "component_reference_merge_fid": int(refs[0]) if len(refs) == 1 else np.nan,
        }
        _update_prefixed_metrics(rec, "component", shape)
        rec.update(_source_composition(group))
        rows.append(rec)
    component_df = pd.DataFrame.from_records(rows)
    component_shape = {int(row["component_id"]): row for row in rows}
    return component_df, comp_sources, comp_geom, component_shape


def _build_completion_candidates(
    *,
    edge_df: pd.DataFrame,
    predicted: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    min_edge_proba: float,
    max_candidate_area: float,
) -> pd.DataFrame:
    component_df, comp_sources, comp_geom, component_shape = _component_tables(predicted, sources)
    comp_lookup = component_df.set_index("component_id").to_dict("index")

    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    source_to_ref = dict(zip(sources["source_fid"].astype(int), sources["reference_merge_fid"]))
    source_geom = dict(zip(sources["source_fid"].astype(int), sources.geometry))
    source_attrs = sources.set_index(sources["source_fid"].astype(int))

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    candidate_edges = edge_df[edge_df["edge_proba"].ge(float(min_edge_proba))].copy()
    for _, row in candidate_edges.iterrows():
        a = int(row["left_source_fid"])
        b = int(row["right_source_fid"])
        if a not in source_to_component or b not in source_to_component:
            continue
        ca = int(source_to_component[a])
        cb = int(source_to_component[b])
        if ca == cb:
            continue
        for comp_id, cand_comp_id, comp_source, cand_source in ((ca, cb, a, b), (cb, ca, b, a)):
            comp = comp_lookup.get(int(comp_id))
            cand = comp_lookup.get(int(cand_comp_id))
            if comp is None or cand is None:
                continue
            if int(comp["component_source_count"]) < 2:
                continue
            if int(cand["component_source_count"]) != 1:
                continue
            if int(comp["component_uprn_count"]) != 1 or int(cand["component_uprn_count"]) != 0:
                continue
            if int(comp["component_reference_merge_fid_count"]) != 1:
                continue
            cand_area = float(source_geom[cand_source].area)
            if cand_area > float(max_candidate_area):
                continue
            key = (int(comp_id), int(cand_source))
            if key in seen:
                continue
            seen.add(key)

            before = component_shape[int(comp_id)]
            candidate_geom = source_geom[int(cand_source)]
            candidate_shape = _shape_metrics(candidate_geom)
            after_geom = shapely.union_all([comp_geom[int(comp_id)], candidate_geom])
            after = _shape_metrics(after_geom)
            candidate_attr = source_attrs.loc[int(cand_source)]
            cand_ref = source_to_ref.get(int(cand_source))
            label = int(pd.notna(cand_ref) and int(cand_ref) == int(comp["component_reference_merge_fid"]))
            centroid = shapely.centroid(comp_geom[int(comp_id)])
            shared_edge_len = float(row["shared_edge_len"])
            rec = {
                "component_id": int(comp_id),
                "candidate_component_id": int(cand_comp_id),
                "component_source_fid": int(comp_source),
                "candidate_source_fid": int(cand_source),
                "component_reference_merge_fid": int(comp["component_reference_merge_fid"]),
                "candidate_reference_merge_fid": int(cand_ref) if pd.notna(cand_ref) else np.nan,
                "label": label,
                "edge_proba": float(row["edge_proba"]),
                "edge_label_from_training": int(row["label"]),
                "role_pair": str(row.get("role_pair", "")),
                "shared_edge_len": shared_edge_len,
                "shared_ratio_small_perimeter": float(row["shared_ratio_small_perimeter"]),
                "edge_union_mrr_ratio": float(row["union_mrr_ratio"]),
                "edge_union_hull_gap_ratio": float(row["union_hull_gap_ratio"]),
                "edge_union_compactness": float(row["union_compactness"]),
                "candidate_area_ratio_to_component": _safe_ratio(cand_area, float(before["component_area"])),
                "candidate_shared_ratio_candidate_perimeter": _safe_ratio(shared_edge_len, float(candidate_shape["perimeter"])),
                "candidate_shared_ratio_component_perimeter": _safe_ratio(shared_edge_len, float(before["component_perimeter"])),
                "candidate_area_to_before_mrr_gap": _safe_ratio(cand_area, float(before["component_mrr_gap_area"])),
                "candidate_area_to_before_hull_gap": _safe_ratio(cand_area, float(before["component_hull_gap_area"])),
                "candidate_role": _role(candidate_attr.get("Theme")),
                "candidate_theme": str(candidate_attr.get("Theme") or ""),
                "candidate_descriptive_group": str(candidate_attr.get("DescriptiveGroup") or ""),
                "candidate_descriptive_term": str(candidate_attr.get("DescriptiveTerm") or ""),
                "candidate_make": str(candidate_attr.get("Make") or ""),
                "component_source_count": int(comp["component_source_count"]),
                "component_building_count": int(comp["component_building_count"]),
                "component_land_count": int(comp["component_land_count"]),
                "component_building_area_ratio": float(comp["component_building_area_ratio"]),
                "component_land_area_ratio": float(comp["component_land_area_ratio"]),
                "component_largest_land_ratio": float(comp["component_largest_land_ratio"]),
                "mrr_gain": float(after["mrr_ratio"] - before["component_mrr_ratio"]),
                "hull_gap_reduction": float(before["component_hull_gap_ratio"] - after["hull_gap_ratio"]),
                "compactness_gain": float(after["compactness"] - before["component_compactness"]),
                "mrr_gap_reduction": float(before["component_mrr_gap_ratio"] - after["mrr_gap_ratio"]),
                "convexity_gain": float(after["convexity"] - before["component_convexity"]),
                "bbox_fill_ratio_gain": float(after["bbox_fill_ratio"] - before["component_bbox_fill_ratio"]),
                "perimeter_mrr_ratio_reduction": float(before["component_perimeter_mrr_ratio"] - after["perimeter_mrr_ratio"]),
                "perimeter_hull_ratio_reduction": float(before["component_perimeter_hull_ratio"] - after["perimeter_hull_ratio"]),
                "boundary_complexity_reduction": float(before["component_boundary_complexity"] - after["boundary_complexity"]),
                "notch_index_reduction": float(before["component_notch_index"] - after["notch_index"]),
                "regularity_score_gain": float(after["regularity_score"] - before["component_regularity_score"]),
                "orthogonal_mean_deviation_reduction": float(
                    before["component_orthogonal_mean_deviation_deg"] - after["orthogonal_mean_deviation_deg"]
                ),
                "orthogonal_p90_deviation_reduction": float(
                    before["component_orthogonal_p90_deviation_deg"] - after["orthogonal_p90_deviation_deg"]
                ),
                "orthogonal_len_ratio_10deg_gain": float(
                    after["orthogonal_len_ratio_10deg"] - before["component_orthogonal_len_ratio_10deg"]
                ),
                "orthogonal_len_ratio_15deg_gain": float(
                    after["orthogonal_len_ratio_15deg"] - before["component_orthogonal_len_ratio_15deg"]
                ),
                "mrr_aspect_ratio_delta": float(after["mrr_aspect_ratio"] - before["component_mrr_aspect_ratio"]),
                "mrr_aspect_ratio_log_delta": float(
                    math.log1p(after["mrr_aspect_ratio"]) - math.log1p(before["component_mrr_aspect_ratio"])
                ),
                "hole_count_delta": float(after["hole_count"] - before["component_hole_count"]),
                "hole_area_ratio_delta": float(after["hole_area_ratio"] - before["component_hole_area_ratio"]),
                "exterior_vertex_count_delta": float(after["exterior_vertex_count"] - before["component_exterior_vertex_count"]),
                "area_added_ratio": _safe_ratio(cand_area, float(after["area"])),
                "mid_x": float(shapely.get_x(centroid)),
                "mid_y": float(shapely.get_y(centroid)),
            }
            _update_prefixed_metrics(rec, "candidate", candidate_shape)
            _update_prefixed_metrics(
                rec,
                "component",
                {name: before[f"component_{name}"] for name in after.keys()},
            )
            _update_prefixed_metrics(rec, "after", after)
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _split_dataset(df: pd.DataFrame, cell_size: float, random_state: int) -> pd.Series:
    x_cell = np.floor(df["mid_x"].astype(float) / cell_size).astype(int)
    y_cell = np.floor(df["mid_y"].astype(float) / cell_size).astype(int)
    groups = x_cell.astype(str) + "_" + y_cell.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, df[TARGET_COL], groups))
    train_val = df.iloc[train_val_idx]
    train_val_groups = groups.iloc[train_val_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=random_state + 1)
    train_rel_idx, val_rel_idx = next(splitter.split(train_val, train_val[TARGET_COL], train_val_groups))
    split = pd.Series("train", index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    split.iloc[train_val_idx[val_rel_idx]] = "validation"
    split.iloc[train_val_idx[train_rel_idx]] = "train"
    df["spatial_group"] = groups
    return split


def _choose_threshold_for_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    candidates = np.where(precision[:-1] >= target_precision)[0]
    if len(candidates) == 0:
        best_idx = int(np.nanargmax(2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)))
    else:
        best_idx = int(candidates[np.argmax(recall[:-1][candidates])])
    f1 = 2 * precision[best_idx] * recall[best_idx] / max(precision[best_idx] + recall[best_idx], 1e-12)
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1),
    }


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "support_positive": int(support[1]),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "f1_negative": float(f1[0]),
        "support_negative": int(support[0]),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).astype(int).tolist(),
    }


def _fixed_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    return {f"{threshold:.2f}": _metrics(y_true, proba, threshold) for threshold in thresholds}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the second-stage WFS merge completion model.")
    parser.add_argument("--input-prediction-gpkg", default=DEFAULT_INPUT_PREDICTION_GPKG)
    parser.add_argument("--edge-csv", default=DEFAULT_EDGE_CSV)
    parser.add_argument("--edge-model-dir", default=DEFAULT_EDGE_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-candidate-edge-proba", type=float, default=0.0)
    parser.add_argument("--max-candidate-area", type=float, default=120.0)
    parser.add_argument("--cell-size", type=float, default=1000.0)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_gpkg = Path(args.input_prediction_gpkg)
    edge_model_dir = Path(args.edge_model_dir)
    _log(f"[INFO] Reading base prediction: {input_gpkg}")
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
    sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")

    _log("[INFO] Loading first-stage edge model and edge candidates")
    edge_model = joblib.load(edge_model_dir / "wfs_merge_edge_model_v1.joblib")
    edge_meta = json.loads((edge_model_dir / "metrics.json").read_text())
    edge_features = edge_meta["feature_columns"]
    edge_df = _add_derived_features(pd.read_csv(args.edge_csv))
    edge_df["edge_proba"] = edge_model.predict_proba(edge_df[edge_features])[:, 1]

    _log("[INFO] Building completion candidate dataset")
    dataset = _build_completion_candidates(
        edge_df=edge_df,
        predicted=predicted,
        sources=sources,
        min_edge_proba=float(args.min_candidate_edge_proba),
        max_candidate_area=float(args.max_candidate_area),
    )
    if dataset.empty:
        raise RuntimeError("No completion candidates were built.")
    dataset["split"] = _split_dataset(dataset, float(args.cell_size), int(args.random_state))

    excluded = ID_COLS | {TARGET_COL, "split", "mid_x", "mid_y"}
    feature_cols = [c for c in dataset.columns if c not in excluded]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols and pd.api.types.is_numeric_dtype(dataset[c])]

    train = dataset[dataset["split"].eq("train")]
    validation = dataset[dataset["split"].eq("validation")]
    test = dataset[dataset["split"].eq("test")]

    _log(f"[INFO] Candidates={len(dataset):,}; label_counts={dataset[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Split sizes: train={len(train):,} validation={len(validation):,} test={len(test):,}")
    _log(f"[INFO] Features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    _log("[INFO] Training completion model")
    pipeline.fit(train[feature_cols], train[TARGET_COL].astype(int))
    val_proba = pipeline.predict_proba(validation[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    all_proba = pd.Series(index=dataset.index, dtype="float64")
    all_proba.loc[train.index] = train_proba
    all_proba.loc[validation.index] = val_proba
    all_proba.loc[test.index] = test_proba
    threshold_info = _choose_threshold_for_precision(validation[TARGET_COL].to_numpy(dtype=int), val_proba, float(args.target_precision))
    threshold = threshold_info["threshold"]

    metrics = {
        "input_prediction_gpkg": str(input_gpkg),
        "edge_csv": str(args.edge_csv),
        "output_dir": str(output_dir),
        "min_candidate_edge_proba": float(args.min_candidate_edge_proba),
        "max_candidate_area": float(args.max_candidate_area),
        "target_precision": float(args.target_precision),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "excluded_columns": sorted(excluded),
        "candidate_rows": int(len(dataset)),
        "positive_rows": int(dataset[TARGET_COL].sum()),
        "negative_rows": int((dataset[TARGET_COL] == 0).sum()),
        "threshold_selected_on_validation": threshold_info,
        "train_at_threshold": _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, threshold),
        "validation_at_threshold": _metrics(validation[TARGET_COL].to_numpy(dtype=int), val_proba, threshold),
        "test_at_threshold": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, threshold),
        "all_rows_fixed_thresholds": _fixed_threshold_metrics(
            dataset[TARGET_COL].to_numpy(dtype=int),
            all_proba.to_numpy(dtype=float),
            [0.5, 0.7, 0.9, 0.95],
        ),
    }

    model_path = output_dir / MODEL_FILE_NAME
    dataset_path = output_dir / CANDIDATES_FILE_NAME
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / PREDICTIONS_FILE_NAME
    _log(f"[INFO] Writing outputs: {output_dir}")
    joblib.dump(pipeline, model_path)
    dataset.to_csv(dataset_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prediction_frames = []
    prediction_report_cols = [
        "component_id",
        "candidate_component_id",
        "component_source_fid",
        "candidate_source_fid",
        "component_reference_merge_fid",
        "candidate_reference_merge_fid",
        "edge_proba",
        "role_pair",
        "candidate_area",
        "candidate_mrr_ratio",
        "candidate_hull_gap_ratio",
        "candidate_perimeter_mrr_ratio",
        "candidate_orthogonal_len_ratio_10deg",
        "candidate_area_ratio_to_component",
        "candidate_shared_ratio_candidate_perimeter",
        "component_mrr_ratio",
        "component_mrr_gap_ratio",
        "component_hull_gap_ratio",
        "component_convexity",
        "component_perimeter_mrr_ratio",
        "component_perimeter_hull_ratio",
        "component_boundary_complexity",
        "component_notch_index",
        "component_regularity_score",
        "component_orthogonal_len_ratio_10deg",
        "after_mrr_ratio",
        "after_mrr_gap_ratio",
        "after_hull_gap_ratio",
        "after_convexity",
        "after_perimeter_mrr_ratio",
        "after_perimeter_hull_ratio",
        "after_boundary_complexity",
        "after_notch_index",
        "after_regularity_score",
        "after_orthogonal_len_ratio_10deg",
        "mrr_gain",
        "mrr_gap_reduction",
        "hull_gap_reduction",
        "convexity_gain",
        "perimeter_mrr_ratio_reduction",
        "perimeter_hull_ratio_reduction",
        "boundary_complexity_reduction",
        "notch_index_reduction",
        "regularity_score_gain",
        "orthogonal_len_ratio_10deg_gain",
        TARGET_COL,
    ]
    prediction_report_cols = [c for c in prediction_report_cols if c in dataset.columns]
    for split_name, rows, proba in (
        ("train", train, train_proba),
        ("validation", validation, val_proba),
        ("test", test, test_proba),
    ):
        out = rows[prediction_report_cols].copy()
        out["split"] = split_name
        out["completion_proba"] = proba
        out["completion_pred"] = out["completion_proba"].ge(threshold).astype(int)
        prediction_frames.append(out)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)

    _log("[DONE] Completion training complete")
    _log(f"[DONE] threshold={threshold:.6f}")
    _log(
        "[DONE] validation precision={:.4f} recall={:.4f} f1={:.4f}".format(
            metrics["validation_at_threshold"]["precision_positive"],
            metrics["validation_at_threshold"]["recall_positive"],
            metrics["validation_at_threshold"]["f1_positive"],
        )
    )
    _log(
        "[DONE] test precision={:.4f} recall={:.4f} f1={:.4f}".format(
            metrics["test_at_threshold"]["precision_positive"],
            metrics["test_at_threshold"]["recall_positive"],
            metrics["test_at_threshold"]["f1_positive"],
        )
    )
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

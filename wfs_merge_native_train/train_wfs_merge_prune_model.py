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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3/"
    "model_predicted_polygons_completion_v3_threshold_090_strict_regularity_guard.gpkg"
)
DEFAULT_QA_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3/"
    "wfs_merge_v3_QA_review_package.gpkg"
)
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_operation_models_v1"
MODEL_FILE_NAME = "wfs_merge_prune_model_v1.joblib"
CANDIDATES_FILE_NAME = "prune_candidates_v1.csv"
PREDICTIONS_FILE_NAME = "prune_candidate_predictions_v1.csv"


TARGET_COL = "label"
CATEGORICAL_FEATURES = [
    "source_role",
    "source_theme",
    "source_descriptive_group",
    "source_descriptive_term",
    "source_make",
]
ID_COLS = {
    "component_id",
    "source_fid",
    "original_pred_component_id",
    "review_fid",
    "reference_merge_fid",
    "label_source",
    "manual_review_class",
    "mid_x",
    "mid_y",
}


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


def _part_count(geom) -> int:
    if geom is None or shapely.is_empty(geom):
        return 0
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        return 1
    if geom_type == "MultiPolygon":
        return len(geom.geoms)
    if geom_type == "GeometryCollection":
        return sum(1 for part in geom.geoms if getattr(part, "geom_type", "") in {"Polygon", "MultiPolygon"})
    return 1


def _update_prefixed(record: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        record[f"{prefix}_{name}"] = float(value)


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
        "component_building_area_ratio": _safe_ratio(building_area, total),
        "component_land_area_ratio": _safe_ratio(land_area, total),
        "component_largest_land_ratio": _safe_ratio(float(areas[is_land].max()) if bool(is_land.any()) else 0.0, total),
    }


def build_prune_candidates(predicted: gpd.GeoDataFrame, sources: gpd.GeoDataFrame) -> pd.DataFrame:
    comp_uprn = dict(zip(predicted["pred_component_id"].astype(int), predicted["pred_uprn_count"].fillna(0).astype(int)))
    records: list[dict[str, Any]] = []

    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        comp_id = int(comp_id)
        if len(group) < 2:
            continue

        full_geom = shapely.union_all(group.geometry.array)
        before = _shape_metrics(full_geom)
        full_centroid = shapely.centroid(full_geom)
        comp = _source_composition(group)
        group_area = float(before["area"])
        group_perimeter = float(before["perimeter"])

        for row in group.itertuples():
            source_fid = int(row.source_fid)
            rest = group[group["source_fid"].astype(int).ne(source_fid)]
            if rest.empty:
                continue
            rest_geom = shapely.union_all(rest.geometry.array)
            source_geom = row.geometry
            after = _shape_metrics(rest_geom)
            source_shape = _shape_metrics(source_geom)
            source_role = _role(getattr(row, "Theme", ""))
            source_area = float(source_shape["area"])
            source_perimeter = float(source_shape["perimeter"])
            shared_len = float(shapely.length(shapely.intersection(shapely.boundary(source_geom), shapely.boundary(rest_geom))))
            source_centroid = shapely.centroid(source_geom)
            centroid_distance = float(shapely.distance(source_centroid, full_centroid))

            rest_theme = rest["Theme"].fillna("").astype(str)
            rest_building_count = int(rest_theme.str.contains("building", case=False, regex=False).sum())
            rest_land_count = int(rest_theme.str.contains("land", case=False, regex=False).sum())

            rec: dict[str, Any] = {
                "component_id": comp_id,
                "source_fid": source_fid,
                "original_pred_component_id": int(getattr(row, "original_pred_component_id", comp_id))
                if pd.notna(getattr(row, "original_pred_component_id", comp_id))
                else comp_id,
                "reference_merge_fid": float(getattr(row, "reference_merge_fid", np.nan))
                if pd.notna(getattr(row, "reference_merge_fid", np.nan))
                else np.nan,
                "source_role": source_role,
                "source_theme": str(getattr(row, "Theme", "") or ""),
                "source_descriptive_group": str(getattr(row, "DescriptiveGroup", "") or ""),
                "source_descriptive_term": str(getattr(row, "DescriptiveTerm", "") or ""),
                "source_make": str(getattr(row, "Make", "") or ""),
                "source_area_ratio": _safe_ratio(source_area, group_area),
                "source_perimeter_ratio": _safe_ratio(source_perimeter, group_perimeter),
                "source_shared_edge_len": shared_len,
                "source_shared_ratio_source_perimeter": _safe_ratio(shared_len, source_perimeter),
                "source_shared_ratio_component_perimeter": _safe_ratio(shared_len, group_perimeter),
                "source_centroid_distance": centroid_distance,
                "source_centroid_distance_norm": _safe_ratio(centroid_distance, float(before["mrr_max_side"])),
                "component_source_count": int(len(group)),
                "component_uprn_count": int(comp_uprn.get(comp_id, 0)),
                "rest_source_count": int(len(rest)),
                "rest_part_count": int(_part_count(rest_geom)),
                "rest_building_count": rest_building_count,
                "rest_land_count": rest_land_count,
                "remove_mrr_gain": float(after["mrr_ratio"] - before["mrr_ratio"]),
                "remove_mrr_gap_reduction": float(before["mrr_gap_ratio"] - after["mrr_gap_ratio"]),
                "remove_hull_gap_reduction": float(before["hull_gap_ratio"] - after["hull_gap_ratio"]),
                "remove_convexity_gain": float(after["convexity"] - before["convexity"]),
                "remove_perimeter_mrr_ratio_reduction": float(before["perimeter_mrr_ratio"] - after["perimeter_mrr_ratio"]),
                "remove_perimeter_hull_ratio_reduction": float(before["perimeter_hull_ratio"] - after["perimeter_hull_ratio"]),
                "remove_boundary_complexity_reduction": float(before["boundary_complexity"] - after["boundary_complexity"]),
                "remove_notch_index_reduction": float(before["notch_index"] - after["notch_index"]),
                "remove_regularity_score_gain": float(after["regularity_score"] - before["regularity_score"]),
                "remove_orthogonal_len_ratio_10deg_gain": float(
                    after["orthogonal_len_ratio_10deg"] - before["orthogonal_len_ratio_10deg"]
                ),
                "remove_mrr_aspect_ratio_delta": float(after["mrr_aspect_ratio"] - before["mrr_aspect_ratio"]),
                "remove_hole_count_delta": float(after["hole_count"] - before["hole_count"]),
                "mid_x": float(shapely.get_x(full_centroid)),
                "mid_y": float(shapely.get_y(full_centroid)),
            }
            rec.update(comp)
            _update_prefixed(rec, "component", before)
            _update_prefixed(rec, "source", source_shape)
            _update_prefixed(rec, "after_remove", after)
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _read_layer_if_exists(path: Path, layer: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame()


def _manual_maps(qa_gpkg: Path) -> tuple[dict[int, str], set[tuple[int, int]], dict[int, int]]:
    annotations = _read_layer_if_exists(qa_gpkg, "10_user_review_04_first15_annotations")
    review_class_by_component: dict[int, str] = {}
    review_fid_by_component: dict[int, int] = {}
    if not annotations.empty:
        for row in annotations.itertuples(index=False):
            comp_id = int(row.pred_component_id)
            review_class_by_component[comp_id] = str(row.user_review_class)
            review_fid_by_component[comp_id] = int(row.review_fid)

    remove_diag = _read_layer_if_exists(qa_gpkg, "11_overmerge_remove_source_diagnostics")
    positive_remove: set[tuple[int, int]] = set()
    if not remove_diag.empty:
        best = remove_diag[remove_diag["remove_rank_in_component"].astype(int).eq(1)].copy()
        for row in best.itertuples(index=False):
            positive_remove.add((int(row.pred_component_id), int(row.candidate_remove_source_fid)))
    return review_class_by_component, positive_remove, review_fid_by_component


def assign_labels(candidates: pd.DataFrame, qa_gpkg: Path) -> pd.DataFrame:
    out = candidates.copy()
    review_class_by_component, positive_remove, review_fid_by_component = _manual_maps(qa_gpkg)
    out["manual_review_class"] = out["component_id"].map(review_class_by_component)
    out["review_fid"] = out["component_id"].map(review_fid_by_component)
    out[TARGET_COL] = np.nan
    out["label_source"] = "unlabeled"
    out["sample_weight"] = 0.0

    for idx, row in out.iterrows():
        key = (int(row["component_id"]), int(row["source_fid"]))
        review_class = row["manual_review_class"]
        if pd.notna(review_class):
            if review_class == "overmerge_neighbor":
                out.at[idx, TARGET_COL] = 1 if key in positive_remove else 0
                out.at[idx, "label_source"] = "manual_overmerge_positive" if key in positive_remove else "manual_overmerge_negative"
                out.at[idx, "sample_weight"] = 50.0 if key in positive_remove else 20.0
            elif review_class in {"ok", "undermerge_missing_own"}:
                out.at[idx, TARGET_COL] = 0
                out.at[idx, "label_source"] = f"manual_{review_class}_negative"
                out.at[idx, "sample_weight"] = 20.0
            continue

        source_is_land = str(row["source_role"]).lower() == "land"
        component_irregular = (
            float(row["component_mrr_ratio"]) < 0.9
            or float(row["component_hull_gap_ratio"]) > 0.08
            or float(row["component_regularity_score"]) < 0.94
        )
        strong_remove_improves_shape = (
            float(row["remove_regularity_score_gain"]) >= 0.08
            and float(row["remove_mrr_gain"]) >= 0.12
            and float(row["remove_hull_gap_reduction"]) >= 0.08
            and float(row["after_remove_regularity_score"]) >= 0.90
        )
        weak_no_remove_signal = (
            float(row["remove_regularity_score_gain"]) <= 0.005
            and float(row["remove_mrr_gain"]) <= 0.02
            and float(row["remove_hull_gap_reduction"]) <= 0.02
        )
        component_already_regular = (
            float(row["component_mrr_ratio"]) >= 0.95
            and float(row["component_hull_gap_ratio"]) <= 0.03
            and float(row["component_regularity_score"]) >= 0.96
        )
        would_remove_last_building = str(row["source_role"]).lower() == "building" and int(row["rest_building_count"]) == 0

        if source_is_land and component_irregular and strong_remove_improves_shape:
            out.at[idx, TARGET_COL] = 1
            out.at[idx, "label_source"] = "weak_shape_positive"
            out.at[idx, "sample_weight"] = 2.0
        elif component_already_regular or weak_no_remove_signal or would_remove_last_building:
            out.at[idx, TARGET_COL] = 0
            out.at[idx, "label_source"] = "weak_negative"
            out.at[idx, "sample_weight"] = 1.0

    return out


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLS | {TARGET_COL, "sample_weight"}
    feature_cols = [c for c in dataset.columns if c not in excluded]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols and pd.api.types.is_numeric_dtype(dataset[c])]
    feature_cols = numeric_cols + categorical_cols
    return feature_cols, numeric_cols, categorical_cols


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "support_positive": int(support[1]),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "f1_negative": float(f1[0]),
        "support_negative": int(support[0]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).astype(int).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the source-level pruning model for WFS merge parcels.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--qa-gpkg", default=DEFAULT_QA_GPKG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    qa_gpkg = Path(args.qa_gpkg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[INFO] Reading prediction: {input_gpkg}")
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
    sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")

    _log("[INFO] Building prune candidates")
    candidates = build_prune_candidates(predicted, sources)
    if candidates.empty:
        raise RuntimeError("No prune candidates were built.")
    dataset = assign_labels(candidates, qa_gpkg)
    labeled = dataset[dataset[TARGET_COL].notna()].copy()
    labeled[TARGET_COL] = labeled[TARGET_COL].astype(int)
    if labeled[TARGET_COL].nunique() < 2:
        raise RuntimeError("Prune training labels have only one class.")

    feature_cols, numeric_cols, categorical_cols = _feature_columns(dataset)
    _log(f"[INFO] Candidates={len(dataset):,}; labeled={len(labeled):,}; label_counts={labeled[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Label sources={labeled['label_source'].value_counts().to_dict()}")
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

    _log("[INFO] Training prune model")
    pipeline.fit(
        labeled[feature_cols],
        labeled[TARGET_COL],
        model__sample_weight=labeled["sample_weight"].astype(float).to_numpy(),
    )
    labeled_proba = pipeline.predict_proba(labeled[feature_cols])[:, 1]
    all_proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]
    dataset["prune_proba"] = all_proba
    dataset["prune_pred_at_threshold"] = dataset["prune_proba"].ge(float(args.threshold)).astype(int)

    manual = labeled[labeled["label_source"].astype(str).str.startswith("manual")].copy()
    manual_proba = pipeline.predict_proba(manual[feature_cols])[:, 1] if not manual.empty else np.asarray([])
    metrics = {
        "input_gpkg": str(input_gpkg),
        "qa_gpkg": str(qa_gpkg),
        "output_dir": str(output_dir),
        "threshold": float(args.threshold),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "excluded_columns": sorted(ID_COLS | {TARGET_COL, "sample_weight"}),
        "candidate_rows": int(len(dataset)),
        "labeled_rows": int(len(labeled)),
        "label_counts": labeled[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "label_source_counts": labeled["label_source"].value_counts().to_dict(),
        "labeled_at_threshold": _metrics(labeled[TARGET_COL].to_numpy(dtype=int), labeled_proba, float(args.threshold)),
        "manual_at_threshold": _metrics(manual[TARGET_COL].to_numpy(dtype=int), manual_proba, float(args.threshold))
        if not manual.empty and manual[TARGET_COL].nunique() > 1
        else None,
    }

    model_path = output_dir / MODEL_FILE_NAME
    candidates_path = output_dir / CANDIDATES_FILE_NAME
    predictions_path = output_dir / PREDICTIONS_FILE_NAME
    metrics_path = output_dir / "prune_metrics.json"
    joblib.dump(pipeline, model_path)
    dataset.to_csv(candidates_path, index=False)

    report_cols = [
        "component_id",
        "source_fid",
        "label",
        "label_source",
        "manual_review_class",
        "review_fid",
        "prune_proba",
        "prune_pred_at_threshold",
        "source_role",
        "source_theme",
        "source_descriptive_group",
        "source_area",
        "source_area_ratio",
        "source_shared_ratio_source_perimeter",
        "component_source_count",
        "component_mrr_ratio",
        "component_hull_gap_ratio",
        "component_regularity_score",
        "remove_mrr_gain",
        "remove_hull_gap_reduction",
        "remove_regularity_score_gain",
        "remove_notch_index_reduction",
        "after_remove_mrr_ratio",
        "after_remove_hull_gap_ratio",
        "after_remove_regularity_score",
    ]
    report_cols = [c for c in report_cols if c in dataset.columns]
    dataset[report_cols].sort_values("prune_proba", ascending=False).to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _log("[DONE] Prune model training complete")
    _log(json.dumps(metrics["labeled_at_threshold"], indent=2))
    if metrics["manual_at_threshold"] is not None:
        _log("[DONE] Manual label metrics:")
        _log(json.dumps(metrics["manual_at_threshold"], indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

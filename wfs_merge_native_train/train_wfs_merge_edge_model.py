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
from shapely.geometry import Point
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_INPUT_CSV = (
    "/data/sheffield/spatial/base-map/"
    "sheffield_wp5_wfs_merge_training_small_edges_50k_uprn_rect_semantic_clean.csv"
)
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1"


TARGET_COL = "label"
ID_COLS = {
    "left_source_fid",
    "right_source_fid",
}
SPLIT_ONLY_COLS = {
    "mid_x",
    "mid_y",
}
LEAKAGE_COLS = {
    "left_merge_fid",
    "right_merge_fid",
    "left_merge_area",
    "right_merge_area",
    "left_merge_source_count",
    "right_merge_source_count",
    "left_merge_stage",
    "right_merge_stage",
    "touches_filtered_parcel",
}
CATEGORICAL_COLS = [
    "left_theme",
    "right_theme",
    "left_role",
    "right_role",
    "left_descriptive_group",
    "right_descriptive_group",
    "left_descriptive_term",
    "right_descriptive_term",
    "left_make",
    "right_make",
    "role_pair",
]
REPORT_COLS = [
    "left_source_fid",
    "right_source_fid",
    "left_merge_fid",
    "right_merge_fid",
    "role_pair",
    "shared_edge_len",
    "shared_ratio_small_perimeter",
    "small_large_area_ratio",
    "union_mrr_ratio",
    "union_hull_gap_ratio",
    "left_area",
    "right_area",
    "left_uprn_count",
    "right_uprn_count",
    "mid_x",
    "mid_y",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_div(num: pd.Series, den: pd.Series | np.ndarray | float) -> pd.Series:
    den_s = pd.Series(den, index=num.index) if not isinstance(den, pd.Series) else den
    return num / den_s.replace(0.0, 1.0)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    left_area = out["left_area"].astype(float)
    right_area = out["right_area"].astype(float)
    left_perimeter = out["left_perimeter"].astype(float)
    right_perimeter = out["right_perimeter"].astype(float)
    shared = out["shared_edge_len"].astype(float)

    out["shared_ratio_left_perimeter"] = _safe_div(shared, left_perimeter)
    out["shared_ratio_right_perimeter"] = _safe_div(shared, right_perimeter)
    out["area_sum"] = left_area + right_area
    out["area_abs_diff"] = (left_area - right_area).abs()
    out["area_balance"] = _safe_div(out[["left_area", "right_area"]].min(axis=1), out[["left_area", "right_area"]].max(axis=1))
    out["perimeter_sum"] = left_perimeter + right_perimeter
    out["perimeter_reduction"] = out["perimeter_sum"] - out["union_perimeter"].astype(float)
    out["perimeter_reduction_ratio"] = _safe_div(out["perimeter_reduction"], out["perimeter_sum"])
    out["union_area_ratio"] = _safe_div(out["union_area"].astype(float), out["area_sum"])
    out["union_mrr_gain_from_best"] = out["union_mrr_ratio"].astype(float) - out[["left_mrr_ratio", "right_mrr_ratio"]].max(axis=1)
    out["union_hull_gap_delta_from_best"] = out["union_hull_gap_ratio"].astype(float) - out[
        ["left_hull_gap_ratio", "right_hull_gap_ratio"]
    ].min(axis=1)
    out["union_compactness_gain_from_best"] = out["union_compactness"].astype(float) - out[
        ["left_compactness", "right_compactness"]
    ].max(axis=1)
    out["shared_edge_log1p"] = np.log1p(shared.clip(lower=0.0))
    out["area_sum_log1p"] = np.log1p(out["area_sum"].clip(lower=0.0))
    out["small_area_log1p"] = np.log1p(out["small_area"].astype(float).clip(lower=0.0))
    out["large_area_log1p"] = np.log1p(out["large_area"].astype(float).clip(lower=0.0))

    left_role = out["left_role"].fillna("").astype(str).str.lower()
    right_role = out["right_role"].fillna("").astype(str).str.lower()
    left_is_building = left_role.eq("building").astype(int)
    right_is_building = right_role.eq("building").astype(int)
    left_is_land = left_role.eq("land").astype(int)
    right_is_land = right_role.eq("land").astype(int)
    out["left_is_building"] = left_is_building
    out["right_is_building"] = right_is_building
    out["left_is_land"] = left_is_land
    out["right_is_land"] = right_is_land
    out["pair_building_count"] = left_is_building + right_is_building
    out["pair_land_count"] = left_is_land + right_is_land
    out["pair_is_building_land"] = ((out["pair_building_count"] == 1) & (out["pair_land_count"] == 1)).astype(int)
    out["pair_is_building_building"] = (out["pair_building_count"] == 2).astype(int)
    out["pair_is_land_land"] = (out["pair_land_count"] == 2).astype(int)
    pair_building_area = left_area * left_is_building + right_area * right_is_building
    pair_land_area = left_area * left_is_land + right_area * right_is_land
    out["pair_building_area"] = pair_building_area
    out["pair_land_area"] = pair_land_area
    out["pair_land_area_ratio"] = _safe_div(pair_land_area, pair_building_area + pair_land_area)
    out["pair_building_area_ratio"] = _safe_div(pair_building_area, pair_building_area + pair_land_area)
    out["pair_largest_land_ratio"] = _safe_div(
        pd.concat([left_area.where(left_is_land.eq(1), 0.0), right_area.where(right_is_land.eq(1), 0.0)], axis=1).max(axis=1),
        pair_building_area + pair_land_area,
    )
    out["has_overlap"] = out["overlap_area"].astype(float).gt(1e-9).astype(int)
    return out


def _make_spatial_split(
    df: pd.DataFrame,
    *,
    cell_size: float,
    test_size: float,
    val_size: float,
    random_state: int,
) -> pd.Series:
    x_cell = np.floor(df["mid_x"].astype(float) / cell_size).astype(int)
    y_cell = np.floor(df["mid_y"].astype(float) / cell_size).astype(int)
    groups = x_cell.astype(str) + "_" + y_cell.astype(str)
    y = df[TARGET_COL].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, y, groups))

    rel_val_size = val_size / (1.0 - test_size)
    train_val = df.iloc[train_val_idx]
    train_val_groups = groups.iloc[train_val_idx]
    train_val_y = y.iloc[train_val_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state + 1)
    train_rel_idx, val_rel_idx = next(splitter.split(train_val, train_val_y, train_val_groups))

    split = pd.Series("train", index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    split.iloc[train_val_idx[val_rel_idx]] = "validation"
    split.iloc[train_val_idx[train_rel_idx]] = "train"
    return split


def _split_counts(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, rows in df.groupby("split"):
        labels = rows[TARGET_COL].value_counts().to_dict()
        out[str(split)] = {
            "rows": int(len(rows)),
            "positive": int(labels.get(1, 0)),
            "negative": int(labels.get(0, 0)),
            "groups": int(rows["spatial_group"].nunique()),
        }
    return out


def _choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
    }


def _threshold_for_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> dict[str, float] | None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    candidates = np.where(precision[:-1] >= target_precision)[0]
    if len(candidates) == 0:
        return None
    best_idx = int(candidates[np.argmax(recall[:-1][candidates])])
    return {
        "target_precision": float(target_precision),
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1.0 - 1e-6))),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "f1_negative": float(f1[0]),
        "support_negative": int(support[0]),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "support_positive": int(support[1]),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "confusion_matrix_labels_0_1": cm.astype(int).tolist(),
    }


def _write_error_gpkg(
    df: pd.DataFrame,
    output_path: Path,
    *,
    threshold: float,
    max_errors_per_layer: int,
) -> None:
    if output_path.exists():
        output_path.unlink()

    wrote_any = False
    for split in ("validation", "test"):
        rows = df[df["split"].eq(split)].copy()
        if rows.empty:
            continue
        rows["pred"] = rows["proba"].ge(threshold).astype(int)
        rows["error_margin"] = np.where(rows[TARGET_COL].eq(1), 1.0 - rows["proba"], rows["proba"])
        cases = {
            f"{split}_false_positives": rows[(rows[TARGET_COL].eq(0)) & (rows["pred"].eq(1))].sort_values(
                "proba", ascending=False
            ),
            f"{split}_false_negatives": rows[(rows[TARGET_COL].eq(1)) & (rows["pred"].eq(0))].sort_values(
                "proba", ascending=True
            ),
        }
        for layer, case in cases.items():
            if case.empty:
                continue
            case = case.head(max_errors_per_layer).copy()
            geom = [Point(float(x), float(y)) for x, y in zip(case["mid_x"], case["mid_y"])]
            gdf = gpd.GeoDataFrame(case, geometry=geom, crs="EPSG:27700")
            gdf.to_file(output_path, layer=layer, driver="GPKG", engine="pyogrio", index=False)
            wrote_any = True

    if not wrote_any:
        empty = gpd.GeoDataFrame({"message": ["no validation/test errors at selected threshold"]}, geometry=[Point(0, 0)], crs="EPSG:27700")
        empty.to_file(output_path, layer="no_errors", driver="GPKG", engine="pyogrio", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline WFS merge edge classifier.")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-size", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--importance-sample", type=int, default=8000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--max-errors-per-layer", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[INFO] Reading training CSV: {input_csv}")
    raw = pd.read_csv(input_csv)
    if TARGET_COL not in raw.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    df = _add_derived_features(raw)

    x_cell = np.floor(df["mid_x"].astype(float) / float(args.cell_size)).astype(int)
    y_cell = np.floor(df["mid_y"].astype(float) / float(args.cell_size)).astype(int)
    df["spatial_group"] = x_cell.astype(str) + "_" + y_cell.astype(str)
    df["split"] = _make_spatial_split(
        df,
        cell_size=float(args.cell_size),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        random_state=int(args.random_state),
    )

    excluded = {TARGET_COL} | ID_COLS | SPLIT_ONLY_COLS | LEAKAGE_COLS | {"split", "spatial_group"}
    feature_cols = [col for col in df.columns if col not in excluded]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in feature_cols]
    numeric_cols = [
        col
        for col in feature_cols
        if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    _log(f"[INFO] Rows={len(df):,}; features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
    _log(f"[INFO] Split counts: {json.dumps(_split_counts(df), indent=2)}")

    train = df[df["split"].eq("train")].copy()
    validation = df[df["split"].eq("validation")].copy()
    test = df[df["split"].eq("test")].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
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
        l2_regularization=float(args.l2),
        class_weight="balanced",
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=25,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    _log("[INFO] Fitting model...")
    pipeline.fit(train[feature_cols], train[TARGET_COL].astype(int))

    _log("[INFO] Predicting validation/test...")
    val_proba = pipeline.predict_proba(validation[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]

    threshold_info = _choose_threshold(validation[TARGET_COL].to_numpy(dtype=int), val_proba)
    threshold = threshold_info["threshold"]
    precision_targets = [
        _threshold_for_precision(validation[TARGET_COL].to_numpy(dtype=int), val_proba, target)
        for target in (0.80, 0.90, 0.95)
    ]
    precision_targets = [item for item in precision_targets if item is not None]

    metrics = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "model": "sklearn.HistGradientBoostingClassifier",
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "excluded_columns": sorted(excluded),
        "split_counts": _split_counts(df),
        "threshold_selected_on_validation": threshold_info,
        "validation_precision_target_thresholds": precision_targets,
        "train_at_selected_threshold": _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, threshold),
        "validation_at_selected_threshold": _metrics(validation[TARGET_COL].to_numpy(dtype=int), val_proba, threshold),
        "test_at_selected_threshold": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, threshold),
        "test_at_threshold_0_5": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, 0.5),
    }

    model_path = output_dir / "wfs_merge_edge_model_v1.joblib"
    metrics_path = output_dir / "metrics.json"
    feature_columns_path = output_dir / "feature_columns.json"
    predictions_path = output_dir / "validation_test_predictions.csv"
    error_gpkg_path = output_dir / "validation_test_errors.gpkg"
    importance_path = output_dir / "permutation_importance.csv"

    _log(f"[INFO] Saving model: {model_path}")
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_columns_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_cols,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "excluded_columns": sorted(excluded),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prediction_frames = []
    for split_name, rows, proba in (
        ("validation", validation, val_proba),
        ("test", test, test_proba),
    ):
        report_cols = [col for col in REPORT_COLS if col in rows.columns]
        out = rows[report_cols + [TARGET_COL]].copy()
        out["split"] = split_name
        out["proba"] = proba
        out["pred_selected_threshold"] = out["proba"].ge(threshold).astype(int)
        out["pred_threshold_0_5"] = out["proba"].ge(0.5).astype(int)
        out["error_type"] = np.where(
            (out[TARGET_COL].eq(0)) & (out["pred_selected_threshold"].eq(1)),
            "false_positive",
            np.where((out[TARGET_COL].eq(1)) & (out["pred_selected_threshold"].eq(0)), "false_negative", "correct"),
        )
        prediction_frames.append(out)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(predictions_path, index=False)
    _write_error_gpkg(predictions, error_gpkg_path, threshold=threshold, max_errors_per_layer=int(args.max_errors_per_layer))

    _log("[INFO] Computing permutation importance on validation sample...")
    importance_rows = validation
    if len(importance_rows) > int(args.importance_sample):
        importance_rows = importance_rows.sample(n=int(args.importance_sample), random_state=int(args.random_state))
    importance = permutation_importance(
        pipeline,
        importance_rows[feature_cols],
        importance_rows[TARGET_COL].astype(int),
        scoring="average_precision",
        n_repeats=int(args.importance_repeats),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(importance_path, index=False)

    _log("[DONE] Training complete.")
    _log(f"[DONE] selected_threshold={threshold:.6f}")
    _log(
        "[DONE] validation AP={:.4f} ROC_AUC={:.4f} F1+={:.4f}".format(
            metrics["validation_at_selected_threshold"]["average_precision"],
            metrics["validation_at_selected_threshold"]["roc_auc"],
            metrics["validation_at_selected_threshold"]["f1_positive"],
        )
    )
    _log(
        "[DONE] test AP={:.4f} ROC_AUC={:.4f} F1+={:.4f}".format(
            metrics["test_at_selected_threshold"]["average_precision"],
            metrics["test_at_selected_threshold"]["roc_auc"],
            metrics["test_at_selected_threshold"]["f1_positive"],
        )
    )
    _log(f"[DONE] outputs: {output_dir}")


if __name__ == "__main__":
    main()

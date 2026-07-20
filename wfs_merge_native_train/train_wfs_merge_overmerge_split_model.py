#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from overmerge_split_features import (
    TARGET_COL,
    build_overmerge_split_candidates,
    feature_columns,
)


DEFAULT_INPUT_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/03_operation_pruned_only.gpkg"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_overmerge_split_model_v1"
MODEL_FILE_NAME = "wfs_merge_overmerge_split_model_v1.joblib"
CANDIDATES_FILE_NAME = "overmerge_split_candidates_v1.csv"
PREDICTIONS_FILE_NAME = "overmerge_split_candidate_predictions_v1.csv"


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_edge_keys(text: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for item in str(text or "").replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        sep = "-" if "-" in item else "|"
        parts = [part.strip() for part in item.split(sep) if part.strip()]
        if len(parts) != 2:
            continue
        try:
            left, right = sorted((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
        out.add((left, right))
    return out


def _edge_key(row: pd.Series) -> tuple[int, int]:
    return tuple(sorted((int(row["left_source_fid"]), int(row["right_source_fid"]))))  # type: ignore[return-value]


def _reference_diff(row: pd.Series) -> bool:
    left = int(row.get("left_merge_fid", -1))
    right = int(row.get("right_merge_fid", -1))
    return left >= 0 and right >= 0 and left != right


def _local_split_rank_score(rows: pd.DataFrame) -> pd.Series:
    local_support = rows["local_aligned_count"].fillna(0).astype(float).clip(lower=0.0)
    side_uprn = rows["split_both_sides_have_uprn"].fillna(0).astype(int)
    return (
        3.0 * side_uprn
        + 1.4 * rows["split_mean_regularity_score"].fillna(0).astype(float)
        + 0.8 * rows["split_min_mrr_ratio"].fillna(0).astype(float)
        - 1.8 * rows["split_area_per_uprn_log_dev_mean"].fillna(2.0).astype(float)
        - 1.0 * rows["split_area_log_dev_mean"].fillna(2.0).astype(float)
        - 0.8 * rows["split_max_hull_gap_ratio"].fillna(1.0).astype(float)
        + 0.05 * np.log1p(local_support)
    )


def assign_labels(
    candidates: pd.DataFrame,
    *,
    manual_positive_edges: set[tuple[int, int]],
    manual_negative_edges: set[tuple[int, int]],
) -> pd.DataFrame:
    out = candidates.copy()
    out[TARGET_COL] = np.nan
    out["sample_weight"] = 0.0
    out["label_source"] = "unlabeled"
    out["manual_label"] = np.nan
    out["manual_reason"] = ""
    out["edge_key"] = out.apply(_edge_key, axis=1)

    reference_diff = out.apply(_reference_diff, axis=1)
    reference_same = out["edge_reference_label"].fillna(-1).astype(int).eq(1)
    usable_split = (
        out["split_both_sides_have_uprn"].fillna(0).astype(int).eq(1)
        & out["local_aligned_count"].fillna(0).astype(float).ge(6.0)
    )
    out["local_split_rank_score"] = _local_split_rank_score(out)

    out.loc[reference_same, TARGET_COL] = 0
    out.loc[reference_same, "sample_weight"] = 1.0
    out.loc[reference_same, "label_source"] = "reference_same_negative"

    reference_diff_idx = out[reference_diff].index
    out.loc[reference_diff_idx, TARGET_COL] = 0
    out.loc[reference_diff_idx, "sample_weight"] = 1.0
    out.loc[reference_diff_idx, "label_source"] = "reference_different_not_best_negative"

    for _component_id, group in out[reference_diff & usable_split].groupby("component_id", sort=True):
        best_idx = group.sort_values(
            [
                "local_split_rank_score",
                "split_area_per_uprn_log_dev_mean",
                "split_area_log_dev_mean",
                "edge_model_proba",
            ],
            ascending=[False, True, True, True],
        ).index[:1]
        out.loc[best_idx, TARGET_COL] = 1
        out.loc[best_idx, "sample_weight"] = 8.0
        out.loc[best_idx, "label_source"] = "reference_different_best_local_mode_positive"

    for idx, key in out["edge_key"].items():
        if key in manual_positive_edges:
            out.at[idx, TARGET_COL] = 1
            out.at[idx, "sample_weight"] = 80.0
            out.at[idx, "label_source"] = "manual_positive"
            out.at[idx, "manual_label"] = 1
            out.at[idx, "manual_reason"] = "user_overmerge_feedback"
        elif key in manual_negative_edges:
            out.at[idx, TARGET_COL] = 0
            out.at[idx, "sample_weight"] = 50.0
            out.at[idx, "label_source"] = "manual_negative"
            out.at[idx, "manual_label"] = 0
            out.at[idx, "manual_reason"] = "user_overmerge_feedback"

    return out.drop(columns=["edge_key"])


def _spatial_split(df: pd.DataFrame, *, cell_size: float, test_size: float, random_state: int) -> pd.Series:
    x = np.floor(df["mid_x"].astype(float) / float(cell_size)).astype(int)
    y = np.floor(df["mid_y"].astype(float) / float(cell_size)).astype(int)
    groups = x.astype(str) + "_" + y.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(random_state))
    train_idx, test_idx = next(splitter.split(df, df[TARGET_COL].astype(int), groups))
    split = pd.Series("train", index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    split.iloc[train_idx] = "train"
    return split


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= float(threshold)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "average_precision": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
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
    parser = argparse.ArgumentParser(description="Train the WFS overmerge bridge-edge split model.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--max-component-area", type=float, default=2000.0)
    parser.add_argument("--max-component-source-count", type=int, default=30)
    parser.add_argument("--min-component-uprn-count", type=int, default=2)
    parser.add_argument("--local-radius", type=float, default=100.0)
    parser.add_argument("--local-angle-tolerance", type=float, default=15.0)
    parser.add_argument("--manual-positive-edges", default="")
    parser.add_argument("--manual-negative-edges", default="")
    parser.add_argument("--cell-size", type=float, default=1000.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_input = Path(args.candidate_input_csv) if str(args.candidate_input_csv).strip() else None
    if candidate_input and candidate_input.exists():
        _log(f"[INFO] Reading candidate CSV: {candidate_input}")
        candidates = pd.read_csv(candidate_input)
    else:
        _log(f"[INFO] Reading prediction: {input_gpkg}")
        predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
        sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")
        edges = gpd.read_file(input_gpkg, layer="predicted_positive_edges", engine="pyogrio")
        _log("[INFO] Building overmerge split candidates")
        candidates_gdf = build_overmerge_split_candidates(
            predicted,
            sources,
            edges,
            max_component_area=float(args.max_component_area),
            max_component_source_count=int(args.max_component_source_count),
            min_component_uprn_count=int(args.min_component_uprn_count),
            local_radius=float(args.local_radius),
            local_angle_tolerance=float(args.local_angle_tolerance),
        )
        candidates = pd.DataFrame(candidates_gdf.drop(columns="geometry"))
        candidates.to_csv(output_dir / CANDIDATES_FILE_NAME, index=False)

    manual_positive_edges = _parse_edge_keys(args.manual_positive_edges)
    manual_negative_edges = _parse_edge_keys(args.manual_negative_edges)
    dataset = assign_labels(
        candidates,
        manual_positive_edges=manual_positive_edges,
        manual_negative_edges=manual_negative_edges,
    )
    labeled = dataset[dataset[TARGET_COL].notna()].copy()
    labeled[TARGET_COL] = labeled[TARGET_COL].astype(int)
    if labeled.empty or labeled[TARGET_COL].nunique() < 2:
        raise RuntimeError("Overmerge split training labels have fewer than two classes.")

    feature_cols, numeric_cols, categorical_cols = feature_columns(dataset)
    split = _spatial_split(
        labeled,
        cell_size=float(args.cell_size),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )
    labeled["split"] = split.to_numpy()
    train = labeled[labeled["split"].eq("train")].copy()
    test = labeled[labeled["split"].eq("test")].copy()

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
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    _log("[INFO] Training overmerge split model")
    pipeline.fit(
        train[feature_cols],
        train[TARGET_COL],
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )
    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    all_proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]
    dataset["overmerge_split_proba"] = all_proba
    dataset["overmerge_split_pred_at_threshold"] = dataset["overmerge_split_proba"].ge(float(args.threshold)).astype(int)

    manual = dataset[dataset["label_source"].astype(str).str.startswith("manual") & dataset[TARGET_COL].notna()].copy()
    manual_proba = pipeline.predict_proba(manual[feature_cols])[:, 1] if not manual.empty else np.asarray([])
    metrics = {
        "input_gpkg": str(input_gpkg),
        "output_dir": str(output_dir),
        "threshold": float(args.threshold),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "candidate_rows": int(len(dataset)),
        "labeled_rows": int(len(labeled)),
        "label_counts": labeled[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "label_source_counts": labeled["label_source"].value_counts().to_dict(),
        "train_at_threshold": _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, float(args.threshold)),
        "test_at_threshold": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(args.threshold)),
        "manual_at_threshold": _metrics(manual[TARGET_COL].to_numpy(dtype=int), manual_proba, float(args.threshold))
        if len(manual) and manual[TARGET_COL].nunique() > 1
        else None,
        "manual_positive_edges": sorted([list(edge) for edge in manual_positive_edges]),
        "manual_negative_edges": sorted([list(edge) for edge in manual_negative_edges]),
    }

    joblib.dump(pipeline, output_dir / MODEL_FILE_NAME)
    dataset.to_csv(output_dir / CANDIDATES_FILE_NAME, index=False)
    report_cols = [
        "component_id",
        "edge_fid",
        "left_source_fid",
        "right_source_fid",
        "left_merge_fid",
        "right_merge_fid",
        "edge_reference_label",
        "label",
        "label_source",
        "sample_weight",
        "overmerge_split_proba",
        "overmerge_split_pred_at_threshold",
        "edge_model_proba",
        "role_pair",
        "component_uprn_count",
        "component_area_to_local_median",
        "component_area_per_uprn_to_local_median",
        "local_aligned_count",
        "local_area_median",
        "split_both_sides_have_uprn",
        "split_area_balance",
        "split_area_log_dev_mean",
        "split_area_per_uprn_log_dev_mean",
        "split_min_regularity_score",
        "split_min_mrr_ratio",
        "split_max_hull_gap_ratio",
    ]
    report_cols = [column for column in report_cols if column in dataset.columns]
    dataset[report_cols].sort_values("overmerge_split_proba", ascending=False).to_csv(
        output_dir / PREDICTIONS_FILE_NAME,
        index=False,
    )
    (output_dir / "overmerge_split_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log("[DONE] Overmerge split model training complete")
    _log(json.dumps(metrics["test_at_threshold"], indent=2))
    if metrics["manual_at_threshold"] is not None:
        _log("[DONE] Manual label metrics:")
        _log(json.dumps(metrics["manual_at_threshold"], indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()


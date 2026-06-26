#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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

from parcel_assembly_features import (
    CATEGORICAL_FEATURES,
    build_parcel_assembly_candidates,
    feature_columns,
    log,
    parse_fid_groups,
)


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_anchor_group_repaired_threshold_085_gate_096.gpkg"
)
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_parcel_assembly_v1"
MODEL_FILE_NAME = "parcel_assembly_model_v1.joblib"
CANDIDATES_FILE_NAME = "parcel_assembly_candidates_v1.csv"
PREDICTIONS_FILE_NAME = "parcel_assembly_predictions_v1.csv"
METRICS_FILE_NAME = "parcel_assembly_metrics_v1.json"
TARGET_COL = "label"


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= float(threshold)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )
    out: dict[str, Any] = {
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
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["average_precision"] = float(average_precision_score(y_true, proba))
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, proba)
        targets: dict[str, Any] = {}
        for target in [0.90, 0.95, 0.97]:
            eligible = np.where(precision_curve[:-1] >= target)[0]
            if len(eligible) == 0:
                targets[str(target)] = None
                continue
            idx = int(eligible[np.argmax(recall_curve[:-1][eligible])])
            targets[str(target)] = {
                "threshold": float(thresholds[idx]),
                "precision": float(precision_curve[idx]),
                "recall": float(recall_curve[idx]),
            }
        out["thresholds_at_precision"] = targets
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one unified parcel assembly model over local component groups.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seed-area", type=float, default=2000.0)
    parser.add_argument("--max-after-area", type=float, default=2000.0)
    parser.add_argument("--max-pair-area", type=float, default=2000.0)
    parser.add_argument("--max-group-size", type=int, default=6)
    parser.add_argument("--top-neighbors", type=int, default=8)
    parser.add_argument("--per-seed-limit", type=int, default=24)
    parser.add_argument("--max-candidate-groups", type=int, default=250000)
    parser.add_argument("--min-shared-edge", type=float, default=0.2)
    parser.add_argument("--query-chunk-size", type=int, default=5000)
    parser.add_argument("--include-all-under-area", action="store_true")
    parser.add_argument("--manual-positive-fid-groups", default="")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_positive_groups = parse_fid_groups(args.manual_positive_fid_groups)

    candidates, edges = build_parcel_assembly_candidates(
        input_gpkg=Path(args.input_gpkg),
        max_seed_area=float(args.max_seed_area),
        max_after_area=float(args.max_after_area),
        max_pair_area=float(args.max_pair_area),
        max_group_size=int(args.max_group_size),
        top_neighbors=int(args.top_neighbors),
        per_seed_limit=int(args.per_seed_limit),
        max_candidate_groups=int(args.max_candidate_groups),
        min_shared_edge=float(args.min_shared_edge),
        query_chunk_size=int(args.query_chunk_size),
        include_all_under_area=bool(args.include_all_under_area),
        manual_positive_fid_groups=manual_positive_groups,
        include_labels=True,
    )
    if candidates.empty:
        raise RuntimeError("No parcel assembly candidates were generated.")
    dataset = pd.DataFrame(candidates.drop(columns="geometry"))
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
    feature_cols, numeric_cols, categorical_cols = feature_columns(dataset)
    log(f"[INFO] Candidates={len(dataset):,}; label_counts={dataset[TARGET_COL].value_counts().to_dict()}")
    log(f"[INFO] Label sources={dataset['label_source'].value_counts().to_dict()}")
    log(f"[INFO] Feature columns={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

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
                [column for column in CATEGORICAL_FEATURES if column in categorical_cols],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_leaf_nodes=23,
        l2_regularization=0.08,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=25,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    groups = dataset["seed_fid"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(dataset, dataset[TARGET_COL], groups=groups))
    train = dataset.iloc[train_idx].copy()
    test = dataset.iloc[test_idx].copy()

    log("[INFO] Training parcel assembly model")
    pipeline.fit(
        train[feature_cols],
        train[TARGET_COL],
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    all_proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]
    dataset["parcel_assembly_proba"] = all_proba
    dataset["parcel_assembly_pred_at_threshold"] = dataset["parcel_assembly_proba"].ge(float(args.threshold)).astype(int)

    test_metrics = _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(args.threshold))
    all_metrics = _metrics(dataset[TARGET_COL].to_numpy(dtype=int), all_proba, float(args.threshold))

    log("[INFO] Refitting final model on all candidates")
    final_pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    final_pipeline.fit(
        dataset[feature_cols],
        dataset[TARGET_COL],
        model__sample_weight=dataset["sample_weight"].astype(float).to_numpy(),
    )

    payload = {
        "pipeline": final_pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "input_gpkg": str(args.input_gpkg),
            "max_seed_area": float(args.max_seed_area),
            "max_after_area": float(args.max_after_area),
            "max_pair_area": float(args.max_pair_area),
            "max_group_size": int(args.max_group_size),
            "top_neighbors": int(args.top_neighbors),
            "per_seed_limit": int(args.per_seed_limit),
            "max_candidate_groups": int(args.max_candidate_groups),
            "min_shared_edge": float(args.min_shared_edge),
            "query_chunk_size": int(args.query_chunk_size),
            "include_all_under_area": bool(args.include_all_under_area),
            "manual_positive_fid_groups": [sorted(group) for group in manual_positive_groups],
            "threshold": float(args.threshold),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    dataset.to_csv(output_dir / CANDIDATES_FILE_NAME, index=False)
    dataset.sort_values("parcel_assembly_proba", ascending=False).head(100000).to_csv(
        output_dir / PREDICTIONS_FILE_NAME,
        index=False,
    )
    metrics = {
        "input_gpkg": str(args.input_gpkg),
        "output_dir": str(output_dir),
        "model": str(output_dir / MODEL_FILE_NAME),
        "edge_rows": int(len(edges)),
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "label_source_counts": dataset["label_source"].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log("[DONE] Parcel assembly model training complete")
    log(json.dumps(test_metrics, indent=2))
    log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

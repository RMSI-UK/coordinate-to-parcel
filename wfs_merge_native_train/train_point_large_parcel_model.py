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
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from point_large_parcel_features import CATEGORICAL_FEATURES, feature_columns, log


DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_point_large_parcel_v1"
MODEL_FILE_NAME = "point_large_parcel_model_v1.joblib"
PREDICTIONS_FILE_NAME = "point_large_parcel_candidate_predictions_v1.csv"
METRICS_FILE_NAME = "point_large_parcel_metrics_v1.json"
TARGET_COL = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prototype model for seed-point large parcel assembly.")
    parser.add_argument("--candidate-csv", action="append", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


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
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for csv_path in args.candidate_csv:
        frame = pd.read_csv(csv_path)
        frame["candidate_csv"] = str(csv_path)
        frames.append(frame)
    dataset = pd.concat(frames, ignore_index=True)
    if TARGET_COL not in dataset.columns:
        raise RuntimeError("Candidate CSV must include a label column.")
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
    feature_cols, numeric_cols, categorical_cols = feature_columns(dataset)
    label_counts = dataset[TARGET_COL].value_counts().to_dict()
    seed_count = int(dataset["seed_fid"].nunique()) if "seed_fid" in dataset.columns else 0
    log(f"[INFO] Rows={len(dataset):,}; seeds={seed_count:,}; labels={label_counts}")
    log(f"[INFO] Feature columns={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

    if len(label_counts) < 2:
        raise RuntimeError("Training requires both positive and negative candidates.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
                    ]
                ),
                [column for column in CATEGORICAL_FEATURES if column in categorical_cols],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.05,
        max_leaf_nodes=17,
        l2_regularization=0.12,
        random_state=int(args.random_state),
        early_stopping=False,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    sample_weight = dataset.get("sample_weight", pd.Series(1.0, index=dataset.index)).astype(float).to_numpy()
    holdout_metrics: dict[str, Any] | None = None
    if seed_count >= 4 and int(np.sum(dataset[TARGET_COL].to_numpy() == 1)) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
        groups = dataset["seed_fid"].astype(int).to_numpy()
        train_idx, test_idx = next(splitter.split(dataset, dataset[TARGET_COL], groups=groups))
        train = dataset.iloc[train_idx].copy()
        test = dataset.iloc[test_idx].copy()
        pipeline.fit(
            train[feature_cols],
            train[TARGET_COL],
            model__sample_weight=train.get("sample_weight", pd.Series(1.0, index=train.index)).astype(float).to_numpy(),
        )
        holdout_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
        holdout_metrics = _metrics(test[TARGET_COL].to_numpy(dtype=int), holdout_proba, float(args.threshold))
        log(f"[INFO] Holdout metrics={holdout_metrics}")
    else:
        log("[WARN] Not enough independent positive seed examples for holdout; fitting prototype on all rows.")

    pipeline.fit(dataset[feature_cols], dataset[TARGET_COL], model__sample_weight=sample_weight)
    proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]
    dataset["point_large_parcel_proba"] = proba
    dataset["point_large_parcel_pred_at_threshold"] = dataset["point_large_parcel_proba"].ge(
        float(args.threshold)
    ).astype(int)
    train_metrics = _metrics(dataset[TARGET_COL].to_numpy(dtype=int), proba, float(args.threshold))

    payload = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "candidate_csv": [str(v) for v in args.candidate_csv],
            "threshold": float(args.threshold),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    predictions_path = output_dir / PREDICTIONS_FILE_NAME
    dataset.sort_values("point_large_parcel_proba", ascending=False).to_csv(predictions_path, index=False)
    metrics = {
        "warning": None
        if holdout_metrics is not None
        else "Prototype only: fewer than 4 independent positive seed examples, so no valid holdout accuracy.",
        "train_or_all_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "seed_count": int(seed_count),
        "feature_columns": feature_cols,
        "model_path": str(output_dir / MODEL_FILE_NAME),
        "predictions_path": str(predictions_path),
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"[INFO] Wrote model {output_dir / MODEL_FILE_NAME}")
    log(f"[INFO] Wrote predictions {predictions_path}")
    log(f"[INFO] Wrote metrics {output_dir / METRICS_FILE_NAME}")


if __name__ == "__main__":
    main()

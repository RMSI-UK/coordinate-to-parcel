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

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_wfs_raw_anchor_group_model import read_candidate_inputs


DEFAULT_CANDIDATE_INPUT = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_candidate_cache_full_v1"
DEFAULT_BASE_MODEL = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_full_v1/wfs_raw_anchor_group_model_v1.joblib"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_residual_trigger_model_v1"

MODEL_FILE_NAME = "wfs_raw_anchor_residual_trigger_model_v1.joblib"
METRICS_FILE_NAME = "wfs_raw_anchor_residual_trigger_metrics_v1.json"
TARGET_COL = "needs_residual"
ID_COLUMNS = {
    "anchor_source_fid",
    "anchor_clean_fids",
    "candidate_clean_fids",
    "candidate_source_fids",
    "target_source_fids",
    "target_train_component_id",
    "label_source",
}
LABEL_DERIVED_MARKERS = ("target_", "source_target_", "label")
CATEGORICAL_FEATURES = ["anchor_role", "role_signature"]


def _log(message: str) -> None:
    print(message, flush=True)


def _threshold_at_recall(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> dict[str, Any] | None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    eligible = np.where(recall[:-1] >= float(target_recall))[0]
    if len(eligible) == 0:
        return None
    idx = int(eligible[np.argmax(precision[:-1][eligible])])
    return {"threshold": float(thresholds[idx]), "precision": float(precision[idx]), "recall": float(recall[idx])}


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = proba >= float(threshold)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred.astype(int), labels=[1, 0], zero_division=0)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "precision_needs_residual": float(precision[0]),
        "recall_needs_residual": float(recall[0]),
        "f1_needs_residual": float(f1[0]),
        "precision_clean_selected": float(precision[1]),
        "recall_clean_selected": float(recall[1]),
        "f1_clean_selected": float(f1[1]),
        "support_needs_residual": int(support[0]),
        "support_clean_selected": int(support[1]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred.astype(int), labels=[0, 1]).astype(int).tolist(),
        "threshold_at_recall_0.90": _threshold_at_recall(y_true, proba, 0.90),
        "threshold_at_recall_0.95": _threshold_at_recall(y_true, proba, 0.95),
        "threshold_at_recall_0.98": _threshold_at_recall(y_true, proba, 0.98),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLUMNS | {
        TARGET_COL,
        "label",
        "sample_weight",
        "raw_anchor_group_pred_at_threshold",
        "selected_label_source",
        "pool_exact_count",
    }
    feature_cols = [
        column
        for column in dataset.columns
        if column not in excluded and not any(str(column).startswith(marker) for marker in LABEL_DERIVED_MARKERS)
    ]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_cols]
    numeric = [column for column in feature_cols if column not in categorical and pd.api.types.is_numeric_dtype(dataset[column])]
    return numeric + categorical, numeric, categorical


def _score_candidates(dataset: pd.DataFrame, base_model_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = joblib.load(base_model_path)
    feature_cols = list(payload["feature_cols"])
    out = dataset.drop(columns=["raw_anchor_group_proba", "raw_anchor_group_pred_at_threshold"], errors="ignore").copy()
    missing = [column for column in feature_cols if column not in out.columns]
    if missing:
        out = pd.concat([out, pd.DataFrame({column: np.nan for column in missing}, index=out.index)], axis=1)
    out["raw_anchor_group_proba"] = payload["pipeline"].predict_proba(out[feature_cols])[:, 1]
    return out, {"base_model": str(base_model_path), "base_feature_count": int(len(feature_cols))}


def _selected_rows_with_pool_features(candidates: pd.DataFrame) -> pd.DataFrame:
    work = candidates.copy()
    work["anchor_source_fid"] = pd.to_numeric(work["anchor_source_fid"], errors="coerce").astype("Int64")
    work = work[work["anchor_source_fid"].notna()].copy()
    work["anchor_source_fid"] = work["anchor_source_fid"].astype("int64")
    if "label" not in work.columns:
        work["label"] = 0
    if "label_source" not in work.columns:
        work["label_source"] = ""
    work["label"] = work["label"].astype(int)
    work["raw_anchor_group_proba"] = pd.to_numeric(work["raw_anchor_group_proba"], errors="coerce").fillna(0.0)
    work = work.sort_values(["anchor_source_fid", "raw_anchor_group_proba"], ascending=[True, False])
    selected = work.groupby("anchor_source_fid", sort=False).head(1).copy()

    group = work.groupby("anchor_source_fid", sort=False)
    summary = group.agg(
        pool_rows=("raw_anchor_group_proba", "size"),
        pool_max_proba=("raw_anchor_group_proba", "max"),
        pool_second_proba=("raw_anchor_group_proba", lambda values: float(values.nlargest(2).iloc[-1]) if len(values) >= 2 else 0.0),
        pool_max_source_count=("candidate_source_count", "max"),
        pool_max_clean_count=("candidate_clean_count", "max"),
        pool_max_area=("candidate_area", "max"),
        pool_max_regularity=("group_regularity_score", "max"),
        pool_min_hull_gap=("group_hull_gap_ratio", "min"),
    )
    selected = selected.join(summary, on="anchor_source_fid", rsuffix="_pool")
    selected["selected_label_source"] = selected["label_source"].astype(str)
    selected["needs_residual"] = selected["label"].astype(int).eq(0).astype(int)
    selected["selected_proba_gap_to_second"] = selected["raw_anchor_group_proba"] - selected["pool_second_proba"]
    selected["selected_source_count_gap_to_pool_max"] = selected["pool_max_source_count"] - selected["candidate_source_count"]
    selected["selected_clean_count_gap_to_pool_max"] = selected["pool_max_clean_count"] - selected["candidate_clean_count"]
    selected["selected_area_to_pool_max"] = selected["candidate_area"] / selected["pool_max_area"].replace(0.0, np.nan)
    selected["selected_area_to_pool_max"] = selected["selected_area_to_pool_max"].fillna(0.0)
    selected["selected_regularity_gap_to_pool_max"] = selected["pool_max_regularity"] - selected["group_regularity_score"]
    selected["selected_hull_gap_delta_to_pool_min"] = selected["group_hull_gap_ratio"] - selected["pool_min_hull_gap"]

    for add in [1, 2, 3]:
        larger = work[work["candidate_source_count"].astype(float).gt(0)].copy()
        selected_count = selected.set_index("anchor_source_fid")["candidate_source_count"].to_dict()
        larger["selected_source_count"] = larger["anchor_source_fid"].map(selected_count)
        larger = larger[larger["candidate_source_count"].eq(larger["selected_source_count"] + add)].copy()
        top = larger.sort_values(["anchor_source_fid", "raw_anchor_group_proba"], ascending=[True, False]).groupby("anchor_source_fid").head(1)
        top = top.set_index("anchor_source_fid")
        selected[f"larger_plus_{add}_exists"] = selected["anchor_source_fid"].isin(top.index).astype(int)
        selected[f"larger_plus_{add}_proba"] = selected["anchor_source_fid"].map(top["raw_anchor_group_proba"]).fillna(0.0)
        selected[f"larger_plus_{add}_proba_delta"] = selected["raw_anchor_group_proba"] - selected[f"larger_plus_{add}_proba"]
        selected[f"larger_plus_{add}_regularity"] = selected["anchor_source_fid"].map(top["group_regularity_score"]).fillna(0.0)
        selected[f"larger_plus_{add}_hull_gap"] = selected["anchor_source_fid"].map(top["group_hull_gap_ratio"]).fillna(0.0)
        selected[f"larger_plus_{add}_area_ratio"] = (
            selected["anchor_source_fid"].map(top["candidate_area"]).fillna(0.0) / selected["candidate_area"].replace(0.0, np.nan)
        ).fillna(0.0)

    return selected.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual trigger model for raw-anchor selected candidates.")
    parser.add_argument("--candidate-input-csv", default=DEFAULT_CANDIDATE_INPUT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-clean-train-rows", type=int, default=8000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-selected-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_candidate_inputs(str(args.candidate_input_csv))
    candidates, score_meta = _score_candidates(candidates, Path(args.base_model))
    dataset = _selected_rows_with_pool_features(candidates)

    positives = dataset[dataset[TARGET_COL].eq(1)]
    negatives = dataset[dataset[TARGET_COL].eq(0)]
    if int(args.max_clean_train_rows) > 0 and len(negatives) > int(args.max_clean_train_rows):
        negatives_fit = negatives.sample(n=int(args.max_clean_train_rows), random_state=int(args.random_state))
    else:
        negatives_fit = negatives
    fit_dataset = pd.concat([positives, negatives_fit], ignore_index=True).sample(frac=1.0, random_state=int(args.random_state))
    fit_dataset["sample_weight"] = 1.0
    fit_dataset.loc[fit_dataset[TARGET_COL].eq(1), "sample_weight"] = max(float(len(negatives_fit)) / max(float(len(positives)), 1.0), 1.0)

    feature_cols, numeric_cols, categorical_cols = _feature_columns(fit_dataset)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(fit_dataset, fit_dataset[TARGET_COL], groups=fit_dataset["anchor_source_fid"].astype(int)))
    train = fit_dataset.iloc[train_idx].copy()
    test = fit_dataset.iloc[test_idx].copy()

    _log(f"[INFO] Selected dataset rows={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Fit rows={len(fit_dataset):,}; labels={fit_dataset[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=4))]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train[feature_cols], train[TARGET_COL], model__sample_weight=train["sample_weight"].to_numpy(dtype="float64"))
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    train_thr = (_threshold_at_recall(train[TARGET_COL].to_numpy(dtype=int), train_proba, 0.95) or {}).get("threshold", 0.5)
    metrics = {
        "model_kind": "wfs_raw_anchor_residual_trigger",
        "candidate_input_csv": str(args.candidate_input_csv),
        "score_meta": score_meta,
        "dataset_rows": int(len(dataset)),
        "dataset_label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "fit_rows": int(len(fit_dataset)),
        "fit_label_counts": fit_dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "threshold_95_recall_from_train": float(train_thr),
        "train_metrics": _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, float(train_thr)),
        "test_metrics": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(train_thr)),
    }

    payload = {
        "model_kind": "wfs_raw_anchor_residual_trigger",
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "candidate_input_csv": str(args.candidate_input_csv),
            "base_model": str(args.base_model),
            "threshold_95_recall_from_train": float(train_thr),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not bool(args.skip_selected_output):
        dataset.to_csv(output_dir / "wfs_raw_anchor_residual_trigger_selected_rows_v1.csv", index=False)
    _log("[DONE] Residual trigger training complete")
    _log(json.dumps(metrics["test_metrics"], indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

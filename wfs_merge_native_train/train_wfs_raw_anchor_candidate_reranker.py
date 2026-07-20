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

from train_wfs_raw_anchor_group_model import _parse_id_set, read_candidate_inputs


DEFAULT_CANDIDATE_INPUT = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_candidate_cache_full_v1"
DEFAULT_BASE_MODEL = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_full_v1/wfs_raw_anchor_group_model_v1.joblib"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_reranker_v1"

MODEL_FILE_NAME = "wfs_raw_anchor_candidate_reranker_v1.joblib"
METRICS_FILE_NAME = "wfs_raw_anchor_candidate_reranker_metrics_v1.json"
TARGET_COL = "label"
LABEL_DERIVED_MARKERS = ("target_", "source_target_", "label")
CATEGORICAL_FEATURES = ["anchor_role", "role_signature"]
ID_COLUMNS = {
    "anchor_source_fid",
    "anchor_clean_fids",
    "candidate_clean_fids",
    "candidate_source_fids",
    "target_source_fids",
    "target_train_component_id",
    "label_source",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _score_candidates(dataset: pd.DataFrame, base_model_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = joblib.load(base_model_path)
    feature_cols = list(payload["feature_cols"])
    out = dataset.drop(columns=["raw_anchor_group_proba", "raw_anchor_group_pred_at_threshold"], errors="ignore").copy()
    missing = [column for column in feature_cols if column not in out.columns]
    if missing:
        out = pd.concat([out, pd.DataFrame({column: np.nan for column in missing}, index=out.index)], axis=1)
    out["raw_anchor_group_proba"] = payload["pipeline"].predict_proba(out[feature_cols])[:, 1]
    return out, {"base_model": str(base_model_path), "base_feature_count": int(len(feature_cols))}


def add_rerank_pool_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["anchor_source_fid"] = pd.to_numeric(out["anchor_source_fid"], errors="coerce")
    out = out[out["anchor_source_fid"].notna()].copy()
    out["anchor_source_fid"] = out["anchor_source_fid"].astype("int64")
    out["raw_anchor_group_proba"] = pd.to_numeric(out["raw_anchor_group_proba"], errors="coerce").fillna(0.0)
    group_key = out["anchor_source_fid"]

    desc_cols = [
        "raw_anchor_group_proba",
        "candidate_source_count",
        "candidate_clean_count",
        "candidate_area",
        "internal_shared_len",
        "anchor_added_shared_len",
        "boundary_simplification",
        "group_regularity_score",
        "regularity_gain_vs_anchor",
        "hull_gap_reduction_vs_anchor",
        "shared_to_external_shared",
    ]
    asc_cols = ["group_hull_gap_ratio", "group_notch_index", "outside_uprn_neighbor_count"]
    for column in desc_cols:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        max_values = values.groupby(group_key).transform("max").replace(0.0, np.nan)
        top_values = values.groupby(group_key).transform("max")
        out[f"rerank_{column}_to_pool_max"] = (values / max_values).fillna(0.0)
        out[f"rerank_{column}_delta_pool_max"] = values - top_values
        out[f"rerank_{column}_rank_desc"] = values.groupby(group_key).rank(method="average", ascending=False)
    for column in asc_cols:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        min_values = values.groupby(group_key).transform("min")
        out[f"rerank_{column}_delta_pool_min"] = values - min_values
        out[f"rerank_{column}_rank_asc"] = values.groupby(group_key).rank(method="average", ascending=True)

    sorted_pool = out.sort_values(["anchor_source_fid", "raw_anchor_group_proba"], ascending=[True, False])
    top = sorted_pool.groupby("anchor_source_fid", sort=False).head(1).set_index("anchor_source_fid")
    for column in [
        "candidate_source_count",
        "candidate_clean_count",
        "candidate_area",
        "group_regularity_score",
        "group_hull_gap_ratio",
        "raw_anchor_group_proba",
    ]:
        if column not in out.columns:
            continue
        top_values = out["anchor_source_fid"].map(top[column]).astype(float)
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        out[f"rerank_{column}_delta_top_base"] = values - top_values

    top_source_text = out["anchor_source_fid"].map(top["candidate_source_fids"]).fillna("").astype(str)
    top_clean_text = out["anchor_source_fid"].map(top["candidate_clean_fids"]).fillna("").astype(str)
    candidate_source_text = out["candidate_source_fids"].fillna("").astype(str)
    candidate_clean_text = out["candidate_clean_fids"].fillna("").astype(str)

    source_features: list[dict[str, float]] = []
    for candidate_text, base_text in zip(candidate_source_text, top_source_text):
        candidate_set = _parse_id_set(candidate_text)
        base_set = _parse_id_set(base_text)
        inter = len(candidate_set & base_set)
        union = len(candidate_set | base_set)
        source_features.append(
            {
                "rerank_source_contains_top": float(base_set.issubset(candidate_set)) if base_set else 0.0,
                "rerank_source_is_subset_of_top": float(candidate_set.issubset(base_set)) if candidate_set else 0.0,
                "rerank_source_extra_vs_top": float(len(candidate_set - base_set)),
                "rerank_source_missing_vs_top": float(len(base_set - candidate_set)),
                "rerank_source_jaccard_to_top": float(inter) / float(union or 1),
            }
        )
    clean_features: list[dict[str, float]] = []
    for candidate_text, base_text in zip(candidate_clean_text, top_clean_text):
        candidate_set = _parse_id_set(candidate_text)
        base_set = _parse_id_set(base_text)
        inter = len(candidate_set & base_set)
        union = len(candidate_set | base_set)
        clean_features.append(
            {
                "rerank_clean_contains_top": float(base_set.issubset(candidate_set)) if base_set else 0.0,
                "rerank_clean_is_subset_of_top": float(candidate_set.issubset(base_set)) if candidate_set else 0.0,
                "rerank_clean_extra_vs_top": float(len(candidate_set - base_set)),
                "rerank_clean_missing_vs_top": float(len(base_set - candidate_set)),
                "rerank_clean_jaccard_to_top": float(inter) / float(union or 1),
            }
        )
    out = pd.concat(
        [
            out.reset_index(drop=True),
            pd.DataFrame(source_features, index=out.index).reset_index(drop=True),
            pd.DataFrame(clean_features, index=out.index).reset_index(drop=True),
        ],
        axis=1,
    )
    return out


def _build_training_rows(candidates: pd.DataFrame, *, top_k: int, include_all_positives: bool) -> pd.DataFrame:
    work = candidates.copy()
    work[TARGET_COL] = work[TARGET_COL].astype(int)
    work = add_rerank_pool_features(work)
    work = work.sort_values(["anchor_source_fid", "raw_anchor_group_proba"], ascending=[True, False])
    top = work.groupby("anchor_source_fid", sort=False).head(int(top_k)).copy()
    if include_all_positives:
        positives = work[work[TARGET_COL].eq(1)].copy()
        rows = pd.concat([top, positives], ignore_index=True)
        rows = rows.drop_duplicates(["anchor_source_fid", "candidate_clean_fids"], keep="first")
    else:
        rows = top
    return rows.reset_index(drop=True)


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLUMNS | {TARGET_COL, "sample_weight", "raw_anchor_rerank_proba"}
    feature_cols = [
        column
        for column in dataset.columns
        if column not in excluded and not any(str(column).startswith(marker) for marker in LABEL_DERIVED_MARKERS)
    ]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_cols]
    numeric = [column for column in feature_cols if column not in categorical and pd.api.types.is_numeric_dtype(dataset[column])]
    return numeric + categorical, numeric, categorical


def _threshold_at_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> dict[str, Any] | None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    eligible = np.where(precision[:-1] >= float(target_precision))[0]
    if len(eligible) == 0:
        return None
    idx = int(eligible[np.argmax(recall[:-1][eligible])])
    return {"threshold": float(thresholds[idx]), "precision": float(precision[idx]), "recall": float(recall[idx])}


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = proba >= float(threshold)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred.astype(int), labels=[1, 0], zero_division=0)
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
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred.astype(int), labels=[0, 1]).astype(int).tolist(),
        "threshold_at_precision_0.95": _threshold_at_precision(y_true, proba, 0.95),
        "threshold_at_precision_0.97": _threshold_at_precision(y_true, proba, 0.97),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a second-stage candidate reranker for raw-anchor WFS merge.")
    parser.add_argument("--candidate-input-csv", default=DEFAULT_CANDIDATE_INPUT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-all-positives", action="store_true")
    parser.add_argument("--max-negative-train-rows", type=int, default=180000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-predictions-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_candidate_inputs(str(args.candidate_input_csv))
    candidates, score_meta = _score_candidates(candidates, Path(args.base_model))
    dataset = _build_training_rows(candidates, top_k=int(args.top_k), include_all_positives=bool(args.include_all_positives))
    dataset["sample_weight"] = 1.0
    positives = dataset[dataset[TARGET_COL].astype(int).eq(1)]
    negatives = dataset[dataset[TARGET_COL].astype(int).eq(0)]
    if int(args.max_negative_train_rows) > 0 and len(negatives) > int(args.max_negative_train_rows):
        negatives = negatives.sample(n=int(args.max_negative_train_rows), random_state=int(args.random_state))
    fit_dataset = pd.concat([positives, negatives], ignore_index=True).sample(frac=1.0, random_state=int(args.random_state))

    feature_cols, numeric_cols, categorical_cols = _feature_columns(dataset)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(fit_dataset, fit_dataset[TARGET_COL], groups=fit_dataset["anchor_source_fid"].astype(int)))
    train = fit_dataset.iloc[train_idx].copy()
    test = fit_dataset.iloc[test_idx].copy()
    _log(f"[INFO] Rerank dataset rows={len(dataset):,}; labels={dataset[TARGET_COL].value_counts().to_dict()}")
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
        max_iter=320,
        learning_rate=0.035,
        max_leaf_nodes=23,
        l2_regularization=0.05,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=25,
        verbose=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(train[feature_cols], train[TARGET_COL].astype(int), model__sample_weight=train["sample_weight"].to_numpy(dtype="float64"))
    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    train_p95 = _threshold_at_precision(train[TARGET_COL].to_numpy(dtype=int), train_proba, 0.95) or {}
    threshold = float(train_p95.get("threshold", 0.5))

    payload = {
        "model_kind": "wfs_raw_anchor_candidate_reranker",
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "candidate_input_csv": str(args.candidate_input_csv),
            "base_model": str(args.base_model),
            "top_k": int(args.top_k),
            "include_all_positives": bool(args.include_all_positives),
            "threshold_95p_from_train": float(threshold),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    metrics = {
        "model_kind": payload["model_kind"],
        "model": str(output_dir / MODEL_FILE_NAME),
        "score_meta": score_meta,
        "candidate_rows": int(len(candidates)),
        "dataset_rows": int(len(dataset)),
        "dataset_label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "fit_rows": int(len(fit_dataset)),
        "fit_label_counts": fit_dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "threshold_95p_from_train": float(threshold),
        "train_metrics": _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, float(threshold)),
        "test_metrics": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(threshold)),
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not bool(args.skip_predictions_output):
        report = test[["anchor_source_fid", "candidate_clean_fids", "candidate_source_fids", "label", "label_source", "raw_anchor_group_proba"]].copy()
        report["raw_anchor_rerank_proba"] = test_proba
        report.sort_values("raw_anchor_rerank_proba", ascending=False).to_csv(output_dir / "wfs_raw_anchor_candidate_reranker_test_predictions_v1.csv", index=False)
    _log("[DONE] Candidate reranker training complete")
    _log(json.dumps(metrics["test_metrics"], indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

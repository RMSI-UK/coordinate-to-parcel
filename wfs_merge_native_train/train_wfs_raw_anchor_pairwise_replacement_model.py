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
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_pairwise_replacement_model_v1"

MODEL_FILE_NAME = "wfs_raw_anchor_pairwise_replacement_model_v1.joblib"
METRICS_FILE_NAME = "wfs_raw_anchor_pairwise_replacement_metrics_v1.json"
TARGET_COL = "replace_top"
LABEL_DERIVED_MARKERS = ("target_", "source_target_", "label", "match_")
ID_COLUMNS = {
    "anchor_source_fid",
    "anchor_clean_fids",
    "candidate_clean_fids",
    "candidate_source_fids",
    "target_source_fids",
    "target_train_component_id",
    "label_source",
    "top_candidate_clean_fids",
    "top_candidate_source_fids",
    "candidate_label_source",
    "top_label_source",
}
CATEGORICAL_FEATURES = ["anchor_role", "role_signature", "top_role_signature"]
TOP_CONTEXT_COLUMNS = [
    "raw_anchor_group_proba",
    "candidate_source_count",
    "candidate_clean_count",
    "candidate_area",
    "internal_shared_len",
    "anchor_added_shared_len",
    "boundary_simplification",
    "group_regularity_score",
    "group_hull_gap_ratio",
    "group_notch_index",
    "outside_uprn_neighbor_count",
    "shared_to_external_shared",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _score_candidates(dataset: pd.DataFrame, base_model_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = joblib.load(base_model_path)
    feature_cols = list(payload["feature_cols"])
    out = dataset.drop(columns=["raw_anchor_group_proba", "raw_anchor_group_pred_at_threshold"], errors="ignore").copy()
    missing = [column for column in feature_cols if column not in out.columns]
    if missing:
        out = pd.concat([out, pd.DataFrame({column: np.nan for column in missing}, index=out.index)], axis=1)
    proba_parts: list[np.ndarray] = []
    chunk_size = 100_000
    for start in range(0, len(out), chunk_size):
        end = min(start + chunk_size, len(out))
        proba_parts.append(payload["pipeline"].predict_proba(out.iloc[start:end][feature_cols])[:, 1])
        if start == 0 or end == len(out) or end % 500_000 < chunk_size:
            _log(f"[INFO] Scored base candidates {end:,}/{len(out):,}")
    out["raw_anchor_group_proba"] = np.concatenate(proba_parts) if proba_parts else np.array([], dtype="float64")
    return out, {"base_model": str(base_model_path), "base_feature_count": int(len(feature_cols))}


def _set_relation_features(candidate_text: object, top_text: object, prefix: str) -> dict[str, float]:
    candidate_set = _parse_id_set(candidate_text)
    top_set = _parse_id_set(top_text)
    inter = len(candidate_set & top_set)
    union = len(candidate_set | top_set)
    return {
        f"{prefix}_contains_top": float(top_set.issubset(candidate_set)) if top_set else 0.0,
        f"{prefix}_is_subset_of_top": float(candidate_set.issubset(top_set)) if candidate_set else 0.0,
        f"{prefix}_extra_vs_top": float(len(candidate_set - top_set)),
        f"{prefix}_missing_vs_top": float(len(top_set - candidate_set)),
        f"{prefix}_jaccard_to_top": float(inter) / float(union or 1),
    }


def _safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = ID_COLUMNS | {TARGET_COL, "label", "sample_weight", "raw_anchor_pair_replace_proba"}
    return [
        column
        for column in frame.columns
        if column not in excluded and not any(str(column).startswith(marker) for marker in LABEL_DERIVED_MARKERS)
    ]


def _add_set_relation_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    source_rel = [
        _set_relation_features(candidate_text, top_text, "pair_source")
        for candidate_text, top_text in zip(frame["candidate_source_fids"], frame["top_candidate_source_fids"], strict=False)
    ]
    clean_rel = [
        _set_relation_features(candidate_text, top_text, "pair_clean")
        for candidate_text, top_text in zip(frame["candidate_clean_fids"], frame["top_candidate_clean_fids"], strict=False)
    ]
    return pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame(source_rel).reset_index(drop=True),
            pd.DataFrame(clean_rel).reset_index(drop=True),
        ],
        axis=1,
    )


def build_pairwise_rows(candidates: pd.DataFrame, *, top_k: int, include_all_positives: bool, add_set_relations: bool = True) -> pd.DataFrame:
    work = candidates.copy()
    work["anchor_source_fid"] = pd.to_numeric(work["anchor_source_fid"], errors="coerce")
    work = work[work["anchor_source_fid"].notna()].copy()
    work["anchor_source_fid"] = work["anchor_source_fid"].astype("int64")
    work["label"] = work["label"].astype(int)
    work["raw_anchor_group_proba"] = pd.to_numeric(work["raw_anchor_group_proba"], errors="coerce").fillna(0.0)
    work = work.sort_values(["anchor_source_fid", "raw_anchor_group_proba"], ascending=[True, False])
    work["_candidate_rank"] = work.groupby("anchor_source_fid", sort=False).cumcount()

    safe_cols = _safe_feature_columns(work)
    numeric_safe = [column for column in safe_cols if pd.api.types.is_numeric_dtype(work[column])]
    categorical_safe = [column for column in ["anchor_role", "role_signature"] if column in safe_cols]

    pool_mask = work["_candidate_rank"].lt(int(top_k))
    if include_all_positives:
        pool_mask = pool_mask | work["label"].eq(1)
    pool = work[pool_mask & work["_candidate_rank"].ne(0)].copy()
    pool = pool.drop_duplicates(["anchor_source_fid", "candidate_clean_fids"], keep="first")

    top_columns = ["anchor_source_fid", "candidate_clean_fids", "candidate_source_fids", "label", "label_source"]
    top_columns.extend([column for column in TOP_CONTEXT_COLUMNS if column in work.columns])
    if "role_signature" in work.columns:
        top_columns.append("role_signature")
    top = work[work["_candidate_rank"].eq(0)][top_columns].copy()
    top = top.rename(
        columns={
            "candidate_clean_fids": "top_candidate_clean_fids",
            "candidate_source_fids": "top_candidate_source_fids",
            "label": "_top_label",
            "label_source": "top_label_source",
            "role_signature": "top_role_signature",
            **{column: f"_top_{column}" for column in TOP_CONTEXT_COLUMNS if column in work.columns},
        }
    )
    out = pool.merge(top, on="anchor_source_fid", how="left", validate="many_to_one")
    out["candidate_label_source"] = out.get("label_source", "").astype(str)
    out[TARGET_COL] = (out["_top_label"].fillna(0).astype(int).eq(0) & out["label"].astype(int).eq(1)).astype(int)

    keep_cols = [
        "anchor_source_fid",
        "candidate_clean_fids",
        "candidate_source_fids",
        "top_candidate_clean_fids",
        "top_candidate_source_fids",
        "candidate_label_source",
        "top_label_source",
        TARGET_COL,
    ]
    keep_cols.extend(numeric_safe)
    keep_cols.extend(categorical_safe)
    if "top_role_signature" in out.columns:
        keep_cols.append("top_role_signature")
    keep_cols = [column for column in dict.fromkeys(keep_cols) if column in out.columns]
    out = out[keep_cols + [column for column in out.columns if str(column).startswith("_top_")]].copy()

    for column in TOP_CONTEXT_COLUMNS:
        top_column = f"_top_{column}"
        if column not in out.columns or top_column not in out.columns:
            continue
        candidate_value = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        top_value = pd.to_numeric(out[top_column], errors="coerce").fillna(0.0)
        out[f"top_{column}"] = top_value
        out[f"pair_delta_{column}"] = candidate_value - top_value
        out[f"pair_ratio_{column}_to_top"] = candidate_value / top_value.where(top_value.ne(0), 1.0)
    out = out.drop(columns=[column for column in out.columns if str(column).startswith("_top_")], errors="ignore")
    if add_set_relations:
        out = _add_set_relation_features(out)
    _log(f"[INFO] Pairwise rows={len(out):,}; labels={out[TARGET_COL].value_counts().to_dict() if not out.empty else {}}")
    return out


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_cols = _safe_feature_columns(dataset)
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
        "precision_replace": float(precision[0]),
        "recall_replace": float(recall[0]),
        "f1_replace": float(f1[0]),
        "precision_keep": float(precision[1]),
        "recall_keep": float(recall[1]),
        "f1_keep": float(f1[1]),
        "support_replace": int(support[0]),
        "support_keep": int(support[1]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred.astype(int), labels=[0, 1]).astype(int).tolist(),
        "threshold_at_precision_0.50": _threshold_at_precision(y_true, proba, 0.50),
        "threshold_at_precision_0.80": _threshold_at_precision(y_true, proba, 0.80),
        "threshold_at_recall_0.80": _threshold_at_recall(y_true, proba, 0.80),
        "threshold_at_recall_0.95": _threshold_at_recall(y_true, proba, 0.95),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pairwise replacement model for candidate-vs-current-top selection.")
    parser.add_argument("--candidate-input-csv", default=DEFAULT_CANDIDATE_INPUT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--include-all-positives", action="store_true")
    parser.add_argument("--max-negative-train-rows", type=int, default=160000)
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
    dataset = build_pairwise_rows(
        candidates,
        top_k=int(args.top_k),
        include_all_positives=bool(args.include_all_positives),
        add_set_relations=False,
    )
    dataset["sample_weight"] = 1.0
    positives = dataset[dataset[TARGET_COL].astype(int).eq(1)]
    negatives = dataset[dataset[TARGET_COL].astype(int).eq(0)]
    if int(args.max_negative_train_rows) > 0 and len(negatives) > int(args.max_negative_train_rows):
        negatives = negatives.sample(n=int(args.max_negative_train_rows), random_state=int(args.random_state))
    fit_dataset = pd.concat([positives, negatives], ignore_index=True).sample(frac=1.0, random_state=int(args.random_state))
    fit_dataset = _add_set_relation_features(fit_dataset)
    fit_dataset.loc[fit_dataset[TARGET_COL].eq(1), "sample_weight"] = max(float(len(negatives)) / max(float(len(positives)), 1.0), 1.0)

    feature_cols, numeric_cols, categorical_cols = _feature_columns(fit_dataset)
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(fit_dataset, fit_dataset[TARGET_COL], groups=fit_dataset["anchor_source_fid"].astype(int)))
    train = fit_dataset.iloc[train_idx].copy()
    test = fit_dataset.iloc[test_idx].copy()
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
    pipeline.fit(train[feature_cols], train[TARGET_COL].astype(int), model__sample_weight=train["sample_weight"].to_numpy(dtype="float64"))
    train_proba = pipeline.predict_proba(train[feature_cols])[:, 1]
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    train_threshold = (_threshold_at_precision(train[TARGET_COL].to_numpy(dtype=int), train_proba, 0.50) or {}).get("threshold", 0.5)

    payload = {
        "model_kind": "wfs_raw_anchor_pairwise_replacement",
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "candidate_input_csv": str(args.candidate_input_csv),
            "base_model": str(args.base_model),
            "top_k": int(args.top_k),
            "include_all_positives": bool(args.include_all_positives),
            "threshold_50p_from_train": float(train_threshold),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    metrics = {
        "model_kind": payload["model_kind"],
        "model": str(output_dir / MODEL_FILE_NAME),
        "score_meta": score_meta,
        "dataset_rows": int(len(dataset)),
        "dataset_label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "fit_rows": int(len(fit_dataset)),
        "fit_label_counts": fit_dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "threshold_50p_from_train": float(train_threshold),
        "train_metrics": _metrics(train[TARGET_COL].to_numpy(dtype=int), train_proba, float(train_threshold)),
        "test_metrics": _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(train_threshold)),
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not bool(args.skip_predictions_output):
        report = test[["anchor_source_fid", "candidate_source_fids", "top_candidate_source_fids", TARGET_COL, "candidate_label_source", "top_label_source"]].copy()
        report["raw_anchor_pair_replace_proba"] = test_proba
        report.sort_values("raw_anchor_pair_replace_proba", ascending=False).to_csv(output_dir / "wfs_raw_anchor_pairwise_replacement_test_predictions_v1.csv", index=False)
    _log("[DONE] Pairwise replacement training complete")
    _log(json.dumps(metrics["test_metrics"], indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

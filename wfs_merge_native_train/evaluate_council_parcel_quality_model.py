#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


TARGET_COL = "label"
DEFAULT_CANDIDATE_CSV = "/data/sheffield/spatial/base-map/tmp/council_parcel_quality_candidates_v7_base_delta.csv"
DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/council_parcel_quality_model_v7_base_delta_hgb800/"
    "council_parcel_quality_model_v1.joblib"
)
DEFAULT_OUTPUT_JSON = (
    "/data/sheffield/spatial/base-map/tmp/council_parcel_quality_model_v7_base_delta_hgb800/"
    "council_parcel_quality_independent_eval.json"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _threshold_at_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> dict[str, Any] | None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    eligible = np.where(precision[:-1] >= float(target_precision))[0]
    if len(eligible) == 0:
        return None
    idx = int(eligible[np.argmax(recall[:-1][eligible])])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


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
        "recall_at_min_precision": _threshold_at_precision(y_true, proba, float(min_precision)),
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
        "threshold_at_precision_0.95": _threshold_at_precision(y_true, proba, 0.95),
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


def _slice_metrics(rows: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    work = rows.copy()
    work["_proba"] = proba
    work["_pred"] = (proba >= float(threshold)).astype(int)
    for neg_type, group in work.groupby("negative_type", dropna=False):
        label_values = set(int(v) for v in group[TARGET_COL].dropna().astype(int).unique())
        if label_values == {1}:
            out[str(neg_type)] = {
                "rows": int(len(group)),
                "label": 1,
                "recall": float(group["_pred"].eq(1).mean()),
                "proba_mean": float(group["_proba"].mean()),
                "proba_p10": float(group["_proba"].quantile(0.10)),
                "proba_p50": float(group["_proba"].quantile(0.50)),
                "proba_p90": float(group["_proba"].quantile(0.90)),
            }
        elif label_values == {0}:
            out[str(neg_type)] = {
                "rows": int(len(group)),
                "label": 0,
                "false_positive_rate": float(group["_pred"].eq(1).mean()),
                "false_positive_rows": int(group["_pred"].eq(1).sum()),
                "proba_mean": float(group["_proba"].mean()),
                "proba_p10": float(group["_proba"].quantile(0.10)),
                "proba_p50": float(group["_proba"].quantile(0.50)),
                "proba_p90": float(group["_proba"].quantile(0.90)),
            }
    return out


def _univariate_feature_audit(rows: pd.DataFrame, feature_cols: list[str], *, limit: int) -> list[dict[str, Any]]:
    y = rows[TARGET_COL].astype(int).to_numpy()
    results: list[dict[str, Any]] = []
    for column in feature_cols:
        if column not in rows.columns or not pd.api.types.is_numeric_dtype(rows[column]):
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        if values.notna().sum() < 20 or values.nunique(dropna=True) < 2:
            continue
        filled = values.fillna(values.median()).to_numpy(dtype="float64")
        try:
            auc = float(roc_auc_score(y, filled))
        except ValueError:
            continue
        strength = max(auc, 1.0 - auc)
        pos = values[rows[TARGET_COL].astype(int).eq(1)]
        neg = values[rows[TARGET_COL].astype(int).eq(0)]
        results.append(
            {
                "feature": str(column),
                "univariate_auc_or_inverse": float(strength),
                "auc_direction_positive_high": bool(auc >= 0.5),
                "positive_median": float(pos.median()) if pos.notna().any() else None,
                "negative_median": float(neg.median()) if neg.notna().any() else None,
                "positive_mean": float(pos.mean()) if pos.notna().any() else None,
                "negative_mean": float(neg.mean()) if neg.notna().any() else None,
            }
        )
    return sorted(results, key=lambda row: -float(row["univariate_auc_or_inverse"]))[: int(limit)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently evaluate a council parcel quality model.")
    parser.add_argument("--candidate-csv", default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--feature-audit-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"[INFO] Reading candidates: {args.candidate_csv}")
    candidates = pd.read_csv(args.candidate_csv, low_memory=False)
    payload = joblib.load(args.model)
    if not isinstance(payload, dict):
        raise RuntimeError("Model payload must be a dictionary.")
    pipeline = payload["pipeline"]
    feature_cols = list(payload["feature_cols"])
    missing = [column for column in feature_cols if column not in candidates.columns]
    if missing:
        raise RuntimeError(f"Candidate CSV missing model features: {missing[:20]}")

    if str(args.split).strip():
        rows = candidates[candidates["split"].astype(str).eq(str(args.split))].copy()
    else:
        rows = candidates.copy()
    if rows.empty:
        raise RuntimeError(f"No rows matched split={args.split!r}")
    y_true = rows[TARGET_COL].astype(int).to_numpy()
    proba = pipeline.predict_proba(rows[feature_cols])[:, 1]
    if args.threshold is None:
        params = dict(payload.get("training_params", {}))
        threshold = float(params.get("threshold_95p_from_train", 0.5))
    else:
        threshold = float(args.threshold)

    report = {
        "candidate_csv": str(args.candidate_csv),
        "model": str(args.model),
        "model_kind": payload.get("model_kind"),
        "split": str(args.split),
        "rows": int(len(rows)),
        "label_counts": rows[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "negative_type_counts": rows["negative_type"].value_counts().to_dict()
        if "negative_type" in rows.columns
        else {},
        "feature_count": int(len(feature_cols)),
        "threshold_used": threshold,
        "goal_gate": _goal_gate(
            y_true,
            proba,
            min_precision=float(args.min_precision),
            min_recall=float(args.min_recall),
        ),
        "metrics_at_threshold": _metrics(y_true, proba, threshold),
        "slice_metrics_at_threshold": _slice_metrics(rows, proba, threshold)
        if "negative_type" in rows.columns
        else {},
        "top_univariate_feature_audit": _univariate_feature_audit(
            rows,
            feature_cols,
            limit=int(args.feature_audit_limit),
        ),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log("[DONE] Independent evaluation complete")
    _log(json.dumps(report["goal_gate"], indent=2))
    _log(f"[DONE] output_json={output_json}")


if __name__ == "__main__":
    main()

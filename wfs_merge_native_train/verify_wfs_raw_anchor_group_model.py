#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4"
DEFAULT_SWEEPS = (
    f"{DEFAULT_MODEL_DIR}/heldout_mod20r0_base_candidate_budget_sweep.csv,"
    f"{DEFAULT_MODEL_DIR}/heldout_mod20r5_base_candidate_budget_sweep.csv"
)
DEFAULT_OUTPUT_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_group_95_95_verification.json"
DEFAULT_MODEL = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_group_model_v1.joblib"
DEFAULT_MODEL_METRICS_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_group_metrics_v1.json"
DEFAULT_SCORED_SUMMARIES = (
    f"{DEFAULT_MODEL_DIR}/heldout_mod20r0_base_scored_candidates_rescore.summary.json,"
    f"{DEFAULT_MODEL_DIR}/heldout_mod20r5_base_scored_candidates_rescore.summary.json"
)
DEFAULT_SPLIT_AUDIT_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_group_split_audit.json"


def _parse_paths(value: str) -> list[Path]:
    paths = [Path(part.strip()) for part in str(value or "").split(",") if part.strip()]
    if not paths:
        raise ValueError("--sweep-csv must contain at least one CSV path")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sweep CSVs: {missing}")
    return paths


def _optional_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in str(value or "").split(",") if part.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    info = {
        "path": str(path),
        "exists": bool(path.exists()),
    }
    if not path.exists():
        return info
    stat = path.stat()
    info.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }
    )
    if include_sha256:
        info["sha256"] = _sha256(path)
    return info


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_summary(path: Path) -> dict[str, Any]:
    info = _file_info(path, include_sha256=True)
    if not path.exists():
        return info
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        info["payload_type"] = type(payload).__name__
        return info
    training_params = dict(payload.get("training_params", {}) or {})
    info.update(
        {
            "model_kind": payload.get("model_kind"),
            "feature_count": len(list(payload.get("feature_cols", []))),
            "training_params": {
                key: training_params.get(key)
                for key in [
                    "wfs_clean_gpkg",
                    "wfs_clean_layer",
                    "target_gpkg",
                    "target_layer",
                    "candidate_input_csv",
                    "target_id_mod",
                    "target_id_remainders",
                    "per_anchor_candidate_limit",
                    "proposal_expanded_candidate_limit",
                    "proposal_keep_per_target",
                    "max_negative_train_rows",
                    "partial_negative_weight",
                    "overmerge_negative_weight",
                    "random_state",
                ]
                if key in training_params
            },
        }
    )
    return info


def _metrics_summary(path: Path) -> dict[str, Any]:
    info = _file_info(path, include_sha256=True)
    metrics = _read_json(path)
    if not metrics:
        return info
    info.update(
        {
            "model": metrics.get("model"),
            "model_kind": metrics.get("model_kind"),
            "candidate_rows": metrics.get("candidate_rows"),
            "label_counts": metrics.get("label_counts"),
            "label_source_counts": metrics.get("label_source_counts"),
            "feature_count": len(list(metrics.get("feature_columns", []))),
        }
    )
    return info


def _scored_summary(path: Path) -> dict[str, Any]:
    info = _file_info(path, include_sha256=True)
    summary = _read_json(path)
    if not summary:
        return info
    info.update(
        {
            "model": summary.get("model"),
            "candidate_input_csv": summary.get("candidate_input_csv"),
            "output_csv": summary.get("output_csv"),
            "rows": summary.get("rows"),
            "threshold": summary.get("threshold"),
            "above_threshold_rows": summary.get("above_threshold_rows"),
            "proba_min": summary.get("proba_min"),
            "proba_max": summary.get("proba_max"),
        }
    )
    return info


def _find_metric_row(frame: pd.DataFrame, *, threshold: float, candidate_budget: int) -> pd.Series:
    required = {
        "threshold",
        "candidate_budget",
        "selected_exact_precision_evaluable",
        "selected_exact_recall_available_targets",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Sweep CSV missing required columns: {sorted(missing)}")
    threshold_values = pd.to_numeric(frame["threshold"], errors="coerce")
    budget_values = pd.to_numeric(frame["candidate_budget"], errors="coerce")
    mask = threshold_values.sub(float(threshold)).abs().le(1e-12) & budget_values.eq(int(candidate_budget))
    rows = frame.loc[mask].copy()
    if rows.empty:
        raise RuntimeError(f"No metric row found for threshold={threshold} and candidate_budget={candidate_budget}")
    return rows.iloc[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify current raw-anchor model sweeps pass the 95/95 gate.")
    parser.add_argument("--sweep-csv", default=DEFAULT_SWEEPS)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-metrics-json", default=DEFAULT_MODEL_METRICS_JSON)
    parser.add_argument("--scored-summary-json", default=DEFAULT_SCORED_SUMMARIES)
    parser.add_argument("--split-audit-json", default=DEFAULT_SPLIT_AUDIT_JSON)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--candidate-budget", type=int, default=96)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    for path in _parse_paths(str(args.sweep_csv)):
        frame = pd.read_csv(path)
        row = _find_metric_row(
            frame,
            threshold=float(args.threshold),
            candidate_budget=int(args.candidate_budget),
        )
        precision = float(row["selected_exact_precision_evaluable"])
        recall = float(row["selected_exact_recall_available_targets"])
        result = {
            "sweep_csv": str(path),
            "sweep_file": _file_info(path, include_sha256=True),
            "threshold": float(row["threshold"]),
            "candidate_budget": int(row["candidate_budget"]),
            "selected_rows": int(row.get("selected_rows", 0)),
            "selected_exact_rows": int(row.get("selected_exact_rows", 0)),
            "selected_partial_rows": int(row.get("selected_partial_rows", 0)),
            "selected_overmerge_rows": int(row.get("selected_overmerge_rows", 0)),
            "selected_mismatch_rows": int(row.get("selected_mismatch_rows", 0)),
            "selected_exact_precision_evaluable": precision,
            "selected_exact_recall_available_targets": recall,
            "passes_precision": bool(precision >= float(args.min_precision)),
            "passes_recall": bool(recall >= float(args.min_recall)),
        }
        result["passes"] = bool(result["passes_precision"] and result["passes_recall"])
        results.append(result)

    summary = {
        "gate": {
            "threshold": float(args.threshold),
            "candidate_budget": int(args.candidate_budget),
            "min_precision": float(args.min_precision),
            "min_recall": float(args.min_recall),
        },
        "sweep_count": int(len(results)),
        "passes": bool(results and all(row["passes"] for row in results)),
        "min_precision_observed": min((row["selected_exact_precision_evaluable"] for row in results), default=None),
        "min_recall_observed": min((row["selected_exact_recall_available_targets"] for row in results), default=None),
        "model": _model_summary(Path(args.model)),
        "model_metrics": _metrics_summary(Path(args.model_metrics_json)),
        "scored_candidate_summaries": [
            _scored_summary(path)
            for path in _optional_paths(str(args.scored_summary_json))
        ],
        "split_audit": _read_json(Path(args.split_audit_json)),
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

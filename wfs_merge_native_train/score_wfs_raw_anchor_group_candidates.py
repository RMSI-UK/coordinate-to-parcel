#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_wfs_raw_anchor_group_model import (
    _add_pool_rank_features,
    read_candidate_inputs,
)


DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/"
    "wfs_raw_anchor_group_model_v1.joblib"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _score_in_batches(
    pipeline: Any,
    rows: pd.DataFrame,
    feature_cols: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    if rows.empty:
        return np.array([], dtype="float64")
    batch_size = max(int(batch_size), 1)
    parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        parts.append(pipeline.predict_proba(rows.iloc[start:stop][feature_cols])[:, 1])
    return np.concatenate(parts) if parts else np.array([], dtype="float64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score raw-WFS anchor group candidate CSVs with a trained model.")
    parser.add_argument("--candidate-input-csv", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--disable-pool-rank-features", action="store_true")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.model)
    if not isinstance(payload, dict) or payload.get("model_kind") != "wfs_raw_anchor_group_scorer":
        raise RuntimeError("--model must point to a wfs_raw_anchor_group_scorer payload.")
    pipeline = payload["pipeline"]
    feature_cols = list(payload["feature_cols"])

    dataset = read_candidate_inputs(str(args.candidate_input_csv))
    dataset = dataset.drop(columns=["raw_anchor_group_proba", "raw_anchor_group_pred_at_threshold"], errors="ignore")
    if bool(args.disable_pool_rank_features):
        _log("[INFO] Pool rank features disabled for scoring")
    else:
        dataset = _add_pool_rank_features(dataset)
    for column in feature_cols:
        if column not in dataset.columns:
            dataset[column] = np.nan

    _log(f"[INFO] Scoring rows={len(dataset):,}; features={len(feature_cols):,}; model={args.model}")
    dataset["raw_anchor_group_proba"] = _score_in_batches(
        pipeline,
        dataset,
        feature_cols,
        batch_size=int(args.batch_size),
    )
    dataset["raw_anchor_group_pred_at_threshold"] = dataset["raw_anchor_group_proba"].ge(float(args.threshold)).astype(int)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    summary = {
        "model": str(args.model),
        "candidate_input_csv": str(args.candidate_input_csv),
        "output_csv": str(output),
        "rows": int(len(dataset)),
        "threshold": float(args.threshold),
        "above_threshold_rows": int(dataset["raw_anchor_group_pred_at_threshold"].sum()),
        "proba_min": float(dataset["raw_anchor_group_proba"].min()) if not dataset.empty else None,
        "proba_max": float(dataset["raw_anchor_group_proba"].max()) if not dataset.empty else None,
    }
    summary_path = Path(args.summary_json) if str(args.summary_json).strip() else output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"[DONE] scored_candidates={output}")
    _log(f"[DONE] summary_json={summary_path}")


if __name__ == "__main__":
    main()

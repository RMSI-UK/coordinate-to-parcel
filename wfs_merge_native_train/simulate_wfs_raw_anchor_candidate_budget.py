#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wfs_merge_native"))
sys.path.insert(0, str(ROOT / "wfs_merge_native_train"))

from apply_wfs_raw_anchor_group_model import _select_candidates  # noqa: E402
from evaluate_wfs_raw_anchor_group_apply import (  # noqa: E402
    build_target_index,
    build_wfs_indexes,
    candidate_pool_summary,
    classify_candidates,
    selected_metrics,
)
from train_wfs_raw_anchor_group_model import read_candidate_inputs  # noqa: E402


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value or "").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("--candidate-budgets must contain at least one integer")
    return sorted(set(out))


def _parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value or "").replace(" ", ",").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("--thresholds must contain at least one number")
    return sorted(set(out))


def _budget_filter(candidates: pd.DataFrame, *, budget: int) -> pd.DataFrame:
    if int(budget) <= 0:
        return candidates.copy()
    group_col = "target_train_component_id" if "target_train_component_id" in candidates.columns else "anchor_source_fid"
    ranks = candidates.groupby(group_col, sort=False).cumcount() + 1
    return candidates.loc[ranks.le(int(budget))].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate candidate budget truncation for raw-anchor scored candidates.")
    parser.add_argument("--candidate-input-csv", required=True)
    parser.add_argument("--candidate-budgets", default="80,100,120,140,160")
    parser.add_argument("--thresholds", default="0,0.001,0.003,0.005,0.007,0.01,0.02")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--target-gpkg", default="/data/sheffield/spatial/base-map/sheffield_wfs_raw_merged_council_train.gpkg")
    parser.add_argument("--target-layer", default="wfs_raw_merged_council_train_merged_only")
    parser.add_argument("--wfs-clean-gpkg", default="/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg")
    parser.add_argument("--wfs-clean-layer", default="wfs_raw_clean")
    parser.add_argument("--uprn-gpkg", default="/data/base-data/osopenuprn_202602.gpkg")
    parser.add_argument("--uprn-layer", default="osopenuprn_address")
    parser.add_argument("--uprn-id-field", default="UPRN")
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--target-id-mod", type=int, default=0)
    parser.add_argument("--target-id-remainders", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_candidate_inputs(str(args.candidate_input_csv))
    required = {"anchor_source_fid", "candidate_clean_fids", "candidate_source_fids", "raw_anchor_group_proba"}
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"Candidate rows missing required columns: {sorted(missing)}")
    candidates["anchor_source_fid"] = pd.to_numeric(candidates["anchor_source_fid"], errors="coerce")
    candidates["raw_anchor_group_proba"] = pd.to_numeric(candidates["raw_anchor_group_proba"], errors="coerce")
    candidates = candidates[candidates["anchor_source_fid"].notna() & candidates["raw_anchor_group_proba"].notna()].copy()
    candidates["anchor_source_fid"] = candidates["anchor_source_fid"].astype("int64")
    for column in ["candidate_clean_fids", "candidate_source_fids", "anchor_clean_fids"]:
        if column in candidates.columns:
            candidates[column] = candidates[column].fillna("").astype(str)

    anchor_owner_by_clean, source_to_clean = build_wfs_indexes(args)
    targets = build_target_index(args, source_to_clean)
    classified = classify_candidates(candidates, targets)

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for budget in _parse_int_list(str(args.candidate_budgets)):
        budgeted = _budget_filter(classified, budget=int(budget))
        budgeted = budgeted.sort_values("raw_anchor_group_proba", ascending=False).drop_duplicates(
            ["anchor_source_fid", "candidate_clean_fids"],
            keep="first",
        )
        summaries[str(budget)] = candidate_pool_summary(budgeted, targets)
        _log(
            "[INFO] Budget "
            f"{budget}: rows={len(budgeted):,}; exact_rows={summaries[str(budget)]['exact_candidate_rows']:,}; "
            f"exact_recall_available_ceiling={summaries[str(budget)]['candidate_pool_exact_recall_ceiling_available']:.4f}"
        )
        for threshold in _parse_float_list(str(args.thresholds)):
            selected, review, conflicts = _select_candidates(
                budgeted,
                threshold=float(threshold),
                anchor_owner_by_clean=anchor_owner_by_clean,
            )
            metrics = selected_metrics(
                threshold=float(threshold),
                selected=selected,
                review=review,
                conflicts=conflicts,
                targets=targets,
            )
            metrics["candidate_budget"] = int(budget)
            metrics["candidate_rows"] = int(len(budgeted))
            rows.append(metrics)
            _log(
                "[INFO] Sweep "
                f"budget={budget}; threshold={threshold:.6g}; selected={len(selected):,}; "
                f"precision={metrics['selected_exact_precision_evaluable']:.4f}; "
                f"recall_available={metrics['selected_exact_recall_available_targets']:.4f}"
            )

    out = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    output_json = Path(args.output_json) if str(args.output_json).strip() else output_csv.with_suffix(".json")
    output_json.write_text(
        json.dumps(
            {
                "candidate_input_csv": str(args.candidate_input_csv),
                "candidate_budgets": _parse_int_list(str(args.candidate_budgets)),
                "thresholds": _parse_float_list(str(args.thresholds)),
                "budget_candidate_pool_summary": summaries,
                "threshold_sweep": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"[DONE] budget_sweep_csv={output_csv}")
    _log(f"[DONE] budget_sweep_json={output_json}")


if __name__ == "__main__":
    main()

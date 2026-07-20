#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRAIN_CANDIDATES = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_candidate_cache_proposal_mod20r1_4_v1,"
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_candidate_cache_proposal_mod20r6_9_v1"
)
DEFAULT_HELDOUT_CANDIDATES = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_candidate_cache_proposal_mod20r0_v1,"
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_candidate_cache_proposal_mod20r5_v1"
)
DEFAULT_OUTPUT_JSON = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/"
    "wfs_raw_anchor_group_split_audit.json"
)


def _candidate_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token)
        if path.is_dir():
            paths.extend(sorted(path.glob("wfs_raw_anchor_group_candidates_part_*.csv")))
        elif any(ch in token for ch in "*?["):
            paths.extend(Path(match) for match in sorted(glob.glob(token)))
        else:
            paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Candidate CSV not found: {missing[:5]}")
    if not unique:
        raise RuntimeError("No candidate CSVs matched")
    return unique


def _scan_target_ids(paths: list[Path], *, chunk_size: int) -> dict[str, Any]:
    target_ids: set[int] = set()
    rows = 0
    files: list[dict[str, Any]] = []
    for path in paths:
        file_rows = 0
        file_ids: set[int] = set()
        for chunk in pd.read_csv(
            path,
            usecols=["target_train_component_id"],
            chunksize=max(int(chunk_size), 1),
        ):
            values = pd.to_numeric(chunk["target_train_component_id"], errors="coerce").dropna().astype("int64")
            ids = set(int(value) for value in values.unique())
            file_ids.update(ids)
            target_ids.update(ids)
            file_rows += int(len(chunk))
        rows += int(file_rows)
        files.append(
            {
                "path": str(path),
                "rows": int(file_rows),
                "unique_target_ids": int(len(file_ids)),
                "remainders_mod20": sorted({int(value) % 20 for value in file_ids}),
            }
        )
    return {
        "candidate_files": int(len(paths)),
        "candidate_rows": int(rows),
        "unique_target_ids": int(len(target_ids)),
        "target_ids": target_ids,
        "remainders_mod20": sorted({int(value) % 20 for value in target_ids}),
        "files": files,
    }


def _public_split_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key != "target_ids"
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw-anchor train/heldout candidate split disjointness.")
    parser.add_argument("--train-candidate-csv", default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--heldout-candidate-csv", default=DEFAULT_HELDOUT_CANDIDATES)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--chunk-size", type=int, default=500000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = _scan_target_ids(_candidate_paths(str(args.train_candidate_csv)), chunk_size=int(args.chunk_size))
    heldout = _scan_target_ids(_candidate_paths(str(args.heldout_candidate_csv)), chunk_size=int(args.chunk_size))
    overlap = sorted(train["target_ids"] & heldout["target_ids"])
    expected_train_remainders = {1, 2, 3, 4, 6, 7, 8, 9}
    expected_heldout_remainders = {0, 5}
    train_remainders = set(train["remainders_mod20"])
    heldout_remainders = set(heldout["remainders_mod20"])
    summary = {
        "train_candidate_csv": str(args.train_candidate_csv),
        "heldout_candidate_csv": str(args.heldout_candidate_csv),
        "train": _public_split_summary(train),
        "heldout": _public_split_summary(heldout),
        "overlap_target_id_count": int(len(overlap)),
        "overlap_target_ids_sample": overlap[:25],
        "expected_train_remainders_mod20": sorted(expected_train_remainders),
        "expected_heldout_remainders_mod20": sorted(expected_heldout_remainders),
        "train_remainders_match_expected": bool(train_remainders == expected_train_remainders),
        "heldout_remainders_match_expected": bool(heldout_remainders == expected_heldout_remainders),
    }
    summary["passes"] = bool(
        len(overlap) == 0
        and summary["train_remainders_match_expected"]
        and summary["heldout_remainders_match_expected"]
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

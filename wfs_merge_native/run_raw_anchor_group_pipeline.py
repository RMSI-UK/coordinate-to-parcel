#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from apply_wfs_raw_anchor_group_model import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_GPKG,
    DEFAULT_PROPOSAL_MODEL,
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_ID_FIELD,
    DEFAULT_UPRN_LAYER,
)
from preprocess_wfs_raw import DEFAULT_OUTPUT_LAYER, DEFAULT_WFS_GPKG, DEFAULT_WFS_LAYER


DEFAULT_WORK_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_pipeline"
DEFAULT_CLEAN_NAME = "wfs_raw_clean_for_anchor_group.gpkg"


def _log(message: str) -> None:
    print(message, flush=True)


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    _log("[RUN] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pure raw-WFS anchor-group parcel merge workflow: "
            "preprocess raw WFS, then apply the proposal model + group scorer."
        )
    )
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--proposal-model", default=DEFAULT_PROPOSAL_MODEL)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--clean-gpkg", default="")
    parser.add_argument("--clean-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--bbox", default="", help="minx,miny,maxx,maxy. Required unless --allow-full-raw is set.")
    parser.add_argument("--allow-full-raw", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--review-threshold", type=float, default=0.0)
    parser.add_argument("--proposal-expanded-candidate-limit", type=int, default=3000)
    parser.add_argument("--proposal-keep-per-anchor", type=int, default=80)
    parser.add_argument("--full-score-per-anchor-limit", type=int, default=96)
    parser.add_argument("--anchor-workers", type=int, default=min(max(mp.cpu_count(), 1), 16))
    parser.add_argument("--debug-layer-limit", type=int, default=0)
    parser.add_argument("--validate-clean-overlaps", action="store_true")
    parser.add_argument("--write-preprocess-debug-layers", action="store_true")
    parser.add_argument("--keep-clean", action="store_true", help="Retain the intermediate clean WFS GPKG.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    if not str(args.bbox).strip() and not bool(args.allow_full_raw):
        raise SystemExit("Refusing to run raw WFS without --bbox. Pass --allow-full-raw only for an intentional full run.")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    preprocess_script = repo_root / "wfs_merge_native_train" / "preprocess_wfs_raw.py"
    apply_script = script_dir / "apply_wfs_raw_anchor_group_model.py"
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_gpkg = Path(args.clean_gpkg) if str(args.clean_gpkg).strip() else work_dir / DEFAULT_CLEAN_NAME
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    preprocess_cmd = [
        str(args.python),
        str(preprocess_script),
        "--wfs-gpkg",
        str(args.wfs_gpkg),
        "--wfs-layer",
        str(args.wfs_layer),
        "--output-gpkg",
        str(clean_gpkg),
        "--output-layer",
        str(args.clean_layer),
        "--overwrite",
    ]
    if str(args.bbox).strip():
        preprocess_cmd.extend(["--bbox", str(args.bbox)])
    if bool(args.validate_clean_overlaps):
        preprocess_cmd.append("--validate-overlaps")
    if bool(args.write_preprocess_debug_layers):
        preprocess_cmd.append("--write-debug-layers")

    apply_cmd = [
        str(args.python),
        str(apply_script),
        "--wfs-clean-gpkg",
        str(clean_gpkg),
        "--wfs-clean-layer",
        str(args.clean_layer),
        "--uprn-gpkg",
        str(args.uprn_gpkg),
        "--uprn-layer",
        str(args.uprn_layer),
        "--uprn-id-field",
        str(args.uprn_id_field),
        "--model",
        str(args.model),
        "--proposal-model",
        str(args.proposal_model),
        "--output-gpkg",
        str(output_gpkg),
        "--threshold",
        str(float(args.threshold)),
        "--review-threshold",
        str(float(args.review_threshold)),
        "--proposal-expanded-candidate-limit",
        str(int(args.proposal_expanded_candidate_limit)),
        "--proposal-keep-per-anchor",
        str(int(args.proposal_keep_per_anchor)),
        "--full-score-per-anchor-limit",
        str(int(args.full_score_per_anchor_limit)),
        "--anchor-workers",
        str(int(args.anchor_workers)),
        "--debug-layer-limit",
        str(int(args.debug_layer_limit)),
    ]

    _run(preprocess_cmd, cwd=repo_root, dry_run=bool(args.dry_run))
    _run(apply_cmd, cwd=repo_root, dry_run=bool(args.dry_run))

    summary = {
        "workflow": "raw_wfs_anchor_group_model",
        "wfs_gpkg": str(args.wfs_gpkg),
        "wfs_layer": str(args.wfs_layer),
        "uprn_gpkg": str(args.uprn_gpkg),
        "uprn_layer": str(args.uprn_layer),
        "bbox": str(args.bbox),
        "clean_gpkg": str(clean_gpkg),
        "clean_layer": str(args.clean_layer),
        "output_gpkg": str(output_gpkg),
        "model": str(args.model),
        "proposal_model": str(args.proposal_model),
        "threshold": float(args.threshold),
        "anchor_workers": int(args.anchor_workers),
        "preprocess_summary": _read_json(clean_gpkg.with_suffix(".preprocess_summary.json")),
        "apply_summary": _read_json(output_gpkg.with_suffix(".summary.json")),
        "elapsed_seconds": float(time.monotonic() - started_at),
    }
    summary_path = output_gpkg.with_suffix(".raw_anchor_pipeline_summary.json")
    if not bool(args.dry_run):
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not bool(args.keep_clean) and not str(args.clean_gpkg).strip():
            # Keep the summary for reproducibility, but remove the bulky temp GPKG by default.
            clean_gpkg.unlink(missing_ok=True)
    _log(f"[DONE] output={output_gpkg}")
    _log(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RAW_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg"
DEFAULT_RAW_WFS_LAYER = "polygons_in_buffers"
DEFAULT_CLEAN_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_CLEAN_WFS_LAYER = "wfs_raw_clean"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_COUNCIL_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons.gpkg"
DEFAULT_COUNCIL_LAYER = "council_polygons"
DEFAULT_ANCHOR_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean_anchor.gpkg"
DEFAULT_ANCHOR_LAYER = "wfs_raw_clean_anchor"
DEFAULT_MAX_ANCHOR_AREA_M2 = 4000.0
DEFAULT_COUNCIL_SINGLE_ANCHOR_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_area05.gpkg"
DEFAULT_COUNCIL_SINGLE_ANCHOR_LAYER = "council_polygons_single_anchor_area05"
DEFAULT_FALLBACK_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_fallback.gpkg"
DEFAULT_FALLBACK_LAYER = "council_polygons_single_anchor_fallback"
DEFAULT_WORKFLOW_SUMMARY = "/data/sheffield/spatial/base-map/wfs_merge_anchor_workflow.summary.json"


def _log(message: str) -> None:
    print(message, flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def _run_command(cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    _log("[RUN] " + " ".join(cmd))
    if dry_run:
        return {
            "status": "dry_run",
            "command": cmd,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    subprocess.run(cmd, check=True)
    finished = datetime.now(timezone.utc)
    return {
        "status": "completed",
        "command": cmd,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }


def _summary_for_output(path: Path) -> dict[str, Any] | None:
    for candidate in (
        path.with_suffix(".summary.json"),
        path.with_suffix(".preprocess_summary.json"),
    ):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"summary_path": str(candidate), "summary_error": "invalid JSON"}
    return None


def _stage_result(
    name: str,
    output_path: Path,
    *,
    command: list[str],
    should_run: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not should_run:
        _log(f"[SKIP] {name}: output exists ({output_path})")
        return {
            "stage": name,
            "status": "skipped_existing",
            "output_gpkg": str(output_path),
            "summary": _summary_for_output(output_path),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(command, dry_run=dry_run)
    result.update(
        {
            "stage": name,
            "output_gpkg": str(output_path),
            "summary": None if dry_run else _summary_for_output(output_path),
        }
    )
    return result


def build_workflow(args: argparse.Namespace) -> dict[str, Any]:
    env_threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    os.environ.update({key: os.environ.get(key, value) for key, value in env_threads.items()})

    python = str(args.python)
    clean_wfs = Path(args.clean_wfs_gpkg)
    anchor_gpkg = Path(args.anchor_gpkg)
    council_single = Path(args.council_single_anchor_gpkg)
    fallback_gpkg = Path(args.fallback_gpkg)

    stages: list[dict[str, Any]] = []

    run_preprocess = bool(args.run_preprocess) or not clean_wfs.exists()
    if not bool(args.skip_preprocess):
        preprocess_cmd = [
            python,
            str(_script_path("preprocess_wfs_raw.py")),
            "--wfs-gpkg",
            str(args.raw_wfs_gpkg),
            "--wfs-layer",
            str(args.raw_wfs_layer),
            "--output-gpkg",
            str(clean_wfs),
            "--output-layer",
            str(args.clean_wfs_layer),
            "--overwrite",
        ]
        if bool(args.validate_preprocess_overlaps):
            preprocess_cmd.append("--validate-overlaps")
        if bool(args.write_preprocess_debug_layers):
            preprocess_cmd.append("--write-debug-layers")
        stages.append(
            _stage_result(
                "preprocess_wfs_raw",
                clean_wfs,
                command=preprocess_cmd,
                should_run=bool(args.force) or run_preprocess,
                dry_run=bool(args.dry_run),
            )
        )

    if not bool(args.skip_anchor):
        anchor_cmd = [
            python,
            str(_script_path("build_anchor_layer.py")),
            "--wfs-gpkg",
            str(clean_wfs),
            "--wfs-layer",
            str(args.clean_wfs_layer),
            "--uprn-gpkg",
            str(args.uprn_gpkg),
            "--uprn-layer",
            str(args.uprn_layer),
            "--output-gpkg",
            str(anchor_gpkg),
            "--output-layer",
            str(args.anchor_layer),
            "--max-anchor-area",
            str(args.max_anchor_area),
            "--overwrite",
        ]
        stages.append(
            _stage_result(
                "build_anchor_layer",
                anchor_gpkg,
                command=anchor_cmd,
                should_run=bool(args.force) or not anchor_gpkg.exists(),
                dry_run=bool(args.dry_run),
            )
        )

    if not bool(args.skip_council_single_anchor):
        council_cmd = [
            python,
            str(_script_path("build_council_single_anchor.py")),
            "--council-gpkg",
            str(args.council_gpkg),
            "--council-layer",
            str(args.council_layer),
            "--anchor-gpkg",
            str(anchor_gpkg),
            "--anchor-layer",
            str(args.anchor_layer),
            "--output-gpkg",
            str(council_single),
            "--output-layer",
            str(args.council_single_anchor_layer),
            "--wfs-gpkg",
            str(clean_wfs),
            "--wfs-layer",
            str(args.clean_wfs_layer),
            "--min-anchor-overlap-area",
            str(args.council_min_anchor_overlap_area),
            "--wfs-match-min-iou",
            str(args.wfs_match_min_iou),
            "--wfs-match-min-regularity",
            str(args.wfs_match_min_regularity),
            "--wfs-match-max-hull-gap",
            str(args.wfs_match_max_hull_gap),
            "--wfs-match-max-hole-area-ratio",
            str(args.wfs_match_max_hole_area_ratio),
            "--wfs-match-min-candidate-inside-ratio",
            str(args.wfs_match_min_candidate_inside_ratio),
            "--wfs-match-min-candidate-cover-ratio",
            str(args.wfs_match_min_candidate_cover_ratio),
            "--wfs-match-max-candidates",
            str(args.wfs_match_max_candidates),
            "--wfs-match-max-selected",
            str(args.wfs_match_max_selected),
            "--overwrite",
        ]
        if not bool(args.enable_wfs_iou_match):
            council_cmd.append("--no-enable-wfs-iou-match")
        stages.append(
            _stage_result(
                "build_council_single_anchor",
                council_single,
                command=council_cmd,
                should_run=bool(args.force) or not council_single.exists(),
                dry_run=bool(args.dry_run),
            )
        )

    if not bool(args.skip_fallback):
        fallback_cmd = [
            python,
            str(_script_path("build_single_anchor_fallback.py")),
            "--wfs-gpkg",
            str(clean_wfs),
            "--wfs-layer",
            str(args.clean_wfs_layer),
            "--anchor-gpkg",
            str(anchor_gpkg),
            "--anchor-layer",
            str(args.anchor_layer),
            "--council-single-anchor-gpkg",
            str(council_single),
            "--council-single-anchor-layer",
            str(args.council_single_anchor_layer),
            "--council-min-anchor-overlap-area",
            str(args.council_min_anchor_overlap_area),
            "--uprn-gpkg",
            str(args.uprn_gpkg),
            "--uprn-layer",
            str(args.uprn_layer),
            "--output-gpkg",
            str(fallback_gpkg),
            "--output-layer",
            str(args.fallback_layer),
            "--fallback-scope",
            str(args.fallback_scope),
            "--min-shared-edge",
            str(args.min_shared_edge),
            "--top-direct",
            str(args.top_direct),
            "--top-indirect-per-direct",
            str(args.top_indirect_per_direct),
            "--max-enum-nodes",
            str(args.max_enum_nodes),
            "--min-completion-regularity",
            str(args.min_completion_regularity),
            "--max-completion-regularity-drop",
            str(args.max_completion_regularity_drop),
            "--max-merge-regularity-loss-vs-anchor",
            str(args.max_merge_regularity_loss_vs_anchor),
            "--strong-completion-min-regularity",
            str(args.strong_completion_min_regularity),
            "--strong-completion-max-hull-gap",
            str(args.strong_completion_max_hull_gap),
            "--strong-completion-max-hole-area-ratio",
            str(args.strong_completion_max_hole_area_ratio),
            "--secondary-direct-min-ratio-of-best",
            str(args.secondary_direct_min_ratio_of_best),
            "--secondary-direct-min-attraction-ratio",
            str(args.secondary_direct_min_attraction_ratio),
            "--max-completion-hull-gap",
            str(args.max_completion_hull_gap),
            "--max-completion-hole-area-ratio",
            str(args.max_completion_hole_area_ratio),
            "--large-anchor-area-threshold",
            str(args.large_anchor_area_threshold),
            "--large-anchor-min-shared-edge",
            str(args.large_anchor_min_shared_edge),
            "--large-anchor-min-attraction-ratio",
            str(args.large_anchor_min_attraction_ratio),
            "--large-anchor-max-regularity-drop",
            str(args.large_anchor_max_regularity_drop),
            "--large-anchor-max-hull-gap-increase",
            str(args.large_anchor_max_hull_gap_increase),
            "--overwrite",
        ]
        stages.append(
            _stage_result(
                "build_single_anchor_fallback",
                fallback_gpkg,
                command=fallback_cmd,
                should_run=bool(args.force) or not fallback_gpkg.exists(),
                dry_run=bool(args.dry_run),
            )
        )

    summary = {
        "workflow": "wfs_merge_anchor",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "run_preprocess": bool(args.run_preprocess),
        "outputs": {
            "clean_wfs_gpkg": str(clean_wfs),
            "anchor_gpkg": str(anchor_gpkg),
            "council_single_anchor_gpkg": str(council_single),
            "fallback_gpkg": str(fallback_gpkg),
        },
        "max_anchor_area_m2": float(args.max_anchor_area),
        "large_anchor_area_threshold_m2": float(args.large_anchor_area_threshold),
        "stages": stages,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end wfs_merge_anchor workflow.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--raw-wfs-gpkg", default=DEFAULT_RAW_WFS_GPKG)
    parser.add_argument("--raw-wfs-layer", default=DEFAULT_RAW_WFS_LAYER)
    parser.add_argument("--clean-wfs-gpkg", default=DEFAULT_CLEAN_WFS_GPKG)
    parser.add_argument("--clean-wfs-layer", default=DEFAULT_CLEAN_WFS_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--council-gpkg", default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--council-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--anchor-gpkg", default=DEFAULT_ANCHOR_GPKG)
    parser.add_argument("--anchor-layer", default=DEFAULT_ANCHOR_LAYER)
    parser.add_argument(
        "--max-anchor-area",
        type=float,
        default=DEFAULT_MAX_ANCHOR_AREA_M2,
        help="Maximum WFS anchor polygon area in square metres; set <=0 to disable.",
    )
    parser.add_argument("--council-single-anchor-gpkg", default=DEFAULT_COUNCIL_SINGLE_ANCHOR_GPKG)
    parser.add_argument("--council-single-anchor-layer", default=DEFAULT_COUNCIL_SINGLE_ANCHOR_LAYER)
    parser.add_argument("--fallback-gpkg", default=DEFAULT_FALLBACK_GPKG)
    parser.add_argument("--fallback-layer", default=DEFAULT_FALLBACK_LAYER)
    parser.add_argument("--fallback-scope", choices=["uncovered", "all"], default="uncovered")
    parser.add_argument("--workflow-summary-json", default=DEFAULT_WORKFLOW_SUMMARY)
    parser.add_argument("--council-min-anchor-overlap-area", type=float, default=0.5)
    parser.add_argument("--enable-wfs-iou-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wfs-match-min-iou", type=float, default=0.90)
    parser.add_argument("--wfs-match-min-regularity", type=float, default=0.70)
    parser.add_argument("--wfs-match-max-hull-gap", type=float, default=0.05)
    parser.add_argument("--wfs-match-max-hole-area-ratio", type=float, default=0.02)
    parser.add_argument("--wfs-match-min-candidate-inside-ratio", type=float, default=0.50)
    parser.add_argument("--wfs-match-min-candidate-cover-ratio", type=float, default=0.01)
    parser.add_argument("--wfs-match-max-candidates", type=int, default=24)
    parser.add_argument("--wfs-match-max-selected", type=int, default=20)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--top-direct", type=int, default=8)
    parser.add_argument("--top-indirect-per-direct", type=int, default=3)
    parser.add_argument("--max-enum-nodes", type=int, default=10)
    parser.add_argument("--min-completion-regularity", type=float, default=0.65)
    parser.add_argument("--max-completion-regularity-drop", type=float, default=0.05)
    parser.add_argument("--max-merge-regularity-loss-vs-anchor", type=float, default=0.05)
    parser.add_argument("--strong-completion-min-regularity", type=float, default=0.70)
    parser.add_argument("--strong-completion-max-hull-gap", type=float, default=0.05)
    parser.add_argument("--strong-completion-max-hole-area-ratio", type=float, default=0.005)
    parser.add_argument("--secondary-direct-min-ratio-of-best", type=float, default=0.75)
    parser.add_argument("--secondary-direct-min-attraction-ratio", type=float, default=0.15)
    parser.add_argument("--max-completion-hull-gap", type=float, default=0.60)
    parser.add_argument("--max-completion-hole-area-ratio", type=float, default=0.02)
    parser.add_argument("--large-anchor-area-threshold", type=float, default=500.0)
    parser.add_argument("--large-anchor-min-shared-edge", type=float, default=8.0)
    parser.add_argument("--large-anchor-min-attraction-ratio", type=float, default=0.08)
    parser.add_argument("--large-anchor-max-regularity-drop", type=float, default=0.01)
    parser.add_argument("--large-anchor-max-hull-gap-increase", type=float, default=0.02)
    parser.add_argument("--run-preprocess", action="store_true", help="Rebuild clean WFS from raw WFS before anchor stages.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-anchor", action="store_true")
    parser.add_argument("--skip-council-single-anchor", action="store_true")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--validate-preprocess-overlaps", action="store_true")
    parser.add_argument("--write-preprocess-debug-layers", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild stage outputs even when they already exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_workflow(args)
    summary_path = Path(args.workflow_summary_json)
    if not bool(args.dry_run):
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
        _log(f"[DONE] Workflow summary: {summary_path}")
    _log(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

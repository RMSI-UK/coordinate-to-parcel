#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyogrio


DEFAULT_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4"
DEFAULT_QUALITY_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_group_95_95_verification.json"
DEFAULT_SPLIT_AUDIT_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_group_split_audit.json"
DEFAULT_RAW_SMOKE_SUMMARY_JSON = (
    f"{DEFAULT_MODEL_DIR}/raw_anchor_pipeline_smoke_elapsed_bbox_431000_386000_431200_386200"
    ".raw_anchor_pipeline_summary.json"
)
DEFAULT_APPLY_BBOX_SUMMARY_JSON = f"{DEFAULT_MODEL_DIR}/default_full_bbox_parallel16_431000_386000_432000_387000.summary.json"
DEFAULT_RUNTIME_ESTIMATE_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_runtime_estimate.json"
DEFAULT_OUTPUT_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_workflow_verification.json"
FORBIDDEN_PRODUCTION_TOKENS = ("council", "reference", "gapfill_council")
REQUIRED_OUTPUT_LAYERS = {"predicted_parcels", "predicted_parcels_merged_only"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_forbidden_tokens(paths: list[Path]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for token in FORBIDDEN_PRODUCTION_TOKENS:
            if token in lower:
                lines = [
                    idx
                    for idx, line in enumerate(text.splitlines(), start=1)
                    if token in line.lower()
                ]
                hits.append(
                    {
                        "path": str(path),
                        "token": token,
                        "line_numbers": lines[:20],
                        "line_count": len(lines),
                    }
                )
    return hits


def _layers(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [str(row[0]) for row in pyogrio.list_layers(path)]


def _run_no_bbox_guard(script: Path, python: str) -> dict[str, Any]:
    proc = subprocess.run(
        [python, str(script), "--dry-run"],
        cwd=str(script.resolve().parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": [python, str(script), "--dry-run"],
        "returncode": int(proc.returncode),
        "output": proc.stdout.strip(),
        "passes": bool(proc.returncode != 0 and "Refusing to run raw WFS without --bbox" in proc.stdout),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the pure raw-anchor production workflow evidence.")
    parser.add_argument("--quality-json", default=DEFAULT_QUALITY_JSON)
    parser.add_argument("--split-audit-json", default=DEFAULT_SPLIT_AUDIT_JSON)
    parser.add_argument("--raw-smoke-summary-json", default=DEFAULT_RAW_SMOKE_SUMMARY_JSON)
    parser.add_argument("--apply-bbox-summary-json", default=DEFAULT_APPLY_BBOX_SUMMARY_JSON)
    parser.add_argument("--runtime-estimate-json", default=DEFAULT_RUNTIME_ESTIMATE_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--runner-script", default="wfs_merge_native/run_raw_anchor_group_pipeline.py")
    parser.add_argument("--apply-script", default="wfs_merge_native/apply_wfs_raw_anchor_group_model.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-raw-smoke-elapsed-seconds", type=float, default=10.0)
    parser.add_argument("--min-apply-bbox-output-parcels", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Path(args.runner_script)
    apply = Path(args.apply_script)
    quality = _read_json(Path(args.quality_json))
    split = _read_json(Path(args.split_audit_json))
    raw_smoke = _read_json(Path(args.raw_smoke_summary_json))
    apply_bbox = _read_json(Path(args.apply_bbox_summary_json))
    runtime_estimate = _read_json(Path(args.runtime_estimate_json))

    smoke_output = Path(str(raw_smoke.get("output_gpkg", "")))
    smoke_layers = set(_layers(smoke_output))
    forbidden_hits = _scan_forbidden_tokens([runner, apply])
    no_bbox_guard = _run_no_bbox_guard(runner, str(args.python))

    checks = {
        "quality_gate_passes": bool(quality.get("passes")),
        "split_audit_passes": bool(split.get("passes")),
        "quality_embeds_split_audit": bool(quality.get("split_audit", {}).get("passes")),
        "production_scripts_have_no_forbidden_tokens": not forbidden_hits,
        "raw_runner_refuses_no_bbox": bool(no_bbox_guard.get("passes")),
        "raw_smoke_workflow_name_ok": raw_smoke.get("workflow") == "raw_wfs_anchor_group_model",
        "raw_smoke_uses_raw_wfs": str(raw_smoke.get("wfs_gpkg", "")).endswith("sheffield_wfs_raw.gpkg"),
        "raw_smoke_has_required_layers": REQUIRED_OUTPUT_LAYERS.issubset(smoke_layers),
        "raw_smoke_elapsed_ok": float(raw_smoke.get("elapsed_seconds", 1e9)) <= float(args.max_raw_smoke_elapsed_seconds),
        "raw_smoke_outputs_parcels": int(raw_smoke.get("apply_summary", {}).get("output_parcels", 0)) > 0,
        "apply_bbox_uses_default_model": str(apply_bbox.get("model", "")) == str(quality.get("model", {}).get("path", "")),
        "apply_bbox_output_scale_ok": int(apply_bbox.get("output_parcels", 0)) >= int(args.min_apply_bbox_output_parcels),
        "runtime_estimate_under_target": bool(
            runtime_estimate.get("estimate", {}).get("estimated_under_target")
        ),
    }
    summary = {
        "passes": bool(all(checks.values())),
        "checks": checks,
        "quality_json": str(args.quality_json),
        "split_audit_json": str(args.split_audit_json),
        "raw_smoke_summary_json": str(args.raw_smoke_summary_json),
        "apply_bbox_summary_json": str(args.apply_bbox_summary_json),
        "runtime_estimate_json": str(args.runtime_estimate_json),
        "runner_script": str(runner),
        "apply_script": str(apply),
        "forbidden_production_tokens": list(FORBIDDEN_PRODUCTION_TOKENS),
        "forbidden_token_hits": forbidden_hits,
        "no_bbox_guard": no_bbox_guard,
        "raw_smoke": {
            "output_gpkg": str(smoke_output),
            "layers": sorted(smoke_layers),
            "elapsed_seconds": raw_smoke.get("elapsed_seconds"),
            "apply_elapsed_seconds": raw_smoke.get("apply_summary", {}).get("elapsed_seconds"),
            "raw_rows": raw_smoke.get("preprocess_summary", {}).get("raw_rows"),
            "clean_rows": raw_smoke.get("preprocess_summary", {}).get("clean_rows"),
            "anchors": raw_smoke.get("apply_summary", {}).get("anchor_rows"),
            "candidate_rows_scored": raw_smoke.get("apply_summary", {}).get("candidate_rows_scored"),
            "selected_groups": raw_smoke.get("apply_summary", {}).get("selected_groups"),
            "output_parcels": raw_smoke.get("apply_summary", {}).get("output_parcels"),
        },
        "apply_bbox": {
            "bbox": apply_bbox.get("bbox"),
            "elapsed_seconds": apply_bbox.get("elapsed_seconds"),
            "output_parcels": apply_bbox.get("output_parcels"),
            "anchor_rows": apply_bbox.get("anchor_rows"),
            "candidate_rows_scored": apply_bbox.get("candidate_rows_scored"),
            "selected_groups": apply_bbox.get("selected_groups"),
            "anchor_workers": apply_bbox.get("anchor_workers"),
        },
        "runtime_estimate": runtime_estimate.get("estimate", {}),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

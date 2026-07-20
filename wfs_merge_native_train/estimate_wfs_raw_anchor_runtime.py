#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyogrio


DEFAULT_WFS_CLEAN_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_CLEAN_LAYER = "wfs_raw_clean"
DEFAULT_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4"
DEFAULT_BBOX_SUMMARY_JSON = f"{DEFAULT_MODEL_DIR}/default_full_bbox_parallel16_431000_386000_432000_387000.summary.json"
DEFAULT_OUTPUT_JSON = f"{DEFAULT_MODEL_DIR}/wfs_raw_anchor_runtime_estimate.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_clean_info(path: Path, layer: str) -> dict[str, Any]:
    info = pyogrio.read_info(path, layer=layer)
    bounds = [float(value) for value in info.get("total_bounds", [])]
    width = bounds[2] - bounds[0] if len(bounds) == 4 else None
    height = bounds[3] - bounds[1] if len(bounds) == 4 else None
    return {
        "path": str(path),
        "layer": str(layer),
        "features": int(info.get("features", 0) or 0),
        "crs": str(info.get("crs")),
        "total_bounds": bounds,
        "extent_width_m": float(width) if width is not None else None,
        "extent_height_m": float(height) if height is not None else None,
        "extent_area_m2": float(width * height) if width is not None and height is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate raw-anchor apply runtime from clean-WFS size and bbox smoke evidence.")
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--bbox-summary-json", default=DEFAULT_BBOX_SUMMARY_JSON)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--target-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = _read_clean_info(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer))
    bbox = _read_json(Path(args.bbox_summary_json))
    clean_rows = int(clean["features"])
    bbox_rows = int(bbox.get("clean_wfs_rows", 0) or 0)
    bbox_seconds = float(bbox.get("elapsed_seconds", 0.0) or 0.0)
    bbox_anchor_rows = int(bbox.get("anchor_rows", 0) or 0)
    bbox_candidate_rows = int(bbox.get("candidate_rows_scored", 0) or 0)
    seconds_per_clean_row = bbox_seconds / bbox_rows if bbox_rows else None
    estimated_apply_seconds = (seconds_per_clean_row * clean_rows) if seconds_per_clean_row is not None else None
    estimated_apply_minutes = (estimated_apply_seconds / 60.0) if estimated_apply_seconds is not None else None
    target_seconds = float(args.target_minutes) * 60.0
    estimated_rows_per_second = (bbox_rows / bbox_seconds) if bbox_seconds else None
    summary = {
        "method": "linear_by_clean_wfs_rows_from_bbox_apply_summary",
        "notes": [
            "This estimates model apply time only from an already-clean WFS layer.",
            "It does not run raw WFS full-area inference.",
            "It does not include full raw-WFS preprocessing time.",
        ],
        "clean_wfs": clean,
        "bbox_evidence": {
            "summary_json": str(args.bbox_summary_json),
            "bbox": bbox.get("bbox"),
            "elapsed_seconds": bbox_seconds,
            "anchor_workers": bbox.get("anchor_workers"),
            "clean_wfs_rows": bbox_rows,
            "anchor_rows": bbox_anchor_rows,
            "candidate_rows_scored": bbox_candidate_rows,
            "output_parcels": bbox.get("output_parcels"),
            "selected_groups": bbox.get("selected_groups"),
        },
        "estimate": {
            "seconds_per_clean_row": seconds_per_clean_row,
            "rows_per_second": estimated_rows_per_second,
            "full_clean_wfs_rows": clean_rows,
            "estimated_apply_seconds": estimated_apply_seconds,
            "estimated_apply_minutes": estimated_apply_minutes,
            "target_minutes": float(args.target_minutes),
            "estimated_under_target": bool(
                estimated_apply_seconds is not None and estimated_apply_seconds <= target_seconds
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

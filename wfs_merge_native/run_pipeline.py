#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg"
DEFAULT_WFS_LAYER = "polygons_in_buffers"
DEFAULT_REFERENCE_GPKG = ""
DEFAULT_REFERENCE_LAYER = "os_wfs_merge"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"

DEFAULT_EDGE_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1"
DEFAULT_COMPLETION_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3"
DEFAULT_OPERATION_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_operation_models_v1"
DEFAULT_PAIR_ANCHOR_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "anchor_problem_detection_model_probe_v1.joblib"
)
DEFAULT_ANCHOR_GROUP_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_train_anchor_group_final_selector_light/"
    "anchor_group_repair_model_v1.joblib"
)

DEFAULT_WORK_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native.gpkg"
DEFAULT_LOG_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native_log.gpkg"
DEFAULT_PROFILE_JOB_ROOT = "/data/file-browser-data/spatial-jobs"

FINAL_SOURCE_LAYER = "predicted_parcels_with_uprn"
FINAL_ALIAS_LAYER = "wfs_raw_merged_native"


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


def _write_final_alias(output_gpkg: Path, source_layer: str, alias_layer: str) -> None:
    _log(f"[INFO] Writing final alias layer: {alias_layer}")
    final = gpd.read_file(output_gpkg, layer=source_layer, engine="pyogrio")
    final.to_file(output_gpkg, layer=alias_layer, driver="GPKG", engine="pyogrio")


def _write_single_layer_output(
    *,
    source_gpkg: Path,
    source_layer: str,
    output_gpkg: Path,
    output_layer: str,
) -> None:
    _log(f"[INFO] Writing clean final output: {output_gpkg} ({output_layer})")
    final = gpd.read_file(source_gpkg, layer=source_layer, engine="pyogrio")
    if output_gpkg.exists():
        output_gpkg.unlink()
    final.to_file(output_gpkg, layer=output_layer, driver="GPKG", engine="pyogrio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the native Sheffield WFS merge production pipeline and write a merged GeoPackage."
        )
    )
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--area-profile", default="", help="Optional AreaProfile JSON used to fill data/model defaults.")
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--reference-gpkg", "--council-land-gpkg", dest="reference_gpkg", default=DEFAULT_REFERENCE_GPKG)
    parser.add_argument("--reference-layer", "--council-land-layer", dest="reference_layer", default=DEFAULT_REFERENCE_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default="UPRN")
    parser.add_argument("--edge-model-dir", default=DEFAULT_EDGE_MODEL_DIR)
    parser.add_argument("--completion-model-dir", default=DEFAULT_COMPLETION_MODEL_DIR)
    parser.add_argument("--operation-model-dir", default=DEFAULT_OPERATION_MODEL_DIR)
    parser.add_argument("--pair-anchor-model", default=DEFAULT_PAIR_ANCHOR_MODEL)
    parser.add_argument("--anchor-group-model", default=DEFAULT_ANCHOR_GROUP_MODEL)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--log-gpkg", default=DEFAULT_LOG_GPKG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--final-layer", default=FINAL_ALIAS_LAYER)
    parser.add_argument("--skip-final-gap-fill", action="store_true")
    parser.add_argument("--keep-output-holes", action="store_true")
    parser.add_argument("--disable-enclosed-gap-fill", action="store_true")
    parser.add_argument("--skip-overmerge-split", action="store_true")
    parser.add_argument("--enclosed-gap-max-area", type=float, default=250.0)
    parser.add_argument("--enclosed-gap-min-shared-edge", type=float, default=0.05)
    parser.add_argument("--min-polygon-part-area", type=float, default=0.01)
    parser.add_argument("--skip-final-alias", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_area_profile(path: str) -> Any:
    code_root = Path(__file__).resolve().parents[2]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from spatial_pipeline.area_profile import load_area_profile

    return load_area_profile(path)


def _default_profile_output_root(area_key: str) -> Path:
    root = Path(os.environ.get("SPATIAL_PIPELINE_JOB_ROOT", DEFAULT_PROFILE_JOB_ROOT))
    return root / "manual" / str(area_key or "area") / "wfs_merge_native"


def _apply_area_profile_defaults(args: argparse.Namespace) -> None:
    if not str(args.area_profile or "").strip():
        return
    profile = _load_area_profile(str(args.area_profile))
    bundle = profile.model_bundle

    wfs_gpkg_was_default = args.wfs_gpkg == DEFAULT_WFS_GPKG
    if args.wfs_gpkg == DEFAULT_WFS_GPKG:
        args.wfs_gpkg = profile.wfs_raw.path
    if wfs_gpkg_was_default and args.wfs_layer == DEFAULT_WFS_LAYER:
        args.wfs_layer = profile.wfs_raw.layer
    if not str(args.reference_gpkg or "").strip():
        args.reference_gpkg = profile.reference.path
    if args.reference_layer == DEFAULT_REFERENCE_LAYER:
        args.reference_layer = profile.reference.layer
    if args.uprn_gpkg == DEFAULT_UPRN_GPKG:
        args.uprn_gpkg = profile.uprn.path
    if args.uprn_layer == DEFAULT_UPRN_LAYER:
        args.uprn_layer = profile.uprn.layer
    if args.edge_model_dir == DEFAULT_EDGE_MODEL_DIR:
        args.edge_model_dir = bundle.edge_model_dir
    if args.completion_model_dir == DEFAULT_COMPLETION_MODEL_DIR:
        args.completion_model_dir = bundle.completion_model_dir
    if args.operation_model_dir == DEFAULT_OPERATION_MODEL_DIR:
        args.operation_model_dir = bundle.operation_model_dir
    if args.pair_anchor_model == DEFAULT_PAIR_ANCHOR_MODEL:
        args.pair_anchor_model = bundle.pair_anchor_model
    if args.anchor_group_model == DEFAULT_ANCHOR_GROUP_MODEL:
        args.anchor_group_model = bundle.anchor_group_model
    if args.final_layer == FINAL_ALIAS_LAYER:
        args.final_layer = profile.native_merge_layer

    default_root = _default_profile_output_root(profile.area_key)
    if args.work_dir == DEFAULT_WORK_DIR:
        args.work_dir = str(default_root / "work")
    if args.output_gpkg == DEFAULT_OUTPUT_GPKG:
        args.output_gpkg = str(default_root / "wfs_raw_merged_native.gpkg")
    if args.log_gpkg == DEFAULT_LOG_GPKG:
        args.log_gpkg = str(default_root / "wfs_raw_merged_native_log.gpkg")


def main() -> None:
    args = parse_args()
    _apply_area_profile_defaults(args)
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    output_gpkg = Path(args.output_gpkg).resolve()
    log_gpkg = Path(args.log_gpkg).resolve()
    clean_output_layer = str(args.final_layer).strip() or FINAL_ALIAS_LAYER
    if output_gpkg == log_gpkg:
        raise ValueError("--output-gpkg and --log-gpkg must be different paths")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    log_gpkg.parent.mkdir(parents=True, exist_ok=True)

    edge_gpkg = work_dir / "01_edge_model.gpkg"
    edge_csv = work_dir / "edge_candidate_predictions_full.csv"
    completion_gpkg = work_dir / "02_completion_model.gpkg"
    operation_gpkg = work_dir / "03_operation_pruned_only.gpkg"
    overmerge_split_gpkg = operation_gpkg if bool(args.skip_overmerge_split) else work_dir / "03b_overmerge_split.gpkg"
    anchor_group_gpkg = log_gpkg if bool(args.skip_final_gap_fill) else work_dir / "04_anchor_group_repaired.gpkg"
    pair_candidate_csv = work_dir / "04_pair_anchor_candidates.csv"
    group_candidate_csv = work_dir / "05_anchor_group_candidates.csv"

    _log("[INFO] Native WFS merge pipeline")
    _log(f"[INFO] work_dir={work_dir}")
    _log(f"[INFO] output_gpkg={output_gpkg}")
    _log(f"[INFO] log_gpkg={log_gpkg}")
    _log(f"[INFO] clean_output_layer={clean_output_layer}")
    reference_gpkg = str(args.reference_gpkg or "").strip()
    reference_enabled = bool(reference_gpkg)
    _log(
        "[INFO] semantic_reference="
        + (f"{reference_gpkg} ({args.reference_layer})" if reference_enabled else "disabled")
    )

    edge_cmd = [
        args.python,
        str(script_dir / "apply_wfs_merge_edge_model.py"),
        "--wfs-gpkg",
        str(args.wfs_gpkg),
        "--wfs-layer",
        str(args.wfs_layer),
        "--uprn-gpkg",
        str(args.uprn_gpkg),
        "--uprn-layer",
        str(args.uprn_layer),
        "--uprn-id-field",
        str(args.uprn_id_field),
        "--edge-model-dir",
        str(args.edge_model_dir),
        "--output-gpkg",
        str(edge_gpkg),
        "--edge-candidate-csv",
        str(edge_csv),
        "--threshold",
        "0.90",
        "--guard-threshold",
        "0.95",
        "--min-component-mrr-ratio",
        "0.90",
        "--max-component-hull-gap-ratio",
        "0.10",
        "--guard-keep-min-proba",
        "0.93",
        "--guard-keep-min-shared-edge",
        "8.0",
        "--comfort-max-reference-area",
        "2000",
        "--comfort-max-reference-source-count",
        "20",
    ]
    if reference_enabled:
        edge_cmd.extend(["--merge-gpkg", reference_gpkg, "--merge-layer", str(args.reference_layer)])
    _run(
        edge_cmd,
        cwd=script_dir,
        dry_run=bool(args.dry_run),
    )

    _run(
        [
            args.python,
            str(script_dir / "apply_wfs_merge_completion_model.py"),
            "--input-prediction-gpkg",
            str(edge_gpkg),
            "--edge-csv",
            str(edge_csv),
            "--edge-model-dir",
            str(args.edge_model_dir),
            "--completion-model-dir",
            str(args.completion_model_dir),
            "--output-gpkg",
            str(completion_gpkg),
            "--threshold",
            "0.90",
            "--max-candidate-area",
            "120",
            "--min-mrr-gain",
            "0",
            "--min-hull-gap-reduction",
            "0",
            "--min-regularity-score-gain",
            "0",
            "--min-boundary-complexity-reduction",
            "0",
            "--min-notch-index-reduction",
            "0",
        ],
        cwd=script_dir,
        dry_run=bool(args.dry_run),
    )

    _run(
        [
            args.python,
            str(script_dir / "apply_wfs_merge_operation_pipeline.py"),
            "--input-gpkg",
            str(completion_gpkg),
            "--prune-model-dir",
            str(args.operation_model_dir),
            "--output-gpkg",
            str(operation_gpkg),
            "--prune-threshold",
            "0.80",
            "--max-prune-component-area",
            "2000",
            "--prune-source-role",
            "land",
            "--max-only-land-prune-source-area-ratio",
            "0.35",
            "--min-prune-mrr-gain",
            "0.10",
            "--min-prune-hull-gap-reduction",
            "0.08",
            "--min-prune-regularity-gain",
            "0.08",
            "--min-after-prune-regularity",
            "0.90",
            "--max-after-prune-hull-gap",
            "0.10",
            "--disable-zero-uprn-attachment",
            "--disable-parcel-completion",
        ],
        cwd=script_dir,
        dry_run=bool(args.dry_run),
    )

    if not bool(args.skip_overmerge_split):
        _run(
            [
                args.python,
                str(script_dir / "apply_wfs_merge_local_mode_split.py"),
                "--input-gpkg",
                str(operation_gpkg),
                "--output-gpkg",
                str(overmerge_split_gpkg),
                "--allow-unflagged-components",
            ],
            cwd=script_dir,
            dry_run=bool(args.dry_run),
        )

    _run(
        [
            args.python,
            str(script_dir / "apply_wfs_merge_anchor_group_repair.py"),
            "--input-gpkg",
            str(overmerge_split_gpkg),
            "--pair-candidate-csv",
            str(pair_candidate_csv),
            "--edge-csv",
            str(edge_csv),
            "--force-rebuild-pair-candidates",
            "--pair-anchor-model",
            str(args.pair_anchor_model),
            "--model",
            str(args.anchor_group_model),
            "--output-gpkg",
            str(anchor_group_gpkg),
            "--candidate-csv",
            str(group_candidate_csv),
            "--candidate-strategy",
            "light",
            "--enclosure-level",
            "pair",
            "--anchor-need-threshold",
            "0.94",
            "--complete-pool-gate-bypass-threshold",
            "0.999",
        ],
        cwd=script_dir,
        dry_run=bool(args.dry_run),
    )

    if not bool(args.skip_final_gap_fill):
        final_layer_arg = "" if bool(args.skip_final_alias) else clean_output_layer
        final_gap_fill_cmd = [
            args.python,
            str(script_dir / "apply_wfs_merge_final_gap_fill.py"),
            "--input-gpkg",
            str(anchor_group_gpkg),
            "--output-gpkg",
            str(log_gpkg),
            "--final-layer",
            final_layer_arg,
            "--enclosed-gap-max-area",
            str(args.enclosed_gap_max_area),
            "--enclosed-gap-min-shared-edge",
            str(args.enclosed_gap_min_shared_edge),
            "--min-polygon-part-area",
            str(args.min_polygon_part_area),
            "--uprn-gpkg",
            str(args.uprn_gpkg),
            "--uprn-layer",
            str(args.uprn_layer),
            "--uprn-id-field",
            str(args.uprn_id_field),
        ]
        if bool(args.keep_output_holes):
            final_gap_fill_cmd.append("--keep-output-holes")
        if bool(args.disable_enclosed_gap_fill):
            final_gap_fill_cmd.append("--disable-enclosed-gap-fill")
        _run(final_gap_fill_cmd, cwd=script_dir, dry_run=bool(args.dry_run))

    clean_source_layer = FINAL_SOURCE_LAYER if bool(args.skip_final_alias) else clean_output_layer
    if bool(args.dry_run):
        _log(
            "[DRY-RUN] Would write clean one-layer output: "
            f"{output_gpkg} ({clean_output_layer}) from {log_gpkg}:{clean_source_layer}"
        )
    else:
        if bool(args.skip_final_gap_fill) and not bool(args.skip_final_alias):
            _write_final_alias(log_gpkg, FINAL_SOURCE_LAYER, clean_output_layer)
        _write_single_layer_output(
            source_gpkg=log_gpkg,
            source_layer=clean_source_layer,
            output_gpkg=output_gpkg,
            output_layer=clean_output_layer,
        )
        summary = {
            "pipeline": "wfs_merge_native",
            "production_inputs": "raw_wfs+uprn+models",
            "reference_enabled": bool(reference_enabled),
            "reference_gpkg": reference_gpkg if reference_enabled else None,
            "reference_layer": str(args.reference_layer) if reference_enabled else None,
            "output_gpkg": str(output_gpkg),
            "log_gpkg": str(log_gpkg),
            "final_layer": None if bool(args.skip_final_alias) else clean_output_layer,
            "clean_output_layer": clean_output_layer,
            "native_final_source_layer": FINAL_SOURCE_LAYER,
            "final_gap_fill_enabled": bool(not args.skip_final_gap_fill),
            "work_dir": str(work_dir),
            "stages": {
                "edge": _read_json(edge_gpkg.with_suffix(".summary.json")),
                "completion": _read_json(completion_gpkg.with_suffix(".summary.json")),
                "operation": _read_json(operation_gpkg.with_suffix(".summary.json")),
                "overmerge_split": {}
                if bool(args.skip_overmerge_split)
                else _read_json(overmerge_split_gpkg.with_suffix(".summary.json")),
                "anchor_group_repair": _read_json(anchor_group_gpkg.with_suffix(".summary.json")),
                "final_gap_fill": {}
                if bool(args.skip_final_gap_fill)
                else _read_json(log_gpkg.with_suffix(".summary.json")),
            },
        }
        pipeline_summary = log_gpkg.with_suffix(".pipeline_summary.json")
        pipeline_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log(f"[DONE] Final output: {output_gpkg}")
        _log(f"[DONE] Final output layer: {clean_output_layer}")
        _log(f"[DONE] Full log output: {log_gpkg}")
        _log(f"[DONE] Pipeline summary: {pipeline_summary}")


if __name__ == "__main__":
    main()

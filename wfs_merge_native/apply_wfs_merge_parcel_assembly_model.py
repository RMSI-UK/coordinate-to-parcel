#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import geopandas as gpd
import joblib
import pandas as pd

from apply_wfs_merge_geometry_completion import _apply_repairs, _read_optional_layer, _write_layer
from parcel_assembly_features import build_parcel_assembly_candidates, log, parse_fid_groups


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_anchor_group_repaired_threshold_085_gate_096.gpkg"
)
DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_parcel_assembly_v1/"
    "parcel_assembly_model_v1.joblib"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_parcel_assembly_v1/"
    "model_predicted_polygons_parcel_assembly_v1.gpkg"
)


def _select_assembly_candidates(
    candidates: gpd.GeoDataFrame,
    *,
    threshold: float,
    review_threshold: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if candidates.empty:
        return candidates.copy(), candidates.copy(), candidates.copy()
    high = candidates[candidates["parcel_assembly_proba"].ge(float(threshold))].copy()
    review = candidates[
        candidates["parcel_assembly_proba"].ge(float(review_threshold))
        & candidates["parcel_assembly_proba"].lt(float(threshold))
    ].copy()
    if high.empty:
        return high, review.sort_values("parcel_assembly_proba", ascending=False), high.copy()
    high = high.sort_values(
        [
            "parcel_assembly_proba",
            "group_size",
            "internal_shared_len",
            "group_regularity_score",
            "group_hull_gap_ratio",
        ],
        ascending=[False, False, False, False, True],
    )
    selected_rows: list[pd.Series] = []
    conflict_rows: list[pd.Series] = []
    used_fids: set[int] = set()
    for _, row in high.iterrows():
        fids = {int(part) for part in str(row.candidate_fids).split("|") if part}
        if fids & used_fids:
            conflict_rows.append(row)
            continue
        selected_rows.append(row)
        used_fids |= fids
    selected = (
        gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=candidates.crs).reset_index(drop=True)
        if selected_rows
        else high.iloc[0:0].copy()
    )
    conflicts = (
        gpd.GeoDataFrame(conflict_rows, geometry="geometry", crs=candidates.crs).reset_index(drop=True)
        if conflict_rows
        else high.iloc[0:0].copy()
    )
    return selected, review.sort_values("parcel_assembly_proba", ascending=False), conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the unified parcel assembly model to component groups.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--review-threshold", type=float, default=0.75)
    parser.add_argument("--max-seed-area", type=float, default=None)
    parser.add_argument("--max-after-area", type=float, default=None)
    parser.add_argument("--max-pair-area", type=float, default=None)
    parser.add_argument("--max-group-size", type=int, default=None)
    parser.add_argument("--top-neighbors", type=int, default=None)
    parser.add_argument("--per-seed-limit", type=int, default=None)
    parser.add_argument("--max-candidate-groups", type=int, default=None)
    parser.add_argument("--min-shared-edge", type=float, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=None)
    parser.add_argument("--include-all-under-area", action="store_true")
    parser.add_argument("--manual-positive-fid-groups", default="")
    parser.add_argument("--top-candidate-layer-limit", type=int, default=20000)
    return parser.parse_args()


def _param(args: argparse.Namespace, params: dict, name: str, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return params.get(name.replace("_", "-"), params.get(name, default))


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    model_path = Path(args.model)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    feature_cols = payload.get("feature_cols") if isinstance(payload, dict) else None
    training_params = payload.get("training_params", {}) if isinstance(payload, dict) else {}
    if not feature_cols:
        raise RuntimeError("Parcel assembly model must provide feature_cols in its joblib payload.")
    manual_positive_groups = parse_fid_groups(args.manual_positive_fid_groups)
    include_all = bool(args.include_all_under_area or training_params.get("include_all_under_area", False))

    candidates, edges = build_parcel_assembly_candidates(
        input_gpkg=input_gpkg,
        max_seed_area=float(_param(args, training_params, "max_seed_area", 2000.0)),
        max_after_area=float(_param(args, training_params, "max_after_area", 2000.0)),
        max_pair_area=float(_param(args, training_params, "max_pair_area", 2000.0)),
        max_group_size=int(_param(args, training_params, "max_group_size", 6)),
        top_neighbors=int(_param(args, training_params, "top_neighbors", 8)),
        per_seed_limit=int(_param(args, training_params, "per_seed_limit", 24)),
        max_candidate_groups=int(_param(args, training_params, "max_candidate_groups", 250000)),
        min_shared_edge=float(_param(args, training_params, "min_shared_edge", 0.2)),
        query_chunk_size=int(_param(args, training_params, "query_chunk_size", 5000)),
        include_all_under_area=include_all,
        manual_positive_fid_groups=manual_positive_groups,
        include_labels=False,
    )
    if candidates.empty:
        raise RuntimeError("No parcel assembly candidates were generated.")
    missing = sorted(set(feature_cols) - set(candidates.columns))
    if missing:
        raise RuntimeError(f"Parcel assembly candidates are missing model features: {missing}")
    candidates["parcel_assembly_proba"] = pipeline.predict_proba(candidates[feature_cols])[:, 1]
    selected, review, conflicts = _select_assembly_candidates(
        candidates,
        threshold=float(args.threshold),
        review_threshold=float(args.review_threshold),
    )
    log(
        "[INFO] Parcel assembly candidates="
        f"{len(candidates):,}; selected={len(selected):,}; review={len(review):,}; conflicts={len(conflicts):,}"
    )

    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio", fid_as_index=True)
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted.index = predicted.index.astype(int)
    predicted["layer_fid"] = predicted.index.astype(int)
    selected_for_apply = selected.copy()
    selected_for_apply["geometry_completion_score"] = selected_for_apply["parcel_assembly_proba"].astype(float)
    sources_new, edges_new, predicted_new = _apply_repairs(
        input_gpkg=input_gpkg,
        output_gpkg=output_gpkg,
        predicted=predicted,
        selected=selected_for_apply,
    )

    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()

    log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(predicted_new, output_gpkg, "predicted_parcels_with_uprn")
    _write_layer(merged_only.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels_merged_only")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only_with_uprn")
    _write_layer(possible_fp.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_false_positive_clusters")
    _write_layer(possible_split.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_split_reference_clusters")
    _write_layer(sources_new, output_gpkg, "prediction_source_polygons")
    _write_layer(edges_new, output_gpkg, "predicted_positive_edges")
    _write_layer(selected, output_gpkg, "parcel_assembly_selected")
    _write_layer(review.head(int(args.top_candidate_layer_limit)), output_gpkg, "parcel_assembly_review_candidates")
    _write_layer(conflicts.head(int(args.top_candidate_layer_limit)), output_gpkg, "parcel_assembly_conflicts")
    top_candidates = candidates.sort_values("parcel_assembly_proba", ascending=False).head(
        int(args.top_candidate_layer_limit)
    )
    _write_layer(top_candidates, output_gpkg, "parcel_assembly_candidates_top")

    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", predicted.crs)
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", predicted.crs)
    if not semantic_reference.empty:
        _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    if not excluded_problem_sources.empty:
        _write_layer(excluded_problem_sources, output_gpkg, "excluded_problem_sources")

    summary = {
        "input_gpkg": str(input_gpkg),
        "model": str(model_path),
        "output_gpkg": str(output_gpkg),
        "threshold": float(args.threshold),
        "review_threshold": float(args.review_threshold),
        "edge_rows": int(len(edges)),
        "candidate_rows": int(len(candidates)),
        "selected_groups": int(len(selected)),
        "review_groups": int(len(review)),
        "conflict_groups": int(len(conflicts)),
        "old_predicted_rows": int(len(predicted)),
        "new_predicted_rows": int(len(predicted_new)),
        "old_possible_split_rows": int(predicted["possible_split_reference"].fillna(0).astype(int).sum()),
        "new_possible_split_rows": int(predicted_new["possible_split_reference"].fillna(0).astype(int).sum()),
        "old_possible_false_positive_rows": int(
            predicted["possible_false_positive_cluster"].fillna(0).astype(int).sum()
        ),
        "new_possible_false_positive_rows": int(
            predicted_new["possible_false_positive_cluster"].fillna(0).astype(int).sum()
        ),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    selected.drop(columns="geometry").to_csv(output_gpkg.with_suffix(".selected.csv"), index=False)
    top_candidates.drop(columns="geometry").to_csv(output_gpkg.with_suffix(".candidates_top.csv"), index=False)
    log(json.dumps(summary, indent=2))
    log("[DONE] Parcel assembly apply complete")


if __name__ == "__main__":
    main()

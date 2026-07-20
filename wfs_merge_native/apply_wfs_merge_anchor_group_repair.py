#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import LineString

from apply_wfs_merge_completion_model import _build_predicted_parcels, _write_layer
from apply_wfs_merge_anchor_repair import build_anchor_candidates, score_anchor_candidates
from apply_wfs_merge_operation_pipeline import _filter_edges_to_current_components
from train_wfs_merge_anchor_group_repair_model import (
    DEFAULT_INPUT_GPKG,
    DEFAULT_PAIR_CANDIDATE_CSV,
    MODEL_FILE_NAME,
    build_anchor_need_candidates,
    build_anchor_group_candidates,
)


DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_train_anchor_group_final_selector_light/"
    f"{MODEL_FILE_NAME}"
)
DEFAULT_EDGE_CSV = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "edge_candidate_predictions_full.csv"
)
DEFAULT_PAIR_ANCHOR_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "anchor_problem_detection_model_probe_v1.joblib"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_anchor_group_repaired_final_selector.gpkg"
)
PASSTHROUGH_DEBUG_LAYERS = [
    "prune_removed_sources",
    "prune_candidate_debug",
    "zero_uprn_attachment_candidates",
    "neighborhood_overmerge_split_components",
    "neighborhood_overmerge_split_removed_edges",
    "neighborhood_overmerge_split_results",
    "local_mode_split_candidates",
    "local_mode_split_selected",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _read_optional_layer(path: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _parse_ids(value: object) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _line_between_components(
    anchor_component_id: int,
    zero_component_id: int,
    geom_by_component: dict[int, Any],
) -> LineString:
    anchor_point = geom_by_component[int(anchor_component_id)].representative_point()
    zero_point = geom_by_component[int(zero_component_id)].representative_point()
    return LineString(
        [
            (shapely.get_x(anchor_point), shapely.get_y(anchor_point)),
            (shapely.get_x(zero_point), shapely.get_y(zero_point)),
        ]
    )


def _candidate_union_geometry(row: pd.Series, geom_by_component: dict[int, Any]):
    geoms = [geom_by_component[int(row["anchor_component_id"])]]
    geoms.extend(geom_by_component[int(comp_id)] for comp_id in _parse_ids(row["zero_component_ids"]))
    return shapely.union_all(geoms)


def _candidate_geometries(
    rows: pd.DataFrame,
    geom_by_component: dict[int, Any],
    crs,
) -> gpd.GeoDataFrame:
    if rows.empty:
        return gpd.GeoDataFrame(rows.copy(), geometry=[], crs=crs)
    geoms = [_candidate_union_geometry(row, geom_by_component) for _, row in rows.iterrows()]
    return gpd.GeoDataFrame(rows.copy(), geometry=geoms, crs=crs)


def _anchor_need_geometries(
    rows: pd.DataFrame,
    geom_by_component: dict[int, Any],
    crs,
) -> gpd.GeoDataFrame:
    if rows.empty:
        return gpd.GeoDataFrame(rows.copy(), geometry=[], crs=crs)
    out = rows.copy()
    geoms = [geom_by_component.get(int(anchor_id)) for anchor_id in out["anchor_component_id"].astype(int)]
    out = out[[geom is not None for geom in geoms]].copy()
    geoms = [geom for geom in geoms if geom is not None]
    return gpd.GeoDataFrame(out, geometry=geoms, crs=crs)


def _select_group_repairs(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    review_threshold: float,
    enable_residual_fallback: bool,
    residual_min_pair_proba: float,
    residual_min_shared_edge: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high = candidates[candidates["anchor_group_repair_proba"].ge(float(threshold))].copy()
    review = candidates[
        candidates["anchor_group_repair_proba"].ge(float(review_threshold))
        & candidates["anchor_group_repair_proba"].lt(float(threshold))
    ].copy()
    if high.empty:
        return high, review, high.copy()
    high["anchor_group_repair_residual_fallback"] = 0
    high["anchor_group_repair_parent_proba"] = np.nan

    residual_lookup: dict[tuple[int, tuple[int, ...]], pd.Series] = {}
    if bool(enable_residual_fallback):
        candidates_sorted = candidates.sort_values("anchor_group_repair_proba", ascending=False)
        for _, row in candidates_sorted.iterrows():
            anchor_id = int(row["anchor_component_id"])
            zero_key = tuple(sorted(_parse_ids(row["zero_component_ids"])))
            if not zero_key:
                continue
            residual_lookup.setdefault((anchor_id, zero_key), row)

    high["_complete_pool_candidate"] = (
        high.get("group_zero_fraction_of_pool", pd.Series(0.0, index=high.index)).fillna(0.0).astype(float).ge(0.999)
        & high.get("omitted_zero_component_count", pd.Series(999.0, index=high.index)).fillna(999.0).astype(float).le(0.0)
    ).astype(int)
    high["_uprn_skeleton_complete_candidate"] = (
        high.get("uprn_skeleton_complete_pool", pd.Series(0.0, index=high.index)).fillna(0.0).astype(float).ge(0.5)
        & high.get(
            "group_zero_uprn_land_building_component_count",
            pd.Series(0.0, index=high.index),
        ).fillna(0.0).astype(float).ge(1.0)
        & high.get(
            "omitted_zero_uprn_land_building_component_count",
            pd.Series(999.0, index=high.index),
        ).fillna(999.0).astype(float).le(0.0)
    ).astype(int)
    high["_omitted_strong_zero_count_sort"] = high.get(
        "omitted_strong_zero_count",
        pd.Series(999.0, index=high.index),
    ).fillna(999.0).astype(float)
    high["_omitted_uprn_skeleton_count_sort"] = high.get(
        "omitted_zero_uprn_land_building_component_count",
        pd.Series(999.0, index=high.index),
    ).fillna(999.0).astype(float)
    high = high.sort_values(
        [
            "anchor_component_id",
            "_uprn_skeleton_complete_candidate",
            "_complete_pool_candidate",
            "_omitted_uprn_skeleton_count_sort",
            "_omitted_strong_zero_count_sort",
            "group_zero_uprn_land_building_fraction_of_pool",
            "group_zero_fraction_of_pool",
            "group_zero_component_count",
            "anchor_group_repair_proba",
            "after_regularity_score",
            "after_mrr_ratio",
            "after_hull_gap_ratio",
        ],
        ascending=[True, False, False, True, True, False, False, False, False, False, False, True],
    ).drop_duplicates("anchor_component_id", keep="first")
    high = high.sort_values(
        [
            "_uprn_skeleton_complete_candidate",
            "_complete_pool_candidate",
            "_omitted_uprn_skeleton_count_sort",
            "_omitted_strong_zero_count_sort",
            "anchor_group_repair_proba",
            "group_zero_component_count",
            "after_regularity_score",
            "after_mrr_ratio",
            "after_hull_gap_ratio",
        ],
        ascending=[False, False, True, True, False, False, False, False, True],
    ).copy()

    kept_rows: list[pd.Series] = []
    skipped_rows: list[pd.Series] = []
    used_anchors: set[int] = set()
    used_zero_components: set[int] = set()
    for _, row in high.iterrows():
        anchor_id = int(row["anchor_component_id"])
        zero_ids = _parse_ids(row["zero_component_ids"])
        if anchor_id in used_anchors:
            skipped_rows.append(row)
            continue
        conflicting_zero_ids = zero_ids & used_zero_components
        if conflicting_zero_ids:
            residual_zero_ids = zero_ids - used_zero_components
            residual_key = tuple(sorted(residual_zero_ids))
            residual_row = residual_lookup.get((anchor_id, residual_key)) if residual_key else None
            if residual_row is not None:
                residual_zero_set = _parse_ids(residual_row["zero_component_ids"])
                residual_pair_ok = (
                    float(residual_row.get("pair_repair_proba_min", 0.0) or 0.0)
                    >= float(residual_min_pair_proba)
                    and float(residual_row.get("pair_shared_edge_sum", 0.0) or 0.0)
                    >= float(residual_min_shared_edge)
                    and float(residual_row.get("after_anchor_building_source_count", 0.0) or 0.0) <= 1.0
                )
                if residual_zero_set and not (residual_zero_set & used_zero_components) and residual_pair_ok:
                    fallback_row = residual_row.copy()
                    fallback_row["anchor_group_repair_residual_fallback"] = 1
                    fallback_row["anchor_group_repair_parent_proba"] = float(row["anchor_group_repair_proba"])
                    fallback_row["anchor_group_repair_proba"] = max(
                        float(fallback_row.get("anchor_group_repair_proba", 0.0) or 0.0),
                        float(threshold) + 1e-6,
                    )
                    fallback_row["anchor_group_repair_pred_at_threshold"] = 1
                    used_anchors.add(anchor_id)
                    used_zero_components |= residual_zero_set
                    kept_rows.append(fallback_row)
            skipped_rows.append(row)
            continue
        used_anchors.add(anchor_id)
        used_zero_components |= zero_ids
        kept_rows.append(row)

    selected = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else high.iloc[0:0].copy()
    conflicts = pd.DataFrame(skipped_rows).reset_index(drop=True) if skipped_rows else high.iloc[0:0].copy()
    return selected, review.reset_index(drop=True), conflicts


def _repair_edges(
    selected: pd.DataFrame,
    sources: gpd.GeoDataFrame,
    geom_by_component: dict[int, Any],
) -> gpd.GeoDataFrame:
    if selected.empty:
        return gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    source_groups = sources.groupby(sources["pred_component_id"].astype(int))
    first_source_by_component = {
        int(comp_id): int(group["source_fid"].astype(int).iloc[0])
        for comp_id, group in source_groups
    }
    if "reference_merge_fid" in sources.columns:
        source_ref = sources.set_index(sources["source_fid"].astype(int))["reference_merge_fid"]
    else:
        source_ref = pd.Series(dtype="float64")
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        anchor_id = int(row.anchor_component_id)
        left_source = int(first_source_by_component.get(anchor_id, -1))
        for zero_id in _parse_ids(row.zero_component_ids):
            right_source = int(first_source_by_component.get(int(zero_id), -1))
            records.append(
                {
                    "left_source_fid": left_source,
                    "right_source_fid": right_source,
                    "left_merge_fid": int(source_ref.get(left_source, -1))
                    if pd.notna(source_ref.get(left_source, np.nan))
                    else -1,
                    "right_merge_fid": int(source_ref.get(right_source, -1))
                    if pd.notna(source_ref.get(right_source, np.nan))
                    else -1,
                    "label": int(getattr(row, "label", -1)),
                    "model_proba": float(getattr(row, "pair_repair_proba_max", np.nan)),
                    "completion_proba": float(row.anchor_group_repair_proba),
                    "role_pair": str(getattr(row, "role_pair_signature", "")),
                    "pred_component_id": anchor_id,
                    "model_stage": "anchor_group_repair",
                    "geometry": _line_between_components(anchor_id, int(zero_id), geom_by_component),
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the UPRN-anchor group repair model.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--pair-candidate-csv", default=DEFAULT_PAIR_CANDIDATE_CSV)
    parser.add_argument("--edge-csv", default=DEFAULT_EDGE_CSV)
    parser.add_argument("--pair-anchor-model", default=DEFAULT_PAIR_ANCHOR_MODEL)
    parser.add_argument("--force-rebuild-pair-candidates", action="store_true")
    parser.add_argument("--pair-candidate-chunksize", type=int, default=250000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--candidate-csv", default="")
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--candidate-strategy", choices=["full", "light"], default="light")
    parser.add_argument("--enclosure-level", choices=["component", "source", "pair", "none"], default="pair")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--review-threshold", type=float, default=None)
    parser.add_argument("--anchor-need-threshold", type=float, default=0.94)
    parser.add_argument("--complete-pool-gate-bypass-threshold", type=float, default=0.999)
    parser.add_argument("--uprn-skeleton-gate-bypass-threshold", type=float, default=0.93)
    parser.add_argument("--uprn-skeleton-gate-bypass-min-group-count", type=int, default=2)
    parser.add_argument("--uprn-skeleton-gate-bypass-min-regularity", type=float, default=0.98)
    parser.add_argument("--uprn-skeleton-gate-bypass-max-after-area", type=float, default=750.0)
    parser.add_argument("--uprn-skeleton-pair-override-min-pair-proba", type=float, default=0.93)
    parser.add_argument("--uprn-skeleton-pair-override-min-regularity", type=float, default=0.93)
    parser.add_argument("--uprn-skeleton-pair-override-max-after-area", type=float, default=500.0)
    parser.add_argument("--uprn-skeleton-pair-override-max-zero-area-ratio", type=float, default=2.0)
    parser.add_argument("--disable-anchor-gate", action="store_true", default=False)
    parser.add_argument("--enable-anchor-gate", dest="disable_anchor_gate", action="store_false")
    parser.add_argument("--disable-residual-fallback", action="store_true")
    parser.add_argument("--residual-fallback-min-pair-proba", type=float, default=0.90)
    parser.add_argument("--residual-fallback-min-shared-edge", type=float, default=6.0)
    parser.add_argument("--top-candidate-layer-limit", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    pair_candidate_csv = Path(args.pair_candidate_csv)
    model_path = Path(args.model)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    feature_cols = payload.get("feature_cols") if isinstance(payload, dict) else None
    model_kind = payload.get("model_kind", "anchor_group_repair_with_gate") if isinstance(payload, dict) else "legacy"
    anchor_gate = payload.get("anchor_gate_pipeline") if isinstance(payload, dict) else None
    anchor_gate_feature_cols = payload.get("anchor_gate_feature_cols") if isinstance(payload, dict) else None
    params = payload.get("training_params", {}) if isinstance(payload, dict) else {}
    if not feature_cols:
        raise RuntimeError("Anchor group repair model must provide feature_cols in its joblib payload.")
    default_threshold = (
        float(params.get("threshold", 0.94))
        if str(model_kind) == "anchor_group_final_selector"
        else 0.78
    )
    threshold = default_threshold if args.threshold is None else float(args.threshold)
    review_threshold = (
        max(0.0, threshold - 0.03)
        if args.review_threshold is None
        else float(args.review_threshold)
    )

    if bool(args.force_rebuild_pair_candidates) or not pair_candidate_csv.exists():
        if not str(args.edge_csv).strip():
            raise RuntimeError("--edge-csv is required when --pair-candidate-csv does not exist.")
        _log(f"[INFO] Building pair anchor candidates: {pair_candidate_csv}")
        pair_candidate_csv.parent.mkdir(parents=True, exist_ok=True)
        pair_candidates, _, _, _ = build_anchor_candidates(
            input_gpkg=input_gpkg,
            edge_csv=Path(args.edge_csv),
            max_edge_chunksize=int(args.pair_candidate_chunksize),
        )
        pair_candidates = score_anchor_candidates(pair_candidates, Path(args.pair_anchor_model))
        pair_candidates.to_csv(pair_candidate_csv, index=False)

    if str(args.candidate_input_csv).strip():
        _log(f"[INFO] Reusing anchor group candidates: {args.candidate_input_csv}")
        candidates = pd.read_csv(args.candidate_input_csv)
        reuse_existing_scores = (
            str(model_kind) != "anchor_group_final_selector"
            and "anchor_group_repair_proba" in candidates.columns
        )
        if not reuse_existing_scores:
            missing = sorted(set(feature_cols) - set(candidates.columns))
            if missing:
                raise RuntimeError(f"Anchor group candidates are missing model features: {missing}")
            candidates["anchor_group_repair_proba"] = pipeline.predict_proba(candidates[feature_cols])[:, 1]
        candidates["anchor_group_repair_model_proba"] = candidates["anchor_group_repair_proba"].astype(float)
        candidates["anchor_group_repair_pred_at_threshold"] = candidates["anchor_group_repair_proba"].ge(
            float(threshold)
        ).astype(int)
    else:
        _log("[INFO] Building anchor group candidates")
        try:
            candidates = build_anchor_group_candidates(
                input_gpkg=input_gpkg,
                pair_candidate_csv=pair_candidate_csv,
                top_zero_neighbors=int(params.get("top_zero_neighbors", 6)),
                max_group_size=int(params.get("max_group_size", 6)),
                max_anchor_area=float(params.get("max_anchor_area", 2000.0)),
                max_zero_area=float(params.get("max_zero_area", 1000.0)),
                max_after_area=float(params.get("max_after_area", 2000.0)),
                max_zero_source_count=int(params.get("max_zero_source_count", 8)),
                candidate_strategy=str(args.candidate_strategy),
                enclosure_level=str(args.enclosure_level),
                manual_positive_groups={},
            )
        except RuntimeError as exc:
            if "No pair candidates remain after group repair filters" not in str(exc):
                raise
            _log("[INFO] No anchor group candidates; carrying prediction forward")
            candidates = pd.DataFrame()
        if not candidates.empty:
            missing = sorted(set(feature_cols) - set(candidates.columns))
            if missing:
                raise RuntimeError(f"Anchor group candidates are missing model features: {missing}")
            candidates["anchor_group_repair_proba"] = pipeline.predict_proba(candidates[feature_cols])[:, 1]
            candidates["anchor_group_repair_model_proba"] = candidates["anchor_group_repair_proba"].astype(float)
            candidates["anchor_group_repair_pred_at_threshold"] = candidates["anchor_group_repair_proba"].ge(
                float(threshold)
            ).astype(int)

    if candidates.empty:
        candidates = candidates.copy()
        candidates["anchor_group_repair_proba"] = pd.Series(dtype="float64")
        candidates["anchor_group_repair_model_proba"] = pd.Series(dtype="float64")
        candidates["anchor_group_repair_pred_at_threshold"] = pd.Series(dtype="int64")

    anchor_need = pd.DataFrame()
    anchor_gate_allowed = (
        str(model_kind) != "anchor_group_final_selector"
        and not args.disable_anchor_gate
        and anchor_gate is not None
        and anchor_gate_feature_cols
    )
    if not candidates.empty and anchor_gate_allowed:
        has_reusable_anchor_gate = (
            "anchor_need_repair_proba" in candidates.columns
            and candidates["anchor_need_repair_proba"].notna().any()
        )
        if not has_reusable_anchor_gate:
            _log("[INFO] Scoring anchor need-repair gate")
            candidates = candidates.drop(columns=["anchor_need_repair_proba"], errors="ignore")
            anchor_need = build_anchor_need_candidates(
                pair_candidate_csv=pair_candidate_csv,
                max_anchor_area=float(params.get("max_anchor_area", 2000.0)),
                max_zero_area=float(params.get("max_zero_area", 1000.0)),
                max_after_area=float(params.get("max_after_area", 2000.0)),
                max_zero_source_count=int(params.get("max_zero_source_count", 8)),
                manual_positive_groups={},
            )
            missing_gate = sorted(set(anchor_gate_feature_cols) - set(anchor_need.columns))
            if missing_gate:
                raise RuntimeError(f"Anchor gate candidates are missing model features: {missing_gate}")
            anchor_need["anchor_need_repair_proba"] = anchor_gate.predict_proba(
                anchor_need[anchor_gate_feature_cols]
            )[:, 1]
            candidates = candidates.merge(
                anchor_need[["anchor_component_id", "anchor_need_repair_proba"]],
                on="anchor_component_id",
                how="left",
            )
        else:
            _log("[INFO] Reusing anchor need-repair gate scores from candidate CSV")
        candidates["anchor_need_repair_proba"] = candidates["anchor_need_repair_proba"].fillna(0.0)
        skeleton_complete = (
            candidates.get("uprn_skeleton_complete_pool", pd.Series(0.0, index=candidates.index))
            .fillna(0.0)
            .astype(float)
            .ge(0.5)
            & candidates.get(
                "group_zero_uprn_land_building_component_count",
                pd.Series(0.0, index=candidates.index),
            )
            .fillna(0.0)
            .astype(float)
            .ge(1.0)
            & candidates.get(
                "omitted_zero_uprn_land_building_component_count",
                pd.Series(999.0, index=candidates.index),
            )
            .fillna(999.0)
            .astype(float)
            .le(0.0)
        )
        skeleton_pair_override = (
            skeleton_complete
            & candidates["group_zero_component_count"].fillna(0).astype(int).eq(1)
            & candidates["anchor_need_repair_proba"].ge(float(args.anchor_need_threshold))
            & candidates.get("pair_repair_proba_min", pd.Series(0.0, index=candidates.index))
            .fillna(0.0)
            .astype(float)
            .ge(float(args.uprn_skeleton_pair_override_min_pair_proba))
            & candidates.get("after_area", pd.Series(np.inf, index=candidates.index))
            .fillna(np.inf)
            .astype(float)
            .le(float(args.uprn_skeleton_pair_override_max_after_area))
            & candidates.get("after_regularity_score", pd.Series(0.0, index=candidates.index))
            .fillna(0.0)
            .astype(float)
            .ge(float(args.uprn_skeleton_pair_override_min_regularity))
            & candidates.get("zero_area_ratio_to_anchor", pd.Series(np.inf, index=candidates.index))
            .fillna(np.inf)
            .astype(float)
            .le(float(args.uprn_skeleton_pair_override_max_zero_area_ratio))
            & candidates["anchor_group_repair_model_proba"].astype(float).lt(float(threshold))
        )
        candidates["anchor_group_repair_skeleton_pair_override"] = skeleton_pair_override.astype(int)
        if skeleton_pair_override.any():
            candidates.loc[skeleton_pair_override, "anchor_group_repair_proba"] = np.maximum(
                candidates.loc[skeleton_pair_override, "anchor_group_repair_proba"].astype(float),
                float(threshold) + 1e-6,
            )
        candidates["anchor_group_repair_pred_at_threshold"] = candidates["anchor_group_repair_proba"].ge(
            float(threshold)
        ).astype(int)
        candidates["anchor_gate_bypass_complete_pool"] = (
            candidates["anchor_group_repair_proba"].astype(float).ge(float(args.complete_pool_gate_bypass_threshold))
            & candidates.get("group_zero_fraction_of_pool", pd.Series(0.0, index=candidates.index))
            .fillna(0.0)
            .astype(float)
            .ge(0.999)
            & candidates.get("omitted_zero_component_count", pd.Series(999.0, index=candidates.index))
            .fillna(999.0)
            .astype(float)
            .le(0.0)
            & candidates["group_zero_component_count"].fillna(0).astype(int).ge(2)
        ).astype(int)
        candidates["anchor_gate_bypass_uprn_skeleton"] = (
            skeleton_complete
            & candidates["anchor_group_repair_proba"].astype(float).ge(
                float(args.uprn_skeleton_gate_bypass_threshold)
            )
            & candidates["group_zero_component_count"]
            .fillna(0)
            .astype(int)
            .ge(int(args.uprn_skeleton_gate_bypass_min_group_count))
            & candidates.get("after_regularity_score", pd.Series(0.0, index=candidates.index))
            .fillna(0.0)
            .astype(float)
            .ge(float(args.uprn_skeleton_gate_bypass_min_regularity))
            & candidates.get("after_area", pd.Series(np.inf, index=candidates.index))
            .fillna(np.inf)
            .astype(float)
            .le(float(args.uprn_skeleton_gate_bypass_max_after_area))
        ).astype(int)
        candidates_for_selection = candidates[
            candidates["anchor_need_repair_proba"].ge(float(args.anchor_need_threshold))
            | candidates["anchor_gate_bypass_complete_pool"].eq(1)
            | candidates["anchor_gate_bypass_uprn_skeleton"].eq(1)
        ].copy()
    else:
        candidates["anchor_need_repair_proba"] = np.nan
        candidates["anchor_gate_bypass_complete_pool"] = 0
        candidates["anchor_gate_bypass_uprn_skeleton"] = 0
        candidates["anchor_group_repair_skeleton_pair_override"] = 0
        candidates_for_selection = candidates.copy()

    if candidates_for_selection.empty:
        selected = candidates_for_selection.copy()
        review = candidates_for_selection.copy()
        conflicts = candidates_for_selection.copy()
    else:
        selected, review, conflicts = _select_group_repairs(
            candidates_for_selection,
            threshold=float(threshold),
            review_threshold=float(review_threshold),
            enable_residual_fallback=not bool(args.disable_residual_fallback),
            residual_min_pair_proba=float(args.residual_fallback_min_pair_proba),
            residual_min_shared_edge=float(args.residual_fallback_min_shared_edge),
        )
    _log(
        "[INFO] Anchor group candidates="
        f"{len(candidates):,}; selected_groups={len(selected):,}; "
        f"review={len(review):,}; conflicts={len(conflicts):,}"
    )

    predicted = pyogrio.read_dataframe(input_gpkg, layer="predicted_parcels_with_uprn")
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted["pred_component_id"] = predicted["pred_component_id"].astype(int)
    old_geom_by_component = predicted.set_index("pred_component_id").geometry.to_dict()

    sources = pyogrio.read_dataframe(input_gpkg, layer="prediction_source_polygons")
    sources = sources[sources.geometry.notna() & ~sources.geometry.is_empty].copy()
    sources["pred_component_id"] = sources["pred_component_id"].astype(int)
    sources_new = sources.copy()
    sources_new["anchor_group_repair_status"] = "base"
    sources_new["anchor_group_repaired_to_component"] = np.nan
    sources_new["anchor_group_repair_proba"] = np.nan
    sources_new["anchor_group_repair_from_components"] = ""

    for row in selected.itertuples(index=False):
        anchor_id = int(row.anchor_component_id)
        zero_ids = _parse_ids(row.zero_component_ids)
        mask = sources_new["pred_component_id"].astype(int).isin(zero_ids)
        sources_new.loc[mask, "anchor_group_repair_status"] = "anchor_group_repaired"
        sources_new.loc[mask, "anchor_group_repaired_to_component"] = anchor_id
        sources_new.loc[mask, "anchor_group_repair_from_components"] = row.zero_component_ids
        sources_new.loc[mask, "anchor_group_repair_proba"] = float(row.anchor_group_repair_proba)
        sources_new.loc[mask, "pred_component_id"] = anchor_id

    edges = pyogrio.read_dataframe(input_gpkg, layer="predicted_positive_edges")
    edges_current = _filter_edges_to_current_components(edges, sources_new)
    repair_edges = _repair_edges(selected, sources, old_geom_by_component)
    if repair_edges.empty:
        edges_new = edges_current.copy()
    else:
        edges_new = pd.concat([edges_current, repair_edges], ignore_index=True)
        edges_new = gpd.GeoDataFrame(edges_new, geometry="geometry", crs=sources.crs)

    predicted_new = _build_predicted_parcels(sources_new, edges_new, predicted)
    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()

    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", sources.crs)
    excluded_gapfill_council_sources = _read_optional_layer(
        input_gpkg,
        "excluded_gapfill_council_sources",
        sources.crs,
    )
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", sources.crs)
    geom_by_component_new = predicted_new.set_index(predicted_new["pred_component_id"].astype(int)).geometry.to_dict()

    candidate_csv = (
        Path(args.candidate_csv)
        if str(args.candidate_csv).strip()
        else output_gpkg.with_suffix(".anchor_group_candidates.csv")
    )
    _log(f"[INFO] Writing candidate CSV: {candidate_csv}")
    candidates.to_csv(candidate_csv, index=False)

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(merged_only.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels_merged_only")
    _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    _write_layer(sources_new, output_gpkg, "prediction_source_polygons")
    _write_layer(excluded_gapfill_council_sources, output_gpkg, "excluded_gapfill_council_sources")
    _write_layer(excluded_problem_sources, output_gpkg, "excluded_problem_sources")
    _write_layer(edges_new, output_gpkg, "predicted_positive_edges")
    _write_layer(possible_fp.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_false_positive_clusters")
    _write_layer(possible_split.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_split_reference_clusters")
    _write_layer(predicted_new, output_gpkg, "predicted_parcels_with_uprn")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only_with_uprn")
    _write_layer(possible_fp, output_gpkg, "possible_false_positive_clusters_with_uprn")
    _write_layer(possible_split, output_gpkg, "possible_split_reference_clusters_with_uprn")
    _write_layer(merged_only[merged_only["pred_uprn_count"].le(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_le1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].eq(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_eq1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].gt(1)].copy(), output_gpkg, "predicted_parcels_merged_only_multi_uprn")
    for layer in PASSTHROUGH_DEBUG_LAYERS:
        passthrough = _read_optional_layer(input_gpkg, layer, sources.crs)
        if not passthrough.empty:
            _write_layer(passthrough, output_gpkg, layer)
    _write_layer(_candidate_geometries(selected, old_geom_by_component, sources.crs), output_gpkg, "anchor_group_repair_selected")
    _write_layer(_candidate_geometries(review, old_geom_by_component, sources.crs), output_gpkg, "anchor_group_repair_review_candidates")
    _write_layer(_candidate_geometries(conflicts, old_geom_by_component, sources.crs), output_gpkg, "anchor_group_repair_conflicts")
    top_candidates = candidates.sort_values("anchor_group_repair_proba", ascending=False).head(
        int(args.top_candidate_layer_limit)
    ).copy()
    _write_layer(_candidate_geometries(top_candidates, old_geom_by_component, sources.crs), output_gpkg, "anchor_group_repair_candidates_top")
    if not anchor_need.empty:
        top_anchor_need = anchor_need.sort_values("anchor_need_repair_proba", ascending=False).head(
            int(args.top_candidate_layer_limit)
        ).copy()
        _write_layer(_anchor_need_geometries(top_anchor_need, old_geom_by_component, sources.crs), output_gpkg, "anchor_need_repair_candidates_top")

    reference_enabled = bool(
        "reference_merge_fid" in sources.columns and sources["reference_merge_fid"].notna().any()
    )
    selected_eval_positive = int(selected["label"].sum()) if reference_enabled and "label" in selected.columns else 0
    selected_zero_components = sum(len(_parse_ids(value)) for value in selected["zero_component_ids"]) if len(selected) else 0
    summary = {
        "input_gpkg": str(input_gpkg),
        "pair_candidate_csv": str(pair_candidate_csv),
        "model": str(model_path),
        "model_kind": str(model_kind),
        "output_gpkg": str(output_gpkg),
        "candidate_csv": str(candidate_csv),
        "candidate_input_csv": str(args.candidate_input_csv),
        "candidate_strategy": str(args.candidate_strategy),
        "enclosure_level": str(args.enclosure_level),
        "threshold": float(threshold),
        "review_threshold": float(review_threshold),
        "anchor_need_threshold": float(args.anchor_need_threshold) if anchor_gate_allowed else None,
        "complete_pool_gate_bypass_threshold": None
        if not anchor_gate_allowed
        else float(args.complete_pool_gate_bypass_threshold),
        "uprn_skeleton_gate_bypass_threshold": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_gate_bypass_threshold),
        "uprn_skeleton_gate_bypass_min_group_count": None
        if not anchor_gate_allowed
        else int(args.uprn_skeleton_gate_bypass_min_group_count),
        "uprn_skeleton_gate_bypass_min_regularity": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_gate_bypass_min_regularity),
        "uprn_skeleton_gate_bypass_max_after_area": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_gate_bypass_max_after_area),
        "uprn_skeleton_pair_override_min_pair_proba": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_pair_override_min_pair_proba),
        "uprn_skeleton_pair_override_min_regularity": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_pair_override_min_regularity),
        "uprn_skeleton_pair_override_max_after_area": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_pair_override_max_after_area),
        "uprn_skeleton_pair_override_max_zero_area_ratio": None
        if not anchor_gate_allowed
        else float(args.uprn_skeleton_pair_override_max_zero_area_ratio),
        "anchor_gate_enabled": bool(anchor_gate_allowed),
        "anchor_group_candidate_rows": int(len(candidates)),
        "anchor_group_candidate_rows_after_gate": int(len(candidates_for_selection)),
        "anchor_gate_bypass_complete_pool_rows": int(candidates["anchor_gate_bypass_complete_pool"].sum())
        if "anchor_gate_bypass_complete_pool" in candidates.columns
        else 0,
        "anchor_gate_bypass_uprn_skeleton_rows": int(candidates["anchor_gate_bypass_uprn_skeleton"].sum())
        if "anchor_gate_bypass_uprn_skeleton" in candidates.columns
        else 0,
        "anchor_group_skeleton_pair_override_rows": int(
            candidates["anchor_group_repair_skeleton_pair_override"].sum()
        )
        if "anchor_group_repair_skeleton_pair_override" in candidates.columns
        else 0,
        "residual_fallback_enabled": bool(not args.disable_residual_fallback),
        "residual_fallback_min_pair_proba": float(args.residual_fallback_min_pair_proba),
        "residual_fallback_min_shared_edge": float(args.residual_fallback_min_shared_edge),
        "anchor_group_residual_fallback_rows": int(
            selected.get(
                "anchor_group_repair_residual_fallback",
                pd.Series(0, index=selected.index),
            )
            .fillna(0)
            .astype(int)
            .sum()
        )
        if len(selected)
        else 0,
        "anchor_group_selected_rows": int(len(selected)),
        "anchor_group_selected_zero_components": int(selected_zero_components),
        "anchor_group_review_rows": int(len(review)),
        "anchor_group_conflict_rows": int(len(conflicts)),
        "reference_enabled": bool(reference_enabled),
        "selected_reference_complete_positive_rows": selected_eval_positive,
        "selected_reference_precision_eval": float(selected_eval_positive / len(selected))
        if reference_enabled and len(selected)
        else None,
        "base_predicted_components": int(len(predicted)),
        "repaired_predicted_components": int(len(predicted_new)),
        "base_possible_false_positive_clusters": int(predicted["possible_false_positive_cluster"].fillna(0).astype(int).sum()),
        "repaired_possible_false_positive_clusters": int(len(possible_fp)),
        "base_possible_split_reference_clusters": int(predicted["possible_split_reference"].fillna(0).astype(int).sum()),
        "repaired_possible_split_reference_clusters": int(len(possible_split)),
        "merged_only_components": int(len(merged_only)),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Anchor group repair apply complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

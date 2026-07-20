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
from apply_wfs_merge_operation_pipeline import _filter_edges_to_current_components
from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_operation_pruned_only_guard_v2.gpkg"
)
DEFAULT_EDGE_CSV = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "edge_candidate_predictions_full.csv"
)
DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "anchor_problem_detection_model_probe_v1.joblib"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_anchor_repaired_threshold_095.gpkg"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _read_optional_layer(path: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _reference_is_single(value: object) -> bool:
    text = str(value or "")
    return bool(text) and "|" not in text


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
    return shapely.union(
        geom_by_component[int(row["anchor_component_id"])],
        geom_by_component[int(row["zero_component_id"])],
    )


def build_anchor_candidates(
    *,
    input_gpkg: Path,
    edge_csv: Path,
    max_edge_chunksize: int,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    _log("[INFO] Reading predicted parcels")
    predicted = pyogrio.read_dataframe(input_gpkg, layer="predicted_parcels_with_uprn")
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted["pred_component_id"] = predicted["pred_component_id"].astype(int)
    predicted_idx = predicted.set_index("pred_component_id")
    comp_uprn = predicted_idx["pred_uprn_count"].fillna(0).astype(int)
    comp_area = predicted_idx["pred_area"].astype(float)
    comp_source_count = predicted_idx["source_count"].astype(int)
    comp_refs = predicted_idx["reference_merge_fids"].fillna("").astype(str)
    comp_split = predicted_idx["possible_split_reference"].fillna(0).astype(int)
    geom_by_component = predicted_idx.geometry.to_dict()

    _log("[INFO] Reading source/component map")
    sources = pyogrio.read_dataframe(input_gpkg, layer="prediction_source_polygons")
    sources = sources[sources.geometry.notna() & ~sources.geometry.is_empty].copy()
    sources["source_fid"] = sources["source_fid"].astype(int)
    sources["pred_component_id"] = sources["pred_component_id"].astype(int)
    source_to_component = sources.set_index("source_fid")["pred_component_id"]

    _log("[INFO] Aggregating source edges to anchor-zero component candidates")
    usecols = [
        "left_source_fid",
        "right_source_fid",
        "shared_edge_len",
        "model_proba",
        "role_pair",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(edge_csv, usecols=usecols, chunksize=max_edge_chunksize):
        chunk["left_comp"] = chunk["left_source_fid"].astype(int).map(source_to_component)
        chunk["right_comp"] = chunk["right_source_fid"].astype(int).map(source_to_component)
        chunk = chunk[chunk["left_comp"].notna() & chunk["right_comp"].notna()].copy()
        if chunk.empty:
            continue
        chunk["left_comp"] = chunk["left_comp"].astype(int)
        chunk["right_comp"] = chunk["right_comp"].astype(int)
        chunk = chunk[chunk["left_comp"].ne(chunk["right_comp"])].copy()
        if chunk.empty:
            continue
        chunk["left_uprn"] = chunk["left_comp"].map(comp_uprn).fillna(-1).astype(int)
        chunk["right_uprn"] = chunk["right_comp"].map(comp_uprn).fillna(-1).astype(int)
        mask = (
            (chunk["left_uprn"].gt(0) & chunk["right_uprn"].eq(0))
            | (chunk["left_uprn"].eq(0) & chunk["right_uprn"].gt(0))
        )
        chunk = chunk[mask].copy()
        if chunk.empty:
            continue
        chunk["anchor_component_id"] = np.where(
            chunk["left_uprn"].gt(0),
            chunk["left_comp"],
            chunk["right_comp"],
        ).astype(int)
        chunk["zero_component_id"] = np.where(
            chunk["left_uprn"].eq(0),
            chunk["left_comp"],
            chunk["right_comp"],
        ).astype(int)
        chunk["anchor_uprn_count"] = np.where(
            chunk["left_uprn"].gt(0),
            chunk["left_uprn"],
            chunk["right_uprn"],
        ).astype(int)
        chunk["zero_uprn_count"] = 0
        chunk["example_left_source_fid"] = chunk["left_source_fid"].astype(int)
        chunk["example_right_source_fid"] = chunk["right_source_fid"].astype(int)
        parts.append(
            chunk[
                [
                    "anchor_component_id",
                    "zero_component_id",
                    "anchor_uprn_count",
                    "zero_uprn_count",
                    "shared_edge_len",
                    "model_proba",
                    "example_left_source_fid",
                    "example_right_source_fid",
                    "role_pair",
                ]
            ]
        )

    if not parts:
        raise RuntimeError("No anchor-zero component candidates were generated.")
    edges = pd.concat(parts, ignore_index=True)

    idx_max = edges.groupby(["anchor_component_id", "zero_component_id"])["model_proba"].idxmax()
    examples = edges.loc[
        idx_max,
        [
            "anchor_component_id",
            "zero_component_id",
            "example_left_source_fid",
            "example_right_source_fid",
            "role_pair",
            "model_proba",
        ],
    ]
    agg = (
        edges.groupby(["anchor_component_id", "zero_component_id"])
        .agg(
            anchor_uprn_count=("anchor_uprn_count", "max"),
            zero_uprn_count=("zero_uprn_count", "max"),
            shared_edge_sum=("shared_edge_len", "sum"),
            shared_edge_max=("shared_edge_len", "max"),
            source_edge_count=("shared_edge_len", "size"),
            edge_proba_max=("model_proba", "max"),
            edge_proba_mean=("model_proba", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(examples, on=["anchor_component_id", "zero_component_id"], how="left")
    agg = agg.rename(columns={"model_proba": "example_model_proba"})

    agg["anchor_area"] = agg["anchor_component_id"].map(comp_area).astype(float)
    agg["zero_area"] = agg["zero_component_id"].map(comp_area).astype(float)
    agg["after_area"] = agg["anchor_area"] + agg["zero_area"]
    agg["anchor_source_count"] = agg["anchor_component_id"].map(comp_source_count).astype(int)
    agg["zero_source_count"] = agg["zero_component_id"].map(comp_source_count).astype(int)
    agg["anchor_reference_fids"] = agg["anchor_component_id"].map(comp_refs)
    agg["zero_reference_fids"] = agg["zero_component_id"].map(comp_refs)
    agg["anchor_possible_split"] = agg["anchor_component_id"].map(comp_split).fillna(0).astype(int)
    agg["zero_possible_split"] = agg["zero_component_id"].map(comp_split).fillna(0).astype(int)
    agg["same_reference_eval"] = (
        agg["anchor_reference_fids"].map(_reference_is_single)
        & agg["anchor_reference_fids"].eq(agg["zero_reference_fids"])
    ).astype(int)

    zero_counts = agg.groupby("zero_component_id")["anchor_component_id"].nunique().rename("neighbor_anchor_count")
    agg = agg.merge(zero_counts, on="zero_component_id", how="left")
    ranked = agg.sort_values(
        ["zero_component_id", "shared_edge_sum", "edge_proba_max"],
        ascending=[True, False, False],
    ).copy()
    ranked["zero_anchor_rank_by_shared"] = ranked.groupby("zero_component_id").cumcount() + 1
    second_shared = ranked[ranked["zero_anchor_rank_by_shared"].eq(2)].set_index("zero_component_id")["shared_edge_sum"]
    best_rank = ranked.set_index(["anchor_component_id", "zero_component_id"])["zero_anchor_rank_by_shared"]
    agg["zero_anchor_rank_by_shared"] = [
        int(best_rank.get((int(row.anchor_component_id), int(row.zero_component_id)), 999))
        for row in agg.itertuples(index=False)
    ]
    agg["second_best_shared_edge_sum"] = agg["zero_component_id"].map(second_shared).fillna(0.0)
    agg["shared_edge_margin_ratio"] = agg["shared_edge_sum"] / agg["second_best_shared_edge_sum"].replace(0.0, np.nan)
    agg["shared_edge_margin_ratio"] = agg["shared_edge_margin_ratio"].fillna(999.0)

    _log("[INFO] Computing candidate union shape metrics")
    anchor_geoms = gpd.GeoSeries(agg["anchor_component_id"].map(geom_by_component).to_list(), crs=predicted.crs)
    zero_geoms = gpd.GeoSeries(agg["zero_component_id"].map(geom_by_component).to_list(), crs=predicted.crs)
    union_geoms = shapely.union(anchor_geoms.array, zero_geoms.array)
    shape = pd.DataFrame.from_records([_shape_metrics(geom) for geom in union_geoms]).add_prefix("candidate_")
    agg = pd.concat([agg.reset_index(drop=True), shape.reset_index(drop=True)], axis=1)
    agg["tier_unique_anchor"] = (
        agg["neighbor_anchor_count"].eq(1)
        & agg["shared_edge_sum"].ge(3.0)
        & agg["after_area"].le(2000.0)
        & agg["zero_area"].le(1000.0)
    ).astype(int)
    agg["tier_clear_anchor"] = (
        agg["shared_edge_sum"].ge(5.0)
        & agg["after_area"].le(2000.0)
        & agg["zero_area"].le(1000.0)
        & (agg["neighbor_anchor_count"].eq(1) | agg["shared_edge_margin_ratio"].ge(1.5))
    ).astype(int)
    agg["tier_shape_supported"] = (
        agg["tier_clear_anchor"].eq(1)
        & agg["candidate_hull_gap_ratio"].le(0.45)
        & agg["candidate_regularity_score"].ge(0.60)
    ).astype(int)
    agg["native_probe_score"] = (
        1.2 * np.log1p(agg["shared_edge_sum"].clip(lower=0))
        + 0.9 * np.minimum(agg["shared_edge_margin_ratio"], 5.0) / 5.0
        + 0.8 * agg["tier_unique_anchor"]
        + 0.6 * agg["candidate_regularity_score"].fillna(0)
        - 0.4 * agg["candidate_hull_gap_ratio"].clip(lower=0, upper=2).fillna(0)
        - 0.25 * np.log1p(agg["neighbor_anchor_count"].clip(lower=1) - 1)
    )
    return agg, predicted, sources, gpd.GeoDataFrame(geometry=[], crs=predicted.crs)


def score_anchor_candidates(candidates: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    feature_cols = payload.get("feature_cols") if isinstance(payload, dict) else None
    if not feature_cols:
        raise RuntimeError("Anchor repair model must provide feature_cols in its joblib payload.")
    missing = sorted(set(feature_cols) - set(candidates.columns))
    if missing:
        raise RuntimeError(f"Anchor candidates are missing model features: {missing}")
    out = candidates.copy()
    out["anchor_repair_proba"] = pipeline.predict_proba(out[feature_cols])[:, 1]
    return out


def select_anchor_repairs(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    review_threshold: float,
    min_proba_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high = candidates[candidates["anchor_repair_proba"].ge(float(threshold))].copy()
    review = candidates[
        candidates["anchor_repair_proba"].ge(float(review_threshold))
        & candidates["anchor_repair_proba"].lt(float(threshold))
    ].copy()
    if high.empty:
        return high, review, high.copy()

    high = high.sort_values(
        ["zero_component_id", "anchor_repair_proba", "shared_edge_sum"],
        ascending=[True, False, False],
    ).copy()
    high["rank_by_zero_proba"] = high.groupby("zero_component_id").cumcount() + 1
    best = high[high["rank_by_zero_proba"].eq(1)].copy()
    second = high[high["rank_by_zero_proba"].eq(2)].set_index("zero_component_id")["anchor_repair_proba"]
    best["second_best_anchor_repair_proba"] = best["zero_component_id"].map(second)
    best["proba_margin_to_second"] = (
        best["anchor_repair_proba"] - best["second_best_anchor_repair_proba"].fillna(-np.inf)
    )
    conflict_mask = best["second_best_anchor_repair_proba"].notna() & best["proba_margin_to_second"].lt(float(min_proba_margin))
    selected = best[~conflict_mask].copy()
    conflicts = high[
        high["zero_component_id"].isin(set(best.loc[conflict_mask, "zero_component_id"].astype(int)))
    ].copy()
    return selected.reset_index(drop=True), review.reset_index(drop=True), conflicts.reset_index(drop=True)


def _repair_edges(
    selected: pd.DataFrame,
    sources: gpd.GeoDataFrame,
    geom_by_component: dict[int, Any],
) -> gpd.GeoDataFrame:
    if selected.empty:
        return gpd.GeoDataFrame(geometry=[], crs=sources.crs)
    source_ref = sources.set_index(sources["source_fid"].astype(int))["reference_merge_fid"]
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        left_source = int(row.example_left_source_fid)
        right_source = int(row.example_right_source_fid)
        records.append(
            {
                "left_source_fid": left_source,
                "right_source_fid": right_source,
                "left_merge_fid": int(source_ref.get(left_source, -1)) if pd.notna(source_ref.get(left_source, np.nan)) else -1,
                "right_merge_fid": int(source_ref.get(right_source, -1)) if pd.notna(source_ref.get(right_source, np.nan)) else -1,
                "label": int(row.same_reference_eval),
                "model_proba": float(row.edge_proba_max),
                "completion_proba": float(row.anchor_repair_proba),
                "role_pair": str(row.role_pair),
                "pred_component_id": int(row.anchor_component_id),
                "model_stage": "anchor_repair",
                "geometry": _line_between_components(
                    int(row.anchor_component_id),
                    int(row.zero_component_id),
                    geom_by_component,
                ),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _candidate_geometries(
    rows: pd.DataFrame,
    geom_by_component: dict[int, Any],
    crs,
) -> gpd.GeoDataFrame:
    if rows.empty:
        return gpd.GeoDataFrame(rows.copy(), geometry=[], crs=crs)
    geoms = [_candidate_union_geometry(row, geom_by_component) for _, row in rows.iterrows()]
    return gpd.GeoDataFrame(rows.copy(), geometry=geoms, crs=crs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply high-confidence UPRN-anchor residual repair to WFS merge output.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--edge-csv", default=DEFAULT_EDGE_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--candidate-csv", default="")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--review-threshold", type=float, default=0.90)
    parser.add_argument("--min-proba-margin", type=float, default=0.05)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--top-candidate-layer-limit", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    edge_csv = Path(args.edge_csv)
    model_path = Path(args.model)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    candidates, predicted, sources, _ = build_anchor_candidates(
        input_gpkg=input_gpkg,
        edge_csv=edge_csv,
        max_edge_chunksize=int(args.chunksize),
    )
    candidates = score_anchor_candidates(candidates, model_path)
    selected, review, conflicts = select_anchor_repairs(
        candidates,
        threshold=float(args.threshold),
        review_threshold=float(args.review_threshold),
        min_proba_margin=float(args.min_proba_margin),
    )
    _log(
        "[INFO] Anchor candidates="
        f"{len(candidates):,}; selected={len(selected):,}; review={len(review):,}; conflicts={len(conflicts):,}"
    )

    sources_new = sources.copy()
    sources_new["anchor_repair_status"] = "base"
    sources_new["anchor_repaired_to_component"] = np.nan
    sources_new["anchor_repair_proba"] = np.nan
    sources_new["anchor_repair_from_component"] = np.nan

    for row in selected.itertuples(index=False):
        zero_id = int(row.zero_component_id)
        anchor_id = int(row.anchor_component_id)
        mask = sources_new["pred_component_id"].astype(int).eq(zero_id)
        sources_new.loc[mask, "anchor_repair_status"] = "anchor_repaired"
        sources_new.loc[mask, "anchor_repaired_to_component"] = anchor_id
        sources_new.loc[mask, "anchor_repair_from_component"] = zero_id
        sources_new.loc[mask, "anchor_repair_proba"] = float(row.anchor_repair_proba)
        sources_new.loc[mask, "pred_component_id"] = anchor_id

    edges = pyogrio.read_dataframe(input_gpkg, layer="predicted_positive_edges")
    edges_current = _filter_edges_to_current_components(edges, sources_new)
    old_geom_by_component = predicted.set_index(predicted["pred_component_id"].astype(int)).geometry.to_dict()
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

    semantic_reference = pyogrio.read_dataframe(input_gpkg, layer="semantic_reference_parcels")
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", sources.crs)
    geom_by_component_new = predicted_new.set_index(predicted_new["pred_component_id"].astype(int)).geometry.to_dict()

    candidate_csv = Path(args.candidate_csv) if str(args.candidate_csv).strip() else output_gpkg.with_suffix(".anchor_candidates.csv")
    _log(f"[INFO] Writing candidate CSV: {candidate_csv}")
    candidates.to_csv(candidate_csv, index=False)

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(merged_only.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels_merged_only")
    _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    _write_layer(sources_new, output_gpkg, "prediction_source_polygons")
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
    _write_layer(_candidate_geometries(selected, old_geom_by_component, sources.crs), output_gpkg, "anchor_repair_selected")
    _write_layer(_candidate_geometries(review, old_geom_by_component, sources.crs), output_gpkg, "anchor_repair_review_candidates")
    _write_layer(_candidate_geometries(conflicts, old_geom_by_component, sources.crs), output_gpkg, "anchor_repair_conflicts")
    top_candidates = candidates.sort_values("anchor_repair_proba", ascending=False).head(int(args.top_candidate_layer_limit)).copy()
    _write_layer(_candidate_geometries(top_candidates, old_geom_by_component, sources.crs), output_gpkg, "anchor_repair_candidates_top")

    selected_eval_positive = int(selected["same_reference_eval"].sum()) if "same_reference_eval" in selected.columns else 0
    summary = {
        "input_gpkg": str(input_gpkg),
        "edge_csv": str(edge_csv),
        "model": str(model_path),
        "output_gpkg": str(output_gpkg),
        "candidate_csv": str(candidate_csv),
        "threshold": float(args.threshold),
        "review_threshold": float(args.review_threshold),
        "min_proba_margin": float(args.min_proba_margin),
        "anchor_candidate_rows": int(len(candidates)),
        "anchor_selected_rows": int(len(selected)),
        "anchor_review_rows": int(len(review)),
        "anchor_conflict_rows": int(len(conflicts)),
        "selected_same_reference_eval_rows": selected_eval_positive,
        "selected_reference_precision_eval": float(selected_eval_positive / len(selected)) if len(selected) else None,
        "base_predicted_components": int(len(predicted)),
        "repaired_predicted_components": int(len(predicted_new)),
        "base_possible_false_positive_clusters": int(predicted["possible_false_positive_cluster"].fillna(0).astype(int).sum()),
        "repaired_possible_false_positive_clusters": int(len(possible_fp)),
        "base_possible_split_reference_clusters": int(predicted["possible_split_reference"].fillna(0).astype(int).sum()),
        "repaired_possible_split_reference_clusters": int(len(possible_split)),
        "merged_only_components": int(len(merged_only)),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Anchor repair apply complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

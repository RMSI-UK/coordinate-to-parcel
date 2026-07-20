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
import shapely
from shapely.geometry import LineString

from train_wfs_merge_completion_model import MODEL_FILE_NAME, _build_completion_candidates, _shape_metrics
from train_wfs_merge_edge_model import _add_derived_features


DEFAULT_INPUT_PREDICTION_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1/"
    "model_predicted_polygons_threshold_090_shape_guard.gpkg"
)
DEFAULT_EDGE_CSV = (
    "/data/sheffield/spatial/base-map/"
    "sheffield_wp5_wfs_merge_training_small_edges_50k_uprn_rect_semantic_clean.csv"
)
DEFAULT_EDGE_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1"
DEFAULT_COMPLETION_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3"
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3/"
    "model_predicted_polygons_completion_v3_threshold_090_shape_guard.gpkg"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _read_optional_layer(path: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _load_edge_predictions(edge_csv: str, edge_model_dir: Path) -> pd.DataFrame:
    edge_model = joblib.load(edge_model_dir / "wfs_merge_edge_model_v1.joblib")
    edge_meta = json.loads((edge_model_dir / "metrics.json").read_text(encoding="utf-8"))
    edge_features = edge_meta["feature_columns"]
    edge_df = _add_derived_features(pd.read_csv(edge_csv))
    edge_df["edge_proba"] = edge_model.predict_proba(edge_df[edge_features])[:, 1]
    return edge_df


def _safe_int(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _source_reference_values(group: gpd.GeoDataFrame) -> list[int]:
    return sorted({int(v) for v in group["reference_merge_fid"].dropna().astype(int)})


def _build_edge_lines(selected: pd.DataFrame, sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if selected.empty:
        return gpd.GeoDataFrame(
            columns=[
                "left_source_fid",
                "right_source_fid",
                "left_merge_fid",
                "right_merge_fid",
                "label",
                "model_proba",
                "completion_proba",
                "role_pair",
                "pred_component_id",
                "model_stage",
                "geometry",
            ],
            geometry="geometry",
            crs=sources.crs,
        )

    geom_by_source = sources.set_index(sources["source_fid"].astype(int)).geometry.to_dict()
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        comp_source = int(row.component_source_fid)
        cand_source = int(row.candidate_source_fid)
        comp_geom = geom_by_source[comp_source]
        cand_geom = geom_by_source[cand_source]
        left_point = comp_geom.representative_point()
        right_point = cand_geom.representative_point()
        geom = LineString([(shapely.get_x(left_point), shapely.get_y(left_point)), (shapely.get_x(right_point), shapely.get_y(right_point))])
        records.append(
            {
                "left_source_fid": comp_source,
                "right_source_fid": cand_source,
                "left_merge_fid": _safe_int(row.component_reference_merge_fid, -1),
                "right_merge_fid": _safe_int(row.candidate_reference_merge_fid, -1),
                "label": int(row.label),
                "model_proba": float(row.edge_proba),
                "completion_proba": float(row.completion_proba),
                "role_pair": str(row.role_pair),
                "pred_component_id": int(row.component_id),
                "model_stage": "completion",
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _component_uprn_counts(predicted: gpd.GeoDataFrame, sources: gpd.GeoDataFrame) -> dict[int, int]:
    if "source_uprn_count" in sources.columns:
        return {
            int(comp_id): int(group["source_uprn_count"].fillna(0).astype(int).sum())
            for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int))
        }
    if "uprn_count" in sources.columns:
        return {
            int(comp_id): int(group["uprn_count"].fillna(0).astype(int).sum())
            for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int))
        }
    if "pred_uprn_count" not in predicted.columns:
        return {}
    old_uprn = dict(
        zip(
            predicted["pred_component_id"].astype(int),
            predicted["pred_uprn_count"].fillna(0).astype(int),
        )
    )
    old_component_by_source = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    out: dict[int, int] = {}
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int)):
        old_components = {int(old_component_by_source[int(source_fid)]) for source_fid in group["source_fid"].astype(int)}
        out[int(comp_id)] = int(sum(old_uprn.get(old_comp, 0) for old_comp in old_components))
    return out


def _build_predicted_parcels(
    sources: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    old_predicted: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    source_uprn = _component_uprn_counts(old_predicted, sources)
    edge_stats: dict[int, dict[str, float]] = {}
    if not edges.empty:
        edge_values = edges.copy()
        edge_values["merge_proba"] = edge_values["completion_proba"].fillna(edge_values["model_proba"]).astype(float)
        for comp_id, group in edge_values.groupby(edge_values["pred_component_id"].astype(int)):
            edge_stats[int(comp_id)] = {
                "predicted_edge_count": int(len(group)),
                "proba_min": float(group["merge_proba"].min()),
                "proba_mean": float(group["merge_proba"].mean()),
                "proba_max": float(group["merge_proba"].max()),
            }

    records: list[dict[str, Any]] = []
    reference_by_component: dict[int, list[int]] = {}
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        geom = shapely.union_all(group.geometry.array)
        shape = _shape_metrics(geom)
        refs = _source_reference_values(group)
        reference_by_component[int(comp_id)] = refs
        stats = edge_stats.get(
            int(comp_id),
            {"predicted_edge_count": 0, "proba_min": np.nan, "proba_mean": np.nan, "proba_max": np.nan},
        )
        semantic_count = int(group["is_semantic_source"].fillna(0).astype(int).sum())
        rec = {
            "pred_component_id": int(comp_id),
            "source_count": int(len(group)),
            "semantic_source_count": semantic_count,
            "outside_source_count": int(len(group) - semantic_count),
            "reference_merge_fid_count": int(len(refs)),
            "reference_merge_fids": "|".join(str(v) for v in refs),
            "max_reference_split_count": 1,
            "predicted_edge_count": stats["predicted_edge_count"],
            "proba_min": stats["proba_min"],
            "proba_mean": stats["proba_mean"],
            "proba_max": stats["proba_max"],
            "has_predicted_merge": int(len(group) > 1),
            "possible_false_positive_cluster": int(len(refs) > 1),
            "possible_split_reference": 0,
            "pred_uprn_count": int(source_uprn.get(int(comp_id), 0)),
            "geometry": geom,
        }
        for name, value in shape.items():
            rec[f"pred_{name}"] = float(value)
        records.append(rec)

    ref_component_counts: dict[int, int] = {}
    for refs in reference_by_component.values():
        for ref in refs:
            ref_component_counts[ref] = ref_component_counts.get(ref, 0) + 1

    for rec in records:
        refs = reference_by_component[int(rec["pred_component_id"])]
        max_split = max([ref_component_counts.get(ref, 1) for ref in refs] or [1])
        rec["max_reference_split_count"] = int(max_split)
        rec["possible_split_reference"] = int(len(refs) == 1 and max_split > 1)

    return gpd.GeoDataFrame(records, geometry="geometry", crs=sources.crs)


def _write_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    clean = gdf.copy().reset_index(drop=True)
    for column in clean.columns:
        if column == clean.geometry.name:
            continue
        values = clean[column].astype("object")
        if any(isinstance(value, (list, tuple, set, dict)) for value in values):
            clean[column] = values.map(
                lambda value: json.dumps(value) if isinstance(value, (list, tuple, set, dict)) else value
            )
    clean.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the second-stage completion model to WFS merge predictions.")
    parser.add_argument("--input-prediction-gpkg", default=DEFAULT_INPUT_PREDICTION_GPKG)
    parser.add_argument("--edge-csv", default=DEFAULT_EDGE_CSV)
    parser.add_argument("--edge-model-dir", default=DEFAULT_EDGE_MODEL_DIR)
    parser.add_argument("--completion-model-dir", default=DEFAULT_COMPLETION_MODEL_DIR)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--min-candidate-edge-proba", type=float, default=0.0)
    parser.add_argument("--max-candidate-area", type=float, default=120.0)
    parser.add_argument("--min-mrr-gain", type=float, default=0.0)
    parser.add_argument("--min-hull-gap-reduction", type=float, default=0.0)
    parser.add_argument("--min-regularity-score-gain", type=float, default=0.0)
    parser.add_argument("--min-boundary-complexity-reduction", type=float, default=-999.0)
    parser.add_argument("--min-notch-index-reduction", type=float, default=-999.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_prediction_gpkg)
    edge_model_dir = Path(args.edge_model_dir)
    completion_model_dir = Path(args.completion_model_dir)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    _log(f"[INFO] Reading base prediction: {input_gpkg}")
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
    sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")
    base_edges = gpd.read_file(input_gpkg, layer="predicted_positive_edges", engine="pyogrio")
    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", sources.crs)
    excluded_gapfill_council_sources = _read_optional_layer(
        input_gpkg,
        "excluded_gapfill_council_sources",
        sources.crs,
    )
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", sources.crs)

    sources = sources.copy()
    sources["original_pred_component_id"] = sources["pred_component_id"].astype(int)
    reference_enabled = bool(
        "reference_merge_fid" in sources.columns and sources["reference_merge_fid"].notna().any()
    )

    _log("[INFO] Building completion candidates")
    edge_df = _load_edge_predictions(args.edge_csv, edge_model_dir)
    candidates = _build_completion_candidates(
        edge_df=edge_df,
        predicted=predicted,
        sources=sources,
        min_edge_proba=float(args.min_candidate_edge_proba),
        max_candidate_area=float(args.max_candidate_area),
        require_component_reference=False,
    )
    if candidates.empty:
        _log("[INFO] No completion candidates; carrying base prediction forward")
        candidates = candidates.copy()
        candidates["completion_pred"] = pd.Series(dtype="int64")
        selected = candidates.iloc[0:0].copy()
    else:
        completion_model = joblib.load(completion_model_dir / MODEL_FILE_NAME)
        completion_meta = json.loads((completion_model_dir / "metrics.json").read_text(encoding="utf-8"))
        feature_cols = completion_meta["feature_columns"]
        missing_features = sorted(set(feature_cols) - set(candidates.columns))
        if missing_features:
            raise RuntimeError(f"Completion candidates are missing model features: {missing_features}")
        candidates["completion_proba"] = completion_model.predict_proba(candidates[feature_cols])[:, 1]
        candidates["completion_pred_raw"] = candidates["completion_proba"].ge(float(args.threshold)).astype(int)
        candidates["shape_guard_pass"] = (
            candidates["mrr_gain"].ge(float(args.min_mrr_gain))
            & candidates["hull_gap_reduction"].ge(float(args.min_hull_gap_reduction))
            & candidates["regularity_score_gain"].ge(float(args.min_regularity_score_gain))
            & candidates["boundary_complexity_reduction"].ge(float(args.min_boundary_complexity_reduction))
            & candidates["notch_index_reduction"].ge(float(args.min_notch_index_reduction))
        ).astype(int)
        candidates["completion_candidate_pass"] = (
            candidates["completion_pred_raw"].eq(1) & candidates["shape_guard_pass"].eq(1)
        ).astype(int)

        selected = candidates[candidates["completion_candidate_pass"].eq(1)].copy()
        selected = selected.sort_values(
            ["completion_proba", "mrr_gain", "hull_gap_reduction"],
            ascending=[False, False, False],
        )
        kept_rows: list[pd.Series] = []
        used_candidate_sources: set[int] = set()
        for _, row in selected.iterrows():
            candidate_source = int(row["candidate_source_fid"])
            if candidate_source in used_candidate_sources:
                continue
            used_candidate_sources.add(candidate_source)
            kept_rows.append(row)
        selected = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else selected.iloc[0:0].copy()
        candidates["completion_pred"] = 0
        selected_keys = {
            (int(row.component_id), int(row.candidate_source_fid))
            for row in selected.itertuples(index=False)
        }
        final_selected_mask = [
            (int(row.component_id), int(row.candidate_source_fid)) in selected_keys
            for row in candidates.itertuples(index=False)
        ]
        candidates.loc[final_selected_mask, "completion_pred"] = 1
        selected["completion_pred"] = 1

    _log(f"[INFO] Completion candidates={len(candidates):,}; selected after guard={len(selected):,}")
    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    if {"candidate_source_fid", "component_id"}.issubset(selected.columns):
        selected_component_by_source = dict(
            zip(selected["candidate_source_fid"].astype(int), selected["component_id"].astype(int))
        )
    else:
        selected_component_by_source = {}
    sources["completion_added_to_component"] = sources["source_fid"].astype(int).map(selected_component_by_source)
    added_mask = sources["completion_added_to_component"].notna()
    sources.loc[added_mask, "pred_component_id"] = sources.loc[added_mask, "completion_added_to_component"].astype(int)
    sources["completion_added"] = added_mask.astype(int)
    sources["completion_source_status"] = np.where(added_mask, "completion_added", "base")

    completion_edges = _build_edge_lines(selected, sources)
    base_edges = base_edges.copy()
    base_edges["completion_proba"] = np.nan
    base_edges["model_stage"] = "edge"
    base_edges["pred_component_id"] = base_edges["pred_component_id"].astype(int)
    edge_cols = [
        "left_source_fid",
        "right_source_fid",
        "left_merge_fid",
        "right_merge_fid",
        "label",
        "model_proba",
        "completion_proba",
        "role_pair",
        "pred_component_id",
        "model_stage",
        "geometry",
    ]
    edges = pd.concat([base_edges[edge_cols], completion_edges[edge_cols]], ignore_index=True)
    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=sources.crs)

    predicted_new = _build_predicted_parcels(sources, edges, predicted)
    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    merged_only_no_uprn = merged_only.drop(columns=["pred_uprn_count"])
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()

    candidate_debug = candidates.copy()
    if "candidate_source_fid" in candidate_debug.columns:
        source_geom = sources.set_index(sources["source_fid"].astype(int)).geometry
        candidate_debug["geometry"] = candidate_debug["candidate_source_fid"].astype(int).map(source_geom)
    else:
        candidate_debug["geometry"] = gpd.GeoSeries([], crs=sources.crs)
    candidate_debug = gpd.GeoDataFrame(candidate_debug, geometry="geometry", crs=sources.crs)
    if "completion_pred" in candidate_debug.columns:
        added_sources = candidate_debug[candidate_debug["completion_pred"].eq(1)].copy()
    else:
        added_sources = candidate_debug.iloc[0:0].copy()

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(merged_only_no_uprn, output_gpkg, "predicted_parcels_merged_only")
    _write_layer(semantic_reference, output_gpkg, "semantic_reference_parcels")
    _write_layer(sources, output_gpkg, "prediction_source_polygons")
    _write_layer(excluded_gapfill_council_sources, output_gpkg, "excluded_gapfill_council_sources")
    _write_layer(excluded_problem_sources, output_gpkg, "excluded_problem_sources")
    _write_layer(edges, output_gpkg, "predicted_positive_edges")
    _write_layer(possible_fp.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_false_positive_clusters")
    _write_layer(possible_split.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_split_reference_clusters")
    _write_layer(predicted_new, output_gpkg, "predicted_parcels_with_uprn")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only_with_uprn")
    _write_layer(possible_fp, output_gpkg, "possible_false_positive_clusters_with_uprn")
    _write_layer(possible_split, output_gpkg, "possible_split_reference_clusters_with_uprn")
    _write_layer(merged_only[merged_only["pred_uprn_count"].le(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_le1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].eq(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_eq1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].gt(1)].copy(), output_gpkg, "predicted_parcels_merged_only_multi_uprn")
    _write_layer(added_sources, output_gpkg, "completion_added_sources")
    _write_layer(candidate_debug, output_gpkg, "completion_candidate_debug")

    summary = {
        "input_prediction_gpkg": str(input_gpkg),
        "output_gpkg": str(output_gpkg),
        "completion_model_dir": str(completion_model_dir),
        "reference_enabled": bool(reference_enabled),
        "threshold": float(args.threshold),
        "min_mrr_gain": float(args.min_mrr_gain),
        "min_hull_gap_reduction": float(args.min_hull_gap_reduction),
        "min_regularity_score_gain": float(args.min_regularity_score_gain),
        "min_boundary_complexity_reduction": float(args.min_boundary_complexity_reduction),
        "min_notch_index_reduction": float(args.min_notch_index_reduction),
        "max_candidate_area": float(args.max_candidate_area),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "selected_label_counts": (
            selected["label"].value_counts().sort_index().astype(int).to_dict()
            if "label" in selected.columns
            else {}
        ),
        "predicted_components": int(len(predicted_new)),
        "merged_only_components": int(len(merged_only)),
        "possible_false_positive_clusters": int(len(possible_fp)),
        "possible_split_reference_clusters": int(len(possible_split)),
        "merged_only_uprn_counts": merged_only["pred_uprn_count"].value_counts().sort_index().astype(int).to_dict(),
        "base_predicted_components": int(len(predicted)),
        "excluded_gapfill_council_sources": int(len(excluded_gapfill_council_sources)),
        "excluded_problem_sources": int(len(excluded_problem_sources)),
    }
    summary_path = output_gpkg.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Completion apply complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

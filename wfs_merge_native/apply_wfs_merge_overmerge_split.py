#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely

from apply_wfs_merge_completion_model import _write_layer
from overmerge_split_features import build_overmerge_split_candidates
from train_wfs_merge_completion_model import _shape_metrics
from train_wfs_merge_overmerge_split_model import MODEL_FILE_NAME


DEFAULT_INPUT_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/03_operation_pruned_only.gpkg"
DEFAULT_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_overmerge_split_model_v1"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/03b_overmerge_split.gpkg"


def _log(message: str) -> None:
    print(message, flush=True)


class _UnionFind:
    def __init__(self, values: pd.Series) -> None:
        self.parent = {int(value): int(value) for value in values.astype(int).tolist()}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def _component_ids(source_ids: pd.Series, edges: pd.DataFrame) -> pd.Series:
    uf = _UnionFind(source_ids)
    source_set = set(source_ids.astype(int))
    for row in edges.itertuples(index=False):
        left = int(row.left_source_fid)
        right = int(row.right_source_fid)
        if left in source_set and right in source_set:
            uf.union(left, right)
    roots = source_ids.astype(int).map(uf.find)
    root_to_component = {root: idx + 1 for idx, root in enumerate(sorted(set(roots.astype(int))))}
    return roots.map(root_to_component).astype(int)


def _reference_values(group: gpd.GeoDataFrame) -> list[int]:
    if "reference_merge_fid" not in group.columns:
        return []
    return sorted({int(value) for value in group["reference_merge_fid"].dropna().astype(int)})


def _build_predicted_parcels_from_sources(sources: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edge_stats: dict[int, dict[str, float]] = {}
    if not edges.empty:
        work = edges.copy()
        if "completion_proba" not in work.columns:
            work["completion_proba"] = np.nan
        work["merge_proba"] = work["completion_proba"].fillna(work["model_proba"]).astype(float)
        for comp_id, group in work.groupby(work["pred_component_id"].astype(int), sort=True):
            edge_stats[int(comp_id)] = {
                "predicted_edge_count": int(len(group)),
                "proba_min": float(group["merge_proba"].min()),
                "proba_mean": float(group["merge_proba"].mean()),
                "proba_max": float(group["merge_proba"].max()),
            }

    records: list[dict[str, object]] = []
    reference_by_component: dict[int, list[int]] = {}
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        comp_id = int(comp_id)
        geom = shapely.union_all(group.geometry.array)
        shape = _shape_metrics(geom)
        refs = _reference_values(group)
        reference_by_component[comp_id] = refs
        stats = edge_stats.get(
            comp_id,
            {"predicted_edge_count": 0, "proba_min": np.nan, "proba_mean": np.nan, "proba_max": np.nan},
        )
        semantic_count = int(group.get("is_semantic_source", pd.Series(0, index=group.index)).fillna(0).astype(int).sum())
        rec: dict[str, object] = {
            "pred_component_id": comp_id,
            "source_count": int(len(group)),
            "semantic_source_count": semantic_count,
            "outside_source_count": int(len(group) - semantic_count),
            "reference_merge_fid_count": int(len(refs)),
            "reference_merge_fids": "|".join(str(value) for value in refs),
            "max_reference_split_count": 1,
            "predicted_edge_count": stats["predicted_edge_count"],
            "proba_min": stats["proba_min"],
            "proba_mean": stats["proba_mean"],
            "proba_max": stats["proba_max"],
            "has_predicted_merge": int(len(group) > 1),
            "possible_false_positive_cluster": int(len(refs) > 1),
            "possible_split_reference": 0,
            "pred_uprn_count": int(group.get("source_uprn_count", pd.Series(0, index=group.index)).fillna(0).astype(int).sum()),
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


def _read_optional_layer(path: Path, layer: str, crs) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _filter_edges_to_current_components(edges: gpd.GeoDataFrame, sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if edges.empty:
        return edges.copy()
    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    out = edges.copy()
    left_component = out["left_source_fid"].astype(int).map(source_to_component)
    right_component = out["right_source_fid"].astype(int).map(source_to_component)
    keep = left_component.notna() & right_component.notna() & left_component.eq(right_component)
    out = out[keep].copy()
    out["pred_component_id"] = left_component[keep].astype(int).to_numpy()
    return out


def _select_splits(
    candidates: gpd.GeoDataFrame,
    *,
    threshold: float,
    min_local_aligned_count: int,
    min_component_area_to_local_median: float,
    max_split_area_per_uprn_log_dev_mean: float,
    min_split_regularity: float,
) -> gpd.GeoDataFrame:
    if candidates.empty:
        return candidates.copy()
    selected = candidates[
        candidates["overmerge_split_proba"].astype(float).ge(float(threshold))
        & candidates["split_both_sides_have_uprn"].fillna(0).astype(int).eq(1)
        & candidates["local_aligned_count"].fillna(0).astype(float).ge(float(min_local_aligned_count))
        & candidates["component_area_to_local_median"].fillna(0).astype(float).ge(float(min_component_area_to_local_median))
        & candidates["split_area_per_uprn_log_dev_mean"].fillna(999.0).astype(float).le(
            float(max_split_area_per_uprn_log_dev_mean)
        )
        & candidates["split_min_regularity_score"].fillna(0.0).astype(float).ge(float(min_split_regularity))
    ].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(
        [
            "overmerge_split_proba",
            "split_area_per_uprn_log_dev_mean",
            "split_area_log_dev_mean",
            "edge_model_proba",
        ],
        ascending=[False, True, True, True],
    )
    return selected.drop_duplicates("component_id", keep="first").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the local-mode overmerge bridge-edge split model.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--candidate-csv", default="")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--review-threshold", type=float, default=0.65)
    parser.add_argument("--max-component-area", type=float, default=2000.0)
    parser.add_argument("--max-component-source-count", type=int, default=30)
    parser.add_argument("--min-component-uprn-count", type=int, default=2)
    parser.add_argument("--local-radius", type=float, default=100.0)
    parser.add_argument("--local-angle-tolerance", type=float, default=15.0)
    parser.add_argument("--min-local-aligned-count", type=int, default=8)
    parser.add_argument("--min-component-area-to-local-median", type=float, default=1.45)
    parser.add_argument("--max-split-area-per-uprn-log-dev-mean", type=float, default=0.70)
    parser.add_argument("--min-split-regularity", type=float, default=0.76)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    _log(f"[INFO] Reading input prediction: {input_gpkg}")
    predicted = gpd.read_file(input_gpkg, layer="predicted_parcels_with_uprn", engine="pyogrio")
    sources = gpd.read_file(input_gpkg, layer="prediction_source_polygons", engine="pyogrio")
    edges = gpd.read_file(input_gpkg, layer="predicted_positive_edges", engine="pyogrio", fid_as_index=True)
    semantic_reference = _read_optional_layer(input_gpkg, "semantic_reference_parcels", sources.crs)
    excluded_problem_sources = _read_optional_layer(input_gpkg, "excluded_problem_sources", sources.crs)

    _log("[INFO] Building overmerge split candidates")
    candidates = build_overmerge_split_candidates(
        predicted,
        sources,
        edges,
        max_component_area=float(args.max_component_area),
        max_component_source_count=int(args.max_component_source_count),
        min_component_uprn_count=int(args.min_component_uprn_count),
        local_radius=float(args.local_radius),
        local_angle_tolerance=float(args.local_angle_tolerance),
    )
    if candidates.empty:
        _log("[INFO] No overmerge split candidates were generated")
        selected = candidates.copy()
        review = candidates.copy()
        scored = candidates.copy()
    else:
        model = joblib.load(model_dir / MODEL_FILE_NAME)
        meta = json.loads((model_dir / "overmerge_split_metrics.json").read_text(encoding="utf-8"))
        feature_cols = meta["feature_columns"]
        missing = sorted(set(feature_cols) - set(candidates.columns))
        if missing:
            raise RuntimeError(f"Overmerge split candidates are missing model features: {missing}")
        scored = candidates.copy()
        scored["overmerge_split_proba"] = model.predict_proba(scored[feature_cols])[:, 1]
        scored["overmerge_split_pred_raw"] = scored["overmerge_split_proba"].ge(float(args.threshold)).astype(int)
        selected = _select_splits(
            scored,
            threshold=float(args.threshold),
            min_local_aligned_count=int(args.min_local_aligned_count),
            min_component_area_to_local_median=float(args.min_component_area_to_local_median),
            max_split_area_per_uprn_log_dev_mean=float(args.max_split_area_per_uprn_log_dev_mean),
            min_split_regularity=float(args.min_split_regularity),
        )
        review = scored[
            scored["overmerge_split_proba"].astype(float).ge(float(args.review_threshold))
            & ~scored["edge_fid"].astype(int).isin(set(selected["edge_fid"].astype(int)))
        ].copy()

    if str(args.candidate_csv).strip():
        candidate_csv = Path(args.candidate_csv)
        candidate_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scored.drop(columns="geometry", errors="ignore")).to_csv(candidate_csv, index=False)

    remove_edge_fids = set(selected["edge_fid"].astype(int)) if not selected.empty else set()
    _log(f"[INFO] Overmerge split candidates={len(scored):,}; selected_edges={len(remove_edge_fids):,}")
    edges_remaining = edges[~edges.index.astype(int).isin(remove_edge_fids)].copy()

    sources_new = sources.copy()
    sources_new["overmerge_split_status"] = "base"
    sources_new["overmerge_split_from_component"] = np.nan
    sources_new["overmerge_split_removed_edge_fid"] = np.nan
    sources_new["overmerge_split_proba"] = np.nan
    if remove_edge_fids:
        affected_components = set(selected["component_id"].astype(int))
        sources_new.loc[
            sources_new["pred_component_id"].astype(int).isin(affected_components),
            "overmerge_split_status",
        ] = "split_component"
        comp_to_edge = selected.set_index("component_id")["edge_fid"].astype(int).to_dict()
        comp_to_proba = selected.set_index("component_id")["overmerge_split_proba"].astype(float).to_dict()
        sources_new["overmerge_split_from_component"] = sources_new["pred_component_id"].where(
            sources_new["pred_component_id"].astype(int).isin(affected_components),
            np.nan,
        )
        sources_new["overmerge_split_removed_edge_fid"] = sources_new["pred_component_id"].astype(int).map(comp_to_edge)
        sources_new["overmerge_split_proba"] = sources_new["pred_component_id"].astype(int).map(comp_to_proba)

    sources_new["pred_component_id"] = _component_ids(sources_new["source_fid"], edges_remaining).to_numpy()
    edges_new = _filter_edges_to_current_components(edges_remaining, sources_new)
    predicted_new = _build_predicted_parcels_from_sources(sources_new, edges_new)

    predicted_no_uprn = predicted_new.drop(columns=["pred_uprn_count"])
    merged_only = predicted_new[predicted_new["source_count"].gt(1)].copy()
    merged_only_no_uprn = merged_only.drop(columns=["pred_uprn_count"])
    possible_fp = predicted_new[predicted_new["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted_new[predicted_new["possible_split_reference"].eq(1)].copy()

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted_no_uprn, output_gpkg, "predicted_parcels")
    _write_layer(merged_only_no_uprn, output_gpkg, "predicted_parcels_merged_only")
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
    _write_layer(selected, output_gpkg, "overmerge_split_selected")
    _write_layer(review, output_gpkg, "overmerge_split_review_candidates")
    top = scored.sort_values("overmerge_split_proba", ascending=False).head(2000).copy() if not scored.empty else scored.copy()
    _write_layer(top, output_gpkg, "overmerge_split_candidates_top")

    summary = {
        "input_gpkg": str(input_gpkg),
        "output_gpkg": str(output_gpkg),
        "model_dir": str(model_dir),
        "threshold": float(args.threshold),
        "review_threshold": float(args.review_threshold),
        "candidate_rows": int(len(scored)),
        "selected_edges": int(len(remove_edge_fids)),
        "review_rows": int(len(review)),
        "selected_components": int(selected["component_id"].nunique()) if not selected.empty else 0,
        "max_component_area": float(args.max_component_area),
        "max_component_source_count": int(args.max_component_source_count),
        "min_component_uprn_count": int(args.min_component_uprn_count),
        "local_radius": float(args.local_radius),
        "local_angle_tolerance": float(args.local_angle_tolerance),
        "min_local_aligned_count": int(args.min_local_aligned_count),
        "min_component_area_to_local_median": float(args.min_component_area_to_local_median),
        "max_split_area_per_uprn_log_dev_mean": float(args.max_split_area_per_uprn_log_dev_mean),
        "min_split_regularity": float(args.min_split_regularity),
        "predicted_components": int(len(predicted_new)),
        "merged_only_components": int(len(merged_only)),
        "possible_false_positive_clusters": int(len(possible_fp)),
        "possible_split_reference_clusters": int(len(possible_split)),
        "merged_only_uprn_counts": merged_only["pred_uprn_count"].value_counts().sort_index().astype(int).to_dict(),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Overmerge split complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.errors import GEOSException

import _bootstrap  # noqa: F401
from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_RAW_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg"
DEFAULT_RAW_LAYER = "polygons_in_buffers"
DEFAULT_NATIVE_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_merge_native.gpkg"
DEFAULT_NATIVE_LAYER = "predicted_parcels_with_uprn"
DEFAULT_COUNCIL_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_merge_council.gpkg"
DEFAULT_COUNCIL_LAYER = "os_wfs_merge"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_merge_final.gpkg"

FINAL_LAYER = "wfs_merged_final"
RETAINED_LAYER = "native_retained"
ACCEPTED_LAYER = "council_group_patches_accepted"
REVIEW_LAYER = "council_group_patches_review"
REJECTED_LAYER = "council_group_patches_rejected"

TEXT_ID_LIMIT = 6000


def _log(message: str) -> None:
    print(message, flush=True)


def _ids_text(values: list[int] | np.ndarray | pd.Series) -> str:
    ids = [str(int(v)) for v in values if pd.notna(v)]
    text = ",".join(ids)
    if len(text) <= TEXT_ID_LIMIT:
        return text
    return text[: TEXT_ID_LIMIT - 20] + f"...(+{max(len(ids) - text[:TEXT_ID_LIMIT].count(','), 0)} ids)"


def _parse_ids(text: Any) -> list[int]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    return [int(token) for token in re.findall(r"\d+", str(text))]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _metric(geom: Any, key: str, default: float = 0.0) -> float:
    try:
        return _safe_float(_shape_metrics(geom).get(key), default)
    except (GEOSException, ValueError, TypeError):
        return default


def _all_metrics(geom: Any) -> dict[str, float]:
    try:
        metrics = _shape_metrics(geom)
    except (GEOSException, ValueError, TypeError):
        return {
            "area": _safe_float(shapely.area(geom)),
            "perimeter": _safe_float(shapely.length(geom)),
            "mrr_ratio": 0.0,
            "mrr_gap_ratio": 0.0,
            "hull_gap_ratio": 0.0,
            "regularity_score": 0.0,
        }
    return {
        "area": _safe_float(metrics.get("area")),
        "perimeter": _safe_float(metrics.get("perimeter")),
        "mrr_ratio": _safe_float(metrics.get("mrr_ratio")),
        "mrr_gap_ratio": _safe_float(metrics.get("mrr_gap_ratio")),
        "hull_gap_ratio": _safe_float(metrics.get("hull_gap_ratio")),
        "regularity_score": _safe_float(metrics.get("regularity_score")),
    }


def _read_layer(path: str, layer: str, *, fid_name: str) -> gpd.GeoDataFrame:
    gdf = pyogrio.read_dataframe(path, layer=layer, fid_as_index=True)
    gdf[fid_name] = gdf.index.astype("int64")
    if gdf.geometry.name != "geometry":
        gdf = gdf.set_geometry(gdf.geometry.name).rename_geometry("geometry")
    return gdf


def _native_complete_like(row: pd.Series) -> bool:
    regularity = _safe_float(row.get("pred_regularity_score"), _safe_float(row.get("regularity_score")))
    mrr_ratio = _safe_float(row.get("pred_mrr_ratio"), _safe_float(row.get("mrr_ratio")))
    hull_gap = _safe_float(row.get("pred_hull_gap_ratio"), _safe_float(row.get("hull_gap_ratio")))
    area = _safe_float(row.get("native_area"))
    uprn = _safe_int(row.get("pred_uprn_count"))
    if area <= 15.0:
        return False
    return uprn > 0 and regularity >= 0.88 and mrr_ratio >= 0.72 and hull_gap <= 0.18


def _candidate_decision(
    *,
    native_count: int,
    council_area: float,
    coverage_ratio: float,
    weighted_native_inside_ratio: float,
    native_uprn_nonzero_count: int,
    native_complete_like_count: int,
    possible_split_ratio: float,
    largest_native_area_ratio: float,
    max_auto_area: float,
    max_review_area: float,
    coverage_threshold: float,
    native_inside_threshold: float,
) -> tuple[str, str, float]:
    complete_ratio = native_complete_like_count / max(native_count, 1)
    complete_uprn_ratio = native_complete_like_count / max(native_uprn_nonzero_count, 1)
    many_complete_native_risk = (
        native_uprn_nonzero_count >= 2
        and native_complete_like_count >= 2
        and complete_ratio >= 0.75
    )
    multi_complete_uprn_overmerge_risk = (
        native_uprn_nonzero_count >= 2
        and native_complete_like_count >= 2
        and complete_uprn_ratio >= 0.75
        and largest_native_area_ratio < 0.55
    )
    distributed_uprn_overmerge_risk = (
        native_uprn_nonzero_count >= 3
        and possible_split_ratio >= 0.80
        and largest_native_area_ratio < 0.45
    )
    overmerge_risk = (
        native_uprn_nonzero_count >= 2
        and native_complete_like_count >= 2
        and possible_split_ratio < 0.80
    ) or (
        native_complete_like_count >= 3
        and council_area > max_auto_area
        and possible_split_ratio < 0.80
    )
    fragment_repair_signal = (
        possible_split_ratio >= 0.50
        or native_complete_like_count <= 1
        or native_uprn_nonzero_count <= 1
    )
    score = (
        100.0 * coverage_ratio
        + 30.0 * weighted_native_inside_ratio
        + 12.0 * possible_split_ratio
        + 6.0 * (1.0 - complete_ratio)
        - min(council_area / max(max_review_area, 1.0), 3.0)
    )

    if native_count < 2:
        return "reject", "single_native_overlap", score
    if coverage_ratio < coverage_threshold:
        return "reject", "low_council_coverage", score
    if weighted_native_inside_ratio < native_inside_threshold:
        return "reject", "native_not_inside_council", score
    if council_area > max_review_area:
        return "reject", "too_large_for_hybrid_patch", score
    if many_complete_native_risk:
        return "reject", "overmerge_risk_many_complete_native", score
    if multi_complete_uprn_overmerge_risk:
        return "reject", "overmerge_risk_multiple_complete_uprn_native", score
    if distributed_uprn_overmerge_risk:
        return "reject", "overmerge_risk_distributed_uprn_native", score
    if overmerge_risk:
        return "reject", "overmerge_risk_multiple_complete_native", score
    if council_area <= max_auto_area and fragment_repair_signal:
        return "accept_prelim", "fragment_repair_high_confidence", score
    if council_area <= max_review_area and fragment_repair_signal:
        return "review", "medium_area_fragment_repair", score
    return "review", "uncertain_native_vs_council_grouping", score


def _build_patch_candidates(
    native: gpd.GeoDataFrame,
    council: gpd.GeoDataFrame,
    *,
    chunk_size: int,
    min_intersection_area: float,
    min_candidate_coverage: float,
    select_native_inside_threshold: float,
    coverage_threshold: float,
    native_inside_threshold: float,
    max_auto_area: float,
    max_review_area: float,
) -> gpd.GeoDataFrame:
    native = native.reset_index(drop=True).copy()
    council = council.reset_index(drop=True).copy()
    native["native_area"] = shapely.area(native.geometry.array).astype(float)
    council["council_area"] = shapely.area(council.geometry.array).astype(float)
    native["native_complete_like"] = native.apply(_native_complete_like, axis=1)

    sindex = native.sindex
    native_geoms = native.geometry
    records: list[dict[str, Any]] = []
    geometries: list[Any] = []

    total = len(council)
    _log(f"[INFO] Scanning council parcels against native base: {total:,} council features")
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk = council.iloc[start:stop]
        if chunk.empty:
            continue

        left, right = sindex.query(chunk.geometry.array, predicate="intersects")
        if len(left) == 0:
            _log(f"[INFO] chunk {start:,}-{stop:,}: no overlaps")
            continue

        by_left: dict[int, list[int]] = {}
        for local_pos, native_pos in zip(left.tolist(), right.tolist()):
            by_left.setdefault(int(local_pos), []).append(int(native_pos))

        for local_pos, native_positions in by_left.items():
            council_row = chunk.iloc[local_pos]
            council_geom = council_row.geometry
            council_area = _safe_float(council_row.get("council_area"))
            if council_area <= 0.0 or council_geom is None or shapely.is_empty(council_geom):
                continue

            npos = np.array(native_positions, dtype=np.int64)
            nrows = native.iloc[npos]
            try:
                inter_geoms = shapely.intersection(nrows.geometry.array, council_geom)
                inter_area = np.asarray(shapely.area(inter_geoms), dtype=float)
            except GEOSException:
                fixed_council = shapely.make_valid(council_geom)
                fixed_native = shapely.make_valid(nrows.geometry.array)
                inter_area = np.asarray(shapely.area(shapely.intersection(fixed_native, fixed_council)), dtype=float)

            native_area = nrows["native_area"].to_numpy(dtype=float)
            valid = np.isfinite(inter_area) & (inter_area >= min_intersection_area) & (native_area > 0)
            if valid.sum() < 2:
                continue

            npos = npos[valid]
            nrows = native.iloc[npos]
            inter_area = inter_area[valid]
            native_area = native_area[valid]
            native_inside = inter_area / np.maximum(native_area, 1e-9)
            council_share = inter_area / max(council_area, 1e-9)
            selected = (native_inside >= select_native_inside_threshold) | (council_share >= 0.02)
            if int(selected.sum()) < 2:
                continue

            selected_npos = npos[selected]
            selected_rows = native.iloc[selected_npos]
            selected_inter_area = inter_area[selected]
            selected_native_area = native_area[selected]
            native_area_sum = float(selected_native_area.sum())
            if native_area_sum <= 0.0:
                continue

            coverage_ratio = float(selected_inter_area.sum() / max(council_area, 1e-9))
            weighted_inside_ratio = float(selected_inter_area.sum() / max(native_area_sum, 1e-9))
            if coverage_ratio < min_candidate_coverage and weighted_inside_ratio < 0.98:
                continue

            native_count = int(len(selected_rows))
            possible_split = selected_rows.get("possible_split_reference", pd.Series(0, index=selected_rows.index))
            possible_split_count = int(pd.to_numeric(possible_split, errors="coerce").fillna(0).gt(0).sum())
            possible_split_ratio = possible_split_count / max(native_count, 1)
            uprn = pd.to_numeric(selected_rows.get("pred_uprn_count", 0), errors="coerce").fillna(0)
            native_uprn_sum = int(uprn.sum())
            native_uprn_nonzero_count = int((uprn > 0).sum())
            native_complete_like_count = int(selected_rows["native_complete_like"].sum())
            largest_native_area_ratio = float(selected_native_area.max() / max(native_area_sum, 1e-9))
            area_delta_ratio = float(abs(native_area_sum - council_area) / max(council_area, 1e-9))

            decision, reason, score = _candidate_decision(
                native_count=native_count,
                council_area=council_area,
                coverage_ratio=coverage_ratio,
                weighted_native_inside_ratio=weighted_inside_ratio,
                native_uprn_nonzero_count=native_uprn_nonzero_count,
                native_complete_like_count=native_complete_like_count,
                possible_split_ratio=possible_split_ratio,
                largest_native_area_ratio=largest_native_area_ratio,
                max_auto_area=max_auto_area,
                max_review_area=max_review_area,
                coverage_threshold=coverage_threshold,
                native_inside_threshold=native_inside_threshold,
            )

            try:
                patch_geom = shapely.union_all(selected_rows.geometry.array)
            except GEOSException:
                patch_geom = shapely.union_all(shapely.make_valid(selected_rows.geometry.array))
            if patch_geom is None or shapely.is_empty(patch_geom):
                continue

            shape = _all_metrics(patch_geom)
            raw_source_fids = _parse_ids(council_row.get("merge_source_fids"))
            records.append(
                {
                    "council_fid": int(council_row["council_fid"]),
                    "decision": decision,
                    "reason": reason,
                    "decision_score": float(score),
                    "native_count": native_count,
                    "native_fids": _ids_text(selected_rows["native_fid"].to_numpy()),
                    "native_uprn_sum": native_uprn_sum,
                    "native_uprn_nonzero_count": native_uprn_nonzero_count,
                    "native_complete_like_count": native_complete_like_count,
                    "possible_split_count": possible_split_count,
                    "possible_split_ratio": float(possible_split_ratio),
                    "coverage_ratio": float(coverage_ratio),
                    "weighted_native_inside_ratio": float(weighted_inside_ratio),
                    "native_area_sum": native_area_sum,
                    "council_area": council_area,
                    "area_delta_ratio": area_delta_ratio,
                    "largest_native_area_ratio": largest_native_area_ratio,
                    "council_merge_stage": str(council_row.get("merge_stage", "")),
                    "council_merge_source_count": _safe_int(council_row.get("merge_source_count")),
                    "raw_source_fid_count": len(raw_source_fids),
                    "raw_source_fids": _ids_text(raw_source_fids),
                    "patch_area": shape["area"],
                    "patch_perimeter": shape["perimeter"],
                    "patch_mrr_ratio": shape["mrr_ratio"],
                    "patch_hull_gap_ratio": shape["hull_gap_ratio"],
                    "patch_regularity_score": shape["regularity_score"],
                }
            )
            geometries.append(patch_geom)

        _log(f"[INFO] chunk {start:,}-{stop:,}: candidates so far={len(records):,}")

    if not records:
        return gpd.GeoDataFrame(records, geometry=[], crs=native.crs)
    return gpd.GeoDataFrame(records, geometry=geometries, crs=native.crs)


def _resolve_accepted_conflicts(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    used_native: set[int] = set()
    accepted_indices: list[Any] = []
    conflict_indices: list[Any] = []
    prelim = out[out["decision"].eq("accept_prelim")].sort_values(
        ["decision_score", "coverage_ratio", "possible_split_ratio"],
        ascending=[False, False, False],
    )
    for idx, row in prelim.iterrows():
        native_ids = set(_parse_ids(row["native_fids"]))
        if not native_ids:
            conflict_indices.append(idx)
            continue
        if used_native.intersection(native_ids):
            conflict_indices.append(idx)
            continue
        used_native.update(native_ids)
        accepted_indices.append(idx)

    if accepted_indices:
        out.loc[accepted_indices, "decision"] = "accept"
        out.loc[accepted_indices, "reason"] = "fragment_repair_high_confidence_selected"
    if conflict_indices:
        out.loc[conflict_indices, "decision"] = "review"
        out.loc[conflict_indices, "reason"] = "conflicts_with_higher_score_patch"
    return out


def _make_final_layers(
    native: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    accepted = candidates[candidates["decision"].eq("accept")].copy()
    used_native: set[int] = set()
    for text in accepted.get("native_fids", pd.Series(dtype=str)):
        used_native.update(_parse_ids(text))

    native = native.copy()
    native["native_area"] = shapely.area(native.geometry.array).astype(float)
    retained = native[~native["native_fid"].isin(used_native)].copy()

    retained_records = pd.DataFrame(
        {
            "hybrid_source": "native",
            "source_native_fids": retained["native_fid"].astype(str),
            "source_council_fid": -1,
            "patch_reason": "",
            "native_count": 1,
            "source_count": pd.to_numeric(retained.get("source_count", 1), errors="coerce").fillna(1).astype(int),
            "pred_uprn_count": pd.to_numeric(retained.get("pred_uprn_count", 0), errors="coerce").fillna(0).astype(int),
            "area": retained["native_area"].astype(float),
            "perimeter": pd.to_numeric(retained.get("pred_perimeter", 0), errors="coerce").fillna(0.0),
            "mrr_ratio": pd.to_numeric(retained.get("pred_mrr_ratio", 0), errors="coerce").fillna(0.0),
            "hull_gap_ratio": pd.to_numeric(retained.get("pred_hull_gap_ratio", 0), errors="coerce").fillna(0.0),
            "regularity_score": pd.to_numeric(retained.get("pred_regularity_score", 0), errors="coerce").fillna(0.0),
        }
    )
    retained_out = gpd.GeoDataFrame(retained_records, geometry=retained.geometry.to_numpy(), crs=native.crs)

    patch_records: list[dict[str, Any]] = []
    patch_geoms: list[Any] = []
    native_by_fid = native.set_index("native_fid", drop=False)
    for _, row in accepted.iterrows():
        native_ids = _parse_ids(row["native_fids"])
        native_rows = native_by_fid.loc[[fid for fid in native_ids if fid in native_by_fid.index]]
        source_count = int(pd.to_numeric(native_rows.get("source_count", 1), errors="coerce").fillna(1).sum())
        patch_records.append(
            {
                "hybrid_source": "council_patch",
                "source_native_fids": row["native_fids"],
                "source_council_fid": int(row["council_fid"]),
                "patch_reason": row["reason"],
                "native_count": int(row["native_count"]),
                "source_count": source_count,
                "pred_uprn_count": int(row["native_uprn_sum"]),
                "area": float(row["patch_area"]),
                "perimeter": float(row["patch_perimeter"]),
                "mrr_ratio": float(row["patch_mrr_ratio"]),
                "hull_gap_ratio": float(row["patch_hull_gap_ratio"]),
                "regularity_score": float(row["patch_regularity_score"]),
            }
        )
        patch_geoms.append(row.geometry)
    patch_out = gpd.GeoDataFrame(patch_records, geometry=patch_geoms, crs=native.crs)

    final = pd.concat([retained_out, patch_out], ignore_index=True)
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=native.crs)
    final.insert(0, "hybrid_id", np.arange(1, len(final) + 1, dtype=np.int64))
    retained_out.insert(0, "hybrid_id", np.arange(1, len(retained_out) + 1, dtype=np.int64))
    patch_out.insert(0, "hybrid_id", np.arange(len(retained_out) + 1, len(retained_out) + len(patch_out) + 1, dtype=np.int64))
    return final, retained_out, patch_out


def _write_layer(gdf: gpd.GeoDataFrame, output_gpkg: Path, layer: str) -> None:
    if gdf.empty:
        _log(f"[WARN] Skip empty layer: {layer}")
        return
    _log(f"[INFO] Writing {layer}: {len(gdf):,} features")
    gdf.to_file(output_gpkg, layer=layer, driver="GPKG", engine="pyogrio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final hybrid WFS merge layer from raw WFS, native merge, and council merge outputs."
    )
    parser.add_argument("--raw-gpkg", default=DEFAULT_RAW_GPKG)
    parser.add_argument("--raw-layer", default=DEFAULT_RAW_LAYER)
    parser.add_argument("--native-gpkg", default=DEFAULT_NATIVE_GPKG)
    parser.add_argument("--native-layer", default=DEFAULT_NATIVE_LAYER)
    parser.add_argument("--council-gpkg", default=DEFAULT_COUNCIL_GPKG)
    parser.add_argument("--council-layer", default=DEFAULT_COUNCIL_LAYER)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--min-intersection-area", type=float, default=0.01)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--select-native-inside-threshold", type=float, default=0.985)
    parser.add_argument("--coverage-threshold", type=float, default=0.985)
    parser.add_argument("--native-inside-threshold", type=float, default=0.985)
    parser.add_argument("--max-auto-area", type=float, default=3000.0)
    parser.add_argument("--max-review-area", type=float, default=12000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_gpkg = Path(args.output_gpkg).resolve()
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if output_gpkg.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_gpkg}")
        output_gpkg.unlink()

    raw_info = pyogrio.read_info(args.raw_gpkg, layer=args.raw_layer)
    _log(
        "[INFO] Raw WFS input: "
        f"{args.raw_gpkg}::{args.raw_layer}, features={raw_info.get('features')}, crs={raw_info.get('crs')}"
    )

    _log(f"[INFO] Reading native: {args.native_gpkg}::{args.native_layer}")
    native = _read_layer(args.native_gpkg, args.native_layer, fid_name="native_fid")
    _log(f"[INFO] Native features={len(native):,}, crs={native.crs}")

    _log(f"[INFO] Reading council: {args.council_gpkg}::{args.council_layer}")
    council = _read_layer(args.council_gpkg, args.council_layer, fid_name="council_fid")
    if council.crs != native.crs:
        council = council.to_crs(native.crs)
    _log(f"[INFO] Council features={len(council):,}, crs={council.crs}")

    candidates = _build_patch_candidates(
        native,
        council,
        chunk_size=args.chunk_size,
        min_intersection_area=args.min_intersection_area,
        min_candidate_coverage=args.min_candidate_coverage,
        select_native_inside_threshold=args.select_native_inside_threshold,
        coverage_threshold=args.coverage_threshold,
        native_inside_threshold=args.native_inside_threshold,
        max_auto_area=args.max_auto_area,
        max_review_area=args.max_review_area,
    )
    candidates = _resolve_accepted_conflicts(candidates)
    final, retained, accepted_patches = _make_final_layers(native, candidates)
    accepted_candidates = candidates[candidates["decision"].eq("accept")].copy()

    review = candidates[candidates["decision"].eq("review")].copy()
    rejected = candidates[candidates["decision"].eq("reject")].copy()
    rejected = rejected.sort_values(["decision_score", "coverage_ratio"], ascending=[False, False]).head(5000)

    _write_layer(final, output_gpkg, FINAL_LAYER)
    _write_layer(retained, output_gpkg, RETAINED_LAYER)
    _write_layer(accepted_candidates, output_gpkg, ACCEPTED_LAYER)
    _write_layer(review, output_gpkg, REVIEW_LAYER)
    _write_layer(rejected, output_gpkg, REJECTED_LAYER)

    summary = {
        "raw_gpkg": args.raw_gpkg,
        "raw_layer": args.raw_layer,
        "raw_features": raw_info.get("features"),
        "native_gpkg": args.native_gpkg,
        "native_layer": args.native_layer,
        "native_features": int(len(native)),
        "council_gpkg": args.council_gpkg,
        "council_layer": args.council_layer,
        "council_features": int(len(council)),
        "output_gpkg": str(output_gpkg),
        "final_layer": FINAL_LAYER,
        "final_features": int(len(final)),
        "native_retained_features": int(len(retained)),
        "accepted_patch_features": int(len(accepted_patches)),
        "accepted_patch_qa_features": int(len(accepted_candidates)),
        "review_patch_features": int(len(review)),
        "rejected_patch_features_written": int(len(rejected)),
        "candidate_features_total": int(len(candidates)),
        "thresholds": {
            "min_candidate_coverage": args.min_candidate_coverage,
            "select_native_inside_threshold": args.select_native_inside_threshold,
            "coverage_threshold": args.coverage_threshold,
            "native_inside_threshold": args.native_inside_threshold,
            "max_auto_area": args.max_auto_area,
            "max_review_area": args.max_review_area,
        },
        "candidate_decisions": candidates["decision"].value_counts(dropna=False).to_dict()
        if not candidates.empty
        else {},
        "candidate_reasons": candidates["reason"].value_counts(dropna=False).head(30).to_dict()
        if not candidates.empty
        else {},
    }
    summary_path = output_gpkg.with_suffix(output_gpkg.suffix + ".hybrid_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[INFO] Summary: {summary_path}")
    _log("[DONE] Hybrid final WFS merge complete")


if __name__ == "__main__":
    main()

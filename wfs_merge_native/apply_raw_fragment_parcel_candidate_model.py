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

from apply_wfs_merge_completion_model import _write_layer
from train_raw_fragment_parcel_candidate_model import (
    _add_pool_rank_features,
    _candidate_features,
    _ids_text,
    _parse_id_set,
)
from train_wfs_raw_anchor_group_model import (
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_ID_FIELD,
    DEFAULT_UPRN_LAYER,
    DEFAULT_WFS_CLEAN_GPKG,
    DEFAULT_WFS_CLEAN_LAYER,
    _add_uprn_counts,
    _build_edges,
    _build_source_indexes,
    _collect_anchor_pool,
    _enumerate_anchor_groups_ordered,
    _parse_bbox,
    _read_clean_wfs,
)


DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/raw_fragment_parcel_candidate_model_full_noleak_poolrank_v1/"
    "raw_fragment_parcel_candidate_model_v1.joblib"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/raw_fragment_parcel_candidate_model_full_noleak_poolrank_v1/"
    "raw_fragment_parcel_candidate_bbox_preview.gpkg"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _param(args: argparse.Namespace, params: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return params.get(name, default)


def _as_source_role(value: object) -> str:
    text = str(value or "").lower()
    if text in {"building", "land", "gapfill", "road", "other"}:
        return text
    return "other"


def _union_geoms(clean_ids: set[int] | frozenset[int], geom_by_clean: dict[int, Any]) -> Any:
    geoms = [geom_by_clean[int(fid)] for fid in sorted(clean_ids) if int(fid) in geom_by_clean]
    if not geoms:
        return None
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _source_indexes_for_fragment_model(
    wfs: gpd.GeoDataFrame,
) -> tuple[dict[int, list[Any]], dict[int, dict[str, Any]]]:
    geom_by_source: dict[int, list[Any]] = {}
    attrs_by_source: dict[int, dict[str, Any]] = {}
    for source_fid, group in wfs.groupby(wfs["source_fid"].astype(int)):
        source = int(source_fid)
        geom_by_source[source] = list(group.geometry)
        roles = [_as_source_role(value) for value in group.get("source_role", group["anchor_role"])]
        role_counts = {role: roles.count(role) for role in sorted(set(roles))}
        attrs_by_source[source] = {
            "source_fid": source,
            "area": float(group["area"].sum()),
            "perimeter": float(group["perimeter"].sum()),
            "source_role": max(role_counts, key=role_counts.get) if role_counts else "other",
            "role_counts": role_counts,
            "part_count": int(len(group)),
            "hole_fill_count": int(pd.to_numeric(group.get("is_polygon_hole_fill", 0), errors="coerce").fillna(0).sum()),
            "gap_fill_count": int(pd.to_numeric(group.get("is_enclosed_gap_fill", 0), errors="coerce").fillna(0).sum()),
        }
    return geom_by_source, attrs_by_source


def _dummy_target_row(anchor_source_fid: int, candidate_source_ids: set[int]) -> pd.Series:
    return pd.Series(
        {
            "anchor_wfs_fid": int(anchor_source_fid),
            "target_source_set": set(int(v) for v in candidate_source_ids),
            "train_component_id": -1,
            "split": "apply",
            "spatial_group": "apply",
            "anchor_kind": "",
        }
    )


def _candidate_clean_ids(source_ids: set[int], source_to_clean: dict[int, list[int]]) -> frozenset[int]:
    clean_ids: set[int] = set()
    for source_fid in source_ids:
        clean_ids.update(int(v) for v in source_to_clean.get(int(source_fid), []))
    return frozenset(clean_ids)


def _score_candidates(
    *,
    anchors: list[int],
    pipeline: Any,
    feature_cols: list[str],
    wfs: gpd.GeoDataFrame,
    source_to_clean: dict[int, list[int]],
    source_by_clean: dict[int, int],
    eligible_clean_ids: set[int],
    adjacency: dict[int, list[tuple[int, float]]],
    geom_by_source: dict[int, list[Any]],
    attrs_by_source: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()
    all_anchor_sources = set(int(v) for v in anchors)
    neighbor_depth = int(_param(args, params, "neighbor_depth", 3))
    max_pool_size = int(_param(args, params, "max_pool_size", 22))
    max_group_size = int(_param(args, params, "max_group_size", 7))
    max_candidate_area = float(_param(args, params, "max_candidate_area", 2500.0))
    per_anchor_candidate_limit = int(_param(args, params, "per_anchor_candidate_limit", 80))

    area_by_clean = wfs.set_index("clean_fid")["area"].astype(float).to_dict()
    for anchor_index, anchor_source_fid in enumerate(anchors, start=1):
        anchor_clean_ids = frozenset(source_to_clean.get(int(anchor_source_fid), []))
        if not anchor_clean_ids:
            continue
        pool = _collect_anchor_pool(
            anchor_clean_ids=anchor_clean_ids,
            positive_clean_ids=frozenset(),
            adjacency=adjacency,
            eligible_clean_ids=eligible_clean_ids,
            max_depth=neighbor_depth,
            max_pool_size=max_pool_size,
        )
        groups = [anchor_clean_ids]
        groups.extend(
            _enumerate_anchor_groups_ordered(
                anchor_clean_ids=anchor_clean_ids,
                pool=pool,
                adjacency=adjacency,
                area_by_clean=area_by_clean,
                max_group_size=max_group_size,
                max_candidate_area=max_candidate_area,
                per_anchor_limit=per_anchor_candidate_limit,
            )
        )
        for group in groups:
            source_ids = {int(source_by_clean[fid]) for fid in group if int(fid) in source_by_clean}
            if int(anchor_source_fid) not in source_ids:
                source_ids.add(int(anchor_source_fid))
            key = (int(anchor_source_fid), _ids_text(source_ids))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rec = _candidate_features(
                candidate_id=f"apply:{int(anchor_source_fid)}:{_ids_text(source_ids)}",
                target_row=_dummy_target_row(int(anchor_source_fid), source_ids),
                candidate_sources=source_ids,
                label=0,
                negative_type="apply",
                geom_by_source=geom_by_source,
                attrs_by_source=attrs_by_source,
                anchor_source_ids={int(anchor_source_fid)},
                other_anchor_source_ids=all_anchor_sources - {int(anchor_source_fid)},
            )
            if rec is None:
                continue
            clean_ids = _candidate_clean_ids(source_ids, source_to_clean)
            rec["anchor_clean_fids"] = _ids_text(anchor_clean_ids)
            rec["candidate_clean_fids"] = _ids_text(clean_ids)
            rec["candidate_source_fids"] = _ids_text(source_ids)
            records.append(rec)
        if anchor_index % 500 == 0:
            _log(f"[INFO] Built apply candidates for anchors={anchor_index:,}/{len(anchors):,}; rows={len(records):,}")

    stats = {
        "anchor_rows": int(len(anchors)),
        "candidate_rows_generated": int(len(records)),
    }
    if not records:
        return pd.DataFrame(), stats

    candidates = pd.DataFrame.from_records(records)
    candidates = _add_pool_rank_features(candidates)
    for column in feature_cols:
        if column not in candidates.columns:
            candidates[column] = np.nan
    candidates["raw_fragment_proba"] = pipeline.predict_proba(candidates[feature_cols])[:, 1]
    candidates["raw_anchor_group_proba"] = candidates["raw_fragment_proba"]
    candidates["group_regularity_score"] = candidates.get("shape_regularity_score", np.nan)
    candidates["group_hull_gap_ratio"] = candidates.get("shape_hull_gap_ratio", np.nan)
    return candidates, stats


def _select_candidates(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    anchor_owner_by_clean: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), candidates.copy(), candidates.copy()
    high = candidates[candidates["raw_fragment_proba"].ge(float(threshold))].copy()
    review = candidates[candidates["raw_fragment_proba"].lt(float(threshold))].copy()
    if high.empty:
        return high, review.sort_values("raw_fragment_proba", ascending=False), high.copy()
    sort_cols = [
        "raw_fragment_proba",
        "candidate_source_count",
        "source_area_sum",
        "shape_regularity_score",
        "shape_hull_gap_ratio",
    ]
    for column in sort_cols:
        if column not in high.columns:
            high[column] = np.nan
    high = high.sort_values(sort_cols, ascending=[False, False, False, False, True])
    selected_rows: list[pd.Series] = []
    conflict_rows: list[pd.Series] = []
    claimed_clean: set[int] = set()
    selected_anchors: set[int] = set()
    for _, row in high.iterrows():
        anchor_source_fid = int(row.anchor_source_fid)
        clean_ids = _parse_id_set(row.candidate_clean_fids)
        other_anchor_clean = {
            clean_id
            for clean_id in clean_ids
            if clean_id in anchor_owner_by_clean and int(anchor_owner_by_clean[clean_id]) != anchor_source_fid
        }
        conflict_reason = ""
        if anchor_source_fid in selected_anchors:
            conflict_reason = "anchor_already_selected"
        elif other_anchor_clean:
            conflict_reason = "contains_other_anchor"
        elif clean_ids & claimed_clean:
            conflict_reason = "clean_fid_already_claimed"
        if conflict_reason:
            conflict = row.copy()
            conflict["conflict_reason"] = conflict_reason
            conflict["conflict_clean_fids"] = _ids_text((clean_ids & claimed_clean) | other_anchor_clean)
            conflict_rows.append(conflict)
            continue
        selected = row.copy()
        selected["conflict_reason"] = ""
        selected["conflict_clean_fids"] = ""
        selected_rows.append(selected)
        selected_anchors.add(anchor_source_fid)
        claimed_clean.update(clean_ids)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else high.iloc[0:0].copy()
    conflicts = pd.DataFrame(conflict_rows).reset_index(drop=True) if conflict_rows else high.iloc[0:0].copy()
    return selected, review.sort_values("raw_fragment_proba", ascending=False), conflicts


def _candidate_geometries(
    rows: pd.DataFrame,
    geom_by_clean: dict[int, Any],
    crs: Any,
    *,
    limit: int,
) -> gpd.GeoDataFrame:
    if rows.empty:
        return gpd.GeoDataFrame(rows.copy(), geometry=[], crs=crs)
    work = rows.sort_values("raw_fragment_proba", ascending=False).head(int(limit)).copy()
    geoms = [_union_geoms(_parse_id_set(value), geom_by_clean) for value in work["candidate_clean_fids"]]
    return gpd.GeoDataFrame(work, geometry=geoms, crs=crs)


def _build_output_parcels(
    *,
    wfs: gpd.GeoDataFrame,
    selected: pd.DataFrame,
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    source_by_clean: dict[int, int],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    claimed_clean: set[int] = set()
    merged_records: list[dict[str, Any]] = []
    for plot_id, row in enumerate(selected.itertuples(index=False), start=1):
        clean_ids = _parse_id_set(getattr(row, "candidate_clean_fids"))
        if not clean_ids:
            continue
        claimed_clean.update(clean_ids)
        source_ids = {int(source_by_clean[fid]) for fid in clean_ids if fid in source_by_clean}
        uprn_count = sum(int(attrs_by_clean[fid].get("uprn_count", 0) or 0) for fid in clean_ids if fid in attrs_by_clean)
        geom = _union_geoms(clean_ids, geom_by_clean)
        merged_records.append(
            {
                "raw_anchor_group_plot_id": int(plot_id),
                "merge_status": "raw_fragment_model_selected",
                "anchor_source_fid": int(getattr(row, "anchor_source_fid")),
                "clean_fids": _ids_text(clean_ids),
                "source_fids": _ids_text(source_ids),
                "clean_count": int(len(clean_ids)),
                "source_count": int(len(source_ids)),
                "uprn_count": int(uprn_count),
                "raw_fragment_proba": float(getattr(row, "raw_fragment_proba")),
                "raw_anchor_group_proba": float(getattr(row, "raw_fragment_proba")),
                "geometry": geom,
            }
        )

    base = wfs[~wfs["clean_fid"].astype(int).isin(claimed_clean)].copy()
    base_records: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        clean_fid = int(row.clean_fid)
        source_fid = int(row.source_fid)
        base_records.append(
            {
                "raw_anchor_group_plot_id": np.nan,
                "merge_status": "base",
                "anchor_source_fid": source_fid if int(getattr(row, "uprn_count", 0) or 0) > 0 else np.nan,
                "clean_fids": str(clean_fid),
                "source_fids": str(source_fid),
                "clean_count": 1,
                "source_count": 1,
                "uprn_count": int(getattr(row, "uprn_count", 0) or 0),
                "raw_fragment_proba": np.nan,
                "raw_anchor_group_proba": np.nan,
                "geometry": getattr(row, "geometry"),
            }
        )
    merged = gpd.GeoDataFrame(merged_records, geometry="geometry", crs=wfs.crs)
    base_gdf = gpd.GeoDataFrame(base_records, geometry="geometry", crs=wfs.crs)
    parcels = pd.concat([merged, base_gdf], ignore_index=True)
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=wfs.crs)
    return parcels, merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the raw-fragment parcel candidate model on a clean WFS bbox.")
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--candidate-output-csv", default="")
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--neighbor-depth", type=int, default=None)
    parser.add_argument("--max-pool-size", type=int, default=None)
    parser.add_argument("--max-group-size", type=int, default=None)
    parser.add_argument("--max-candidate-area", type=float, default=None)
    parser.add_argument("--per-anchor-candidate-limit", type=int, default=None)
    parser.add_argument("--top-neighbors", type=int, default=14)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--edge-query-chunk-size", type=int, default=20000)
    parser.add_argument("--edge-calc-chunk-size", type=int, default=50000)
    parser.add_argument("--debug-layer-limit", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    payload = joblib.load(args.model)
    if not isinstance(payload, dict) or payload.get("model_kind") != "raw_fragment_parcel_candidate_scorer":
        raise RuntimeError("Model payload must be a raw_fragment_parcel_candidate_scorer.")
    pipeline = payload["pipeline"]
    feature_cols = list(payload["feature_cols"])
    params = dict(payload.get("training_params", {}))
    threshold = float(args.threshold if args.threshold is not None else params.get("threshold_95p_from_train", 0.5))

    bbox = _parse_bbox(args.bbox)
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), bbox)
    wfs = _add_uprn_counts(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
    )
    wfs["source_role"] = wfs["anchor_role"].map(_as_source_role)
    source_to_clean, source_by_clean = _build_source_indexes(wfs)
    geom_by_clean = wfs.set_index("clean_fid").geometry.to_dict()
    attrs_by_clean = wfs.set_index("clean_fid").drop(columns="geometry").to_dict("index")
    geom_by_source, attrs_by_source = _source_indexes_for_fragment_model(wfs)

    anchor_mask = wfs["uprn_count"].astype(int).gt(0) & wfs["anchor_role"].isin(["building", "land"])
    anchors = sorted(set(int(value) for value in wfs.loc[anchor_mask, "source_fid"].astype(int)))
    if int(args.max_anchors) > 0:
        anchors = anchors[: int(args.max_anchors)]
    anchor_owner_by_clean = {
        int(clean_id): int(source_fid)
        for source_fid in anchors
        for clean_id in source_to_clean.get(int(source_fid), [])
    }
    _log(f"[INFO] Production anchors={len(anchors):,}")

    nodes = wfs[wfs["plot_eligible"].astype(bool)].copy()
    eligible_clean_ids = set(nodes["clean_fid"].astype(int))
    _edges, adjacency, _shared_by_pair = _build_edges(
        nodes,
        min_shared_edge=float(args.min_shared_edge),
        top_neighbors=int(args.top_neighbors),
        query_chunk_size=int(args.edge_query_chunk_size),
        edge_calc_chunk_size=int(args.edge_calc_chunk_size),
    )
    candidates, score_stats = _score_candidates(
        anchors=anchors,
        pipeline=pipeline,
        feature_cols=feature_cols,
        wfs=wfs,
        source_to_clean=source_to_clean,
        source_by_clean=source_by_clean,
        eligible_clean_ids=eligible_clean_ids,
        adjacency=adjacency,
        geom_by_source=geom_by_source,
        attrs_by_source=attrs_by_source,
        args=args,
        params=params,
    )
    if str(args.candidate_output_csv).strip():
        candidate_output = Path(args.candidate_output_csv)
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_output, index=False)
        _log(f"[INFO] Wrote scored candidate cache: rows={len(candidates):,}; path={candidate_output}")

    selected, review, conflicts = _select_candidates(
        candidates,
        threshold=threshold,
        anchor_owner_by_clean=anchor_owner_by_clean,
    )
    parcels, merged_only = _build_output_parcels(
        wfs=wfs,
        selected=selected,
        geom_by_clean=geom_by_clean,
        attrs_by_clean=attrs_by_clean,
        source_by_clean=source_by_clean,
    )

    _write_layer(parcels, output_gpkg, "predicted_parcels")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only")
    _write_layer(
        _candidate_geometries(selected, geom_by_clean, wfs.crs, limit=int(args.debug_layer_limit)),
        output_gpkg,
        "raw_anchor_group_selected",
    )
    _write_layer(
        _candidate_geometries(review, geom_by_clean, wfs.crs, limit=int(args.debug_layer_limit)),
        output_gpkg,
        "raw_anchor_group_review_candidates",
    )
    _write_layer(
        _candidate_geometries(conflicts, geom_by_clean, wfs.crs, limit=int(args.debug_layer_limit)),
        output_gpkg,
        "raw_anchor_group_conflicts",
    )
    if not candidates.empty:
        candidates.sort_values("raw_fragment_proba", ascending=False).head(int(args.debug_layer_limit)).to_csv(
            output_gpkg.with_suffix(".top_candidates.csv"),
            index=False,
        )
    selected.to_csv(output_gpkg.with_suffix(".selected.csv"), index=False)

    selected_clean_count = sum(len(_parse_id_set(value)) for value in selected.get("candidate_clean_fids", []))
    summary = {
        "model": str(args.model),
        "model_kind": payload.get("model_kind"),
        "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
        "wfs_clean_layer": str(args.wfs_clean_layer),
        "uprn_gpkg": str(args.uprn_gpkg),
        "uprn_layer": str(args.uprn_layer),
        "bbox": str(args.bbox),
        "threshold": float(threshold),
        "clean_wfs_rows": int(len(wfs)),
        "anchor_rows": int(len(anchors)),
        "candidate_rows_scored": int(len(candidates)),
        "selected_groups": int(len(selected)),
        "selected_clean_fids": int(selected_clean_count),
        "conflict_rows": int(len(conflicts)),
        "review_rows": int(len(review)),
        "output_parcels": int(len(parcels)),
        "merged_only_rows": int(len(merged_only)),
        "score_stats": score_stats,
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Raw-fragment parcel candidate apply complete")
    _log(json.dumps(summary, indent=2))
    _log(f"[DONE] output={output_gpkg}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely

from apply_wfs_merge_completion_model import _write_layer
from train_wfs_raw_anchor_group_model import (
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_ID_FIELD,
    DEFAULT_UPRN_LAYER,
    DEFAULT_WFS_CLEAN_GPKG,
    DEFAULT_WFS_CLEAN_LAYER,
    _add_uprn_counts,
    _add_uprn_counts_cached,
    _add_pool_rank_features,
    _build_adjacency_lookup,
    _build_edges,
    _build_edges_cached,
    _build_source_indexes,
    _candidate_features,
    _collect_anchor_pool,
    _enumerate_anchor_groups,
    _enumerate_anchor_groups_ordered,
    _enumerate_anchor_groups_with_shape_supplement,
    _ids_text,
    _parse_bbox,
    _parse_id_set,
    _read_clean_wfs,
    read_candidate_inputs,
)
from train_wfs_raw_anchor_candidate_proposal_model import build_cheap_clean_attrs, cheap_candidate_features
from train_wfs_merge_completion_model import _shape_metrics
from train_wfs_raw_anchor_pairwise_replacement_model import build_pairwise_rows


DEFAULT_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/"
    "wfs_raw_anchor_group_model_v1.joblib"
)
DEFAULT_PROPOSAL_MODEL = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_proposal_model_full_sampled_v1/"
    "wfs_raw_anchor_candidate_proposal_model_v1.joblib"
)
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/"
    "wfs_raw_anchor_group_applied.gpkg"
)
_ANCHOR_SCORE_WORKER_CONTEXT: dict[str, Any] | None = None


def _log(message: str) -> None:
    print(message, flush=True)


def _param(args: argparse.Namespace, params: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return params.get(name, default)


def _union_geoms(clean_ids: set[int] | frozenset[int], geom_by_clean: dict[int, Any]) -> Any:
    geoms = [geom_by_clean[int(fid)] for fid in sorted(clean_ids) if int(fid) in geom_by_clean]
    if not geoms:
        return None
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _init_anchor_score_worker(context: dict[str, Any]) -> None:
    global _ANCHOR_SCORE_WORKER_CONTEXT
    _ANCHOR_SCORE_WORKER_CONTEXT = context


def _score_anchor_batch_worker(task: tuple[int, list[int]]) -> tuple[int, pd.DataFrame, dict[str, int]]:
    if _ANCHOR_SCORE_WORKER_CONTEXT is None:
        raise RuntimeError("Anchor score worker context is not initialized.")
    batch_index, batch_anchors = task
    return _score_anchor_batch(batch_index=batch_index, batch_anchors=batch_anchors, context=_ANCHOR_SCORE_WORKER_CONTEXT)


def _score_anchor_batch(
    *,
    batch_index: int,
    batch_anchors: list[int],
    context: dict[str, Any],
) -> tuple[int, pd.DataFrame, dict[str, int]]:
    source_to_clean: dict[int, list[int]] = context["source_to_clean"]
    source_by_clean: dict[int, int] = context["source_by_clean"]
    eligible_clean_ids: set[int] = context["eligible_clean_ids"]
    adjacency: dict[int, list[tuple[int, float]]] = context["adjacency"]
    shared_by_pair: dict[tuple[int, int], float] = context["shared_by_pair"]
    adjacency_lookup: dict[int, dict[int, float]] | None = context["adjacency_lookup"]
    geom_by_clean: dict[int, Any] = context["geom_by_clean"]
    attrs_by_clean: dict[int, dict[str, Any]] = context["attrs_by_clean"]
    shape_by_clean: dict[int, dict[str, float]] = context["shape_by_clean"]
    area_by_clean: dict[int, float] = context["area_by_clean"]
    cheap_attrs_by_clean: dict[int, tuple[int | None, float, float, int, str]] = context["cheap_attrs_by_clean"]
    uprn_clean_ids: set[int] = context["uprn_clean_ids"]
    building_clean_ids: set[int] = context["building_clean_ids"]
    pipeline: Any = context["pipeline"]
    feature_cols: list[str] = context["feature_cols"]
    proposal_pipeline: Any | None = context["proposal_pipeline"]
    proposal_feature_cols: list[str] = context["proposal_feature_cols"]
    perimeter_by_clean: dict[int, float] = context["perimeter_by_clean"]
    shape_by_group: dict[frozenset[int], dict[str, float]] = context["shape_by_group"]
    params: dict[str, Any] = context["score_params"]

    neighbor_depth = int(params["neighbor_depth"])
    max_pool_size = int(params["max_pool_size"])
    max_group_size = int(params["max_group_size"])
    max_candidate_area = float(params["max_candidate_area"])
    per_anchor_candidate_limit = int(params["per_anchor_candidate_limit"])
    full_score_per_anchor_limit = int(params["full_score_per_anchor_limit"])
    shape_supplement_pool_limit = int(params["shape_supplement_pool_limit"])
    shape_supplement_keep = int(params["shape_supplement_keep"])
    proposal_expanded_candidate_limit = int(params["proposal_expanded_candidate_limit"])
    proposal_keep_per_anchor = int(params["proposal_keep_per_anchor"])
    proposal_include_base_candidates = bool(params["proposal_include_base_candidates"])
    review_threshold = float(params["review_threshold"])

    stats = {
        "candidate_rows_generated": 0,
        "candidate_rows": 0,
        "review_candidate_rows": 0,
        "anchors_without_candidates": 0,
    }
    seen_keys: set[tuple[int, str]] = set()
    records: list[dict[str, Any]] = []
    for anchor_source_fid in batch_anchors:
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
        if shape_supplement_pool_limit > per_anchor_candidate_limit:
            groups = _enumerate_anchor_groups_with_shape_supplement(
                anchor_clean_ids=anchor_clean_ids,
                pool=pool,
                adjacency=adjacency,
                shared_by_pair=shared_by_pair,
                area_by_clean=area_by_clean,
                perimeter_by_clean=perimeter_by_clean,
                max_group_size=max_group_size,
                max_candidate_area=max_candidate_area,
                per_anchor_limit=per_anchor_candidate_limit,
                shape_supplement_pool_limit=shape_supplement_pool_limit,
                shape_supplement_keep=shape_supplement_keep,
                adjacency_lookup=adjacency_lookup,
            )
        elif proposal_pipeline is not None and proposal_keep_per_anchor > 0:
            expanded_groups = _enumerate_anchor_groups_ordered(
                anchor_clean_ids=anchor_clean_ids,
                pool=pool,
                adjacency=adjacency,
                area_by_clean=area_by_clean,
                max_group_size=max_group_size,
                max_candidate_area=max_candidate_area,
                per_anchor_limit=proposal_expanded_candidate_limit,
                adjacency_lookup=adjacency_lookup,
            )
            proposal_records = [
                cheap_candidate_features(
                    anchor_source_fid=int(anchor_source_fid),
                    anchor_clean_ids=anchor_clean_ids,
                    candidate_clean_ids=group,
                    enum_rank=rank,
                    target_train_component_id=-1,
                    source_by_clean=source_by_clean,
                    attrs_by_clean=attrs_by_clean,
                    area_by_clean=area_by_clean,
                    perimeter_by_clean=perimeter_by_clean,
                    adjacency=adjacency,
                    shared_by_pair=shared_by_pair,
                    include_ids=False,
                    cheap_attrs_by_clean=cheap_attrs_by_clean,
                    uprn_clean_ids=uprn_clean_ids,
                    building_clean_ids=building_clean_ids,
                )
                for rank, group in enumerate(expanded_groups, start=1)
            ]
            if proposal_records:
                proposal_frame = pd.DataFrame.from_records(proposal_records)
                for column in proposal_feature_cols:
                    if column not in proposal_frame.columns:
                        proposal_frame[column] = np.nan
                proposal_scores = proposal_pipeline.predict_proba(proposal_frame[proposal_feature_cols])[:, 1]
                fast_scores = pd.to_numeric(
                    proposal_frame["fast_shape_score"],
                    errors="coerce",
                ).fillna(0.0).to_numpy(dtype="float64")
                enum_rank_values = pd.to_numeric(
                    proposal_frame["enum_rank"],
                    errors="coerce",
                ).to_numpy(dtype="float64")
                enum_ranks = np.where(
                    np.isfinite(enum_rank_values),
                    enum_rank_values,
                    np.arange(1, len(expanded_groups) + 1, dtype="float64"),
                ).astype("int64")
                order = np.lexsort((enum_ranks, -fast_scores, -proposal_scores)).tolist()
                proposal_groups = [expanded_groups[idx] for idx in order[:proposal_keep_per_anchor]]
            else:
                proposal_groups = []
            base_groups = []
            needs_base_fallback = (
                proposal_include_base_candidates
                or not proposal_groups
                or full_score_per_anchor_limit <= 0
                or len(proposal_groups) < full_score_per_anchor_limit
            )
            if needs_base_fallback:
                base_groups = list(expanded_groups[:per_anchor_candidate_limit])
            seen_groups: set[frozenset[int]] = set()
            groups = []
            for group in list(proposal_groups) + list(base_groups):
                if group in seen_groups:
                    continue
                seen_groups.add(group)
                groups.append(group)
        else:
            groups = _enumerate_anchor_groups_ordered(
                anchor_clean_ids=anchor_clean_ids,
                pool=pool,
                adjacency=adjacency,
                area_by_clean=area_by_clean,
                max_group_size=max_group_size,
                max_candidate_area=max_candidate_area,
                per_anchor_limit=per_anchor_candidate_limit,
                adjacency_lookup=adjacency_lookup,
            )
        stats["candidate_rows_generated"] += int(len(groups))
        if full_score_per_anchor_limit > 0:
            groups = groups[:full_score_per_anchor_limit]
        if not groups:
            stats["anchors_without_candidates"] += 1
            continue
        for group in groups:
            key = (int(anchor_source_fid), _ids_text(group))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                records.append(
                    _candidate_features(
                        anchor_source_fid=int(anchor_source_fid),
                        anchor_clean_ids=anchor_clean_ids,
                        candidate_clean_ids=group,
                        target_source_ids=set(),
                        target_train_component_id=-1,
                        target_missing_source_count=0,
                        geom_by_clean=geom_by_clean,
                        attrs_by_clean=attrs_by_clean,
                        shape_by_clean=shape_by_clean,
                        source_by_clean=source_by_clean,
                        adjacency=adjacency,
                        shared_by_pair=shared_by_pair,
                        shape_by_group=shape_by_group,
                    )
                )
            except Exception:
                continue

    if not records:
        return batch_index, pd.DataFrame(), stats
    candidates = pd.DataFrame.from_records(records)
    candidates = _add_pool_rank_features(candidates)
    stats["candidate_rows"] = int(len(candidates))
    missing_features = [column for column in feature_cols if column not in candidates.columns]
    for column in missing_features:
        candidates[column] = np.nan
    candidates["raw_anchor_group_proba"] = pipeline.predict_proba(candidates[feature_cols])[:, 1]
    keep = candidates["raw_anchor_group_proba"].ge(review_threshold)
    kept = candidates.loc[keep].copy()
    stats["review_candidate_rows"] = int(len(kept))
    return batch_index, kept, stats


def _score_anchor_candidates(
    *,
    anchors: list[int],
    pipeline: Any,
    feature_cols: list[str],
    wfs: gpd.GeoDataFrame,
    source_to_clean: dict[int, list[int]],
    source_by_clean: dict[int, int],
    eligible_clean_ids: set[int],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    adjacency_lookup: dict[int, dict[int, float]] | None,
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    shape_by_clean: dict[int, dict[str, float]],
    area_by_clean: dict[int, float],
    args: argparse.Namespace,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    review_rows: list[pd.DataFrame] = []
    stats = {
        "anchor_rows": int(len(anchors)),
        "candidate_rows_generated": 0,
        "candidate_rows": 0,
        "review_candidate_rows": 0,
        "anchors_without_candidates": 0,
    }
    neighbor_depth = int(_param(args, params, "neighbor_depth", 3))
    max_pool_size = int(_param(args, params, "max_pool_size", 16))
    max_group_size = int(_param(args, params, "max_group_size", 7))
    max_candidate_area = float(_param(args, params, "max_candidate_area", 2500.0))
    per_anchor_candidate_limit = int(_param(args, params, "per_anchor_candidate_limit", 40))
    full_score_per_anchor_limit = int(_param(args, params, "full_score_per_anchor_limit", 12))
    shape_supplement_pool_limit = int(_param(args, params, "shape_supplement_pool_limit", 0))
    shape_supplement_keep = int(_param(args, params, "shape_supplement_keep", 0))
    proposal_payload = None
    proposal_pipeline = None
    proposal_feature_cols: list[str] = []
    if str(args.proposal_model).strip():
        proposal_payload = joblib.load(str(args.proposal_model))
        if not isinstance(proposal_payload, dict) or proposal_payload.get("model_kind") != "wfs_raw_anchor_candidate_proposal_ranker":
            raise RuntimeError("--proposal-model must be a wfs_raw_anchor_candidate_proposal_ranker payload.")
        proposal_pipeline = proposal_payload["pipeline"]
        proposal_feature_cols = list(proposal_payload["feature_cols"])
        _log(
            "[INFO] Candidate proposal model enabled: "
            f"model={args.proposal_model}; expanded_limit={int(args.proposal_expanded_candidate_limit)}; "
            f"keep={int(args.proposal_keep_per_anchor)}"
        )
    perimeter_by_clean = {
        int(clean_fid): float(attrs.get("perimeter", 0.0) or 0.0)
        for clean_fid, attrs in attrs_by_clean.items()
    }
    cheap_attrs_by_clean = build_cheap_clean_attrs(
        source_by_clean=source_by_clean,
        attrs_by_clean=attrs_by_clean,
        area_by_clean=area_by_clean,
        perimeter_by_clean=perimeter_by_clean,
    )
    uprn_clean_ids = {
        int(clean_fid)
        for clean_fid, attrs in attrs_by_clean.items()
        if int(attrs.get("uprn_count", 0) or 0) > 0
    }
    building_clean_ids = {
        int(clean_fid)
        for clean_fid, attrs in attrs_by_clean.items()
        if str(attrs.get("anchor_role", "") or "") == "building"
    }
    score_params = {
        "neighbor_depth": neighbor_depth,
        "max_pool_size": max_pool_size,
        "max_group_size": max_group_size,
        "max_candidate_area": max_candidate_area,
        "per_anchor_candidate_limit": per_anchor_candidate_limit,
        "full_score_per_anchor_limit": full_score_per_anchor_limit,
        "shape_supplement_pool_limit": shape_supplement_pool_limit,
        "shape_supplement_keep": shape_supplement_keep,
        "proposal_expanded_candidate_limit": int(args.proposal_expanded_candidate_limit),
        "proposal_keep_per_anchor": int(args.proposal_keep_per_anchor),
        "proposal_include_base_candidates": bool(getattr(args, "proposal_include_base_candidates", False)),
        "review_threshold": float(args.review_threshold),
    }
    worker_context: dict[str, Any] = {
        "source_to_clean": source_to_clean,
        "source_by_clean": source_by_clean,
        "eligible_clean_ids": eligible_clean_ids,
        "adjacency": adjacency,
        "shared_by_pair": shared_by_pair,
        "adjacency_lookup": adjacency_lookup,
        "geom_by_clean": geom_by_clean,
        "attrs_by_clean": attrs_by_clean,
        "shape_by_clean": shape_by_clean,
        "area_by_clean": area_by_clean,
        "perimeter_by_clean": perimeter_by_clean,
        "cheap_attrs_by_clean": cheap_attrs_by_clean,
        "uprn_clean_ids": uprn_clean_ids,
        "building_clean_ids": building_clean_ids,
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "proposal_pipeline": proposal_pipeline,
        "proposal_feature_cols": proposal_feature_cols,
        "shape_by_group": {},
        "score_params": score_params,
    }
    worker_count = max(int(getattr(args, "anchor_workers", 1) or 1), 1)
    if worker_count > 1 and "fork" not in mp.get_all_start_methods():
        _log("[WARN] Multiprocessing fork start method is unavailable; falling back to --anchor-workers 1")
        worker_count = 1
    configured_batch_size = max(int(args.anchor_batch_size), 1)
    if worker_count > 1:
        configured_batch_size = min(
            configured_batch_size,
            max(1, int(math.ceil(len(anchors) / max(worker_count * 2, 1)))),
        )
    tasks = [
        (batch_index, anchors[start : min(start + configured_batch_size, len(anchors))])
        for batch_index, start in enumerate(range(0, len(anchors), configured_batch_size), start=1)
    ]
    if worker_count > 1 and len(tasks) > 1:
        _log(
            "[INFO] Anchor scoring workers enabled: "
            f"workers={worker_count}; batches={len(tasks):,}; batch_size={configured_batch_size:,}"
        )
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=worker_count,
            initializer=_init_anchor_score_worker,
            initargs=(worker_context,),
        ) as pool:
            results = list(pool.imap(_score_anchor_batch_worker, tasks))
    else:
        results = [
            _score_anchor_batch(batch_index=batch_index, batch_anchors=batch_anchors, context=worker_context)
            for batch_index, batch_anchors in tasks
        ]

    for batch_index, kept, batch_stats in sorted(results, key=lambda item: item[0]):
        stats["candidate_rows_generated"] += int(batch_stats.get("candidate_rows_generated", 0))
        stats["candidate_rows"] += int(batch_stats.get("candidate_rows", 0))
        stats["review_candidate_rows"] += int(batch_stats.get("review_candidate_rows", 0))
        stats["anchors_without_candidates"] += int(batch_stats.get("anchors_without_candidates", 0))
        if not kept.empty:
            review_rows.append(kept)
        _log(
            "[INFO] Scored anchor batch "
            f"{batch_index}: anchors={len(tasks[batch_index - 1][1]):,}; "
            f"candidates={int(batch_stats.get('candidate_rows', 0)):,}; "
            f"review_kept={int(batch_stats.get('review_candidate_rows', 0)):,}"
        )

    if not review_rows:
        return pd.DataFrame(), stats
    return pd.concat(review_rows, ignore_index=True), stats


def _select_candidates(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    anchor_owner_by_clean: dict[int, int],
    reject_candidate_holes: bool = False,
    allow_holed_anchor_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), candidates.copy(), candidates.copy()
    high = candidates[candidates["raw_anchor_group_proba"].ge(float(threshold))].copy()
    review = candidates[candidates["raw_anchor_group_proba"].lt(float(threshold))].copy()
    if high.empty:
        return high, review.sort_values("raw_anchor_group_proba", ascending=False), high.copy()
    high = high.sort_values(
        [
            "raw_anchor_group_proba",
            "candidate_source_count",
            "internal_shared_len",
            "group_regularity_score",
            "group_hull_gap_ratio",
        ],
        ascending=[False, False, False, False, True],
    )
    selected_rows: list[pd.Series] = []
    conflict_rows: list[pd.Series] = []
    claimed_clean: set[int] = set()
    selected_anchors: set[int] = set()
    for _, row in high.iterrows():
        anchor_source_fid = int(row.anchor_source_fid)
        clean_ids = _parse_id_set(row.candidate_clean_fids)
        anchor_clean_ids = _parse_id_set(row.anchor_clean_fids)
        has_candidate_hole = False
        if bool(reject_candidate_holes):
            fallback_value = row.get("is_anchor_fallback_candidate", 0)
            is_anchor_fallback = False if pd.isna(fallback_value) else bool(int(fallback_value or 0) > 0)
            hole_count = pd.to_numeric(
                pd.Series([row.get("group_hole_count", 0)]),
                errors="coerce",
            ).fillna(0.0).iloc[0]
            hole_area_ratio = pd.to_numeric(
                pd.Series([row.get("group_hole_area_ratio", 0)]),
                errors="coerce",
            ).fillna(0.0).iloc[0]
            has_candidate_hole = bool(
                (float(hole_count) > 0.0 or float(hole_area_ratio) > 0.0)
                and not (bool(allow_holed_anchor_fallback) and is_anchor_fallback)
            )
        other_anchor_clean = {
            clean_id
            for clean_id in clean_ids
            if clean_id in anchor_owner_by_clean and int(anchor_owner_by_clean[clean_id]) != anchor_source_fid
        }
        conflict_reason = ""
        if has_candidate_hole:
            conflict_reason = "candidate_has_hole"
        elif anchor_source_fid in selected_anchors:
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
        claimed_clean.update(anchor_clean_ids)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else high.iloc[0:0].copy()
    conflicts = pd.DataFrame(conflict_rows).reset_index(drop=True) if conflict_rows else high.iloc[0:0].copy()
    return selected, review.sort_values("raw_anchor_group_proba", ascending=False), conflicts


def _row_contains_other_anchor(row: pd.Series, anchor_owner_by_clean: dict[int, int]) -> bool:
    anchor_source_fid = int(row.anchor_source_fid)
    clean_ids = _parse_id_set(row.candidate_clean_fids)
    return any(
        clean_id in anchor_owner_by_clean and int(anchor_owner_by_clean[clean_id]) != anchor_source_fid
        for clean_id in clean_ids
    )


def _has_holes(geom: Any) -> bool:
    if geom is None or geom.is_empty:
        return False
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    return any(len(getattr(part, "interiors", [])) > 0 for part in parts)


def _hole_polygons(geom: Any) -> list[Any]:
    if geom is None or geom.is_empty:
        return []
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    holes = []
    for part in parts:
        if getattr(part, "geom_type", "") != "Polygon":
            continue
        holes.extend(shapely.Polygon(ring) for ring in getattr(part, "interiors", []))
    return [hole for hole in holes if hole is not None and not hole.is_empty and float(hole.area) > 1e-9]


def _fill_polygon_holes(geom: Any) -> Any:
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return shapely.Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([shapely.Polygon(part.exterior) for part in geom.geoms if not part.is_empty])
    if geom.geom_type == "GeometryCollection":
        polygons = [
            shapely.Polygon(part.exterior)
            for part in geom.geoms
            if part.geom_type == "Polygon" and not part.is_empty
        ]
        if polygons:
            return shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    return geom


def _attach_leftover_zero_uprn(
    selected: pd.DataFrame,
    *,
    wfs: gpd.GeoDataFrame,
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    source_by_clean: dict[int, int],
    max_passes: int,
    min_shared_edge: float,
    max_added_area: float,
    max_group_area: float,
    max_hull_gap_ratio: float,
    regularity_drop: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    if selected.empty or int(max_passes) <= 0:
        return selected, selected.iloc[0:0].copy(), {
            "leftover_attach_enabled": int(int(max_passes) > 0),
            "leftover_attach_rows": 0,
            "leftover_attach_candidates": 0,
            "leftover_attach_passes": 0,
        }

    work = selected.copy().reset_index(drop=True)
    group_clean_ids: list[set[int]] = [
        set(_parse_id_set(value))
        for value in work["candidate_clean_fids"].fillna("").astype(str)
    ]
    group_geoms = [_union_geoms(ids, geom_by_clean) for ids in group_clean_ids]
    group_shapes = [
        _shape_metrics(geom) if geom is not None and not geom.is_empty else {}
        for geom in group_geoms
    ]
    claimed_clean = {int(clean_id) for ids in group_clean_ids for clean_id in ids}

    def is_leftover_candidate(row: Any) -> bool:
        clean_id = int(row.clean_fid)
        if clean_id in claimed_clean:
            return False
        attrs = attrs_by_clean.get(clean_id, {})
        if int(attrs.get("uprn_count", 0) or 0) > 0:
            return False
        if not bool(attrs.get("plot_eligible", False)):
            return False
        role = str(attrs.get("anchor_role", "") or "")
        if role not in {"building", "land"}:
            return False
        area = float(attrs.get("area", 0.0) or 0.0)
        if float(max_added_area) > 0.0 and area > float(max_added_area):
            return False
        return True

    leftover_ids = [
        int(row.clean_fid)
        for row in wfs.itertuples(index=False)
        if is_leftover_candidate(row)
    ]
    leftover_id_set = set(leftover_ids)
    attach_records: list[dict[str, Any]] = []
    attempted = 0
    passes_run = 0
    for pass_index in range(1, int(max_passes) + 1):
        candidates = [clean_id for clean_id in leftover_ids if clean_id not in claimed_clean]
        if not candidates:
            break
        passes_run = pass_index
        tree_geoms = np.asarray(group_geoms, dtype=object)
        tree = shapely.STRtree(tree_geoms)
        leftover_geoms = np.asarray([geom_by_clean[fid] for fid in candidates], dtype=object)
        leftover_tree = shapely.STRtree(leftover_geoms) if len(leftover_geoms) else None
        leftover_tree_ids = [int(fid) for fid in candidates]

        def contained_hole_fillers(candidate_geom: Any, primary_clean_id: int) -> set[int] | None:
            holes = _hole_polygons(candidate_geom)
            if not holes:
                return set()
            if leftover_tree is None:
                return None
            fillers: set[int] = set()
            for hole in holes:
                hit_ids: list[int] = []
                hit_geoms: list[Any] = []
                for hit_index in leftover_tree.query(hole, predicate="intersects"):
                    filler_id = int(leftover_tree_ids[int(hit_index)])
                    if filler_id == int(primary_clean_id) or filler_id in claimed_clean:
                        continue
                    if filler_id not in leftover_id_set:
                        continue
                    filler_geom = geom_by_clean.get(filler_id)
                    if filler_geom is None or filler_geom.is_empty:
                        continue
                    overlap_area = float(shapely.area(shapely.intersection(filler_geom, hole)))
                    if overlap_area <= 1e-8:
                        continue
                    hit_ids.append(filler_id)
                    hit_geoms.append(filler_geom)
                if not hit_geoms:
                    return None
                covered = shapely.union_all(np.asarray([shapely.intersection(geom, hole) for geom in hit_geoms], dtype=object))
                coverage = float(shapely.area(covered)) / max(float(hole.area), 1e-9)
                if coverage < 0.98:
                    return None
                fillers.update(hit_ids)
            return fillers

        changed = False
        for clean_id in sorted(candidates, key=lambda fid: float(attrs_by_clean.get(fid, {}).get("area", 0.0) or 0.0)):
            if clean_id in claimed_clean:
                continue
            geom = geom_by_clean.get(int(clean_id))
            if geom is None or geom.is_empty:
                continue
            candidate_group_indexes = tree.query(geom, predicate="intersects")
            best: tuple[tuple[float, float, float], int, Any, dict[str, float], set[int]] | None = None
            for group_index in candidate_group_indexes:
                group_index = int(group_index)
                group_geom = group_geoms[group_index]
                if group_geom is None or group_geom.is_empty:
                    continue
                shared = float(shapely.length(shapely.intersection(shapely.boundary(group_geom), shapely.boundary(geom))))
                if shared < float(min_shared_edge):
                    continue
                attempted += 1
                new_geom = shapely.union_all(np.asarray([group_geom, geom], dtype=object))
                if new_geom is None or new_geom.is_empty:
                    continue
                if not bool(shapely.is_valid(new_geom)):
                    new_geom = shapely.make_valid(new_geom)
                hole_filler_ids: set[int] = set()
                if _has_holes(new_geom):
                    maybe_fillers = contained_hole_fillers(new_geom, int(clean_id))
                    if maybe_fillers is None:
                        continue
                    hole_filler_ids = set(maybe_fillers)
                    if hole_filler_ids:
                        new_geom = shapely.union_all(
                            np.asarray(
                                [group_geom, geom] + [geom_by_clean[fid] for fid in sorted(hole_filler_ids)],
                                dtype=object,
                            )
                        )
                        if new_geom is None or new_geom.is_empty:
                            continue
                        if not bool(shapely.is_valid(new_geom)):
                            new_geom = shapely.make_valid(new_geom)
                    if _has_holes(new_geom):
                        continue
                shape = _shape_metrics(new_geom)
                if float(max_group_area) > 0.0 and float(shape.get("area", 0.0)) > float(max_group_area):
                    continue
                if float(shape.get("hull_gap_ratio", 0.0)) > float(max_hull_gap_ratio):
                    continue
                current_shape = group_shapes[group_index] or {}
                current_regularity = float(current_shape.get("regularity_score", 0.0) or 0.0)
                if float(shape.get("regularity_score", 0.0)) < current_regularity - float(regularity_drop):
                    continue
                score = (
                    shared,
                    float(shape.get("regularity_score", 0.0)),
                    -float(shape.get("hull_gap_ratio", 0.0)),
                )
                if best is None or score > best[0]:
                    best = (score, group_index, new_geom, shape, hole_filler_ids)
            if best is None:
                continue
            score, group_index, new_geom, shape, hole_filler_ids = best
            shared = float(score[0])
            regularity = float(score[1])
            group_clean_ids[group_index].add(int(clean_id))
            group_clean_ids[group_index].update(int(fid) for fid in hole_filler_ids)
            group_geoms[group_index] = new_geom
            group_shapes[group_index] = shape
            claimed_clean.add(int(clean_id))
            claimed_clean.update(int(fid) for fid in hole_filler_ids)
            changed = True
            attach_records.append(
                {
                    "attached_clean_fid": int(clean_id),
                    "attached_source_fid": int(source_by_clean.get(int(clean_id), clean_id)),
                    "hole_filler_clean_fids": _ids_text(hole_filler_ids),
                    "hole_filler_source_fids": _ids_text(
                        int(source_by_clean.get(int(fid), fid)) for fid in sorted(hole_filler_ids)
                    ),
                    "target_anchor_source_fid": int(work.at[group_index, "anchor_source_fid"]),
                    "target_row_index": int(group_index),
                    "shared_edge_m": float(shared),
                    "attached_area": float(attrs_by_clean.get(int(clean_id), {}).get("area", 0.0) or 0.0),
                    "new_group_area": float(shape.get("area", 0.0)),
                    "new_group_regularity_score": float(regularity),
                    "new_group_hull_gap_ratio": float(shape.get("hull_gap_ratio", 0.0)),
                    "geometry": geom,
                }
            )
        if not changed:
            break

    for row_index, clean_ids in enumerate(group_clean_ids):
        source_ids = {int(source_by_clean[fid]) for fid in clean_ids if fid in source_by_clean}
        work.at[row_index, "candidate_clean_fids"] = _ids_text(clean_ids)
        work.at[row_index, "candidate_source_fids"] = _ids_text(source_ids)
        work.at[row_index, "candidate_clean_count"] = int(len(clean_ids))
        work.at[row_index, "candidate_source_count"] = int(len(source_ids))

    attached = (
        gpd.GeoDataFrame(attach_records, geometry="geometry", crs=wfs.crs)
        if attach_records
        else gpd.GeoDataFrame(attach_records, geometry=[], crs=wfs.crs)
    )
    stats = {
        "leftover_attach_enabled": 1,
        "leftover_attach_rows": int(len(attached)),
        "leftover_attach_candidates": int(len(leftover_ids)),
        "leftover_attach_attempts": int(attempted),
        "leftover_attach_passes": int(passes_run),
        "leftover_attach_max_added_area": float(max_added_area),
        "leftover_attach_max_group_area": float(max_group_area),
        "leftover_attach_max_hull_gap_ratio": float(max_hull_gap_ratio),
        "leftover_attach_regularity_drop": float(regularity_drop),
    }
    return work, attached, stats


def _build_anchor_fallback_candidates(
    *,
    anchors: list[int],
    source_to_clean: dict[int, list[int]],
    source_by_clean: dict[int, int],
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    existing_candidates: pd.DataFrame,
    proba: float,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    existing_keys: set[tuple[int, str]] = set()
    if not existing_candidates.empty and {"anchor_source_fid", "candidate_clean_fids"}.issubset(existing_candidates.columns):
        existing_keys = {
            (int(row.anchor_source_fid), str(row.candidate_clean_fids))
            for row in existing_candidates[["anchor_source_fid", "candidate_clean_fids"]].itertuples(index=False)
        }

    candidate_anchor_set = (
        set(existing_candidates["anchor_source_fid"].astype(int))
        if not existing_candidates.empty and "anchor_source_fid" in existing_candidates.columns
        else set()
    )
    rows: list[dict[str, Any]] = []
    anchors_without_input_candidates = 0
    for anchor_source_fid in anchors:
        anchor_source_fid = int(anchor_source_fid)
        anchor_clean_ids = frozenset(int(value) for value in source_to_clean.get(anchor_source_fid, []))
        if not anchor_clean_ids:
            continue
        key = (anchor_source_fid, _ids_text(anchor_clean_ids))
        if key in existing_keys:
            continue
        if anchor_source_fid not in candidate_anchor_set:
            anchors_without_input_candidates += 1
        geom = _union_geoms(anchor_clean_ids, geom_by_clean)
        shape = _shape_metrics(geom) if geom is not None and not geom.is_empty else {}
        source_ids = {int(source_by_clean[fid]) for fid in anchor_clean_ids if fid in source_by_clean}
        uprn_count = sum(int(attrs_by_clean[fid].get("uprn_count", 0) or 0) for fid in anchor_clean_ids if fid in attrs_by_clean)
        roles = [str(attrs_by_clean[fid].get("anchor_role", "other") or "other") for fid in anchor_clean_ids if fid in attrs_by_clean]
        role_signature = "|".join(f"{role}:{roles.count(role)}" for role in sorted(set(roles))) if roles else ""
        rows.append(
            {
                "anchor_source_fid": anchor_source_fid,
                "anchor_clean_fids": _ids_text(anchor_clean_ids),
                "candidate_clean_fids": _ids_text(anchor_clean_ids),
                "candidate_source_fids": _ids_text(source_ids),
                "target_source_fids": "",
                "target_train_component_id": -1,
                "target_missing_source_count": 0,
                "candidate_clean_count": int(len(anchor_clean_ids)),
                "candidate_source_count": int(len(source_ids)),
                "anchor_clean_count": int(len(anchor_clean_ids)),
                "added_clean_count": 0,
                "added_source_count": 0,
                "candidate_area": float(shape.get("area", 0.0)),
                "anchor_area": float(shape.get("area", 0.0)),
                "added_area_sum": 0.0,
                "internal_shared_len": 0.0,
                "uprn_count": int(uprn_count),
                "anchor_uprn_count": int(uprn_count),
                "added_uprn_count": 0,
                "added_uprn_polygon_count": 0,
                "role_signature": role_signature,
                "anchor_role": roles[0] if roles else "",
                "group_area": float(shape.get("area", 0.0)),
                "group_hull_gap_ratio": float(shape.get("hull_gap_ratio", 0.0)),
                "group_hole_count": float(shape.get("hole_count", 0.0)),
                "group_hole_area_ratio": float(shape.get("hole_area_ratio", 0.0)),
                "group_regularity_score": float(shape.get("regularity_score", 0.0)),
                "raw_anchor_group_proba": float(proba),
                "is_anchor_fallback_candidate": 1,
            }
        )
    fallback = pd.DataFrame.from_records(rows)
    stats = {
        "anchor_fallback_enabled": 1,
        "anchor_fallback_rows": int(len(fallback)),
        "anchor_fallback_anchors_without_input_candidates": int(anchors_without_input_candidates),
        "anchor_fallback_proba": float(proba),
    }
    return fallback, stats


def _apply_residual_completion_selector(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    anchor_owner_by_clean: dict[int, int],
    reject_candidate_holes: bool,
    allow_holed_anchor_fallback: bool,
    proba_delta: float,
    regularity_drop: float,
    hull_gap_add: float,
    source_count_add: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if candidates.empty or selected.empty or float(proba_delta) <= 0.0:
        return selected, selected.iloc[0:0].copy(), {
            "residual_enabled": 0,
            "residual_replacement_attempts": 0,
            "residual_selected_groups_before_conflict": int(len(selected)),
            "residual_selected_groups": int(len(selected)),
            "residual_conflict_rows": 0,
        }

    by_anchor = {
        int(anchor): group.copy()
        for anchor, group in candidates.groupby(candidates["anchor_source_fid"].astype(int), sort=False)
    }
    preferred_rows: list[pd.Series] = []
    replacement_attempts = 0
    for _, row in selected.iterrows():
        anchor = int(row.anchor_source_fid)
        pool = by_anchor.get(anchor)
        if pool is None or pool.empty:
            preferred_rows.append(row)
            continue
        target_source_count = int(row.candidate_source_count) + int(source_count_add)
        candidates_for_anchor = pool[
            pool["raw_anchor_group_proba"].astype(float).ge(float(row.raw_anchor_group_proba) - float(proba_delta))
            & pool["candidate_source_count"].astype(int).eq(target_source_count)
            & pool["group_regularity_score"].astype(float).ge(float(row.group_regularity_score) - float(regularity_drop))
            & pool["group_hull_gap_ratio"].astype(float).le(float(row.group_hull_gap_ratio) + float(hull_gap_add))
        ].copy()
        if not candidates_for_anchor.empty:
            candidates_for_anchor = candidates_for_anchor[
                ~candidates_for_anchor.apply(lambda candidate: _row_contains_other_anchor(candidate, anchor_owner_by_clean), axis=1)
            ].copy()
        if bool(reject_candidate_holes) and not candidates_for_anchor.empty:
            hole_count = pd.to_numeric(
                candidates_for_anchor.get("group_hole_count", 0),
                errors="coerce",
            ).fillna(0.0)
            hole_area_ratio = pd.to_numeric(
                candidates_for_anchor.get("group_hole_area_ratio", 0),
                errors="coerce",
            ).fillna(0.0)
            candidates_for_anchor = candidates_for_anchor[
                hole_count.le(0.0) & hole_area_ratio.le(0.0)
            ].copy()
        if candidates_for_anchor.empty:
            preferred_rows.append(row)
            continue
        replacement_attempts += 1
        candidates_for_anchor = candidates_for_anchor.sort_values(
            ["raw_anchor_group_proba", "group_regularity_score", "candidate_area"],
            ascending=[False, False, False],
        )
        preferred_rows.append(candidates_for_anchor.iloc[0])

    preferred = pd.DataFrame(preferred_rows).reset_index(drop=True)
    residual_selected, _residual_review, residual_conflicts = _select_candidates(
        preferred,
        threshold=0.0,
        anchor_owner_by_clean=anchor_owner_by_clean,
        reject_candidate_holes=bool(reject_candidate_holes),
        allow_holed_anchor_fallback=bool(allow_holed_anchor_fallback),
    )
    stats = {
        "residual_enabled": 1,
        "residual_replacement_attempts": int(replacement_attempts),
        "residual_selected_groups_before_conflict": int(len(preferred)),
        "residual_selected_groups": int(len(residual_selected)),
        "residual_conflict_rows": int(len(residual_conflicts)),
    }
    return residual_selected, residual_conflicts, stats


def _apply_pairwise_replacement_selector(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    pairwise_model_path: str,
    pairwise_threshold: float,
    pairwise_top_k: int,
    anchor_owner_by_clean: dict[int, int],
    reject_candidate_holes: bool,
    allow_holed_anchor_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if candidates.empty or selected.empty or not str(pairwise_model_path).strip():
        return selected, selected.iloc[0:0].copy(), {
            "pairwise_enabled": 0,
            "pairwise_rows": 0,
            "pairwise_above_threshold": 0,
            "pairwise_replacement_attempts": 0,
            "pairwise_selected_groups_before_conflict": int(len(selected)),
            "pairwise_selected_groups": int(len(selected)),
            "pairwise_conflict_rows": 0,
        }

    payload = joblib.load(pairwise_model_path)
    if not isinstance(payload, dict) or payload.get("model_kind") != "wfs_raw_anchor_pairwise_replacement":
        raise RuntimeError("Pairwise payload must be a wfs_raw_anchor_pairwise_replacement model.")
    feature_cols = list(payload["feature_cols"])
    work = candidates.copy()
    if "label" not in work.columns:
        work["label"] = 0
    if "label_source" not in work.columns:
        work["label_source"] = ""
    pairwise_rows = build_pairwise_rows(
        work,
        top_k=int(pairwise_top_k),
        include_all_positives=False,
        add_set_relations=True,
    )
    if pairwise_rows.empty:
        return selected, selected.iloc[0:0].copy(), {
            "pairwise_enabled": 1,
            "pairwise_rows": 0,
            "pairwise_above_threshold": 0,
            "pairwise_replacement_attempts": 0,
            "pairwise_selected_groups_before_conflict": int(len(selected)),
            "pairwise_selected_groups": int(len(selected)),
            "pairwise_conflict_rows": 0,
        }
    for column in feature_cols:
        if column not in pairwise_rows.columns:
            pairwise_rows[column] = np.nan
    pairwise_rows["raw_anchor_pair_replace_proba"] = payload["pipeline"].predict_proba(pairwise_rows[feature_cols])[:, 1]
    replace_rows = pairwise_rows[pairwise_rows["raw_anchor_pair_replace_proba"].ge(float(pairwise_threshold))].copy()
    replace_rows = replace_rows.sort_values(
        ["raw_anchor_pair_replace_proba", "raw_anchor_group_proba", "candidate_source_count"],
        ascending=[False, False, False],
    )
    candidate_key_cols = ["anchor_source_fid", "candidate_clean_fids"]
    replacement_by_anchor = {
        int(row.anchor_source_fid): row
        for row in replace_rows.drop_duplicates(["anchor_source_fid"], keep="first").itertuples(index=False)
    }
    candidate_lookup = candidates.drop_duplicates(candidate_key_cols, keep="first").set_index(candidate_key_cols, drop=False)

    preferred_rows: list[pd.Series] = []
    attempts = 0
    for _, row in selected.iterrows():
        replacement = replacement_by_anchor.get(int(row.anchor_source_fid))
        if replacement is None:
            preferred_rows.append(row)
            continue
        if str(getattr(replacement, "top_candidate_clean_fids", "")) != str(row.candidate_clean_fids):
            preferred_rows.append(row)
            continue
        key = (int(row.anchor_source_fid), str(getattr(replacement, "candidate_clean_fids", "")))
        if key not in candidate_lookup.index:
            preferred_rows.append(row)
            continue
        candidate = candidate_lookup.loc[key].copy()
        if isinstance(candidate, pd.DataFrame):
            candidate = candidate.iloc[0].copy()
        if _row_contains_other_anchor(candidate, anchor_owner_by_clean):
            preferred_rows.append(row)
            continue
        candidate["raw_anchor_pair_replace_proba"] = float(getattr(replacement, "raw_anchor_pair_replace_proba"))
        attempts += 1
        preferred_rows.append(candidate)

    preferred = pd.DataFrame(preferred_rows).reset_index(drop=True)
    pairwise_selected, _pairwise_review, pairwise_conflicts = _select_candidates(
        preferred,
        threshold=0.0,
        anchor_owner_by_clean=anchor_owner_by_clean,
        reject_candidate_holes=bool(reject_candidate_holes),
        allow_holed_anchor_fallback=bool(allow_holed_anchor_fallback),
    )
    stats = {
        "pairwise_enabled": 1,
        "pairwise_rows": int(len(pairwise_rows)),
        "pairwise_above_threshold": int(len(replace_rows)),
        "pairwise_replacement_attempts": int(attempts),
        "pairwise_selected_groups_before_conflict": int(len(preferred)),
        "pairwise_selected_groups": int(len(pairwise_selected)),
        "pairwise_conflict_rows": int(len(pairwise_conflicts)),
    }
    return pairwise_selected, pairwise_conflicts, stats


def _candidate_geometries(
    rows: pd.DataFrame,
    geom_by_clean: dict[int, Any],
    crs: Any,
    *,
    limit: int,
) -> gpd.GeoDataFrame:
    if rows.empty:
        return gpd.GeoDataFrame(rows.copy(), geometry=[], crs=crs)
    work = rows.sort_values("raw_anchor_group_proba", ascending=False).head(int(limit)).copy()
    geoms = [_union_geoms(_parse_id_set(value), geom_by_clean) for value in work["candidate_clean_fids"]]
    return gpd.GeoDataFrame(work, geometry=geoms, crs=crs)


def _build_output_parcels(
    *,
    wfs: gpd.GeoDataFrame,
    selected: pd.DataFrame,
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    source_by_clean: dict[int, int],
    fill_output_holes: bool,
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
        if bool(fill_output_holes):
            geom = _fill_polygon_holes(geom)
        merged_records.append(
            {
                "raw_anchor_group_plot_id": int(plot_id),
                "merge_status": "raw_anchor_group_selected",
                "anchor_source_fid": int(getattr(row, "anchor_source_fid")),
                "clean_fids": _ids_text(clean_ids),
                "source_fids": _ids_text(source_ids),
                "clean_count": int(len(clean_ids)),
                "source_count": int(len(source_ids)),
                "uprn_count": int(uprn_count),
                "raw_anchor_group_proba": float(getattr(row, "raw_anchor_group_proba")),
                "geometry": geom,
            }
        )

    base = wfs[~wfs["clean_fid"].astype(int).isin(claimed_clean)].copy()
    base_records = []
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
                "raw_anchor_group_proba": np.nan,
                "geometry": _fill_polygon_holes(getattr(row, "geometry")) if bool(fill_output_holes) else getattr(row, "geometry"),
            }
        )
    merged = gpd.GeoDataFrame(merged_records, geometry="geometry", crs=wfs.crs)
    base_gdf = gpd.GeoDataFrame(base_records, geometry="geometry", crs=wfs.crs)
    parcels = pd.concat([merged, base_gdf], ignore_index=True)
    parcels = gpd.GeoDataFrame(parcels, geometry="geometry", crs=wfs.crs)
    return parcels, merged


def _build_found_anchors_layer(wfs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if wfs.empty:
        return gpd.GeoDataFrame(geometry=[], crs=wfs.crs)
    work = wfs[
        wfs["uprn_count"].astype(int).gt(0)
        & wfs["anchor_role"].isin(["building", "land"])
    ].copy()
    if work.empty:
        return gpd.GeoDataFrame(geometry=[], crs=wfs.crs)
    work["runtime_clean_fid"] = work["clean_fid"].astype("int64")
    keep_cols = [
        "runtime_clean_fid",
        "source_fid",
        "uprn_count",
        "anchor_role",
        "Theme",
        "DescriptiveGroup",
        "DescriptiveTerm",
        "area",
        "perimeter",
        "geometry",
    ]
    keep_cols = [column for column in keep_cols if column in work.columns]
    return gpd.GeoDataFrame(work[keep_cols].copy(), geometry="geometry", crs=wfs.crs)


def _build_clean_coverage_audit(
    *,
    wfs: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    clean_to_output_rows: dict[int, list[int]] = {}
    clean_to_output_status: dict[int, list[str]] = {}
    clean_to_output_sources: dict[int, list[str]] = {}
    unknown_clean_ids: set[int] = set()
    valid_clean_ids = set(int(value) for value in wfs["clean_fid"].astype(int))
    for output_row_id, row in enumerate(parcels.itertuples(index=False), start=1):
        clean_ids = _parse_id_set(getattr(row, "clean_fids", ""))
        for clean_id in clean_ids:
            clean_id = int(clean_id)
            if clean_id not in valid_clean_ids:
                unknown_clean_ids.add(clean_id)
                continue
            clean_to_output_rows.setdefault(clean_id, []).append(int(output_row_id))
            clean_to_output_status.setdefault(clean_id, []).append(str(getattr(row, "merge_status", "")))
            clean_to_output_sources.setdefault(clean_id, []).append(str(getattr(row, "source_fids", "")))

    covered_clean_ids = set(clean_to_output_rows)
    missing_clean_ids = valid_clean_ids - covered_clean_ids
    duplicate_clean_ids = {
        clean_id
        for clean_id, output_rows in clean_to_output_rows.items()
        if len(output_rows) > 1
    }
    issue_clean_ids = sorted(missing_clean_ids | duplicate_clean_ids)
    records: list[dict[str, Any]] = []
    if issue_clean_ids:
        attrs = wfs.set_index(wfs["clean_fid"].astype(int))
        for clean_id in issue_clean_ids:
            row = attrs.loc[int(clean_id)]
            is_missing = int(clean_id in missing_clean_ids)
            is_duplicate = int(clean_id in duplicate_clean_ids)
            records.append(
                {
                    "coverage_issue": (
                        "missing"
                        if is_missing and not is_duplicate
                        else "duplicate"
                        if is_duplicate and not is_missing
                        else "missing_and_duplicate"
                    ),
                    "clean_fid": int(clean_id),
                    "source_fid": int(row.source_fid),
                    "uprn_count": int(getattr(row, "uprn_count", 0) or 0),
                    "anchor_role": str(getattr(row, "anchor_role", "") or ""),
                    "Theme": str(getattr(row, "Theme", "") or ""),
                    "DescriptiveGroup": str(getattr(row, "DescriptiveGroup", "") or ""),
                    "DescriptiveTerm": str(getattr(row, "DescriptiveTerm", "") or ""),
                    "output_row_ids": _ids_text(clean_to_output_rows.get(int(clean_id), [])),
                    "output_merge_statuses": "|".join(clean_to_output_status.get(int(clean_id), [])),
                    "output_source_fids": "|".join(clean_to_output_sources.get(int(clean_id), [])),
                    "geometry": row.geometry,
                }
            )
    audit = (
        gpd.GeoDataFrame(records, geometry="geometry", crs=wfs.crs)
        if records
        else gpd.GeoDataFrame(records, geometry=[], crs=wfs.crs)
    )
    stats = {
        "coverage_input_clean_fids": int(len(valid_clean_ids)),
        "coverage_output_clean_fids": int(len(covered_clean_ids)),
        "coverage_missing_clean_fids": int(len(missing_clean_ids)),
        "coverage_duplicate_clean_fids": int(len(duplicate_clean_ids)),
        "coverage_unknown_clean_fids": int(len(unknown_clean_ids)),
        "coverage_audit_rows": int(len(audit)),
    }
    return audit, stats


def _absorb_zero_uprn_hole_fillers(
    *,
    parcels: gpd.GeoDataFrame,
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    source_by_clean: dict[int, int],
    coverage_threshold: float,
) -> tuple[gpd.GeoDataFrame, dict[str, int | float]]:
    if parcels.empty:
        return parcels, {
            "hole_filler_absorb_enabled": 1,
            "hole_filler_absorbed_clean_fids": 0,
            "hole_filler_absorbed_target_rows": 0,
            "hole_filler_coverage_threshold": float(coverage_threshold),
        }

    work = parcels.copy().reset_index(drop=True)
    row_clean_ids = [_parse_id_set(value) for value in work["clean_fids"].fillna("").astype(str)]
    candidate_clean_ids: list[int] = []
    candidate_row_by_clean: dict[int, int] = {}
    for row_index, clean_ids in enumerate(row_clean_ids):
        if len(clean_ids) != 1:
            continue
        clean_id = next(iter(clean_ids))
        attrs = attrs_by_clean.get(int(clean_id), {})
        if int(attrs.get("uprn_count", 0) or 0) > 0:
            continue
        if not bool(attrs.get("plot_eligible", False)):
            continue
        geom = geom_by_clean.get(int(clean_id))
        if geom is None or geom.is_empty:
            continue
        candidate_clean_ids.append(int(clean_id))
        candidate_row_by_clean[int(clean_id)] = int(row_index)

    if not candidate_clean_ids:
        return work, {
            "hole_filler_absorb_enabled": 1,
            "hole_filler_absorbed_clean_fids": 0,
            "hole_filler_absorbed_target_rows": 0,
            "hole_filler_coverage_threshold": float(coverage_threshold),
        }

    candidate_geoms = np.asarray([geom_by_clean[clean_id] for clean_id in candidate_clean_ids], dtype=object)
    tree = shapely.STRtree(candidate_geoms)
    assignments: dict[int, tuple[int, float]] = {}
    threshold = float(coverage_threshold)
    for row_index, row in enumerate(work.itertuples(index=False)):
        geom = getattr(row, "geometry")
        if geom is None or geom.is_empty:
            continue
        holes = _hole_polygons(geom)
        if not holes:
            continue
        own_clean_ids = row_clean_ids[row_index]
        for hole in holes:
            for hit_index in tree.query(hole, predicate="intersects"):
                clean_id = int(candidate_clean_ids[int(hit_index)])
                if clean_id in own_clean_ids:
                    continue
                candidate_row_index = candidate_row_by_clean.get(clean_id)
                if candidate_row_index is None or int(candidate_row_index) == int(row_index):
                    continue
                candidate_geom = geom_by_clean.get(clean_id)
                if candidate_geom is None or candidate_geom.is_empty:
                    continue
                overlap_area = float(shapely.area(shapely.intersection(candidate_geom, hole)))
                coverage = overlap_area / max(float(shapely.area(candidate_geom)), 1e-9)
                if coverage < threshold:
                    continue
                previous = assignments.get(clean_id)
                if previous is None or float(coverage) > float(previous[1]):
                    assignments[clean_id] = (int(row_index), float(coverage))

    if not assignments:
        return work, {
            "hole_filler_absorb_enabled": 1,
            "hole_filler_absorbed_clean_fids": 0,
            "hole_filler_absorbed_target_rows": 0,
            "hole_filler_coverage_threshold": float(coverage_threshold),
        }

    absorbed_by_target: dict[int, set[int]] = {}
    for clean_id, (target_row, _coverage) in assignments.items():
        absorbed_by_target.setdefault(int(target_row), set()).add(int(clean_id))
    absorbed_clean_ids = set(int(clean_id) for clean_id in assignments)

    for row_index, absorbed_ids in absorbed_by_target.items():
        clean_ids = set(row_clean_ids[int(row_index)])
        clean_ids.update(int(clean_id) for clean_id in absorbed_ids)
        geoms = [work.at[int(row_index), "geometry"]]
        geoms.extend(geom_by_clean[int(clean_id)] for clean_id in sorted(absorbed_ids) if int(clean_id) in geom_by_clean)
        geom = shapely.union_all(np.asarray(geoms, dtype=object))
        if geom is not None and not geom.is_empty and not bool(shapely.is_valid(geom)):
            geom = shapely.make_valid(geom)
        source_ids = {int(source_by_clean[clean_id]) for clean_id in clean_ids if clean_id in source_by_clean}
        uprn_count = sum(int(attrs_by_clean[clean_id].get("uprn_count", 0) or 0) for clean_id in clean_ids if clean_id in attrs_by_clean)
        work.at[int(row_index), "geometry"] = geom
        work.at[int(row_index), "clean_fids"] = _ids_text(clean_ids)
        work.at[int(row_index), "source_fids"] = _ids_text(source_ids)
        work.at[int(row_index), "clean_count"] = int(len(clean_ids))
        work.at[int(row_index), "source_count"] = int(len(source_ids))
        work.at[int(row_index), "uprn_count"] = int(uprn_count)
        work.at[int(row_index), "absorbed_hole_filler_count"] = int(len(absorbed_ids))
        work.at[int(row_index), "absorbed_hole_filler_clean_fids"] = _ids_text(absorbed_ids)
        work.at[int(row_index), "absorbed_hole_filler_source_fids"] = _ids_text(
            int(source_by_clean[clean_id]) for clean_id in sorted(absorbed_ids) if clean_id in source_by_clean
        )

    keep_rows = []
    for row_index, clean_ids in enumerate(row_clean_ids):
        if clean_ids and clean_ids.issubset(absorbed_clean_ids):
            continue
        keep_rows.append(int(row_index))
    out = work.iloc[keep_rows].copy().reset_index(drop=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=parcels.crs)
    stats = {
        "hole_filler_absorb_enabled": 1,
        "hole_filler_absorbed_clean_fids": int(len(absorbed_clean_ids)),
        "hole_filler_absorbed_target_rows": int(len(absorbed_by_target)),
        "hole_filler_coverage_threshold": float(coverage_threshold),
    }
    return out, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply raw-clean WFS anchor group model.")
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--candidate-output-csv", default="")
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--review-threshold", type=float, default=0.0)
    parser.add_argument("--anchor-batch-size", type=int, default=2000)
    parser.add_argument("--anchor-workers", type=int, default=min(max(mp.cpu_count(), 1), 16))
    parser.add_argument("--neighbor-depth", type=int, default=None)
    parser.add_argument("--max-pool-size", type=int, default=None)
    parser.add_argument("--max-group-size", type=int, default=None)
    parser.add_argument("--max-candidate-area", type=float, default=8000.0)
    parser.add_argument("--per-anchor-candidate-limit", type=int, default=None)
    parser.add_argument("--shape-supplement-pool-limit", type=int, default=None)
    parser.add_argument("--shape-supplement-keep", type=int, default=None)
    parser.add_argument("--full-score-per-anchor-limit", type=int, default=None)
    parser.add_argument("--proposal-model", default="")
    parser.add_argument("--proposal-expanded-candidate-limit", type=int, default=3000)
    parser.add_argument("--proposal-keep-per-anchor", type=int, default=80)
    parser.add_argument("--proposal-include-base-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-neighbors", type=int, default=None)
    parser.add_argument("--min-shared-edge", type=float, default=None)
    parser.add_argument("--edge-query-chunk-size", type=int, default=20000)
    parser.add_argument("--edge-calc-chunk-size", type=int, default=50000)
    parser.add_argument("--context-cache-dir", default="")
    parser.add_argument("--debug-layer-limit", type=int, default=20000)
    parser.add_argument("--pairwise-replacement-model", default="")
    parser.add_argument("--pairwise-replacement-threshold", type=float, default=1.1)
    parser.add_argument("--pairwise-top-k", type=int, default=30)
    parser.add_argument("--reject-candidate-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-fallback-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-fallback-proba", type=float, default=0.0)
    parser.add_argument("--allow-holed-anchor-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-output-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leftover-attach-zero-uprn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leftover-attach-passes", type=int, default=3)
    parser.add_argument("--leftover-attach-min-shared-edge", type=float, default=0.05)
    parser.add_argument("--leftover-attach-max-added-area", type=float, default=2500.0)
    parser.add_argument("--leftover-attach-max-group-area", type=float, default=12000.0)
    parser.add_argument("--leftover-attach-max-hull-gap-ratio", type=float, default=0.65)
    parser.add_argument("--leftover-attach-regularity-drop", type=float, default=0.25)
    parser.add_argument("--residual-completion-delta", type=float, default=0.0)
    parser.add_argument("--residual-completion-regularity-drop", type=float, default=0.0)
    parser.add_argument("--residual-completion-hull-gap-add", type=float, default=0.0)
    parser.add_argument("--residual-completion-source-count-add", type=int, default=1)
    parser.add_argument("--coverage-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--absorb-hole-fillers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hole-filler-coverage-threshold", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    payload = joblib.load(args.model)
    if not isinstance(payload, dict) or payload.get("model_kind") != "wfs_raw_anchor_group_scorer":
        raise RuntimeError("Model payload must be a wfs_raw_anchor_group_scorer.")
    pipeline = payload["pipeline"]
    feature_cols = list(payload["feature_cols"])
    params = dict(payload.get("training_params", {}))

    bbox = _parse_bbox(args.bbox)
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), bbox)
    wfs = _add_uprn_counts_cached(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
        context_cache_dir=str(args.context_cache_dir),
    )
    source_to_clean, source_by_clean = _build_source_indexes(wfs)
    geom_by_clean = wfs.set_index("clean_fid").geometry.to_dict()
    attrs_by_clean = wfs.set_index("clean_fid").drop(columns="geometry").to_dict("index")

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
    if str(args.candidate_input_csv).strip():
        candidates = read_candidate_inputs(str(args.candidate_input_csv))
        if "raw_anchor_group_proba" not in candidates.columns:
            raise RuntimeError("--candidate-input-csv must contain raw_anchor_group_proba")
        before_filter = len(candidates)
        candidates = candidates[candidates["raw_anchor_group_proba"].astype(float).ge(float(args.review_threshold))].copy()
        score_stats = {
            "anchor_rows": int(len(anchors)),
            "candidate_rows": 0,
            "loaded_candidate_rows": int(before_filter),
            "review_candidate_rows": int(len(candidates)),
        }
        _log(
            "[INFO] Reusing scored candidates: "
            f"loaded={before_filter:,}; kept_after_review_threshold={len(candidates):,}"
        )
    else:
        nodes = wfs[wfs["plot_eligible"].astype(bool)].copy()
        eligible_clean_ids = set(nodes["clean_fid"].astype(int))
        _edges, adjacency, shared_by_pair = _build_edges_cached(
            nodes,
            min_shared_edge=float(_param(args, params, "min_shared_edge", 0.05)),
            top_neighbors=int(_param(args, params, "top_neighbors", 14)),
            query_chunk_size=int(args.edge_query_chunk_size),
            edge_calc_chunk_size=int(args.edge_calc_chunk_size),
            context_cache_dir=str(args.context_cache_dir),
        )
        shape_by_clean: dict[int, dict[str, float]] = {}
        adjacency_lookup = _build_adjacency_lookup(adjacency)
        area_by_clean = wfs.set_index("clean_fid")["area"].astype(float).to_dict()
        candidates, score_stats = _score_anchor_candidates(
            anchors=anchors,
            pipeline=pipeline,
            feature_cols=feature_cols,
            wfs=wfs,
            source_to_clean=source_to_clean,
            source_by_clean=source_by_clean,
            eligible_clean_ids=eligible_clean_ids,
            adjacency=adjacency,
            shared_by_pair=shared_by_pair,
            adjacency_lookup=adjacency_lookup,
            geom_by_clean=geom_by_clean,
            attrs_by_clean=attrs_by_clean,
            shape_by_clean=shape_by_clean,
            area_by_clean=area_by_clean,
            args=args,
            params=params,
        )
    anchor_fallback_stats: dict[str, int | float] = {
        "anchor_fallback_enabled": int(bool(args.anchor_fallback_candidates)),
        "anchor_fallback_rows": 0,
        "anchor_fallback_anchors_without_input_candidates": 0,
        "anchor_fallback_proba": float(args.anchor_fallback_proba),
    }
    if bool(args.anchor_fallback_candidates):
        fallback_candidates, anchor_fallback_stats = _build_anchor_fallback_candidates(
            anchors=anchors,
            source_to_clean=source_to_clean,
            source_by_clean=source_by_clean,
            geom_by_clean=geom_by_clean,
            attrs_by_clean=attrs_by_clean,
            existing_candidates=candidates,
            proba=float(args.anchor_fallback_proba),
        )
        if not fallback_candidates.empty:
            candidates = pd.concat([candidates, fallback_candidates], ignore_index=True, sort=False)
        score_stats["anchor_fallback_rows"] = int(anchor_fallback_stats["anchor_fallback_rows"])
        score_stats["anchor_fallback_anchors_without_input_candidates"] = int(
            anchor_fallback_stats["anchor_fallback_anchors_without_input_candidates"]
        )
        _log(
            "[INFO] Anchor fallback candidates: "
            f"rows={int(anchor_fallback_stats['anchor_fallback_rows']):,}; "
            f"anchors_without_input_candidates="
            f"{int(anchor_fallback_stats['anchor_fallback_anchors_without_input_candidates']):,}; "
            f"proba={float(anchor_fallback_stats['anchor_fallback_proba']):.6g}"
        )
    if str(args.candidate_output_csv).strip():
        candidate_output = Path(args.candidate_output_csv)
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_output, index=False)
        _log(f"[INFO] Wrote scored candidate cache: rows={len(candidates):,}; path={candidate_output}")
    allow_holed_anchor_fallback = bool(args.allow_holed_anchor_fallback) and bool(args.fill_output_holes)
    selected, review, conflicts = _select_candidates(
        candidates,
        threshold=float(args.threshold),
        anchor_owner_by_clean=anchor_owner_by_clean,
        reject_candidate_holes=bool(args.reject_candidate_holes),
        allow_holed_anchor_fallback=bool(allow_holed_anchor_fallback),
    )
    residual_stats = {
        "residual_enabled": 0,
        "residual_replacement_attempts": 0,
        "residual_selected_groups_before_conflict": int(len(selected)),
        "residual_selected_groups": int(len(selected)),
        "residual_conflict_rows": 0,
    }
    pairwise_stats = {
        "pairwise_enabled": 0,
        "pairwise_rows": 0,
        "pairwise_above_threshold": 0,
        "pairwise_replacement_attempts": 0,
        "pairwise_selected_groups_before_conflict": int(len(selected)),
        "pairwise_selected_groups": int(len(selected)),
        "pairwise_conflict_rows": 0,
    }
    if str(args.pairwise_replacement_model).strip() and float(args.pairwise_replacement_threshold) <= 1.0:
        selected, pairwise_conflicts, pairwise_stats = _apply_pairwise_replacement_selector(
            candidates,
            selected,
            pairwise_model_path=str(args.pairwise_replacement_model),
            pairwise_threshold=float(args.pairwise_replacement_threshold),
            pairwise_top_k=int(args.pairwise_top_k),
            anchor_owner_by_clean=anchor_owner_by_clean,
            reject_candidate_holes=bool(args.reject_candidate_holes),
            allow_holed_anchor_fallback=bool(allow_holed_anchor_fallback),
        )
        if not pairwise_conflicts.empty:
            conflicts = pd.concat([conflicts, pairwise_conflicts], ignore_index=True)
    if float(args.residual_completion_delta) > 0.0:
        selected, residual_conflicts, residual_stats = _apply_residual_completion_selector(
            candidates,
            selected,
            anchor_owner_by_clean=anchor_owner_by_clean,
            reject_candidate_holes=bool(args.reject_candidate_holes),
            allow_holed_anchor_fallback=bool(allow_holed_anchor_fallback),
            proba_delta=float(args.residual_completion_delta),
            regularity_drop=float(args.residual_completion_regularity_drop),
            hull_gap_add=float(args.residual_completion_hull_gap_add),
            source_count_add=int(args.residual_completion_source_count_add),
        )
        if not residual_conflicts.empty:
            conflicts = pd.concat([conflicts, residual_conflicts], ignore_index=True)
    leftover_attach_stats: dict[str, int | float] = {
        "leftover_attach_enabled": int(bool(args.leftover_attach_zero_uprn)),
        "leftover_attach_rows": 0,
        "leftover_attach_candidates": 0,
        "leftover_attach_attempts": 0,
        "leftover_attach_passes": 0,
    }
    leftover_attached = gpd.GeoDataFrame([], geometry=[], crs=wfs.crs)
    if bool(args.leftover_attach_zero_uprn):
        selected, leftover_attached, leftover_attach_stats = _attach_leftover_zero_uprn(
            selected,
            wfs=wfs,
            geom_by_clean=geom_by_clean,
            attrs_by_clean=attrs_by_clean,
            source_by_clean=source_by_clean,
            max_passes=int(args.leftover_attach_passes),
            min_shared_edge=float(args.leftover_attach_min_shared_edge),
            max_added_area=float(args.leftover_attach_max_added_area),
            max_group_area=float(args.leftover_attach_max_group_area),
            max_hull_gap_ratio=float(args.leftover_attach_max_hull_gap_ratio),
            regularity_drop=float(args.leftover_attach_regularity_drop),
        )
    parcels, merged_only = _build_output_parcels(
        wfs=wfs,
        selected=selected,
        geom_by_clean=geom_by_clean,
        attrs_by_clean=attrs_by_clean,
        source_by_clean=source_by_clean,
        fill_output_holes=False,
    )
    hole_filler_absorb_stats: dict[str, int | float] = {
        "hole_filler_absorb_enabled": int(bool(args.absorb_hole_fillers)),
        "hole_filler_absorbed_clean_fids": 0,
        "hole_filler_absorbed_target_rows": 0,
        "hole_filler_coverage_threshold": float(args.hole_filler_coverage_threshold),
    }
    if bool(args.absorb_hole_fillers):
        parcels, hole_filler_absorb_stats = _absorb_zero_uprn_hole_fillers(
            parcels=parcels,
            geom_by_clean=geom_by_clean,
            attrs_by_clean=attrs_by_clean,
            source_by_clean=source_by_clean,
            coverage_threshold=float(args.hole_filler_coverage_threshold),
        )
    if bool(args.fill_output_holes) and not parcels.empty:
        parcels = parcels.copy()
        parcels.geometry = parcels.geometry.map(_fill_polygon_holes)
    merged_only = parcels[parcels["merge_status"].astype(str).eq("raw_anchor_group_selected")].copy()
    merged_only = gpd.GeoDataFrame(merged_only, geometry="geometry", crs=parcels.crs)
    coverage_audit, coverage_stats = _build_clean_coverage_audit(wfs=wfs, parcels=parcels)
    found_anchors = _build_found_anchors_layer(wfs)

    _write_layer(parcels, output_gpkg, "predicted_parcels")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only")
    if int(args.debug_layer_limit) > 0:
        _write_layer(found_anchors, output_gpkg, "found_anchors")
        if bool(args.coverage_audit):
            _write_layer(coverage_audit, output_gpkg, "raw_clean_coverage_audit")
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
        if not leftover_attached.empty:
            _write_layer(
                leftover_attached.head(int(args.debug_layer_limit)),
                output_gpkg,
                "raw_anchor_group_leftover_attached",
            )
    if int(args.debug_layer_limit) > 0 and not candidates.empty:
        candidates.drop(columns=[], errors="ignore").sort_values("raw_anchor_group_proba", ascending=False).head(
            int(args.debug_layer_limit)
        ).to_csv(output_gpkg.with_suffix(".top_candidates.csv"), index=False)
    selected.drop(columns=[], errors="ignore").to_csv(output_gpkg.with_suffix(".selected.csv"), index=False)

    selected_clean_count = sum(len(_parse_id_set(value)) for value in selected.get("candidate_clean_fids", []))
    summary = {
        "model": str(args.model),
        "model_kind": payload.get("model_kind"),
        "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
        "wfs_clean_layer": str(args.wfs_clean_layer),
        "uprn_gpkg": str(args.uprn_gpkg),
        "uprn_layer": str(args.uprn_layer),
        "bbox": str(args.bbox),
        "threshold": float(args.threshold),
        "review_threshold": float(args.review_threshold),
        "clean_wfs_rows": int(len(wfs)),
        "anchor_rows": int(len(anchors)),
        "candidate_rows_scored": int(score_stats.get("candidate_rows", 0)),
        "review_candidate_rows": int(len(candidates)),
        "selected_groups": int(len(selected)),
        "selected_clean_fids": int(selected_clean_count),
        "conflict_rows": int(len(conflicts)),
        "review_rows": int(len(review)),
        "output_parcels": int(len(parcels)),
        "merged_only_rows": int(len(merged_only)),
        "found_anchor_rows": int(len(found_anchors)),
        **coverage_stats,
        "score_stats": score_stats,
        "neighbor_depth": int(_param(args, params, "neighbor_depth", 3)),
        "max_pool_size": int(_param(args, params, "max_pool_size", 16)),
        "max_group_size": int(_param(args, params, "max_group_size", 7)),
        "max_candidate_area": float(_param(args, params, "max_candidate_area", 2500.0)),
        "per_anchor_candidate_limit": int(_param(args, params, "per_anchor_candidate_limit", 40)),
        "full_score_per_anchor_limit": int(_param(args, params, "full_score_per_anchor_limit", 12)),
        "top_neighbors": int(_param(args, params, "top_neighbors", 14)),
        "min_shared_edge": float(_param(args, params, "min_shared_edge", 0.05)),
        "shape_supplement_pool_limit": int(_param(args, params, "shape_supplement_pool_limit", 0)),
        "shape_supplement_keep": int(_param(args, params, "shape_supplement_keep", 0)),
        "proposal_model": str(args.proposal_model),
        "proposal_expanded_candidate_limit": int(args.proposal_expanded_candidate_limit),
        "proposal_keep_per_anchor": int(args.proposal_keep_per_anchor),
        "proposal_include_base_candidates": bool(args.proposal_include_base_candidates),
        "reject_candidate_holes": bool(args.reject_candidate_holes),
        "allow_holed_anchor_fallback": bool(allow_holed_anchor_fallback),
        "fill_output_holes": bool(args.fill_output_holes),
        "hole_filler_absorb_stats": hole_filler_absorb_stats,
        "anchor_fallback_stats": anchor_fallback_stats,
        "anchor_workers": int(args.anchor_workers),
        "context_cache_dir": str(args.context_cache_dir),
        "pairwise_stats": pairwise_stats,
        "pairwise_replacement_model": str(args.pairwise_replacement_model),
        "pairwise_replacement_threshold": float(args.pairwise_replacement_threshold),
        "pairwise_top_k": int(args.pairwise_top_k),
        "residual_stats": residual_stats,
        "residual_completion_delta": float(args.residual_completion_delta),
        "residual_completion_regularity_drop": float(args.residual_completion_regularity_drop),
        "residual_completion_hull_gap_add": float(args.residual_completion_hull_gap_add),
        "residual_completion_source_count_add": int(args.residual_completion_source_count_add),
        "leftover_attach_stats": leftover_attach_stats,
        "elapsed_seconds": float(time.monotonic() - started_at),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Raw anchor group apply complete")
    _log(json.dumps(summary, indent=2))
    _log(f"[DONE] output={output_gpkg}")


if __name__ == "__main__":
    main()

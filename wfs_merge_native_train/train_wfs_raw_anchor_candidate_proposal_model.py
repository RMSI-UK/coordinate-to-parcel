#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from train_wfs_raw_anchor_group_model import (
    DEFAULT_TARGET_GPKG,
    DEFAULT_TARGET_LAYER,
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_ID_FIELD,
    DEFAULT_UPRN_LAYER,
    DEFAULT_WFS_CLEAN_GPKG,
    DEFAULT_WFS_CLEAN_LAYER,
    _add_uprn_counts,
    _build_adjacency_lookup,
    _build_edges,
    _build_source_indexes,
    _collect_anchor_pool,
    _enumerate_anchor_groups_ordered,
    _ids_text,
    _parse_bbox,
    _read_clean_wfs,
    _read_targets,
    _safe_ratio,
    _target_clean_set,
)


DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_proposal_model_v1"
MODEL_FILE_NAME = "wfs_raw_anchor_candidate_proposal_model_v1.joblib"
METRICS_FILE_NAME = "wfs_raw_anchor_candidate_proposal_metrics_v1.json"
TARGET_COL = "label"
ID_COLUMNS = {
    "target_train_component_id",
    "anchor_source_fid",
    "candidate_clean_fids",
    "candidate_source_fids",
    "target_clean_fids",
    "target_source_fids",
}
ROLE_ORDER = ("building", "land", "road", "gapfill", "other")


def build_cheap_clean_attrs(
    *,
    source_by_clean: dict[int, int],
    attrs_by_clean: dict[int, dict[str, Any]],
    area_by_clean: dict[int, float],
    perimeter_by_clean: dict[int, float],
) -> dict[int, tuple[int | None, float, float, int, str]]:
    out: dict[int, tuple[int | None, float, float, int, str]] = {}
    all_ids = set(int(v) for v in area_by_clean)
    all_ids.update(int(v) for v in perimeter_by_clean)
    all_ids.update(int(v) for v in attrs_by_clean)
    all_ids.update(int(v) for v in source_by_clean)
    for clean_fid in all_ids:
        attrs = attrs_by_clean.get(int(clean_fid), {})
        role = str(attrs.get("anchor_role", "other") or "other")
        if role not in ROLE_ORDER:
            role = "other"
        source_fid = source_by_clean.get(int(clean_fid))
        out[int(clean_fid)] = (
            int(source_fid) if source_fid is not None else None,
            float(area_by_clean.get(int(clean_fid), 0.0)),
            float(perimeter_by_clean.get(int(clean_fid), 0.0)),
            int(attrs.get("uprn_count", 0) or 0),
            role,
        )
    return out


def _log(message: str) -> None:
    print(message, flush=True)


def _source_ids_for_group(group: frozenset[int], source_by_clean: dict[int, int]) -> frozenset[int]:
    return frozenset(int(source_by_clean[fid]) for fid in group if int(fid) in source_by_clean)


def cheap_candidate_features(
    *,
    anchor_source_fid: int,
    anchor_clean_ids: frozenset[int],
    candidate_clean_ids: frozenset[int],
    enum_rank: int,
    target_train_component_id: int,
    source_by_clean: dict[int, int],
    attrs_by_clean: dict[int, dict[str, Any]],
    area_by_clean: dict[int, float],
    perimeter_by_clean: dict[int, float],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    include_ids: bool = True,
    cheap_attrs_by_clean: dict[int, tuple[int | None, float, float, int, str]] | None = None,
    uprn_clean_ids: set[int] | None = None,
    building_clean_ids: set[int] | None = None,
) -> dict[str, Any]:
    clean_ids = tuple(sorted(int(v) for v in candidate_clean_ids))
    anchor_ids = frozenset(int(v) for v in anchor_clean_ids)
    added_ids = tuple(fid for fid in clean_ids if fid not in anchor_ids)
    candidate_set = frozenset(clean_ids)
    source_ids: set[int] = set()
    role_counts = {role: 0 for role in ROLE_ORDER}
    added_role_counts = {role: 0 for role in ROLE_ORDER}

    area_sum = 0.0
    anchor_area = 0.0
    added_area_sum = 0.0
    added_area_max = 0.0
    added_area_count = 0
    largest_area = 0.0
    smallest_area = math.inf
    perimeter_sum = 0.0
    uprn_count = 0
    for fid in clean_ids:
        if cheap_attrs_by_clean is not None:
            source_fid, area, perimeter, fid_uprn_count, role = cheap_attrs_by_clean.get(
                int(fid),
                (None, 0.0, 0.0, 0, "other"),
            )
        else:
            area = float(area_by_clean.get(fid, 0.0))
            perimeter = float(perimeter_by_clean.get(fid, 0.0))
            attrs = attrs_by_clean.get(fid, {})
            source_fid = source_by_clean.get(fid)
            fid_uprn_count = int(attrs.get("uprn_count", 0) or 0)
            role = str(attrs.get("anchor_role", "other") or "other")
            if role not in role_counts:
                role = "other"
        if source_fid is not None:
            source_ids.add(int(source_fid))
        area_sum += area
        perimeter_sum += perimeter
        largest_area = max(largest_area, area)
        smallest_area = min(smallest_area, area)
        uprn_count += int(fid_uprn_count)
        role_counts[role] += 1
        if fid in anchor_ids:
            anchor_area += area
        else:
            added_area_sum += area
            added_area_max = max(added_area_max, area)
            added_area_count += 1
            added_role_counts[role] += 1
    if smallest_area == math.inf:
        smallest_area = 0.0

    internal_shared = 0.0
    anchor_added_shared = 0.0
    for idx, left in enumerate(clean_ids):
        for right in clean_ids[idx + 1 :]:
            shared = float(shared_by_pair.get((left, right), 0.0))
            internal_shared += shared
            if (left in anchor_ids and right not in anchor_ids) or (right in anchor_ids and left not in anchor_ids):
                anchor_added_shared += shared

    external_shared = 0.0
    outside_neighbor_ids: set[int] = set()
    outside_uprn_neighbor_ids: set[int] = set()
    outside_building_neighbor_ids: set[int] = set()
    for fid in clean_ids:
        for neighbor, shared in adjacency.get(int(fid), []):
            neighbor = int(neighbor)
            if neighbor in candidate_set:
                continue
            external_shared += float(shared)
            outside_neighbor_ids.add(neighbor)
            if (
                neighbor in uprn_clean_ids
                if uprn_clean_ids is not None
                else int(attrs_by_clean.get(neighbor, {}).get("uprn_count", 0) or 0) > 0
            ):
                outside_uprn_neighbor_ids.add(neighbor)
            if (
                neighbor in building_clean_ids
                if building_clean_ids is not None
                else str(attrs_by_clean.get(neighbor, {}).get("anchor_role", "") or "") == "building"
            ):
                outside_building_neighbor_ids.add(neighbor)

    outer_perimeter = max(perimeter_sum - (2.0 * internal_shared), 1e-6)
    compactness = (4.0 * math.pi * area_sum / (outer_perimeter * outer_perimeter)) if area_sum > 0.0 else 0.0
    boundary_simplification = _safe_ratio(perimeter_sum - outer_perimeter, perimeter_sum)

    record: dict[str, Any] = {
        "enum_rank": int(enum_rank),
        "candidate_clean_count": int(len(clean_ids)),
        "candidate_source_count": int(len(source_ids)),
        "anchor_clean_count": int(len(anchor_ids)),
        "added_clean_count": int(len(added_ids)),
        "added_source_count": int(max(len(source_ids) - 1, 0)),
        "area_sum": area_sum,
        "anchor_area": anchor_area,
        "area_to_anchor": _safe_ratio(area_sum, anchor_area),
        "added_area_sum": float(added_area_sum),
        "added_area_max": float(added_area_max),
        "added_area_mean": _safe_ratio(float(added_area_sum), float(added_area_count)),
        "largest_part_area_ratio": _safe_ratio(float(largest_area), area_sum),
        "smallest_part_area_ratio": _safe_ratio(float(smallest_area), area_sum),
        "perimeter_sum": perimeter_sum,
        "internal_shared_len": float(internal_shared),
        "anchor_added_shared_len": float(anchor_added_shared),
        "external_shared_len": float(external_shared),
        "shared_to_external_shared": _safe_ratio(float(internal_shared), float(external_shared)),
        "outer_perimeter_approx": float(outer_perimeter),
        "compactness_approx": float(compactness),
        "boundary_simplification_approx": float(boundary_simplification),
        "uprn_count": int(uprn_count),
        "outside_neighbor_count": int(len(outside_neighbor_ids)),
        "outside_uprn_neighbor_count": int(len(outside_uprn_neighbor_ids)),
        "outside_building_neighbor_count": int(len(outside_building_neighbor_ids)),
        "fast_shape_score": float(
            (2.5 * compactness)
            + (2.0 * boundary_simplification)
            + (0.03 * len(clean_ids))
            - (0.00005 * abs(area_sum - 350.0))
        ),
    }
    if bool(include_ids):
        record.update(
            {
                "target_train_component_id": int(target_train_component_id),
                "anchor_source_fid": int(anchor_source_fid),
                "candidate_clean_fids": _ids_text(candidate_clean_ids),
                "candidate_source_fids": _ids_text(source_ids),
            }
        )
    for role in ROLE_ORDER:
        record[f"role_{role}_count"] = int(role_counts[role])
        record[f"added_role_{role}_count"] = int(added_role_counts[role])
    return record


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataset.columns
        if column not in ID_COLUMNS | {TARGET_COL, "sample_weight", "proposal_proba"}
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def _stable_seed(*parts: object, random_state: int) -> int:
    text = "|".join(str(part) for part in (*parts, random_state))
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32)


def _sample_group_indices(
    *,
    groups: list[frozenset[int]],
    target_clean_ids: frozenset[int],
    target_source_ids: set[int],
    source_by_clean: dict[int, int],
    args: argparse.Namespace,
    target_train_component_id: int,
) -> set[int]:
    if int(args.sample_negatives_per_target) <= 0:
        return set(range(len(groups)))

    selected: set[int] = set()
    exact_indexes: list[int] = []
    partial_indexes: list[int] = []
    overmerge_indexes: list[int] = []
    overlap_mismatch_indexes: list[int] = []
    for idx, group in enumerate(groups):
        if group == target_clean_ids:
            exact_indexes.append(idx)
            continue
        source_ids = _source_ids_for_group(group, source_by_clean)
        if source_ids and source_ids < target_source_ids:
            partial_indexes.append(idx)
        elif source_ids and target_source_ids < source_ids:
            overmerge_indexes.append(idx)
        elif source_ids & target_source_ids:
            overlap_mismatch_indexes.append(idx)

    selected.update(exact_indexes)
    top_rank_keep = max(int(args.sample_top_rank_negatives), 0)
    selected.update(idx for idx in range(min(len(groups), top_rank_keep)) if idx not in exact_indexes)

    hard_keep = max(int(args.sample_hard_negatives_per_class), 0)
    for indexes in [partial_indexes, overmerge_indexes, overlap_mismatch_indexes]:
        selected.update(indexes[:hard_keep])

    budget = max(int(args.sample_negatives_per_target) + len(exact_indexes), len(selected))
    remaining_budget = max(budget - len(selected), 0)
    if remaining_budget > 0:
        remaining = [idx for idx in range(len(groups)) if idx not in selected]
        if remaining:
            rng = np.random.default_rng(
                _stable_seed(target_train_component_id, len(groups), random_state=int(args.random_state))
            )
            take = min(remaining_budget, len(remaining))
            selected.update(int(idx) for idx in rng.choice(np.asarray(remaining, dtype="int64"), size=take, replace=False))
    return selected


def build_proposal_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    bbox = _parse_bbox(args.bbox)
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), bbox)
    wfs = _add_uprn_counts(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
    )
    target = _read_targets(Path(args.target_gpkg), str(args.target_layer), bbox, int(args.max_target_rows))
    source_to_clean, source_by_clean = _build_source_indexes(wfs)
    nodes = wfs[wfs["plot_eligible"].astype(bool)].copy()
    eligible_clean_ids = set(nodes["clean_fid"].astype(int))
    _edges, adjacency, shared_by_pair = _build_edges(
        nodes,
        min_shared_edge=float(args.min_shared_edge),
        top_neighbors=int(args.top_neighbors),
        query_chunk_size=int(args.edge_query_chunk_size),
        edge_calc_chunk_size=int(args.edge_calc_chunk_size),
    )
    adjacency_lookup = _build_adjacency_lookup(adjacency)
    attrs_by_clean = wfs.set_index("clean_fid").drop(columns="geometry").to_dict("index")
    area_by_clean = wfs.set_index("clean_fid")["area"].astype(float).to_dict()
    perimeter_by_clean = wfs.set_index("clean_fid")["perimeter"].astype(float).to_dict()

    records: list[dict[str, Any]] = []
    target_rows = 0
    skipped_missing_sources = 0
    skipped_no_anchor = 0
    generated_exact_targets = 0
    expanded_group_count = 0
    for row_index, row in enumerate(target.itertuples(index=False), start=1):
        target_rows += 1
        target_source_ids = set(int(v) for v in getattr(row, "target_source_set"))
        anchor_source_fid = int(row.anchor_source_fid)
        target_train_component_id = int(row.train_component_id)
        anchor_clean_ids = frozenset(source_to_clean.get(anchor_source_fid, []))
        target_clean_ids, missing_source_ids = _target_clean_set(target_source_ids, source_to_clean)
        if missing_source_ids:
            skipped_missing_sources += 1
            continue
        if not anchor_clean_ids or not target_clean_ids:
            skipped_no_anchor += 1
            continue
        pool = _collect_anchor_pool(
            anchor_clean_ids=anchor_clean_ids,
            positive_clean_ids=frozenset(),
            adjacency=adjacency,
            eligible_clean_ids=eligible_clean_ids,
            max_depth=int(args.neighbor_depth),
            max_pool_size=int(args.max_pool_size),
        )
        groups = _enumerate_anchor_groups_ordered(
            anchor_clean_ids=anchor_clean_ids,
            pool=pool,
            adjacency=adjacency,
            area_by_clean=area_by_clean,
            max_group_size=int(args.max_group_size),
            max_candidate_area=float(args.max_candidate_area),
            per_anchor_limit=int(args.expanded_candidate_limit),
            adjacency_lookup=adjacency_lookup,
        )
        expanded_group_count += int(len(groups))
        if target_clean_ids in set(groups):
            generated_exact_targets += 1
        selected_group_indexes = _sample_group_indices(
            groups=groups,
            target_clean_ids=target_clean_ids,
            target_source_ids=target_source_ids,
            source_by_clean=source_by_clean,
            args=args,
            target_train_component_id=target_train_component_id,
        )
        for rank, group in enumerate(groups, start=1):
            if (rank - 1) not in selected_group_indexes:
                continue
            record = cheap_candidate_features(
                anchor_source_fid=anchor_source_fid,
                anchor_clean_ids=anchor_clean_ids,
                candidate_clean_ids=group,
                enum_rank=rank,
                target_train_component_id=target_train_component_id,
                source_by_clean=source_by_clean,
                attrs_by_clean=attrs_by_clean,
                area_by_clean=area_by_clean,
                perimeter_by_clean=perimeter_by_clean,
                adjacency=adjacency,
                shared_by_pair=shared_by_pair,
            )
            record["target_clean_fids"] = _ids_text(target_clean_ids)
            record["target_source_fids"] = _ids_text(target_source_ids)
            record[TARGET_COL] = int(group == target_clean_ids)
            records.append(record)
        if row_index % int(args.log_every_targets) == 0:
            _log(
                "[INFO] Proposal targets "
                f"{row_index:,}/{len(target):,}; sampled_rows={len(records):,}; "
                f"expanded_groups={expanded_group_count:,}; generated_exact_targets={generated_exact_targets:,}"
            )

    if not records:
        raise RuntimeError("No proposal candidates were generated.")
    dataset = pd.DataFrame.from_records(records)
    summary = {
        "target_rows": int(target_rows),
        "candidate_rows": int(len(dataset)),
        "expanded_group_count": int(expanded_group_count),
        "positive_rows": int(dataset[TARGET_COL].astype(int).sum()),
        "generated_exact_targets": int(generated_exact_targets),
        "skipped_missing_source_targets": int(skipped_missing_sources),
        "skipped_no_anchor_targets": int(skipped_no_anchor),
        "generation_exact_recall_all_targets": _safe_ratio(float(generated_exact_targets), float(target_rows)),
        "generation_exact_recall_available_targets": _safe_ratio(
            float(generated_exact_targets),
            float(target_rows - skipped_missing_sources - skipped_no_anchor),
        ),
        "sample_negatives_per_target": int(args.sample_negatives_per_target),
        "sample_top_rank_negatives": int(args.sample_top_rank_negatives),
        "sample_hard_negatives_per_class": int(args.sample_hard_negatives_per_class),
    }
    _log("[INFO] Proposal candidate summary:")
    _log(json.dumps(summary, indent=2))
    return dataset, summary


def _sample_fit_rows(dataset: pd.DataFrame, *, max_negative_rows: int, random_state: int) -> pd.DataFrame:
    positive = dataset[dataset[TARGET_COL].astype(int).eq(1)]
    negative = dataset[dataset[TARGET_COL].astype(int).eq(0)]
    if int(max_negative_rows) > 0 and len(negative) > int(max_negative_rows):
        negative = negative.sample(n=int(max_negative_rows), random_state=int(random_state))
    fit = pd.concat([positive, negative], ignore_index=True).sample(frac=1.0, random_state=int(random_state))
    fit["sample_weight"] = 1.0
    fit.loc[fit[TARGET_COL].astype(int).eq(1), "sample_weight"] = max(
        float(len(negative)) / max(float(len(positive)), 1.0),
        1.0,
    )
    return fit


def _rank_recall_by_group(dataset: pd.DataFrame, score_col: str, ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups_with_positive = 0
    for k in ks:
        ok = 0
        denom = 0
        for _target_id, group in dataset.groupby("target_train_component_id", sort=False):
            if int(group[TARGET_COL].astype(int).sum()) == 0:
                continue
            denom += 1
            top = group.sort_values([score_col, "fast_shape_score"], ascending=[False, False]).head(int(k))
            ok += int(top[TARGET_COL].astype(int).sum() > 0)
        groups_with_positive = max(groups_with_positive, denom)
        out[f"recall_at_{int(k)}"] = {
            "ok": int(ok),
            "denom": int(denom),
            "recall": _safe_ratio(float(ok), float(denom)),
        }
    out["groups_with_positive"] = int(groups_with_positive)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight raw-anchor candidate proposal ranker.")
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--target-gpkg", default=DEFAULT_TARGET_GPKG)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--neighbor-depth", type=int, default=3)
    parser.add_argument("--max-pool-size", type=int, default=22)
    parser.add_argument("--max-group-size", type=int, default=10)
    parser.add_argument("--max-candidate-area", type=float, default=6000.0)
    parser.add_argument("--expanded-candidate-limit", type=int, default=3000)
    parser.add_argument("--top-neighbors", type=int, default=14)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--edge-query-chunk-size", type=int, default=20000)
    parser.add_argument("--edge-calc-chunk-size", type=int, default=50000)
    parser.add_argument("--sample-negatives-per-target", type=int, default=80)
    parser.add_argument("--sample-top-rank-negatives", type=int, default=25)
    parser.add_argument("--sample-hard-negatives-per-class", type=int, default=20)
    parser.add_argument("--max-negative-train-rows", type=int, default=160000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rank-recall-at", default="20,40,80,120,160,200")
    parser.add_argument("--log-every-targets", type=int, default=500)
    parser.add_argument("--skip-dataset-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, build_summary = build_proposal_dataset(args)
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
    feature_cols = _feature_columns(dataset)
    if int(dataset[TARGET_COL].sum()) == 0:
        raise RuntimeError("No positive proposal rows are available.")
    fit_dataset = _sample_fit_rows(
        dataset,
        max_negative_rows=int(args.max_negative_train_rows),
        random_state=int(args.random_state),
    )

    groups = fit_dataset["anchor_source_fid"].astype(int).to_numpy()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.random_state)).split(
            fit_dataset,
            fit_dataset[TARGET_COL],
            groups=groups,
        )
    )
    train = fit_dataset.iloc[train_idx].copy()
    test = fit_dataset.iloc[test_idx].copy()
    _log(
        "[INFO] Training proposal model: "
        f"fit_rows={len(fit_dataset):,}; train={len(train):,}; test={len(test):,}; "
        f"fit_labels={fit_dataset[TARGET_COL].value_counts().to_dict()}; features={len(feature_cols)}"
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(
        train[feature_cols],
        train[TARGET_COL],
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    test = test.copy()
    test["proposal_proba"] = test_proba
    ks = [int(part.strip()) for part in str(args.rank_recall_at).split(",") if part.strip()]
    test_metrics = {
        "rows": int(len(test)),
        "positive_rows": int(test[TARGET_COL].astype(int).sum()),
        "roc_auc": float(roc_auc_score(test[TARGET_COL], test_proba)),
        "average_precision": float(average_precision_score(test[TARGET_COL], test_proba)),
        "rank_recall": _rank_recall_by_group(test, "proposal_proba", ks),
    }

    final_model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=1,
    )
    final_pipeline = Pipeline([("preprocess", preprocessor), ("model", final_model)])
    final_pipeline.fit(
        fit_dataset[feature_cols],
        fit_dataset[TARGET_COL],
        model__sample_weight=fit_dataset["sample_weight"].astype(float).to_numpy(),
    )

    payload = {
        "model_kind": "wfs_raw_anchor_candidate_proposal_ranker",
        "pipeline": final_pipeline,
        "feature_cols": feature_cols,
        "training_params": {
            "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
            "wfs_clean_layer": str(args.wfs_clean_layer),
            "uprn_gpkg": str(args.uprn_gpkg),
            "uprn_layer": str(args.uprn_layer),
            "target_gpkg": str(args.target_gpkg),
            "target_layer": str(args.target_layer),
            "bbox": str(args.bbox),
            "max_target_rows": int(args.max_target_rows),
            "neighbor_depth": int(args.neighbor_depth),
            "max_pool_size": int(args.max_pool_size),
            "max_group_size": int(args.max_group_size),
            "max_candidate_area": float(args.max_candidate_area),
            "expanded_candidate_limit": int(args.expanded_candidate_limit),
            "top_neighbors": int(args.top_neighbors),
            "min_shared_edge": float(args.min_shared_edge),
            "max_negative_train_rows": int(args.max_negative_train_rows),
            "sample_negatives_per_target": int(args.sample_negatives_per_target),
            "sample_top_rank_negatives": int(args.sample_top_rank_negatives),
            "sample_hard_negatives_per_class": int(args.sample_hard_negatives_per_class),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)
    metrics = {
        "model_kind": payload["model_kind"],
        "model": str(output_dir / MODEL_FILE_NAME),
        "build_summary": build_summary,
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "feature_columns": feature_cols,
        "test_metrics": test_metrics,
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not bool(args.skip_dataset_output):
        dataset.head(500000).to_csv(output_dir / "wfs_raw_anchor_candidate_proposal_dataset_sample_v1.csv", index=False)
    _log("[DONE] Proposal model training complete")
    _log(json.dumps(test_metrics, indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
import pyogrio
import shapely
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_INPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/"
    "03_operation_pruned_only.gpkg"
)
DEFAULT_PAIR_CANDIDATE_CSV = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_anchor_group_v2_spatial/"
    "04_pair_anchor_candidates_v2.csv"
)
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_anchor_group_v3_uprn_skeleton"
MODEL_FILE_NAME = "anchor_group_repair_model_v1.joblib"
CANDIDATES_FILE_NAME = "anchor_group_repair_candidates_v1.csv"
PREDICTIONS_FILE_NAME = "anchor_group_repair_predictions_v1.csv"

TARGET_COL = "label"
CATEGORICAL_FEATURES = ["role_pair_signature"]
ID_COLS = {
    "anchor_component_id",
    "zero_component_ids",
    "candidate_source_fids",
    "source_target_best_fids",
    "anchor_reference_fids",
    "zero_reference_fids",
    "reference_positive_zero_component_ids",
    "label_source",
}
MODEL_OUTPUT_COLS = {
    "anchor_group_repair_proba",
    "anchor_group_repair_model_proba",
    "anchor_group_repair_pred_at_threshold",
    "anchor_need_repair_proba",
    "anchor_gate_bypass_complete_pool",
    "anchor_gate_bypass_uprn_skeleton",
    "anchor_group_repair_skeleton_pair_override",
    "anchor_group_repair_residual_fallback",
    "anchor_group_repair_parent_proba",
}
REFERENCE_DERIVED_FEATURE_MARKERS = ("reference", "same_reference", "possible_split")
LABEL_DERIVED_FEATURE_MARKERS = REFERENCE_DERIVED_FEATURE_MARKERS + ("source_target",)


def _model_kind_for_label_mode(label_mode: str) -> str:
    if label_mode == "final_selection":
        return "anchor_group_final_selector"
    if label_mode == "source_target":
        return "anchor_group_source_target_pure"
    return "anchor_group_repair_with_gate"


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _parse_groups(text: str) -> dict[int, set[int]]:
    groups: dict[int, set[int]] = {}
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        anchor_text, zero_text = item.split(":", 1)
        try:
            anchor_id = int(anchor_text.strip())
        except ValueError:
            continue
        zero_ids: set[int] = set()
        for part in zero_text.replace(",", "|").split("|"):
            part = part.strip()
            if not part:
                continue
            try:
                zero_ids.add(int(part))
            except ValueError:
                continue
        if zero_ids:
            groups[anchor_id] = zero_ids
    return groups


def _ids_text(values: set[int] | tuple[int, ...] | list[int]) -> str:
    return "|".join(str(v) for v in sorted(int(x) for x in values))


def _group_key(anchor_id: object, zero_ids: object) -> tuple[int, str]:
    return int(anchor_id), _ids_text(_parse_id_set(zero_ids))


def _parse_id_set(value: object) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _update_prefixed(row: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = float(value)


def _source_theme_text(sources: pd.DataFrame) -> pd.Series:
    pieces = []
    for column in ["role", "Theme", "DescriptiveGroup", "DescriptiveTerm"]:
        if column in sources.columns:
            pieces.append(sources[column].fillna("").astype(str))
    if not pieces:
        return pd.Series("", index=sources.index)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.str.cat(piece, sep=" ")
    return out.str.lower()


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be minx,miny,maxx,maxy")
    minx, miny, maxx, maxy = (float(part) for part in parts)
    if minx >= maxx or miny >= maxy:
        raise ValueError("--bbox must satisfy minx < maxx and miny < maxy")
    return minx, miny, maxx, maxy


def _component_ids_in_bbox(input_gpkg: Path, bbox: tuple[float, float, float, float] | None) -> set[int] | None:
    if bbox is None:
        return None
    predicted = pyogrio.read_dataframe(
        input_gpkg,
        layer="predicted_parcels_with_uprn",
        columns=["pred_component_id"],
        bbox=bbox,
    )
    if predicted.empty:
        return set()
    return {int(value) for value in predicted["pred_component_id"].dropna().astype(int)}


def _source_enclosure_stats(
    sources: pd.DataFrame,
    candidate_mask: pd.Series,
    *,
    min_boundary_ratio: float,
    min_neighbors: int,
    min_shared_edge: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    enclosed = pd.Series(False, index=sources.index)
    boundary_ratio = pd.Series(0.0, index=sources.index)
    neighbor_count = pd.Series(0, index=sources.index)
    candidate_positions = np.flatnonzero(candidate_mask.fillna(False).to_numpy(dtype=bool))
    if len(candidate_positions) == 0:
        return enclosed, boundary_ratio, neighbor_count

    geoms = np.asarray(sources.geometry.to_numpy(), dtype=object)
    tree = shapely.STRtree(geoms)
    index_values = sources.index.to_numpy()
    perimeters = np.asarray(shapely.length(geoms), dtype="float64")
    shared_sum_by_pos = np.zeros(len(sources), dtype="float64")
    shared_neighbor_count_by_pos = np.zeros(len(sources), dtype="int32")
    chunk_size = 1000

    for start in range(0, len(candidate_positions), chunk_size):
        chunk_positions = candidate_positions[start : start + chunk_size]
        chunk_geoms = geoms[chunk_positions]
        try:
            raw_pairs = tree.query(chunk_geoms, predicate="touches")
        except TypeError:
            raw_pairs = tree.query(chunk_geoms)
        if not isinstance(raw_pairs, np.ndarray) or raw_pairs.size == 0:
            continue

        if raw_pairs.ndim == 2:
            left_local = raw_pairs[0].astype("int64", copy=False)
            right_pos = raw_pairs[1].astype("int64", copy=False)
            left_pos = chunk_positions[left_local]
        else:
            chunk_shared = np.zeros(len(chunk_positions), dtype="float64")
            chunk_count = np.zeros(len(chunk_positions), dtype="int32")
            for local_idx, candidate_pos in enumerate(chunk_positions):
                neighbor_positions = np.asarray(raw_pairs, dtype="int64")
                keep = neighbor_positions != int(candidate_pos)
                if not bool(np.any(keep)):
                    continue
                neighbor_positions = neighbor_positions[keep]
                shared = np.asarray(
                    shapely.length(
                        shapely.intersection(
                            shapely.boundary(np.repeat(geoms[int(candidate_pos)], len(neighbor_positions))),
                            shapely.boundary(geoms[neighbor_positions]),
                        )
                    ),
                    dtype="float64",
                )
                good = shared >= float(min_shared_edge)
                chunk_shared[local_idx] = float(shared[good].sum())
                chunk_count[local_idx] = int(np.sum(good))
            shared_sum_by_pos[chunk_positions] += chunk_shared
            shared_neighbor_count_by_pos[chunk_positions] += chunk_count
            continue

        keep = left_pos != right_pos
        if not bool(np.any(keep)):
            continue
        left_pos = left_pos[keep]
        right_pos = right_pos[keep]
        left_local = left_local[keep]
        shared = np.asarray(
            shapely.length(
                shapely.intersection(
                    shapely.boundary(geoms[left_pos]),
                    shapely.boundary(geoms[right_pos]),
                )
            ),
            dtype="float64",
        )
        good = shared >= float(min_shared_edge)
        if not bool(np.any(good)):
            continue
        chunk_shared = np.zeros(len(chunk_positions), dtype="float64")
        chunk_count = np.zeros(len(chunk_positions), dtype="int32")
        np.add.at(chunk_shared, left_local[good], shared[good])
        np.add.at(chunk_count, left_local[good], 1)
        shared_sum_by_pos[chunk_positions] += chunk_shared
        shared_neighbor_count_by_pos[chunk_positions] += chunk_count

    valid = perimeters[candidate_positions] > 0.0
    valid_positions = candidate_positions[valid]
    ratios = np.zeros(len(valid_positions), dtype="float64")
    ratios[:] = np.minimum(shared_sum_by_pos[valid_positions] / perimeters[valid_positions], 1.0)
    counts = shared_neighbor_count_by_pos[valid_positions]
    source_indexes = index_values[valid_positions]
    boundary_ratio.loc[source_indexes] = ratios
    neighbor_count.loc[source_indexes] = counts
    enclosed.loc[source_indexes] = (ratios >= float(min_boundary_ratio)) & (counts >= int(min_neighbors))

    return enclosed, boundary_ratio, neighbor_count


def _reference_is_single(value: object) -> bool:
    text = str(value or "")
    return bool(text) and "|" not in text


def _label_group(
    *,
    anchor_id: int,
    zero_ids: set[int],
    reference_positive_zero_ids: set[int],
    manual_positive_groups: dict[int, set[int]],
) -> tuple[int, str, float]:
    manual = manual_positive_groups.get(int(anchor_id))
    if manual:
        if zero_ids == manual:
            return 1, "manual_complete_positive", 500.0
        if zero_ids & manual:
            return 0, "manual_partial_or_overmerge_negative", 2.0

    if reference_positive_zero_ids:
        if zero_ids == reference_positive_zero_ids:
            return 1, "reference_complete_positive", 6.0
        if zero_ids < reference_positive_zero_ids:
            return 0, "reference_partial_negative", 5.0
        if zero_ids & reference_positive_zero_ids:
            return 0, "reference_overmerge_negative", 5.0
    return 0, "reference_different_negative", 1.0


def _apply_manual_group_labels(
    dataset: pd.DataFrame,
    manual_positive_groups: dict[int, set[int]],
) -> pd.DataFrame:
    if not manual_positive_groups:
        return dataset
    out = dataset.copy()
    for anchor_id, manual_zero_ids in manual_positive_groups.items():
        anchor_mask = out["anchor_component_id"].astype(int).eq(int(anchor_id))
        if not bool(anchor_mask.any()):
            continue
        for idx in out[anchor_mask].index:
            zero_ids = _parse_id_set(out.at[idx, "zero_component_ids"])
            if zero_ids == manual_zero_ids:
                out.at[idx, TARGET_COL] = 1
                out.at[idx, "label_source"] = "manual_complete_positive"
                out.at[idx, "sample_weight"] = 500.0
            elif zero_ids & manual_zero_ids:
                out.at[idx, TARGET_COL] = 0
                out.at[idx, "label_source"] = "manual_partial_or_overmerge_negative"
                out.at[idx, "sample_weight"] = 2.0
    return out


def _apply_spatial_pattern_weak_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {
        "label_source",
        "group_zero_component_count",
        "local_context_score",
        "local_same_pattern_anchor_child_count_65m",
        "after_regularity_score",
        "after_mrr_ratio",
        "after_hull_gap_ratio",
        "pair_shared_edge_sum",
    }
    if not required.issubset(dataset.columns):
        return dataset
    out = dataset.copy()
    eligible = out[
        out["label_source"].astype(str).eq("reference_different_negative")
        & out["group_zero_component_count"].astype(float).ge(2)
        & out["local_context_score"].astype(float).ge(2.0)
        & out["local_same_pattern_anchor_child_count_65m"].astype(float).ge(8)
        & out["after_regularity_score"].astype(float).ge(0.94)
        & out["after_mrr_ratio"].astype(float).ge(0.88)
        & out["after_hull_gap_ratio"].astype(float).le(0.04)
        & out["pair_shared_edge_sum"].astype(float).ge(6.0)
    ].copy()
    if eligible.empty:
        return out
    chosen = eligible.sort_values(
        [
            "anchor_component_id",
            "group_zero_component_count",
            "after_regularity_score",
            "pair_shared_edge_sum",
        ],
        ascending=[True, False, False, False],
    ).drop_duplicates("anchor_component_id", keep="first")
    out.loc[chosen.index, TARGET_COL] = 1
    out.loc[chosen.index, "label_source"] = "spatial_pattern_complete_positive"
    out.loc[chosen.index, "sample_weight"] = 8.0
    return out


def _apply_uprn_skeleton_weak_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {
        "label_source",
        "anchor_uprn_count",
        "group_zero_component_count",
        "group_zero_uprn_land_building_component_count",
        "omitted_zero_uprn_land_building_component_count",
        "group_zero_uprn_land_building_fraction_of_pool",
        "after_area",
        "after_regularity_score",
        "after_hull_gap_ratio",
        "pair_shared_edge_sum",
        "pair_repair_proba_min",
        "pair_zero_anchor_rank_max",
        "pair_shared_margin_min",
    }
    if not required.issubset(dataset.columns):
        return dataset
    out = dataset.copy()
    skeleton_pool = out.get(
        "anchor_pool_zero_uprn_land_building_component_count",
        pd.Series(0.0, index=out.index),
    ).fillna(0.0).astype(float)
    skeleton_count = out["group_zero_uprn_land_building_component_count"].fillna(0).astype(float)
    omitted = out["omitted_zero_uprn_land_building_component_count"].fillna(999.0).astype(float)
    complete_skeleton = (
        skeleton_pool.gt(0)
        & omitted.le(0)
        & out["group_zero_uprn_land_building_fraction_of_pool"].fillna(0.0).astype(float).ge(0.999)
    )
    unambiguous_anchor = (
        out["pair_zero_anchor_rank_max"].fillna(999.0).astype(float).le(1.0)
        | out["pair_shared_margin_min"].fillna(0.0).astype(float).ge(1.25)
    )
    eligible = out[
        out["label_source"].astype(str).eq("reference_different_negative")
        & out["anchor_uprn_count"].fillna(0).astype(float).gt(0)
        & skeleton_count.ge(1)
        & skeleton_count.eq(out["group_zero_component_count"].fillna(0).astype(float))
        & complete_skeleton
        & out["after_area"].fillna(999999.0).astype(float).le(2000.0)
        & out["after_regularity_score"].fillna(0.0).astype(float).ge(0.58)
        & out["after_hull_gap_ratio"].fillna(9.0).astype(float).le(0.50)
        & out["pair_shared_edge_sum"].fillna(0.0).astype(float).ge(1.0)
        & out["pair_repair_proba_min"].fillna(0.0).astype(float).ge(0.05)
        & unambiguous_anchor
    ].copy()
    if not eligible.empty:
        chosen = eligible.sort_values(
            [
                "anchor_component_id",
                "group_zero_uprn_land_building_component_count",
                "after_regularity_score",
                "pair_shared_edge_sum",
                "pair_repair_proba_min",
            ],
            ascending=[True, False, False, False, False],
        ).drop_duplicates("anchor_component_id", keep="first")
        out.loc[chosen.index, TARGET_COL] = 1
        out.loc[chosen.index, "label_source"] = "uprn_skeleton_complete_positive"
        out.loc[chosen.index, "sample_weight"] = 14.0

    partial = (
        out["label_source"].astype(str).eq("reference_different_negative")
        & out["anchor_uprn_count"].fillna(0).astype(float).gt(0)
        & skeleton_count.ge(1)
        & omitted.gt(0)
        & out["after_area"].fillna(999999.0).astype(float).le(2000.0)
    )
    out.loc[partial, "label_source"] = "uprn_skeleton_partial_negative"
    out.loc[partial, "sample_weight"] = np.maximum(
        out.loc[partial, "sample_weight"].astype(float).to_numpy(),
        np.full(int(partial.sum()), 3.0),
    )
    return out


def _apply_enclosed_zero_uprn_weak_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {
        "label_source",
        "anchor_pool_enclosed_zero_uprn_land_building_component_count",
        "group_enclosed_zero_uprn_land_building_component_count",
        "group_enclosed_zero_uprn_land_building_fraction_of_pool",
        "omitted_enclosed_zero_uprn_land_building_component_count",
        "after_area",
        "after_regularity_score",
        "after_hull_gap_ratio",
    }
    if not required.issubset(dataset.columns):
        return dataset
    out = dataset.copy()
    label_source = out["label_source"].astype(str)
    pool = out["anchor_pool_enclosed_zero_uprn_land_building_component_count"].fillna(0).astype(float)
    selected = out["group_enclosed_zero_uprn_land_building_component_count"].fillna(0).astype(float)
    omitted = out["omitted_enclosed_zero_uprn_land_building_component_count"].fillna(0).astype(float)
    fraction = out["group_enclosed_zero_uprn_land_building_fraction_of_pool"].fillna(0.0).astype(float)
    after_area = out["after_area"].fillna(999999.0).astype(float)
    regularity = out["after_regularity_score"].fillna(0.0).astype(float)
    hull_gap = out["after_hull_gap_ratio"].fillna(9.0).astype(float)

    complete = pool.gt(0) & selected.gt(0) & omitted.le(0) & fraction.ge(0.999)
    shape_ok = after_area.le(2000.0) & regularity.ge(0.58) & hull_gap.le(0.50)
    positive = (
        complete
        & shape_ok
        & label_source.isin(["reference_different_negative", "uprn_skeleton_partial_negative"])
    )
    if bool(positive.any()):
        chosen = out.loc[positive].sort_values(
            [
                "anchor_component_id",
                "group_enclosed_zero_uprn_land_building_component_count",
                "after_regularity_score",
                "pair_shared_edge_sum",
            ],
            ascending=[True, False, False, False],
        ).drop_duplicates("anchor_component_id", keep="first")
        out.loc[chosen.index, TARGET_COL] = 1
        out.loc[chosen.index, "label_source"] = "enclosed_zero_uprn_complete_positive"
        out.loc[chosen.index, "sample_weight"] = 24.0

    partial = (
        pool.gt(0)
        & omitted.gt(0)
        & after_area.le(2000.0)
        & ~label_source.eq("manual_complete_positive")
    )
    if bool(partial.any()):
        out.loc[partial, TARGET_COL] = 0
        out.loc[partial, "label_source"] = "enclosed_zero_uprn_partial_negative"
        out.loc[partial, "sample_weight"] = np.maximum(
            out.loc[partial, "sample_weight"].astype(float).to_numpy(),
            np.full(int(partial.sum()), 12.0),
        )

    irregular = (
        pool.gt(0)
        & selected.gt(0)
        & (regularity.lt(0.45) | hull_gap.gt(0.65))
        & ~out["label_source"].astype(str).eq("manual_complete_positive")
    )
    if bool(irregular.any()):
        out.loc[irregular, TARGET_COL] = 0
        out.loc[irregular, "label_source"] = "enclosed_zero_uprn_irregular_negative"
        out.loc[irregular, "sample_weight"] = np.maximum(
            out.loc[irregular, "sample_weight"].astype(float).to_numpy(),
            np.full(int(irregular.sum()), 18.0),
        )
    return out


def _apply_anchor_building_guard_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {
        "anchor_anchor_building_source_count",
        "group_anchor_building_source_count",
        "label_source",
        "sample_weight",
    }
    if not required.issubset(dataset.columns):
        return dataset
    out = dataset.copy()
    anchor_has_building = out["anchor_anchor_building_source_count"].fillna(0).astype(float).gt(0)
    group_anchor_buildings = out["group_anchor_building_source_count"].fillna(0).astype(float)
    adds_second_anchor_building = (anchor_has_building & group_anchor_buildings.gt(0)) | group_anchor_buildings.gt(1)
    if not bool(adds_second_anchor_building.any()):
        return out
    out.loc[adds_second_anchor_building, TARGET_COL] = 0
    out.loc[adds_second_anchor_building, "label_source"] = "adds_second_anchor_building_negative"
    out.loc[adds_second_anchor_building, "sample_weight"] = np.maximum(
        out.loc[adds_second_anchor_building, "sample_weight"].astype(float).to_numpy(),
        np.full(int(adds_second_anchor_building.sum()), 80.0),
    )
    return out


def _apply_final_selection_labels(
    dataset: pd.DataFrame,
    *,
    selection_gpkg: Path,
    selection_layer: str,
    positive_weight: float,
    negative_weight: float,
) -> pd.DataFrame:
    if not selection_gpkg.exists():
        raise FileNotFoundError(f"Selection GeoPackage does not exist: {selection_gpkg}")
    selected = pyogrio.read_dataframe(
        selection_gpkg,
        layer=selection_layer,
        columns=["anchor_component_id", "zero_component_ids"],
    )
    selected_keys = {
        _group_key(row.anchor_component_id, row.zero_component_ids)
        for row in selected.itertuples(index=False)
    }
    out = dataset.copy()
    keys = [_group_key(row.anchor_component_id, row.zero_component_ids) for row in out.itertuples(index=False)]
    positive = pd.Series([key in selected_keys for key in keys], index=out.index)
    out[TARGET_COL] = positive.astype(int)
    out["label_source"] = np.where(
        positive,
        "final_selection_positive",
        "final_selection_negative",
    )
    out["sample_weight"] = np.where(
        positive,
        float(positive_weight),
        float(negative_weight),
    )
    _log(
        "[INFO] Applied final-selection labels: "
        f"selected_layer_rows={len(selected):,}; matched_positive={int(positive.sum()):,}; "
        f"candidate_rows={len(out):,}"
    )
    if len(selected_keys) and int(positive.sum()) == 0:
        raise RuntimeError("Final-selection labels did not match any anchor group candidates.")
    return out


def _load_component_source_fids(input_gpkg: Path) -> dict[int, frozenset[int]]:
    sources = pyogrio.read_dataframe(
        input_gpkg,
        layer="prediction_source_polygons",
        columns=["pred_component_id", "source_fid"],
        read_geometry=False,
    )
    required = {"pred_component_id", "source_fid"}
    if not required.issubset(sources.columns):
        raise RuntimeError(
            "prediction_source_polygons must contain pred_component_id and source_fid "
            "for --label-mode source_target"
        )
    sources = sources.dropna(subset=["pred_component_id", "source_fid"]).copy()
    sources["pred_component_id"] = sources["pred_component_id"].astype(int)
    sources["source_fid"] = sources["source_fid"].astype(int)
    return {
        int(component_id): frozenset(int(value) for value in group["source_fid"].to_numpy())
        for component_id, group in sources.groupby("pred_component_id", sort=False)
    }


def _load_source_target_sets(
    target_gpkg: Path,
    target_layer: str,
) -> tuple[dict[frozenset[int], int], dict[int, set[int]], list[frozenset[int]]]:
    if not target_gpkg.exists():
        raise FileNotFoundError(f"Source-target GeoPackage does not exist: {target_gpkg}")
    target = pyogrio.read_dataframe(
        target_gpkg,
        layer=target_layer,
        columns=["train_component_id", "source_wfs_fids"],
        read_geometry=False,
    )
    if "source_wfs_fids" not in target.columns:
        raise RuntimeError(f"{target_layer} must contain source_wfs_fids")

    target_sets: list[frozenset[int]] = []
    target_set_to_id: dict[frozenset[int], int] = {}
    fid_to_target_ids: dict[int, set[int]] = {}
    duplicate_count = 0
    for idx, row in enumerate(target.itertuples(index=False), start=1):
        source_set = frozenset(_parse_id_set(getattr(row, "source_wfs_fids")))
        if not source_set:
            continue
        component_id = getattr(row, "train_component_id", None)
        try:
            target_id = int(component_id)
        except (TypeError, ValueError):
            target_id = int(idx)
        if source_set in target_set_to_id:
            duplicate_count += 1
            continue
        target_sets.append(source_set)
        target_set_to_id[source_set] = target_id
        target_index = len(target_sets) - 1
        for source_fid in source_set:
            fid_to_target_ids.setdefault(int(source_fid), set()).add(target_index)

    _log(
        "[INFO] Loaded source-target labels: "
        f"target_rows={len(target):,}; unique_source_sets={len(target_sets):,}; duplicates={duplicate_count:,}"
    )
    if not target_sets:
        raise RuntimeError("No source-target label sets were loaded.")
    return target_set_to_id, fid_to_target_ids, target_sets


def _apply_source_target_labels(
    dataset: pd.DataFrame,
    *,
    input_gpkg: Path,
    target_gpkg: Path,
    target_layer: str,
    positive_weight: float,
    negative_weight: float,
) -> pd.DataFrame:
    component_sources = _load_component_source_fids(input_gpkg)
    target_set_to_id, fid_to_target_ids, target_sets = _load_source_target_sets(target_gpkg, target_layer)

    labels: list[int] = []
    label_sources: list[str] = []
    weights: list[float] = []
    candidate_source_texts: list[str] = []
    best_target_texts: list[str] = []
    best_overlap_counts: list[int] = []
    best_overlap_ratios: list[float] = []
    exact_positive_count = 0
    missing_component_count = 0

    for row in dataset.itertuples(index=False):
        component_ids = {int(getattr(row, "anchor_component_id"))}
        component_ids |= _parse_id_set(getattr(row, "zero_component_ids"))
        missing = [component_id for component_id in component_ids if component_id not in component_sources]
        if missing:
            missing_component_count += 1

        candidate_sources: set[int] = set()
        for component_id in component_ids:
            candidate_sources.update(component_sources.get(component_id, frozenset()))
        candidate_set = frozenset(candidate_sources)
        candidate_source_texts.append(_ids_text(candidate_set))

        target_indexes: set[int] = set()
        for source_fid in candidate_set:
            target_indexes.update(fid_to_target_ids.get(int(source_fid), set()))

        if candidate_set in target_set_to_id:
            labels.append(1)
            label_sources.append("source_target_complete_positive")
            weights.append(float(positive_weight))
            best_target_texts.append(_ids_text(candidate_set))
            best_overlap_counts.append(len(candidate_set))
            best_overlap_ratios.append(1.0)
            exact_positive_count += 1
            continue

        best_target: frozenset[int] = frozenset()
        best_overlap = 0
        for target_index in target_indexes:
            target_set = target_sets[int(target_index)]
            overlap = len(candidate_set & target_set)
            if overlap > best_overlap:
                best_overlap = int(overlap)
                best_target = target_set

        labels.append(0)
        weights.append(float(negative_weight))
        best_target_texts.append(_ids_text(best_target))
        best_overlap_counts.append(int(best_overlap))
        best_overlap_ratios.append(_safe_ratio(float(best_overlap), float(max(len(candidate_set), len(best_target)))))

        if not candidate_set:
            label_sources.append("source_target_empty_candidate_negative")
        elif not best_target:
            label_sources.append("source_target_unmatched_negative")
        elif set(candidate_set) < set(best_target):
            label_sources.append("source_target_partial_negative")
        elif set(best_target) < set(candidate_set):
            label_sources.append("source_target_overmerge_negative")
        else:
            label_sources.append("source_target_mismatch_negative")

    out = dataset.copy()
    out[TARGET_COL] = np.asarray(labels, dtype="int32")
    out["label_source"] = label_sources
    out["sample_weight"] = np.asarray(weights, dtype="float64")
    out["candidate_source_fids"] = candidate_source_texts
    out["source_target_best_fids"] = best_target_texts
    out["source_target_best_overlap_count"] = np.asarray(best_overlap_counts, dtype="int32")
    out["source_target_best_overlap_ratio"] = np.asarray(best_overlap_ratios, dtype="float64")

    _log(
        "[INFO] Applied source-target labels: "
        f"candidate_rows={len(out):,}; matched_positive={exact_positive_count:,}; "
        f"missing_component_rows={missing_component_count:,}"
    )
    if exact_positive_count == 0:
        raise RuntimeError("Source-target labels did not match any anchor group candidates.")
    return out


def _add_pool_completion_features(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {
        "anchor_component_id",
        "zero_component_ids",
        "group_zero_component_count",
        "pair_shared_edge_sum",
        "pair_repair_proba_mean",
        "pair_repair_proba_max",
    }
    if not required.issubset(dataset.columns):
        return dataset
    out = dataset.copy()
    feature_rows: list[dict[str, float]] = []
    for _, group in out.groupby("anchor_component_id", sort=False):
        pool_ids: set[int] = set()
        for value in group["zero_component_ids"]:
            pool_ids |= _parse_id_set(value)
        singleton_stats: dict[int, dict[str, float]] = {}
        singleton = group[group["group_zero_component_count"].astype(int).eq(1)]
        for _, row in singleton.iterrows():
            ids = _parse_id_set(row["zero_component_ids"])
            if len(ids) != 1:
                continue
            zero_id = next(iter(ids))
            singleton_stats[zero_id] = {
                "shared": float(row.get("pair_shared_edge_sum", 0.0) or 0.0),
                "proba": float(row.get("pair_repair_proba_mean", 0.0) or 0.0),
                "reg": float(row.get("after_regularity_score", 0.0) or 0.0),
                "hull": float(row.get("after_hull_gap_ratio", 9.0) or 9.0),
                "uprn_skeleton_component": float(
                    row.get("group_zero_uprn_land_building_component_count", 0.0) or 0.0
                ),
                "uprn_skeleton_small_component": float(
                    row.get("group_small_zero_uprn_land_building_component_count", 0.0) or 0.0
                ),
                "uprn_skeleton_medium_component": float(
                    row.get("group_medium_zero_uprn_land_building_component_count", 0.0) or 0.0
                ),
                "uprn_skeleton_area": float(row.get("group_zero_uprn_land_building_area_sum", 0.0) or 0.0),
                "enclosed_skeleton_component": float(
                    row.get("group_enclosed_zero_uprn_land_building_component_count", 0.0) or 0.0
                ),
                "enclosed_skeleton_area": float(
                    row.get("group_enclosed_zero_uprn_land_building_area_sum", 0.0) or 0.0
                ),
            }
        for idx, row in group.iterrows():
            selected_ids = _parse_id_set(row["zero_component_ids"])
            omitted_ids = pool_ids - selected_ids
            omitted_stats = [singleton_stats.get(int(zero_id), {}) for zero_id in omitted_ids]
            omitted_shared = [float(stats.get("shared", 0.0)) for stats in omitted_stats]
            omitted_proba = [float(stats.get("proba", 0.0)) for stats in omitted_stats]
            pool_skeleton_count = float(
                sum(float(stats.get("uprn_skeleton_component", 0.0)) for stats in singleton_stats.values())
            )
            pool_skeleton_area = float(
                sum(float(stats.get("uprn_skeleton_area", 0.0)) for stats in singleton_stats.values())
            )
            omitted_skeleton_count = float(
                sum(float(stats.get("uprn_skeleton_component", 0.0)) for stats in omitted_stats)
            )
            omitted_skeleton_area = float(
                sum(float(stats.get("uprn_skeleton_area", 0.0)) for stats in omitted_stats)
            )
            omitted_skeleton_small_count = float(
                sum(float(stats.get("uprn_skeleton_small_component", 0.0)) for stats in omitted_stats)
            )
            omitted_skeleton_medium_count = float(
                sum(float(stats.get("uprn_skeleton_medium_component", 0.0)) for stats in omitted_stats)
            )
            selected_skeleton_count = float(row.get("group_zero_uprn_land_building_component_count", 0.0) or 0.0)
            selected_skeleton_area = float(row.get("group_zero_uprn_land_building_area_sum", 0.0) or 0.0)
            pool_enclosed_count = float(
                sum(float(stats.get("enclosed_skeleton_component", 0.0)) for stats in singleton_stats.values())
            )
            pool_enclosed_area = float(
                sum(float(stats.get("enclosed_skeleton_area", 0.0)) for stats in singleton_stats.values())
            )
            omitted_enclosed_count = float(
                sum(float(stats.get("enclosed_skeleton_component", 0.0)) for stats in omitted_stats)
            )
            omitted_enclosed_area = float(
                sum(float(stats.get("enclosed_skeleton_area", 0.0)) for stats in omitted_stats)
            )
            selected_enclosed_count = float(
                row.get("group_enclosed_zero_uprn_land_building_component_count", 0.0) or 0.0
            )
            selected_enclosed_area = float(
                row.get("group_enclosed_zero_uprn_land_building_area_sum", 0.0) or 0.0
            )
            omitted_strong = [
                stats
                for stats in omitted_stats
                if float(stats.get("shared", 0.0)) >= 1.0
                and float(stats.get("proba", 0.0)) >= 0.25
                and float(stats.get("reg", 0.0)) >= 0.80
                and float(stats.get("hull", 9.0)) <= 0.20
            ]
            pool_count = len(pool_ids)
            feature_rows.append(
                {
                    "_idx": idx,
                    "anchor_pool_zero_count": float(pool_count),
                    "group_zero_fraction_of_pool": _safe_ratio(len(selected_ids), pool_count),
                    "omitted_zero_component_count": float(len(omitted_ids)),
                    "omitted_pair_shared_edge_sum": float(sum(omitted_shared)),
                    "omitted_pair_shared_edge_max": float(max(omitted_shared) if omitted_shared else 0.0),
                    "omitted_pair_repair_proba_max": float(max(omitted_proba) if omitted_proba else 0.0),
                    "omitted_pair_repair_proba_mean": float(np.mean(omitted_proba) if omitted_proba else 0.0),
                    "omitted_strong_zero_count": float(len(omitted_strong)),
                    "anchor_pool_zero_uprn_land_building_component_count": pool_skeleton_count,
                    "anchor_pool_zero_uprn_land_building_area_sum": pool_skeleton_area,
                    "group_zero_uprn_land_building_fraction_of_pool": _safe_ratio(
                        selected_skeleton_count,
                        pool_skeleton_count,
                    ),
                    "group_zero_uprn_land_building_area_fraction_of_pool": _safe_ratio(
                        selected_skeleton_area,
                        pool_skeleton_area,
                    ),
                    "omitted_zero_uprn_land_building_component_count": omitted_skeleton_count,
                    "omitted_zero_uprn_land_building_area_sum": omitted_skeleton_area,
                    "omitted_small_zero_uprn_land_building_component_count": omitted_skeleton_small_count,
                    "omitted_medium_zero_uprn_land_building_component_count": omitted_skeleton_medium_count,
                    "uprn_skeleton_complete_pool": float(
                        pool_skeleton_count > 0 and omitted_skeleton_count <= 0.0
                    ),
                    "anchor_pool_enclosed_zero_uprn_land_building_component_count": pool_enclosed_count,
                    "anchor_pool_enclosed_zero_uprn_land_building_area_sum": pool_enclosed_area,
                    "group_enclosed_zero_uprn_land_building_fraction_of_pool": _safe_ratio(
                        selected_enclosed_count,
                        pool_enclosed_count,
                    ),
                    "group_enclosed_zero_uprn_land_building_area_fraction_of_pool": _safe_ratio(
                        selected_enclosed_area,
                        pool_enclosed_area,
                    ),
                    "omitted_enclosed_zero_uprn_land_building_component_count": omitted_enclosed_count,
                    "omitted_enclosed_zero_uprn_land_building_area_sum": omitted_enclosed_area,
                    "enclosed_zero_uprn_complete_pool": float(
                        pool_enclosed_count > 0 and omitted_enclosed_count <= 0.0
                    ),
                }
            )
    if not feature_rows:
        return out
    features = pd.DataFrame.from_records(feature_rows).set_index("_idx")
    for column in features.columns:
        out[column] = features[column]
    return out


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLS | MODEL_OUTPUT_COLS | {TARGET_COL, "sample_weight"}
    feature_cols = [
        c
        for c in dataset.columns
        if c not in excluded and not any(marker in c for marker in LABEL_DERIVED_FEATURE_MARKERS)
    ]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    numeric_cols = [
        c for c in feature_cols if c not in categorical_cols and pd.api.types.is_numeric_dtype(dataset[c])
    ]
    return numeric_cols + categorical_cols, numeric_cols, categorical_cols


def _thresholds_at_precision(y_true: np.ndarray, proba: np.ndarray, targets: list[float]) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    out: dict[str, Any] = {}
    for target in targets:
        eligible = np.where(precision[:-1] >= float(target))[0]
        if len(eligible) == 0:
            out[str(target)] = None
            continue
        idx = int(eligible[np.argmax(recall[:-1][eligible])])
        out[str(target)] = {
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
        }
    return out


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= float(threshold)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    out: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "precision_positive": float(precision[1]),
        "recall_positive": float(recall[1]),
        "f1_positive": float(f1[1]),
        "support_positive": int(support[1]),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "f1_negative": float(f1[0]),
        "support_negative": int(support[0]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).astype(int).tolist(),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["average_precision"] = float(average_precision_score(y_true, proba))
        out["thresholds_at_precision"] = _thresholds_at_precision(y_true, proba, [0.9, 0.95, 0.97])
    return out


def _sample_fit_rows(dataset: pd.DataFrame, *, max_negative_rows: int, random_state: int) -> pd.DataFrame:
    if int(max_negative_rows) <= 0:
        return dataset.copy()
    positive = dataset[dataset[TARGET_COL].astype(int).eq(1)].copy()
    negative = dataset[dataset[TARGET_COL].astype(int).eq(0)].copy()
    if len(negative) > int(max_negative_rows):
        negative = negative.sample(n=int(max_negative_rows), random_state=int(random_state))
    return pd.concat([positive, negative], ignore_index=True).sample(frac=1.0, random_state=int(random_state))


def _read_pair_candidates(path: Path) -> pd.DataFrame:
    usecols = [
        "anchor_component_id",
        "zero_component_id",
        "anchor_uprn_count",
        "zero_uprn_count",
        "shared_edge_sum",
        "shared_edge_max",
        "source_edge_count",
        "edge_proba_max",
        "edge_proba_mean",
        "role_pair",
        "anchor_area",
        "zero_area",
        "after_area",
        "anchor_source_count",
        "zero_source_count",
        "anchor_reference_fids",
        "zero_reference_fids",
        "same_reference_eval",
        "neighbor_anchor_count",
        "zero_anchor_rank_by_shared",
        "second_best_shared_edge_sum",
        "shared_edge_margin_ratio",
        "candidate_mrr_ratio",
        "candidate_hull_gap_ratio",
        "candidate_regularity_score",
        "tier_unique_anchor",
        "tier_clear_anchor",
        "tier_shape_supported",
        "native_probe_score",
        "anchor_repair_proba",
    ]
    pair = pd.read_csv(path, usecols=lambda col: col in usecols)
    pair["anchor_component_id"] = pair["anchor_component_id"].astype(int)
    pair["zero_component_id"] = pair["zero_component_id"].astype(int)
    if "anchor_uprn_count" not in pair.columns:
        pair["anchor_uprn_count"] = 1
    if "zero_uprn_count" not in pair.columns:
        pair["zero_uprn_count"] = 0
    return pair


def build_anchor_need_candidates(
    *,
    pair_candidate_csv: Path,
    max_anchor_area: float,
    max_zero_area: float,
    max_after_area: float,
    max_zero_source_count: int,
    allowed_anchor_ids: set[int] | None = None,
    manual_positive_groups: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    manual_positive_groups = manual_positive_groups or {}
    pair = _read_pair_candidates(pair_candidate_csv)
    if allowed_anchor_ids is not None:
        pair = pair[pair["anchor_component_id"].isin({int(v) for v in allowed_anchor_ids})].copy()
    pair = pair[
        pair["anchor_area"].astype(float).le(float(max_anchor_area))
        & pair["zero_area"].astype(float).le(float(max_zero_area))
        & pair["after_area"].astype(float).le(float(max_after_area))
        & pair["zero_source_count"].fillna(999).astype(int).le(int(max_zero_source_count))
    ].copy()
    records: list[dict[str, Any]] = []
    for anchor_id, group in pair.groupby("anchor_component_id", sort=True):
        anchor_id = int(anchor_id)
        group = group.sort_values(
            ["anchor_repair_proba", "shared_edge_sum", "edge_proba_max"],
            ascending=[False, False, False],
        )
        if group.empty:
            continue
        top = group.iloc[0]
        role_text = group.get("role_pair", pd.Series("", index=group.index)).fillna("").astype(str).str.lower()
        zero_land_building = (
            group.get("zero_uprn_count", pd.Series(0, index=group.index)).fillna(0).astype(int).eq(0)
            & role_text.str.contains("land|building", regex=True, na=False)
        )
        zero_land_building_area = group.loc[zero_land_building, "zero_area"].astype(float)
        zero_land_building_proba = group.loc[zero_land_building, "anchor_repair_proba"].astype(float)
        zero_land_building_shared = group.loc[zero_land_building, "shared_edge_sum"].astype(float)
        repair_proba = group["anchor_repair_proba"].astype(float).to_numpy()
        shared = group["shared_edge_sum"].astype(float).to_numpy()
        regularity = group["candidate_regularity_score"].astype(float).to_numpy()
        hull = group["candidate_hull_gap_ratio"].astype(float).to_numpy()
        mrr = group["candidate_mrr_ratio"].astype(float).to_numpy()
        ranks = group["zero_anchor_rank_by_shared"].astype(float).to_numpy()
        neighbor_counts = group["neighbor_anchor_count"].astype(float).to_numpy()
        is_manual = anchor_id in manual_positive_groups
        label = int(group["same_reference_eval"].fillna(0).astype(int).eq(1).any() or is_manual)
        records.append(
            {
                "anchor_component_id": anchor_id,
                "label": label,
                "sample_weight": 30.0 if is_manual else (4.0 if label else 1.0),
                "candidate_count": int(len(group)),
                "anchor_area": float(top["anchor_area"]),
                "anchor_source_count": float(top["anchor_source_count"]),
                "anchor_uprn_count": float(top.get("anchor_uprn_count", 1)),
                "zero_uprn_count_max": float(group.get("zero_uprn_count", pd.Series(0, index=group.index)).max()),
                "zero_uprn_land_building_candidate_count": float(zero_land_building.sum()),
                "zero_uprn_land_building_candidate_fraction": _safe_ratio(float(zero_land_building.sum()), len(group)),
                "zero_uprn_land_building_area_sum": float(zero_land_building_area.sum()),
                "zero_uprn_land_building_area_max": float(
                    zero_land_building_area.max() if len(zero_land_building_area) else 0.0
                ),
                "small_zero_uprn_land_building_candidate_count": float(
                    zero_land_building_area.le(250.0).sum()
                ),
                "medium_zero_uprn_land_building_candidate_count": float(
                    zero_land_building_area.le(1000.0).sum()
                ),
                "zero_uprn_land_building_proba_max": float(
                    zero_land_building_proba.max() if len(zero_land_building_proba) else 0.0
                ),
                "zero_uprn_land_building_proba_mean": float(
                    zero_land_building_proba.mean() if len(zero_land_building_proba) else 0.0
                ),
                "zero_uprn_land_building_shared_sum": float(zero_land_building_shared.sum()),
                "zero_uprn_land_building_shared_max": float(
                    zero_land_building_shared.max() if len(zero_land_building_shared) else 0.0
                ),
                "proba_max": float(repair_proba.max()),
                "proba_mean": float(repair_proba.mean()),
                "proba_top2_margin": float(repair_proba[0] - repair_proba[1]) if len(repair_proba) > 1 else 1.0,
                "proba_ge_08": int(np.sum(repair_proba >= 0.8)),
                "proba_ge_09": int(np.sum(repair_proba >= 0.9)),
                "proba_ge_095": int(np.sum(repair_proba >= 0.95)),
                "shared_max": float(shared.max()),
                "shared_sum_top3": float(shared[:3].sum()),
                "shared_mean": float(shared.mean()),
                "reg_at_top": float(top["candidate_regularity_score"]),
                "hull_at_top": float(top["candidate_hull_gap_ratio"]),
                "mrr_at_top": float(top["candidate_mrr_ratio"]),
                "reg_max": float(regularity.max()),
                "hull_min": float(hull.min()),
                "mrr_max": float(mrr.max()),
                "rank_min": float(ranks.min()),
                "rank1_count": int(np.sum(ranks <= 1.0)),
                "neighbor_anchor_count_min": float(neighbor_counts.min()),
                "neighbor_anchor_count_max": float(neighbor_counts.max()),
                "tier_unique_sum": int(group["tier_unique_anchor"].fillna(0).astype(int).sum()),
                "tier_clear_sum": int(group["tier_clear_anchor"].fillna(0).astype(int).sum()),
                "tier_shape_sum": int(group["tier_shape_supported"].fillna(0).astype(int).sum()),
                "zero_area_top": float(top["zero_area"]),
                "after_area_top": float(top["after_area"]),
            }
        )
    if not records:
        raise RuntimeError("No anchor need candidates were generated.")
    return pd.DataFrame.from_records(records)


def _mrr_orientation_deg(geom: Any) -> float:
    mrr = shapely.minimum_rotated_rectangle(geom)
    if not hasattr(mrr, "exterior"):
        return 0.0
    coords = list(mrr.exterior.coords)
    best_length = 0.0
    best_angle = 0.0
    for start, end in zip(coords, coords[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if length > best_length:
            best_length = length
            best_angle = math.degrees(math.atan2(dy, dx)) % 180.0
    return float(best_angle)


def _angle_delta_deg(angle: np.ndarray | float, reference_angle: float) -> np.ndarray | float:
    return np.abs(((angle - float(reference_angle) + 45.0) % 90.0) - 45.0)


def _component_geometries(
    input_gpkg: Path,
    *,
    enclosure_component_ids: set[int] | None = None,
    enclosure_level: str = "component",
    source_enclosed_boundary_ratio: float = 0.90,
    source_enclosed_min_neighbors: int = 2,
    source_enclosed_min_shared_edge: float = 0.05,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]], Any, dict[str, Any]]:
    predicted = pyogrio.read_dataframe(input_gpkg, layer="predicted_parcels_with_uprn")
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted["pred_component_id"] = predicted["pred_component_id"].astype(int)
    geom_by_component = predicted.set_index("pred_component_id").geometry.to_dict()
    attrs = predicted.set_index("pred_component_id").drop(columns="geometry").to_dict("index")

    sources = pyogrio.read_dataframe(input_gpkg, layer="prediction_source_polygons")
    sources = sources[sources.geometry.notna() & ~sources.geometry.is_empty].copy()
    sources["pred_component_id"] = sources["pred_component_id"].astype(int)
    source_uprn = pd.to_numeric(sources.get("source_uprn_count", 0), errors="coerce").fillna(0).astype(int)
    theme_text = _source_theme_text(sources)
    building_source = theme_text.str.contains("building", regex=False, na=False)
    land_source = theme_text.str.contains("land", regex=False, na=False)
    land_building_source = building_source | land_source
    enclosure_candidate = source_uprn.eq(0) & land_building_source
    if enclosure_component_ids is not None:
        enclosure_candidate &= sources["pred_component_id"].isin({int(v) for v in enclosure_component_ids})
    if str(enclosure_level).lower() == "source":
        _log(f"[INFO] Computing source-level enclosed zero-UPRN signals for {int(enclosure_candidate.sum()):,} sources")
        enclosed_source, enclosed_boundary_ratio, enclosed_neighbor_count = _source_enclosure_stats(
            sources,
            enclosure_candidate,
            min_boundary_ratio=float(source_enclosed_boundary_ratio),
            min_neighbors=int(source_enclosed_min_neighbors),
            min_shared_edge=float(source_enclosed_min_shared_edge),
        )
    else:
        enclosed_source = pd.Series(False, index=sources.index)
        enclosed_boundary_ratio = pd.Series(0.0, index=sources.index)
        enclosed_neighbor_count = pd.Series(0, index=sources.index)
    source_area = sources.geometry.area.astype(float)
    enclosed_zero_land_building = source_uprn.eq(0) & land_building_source & enclosed_source
    source_stats = sources.assign(
        _uprn_source=source_uprn.gt(0).astype(int),
        _zero_uprn_source=source_uprn.eq(0).astype(int),
        _building_source=building_source.astype(int),
        _land_source=land_source.astype(int),
        _land_building_source=land_building_source.astype(int),
        _zero_uprn_land_building_source=(source_uprn.eq(0) & land_building_source).astype(int),
        _uprn_land_building_source=(source_uprn.gt(0) & land_building_source).astype(int),
        _anchor_building_source=(source_uprn.gt(0) & building_source).astype(int),
        _enclosed_zero_uprn_land_building_source=enclosed_zero_land_building.astype(int),
        _enclosed_zero_uprn_land_building_area=np.where(enclosed_zero_land_building, source_area, 0.0),
        _enclosed_zero_uprn_land_building_boundary_ratio=np.where(
            enclosed_zero_land_building,
            enclosed_boundary_ratio.astype(float),
            0.0,
        ),
        _enclosed_zero_uprn_land_building_neighbor_count=np.where(
            enclosed_zero_land_building,
            enclosed_neighbor_count.astype(int),
            0,
        ),
    ).groupby("pred_component_id").agg(
        component_source_count_from_sources=("pred_component_id", "size"),
        component_uprn_source_count=("_uprn_source", "sum"),
        component_zero_uprn_source_count=("_zero_uprn_source", "sum"),
        component_building_source_count=("_building_source", "sum"),
        component_land_source_count=("_land_source", "sum"),
        component_land_building_source_count=("_land_building_source", "sum"),
        component_zero_uprn_land_building_source_count=("_zero_uprn_land_building_source", "sum"),
        component_uprn_land_building_source_count=("_uprn_land_building_source", "sum"),
        component_anchor_building_source_count=("_anchor_building_source", "sum"),
        component_enclosed_zero_uprn_land_building_source_count=(
            "_enclosed_zero_uprn_land_building_source",
            "sum",
        ),
        component_enclosed_zero_uprn_land_building_area_sum=(
            "_enclosed_zero_uprn_land_building_area",
            "sum",
        ),
        component_enclosed_zero_uprn_land_building_boundary_ratio_max=(
            "_enclosed_zero_uprn_land_building_boundary_ratio",
            "max",
        ),
        component_enclosed_zero_uprn_land_building_neighbor_count_max=(
            "_enclosed_zero_uprn_land_building_neighbor_count",
            "max",
        ),
    )
    for comp_id, row in source_stats.iterrows():
        values: dict[str, Any] = {}
        for key, value in row.to_dict().items():
            if key.endswith("_area_sum") or key.endswith("_ratio_max"):
                values[key] = float(value)
            else:
                values[key] = int(value)
        attrs.setdefault(int(comp_id), {}).update(values)

    component_ids = np.asarray(sorted(geom_by_component), dtype="int64")
    geoms = [geom_by_component[int(comp_id)] for comp_id in component_ids]
    tree = shapely.STRtree(geoms)
    geom_id_to_index = {id(geom): idx for idx, geom in enumerate(geoms)}
    areas = np.asarray([float(shapely.area(geom)) for geom in geoms], dtype="float64")
    orientations = np.asarray([_mrr_orientation_deg(geom) for geom in geoms], dtype="float64")
    regularity = np.asarray(
        [float(attrs.get(int(comp_id), {}).get("pred_regularity_score", 0.0) or 0.0) for comp_id in component_ids],
        dtype="float64",
    )
    hull_gap = np.asarray(
        [float(attrs.get(int(comp_id), {}).get("pred_hull_gap_ratio", 9.0) or 9.0) for comp_id in component_ids],
        dtype="float64",
    )
    mrr_ratio = np.asarray(
        [float(attrs.get(int(comp_id), {}).get("pred_mrr_ratio", 0.0) or 0.0) for comp_id in component_ids],
        dtype="float64",
    )
    uprn_count = np.asarray(
        [int(attrs.get(int(comp_id), {}).get("pred_uprn_count", 0) or 0) for comp_id in component_ids],
        dtype="int64",
    )
    uprn_source_count = np.asarray(
        [int(attrs.get(int(comp_id), {}).get("component_uprn_source_count", 0) or 0) for comp_id in component_ids],
        dtype="int64",
    )
    zero_uprn_source_count = np.asarray(
        [int(attrs.get(int(comp_id), {}).get("component_zero_uprn_source_count", 0) or 0) for comp_id in component_ids],
        dtype="int64",
    )
    land_building_source_count = np.asarray(
        [
            int(attrs.get(int(comp_id), {}).get("component_land_building_source_count", 0) or 0)
            for comp_id in component_ids
        ],
        dtype="int64",
    )
    zero_uprn_land_building_source_count = np.asarray(
        [
            int(attrs.get(int(comp_id), {}).get("component_zero_uprn_land_building_source_count", 0) or 0)
            for comp_id in component_ids
        ],
        dtype="int64",
    )
    anchor_child_pattern = (uprn_source_count > 0) & (zero_uprn_source_count > 0)
    zero_uprn_land_building_pattern = (uprn_count == 0) & (zero_uprn_land_building_source_count > 0)
    if str(enclosure_level).lower() == "component":
        allowed = {int(v) for v in enclosure_component_ids} if enclosure_component_ids is not None else set(component_ids)
        component_candidate_mask = predicted["pred_component_id"].astype(int).isin(allowed)
        component_candidate_mask &= predicted["pred_component_id"].astype(int).map(
            lambda comp_id: int(attrs.get(int(comp_id), {}).get("pred_uprn_count", 0) or 0) == 0
            and int(
                attrs.get(int(comp_id), {}).get(
                    "component_zero_uprn_land_building_source_count",
                    0,
                )
                or 0
            )
            > 0
        )
        _log(
            "[INFO] Computing component-level enclosed zero-UPRN signals for "
            f"{int(component_candidate_mask.sum()):,} components"
        )
        component_enclosed, component_boundary_ratio, component_neighbor_count = _source_enclosure_stats(
            predicted,
            component_candidate_mask,
            min_boundary_ratio=float(source_enclosed_boundary_ratio),
            min_neighbors=int(source_enclosed_min_neighbors),
            min_shared_edge=float(source_enclosed_min_shared_edge),
        )
        enclosed_component_ids = set(
            predicted.loc[component_enclosed, "pred_component_id"].dropna().astype(int).tolist()
        )
        for comp_id in enclosed_component_ids:
            comp_attrs = attrs.setdefault(int(comp_id), {})
            zero_source_count = int(comp_attrs.get("component_zero_uprn_land_building_source_count", 0) or 0)
            comp_attrs["component_enclosed_zero_uprn_land_building_source_count"] = zero_source_count
            comp_attrs["component_enclosed_zero_uprn_land_building_area_sum"] = float(
                shapely.area(geom_by_component[int(comp_id)])
            )
        ratio_by_component = predicted.loc[component_enclosed, ["pred_component_id"]].copy()
        ratio_by_component["boundary_ratio"] = component_boundary_ratio.loc[component_enclosed].astype(float).to_numpy()
        ratio_by_component["neighbor_count"] = component_neighbor_count.loc[component_enclosed].astype(int).to_numpy()
        for row in ratio_by_component.itertuples(index=False):
            comp_attrs = attrs.setdefault(int(row.pred_component_id), {})
            comp_attrs["component_enclosed_zero_uprn_land_building_boundary_ratio_max"] = float(row.boundary_ratio)
            comp_attrs["component_enclosed_zero_uprn_land_building_neighbor_count_max"] = int(row.neighbor_count)

    local_context = {
        "ids": component_ids,
        "geoms": geoms,
        "tree": tree,
        "geom_id_to_index": geom_id_to_index,
        "areas": areas,
        "orientations": orientations,
        "regularity": regularity,
        "hull_gap": hull_gap,
        "mrr_ratio": mrr_ratio,
        "uprn_count": uprn_count,
        "land_building_source_count": land_building_source_count,
        "zero_uprn_land_building_source_count": zero_uprn_land_building_source_count,
        "anchor_child_pattern": anchor_child_pattern,
        "zero_uprn_land_building_pattern": zero_uprn_land_building_pattern,
    }
    return geom_by_component, attrs, predicted.crs, local_context


def _neighbor_cache_for_anchor(
    *,
    anchor_id: int,
    anchor_geom: Any,
    local_context: dict[str, Any],
    radius: float = 65.0,
) -> dict[str, np.ndarray]:
    raw = local_context["tree"].query(shapely.buffer(anchor_geom, float(radius)))
    if len(raw) and not isinstance(raw[0], (int, np.integer)):
        idx = np.asarray([local_context["geom_id_to_index"][id(geom)] for geom in raw], dtype="int64")
    else:
        idx = np.asarray(raw, dtype="int64")
    return {
        "ids": local_context["ids"][idx],
        "areas": local_context["areas"][idx],
        "orientations": local_context["orientations"][idx],
        "regularity": local_context["regularity"][idx],
        "hull_gap": local_context["hull_gap"][idx],
        "mrr_ratio": local_context["mrr_ratio"][idx],
        "uprn_count": local_context["uprn_count"][idx],
        "land_building_source_count": local_context["land_building_source_count"][idx],
        "zero_uprn_land_building_source_count": local_context["zero_uprn_land_building_source_count"][idx],
        "anchor_child_pattern": local_context["anchor_child_pattern"][idx],
        "zero_uprn_land_building_pattern": local_context["zero_uprn_land_building_pattern"][idx],
    }


def _local_pattern_features(
    *,
    cache: dict[str, np.ndarray],
    reference_area: float,
    reference_orientation: float,
    exclude_component_ids: set[int],
) -> dict[str, float]:
    ids = cache["ids"]
    if len(ids) == 0:
        return {
            "local_neighbor_count_65m": 0.0,
            "local_regular_uprn_count_65m": 0.0,
            "local_same_orientation_count_65m": 0.0,
            "local_same_pattern_count_65m": 0.0,
            "local_same_pattern_ratio_65m": 0.0,
            "local_anchor_child_pattern_count_65m": 0.0,
            "local_anchor_child_pattern_ratio_65m": 0.0,
            "local_same_pattern_anchor_child_count_65m": 0.0,
            "local_same_pattern_anchor_child_ratio_65m": 0.0,
            "local_same_pattern_area_median": 0.0,
            "local_same_pattern_area_iqr": 0.0,
            "area_to_local_same_pattern_median": 0.0,
            "local_same_pattern_orientation_delta_median": 45.0,
            "local_context_score": 0.0,
            "local_zero_uprn_land_building_count_65m": 0.0,
            "local_small_zero_uprn_land_building_count_65m": 0.0,
            "local_medium_zero_uprn_land_building_count_65m": 0.0,
            "local_zero_uprn_land_building_area_sum_65m": 0.0,
            "local_zero_uprn_land_building_area_median_65m": 0.0,
            "local_zero_uprn_land_building_ratio_65m": 0.0,
        }
    keep = ~np.isin(ids, np.asarray(list(exclude_component_ids), dtype="int64"))
    if not bool(np.any(keep)):
        keep = np.zeros_like(ids, dtype=bool)
    areas = cache["areas"][keep]
    orientation_delta = _angle_delta_deg(cache["orientations"][keep], float(reference_orientation))
    regular_uprn = (
        (cache["uprn_count"][keep] > 0)
        & (cache["regularity"][keep] >= 0.90)
        & (cache["hull_gap"][keep] <= 0.12)
        & (cache["mrr_ratio"][keep] >= 0.80)
    )
    same_orientation = orientation_delta <= 15.0
    area_like = (areas >= float(reference_area) * 0.35) & (areas <= float(reference_area) * 2.50)
    same_pattern = regular_uprn & same_orientation & area_like
    anchor_child = cache["anchor_child_pattern"][keep]
    zero_uprn_land_building = cache["zero_uprn_land_building_pattern"][keep]
    same_pattern_anchor_child = same_pattern & anchor_child
    local_count = int(len(areas))
    same_pattern_areas = areas[same_pattern]
    same_pattern_deltas = orientation_delta[same_pattern]
    zero_uprn_land_building_areas = areas[zero_uprn_land_building]
    median_area = float(np.median(same_pattern_areas)) if len(same_pattern_areas) else 0.0
    q75 = float(np.percentile(same_pattern_areas, 75)) if len(same_pattern_areas) else 0.0
    q25 = float(np.percentile(same_pattern_areas, 25)) if len(same_pattern_areas) else 0.0
    same_pattern_count = int(np.sum(same_pattern))
    anchor_child_count = int(np.sum(anchor_child))
    same_pattern_anchor_child_count = int(np.sum(same_pattern_anchor_child))
    context_score = (
        min(same_pattern_count, 12) / 12.0
        + min(same_pattern_anchor_child_count, 8) / 8.0
        + min(anchor_child_count, 12) / 24.0
    )
    return {
        "local_neighbor_count_65m": float(local_count),
        "local_regular_uprn_count_65m": float(np.sum(regular_uprn)),
        "local_same_orientation_count_65m": float(np.sum(same_orientation & regular_uprn)),
        "local_same_pattern_count_65m": float(same_pattern_count),
        "local_same_pattern_ratio_65m": _safe_ratio(same_pattern_count, local_count),
        "local_anchor_child_pattern_count_65m": float(anchor_child_count),
        "local_anchor_child_pattern_ratio_65m": _safe_ratio(anchor_child_count, local_count),
        "local_same_pattern_anchor_child_count_65m": float(same_pattern_anchor_child_count),
        "local_same_pattern_anchor_child_ratio_65m": _safe_ratio(same_pattern_anchor_child_count, same_pattern_count),
        "local_same_pattern_area_median": median_area,
        "local_same_pattern_area_iqr": float(q75 - q25),
        "area_to_local_same_pattern_median": _safe_ratio(float(reference_area), median_area),
        "local_same_pattern_orientation_delta_median": float(np.median(same_pattern_deltas))
        if len(same_pattern_deltas)
        else 45.0,
        "local_context_score": float(context_score),
        "local_zero_uprn_land_building_count_65m": float(np.sum(zero_uprn_land_building)),
        "local_small_zero_uprn_land_building_count_65m": float(
            np.sum(zero_uprn_land_building & (areas <= 250.0))
        ),
        "local_medium_zero_uprn_land_building_count_65m": float(
            np.sum(zero_uprn_land_building & (areas <= 1000.0))
        ),
        "local_zero_uprn_land_building_area_sum_65m": float(zero_uprn_land_building_areas.sum()),
        "local_zero_uprn_land_building_area_median_65m": float(
            np.median(zero_uprn_land_building_areas) if len(zero_uprn_land_building_areas) else 0.0
        ),
        "local_zero_uprn_land_building_ratio_65m": _safe_ratio(float(np.sum(zero_uprn_land_building)), local_count),
    }


def _pool_for_anchor(
    group: pd.DataFrame,
    *,
    top_zero_neighbors: int,
    manual_positive_groups: dict[int, set[int]],
) -> pd.DataFrame:
    anchor_id = int(group["anchor_component_id"].iloc[0])
    ordered = group.sort_values(
        ["anchor_repair_proba", "shared_edge_sum", "edge_proba_max"],
        ascending=[False, False, False],
    ).drop_duplicates("zero_component_id", keep="first")
    pool_ids = set(ordered.head(int(top_zero_neighbors))["zero_component_id"].astype(int))
    manual = manual_positive_groups.get(anchor_id)
    if manual:
        pool_ids |= {int(v) for v in manual}
    return ordered[ordered["zero_component_id"].astype(int).isin(pool_ids)].copy()


def _candidate_zero_combinations(
    *,
    zero_ids: list[int],
    rows_by_zero: dict[int, Any],
    attrs_by_component: dict[int, dict[str, Any]],
    max_group_size: int,
    candidate_strategy: str,
) -> list[tuple[int, ...]]:
    max_size = min(int(max_group_size), len(zero_ids))
    if max_size <= 0:
        return []
    if str(candidate_strategy).lower() == "full":
        return [
            tuple(int(v) for v in combo)
            for size in range(1, max_size + 1)
            for combo in itertools.combinations(sorted(zero_ids), size)
        ]

    combo_sets: set[tuple[int, ...]] = set()

    def add(values: list[int] | tuple[int, ...]) -> None:
        unique_values = []
        seen: set[int] = set()
        for value in values:
            int_value = int(value)
            if int_value in seen:
                continue
            seen.add(int_value)
            unique_values.append(int_value)
        if not unique_values or len(unique_values) > max_size:
            return
        combo_sets.add(tuple(sorted(unique_values)))

    ordered_ids = [int(v) for v in zero_ids]
    for zero_id in ordered_ids:
        add([zero_id])
    add(ordered_ids)
    add(ordered_ids[:2])
    add(ordered_ids[:3])

    strong_ids = [
        int(zero_id)
        for zero_id in ordered_ids
        if float(getattr(rows_by_zero[int(zero_id)], "anchor_repair_proba", 0.0) or 0.0) >= 0.60
        or (
            float(getattr(rows_by_zero[int(zero_id)], "shared_edge_sum", 0.0) or 0.0) >= 6.0
            and float(getattr(rows_by_zero[int(zero_id)], "zero_anchor_rank_by_shared", 999.0) or 999.0) <= 1.0
        )
    ]
    add(strong_ids)
    add(strong_ids[:2])
    add(strong_ids[:3])

    skeleton_ids = [
        int(zero_id)
        for zero_id in ordered_ids
        if int(
            attrs_by_component.get(int(zero_id), {}).get(
                "component_zero_uprn_land_building_source_count",
                0,
            )
            or 0
        )
        > 0
    ]
    enclosed_ids = [
        int(zero_id)
        for zero_id in ordered_ids
        if int(
            attrs_by_component.get(int(zero_id), {}).get(
                "component_enclosed_zero_uprn_land_building_source_count",
                0,
            )
            or 0
        )
        > 0
    ]
    add(skeleton_ids)
    add(enclosed_ids)
    add(strong_ids + skeleton_ids)
    add(strong_ids + enclosed_ids)
    return sorted(combo_sets, key=lambda combo: (len(combo), combo))


def build_anchor_group_candidates(
    *,
    input_gpkg: Path,
    pair_candidate_csv: Path,
    top_zero_neighbors: int,
    max_group_size: int,
    max_anchor_area: float,
    max_zero_area: float,
    max_after_area: float,
    max_zero_source_count: int,
    allowed_anchor_ids: set[int] | None = None,
    candidate_strategy: str = "full",
    enclosure_level: str = "component",
    source_enclosed_boundary_ratio: float = 0.90,
    source_enclosed_min_neighbors: int = 2,
    source_enclosed_min_shared_edge: float = 0.05,
    manual_positive_groups: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    manual_positive_groups = manual_positive_groups or {}
    _log("[INFO] Reading pair anchor candidates")
    pair = _read_pair_candidates(pair_candidate_csv)
    if allowed_anchor_ids is not None:
        pair = pair[pair["anchor_component_id"].isin({int(v) for v in allowed_anchor_ids})].copy()
    pair = pair[
        pair["anchor_area"].astype(float).le(float(max_anchor_area))
        & pair["zero_area"].astype(float).le(float(max_zero_area))
        & pair["zero_source_count"].fillna(999).astype(int).le(int(max_zero_source_count))
    ].copy()
    if pair.empty:
        raise RuntimeError("No pair candidates remain after group repair filters.")

    _log("[INFO] Reading component geometries")
    enclosure_component_ids = set(pair["anchor_component_id"].astype(int)) | set(pair["zero_component_id"].astype(int))
    geom_by_component, attrs_by_component, _, local_context = _component_geometries(
        input_gpkg,
        enclosure_component_ids=enclosure_component_ids,
        enclosure_level=str(enclosure_level),
        source_enclosed_boundary_ratio=float(source_enclosed_boundary_ratio),
        source_enclosed_min_neighbors=int(source_enclosed_min_neighbors),
        source_enclosed_min_shared_edge=float(source_enclosed_min_shared_edge),
    )
    pair = pair[
        pair["anchor_component_id"].isin(geom_by_component)
        & pair["zero_component_id"].isin(geom_by_component)
    ].copy()

    positive_by_anchor: dict[int, set[int]] = {}
    same_ref = (
        pair["same_reference_eval"].fillna(0).astype(int).eq(1)
        & pair["anchor_reference_fids"].map(_reference_is_single)
    )
    for anchor_id, values in pair.loc[same_ref].groupby("anchor_component_id")["zero_component_id"]:
        positive_by_anchor[int(anchor_id)] = {int(v) for v in values}

    anchor_shape_cache: dict[int, dict[str, float]] = {}
    anchor_neighbor_cache: dict[int, dict[str, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for anchor_id, group in pair.groupby("anchor_component_id", sort=True):
        anchor_id = int(anchor_id)
        if anchor_id not in geom_by_component:
            continue
        pool = _pool_for_anchor(
            group,
            top_zero_neighbors=int(top_zero_neighbors),
            manual_positive_groups=manual_positive_groups,
        )
        if pool.empty:
            continue
        rows_by_zero = {
            int(row.zero_component_id): row
            for row in pool.sort_values("anchor_repair_proba", ascending=False).itertuples(index=False)
        }
        zero_ids = [int(zero_id) for zero_id in rows_by_zero]
        if not zero_ids:
            continue
        anchor_geom = geom_by_component[anchor_id]
        if anchor_id not in anchor_shape_cache:
            anchor_shape_cache[anchor_id] = _shape_metrics(anchor_geom)
        anchor_shape = anchor_shape_cache[anchor_id]
        anchor_orientation = _mrr_orientation_deg(anchor_geom)
        if anchor_id not in anchor_neighbor_cache:
            anchor_neighbor_cache[anchor_id] = _neighbor_cache_for_anchor(
                anchor_id=anchor_id,
                anchor_geom=anchor_geom,
                local_context=local_context,
            )
        neighbor_cache = anchor_neighbor_cache[anchor_id]
        anchor_attrs = attrs_by_component.get(anchor_id, {})
        reference_positive_zero_ids = positive_by_anchor.get(anchor_id, set()) & set(zero_ids)
        candidate_combos = _candidate_zero_combinations(
            zero_ids=zero_ids,
            rows_by_zero=rows_by_zero,
            attrs_by_component=attrs_by_component,
            max_group_size=int(max_group_size),
            candidate_strategy=str(candidate_strategy),
        )

        for combo in candidate_combos:
                size = len(combo)
                combo_set = {int(v) for v in combo}
                pair_rows = [rows_by_zero[int(v)] for v in combo]
                zero_geoms = [geom_by_component[int(v)] for v in combo]
                zero_group_geom = shapely.union_all(zero_geoms)
                after_geom = shapely.union_all([anchor_geom, zero_group_geom])
                after_area = float(shapely.area(after_geom))
                if after_area > float(max_after_area):
                    continue
                zero_shape = _shape_metrics(zero_group_geom)
                after_shape = _shape_metrics(after_geom)
                after_orientation = _mrr_orientation_deg(after_geom)
                local_features = _local_pattern_features(
                    cache=neighbor_cache,
                    reference_area=after_area,
                    reference_orientation=after_orientation,
                    exclude_component_ids={anchor_id} | combo_set,
                )
                anchor_local_features = {
                    f"anchor_{key}": value
                    for key, value in _local_pattern_features(
                        cache=neighbor_cache,
                        reference_area=float(shapely.area(anchor_geom)),
                        reference_orientation=anchor_orientation,
                        exclude_component_ids={anchor_id},
                    ).items()
                }
                zero_area_values = np.asarray([float(getattr(row, "zero_area")) for row in pair_rows], dtype="float64")
                repair_probas = np.asarray([float(getattr(row, "anchor_repair_proba")) for row in pair_rows], dtype="float64")
                shared_edges = np.asarray([float(getattr(row, "shared_edge_sum")) for row in pair_rows], dtype="float64")
                edge_probas = np.asarray([float(getattr(row, "edge_proba_max")) for row in pair_rows], dtype="float64")
                ranks = np.asarray([float(getattr(row, "zero_anchor_rank_by_shared")) for row in pair_rows], dtype="float64")
                neighbor_counts = np.asarray([float(getattr(row, "neighbor_anchor_count")) for row in pair_rows], dtype="float64")
                margins = np.asarray([float(getattr(row, "shared_edge_margin_ratio")) for row in pair_rows], dtype="float64")
                zero_source_count = int(sum(int(getattr(row, "zero_source_count")) for row in pair_rows))
                anchor_area = float(getattr(pair_rows[0], "anchor_area"))
                zero_area_sum = float(zero_area_values.sum())
                zero_attrs = [attrs_by_component.get(int(zero_id), {}) for zero_id in combo]
                zero_multi_source_count = int(sum(int(attrs.get("source_count", 0) or 0) > 1 for attrs in zero_attrs))
                zero_possible_split_count = int(
                    sum(int(attrs.get("possible_split_reference", 0) or 0) > 0 for attrs in zero_attrs)
                )
                zero_regular_count = int(
                    sum(float(attrs.get("pred_regularity_score", 0.0) or 0.0) >= 0.90 for attrs in zero_attrs)
                )
                zero_anchor_child_like_count = int(
                    sum(
                        int(attrs.get("component_uprn_source_count", 0) or 0) > 0
                        and int(attrs.get("component_zero_uprn_source_count", 0) or 0) > 0
                        for attrs in zero_attrs
                    )
                )
                zero_land_building_source_counts = np.asarray(
                    [
                        int(attrs.get("component_land_building_source_count", 0) or 0)
                        for attrs in zero_attrs
                    ],
                    dtype="float64",
                )
                zero_uprn_land_building_source_counts = np.asarray(
                    [
                        int(attrs.get("component_zero_uprn_land_building_source_count", 0) or 0)
                        for attrs in zero_attrs
                    ],
                    dtype="float64",
                )
                zero_anchor_building_source_counts = np.asarray(
                    [
                        int(attrs.get("component_anchor_building_source_count", 0) or 0)
                        for attrs in zero_attrs
                    ],
                    dtype="float64",
                )
                zero_enclosed_uprn_land_building_source_counts = np.asarray(
                    [
                        int(attrs.get("component_enclosed_zero_uprn_land_building_source_count", 0) or 0)
                        for attrs in zero_attrs
                    ],
                    dtype="float64",
                )
                zero_enclosed_uprn_land_building_area_sums = np.asarray(
                    [
                        float(attrs.get("component_enclosed_zero_uprn_land_building_area_sum", 0.0) or 0.0)
                        for attrs in zero_attrs
                    ],
                    dtype="float64",
                )
                zero_uprn_land_building_mask = zero_uprn_land_building_source_counts > 0
                zero_uprn_land_building_areas = zero_area_values[zero_uprn_land_building_mask]
                zero_uprn_land_building_count = int(np.sum(zero_uprn_land_building_mask))
                if str(enclosure_level).lower() == "pair":
                    pair_enclosed_mask = zero_uprn_land_building_mask & (neighbor_counts >= 2.0)
                    zero_enclosed_uprn_land_building_source_counts = np.where(
                        pair_enclosed_mask,
                        zero_uprn_land_building_source_counts,
                        0.0,
                    )
                    zero_enclosed_uprn_land_building_area_sums = np.where(
                        pair_enclosed_mask,
                        zero_area_values,
                        0.0,
                    )
                zero_enclosed_uprn_land_building_mask = zero_enclosed_uprn_land_building_source_counts > 0
                zero_enclosed_uprn_land_building_count = int(np.sum(zero_enclosed_uprn_land_building_mask))
                small_zero_uprn_land_building_count = int(
                    np.sum(zero_uprn_land_building_mask & (zero_area_values <= 250.0))
                )
                medium_zero_uprn_land_building_count = int(
                    np.sum(zero_uprn_land_building_mask & (zero_area_values <= 1000.0))
                )
                label, label_source, sample_weight = _label_group(
                    anchor_id=anchor_id,
                    zero_ids=combo_set,
                    reference_positive_zero_ids=reference_positive_zero_ids,
                    manual_positive_groups=manual_positive_groups,
                )
                role_signature = "|".join(sorted({str(getattr(row, "role_pair") or "") for row in pair_rows}))
                zero_refs = sorted({str(getattr(row, "zero_reference_fids") or "") for row in pair_rows})
                rec: dict[str, Any] = {
                    "anchor_component_id": anchor_id,
                    "zero_component_ids": _ids_text(combo_set),
                    "anchor_reference_fids": str(getattr(pair_rows[0], "anchor_reference_fids") or ""),
                    "zero_reference_fids": "|".join(zero_refs),
                    "reference_positive_zero_component_ids": _ids_text(reference_positive_zero_ids),
                    "label": int(label),
                    "label_source": label_source,
                    "sample_weight": float(sample_weight),
                    "group_zero_component_count": int(size),
                    "group_zero_source_count": int(zero_source_count),
                    "anchor_source_count": int(getattr(pair_rows[0], "anchor_source_count")),
                    "anchor_area": anchor_area,
                    "zero_area_sum": zero_area_sum,
                    "zero_area_max": float(zero_area_values.max()),
                    "zero_area_min": float(zero_area_values.min()),
                    "zero_area_mean": float(zero_area_values.mean()),
                    "after_area": after_area,
                    "zero_area_ratio_to_anchor": _safe_ratio(zero_area_sum, anchor_area),
                    "zero_area_ratio_to_after": _safe_ratio(zero_area_sum, after_area),
                    "after_area_ratio_to_anchor": _safe_ratio(after_area, anchor_area),
                    "pair_repair_proba_min": float(repair_probas.min()),
                    "pair_repair_proba_mean": float(repair_probas.mean()),
                    "pair_repair_proba_max": float(repair_probas.max()),
                    "pair_repair_proba_std": float(repair_probas.std(ddof=0)),
                    "pair_shared_edge_sum": float(shared_edges.sum()),
                    "pair_shared_edge_max": float(shared_edges.max()),
                    "pair_shared_edge_mean": float(shared_edges.mean()),
                    "pair_edge_proba_max": float(edge_probas.max()),
                    "pair_edge_proba_mean": float(edge_probas.mean()),
                    "pair_edge_proba_min": float(edge_probas.min()),
                    "pair_neighbor_anchor_count_max": float(neighbor_counts.max()),
                    "pair_neighbor_anchor_count_mean": float(neighbor_counts.mean()),
                    "pair_zero_anchor_rank_min": float(ranks.min()),
                    "pair_zero_anchor_rank_mean": float(ranks.mean()),
                    "pair_zero_anchor_rank_max": float(ranks.max()),
                    "pair_rank1_count": int(np.sum(ranks <= 1.0)),
                    "pair_shared_margin_min": float(margins.min()),
                    "pair_shared_margin_mean": float(np.minimum(margins, 999.0).mean()),
                    "tier_unique_anchor_count": int(sum(int(getattr(row, "tier_unique_anchor")) for row in pair_rows)),
                    "tier_clear_anchor_count": int(sum(int(getattr(row, "tier_clear_anchor")) for row in pair_rows)),
                    "tier_shape_supported_count": int(sum(int(getattr(row, "tier_shape_supported")) for row in pair_rows)),
                    "role_pair_signature": role_signature,
                    "anchor_uprn_source_count": int(anchor_attrs.get("component_uprn_source_count", 0) or 0),
                    "anchor_zero_uprn_source_count": int(anchor_attrs.get("component_zero_uprn_source_count", 0) or 0),
                    "anchor_building_source_count": int(anchor_attrs.get("component_building_source_count", 0) or 0),
                    "anchor_land_source_count": int(anchor_attrs.get("component_land_source_count", 0) or 0),
                    "anchor_land_building_source_count": int(
                        anchor_attrs.get("component_land_building_source_count", 0) or 0
                    ),
                    "anchor_zero_uprn_land_building_source_count": int(
                        anchor_attrs.get("component_zero_uprn_land_building_source_count", 0) or 0
                    ),
                    "anchor_uprn_land_building_source_count": int(
                        anchor_attrs.get("component_uprn_land_building_source_count", 0) or 0
                    ),
                    "anchor_anchor_building_source_count": int(
                        anchor_attrs.get("component_anchor_building_source_count", 0) or 0
                    ),
                    "anchor_enclosed_zero_uprn_land_building_source_count": int(
                        anchor_attrs.get("component_enclosed_zero_uprn_land_building_source_count", 0) or 0
                    ),
                    "anchor_enclosed_zero_uprn_land_building_area_sum": float(
                        anchor_attrs.get("component_enclosed_zero_uprn_land_building_area_sum", 0.0) or 0.0
                    ),
                    "anchor_has_zero_uprn_child_source": int(
                        int(anchor_attrs.get("component_zero_uprn_source_count", 0) or 0) > 0
                    ),
                    "group_anchor_building_source_count": int(zero_anchor_building_source_counts.sum()),
                    "after_anchor_building_source_count": int(
                        int(anchor_attrs.get("component_anchor_building_source_count", 0) or 0)
                        + int(zero_anchor_building_source_counts.sum())
                    ),
                    "zero_multi_source_component_count": zero_multi_source_count,
                    "zero_possible_split_component_count": zero_possible_split_count,
                    "zero_regular_component_count": zero_regular_count,
                    "zero_anchor_child_like_component_count": zero_anchor_child_like_count,
                    "group_land_building_source_count": int(zero_land_building_source_counts.sum()),
                    "group_zero_uprn_land_building_source_count": int(
                        zero_uprn_land_building_source_counts.sum()
                    ),
                    "group_zero_uprn_land_building_component_count": zero_uprn_land_building_count,
                    "group_small_zero_uprn_land_building_component_count": small_zero_uprn_land_building_count,
                    "group_medium_zero_uprn_land_building_component_count": medium_zero_uprn_land_building_count,
                    "group_zero_uprn_land_building_component_fraction": _safe_ratio(
                        zero_uprn_land_building_count,
                        size,
                    ),
                    "group_zero_uprn_land_building_area_sum": float(
                        zero_uprn_land_building_areas.sum()
                    ),
                    "group_zero_uprn_land_building_area_max": float(
                        zero_uprn_land_building_areas.max() if len(zero_uprn_land_building_areas) else 0.0
                    ),
                    "group_zero_uprn_land_building_area_fraction": _safe_ratio(
                        float(zero_uprn_land_building_areas.sum()),
                        zero_area_sum,
                    ),
                    "group_enclosed_zero_uprn_land_building_source_count": int(
                        zero_enclosed_uprn_land_building_source_counts.sum()
                    ),
                    "group_enclosed_zero_uprn_land_building_component_count": zero_enclosed_uprn_land_building_count,
                    "group_enclosed_zero_uprn_land_building_component_fraction": _safe_ratio(
                        zero_enclosed_uprn_land_building_count,
                        size,
                    ),
                    "group_enclosed_zero_uprn_land_building_area_sum": float(
                        zero_enclosed_uprn_land_building_area_sums.sum()
                    ),
                    "group_enclosed_zero_uprn_land_building_area_fraction": _safe_ratio(
                        float(zero_enclosed_uprn_land_building_area_sums.sum()),
                        zero_area_sum,
                    ),
                    "mrr_gain": float(after_shape["mrr_ratio"] - anchor_shape["mrr_ratio"]),
                    "mrr_gap_reduction": float(anchor_shape["mrr_gap_ratio"] - after_shape["mrr_gap_ratio"]),
                    "hull_gap_reduction": float(anchor_shape["hull_gap_ratio"] - after_shape["hull_gap_ratio"]),
                    "convexity_gain": float(after_shape["convexity"] - anchor_shape["convexity"]),
                    "boundary_complexity_reduction": float(
                        anchor_shape["boundary_complexity"] - after_shape["boundary_complexity"]
                    ),
                    "notch_index_reduction": float(anchor_shape["notch_index"] - after_shape["notch_index"]),
                    "regularity_score_gain": float(after_shape["regularity_score"] - anchor_shape["regularity_score"]),
                    "anchor_uprn_count": int(anchor_attrs.get("pred_uprn_count", 1) or 0),
                    "after_uprn_count": int(anchor_attrs.get("pred_uprn_count", 1) or 0),
                    "after_mrr_orientation_deg": after_orientation,
                    "anchor_mrr_orientation_deg": anchor_orientation,
                }
                rec.update(local_features)
                rec.update(anchor_local_features)
                _update_prefixed(rec, "anchor", anchor_shape)
                _update_prefixed(rec, "zero_group", zero_shape)
                _update_prefixed(rec, "after", after_shape)
                records.append(rec)

    if not records:
        raise RuntimeError("No anchor group candidates were generated.")
    return _add_pool_completion_features(pd.DataFrame.from_records(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UPRN-anchor group repair model.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--pair-candidate-csv", default=DEFAULT_PAIR_CANDIDATE_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bbox", default="", help="Optional smoke-test bbox: minx,miny,maxx,maxy.")
    parser.add_argument("--top-zero-neighbors", type=int, default=6)
    parser.add_argument("--max-group-size", type=int, default=6)
    parser.add_argument("--max-anchor-area", type=float, default=2000.0)
    parser.add_argument("--max-zero-area", type=float, default=1000.0)
    parser.add_argument("--max-after-area", type=float, default=2000.0)
    parser.add_argument("--max-zero-source-count", type=int, default=8)
    parser.add_argument("--candidate-strategy", choices=["full", "light"], default="full")
    parser.add_argument("--enclosure-level", choices=["component", "source", "pair", "none"], default="component")
    parser.add_argument("--source-enclosed-boundary-ratio", type=float, default=0.90)
    parser.add_argument("--source-enclosed-min-neighbors", type=int, default=2)
    parser.add_argument("--source-enclosed-min-shared-edge", type=float, default=0.05)
    parser.add_argument("--manual-positive-groups", default="")
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--anchor-need-input-csv", default="")
    parser.add_argument(
        "--label-mode",
        choices=["weak_reference", "final_selection", "source_target"],
        default="weak_reference",
        help=(
            "Use weak/reference labels, distill labels from an already selected anchor-group layer, "
            "or train directly against target source_wfs_fids groups."
        ),
    )
    parser.add_argument("--selection-gpkg", default="")
    parser.add_argument("--selection-layer", default="anchor_group_repair_selected")
    parser.add_argument("--final-selection-positive-weight", type=float, default=16.0)
    parser.add_argument("--final-selection-negative-weight", type=float, default=1.0)
    parser.add_argument("--source-target-gpkg", default="")
    parser.add_argument("--source-target-layer", default="wfs_raw_merged_council_train_merged_only")
    parser.add_argument("--source-target-positive-weight", type=float, default=1.0)
    parser.add_argument("--source-target-negative-weight", type=float, default=1.0)
    parser.add_argument("--skip-candidate-output", action="store_true")
    parser.add_argument("--max-negative-train-rows", type=int, default=120000)
    parser.add_argument("--max-negative-gate-train-rows", type=int, default=80000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_gpkg = Path(args.input_gpkg)
    pair_candidate_csv = Path(args.pair_candidate_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_positive_groups = _parse_groups(args.manual_positive_groups)
    bbox = _parse_bbox(args.bbox)
    allowed_anchor_ids = _component_ids_in_bbox(input_gpkg, bbox)
    if allowed_anchor_ids is not None:
        _log(f"[INFO] Bbox anchor components={len(allowed_anchor_ids):,}; bbox={bbox}")

    if str(args.candidate_input_csv).strip():
        dataset = pd.read_csv(args.candidate_input_csv)
        if allowed_anchor_ids is not None:
            before = len(dataset)
            dataset = dataset[dataset["anchor_component_id"].astype(int).isin(allowed_anchor_ids)].copy()
            _log(f"[INFO] Bbox-filtered reused candidate dataset: {before:,} -> {len(dataset):,}")
        _log(f"[INFO] Reusing candidate dataset: {args.candidate_input_csv}")
    else:
        dataset = build_anchor_group_candidates(
            input_gpkg=input_gpkg,
            pair_candidate_csv=pair_candidate_csv,
            top_zero_neighbors=int(args.top_zero_neighbors),
            max_group_size=int(args.max_group_size),
            max_anchor_area=float(args.max_anchor_area),
            max_zero_area=float(args.max_zero_area),
            max_after_area=float(args.max_after_area),
            max_zero_source_count=int(args.max_zero_source_count),
            allowed_anchor_ids=allowed_anchor_ids,
            candidate_strategy=str(args.candidate_strategy),
            enclosure_level=str(args.enclosure_level),
            source_enclosed_boundary_ratio=float(args.source_enclosed_boundary_ratio),
            source_enclosed_min_neighbors=int(args.source_enclosed_min_neighbors),
            source_enclosed_min_shared_edge=float(args.source_enclosed_min_shared_edge),
            manual_positive_groups=manual_positive_groups,
        )
    dataset = _add_pool_completion_features(dataset)
    label_mode = str(args.label_mode)
    if label_mode == "final_selection":
        if not str(args.selection_gpkg).strip():
            raise RuntimeError("--selection-gpkg is required when --label-mode final_selection")
        dataset = _apply_final_selection_labels(
            dataset,
            selection_gpkg=Path(args.selection_gpkg),
            selection_layer=str(args.selection_layer),
            positive_weight=float(args.final_selection_positive_weight),
            negative_weight=float(args.final_selection_negative_weight),
        )
    elif label_mode == "source_target":
        if not str(args.source_target_gpkg).strip():
            raise RuntimeError("--source-target-gpkg is required when --label-mode source_target")
        dataset = _apply_source_target_labels(
            dataset,
            input_gpkg=input_gpkg,
            target_gpkg=Path(args.source_target_gpkg),
            target_layer=str(args.source_target_layer),
            positive_weight=float(args.source_target_positive_weight),
            negative_weight=float(args.source_target_negative_weight),
        )
    else:
        dataset = _apply_spatial_pattern_weak_labels(dataset)
        dataset = _apply_uprn_skeleton_weak_labels(dataset)
        dataset = _apply_enclosed_zero_uprn_weak_labels(dataset)
        dataset = _apply_manual_group_labels(dataset, manual_positive_groups)
        dataset = _apply_anchor_building_guard_labels(dataset)
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
    if bool(args.skip_candidate_output):
        _log("[INFO] Skipping candidate dataset write")
    else:
        dataset.to_csv(output_dir / CANDIDATES_FILE_NAME, index=False)
        _log(f"[INFO] Wrote candidate dataset: {output_dir / CANDIDATES_FILE_NAME}")
    fit_dataset = _sample_fit_rows(
        dataset,
        max_negative_rows=int(args.max_negative_train_rows),
        random_state=int(args.random_state),
    )
    _log(
        "[INFO] Fit rows="
        f"{len(fit_dataset):,}; fit_label_counts={fit_dataset[TARGET_COL].value_counts().to_dict()}"
    )
    feature_cols, numeric_cols, categorical_cols = _feature_columns(dataset)
    _log(f"[INFO] Candidates={len(dataset):,}; label_counts={dataset[TARGET_COL].value_counts().to_dict()}")
    _log(f"[INFO] Label sources={dataset['label_source'].value_counts().to_dict()}")
    _log(f"[INFO] Features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=4)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=19,
        l2_regularization=0.08,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    groups = fit_dataset["anchor_component_id"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(fit_dataset, fit_dataset[TARGET_COL], groups=groups))
    train = fit_dataset.iloc[train_idx].copy()
    test = fit_dataset.iloc[test_idx].copy()

    _log("[INFO] Training anchor group repair model")
    pipeline.fit(
        train[feature_cols],
        train[TARGET_COL],
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    all_proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]
    dataset["anchor_group_repair_proba"] = all_proba
    dataset["anchor_group_repair_pred_at_threshold"] = dataset["anchor_group_repair_proba"].ge(
        float(args.threshold)
    ).astype(int)

    test_metrics = _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(args.threshold))
    all_metrics = _metrics(dataset[TARGET_COL].to_numpy(dtype=int), all_proba, float(args.threshold))

    _log("[INFO] Refitting final model on all candidates")
    final_pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    final_pipeline.fit(
        fit_dataset[feature_cols],
        fit_dataset[TARGET_COL],
        model__sample_weight=fit_dataset["sample_weight"].astype(float).to_numpy(),
    )

    anchor_need = pd.DataFrame()
    anchor_feature_cols: list[str] = []
    gate_test_metrics: dict[str, Any] | None = None
    gate_all_metrics: dict[str, Any] | None = None
    final_anchor_gate = None
    train_anchor_gate = label_mode == "weak_reference"
    if train_anchor_gate and str(args.anchor_need_input_csv).strip():
        anchor_need = pd.read_csv(args.anchor_need_input_csv)
        if allowed_anchor_ids is not None:
            before = len(anchor_need)
            anchor_need = anchor_need[anchor_need["anchor_component_id"].astype(int).isin(allowed_anchor_ids)].copy()
            _log(f"[INFO] Bbox-filtered reused anchor need-repair candidates: {before:,} -> {len(anchor_need):,}")
        _log(f"[INFO] Reusing anchor need-repair gate candidates: {args.anchor_need_input_csv}")
    elif train_anchor_gate:
        _log("[INFO] Building anchor need-repair gate candidates")
        anchor_need = build_anchor_need_candidates(
            pair_candidate_csv=pair_candidate_csv,
            max_anchor_area=float(args.max_anchor_area),
            max_zero_area=float(args.max_zero_area),
            max_after_area=float(args.max_after_area),
            max_zero_source_count=int(args.max_zero_source_count),
            allowed_anchor_ids=allowed_anchor_ids,
            manual_positive_groups=manual_positive_groups,
        )

    if train_anchor_gate:
        anchor_need_fit = _sample_fit_rows(
            anchor_need,
            max_negative_rows=int(args.max_negative_gate_train_rows),
            random_state=int(args.random_state),
        )
        anchor_feature_cols = [
            c
            for c in anchor_need.columns
            if c not in {"anchor_component_id", TARGET_COL, "sample_weight"}
            and not any(marker in c for marker in LABEL_DERIVED_FEATURE_MARKERS)
        ]
        anchor_gate = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=int(args.random_state),
                        early_stopping=True,
                        n_iter_no_change=20,
                        verbose=1,
                    ),
                ),
            ]
        )
        gate_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
        gate_train_idx, gate_test_idx = next(
            gate_splitter.split(
                anchor_need_fit,
                anchor_need_fit[TARGET_COL],
                groups=anchor_need_fit["anchor_component_id"].astype(int),
            )
        )
        gate_train = anchor_need_fit.iloc[gate_train_idx].copy()
        gate_test = anchor_need_fit.iloc[gate_test_idx].copy()
        _log("[INFO] Training anchor need-repair gate")
        anchor_gate.fit(
            gate_train[anchor_feature_cols],
            gate_train[TARGET_COL],
            model__sample_weight=gate_train["sample_weight"].astype(float).to_numpy(),
        )
        gate_test_proba = anchor_gate.predict_proba(gate_test[anchor_feature_cols])[:, 1]
        gate_all_proba = anchor_gate.predict_proba(anchor_need[anchor_feature_cols])[:, 1]
        anchor_need["anchor_need_repair_proba"] = gate_all_proba
        gate_test_metrics = _metrics(gate_test[TARGET_COL].to_numpy(dtype=int), gate_test_proba, 0.8)
        gate_all_metrics = _metrics(anchor_need[TARGET_COL].to_numpy(dtype=int), gate_all_proba, 0.8)

        final_anchor_gate = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=int(args.random_state),
                        early_stopping=True,
                        n_iter_no_change=20,
                        verbose=1,
                    ),
                ),
            ]
        )
        final_anchor_gate.fit(
            anchor_need_fit[anchor_feature_cols],
            anchor_need_fit[TARGET_COL],
            model__sample_weight=anchor_need_fit["sample_weight"].astype(float).to_numpy(),
        )

    payload = {
        "model_kind": _model_kind_for_label_mode(label_mode),
        "pipeline": final_pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "label_mode": label_mode,
            "input_gpkg": str(input_gpkg),
            "pair_candidate_csv": str(pair_candidate_csv),
            "bbox": list(bbox) if bbox is not None else None,
            "top_zero_neighbors": int(args.top_zero_neighbors),
            "max_group_size": int(args.max_group_size),
            "max_anchor_area": float(args.max_anchor_area),
            "max_zero_area": float(args.max_zero_area),
            "max_after_area": float(args.max_after_area),
            "max_zero_source_count": int(args.max_zero_source_count),
            "candidate_strategy": str(args.candidate_strategy),
            "enclosure_level": str(args.enclosure_level),
            "source_enclosed_boundary_ratio": float(args.source_enclosed_boundary_ratio),
            "source_enclosed_min_neighbors": int(args.source_enclosed_min_neighbors),
            "source_enclosed_min_shared_edge": float(args.source_enclosed_min_shared_edge),
            "manual_positive_groups": {str(k): sorted(v) for k, v in manual_positive_groups.items()},
            "candidate_input_csv": str(args.candidate_input_csv),
            "anchor_need_input_csv": str(args.anchor_need_input_csv),
            "selection_gpkg": str(args.selection_gpkg),
            "selection_layer": str(args.selection_layer),
            "source_target_gpkg": str(args.source_target_gpkg),
            "source_target_layer": str(args.source_target_layer),
            "source_target_positive_weight": float(args.source_target_positive_weight),
            "source_target_negative_weight": float(args.source_target_negative_weight),
            "skip_candidate_output": bool(args.skip_candidate_output),
            "max_negative_train_rows": int(args.max_negative_train_rows),
            "max_negative_gate_train_rows": int(args.max_negative_gate_train_rows),
            "threshold": float(args.threshold),
            "random_state": int(args.random_state),
        },
    }
    if label_mode == "weak_reference":
        payload["training_params"]["uprn_skeleton_logic"] = {
            "description": (
                "UPRN points usually anchor the main building; adjacent zero-UPRN land/building "
                "components around non-large anchors are treated as completion skeleton candidates. "
                "A repaired plot should contain at most one UPRN-bearing building source, and "
                "enclosed zero-UPRN land/building components are treated as must-complete skeleton."
            ),
            "max_after_area": 2000.0,
            "small_zero_area": 250.0,
            "medium_zero_area": 1000.0,
            "max_anchor_building_sources_after_merge": 1,
        }
    if final_anchor_gate is not None:
        payload["anchor_gate_pipeline"] = final_anchor_gate
        payload["anchor_gate_feature_cols"] = anchor_feature_cols
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)

    if not anchor_need.empty:
        anchor_need.to_csv(output_dir / "anchor_need_repair_candidates_v1.csv", index=False)
    report_cols = [
        "anchor_component_id",
        "zero_component_ids",
        "label",
        "label_source",
        "sample_weight",
        "anchor_group_repair_proba",
        "anchor_group_repair_pred_at_threshold",
        "candidate_source_fids",
        "source_target_best_fids",
        "source_target_best_overlap_count",
        "source_target_best_overlap_ratio",
        "group_zero_component_count",
        "group_zero_source_count",
        "anchor_pool_zero_count",
        "group_zero_fraction_of_pool",
        "omitted_zero_component_count",
        "omitted_strong_zero_count",
        "omitted_pair_shared_edge_sum",
        "omitted_pair_repair_proba_max",
        "anchor_pool_zero_uprn_land_building_component_count",
        "group_zero_uprn_land_building_component_count",
        "group_zero_uprn_land_building_fraction_of_pool",
        "omitted_zero_uprn_land_building_component_count",
        "omitted_zero_uprn_land_building_area_sum",
        "uprn_skeleton_complete_pool",
        "after_anchor_building_source_count",
        "anchor_pool_enclosed_zero_uprn_land_building_component_count",
        "group_enclosed_zero_uprn_land_building_component_count",
        "group_enclosed_zero_uprn_land_building_fraction_of_pool",
        "omitted_enclosed_zero_uprn_land_building_component_count",
        "enclosed_zero_uprn_complete_pool",
        "anchor_area",
        "zero_area_sum",
        "after_area",
        "pair_repair_proba_min",
        "pair_repair_proba_mean",
        "pair_shared_edge_sum",
        "pair_neighbor_anchor_count_max",
        "pair_zero_anchor_rank_min",
        "mrr_gain",
        "hull_gap_reduction",
        "regularity_score_gain",
        "after_mrr_ratio",
        "after_hull_gap_ratio",
        "after_regularity_score",
        "reference_positive_zero_component_ids",
        "anchor_reference_fids",
        "zero_reference_fids",
    ]
    report_cols = [column for column in report_cols if column in dataset.columns]
    dataset[report_cols].sort_values("anchor_group_repair_proba", ascending=False).to_csv(
        output_dir / PREDICTIONS_FILE_NAME,
        index=False,
    )
    metrics = {
        "model_kind": _model_kind_for_label_mode(label_mode),
        "input_gpkg": str(input_gpkg),
        "pair_candidate_csv": str(pair_candidate_csv),
        "bbox": list(bbox) if bbox is not None else None,
        "output_dir": str(output_dir),
        "model": str(output_dir / MODEL_FILE_NAME),
        "threshold": float(args.threshold),
        "label_mode": label_mode,
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "label_source_counts": dataset["label_source"].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
    }
    if train_anchor_gate:
        metrics.update(
            {
                "anchor_gate_feature_columns": anchor_feature_cols,
                "anchor_gate_test_metrics": gate_test_metrics,
                "anchor_gate_all_metrics": gate_all_metrics,
            }
        )
    (output_dir / "anchor_group_repair_metrics_v1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log("[DONE] Anchor group repair model training complete")
    _log(json.dumps(test_metrics, indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

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
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_operation_pruned_only_guard_v2.gpkg"
)
DEFAULT_PAIR_CANDIDATE_CSV = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/"
    "model_predicted_polygons_anchor_repaired_threshold_097.anchor_candidates.csv"
)
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback"
MODEL_FILE_NAME = "anchor_group_repair_model_v1.joblib"
CANDIDATES_FILE_NAME = "anchor_group_repair_candidates_v1.csv"
PREDICTIONS_FILE_NAME = "anchor_group_repair_predictions_v1.csv"

TARGET_COL = "label"
CATEGORICAL_FEATURES = ["role_pair_signature"]
ID_COLS = {
    "anchor_component_id",
    "zero_component_ids",
    "anchor_reference_fids",
    "zero_reference_fids",
    "reference_positive_zero_component_ids",
    "label_source",
}


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


def _update_prefixed(row: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = float(value)


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
            return 1, "manual_complete_positive", 60.0
        if zero_ids & manual:
            return 0, "manual_partial_or_overmerge_negative", 18.0

    if reference_positive_zero_ids:
        if zero_ids == reference_positive_zero_ids:
            return 1, "reference_complete_positive", 6.0
        if zero_ids < reference_positive_zero_ids:
            return 0, "reference_partial_negative", 5.0
        if zero_ids & reference_positive_zero_ids:
            return 0, "reference_overmerge_negative", 5.0
    return 0, "reference_different_negative", 1.0


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLS | {TARGET_COL, "sample_weight"}
    feature_cols = [c for c in dataset.columns if c not in excluded]
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


def _read_pair_candidates(path: Path) -> pd.DataFrame:
    usecols = [
        "anchor_component_id",
        "zero_component_id",
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
    return pair


def build_anchor_need_candidates(
    *,
    pair_candidate_csv: Path,
    max_anchor_area: float,
    max_zero_area: float,
    max_after_area: float,
    max_zero_source_count: int,
    manual_positive_groups: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    manual_positive_groups = manual_positive_groups or {}
    pair = _read_pair_candidates(pair_candidate_csv)
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


def _component_geometries(input_gpkg: Path) -> tuple[dict[int, Any], dict[int, dict[str, Any]], Any]:
    predicted = pyogrio.read_dataframe(input_gpkg, layer="predicted_parcels_with_uprn")
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    predicted["pred_component_id"] = predicted["pred_component_id"].astype(int)
    geom_by_component = predicted.set_index("pred_component_id").geometry.to_dict()
    attrs = predicted.set_index("pred_component_id").drop(columns="geometry").to_dict("index")
    return geom_by_component, attrs, predicted.crs


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
    manual_positive_groups: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    manual_positive_groups = manual_positive_groups or {}
    _log("[INFO] Reading pair anchor candidates")
    pair = _read_pair_candidates(pair_candidate_csv)
    pair = pair[
        pair["anchor_area"].astype(float).le(float(max_anchor_area))
        & pair["zero_area"].astype(float).le(float(max_zero_area))
        & pair["zero_source_count"].fillna(999).astype(int).le(int(max_zero_source_count))
    ].copy()
    if pair.empty:
        raise RuntimeError("No pair candidates remain after group repair filters.")

    _log("[INFO] Reading component geometries")
    geom_by_component, attrs_by_component, _ = _component_geometries(input_gpkg)
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
        zero_ids = sorted(rows_by_zero)
        if not zero_ids:
            continue
        anchor_geom = geom_by_component[anchor_id]
        if anchor_id not in anchor_shape_cache:
            anchor_shape_cache[anchor_id] = _shape_metrics(anchor_geom)
        anchor_shape = anchor_shape_cache[anchor_id]
        anchor_attrs = attrs_by_component.get(anchor_id, {})
        reference_positive_zero_ids = positive_by_anchor.get(anchor_id, set()) & set(zero_ids)

        for size in range(1, min(int(max_group_size), len(zero_ids)) + 1):
            for combo in itertools.combinations(zero_ids, size):
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
                }
                _update_prefixed(rec, "anchor", anchor_shape)
                _update_prefixed(rec, "zero_group", zero_shape)
                _update_prefixed(rec, "after", after_shape)
                records.append(rec)

    if not records:
        raise RuntimeError("No anchor group candidates were generated.")
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UPRN-anchor group repair model.")
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--pair-candidate-csv", default=DEFAULT_PAIR_CANDIDATE_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-zero-neighbors", type=int, default=6)
    parser.add_argument("--max-group-size", type=int, default=6)
    parser.add_argument("--max-anchor-area", type=float, default=2000.0)
    parser.add_argument("--max-zero-area", type=float, default=1000.0)
    parser.add_argument("--max-after-area", type=float, default=2000.0)
    parser.add_argument("--max-zero-source-count", type=int, default=8)
    parser.add_argument("--manual-positive-groups", default="")
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

    dataset = build_anchor_group_candidates(
        input_gpkg=input_gpkg,
        pair_candidate_csv=pair_candidate_csv,
        top_zero_neighbors=int(args.top_zero_neighbors),
        max_group_size=int(args.max_group_size),
        max_anchor_area=float(args.max_anchor_area),
        max_zero_area=float(args.max_zero_area),
        max_after_area=float(args.max_after_area),
        max_zero_source_count=int(args.max_zero_source_count),
        manual_positive_groups=manual_positive_groups,
    )
    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
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
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=19,
        l2_regularization=0.08,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    groups = dataset["anchor_component_id"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(dataset, dataset[TARGET_COL], groups=groups))
    train = dataset.iloc[train_idx].copy()
    test = dataset.iloc[test_idx].copy()

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
        dataset[feature_cols],
        dataset[TARGET_COL],
        model__sample_weight=dataset["sample_weight"].astype(float).to_numpy(),
    )

    _log("[INFO] Building anchor need-repair gate candidates")
    anchor_need = build_anchor_need_candidates(
        pair_candidate_csv=pair_candidate_csv,
        max_anchor_area=float(args.max_anchor_area),
        max_zero_area=float(args.max_zero_area),
        max_after_area=float(args.max_after_area),
        max_zero_source_count=int(args.max_zero_source_count),
        manual_positive_groups=manual_positive_groups,
    )
    anchor_feature_cols = [
        c for c in anchor_need.columns if c not in {"anchor_component_id", TARGET_COL, "sample_weight"}
    ]
    anchor_gate = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.05,
                    random_state=int(args.random_state),
                    early_stopping=True,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    gate_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
    gate_train_idx, gate_test_idx = next(
        gate_splitter.split(anchor_need, anchor_need[TARGET_COL], groups=anchor_need["anchor_component_id"].astype(int))
    )
    gate_train = anchor_need.iloc[gate_train_idx].copy()
    gate_test = anchor_need.iloc[gate_test_idx].copy()
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
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.05,
                    random_state=int(args.random_state),
                    early_stopping=True,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    final_anchor_gate.fit(
        anchor_need[anchor_feature_cols],
        anchor_need[TARGET_COL],
        model__sample_weight=anchor_need["sample_weight"].astype(float).to_numpy(),
    )

    payload = {
        "pipeline": final_pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "anchor_gate_pipeline": final_anchor_gate,
        "anchor_gate_feature_cols": anchor_feature_cols,
        "training_params": {
            "input_gpkg": str(input_gpkg),
            "pair_candidate_csv": str(pair_candidate_csv),
            "top_zero_neighbors": int(args.top_zero_neighbors),
            "max_group_size": int(args.max_group_size),
            "max_anchor_area": float(args.max_anchor_area),
            "max_zero_area": float(args.max_zero_area),
            "max_after_area": float(args.max_after_area),
            "max_zero_source_count": int(args.max_zero_source_count),
            "manual_positive_groups": {str(k): sorted(v) for k, v in manual_positive_groups.items()},
            "threshold": float(args.threshold),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)

    dataset.to_csv(output_dir / CANDIDATES_FILE_NAME, index=False)
    anchor_need.to_csv(output_dir / "anchor_need_repair_candidates_v1.csv", index=False)
    report_cols = [
        "anchor_component_id",
        "zero_component_ids",
        "label",
        "label_source",
        "sample_weight",
        "anchor_group_repair_proba",
        "anchor_group_repair_pred_at_threshold",
        "group_zero_component_count",
        "group_zero_source_count",
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
    dataset[report_cols].sort_values("anchor_group_repair_proba", ascending=False).to_csv(
        output_dir / PREDICTIONS_FILE_NAME,
        index=False,
    )
    metrics = {
        "input_gpkg": str(input_gpkg),
        "pair_candidate_csv": str(pair_candidate_csv),
        "output_dir": str(output_dir),
        "model": str(output_dir / MODEL_FILE_NAME),
        "threshold": float(args.threshold),
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "label_source_counts": dataset["label_source"].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "anchor_gate_feature_columns": anchor_feature_cols,
        "anchor_gate_test_metrics": gate_test_metrics,
        "anchor_gate_all_metrics": gate_all_metrics,
    }
    (output_dir / "anchor_group_repair_metrics_v1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log("[DONE] Anchor group repair model training complete")
    _log(json.dumps(test_metrics, indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

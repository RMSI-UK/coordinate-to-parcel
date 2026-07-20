#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wfs_merge_native"))
sys.path.insert(0, str(ROOT / "wfs_merge_native_train"))

from apply_wfs_raw_anchor_group_model import _select_candidates  # noqa: E402
from train_wfs_raw_anchor_group_model import (  # noqa: E402
    DEFAULT_TARGET_GPKG,
    DEFAULT_TARGET_LAYER,
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_ID_FIELD,
    DEFAULT_UPRN_LAYER,
    DEFAULT_WFS_CLEAN_GPKG,
    DEFAULT_WFS_CLEAN_LAYER,
    _add_uprn_counts,
    _build_source_indexes,
    _ids_text,
    _parse_bbox,
    _parse_id_set,
    _read_clean_wfs,
    _read_targets,
    read_candidate_inputs,
)


DEFAULT_APPLY_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_full_v1/"
    "wfs_raw_anchor_group_applied_full.gpkg"
)
DEFAULT_THRESHOLDS = "0.5,0.7,0.8,0.856,0.9,0.95,0.97"
APPLY_CANDIDATE_LAYERS = [
    "raw_anchor_group_selected",
    "raw_anchor_group_review_candidates",
    "raw_anchor_group_conflicts",
]


@dataclass(frozen=True)
class TargetIndex:
    target_sets: list[frozenset[int]]
    target_ids: list[int]
    exact_to_target_ids: dict[frozenset[int], list[int]]
    source_to_target_indexes: dict[int, set[int]]
    missing_source_target_ids: set[int]


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _thresholds(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value or "").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("--thresholds must contain at least one numeric value")
    return sorted(set(out))


def _parse_int_list(value: str) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _read_gpkg_layer_without_geometry(path: Path, layer: str) -> pd.DataFrame:
    try:
        frame = pyogrio.read_dataframe(path, layer=layer, read_geometry=False)
    except TypeError:
        frame = pyogrio.read_dataframe(path, layer=layer)
    if isinstance(frame, gpd.GeoDataFrame):
        geom_column = frame.geometry.name if frame.geometry.name in frame.columns else "geometry"
        frame = pd.DataFrame(frame.drop(columns=[geom_column], errors="ignore"))
    else:
        frame = pd.DataFrame(frame).drop(columns=["geometry"], errors="ignore")
    return frame


def _available_layers(path: Path) -> set[str]:
    layers = pyogrio.list_layers(path)
    return {str(row[0]) for row in layers}


def read_apply_candidates(path: Path) -> pd.DataFrame:
    available = _available_layers(path)
    frames: list[pd.DataFrame] = []
    for layer in APPLY_CANDIDATE_LAYERS:
        if layer not in available:
            _log(f"[WARN] Apply GPKG layer not found, skipping: {layer}")
            continue
        frame = _read_gpkg_layer_without_geometry(path, layer)
        frame["candidate_layer"] = layer
        frames.append(frame)
        _log(f"[INFO] Read apply candidate layer: {layer}; rows={len(frame):,}")
    if not frames:
        raise RuntimeError(f"No candidate layers found in {path}")
    candidates = pd.concat(frames, ignore_index=True)
    return _clean_candidate_rows(candidates)


def _clean_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"anchor_source_fid", "candidate_clean_fids", "candidate_source_fids", "raw_anchor_group_proba"}
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"Candidate rows missing required columns: {sorted(missing)}")
    out = candidates.copy()
    out["anchor_source_fid"] = pd.to_numeric(out["anchor_source_fid"], errors="coerce")
    out["raw_anchor_group_proba"] = pd.to_numeric(out["raw_anchor_group_proba"], errors="coerce")
    out = out[out["anchor_source_fid"].notna() & out["raw_anchor_group_proba"].notna()].copy()
    out["anchor_source_fid"] = out["anchor_source_fid"].astype("int64")
    for column in ["candidate_clean_fids", "candidate_source_fids", "anchor_clean_fids"]:
        if column in out.columns:
            out[column] = out[column].fillna("").astype(str)
    out = out.sort_values("raw_anchor_group_proba", ascending=False).drop_duplicates(
        ["anchor_source_fid", "candidate_clean_fids"],
        keep="first",
    )
    out = out.reset_index(drop=True)
    _log(f"[INFO] Candidate rows after cleanup/dedupe={len(out):,}")
    return out


def build_target_index(args: argparse.Namespace, source_to_clean: dict[int, list[int]]) -> TargetIndex:
    target = _read_targets(
        Path(args.target_gpkg),
        str(args.target_layer),
        _parse_bbox(args.bbox),
        int(args.max_target_rows),
    )
    if int(args.target_id_mod) > 0:
        remainders = _parse_int_list(str(args.target_id_remainders))
        if not remainders:
            raise ValueError("--target-id-remainders is required when --target-id-mod is set")
        target = target[
            target["train_component_id"].astype(int).mod(int(args.target_id_mod)).isin(remainders)
        ].copy()
        _log(
            "[INFO] Target id modulo filter applied: "
            f"mod={int(args.target_id_mod)}; remainders={sorted(remainders)}; rows={len(target):,}"
        )
    target_sets: list[frozenset[int]] = []
    target_ids: list[int] = []
    exact_to_target_ids: dict[frozenset[int], list[int]] = {}
    source_to_target_indexes: dict[int, set[int]] = {}
    missing_source_target_ids: set[int] = set()
    for idx, row in enumerate(target.itertuples(index=False)):
        source_ids = getattr(row, "target_source_set", None)
        if not isinstance(source_ids, set):
            source_ids = _parse_id_set(getattr(row, "source_wfs_fids", ""))
        source_set = frozenset(int(v) for v in source_ids)
        if len(source_set) < 2:
            continue
        target_id = int(getattr(row, "train_component_id"))
        if any(int(source_fid) not in source_to_clean for source_fid in source_set):
            missing_source_target_ids.add(target_id)
        target_sets.append(source_set)
        target_ids.append(target_id)
        exact_to_target_ids.setdefault(source_set, []).append(target_id)
        target_index = len(target_sets) - 1
        for source_fid in source_set:
            source_to_target_indexes.setdefault(int(source_fid), set()).add(target_index)
    _log(
        "[INFO] Target index: "
        f"target_rows={len(target_sets):,}; exact_sets={len(exact_to_target_ids):,}; "
        f"source_members={len(source_to_target_indexes):,}; "
        f"missing_source_targets={len(missing_source_target_ids):,}"
    )
    return TargetIndex(
        target_sets=target_sets,
        target_ids=target_ids,
        exact_to_target_ids=exact_to_target_ids,
        source_to_target_indexes=source_to_target_indexes,
        missing_source_target_ids=missing_source_target_ids,
    )


def build_wfs_indexes(args: argparse.Namespace) -> tuple[dict[int, int], dict[int, list[int]]]:
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), _parse_bbox(args.bbox))
    wfs = _add_uprn_counts(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
    )
    source_to_clean, _source_by_clean = _build_source_indexes(wfs)
    anchor_mask = wfs["uprn_count"].astype(int).gt(0) & wfs["anchor_role"].isin(["building", "land"])
    anchors = sorted(set(int(value) for value in wfs.loc[anchor_mask, "source_fid"].astype(int)))
    owner = {
        int(clean_id): int(source_fid)
        for source_fid in anchors
        for clean_id in source_to_clean.get(int(source_fid), [])
    }
    _log(f"[INFO] Anchor owner index: anchors={len(anchors):,}; owned_clean_fids={len(owner):,}")
    return owner, source_to_clean


def build_anchor_owner_by_clean(args: argparse.Namespace) -> dict[int, int]:
    owner, _source_to_clean = build_wfs_indexes(args)
    return owner


def _classify_source_set(source_ids: frozenset[int], targets: TargetIndex) -> tuple[str, int | None, str, int, float]:
    if not source_ids:
        return "empty", None, "", 0, 0.0
    exact_ids = targets.exact_to_target_ids.get(source_ids)
    if exact_ids:
        return "exact", int(exact_ids[0]), _ids_text(source_ids), len(source_ids), 1.0

    candidate_target_indexes: set[int] = set()
    for source_fid in source_ids:
        candidate_target_indexes.update(targets.source_to_target_indexes.get(int(source_fid), set()))
    if not candidate_target_indexes:
        return "not_in_council_train", None, "", 0, 0.0

    best_index = -1
    best_overlap = -1
    best_jaccard = -1.0
    for target_index in candidate_target_indexes:
        target_set = targets.target_sets[int(target_index)]
        overlap = len(source_ids & target_set)
        union = len(source_ids | target_set)
        jaccard = _safe_ratio(float(overlap), float(union))
        if overlap > best_overlap or (overlap == best_overlap and jaccard > best_jaccard):
            best_index = int(target_index)
            best_overlap = int(overlap)
            best_jaccard = float(jaccard)

    target_set = targets.target_sets[best_index]
    target_id = int(targets.target_ids[best_index])
    if source_ids < target_set:
        match_class = "partial"
    elif target_set < source_ids:
        match_class = "overmerge"
    else:
        match_class = "mismatch"
    return match_class, target_id, _ids_text(target_set), int(best_overlap), float(best_jaccard)


def classify_candidates(candidates: pd.DataFrame, targets: TargetIndex) -> pd.DataFrame:
    records: list[tuple[str, int | None, str, int, float]] = []
    for value in candidates["candidate_source_fids"]:
        records.append(_classify_source_set(frozenset(_parse_id_set(value)), targets))
    classified = candidates.copy()
    classified["match_class"] = [row[0] for row in records]
    classified["matched_target_id"] = [row[1] for row in records]
    classified["matched_target_source_fids"] = [row[2] for row in records]
    classified["target_overlap_count"] = [row[3] for row in records]
    classified["target_jaccard"] = [row[4] for row in records]
    _log(f"[INFO] Candidate match classes: {classified['match_class'].value_counts().to_dict()}")
    return classified


def _proba_quantiles(rows: pd.DataFrame) -> dict[str, float | None]:
    if rows.empty:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    values = rows["raw_anchor_group_proba"].astype(float).to_numpy()
    return {
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def candidate_pool_summary(candidates: pd.DataFrame, targets: TargetIndex) -> dict[str, Any]:
    exact = candidates[candidates["match_class"].eq("exact")]
    counts = candidates["match_class"].value_counts().to_dict()
    proba_by_class = {
        match_class: _proba_quantiles(group)
        for match_class, group in candidates.groupby("match_class", dropna=False)
    }
    unique_exact_targets = int(exact["matched_target_id"].dropna().astype(int).nunique())
    available_target_rows = int(len(targets.target_sets) - len(targets.missing_source_target_ids))
    return {
        "candidate_rows": int(len(candidates)),
        "target_rows": int(len(targets.target_sets)),
        "missing_source_target_rows": int(len(targets.missing_source_target_ids)),
        "available_target_rows": available_target_rows,
        "match_class_counts": {str(k): int(v) for k, v in counts.items()},
        "exact_candidate_rows": int(len(exact)),
        "exact_candidate_unique_targets": unique_exact_targets,
        "candidate_pool_exact_recall_ceiling": _safe_ratio(float(unique_exact_targets), float(len(targets.target_sets))),
        "candidate_pool_exact_recall_ceiling_available": _safe_ratio(
            float(unique_exact_targets),
            float(available_target_rows),
        ),
        "proba_quantiles_by_match_class": proba_by_class,
    }


def selected_metrics(
    *,
    threshold: float,
    selected: pd.DataFrame,
    review: pd.DataFrame,
    conflicts: pd.DataFrame,
    targets: TargetIndex,
) -> dict[str, Any]:
    counts = selected["match_class"].value_counts().to_dict() if "match_class" in selected.columns else {}
    exact_rows = int(counts.get("exact", 0))
    not_in_rows = int(counts.get("not_in_council_train", 0))
    empty_rows = int(counts.get("empty", 0))
    evaluable_rows = int(len(selected) - not_in_rows - empty_rows)
    exact_unique_targets = (
        int(selected.loc[selected["match_class"].eq("exact"), "matched_target_id"].dropna().astype(int).nunique())
        if "match_class" in selected.columns and not selected.empty
        else 0
    )
    available_target_rows = int(len(targets.target_sets) - len(targets.missing_source_target_ids))
    return {
        "threshold": float(threshold),
        "selected_rows": int(len(selected)),
        "selected_exact_rows": exact_rows,
        "selected_partial_rows": int(counts.get("partial", 0)),
        "selected_overmerge_rows": int(counts.get("overmerge", 0)),
        "selected_mismatch_rows": int(counts.get("mismatch", 0)),
        "selected_not_in_council_train_rows": not_in_rows,
        "selected_empty_rows": empty_rows,
        "selected_evaluable_rows": evaluable_rows,
        "selected_exact_unique_targets": exact_unique_targets,
        "selected_exact_precision_evaluable": _safe_ratio(float(exact_rows), float(evaluable_rows)),
        "selected_exact_precision_all": _safe_ratio(float(exact_rows), float(len(selected))),
        "selected_exact_recall_targets": _safe_ratio(float(exact_unique_targets), float(len(targets.target_sets))),
        "selected_exact_recall_available_targets": _safe_ratio(
            float(exact_unique_targets),
            float(available_target_rows),
        ),
        "review_rows": int(len(review)),
        "conflict_rows": int(len(conflicts)),
        "high_candidate_rows": int(len(selected) + len(conflicts)),
    }


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if str(args.output_csv).strip():
        csv_path = Path(args.output_csv)
    elif str(args.candidate_input_csv).strip():
        csv_path = Path(args.candidate_input_csv).with_suffix(".threshold_sweep.csv")
    else:
        csv_path = Path(args.apply_gpkg).with_suffix(".threshold_sweep.csv")

    if str(args.output_json).strip():
        json_path = Path(args.output_json)
    else:
        json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw-anchor-group production candidates against council labels.")
    parser.add_argument("--apply-gpkg", default=DEFAULT_APPLY_GPKG)
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--target-gpkg", default=DEFAULT_TARGET_GPKG)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--target-id-mod", type=int, default=0)
    parser.add_argument("--target-id-remainders", default="")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--classified-candidate-output-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.candidate_input_csv).strip():
        candidates = _clean_candidate_rows(read_candidate_inputs(str(args.candidate_input_csv)))
    else:
        candidates = read_apply_candidates(Path(args.apply_gpkg))

    anchor_owner_by_clean, source_to_clean = build_wfs_indexes(args)
    targets = build_target_index(args, source_to_clean)
    candidates = classify_candidates(candidates, targets)

    rows: list[dict[str, Any]] = []
    for threshold in _thresholds(args.thresholds):
        selected, review, conflicts = _select_candidates(
            candidates,
            threshold=float(threshold),
            anchor_owner_by_clean=anchor_owner_by_clean,
        )
        rows.append(
            selected_metrics(
                threshold=float(threshold),
                selected=selected,
                review=review,
                conflicts=conflicts,
                targets=targets,
            )
        )
        _log(
            "[INFO] Sweep "
            f"threshold={threshold:.6g}; selected={len(selected):,}; "
            f"exact={rows[-1]['selected_exact_rows']:,}; "
            f"evaluable_precision={rows[-1]['selected_exact_precision_evaluable']:.4f}; "
            f"target_recall_available={rows[-1]['selected_exact_recall_available_targets']:.4f}; "
            f"target_recall_all={rows[-1]['selected_exact_recall_targets']:.4f}"
        )

    csv_path, json_path = output_paths(args)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(csv_path, index=False)
    summary = {
        "apply_gpkg": str(args.apply_gpkg),
        "candidate_input_csv": str(args.candidate_input_csv),
        "target_gpkg": str(args.target_gpkg),
        "target_layer": str(args.target_layer),
        "bbox": str(args.bbox),
        "target_id_mod": int(args.target_id_mod),
        "target_id_remainders": sorted(_parse_int_list(str(args.target_id_remainders))),
        "candidate_pool_summary": candidate_pool_summary(candidates, targets),
        "threshold_sweep": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if str(args.classified_candidate_output_csv).strip():
        path = Path(args.classified_candidate_output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(path, index=False)
        _log(f"[INFO] Wrote classified candidates: rows={len(candidates):,}; path={path}")
    _log(f"[DONE] threshold_sweep_csv={csv_path}")
    _log(f"[DONE] threshold_sweep_json={json_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
import joblib
import pandas as pd
import pyogrio


BASE_DIR = Path("/data/sheffield/spatial/base-map")
DEFAULT_PARCEL_GPKG = BASE_DIR / "sheffield_wfs_merged_council_train_parcel.gpkg"
DEFAULT_INPUT_GPKG = BASE_DIR / "sheffield_wfs_merged_council_train_molecular_depth3.gpkg"
DEFAULT_EDGE_CACHE = (
    BASE_DIR
    / "tmp/wfs_raw_anchor_group_model_completeness_v2_context_cache/"
    / "shared_edges_e455305190c051e0db7e7441.joblib"
)
PARCEL_LAYER = "train_parcel_label"
INPUT_LAYER = "train_input_molecular_depth3"
FEATURE_POLICY = "anchor_is_model_feature"


def _log(message: str) -> None:
    print(message, flush=True)


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp_anchor_feature{path.suffix}")


def _as_int_series(frame: pd.DataFrame, column: str, default: int = 0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="int64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype("int64")


def _parse_ids_text(value: Any) -> set[int]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    out: set[int] = set()
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(float(part)))
        except ValueError:
            continue
    return out


def _find_valid_labels(labels: gpd.GeoDataFrame, inputs: gpd.GeoDataFrame) -> tuple[set[int], pd.DataFrame]:
    depth0 = inputs[
        _as_int_series(inputs, "neighbor_depth").eq(0)
        & _as_int_series(inputs, "label_id").gt(0)
    ].copy()
    depth0["_building_uprn_anchor"] = (
        _as_int_series(depth0, "is_building_theme").eq(1)
        & _as_int_series(depth0, "uprn_count").gt(0)
    ).astype("int64")

    per_label = depth0.groupby("label_id", as_index=True).agg(
        building_uprn_anchor_count=("_building_uprn_anchor", "sum"),
        depth0_row_count=("raw_clean_fid", "size"),
    )
    complete_labels = set(
        _as_int_series(labels, "label_id")[
            _as_int_series(labels, "source_complete_in_input").eq(1)
        ].astype(int)
    )
    valid_ids = set(
        int(label_id)
        for label_id in per_label.index[
            per_label["building_uprn_anchor_count"].eq(1)
            & per_label.index.astype(int).isin(complete_labels)
        ]
    )

    anchors = depth0[
        depth0["label_id"].astype(int).isin(valid_ids)
        & depth0["_building_uprn_anchor"].eq(1)
    ][["label_id", "raw_clean_fid", "source_fid", "uprn_count"]].copy()
    anchors = anchors.rename(
        columns={
            "raw_clean_fid": "anchor_raw_clean_fid",
            "source_fid": "anchor_source_fid",
            "uprn_count": "anchor_uprn_count",
        }
    )
    anchors["label_id"] = anchors["label_id"].astype("int64")
    anchors["anchor_raw_clean_fid"] = anchors["anchor_raw_clean_fid"].astype("int64")
    anchors["anchor_source_fid"] = anchors["anchor_source_fid"].astype("int64")
    anchors["anchor_uprn_count"] = anchors["anchor_uprn_count"].astype("int64")
    anchors["anchor_kind"] = "building"
    anchors["anchor_polygon_count"] = 1
    anchors["anchor_is_building"] = 1

    anchor_counts = anchors.groupby("label_id").size()
    if not anchor_counts.eq(1).all() or len(anchor_counts) != len(valid_ids):
        bad = anchor_counts[~anchor_counts.eq(1)].head(10).to_dict()
        raise RuntimeError(f"Valid label anchor count invariant failed: {bad}")

    return valid_ids, anchors


def _expand_depths(
    *,
    inputs: gpd.GeoDataFrame,
    valid_label_ids: set[int],
    edge_cache_path: Path,
    max_depth: int,
) -> dict[int, int]:
    depth0 = inputs[
        _as_int_series(inputs, "neighbor_depth").eq(0)
        & _as_int_series(inputs, "label_id").isin(valid_label_ids)
    ]
    seeds = set(int(v) for v in depth0["raw_clean_fid"])
    if not seeds:
        raise RuntimeError("No valid depth0 seeds after anchor filtering.")

    universe = set(int(v) for v in inputs["raw_clean_fid"])
    cache = joblib.load(edge_cache_path)
    adjacency = cache["adjacency"]

    distances: dict[int, int] = {node: 0 for node in seeds}
    frontier = set(seeds)
    for depth in range(1, int(max_depth) + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            for neighbor, _shared_edge_len in adjacency.get(int(node), ()):
                neighbor = int(neighbor)
                if neighbor in universe and neighbor not in distances:
                    distances[neighbor] = depth
                    next_frontier.add(neighbor)
        _log(f"[INFO] retained-label BFS depth={depth}: new={len(next_frontier):,}; total={len(distances):,}")
        frontier = next_frontier
    return distances


def _standardize_labels(labels: gpd.GeoDataFrame, valid_label_ids: set[int], anchors: pd.DataFrame) -> gpd.GeoDataFrame:
    out = labels[_as_int_series(labels, "label_id").isin(valid_label_ids)].copy()
    out = out.merge(anchors, on="label_id", how="left", validate="one_to_one")
    out["label_training_status"] = "available"
    out["model_feature_policy"] = FEATURE_POLICY
    out["source_complete_in_input"] = 1
    out["missing_source_count"] = 0
    out["missing_source_fids"] = ""
    for column in [
        "anchor_raw_clean_fid",
        "anchor_source_fid",
        "anchor_uprn_count",
        "anchor_polygon_count",
        "anchor_is_building",
    ]:
        out[column] = _as_int_series(out, column)
    out["anchor_kind"] = out["anchor_kind"].fillna("building").astype(str)
    out = out.sort_values("label_id").reset_index(drop=True)
    return out


def _clear_neighbor_label_fields(inputs: gpd.GeoDataFrame, mask: pd.Series) -> None:
    int_defaults = {
        "label_id": -1,
        "label_fid": -1,
        "label_source_order": -1,
        "label_source_count": 0,
        "label_uprn_count": 0,
        "council_fid": -1,
        "source_complete_in_label": 0,
    }
    for column, value in int_defaults.items():
        if column in inputs.columns:
            inputs.loc[mask, column] = int(value)
    float_defaults = ["council_coverage", "wfs_coverage", "selection_score"]
    for column in float_defaults:
        if column in inputs.columns:
            inputs.loc[mask, column] = 0.0
    str_defaults = ["council_label", "council_name"]
    for column in str_defaults:
        if column in inputs.columns:
            inputs.loc[mask, column] = ""


def _standardize_inputs(
    inputs: gpd.GeoDataFrame,
    distances: dict[int, int],
    valid_label_ids: set[int],
    anchors: pd.DataFrame,
) -> gpd.GeoDataFrame:
    out = inputs[inputs["raw_clean_fid"].astype(int).isin(distances)].copy()
    out["neighbor_depth"] = out["raw_clean_fid"].astype(int).map(distances).astype("int64")
    out = out.sort_values(["neighbor_depth", "label_id", "raw_clean_fid"]).reset_index(drop=True)
    out["input_id"] = range(1, len(out) + 1)
    out["input_fid"] = out["input_id"]
    out["input_training_status"] = "available"
    out["model_feature_policy"] = FEATURE_POLICY

    out["is_label_member"] = (
        out["neighbor_depth"].eq(0)
        & _as_int_series(out, "label_id").isin(valid_label_ids)
    ).astype("int64")
    out["is_depth_neighbor"] = out["neighbor_depth"].gt(0).astype("int64")
    out["source_complete_in_label"] = out["is_label_member"].astype("int64")

    neighbor_mask = out["neighbor_depth"].gt(0)
    _clear_neighbor_label_fields(out, neighbor_mask)

    out["has_uprn"] = _as_int_series(out, "uprn_count").gt(0).astype("int64")
    out["is_building_uprn_anchor"] = (
        _as_int_series(out, "is_building_theme").eq(1)
        & _as_int_series(out, "uprn_count").gt(0)
    ).astype("int64")
    out["is_nonanchor_uprn"] = (
        out["has_uprn"].eq(1)
        & out["is_building_uprn_anchor"].eq(0)
    ).astype("int64")

    label_anchor_raw_ids = set(int(v) for v in anchors["anchor_raw_clean_fid"])
    out["is_building_label_anchor"] = (
        out["is_label_member"].eq(1)
        & out["raw_clean_fid"].astype(int).isin(label_anchor_raw_ids)
    ).astype("int64")

    # Keep the older aux columns aligned for diagnostics/backward compatibility.
    out["aux_is_building_uprn_anchor"] = out["is_building_uprn_anchor"]
    out["aux_is_building_label_anchor"] = out["is_building_label_anchor"]
    out["aux_is_nonanchor_uprn"] = out["is_nonanchor_uprn"]

    for column in [
        "input_id",
        "input_fid",
        "neighbor_depth",
        "is_label_member",
        "is_depth_neighbor",
        "raw_clean_fid",
        "raw_clean_attr_fid",
        "source_fid",
        "uprn_count",
        "has_uprn",
        "is_building_theme",
        "zero_uprn_plot_eligible",
        "source_complete_in_label",
        "label_id",
        "label_fid",
        "label_source_order",
        "label_source_count",
        "label_uprn_count",
        "council_fid",
        "plot_eligible",
        "is_building_uprn_anchor",
        "is_building_label_anchor",
        "is_nonanchor_uprn",
        "aux_is_building_uprn_anchor",
        "aux_is_building_label_anchor",
        "aux_is_nonanchor_uprn",
    ]:
        if column in out.columns:
            out[column] = _as_int_series(out, column)
    return out


def _qa(
    labels: gpd.GeoDataFrame,
    inputs: gpd.GeoDataFrame,
    *,
    expected_label_rows: int,
    expected_input_rows: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    depth_counts = {
        int(depth): int(count)
        for depth, count in Counter(_as_int_series(inputs, "neighbor_depth")).items()
    }
    report["label_rows"] = int(len(labels))
    report["input_rows"] = int(len(inputs))
    report["input_depth_counts"] = dict(sorted(depth_counts.items()))
    report["label_rows_expected"] = int(expected_label_rows)
    report["input_rows_expected"] = int(expected_input_rows)

    failures: list[str] = []
    if len(labels) != int(expected_label_rows):
        failures.append(f"label rows {len(labels)} != {expected_label_rows}")
    if len(inputs) != int(expected_input_rows):
        failures.append(f"input rows {len(inputs)} != {expected_input_rows}")

    depth0 = inputs[_as_int_series(inputs, "neighbor_depth").eq(0)].copy()
    depth_neighbors = inputs[_as_int_series(inputs, "neighbor_depth").gt(0)].copy()

    label_anchor_counts = depth0.groupby("label_id")["is_building_label_anchor"].sum()
    bad_anchor_counts = label_anchor_counts[~label_anchor_counts.eq(1)]
    report["labels_with_exactly_one_building_uprn_anchor"] = int(label_anchor_counts.eq(1).sum())
    report["bad_label_anchor_count"] = int(len(bad_anchor_counts))
    if len(bad_anchor_counts) > 0:
        failures.append(f"labels without exactly one building label anchor: {len(bad_anchor_counts)}")

    anchor_rows = depth0[_as_int_series(depth0, "is_building_label_anchor").eq(1)]
    nonbuilding_anchors = int((anchor_rows["is_building_theme"].astype(int) != 1).sum())
    nonpositive_anchor_uprn = int((anchor_rows["uprn_count"].astype(int) <= 0).sum())
    report["label_anchor_non_building"] = nonbuilding_anchors
    report["label_anchor_uprn_count_le_0"] = nonpositive_anchor_uprn
    if nonbuilding_anchors:
        failures.append(f"label anchor non-building rows: {nonbuilding_anchors}")
    if nonpositive_anchor_uprn:
        failures.append(f"label anchor uprn_count <= 0 rows: {nonpositive_anchor_uprn}")

    raw_clean_duplicates = int(inputs["raw_clean_fid"].duplicated().sum())
    report["input_raw_clean_fid_duplicates"] = raw_clean_duplicates
    if raw_clean_duplicates:
        failures.append(f"input raw_clean_fid duplicates: {raw_clean_duplicates}")

    depth_neighbor_bad_label_id = int((_as_int_series(depth_neighbors, "label_id") != -1).sum())
    depth0_bad_label_id = int((_as_int_series(depth0, "label_id") <= 0).sum())
    report["depth1_2_3_label_id_not_minus1"] = depth_neighbor_bad_label_id
    report["depth0_label_id_not_positive"] = depth0_bad_label_id
    if depth_neighbor_bad_label_id:
        failures.append(f"depth1/2/3 label_id != -1: {depth_neighbor_bad_label_id}")
    if depth0_bad_label_id:
        failures.append(f"depth0 label_id <= 0: {depth0_bad_label_id}")

    source_conflicts = (
        depth0.groupby("source_fid")["label_id"].nunique().loc[lambda s: s.gt(1)]
        if "source_fid" in depth0.columns
        else pd.Series(dtype="int64")
    )
    report["depth0_source_fid_cross_label_conflicts"] = int(len(source_conflicts))
    if len(source_conflicts) > 0:
        failures.append(f"depth0 source_fid cross-label conflicts: {len(source_conflicts)}")

    depth0_sources = depth0.groupby("label_id")["source_fid"].agg(lambda s: set(int(v) for v in s))
    incomplete_source_labels = 0
    for row in labels[["label_id", "label_source_fids"]].itertuples(index=False):
        expected = _parse_ids_text(row.label_source_fids)
        actual = depth0_sources.get(int(row.label_id), set())
        if expected != actual:
            incomplete_source_labels += 1
    report["label_source_fids_in_input_depth0_incomplete"] = int(incomplete_source_labels)
    if incomplete_source_labels:
        failures.append(f"label source_fids not complete in input depth0: {incomplete_source_labels}")

    report["failures"] = failures
    report["all_ok"] = not failures
    if failures:
        raise RuntimeError("QA failed: " + "; ".join(failures))
    return report


def _write_gpkg(frame: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    temp = _tmp_path(path)
    if temp.exists():
        temp.unlink()
    pyogrio.write_dataframe(frame, temp, layer=layer, driver="GPKG")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and standardize council-derived WFS training data with building+UPRN anchor features."
    )
    parser.add_argument("--parcel-gpkg", default=str(DEFAULT_PARCEL_GPKG))
    parser.add_argument("--parcel-layer", default=PARCEL_LAYER)
    parser.add_argument("--input-gpkg", default=str(DEFAULT_INPUT_GPKG))
    parser.add_argument("--input-layer", default=INPUT_LAYER)
    parser.add_argument("--edge-cache", default=str(DEFAULT_EDGE_CACHE))
    parser.add_argument("--output-parcel-gpkg", default=str(DEFAULT_PARCEL_GPKG))
    parser.add_argument("--output-input-gpkg", default=str(DEFAULT_INPUT_GPKG))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--expected-label-rows", type=int, default=37900)
    parser.add_argument("--expected-input-rows", type=int, default=320910)
    parser.add_argument(
        "--summary-json",
        default=str(BASE_DIR / "sheffield_wfs_merged_council_train_dataset_summary.json"),
    )
    parser.add_argument(
        "--cleaning-summary-json",
        default=str(BASE_DIR / "sheffield_wfs_merged_council_train_cleaning_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parcel_gpkg = Path(args.parcel_gpkg)
    input_gpkg = Path(args.input_gpkg)
    output_parcel_gpkg = Path(args.output_parcel_gpkg)
    output_input_gpkg = Path(args.output_input_gpkg)
    edge_cache = Path(args.edge_cache)

    _log(f"[INFO] Reading labels: {parcel_gpkg}:{args.parcel_layer}")
    labels = gpd.read_file(parcel_gpkg, layer=str(args.parcel_layer), engine="pyogrio")
    _log(f"[INFO] Reading input molecules: {input_gpkg}:{args.input_layer}")
    inputs = gpd.read_file(input_gpkg, layer=str(args.input_layer), engine="pyogrio")

    valid_label_ids, anchors = _find_valid_labels(labels, inputs)
    _log(f"[INFO] Valid labels with exactly one building+UPRN anchor and complete depth0 sources: {len(valid_label_ids):,}")
    distances = _expand_depths(
        inputs=inputs,
        valid_label_ids=valid_label_ids,
        edge_cache_path=edge_cache,
        max_depth=int(args.max_depth),
    )

    standardized_labels = _standardize_labels(labels, valid_label_ids, anchors)
    standardized_inputs = _standardize_inputs(inputs, distances, valid_label_ids, anchors)
    qa_report = _qa(
        standardized_labels,
        standardized_inputs,
        expected_label_rows=int(args.expected_label_rows),
        expected_input_rows=int(args.expected_input_rows),
    )

    _log(f"[INFO] Writing standardized labels: {output_parcel_gpkg}:{PARCEL_LAYER}")
    _write_gpkg(standardized_labels, output_parcel_gpkg, PARCEL_LAYER)
    _log(f"[INFO] Writing standardized input molecules: {output_input_gpkg}:{INPUT_LAYER}")
    _write_gpkg(standardized_inputs, output_input_gpkg, INPUT_LAYER)

    summary = {
        "parcel_path": str(output_parcel_gpkg),
        "parcel_layer": PARCEL_LAYER,
        "input_path": str(output_input_gpkg),
        "input_layer": INPUT_LAYER,
        "source_parcel_path": str(parcel_gpkg),
        "source_input_path": str(input_gpkg),
        "edge_cache": str(edge_cache),
        "model_feature_policy": FEATURE_POLICY,
        "anchor_rule": "depth0 member with is_building_theme=1 and uprn_count>0; exactly one per label",
        "anchor_is_training_feature": True,
        "max_depth": int(args.max_depth),
        "label_rows": int(len(standardized_labels)),
        "input_rows": int(len(standardized_inputs)),
        "input_depth_counts": qa_report["input_depth_counts"],
        "global_building_uprn_anchor_rows": int(standardized_inputs["is_building_uprn_anchor"].sum()),
        "label_anchor_rows": int(standardized_inputs["is_building_label_anchor"].sum()),
        "nonanchor_uprn_rows": int(standardized_inputs["is_nonanchor_uprn"].sum()),
        "qa": qa_report,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cleaning_summary = {
        **summary,
        "original_label_rows": int(len(labels)),
        "clean_label_rows": int(len(standardized_labels)),
        "invalid_label_rows": int(len(labels) - len(standardized_labels)),
        "input_rows_before": int(len(inputs)),
        "input_rows_after": int(len(standardized_inputs)),
        "valid_seed_raw_clean_fids": int((standardized_inputs["neighbor_depth"].astype(int) == 0).sum()),
        "depth_gt0_label_id_sentinel_minus1": int(
            (
                (standardized_inputs["neighbor_depth"].astype(int) > 0)
                & (standardized_inputs["label_id"].astype(int) == -1)
            ).sum()
        ),
    }
    Path(args.cleaning_summary_json).write_text(json.dumps(cleaning_summary, indent=2), encoding="utf-8")

    _log("[DONE] Anchor-feature training dataset standardized")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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
import pyogrio
import shapely
from shapely.geometry import LineString

from apply_wfs_merge_completion_model import _write_layer
from prepare_wfs_merge_training_dataset import (
    DEFAULT_MERGE_LAYER,
    DEFAULT_UPRN_GPKG,
    DEFAULT_UPRN_LAYER,
    DEFAULT_WFS_GPKG,
    DEFAULT_WFS_LAYER,
    _candidate_pairs,
    _load_uprn_counts,
    _parts,
    _read_wfs,
    _shape_metrics,
)
from train_wfs_merge_completion_model import _shape_metrics as _parcel_shape_metrics
from train_wfs_merge_edge_model import _add_derived_features


DEFAULT_EDGE_MODEL_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1"
DEFAULT_REFERENCE_GPKG = ""
DEFAULT_OUTPUT_GPKG = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v1/"
    "model_predicted_polygons_edge_threshold_090_shape_guard.gpkg"
)
DEFAULT_EDGE_CANDIDATE_CSV = (
    "/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v1/"
    "edge_candidate_predictions_full.csv"
)


def _log(message: str) -> None:
    print(message, flush=True)


def _source_to_merge_table(merge: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for merge_fid, row in merge.iterrows():
        source_fids = _parts(row.get("merge_source_fids"))
        if not source_fids and pd.notna(row.get("source_fid")):
            source_fids = _parts(row.get("source_fid"))
        geometry = row.get("geometry") if "geometry" in row.index else None
        geometry_area = float(geometry.area) if geometry is not None and not pd.isna(geometry) else 0.0
        for source_fid in source_fids:
            records.append(
                {
                    "source_fid": int(source_fid),
                    "merge_fid": int(merge_fid),
                    "merge_area": float(row.get("merge_area", geometry_area)),
                    "merge_source_count": int(row.get("merge_source_count", len(source_fids) or 1) or 1),
                    "merge_stage": str(row.get("merge_stage", "") or ""),
                }
            )
    if not records:
        return pd.DataFrame(columns=["source_fid", "merge_fid", "merge_area", "merge_source_count", "merge_stage"])
    out = pd.DataFrame.from_records(records)
    out = out.sort_values(["source_fid", "merge_source_count", "merge_fid"])
    return out.drop_duplicates("source_fid", keep="first").reset_index(drop=True)


def _empty_semantic_reference(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "merge_fid": pd.Series(dtype="int64"),
            "merge_area": pd.Series(dtype="float64"),
            "merge_source_count": pd.Series(dtype="int64"),
            "merge_stage": pd.Series(dtype="object"),
            "merge_source_fids": pd.Series(dtype="object"),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _normalise_reference_columns(reference: pd.DataFrame, has_geometry: bool) -> pd.DataFrame:
    reference.index.name = "merge_fid"
    if has_geometry:
        reference = reference[reference.geometry.notna() & ~reference.geometry.is_empty].copy()
    else:
        reference = reference.copy()
    reference["merge_fid"] = reference.index.astype(int)
    reference.index.name = None
    if "merge_area" not in reference.columns:
        if has_geometry:
            reference["merge_area"] = reference.geometry.area.astype(float)
        else:
            reference["merge_area"] = 0.0
    if "merge_source_fids" not in reference.columns:
        reference["merge_source_fids"] = ""
    if "merge_source_count" not in reference.columns:
        reference["merge_source_count"] = reference["merge_source_fids"].map(lambda value: len(_parts(value)) or 1)
    if "merge_stage" not in reference.columns:
        reference["merge_stage"] = ""
    return reference


def _load_semantic_reference(path: str, layer: str, crs, *, load_geometry: bool = False) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    if not str(path or "").strip():
        _log("[INFO] Semantic reference disabled; production inference uses raw WFS + UPRN + model only.")
        return _empty_semantic_reference(crs), _source_to_merge_table(_empty_semantic_reference(crs))
    if not Path(path).exists():
        raise FileNotFoundError(f"Semantic reference GeoPackage does not exist: {path}")
    info = pyogrio.read_info(path, layer=layer)
    fields = {str(field) for field in info.get("fields", [])}
    feature_count = int(info.get("features") or 0)
    has_source_mapping = bool({"merge_source_fids", "source_fid"} & fields)
    if not load_geometry and not has_source_mapping:
        _log(
            f"[INFO] Semantic reference has no source mapping fields; "
            f"skipping geometry load (rows={feature_count:,})."
        )
        return _empty_semantic_reference(crs), _source_to_merge_table(_empty_semantic_reference(crs))

    if load_geometry:
        _log(f"[INFO] Reading semantic reference geometry: {path} ({layer})")
        reference = gpd.read_file(path, layer=layer, engine="pyogrio", fid_as_index=True)
        reference = _normalise_reference_columns(reference, has_geometry=True)
    else:
        columns = [
            column
            for column in ("merge_source_fids", "source_fid", "merge_area", "merge_source_count", "merge_stage")
            if column in fields
        ]
        _log(f"[INFO] Reading semantic reference attributes: {path} ({layer})")
        reference = gpd.read_file(
            path,
            layer=layer,
            engine="pyogrio",
            fid_as_index=True,
            ignore_geometry=True,
            columns=columns,
        )
        reference = _normalise_reference_columns(reference, has_geometry=False)
    labels = _source_to_merge_table(reference)
    _log(f"[INFO] Semantic reference rows={feature_count:,}; source labels={len(labels):,}")
    if not load_geometry:
        return _empty_semantic_reference(crs), labels
    return reference, labels


def _local_semantic_reference(reference: gpd.GeoDataFrame, sources: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    if reference.empty or "reference_merge_fid" not in sources.columns:
        return _empty_semantic_reference(crs)
    local_refs = set(sources["reference_merge_fid"].dropna().astype(int))
    if not local_refs:
        return _empty_semantic_reference(crs)
    return reference[reference["merge_fid"].astype(int).isin(local_refs)].copy()


def _lookup_series(labels: pd.DataFrame, source_ids: pd.Series, column: str, default: object) -> pd.Series:
    if labels.empty or column not in labels.columns:
        return pd.Series(default, index=source_ids.index)
    mapping = labels.set_index("source_fid")[column]
    return source_ids.astype(int).map(mapping).fillna(default)


def _gapfill_council_mask(wfs: gpd.GeoDataFrame) -> pd.Series:
    pieces = []
    for column in ("GmlID", "TOID"):
        if column in wfs.columns:
            pieces.append(wfs[column].fillna("").astype(str).str.lower())
    if not pieces:
        return pd.Series(False, index=wfs.index)
    text = pieces[0]
    for piece in pieces[1:]:
        text = text.str.cat(piece, sep="|")
    return text.str.contains("gapfill_council", regex=False)


def _add_pair_features_for_apply(
    pairs: pd.DataFrame,
    pool: gpd.GeoDataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    attrs = pool.set_index("source_fid")
    left = attrs.loc[pairs["pair_a"].to_numpy()]
    right = attrs.loc[pairs["pair_b"].to_numpy()]

    left_geoms = left.geometry.array
    right_geoms = right.geometry.array
    union_geoms = shapely.union(left_geoms, right_geoms)

    left_shape = _shape_metrics(left_geoms).add_prefix("left_")
    right_shape = _shape_metrics(right_geoms).add_prefix("right_")
    union_shape = _shape_metrics(union_geoms).add_prefix("union_")

    left_source = pairs["pair_a"].astype(int).reset_index(drop=True)
    right_source = pairs["pair_b"].astype(int).reset_index(drop=True)
    left_merge = _lookup_series(labels, left_source, "merge_fid", -1).astype(int)
    right_merge = _lookup_series(labels, right_source, "merge_fid", -1).astype(int)

    out = pd.DataFrame(
        {
            "left_source_fid": left_source.to_numpy(),
            "right_source_fid": right_source.to_numpy(),
            "label": ((left_merge.to_numpy() == right_merge.to_numpy()) & (left_merge.to_numpy() >= 0)).astype(int),
            "left_merge_fid": left_merge.to_numpy(),
            "right_merge_fid": right_merge.to_numpy(),
            "left_merge_area": _lookup_series(labels, left_source, "merge_area", 0.0).astype(float).to_numpy(),
            "right_merge_area": _lookup_series(labels, right_source, "merge_area", 0.0).astype(float).to_numpy(),
            "left_merge_source_count": _lookup_series(labels, left_source, "merge_source_count", 1).astype(int).to_numpy(),
            "right_merge_source_count": _lookup_series(labels, right_source, "merge_source_count", 1).astype(int).to_numpy(),
            "left_merge_stage": _lookup_series(labels, left_source, "merge_stage", "").astype(str).to_numpy(),
            "right_merge_stage": _lookup_series(labels, right_source, "merge_stage", "").astype(str).to_numpy(),
            "left_theme": left["Theme"].fillna("").astype(str).to_numpy(),
            "right_theme": right["Theme"].fillna("").astype(str).to_numpy(),
            "left_role": left["role"].astype(str).to_numpy(),
            "right_role": right["role"].astype(str).to_numpy(),
            "left_descriptive_group": left["DescriptiveGroup"].fillna("").astype(str).to_numpy(),
            "right_descriptive_group": right["DescriptiveGroup"].fillna("").astype(str).to_numpy(),
            "left_descriptive_term": left["DescriptiveTerm"].fillna("").astype(str).to_numpy(),
            "right_descriptive_term": right["DescriptiveTerm"].fillna("").astype(str).to_numpy(),
            "left_make": left["Make"].fillna("").astype(str).to_numpy(),
            "right_make": right["Make"].fillna("").astype(str).to_numpy(),
            "left_physical_level": left["PhysicalLevel"].fillna(-9999).astype(int).to_numpy(),
            "right_physical_level": right["PhysicalLevel"].fillna(-9999).astype(int).to_numpy(),
            "left_uprn_count": left["uprn_count"].fillna(0).astype(int).to_numpy(),
            "right_uprn_count": right["uprn_count"].fillna(0).astype(int).to_numpy(),
            "shared_edge_len": pairs["shared_edge_len"].astype(float).to_numpy(),
            "overlap_area": pairs["overlap_area"].astype(float).to_numpy(),
        }
    )
    out = pd.concat([out, left_shape, right_shape, union_shape], axis=1)
    small_area = out[["left_area", "right_area"]].min(axis=1)
    large_area = out[["left_area", "right_area"]].max(axis=1)
    small_perimeter = out[["left_perimeter", "right_perimeter"]].min(axis=1)
    out["small_area"] = small_area
    out["large_area"] = large_area
    out["small_large_area_ratio"] = small_area / large_area.replace(0.0, 1.0)
    out["shared_ratio_small_perimeter"] = out["shared_edge_len"] / small_perimeter.replace(0.0, 1.0)
    out["role_pair"] = out["left_role"] + "__" + out["right_role"]
    out["same_descriptive_group"] = (
        out["left_descriptive_group"].ne("") & out["left_descriptive_group"].eq(out["right_descriptive_group"])
    ).astype(int)
    out["same_descriptive_term"] = (
        out["left_descriptive_term"].ne("") & out["left_descriptive_term"].eq(out["right_descriptive_term"])
    ).astype(int)
    out["same_make"] = (out["left_make"].ne("") & out["left_make"].eq(out["right_make"])).astype(int)
    out["same_physical_level"] = out["left_physical_level"].eq(out["right_physical_level"]).astype(int)
    out["uprn_count_sum"] = out["left_uprn_count"] + out["right_uprn_count"]
    out["both_have_uprn"] = ((out["left_uprn_count"] > 0) & (out["right_uprn_count"] > 0)).astype(int)
    out["one_has_uprn"] = ((out["left_uprn_count"] > 0) ^ (out["right_uprn_count"] > 0)).astype(int)
    out["neither_has_uprn"] = ((out["left_uprn_count"] == 0) & (out["right_uprn_count"] == 0)).astype(int)
    out["mid_x"] = (shapely.get_x(shapely.centroid(left_geoms)) + shapely.get_x(shapely.centroid(right_geoms))) / 2.0
    out["mid_y"] = (shapely.get_y(shapely.centroid(left_geoms)) + shapely.get_y(shapely.centroid(right_geoms))) / 2.0
    return out


class _UnionFind:
    def __init__(self, values: pd.Series) -> None:
        self.parent = {int(v): int(v) for v in values.astype(int).tolist()}

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
    for row in edges.itertuples(index=False):
        uf.union(int(row.left_source_fid), int(row.right_source_fid))
    roots = source_ids.astype(int).map(uf.find)
    unique_roots = {root: idx + 1 for idx, root in enumerate(sorted(set(roots.astype(int))))}
    return roots.map(unique_roots).astype(int)


def _guard_edges(
    wfs: gpd.GeoDataFrame,
    selected_edges: pd.DataFrame,
    *,
    guard_threshold: float,
    min_component_mrr_ratio: float,
    max_component_hull_gap_ratio: float,
    keep_high_support_edges: bool,
    keep_min_proba: float,
    keep_min_shared_edge: float,
    keep_same_ref_zero_uprn_edges: bool,
    keep_same_ref_zero_uprn_min_proba: float,
    keep_same_ref_zero_uprn_min_shared_edge: float,
    keep_same_ref_zero_uprn_max_small_area: float,
    keep_same_ref_zero_uprn_max_large_area: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if selected_edges.empty:
        return selected_edges.copy(), {
            "guarded_components": 0,
            "removed_low_confidence_edges_in_guarded_components": 0,
            "kept_high_support_edges_in_guarded_components": 0,
            "kept_same_ref_zero_uprn_edges_in_guarded_components": 0,
        }

    component_ids = _component_ids(wfs["source_fid"], selected_edges)
    work = wfs[["source_fid", "geometry"]].copy()
    work["pred_component_id"] = component_ids.to_numpy()
    bad_components: set[int] = set()
    for comp_id, group in work.groupby("pred_component_id", sort=True):
        if len(group) <= 1:
            continue
        geom = shapely.union_all(group.geometry.array)
        shape = _parcel_shape_metrics(geom)
        if (
            float(shape["mrr_ratio"]) < float(min_component_mrr_ratio)
            or float(shape["hull_gap_ratio"]) > float(max_component_hull_gap_ratio)
        ):
            bad_components.add(int(comp_id))
    if not bad_components:
        return selected_edges.copy(), {
            "guarded_components": 0,
            "removed_low_confidence_edges_in_guarded_components": 0,
            "kept_high_support_edges_in_guarded_components": 0,
            "kept_same_ref_zero_uprn_edges_in_guarded_components": 0,
        }

    source_to_component = dict(zip(work["source_fid"].astype(int), work["pred_component_id"].astype(int)))
    edge_component = selected_edges["left_source_fid"].astype(int).map(source_to_component)
    remove_mask = edge_component.isin(bad_components) & selected_edges["model_proba"].lt(float(guard_threshold))
    keep_mask = pd.Series(False, index=selected_edges.index)
    same_ref_zero_uprn_keep_mask = pd.Series(False, index=selected_edges.index)
    if keep_high_support_edges and remove_mask.any():
        role_pair = selected_edges["role_pair"].fillna("").astype(str)
        is_building_land = role_pair.isin(["building__land", "land__building"])
        one_has_uprn = selected_edges.get("one_has_uprn", 0)
        if not isinstance(one_has_uprn, pd.Series):
            one_has_uprn = pd.Series(0, index=selected_edges.index)
        keep_mask = (
            remove_mask
            & is_building_land
            & one_has_uprn.fillna(0).astype(int).eq(1)
            & selected_edges["model_proba"].ge(float(keep_min_proba))
            & selected_edges["shared_edge_len"].ge(float(keep_min_shared_edge))
        )
    if keep_same_ref_zero_uprn_edges and remove_mask.any():
        role_pair = selected_edges["role_pair"].fillna("").astype(str)
        is_building_land = role_pair.isin(["building__land", "land__building"])
        neither_has_uprn = selected_edges.get("neither_has_uprn", 0)
        if not isinstance(neither_has_uprn, pd.Series):
            neither_has_uprn = pd.Series(0, index=selected_edges.index)
        same_reference = (
            selected_edges["left_merge_fid"].fillna(-1).astype(int)
            .eq(selected_edges["right_merge_fid"].fillna(-2).astype(int))
            & selected_edges["left_merge_fid"].fillna(-1).astype(int).ge(0)
        )
        small_area = selected_edges[["left_area", "right_area"]].min(axis=1).astype(float)
        large_area = selected_edges[["left_area", "right_area"]].max(axis=1).astype(float)
        same_ref_zero_uprn_keep_mask = (
            remove_mask
            & is_building_land
            & neither_has_uprn.fillna(0).astype(int).eq(1)
            & same_reference
            & selected_edges["model_proba"].ge(float(keep_same_ref_zero_uprn_min_proba))
            & selected_edges["shared_edge_len"].ge(float(keep_same_ref_zero_uprn_min_shared_edge))
            & small_area.le(float(keep_same_ref_zero_uprn_max_small_area))
            & large_area.le(float(keep_same_ref_zero_uprn_max_large_area))
        )
    combined_keep_mask = keep_mask | same_ref_zero_uprn_keep_mask
    remove_mask &= ~combined_keep_mask
    guarded = selected_edges[~remove_mask].copy()
    return guarded, {
        "guarded_components": int(len(bad_components)),
        "removed_low_confidence_edges_in_guarded_components": int(remove_mask.sum()),
        "kept_high_support_edges_in_guarded_components": int(keep_mask.sum()),
        "kept_same_ref_zero_uprn_edges_in_guarded_components": int(same_ref_zero_uprn_keep_mask.sum()),
    }


def _edge_lines(edges: pd.DataFrame, sources: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    columns = [
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
    if edges.empty:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=sources.crs)
    geom_by_source = sources.set_index(sources["source_fid"].astype(int)).geometry.to_dict()
    records: list[dict[str, Any]] = []
    for row in edges.itertuples(index=False):
        left_geom = geom_by_source[int(row.left_source_fid)]
        right_geom = geom_by_source[int(row.right_source_fid)]
        left_point = left_geom.representative_point()
        right_point = right_geom.representative_point()
        records.append(
            {
                "left_source_fid": int(row.left_source_fid),
                "right_source_fid": int(row.right_source_fid),
                "left_merge_fid": int(row.left_merge_fid),
                "right_merge_fid": int(row.right_merge_fid),
                "label": int(row.label),
                "model_proba": float(row.model_proba),
                "completion_proba": np.nan,
                "role_pair": str(row.role_pair),
                "pred_component_id": int(row.pred_component_id),
                "model_stage": "edge",
                "geometry": LineString(
                    [
                        (shapely.get_x(left_point), shapely.get_y(left_point)),
                        (shapely.get_x(right_point), shapely.get_y(right_point)),
                    ]
                ),
            }
        )
    return gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=sources.crs)


def _reference_values(group: gpd.GeoDataFrame) -> list[int]:
    return sorted({int(v) for v in group["reference_merge_fid"].dropna().astype(int)})


def _build_predicted_parcels(sources: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edge_stats: dict[int, dict[str, float]] = {}
    if not edges.empty:
        for comp_id, group in edges.groupby(edges["pred_component_id"].astype(int)):
            edge_stats[int(comp_id)] = {
                "predicted_edge_count": int(len(group)),
                "proba_min": float(group["model_proba"].min()),
                "proba_mean": float(group["model_proba"].mean()),
                "proba_max": float(group["model_proba"].max()),
            }

    records: list[dict[str, Any]] = []
    reference_by_component: dict[int, list[int]] = {}
    for comp_id, group in sources.groupby(sources["pred_component_id"].astype(int), sort=True):
        geom = shapely.union_all(group.geometry.array)
        shape = _parcel_shape_metrics(geom)
        refs = _reference_values(group)
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
            "pred_uprn_count": int(group["source_uprn_count"].fillna(0).astype(int).sum()),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the first-stage WFS merge edge model to raw WFS polygons.")
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--merge-gpkg", "--reference-gpkg", "--council-land-gpkg", dest="merge_gpkg", default=DEFAULT_REFERENCE_GPKG)
    parser.add_argument("--merge-layer", default=DEFAULT_MERGE_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default="UPRN")
    parser.add_argument("--edge-model-dir", default=DEFAULT_EDGE_MODEL_DIR)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--edge-candidate-csv", default=DEFAULT_EDGE_CANDIDATE_CSV)
    parser.add_argument("--comfort-zone", choices=["none", "reference-small-uprn"], default="none")
    parser.add_argument("--comfort-max-reference-area", type=float, default=2000.0)
    parser.add_argument("--comfort-max-reference-source-count", type=int, default=20)
    parser.add_argument("--comfort-min-uprn", type=int, default=1)
    parser.add_argument("--comfort-max-uprn", type=int, default=2)
    parser.add_argument("--comfort-require-building-land", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--guard-threshold", type=float, default=0.95)
    parser.add_argument("--min-component-mrr-ratio", type=float, default=0.90)
    parser.add_argument("--max-component-hull-gap-ratio", type=float, default=0.10)
    parser.add_argument("--guard-keep-high-support-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guard-keep-min-proba", type=float, default=0.93)
    parser.add_argument("--guard-keep-min-shared-edge", type=float, default=8.0)
    parser.add_argument("--guard-keep-same-ref-zero-uprn-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guard-keep-same-ref-zero-uprn-min-proba", type=float, default=0.90)
    parser.add_argument("--guard-keep-same-ref-zero-uprn-min-shared-edge", type=float, default=6.0)
    parser.add_argument("--guard-keep-same-ref-zero-uprn-max-small-area", type=float, default=80.0)
    parser.add_argument("--guard-keep-same-ref-zero-uprn-max-large-area", type=float, default=500.0)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--max-overlap-area", type=float, default=1e-6)
    parser.add_argument("--include-terms", default="building,land")
    parser.add_argument("--exclude-gapfill-council", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exclude-problem-gapfill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--problem-gapfill-max-uprn", type=int, default=2)
    parser.add_argument("--problem-gapfill-min-mrr-ratio", type=float, default=0.75)
    parser.add_argument("--problem-gapfill-max-hull-gap-ratio", type=float, default=0.25)
    parser.add_argument("--no-shape-guard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_gpkg = Path(args.output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    edge_model_dir = Path(args.edge_model_dir)

    include_terms = {term.strip().lower() for term in str(args.include_terms or "").split(",") if term.strip()}
    wfs = _read_wfs(args.wfs_gpkg, args.wfs_layer, include_terms, True)
    wfs["uprn_count"] = _load_uprn_counts(
        args.uprn_gpkg,
        args.uprn_layer,
        args.uprn_id_field,
        wfs,
        True,
    ).to_numpy()
    semantic_reference, labels = _load_semantic_reference(
        args.merge_gpkg,
        args.merge_layer,
        wfs.crs,
        load_geometry=args.comfort_zone == "reference-small-uprn",
    )
    reference_enabled = bool(str(args.merge_gpkg or "").strip())
    label_lookup = labels.set_index("source_fid") if not labels.empty else pd.DataFrame()
    wfs["reference_merge_fid"] = wfs["source_fid"].astype(int).map(label_lookup["merge_fid"] if not label_lookup.empty else {})
    wfs["is_semantic_source"] = wfs["reference_merge_fid"].notna().astype(int)
    wfs["source_uprn_count"] = wfs["uprn_count"].fillna(0).astype(int)
    excluded_problem_sources = gpd.GeoDataFrame(geometry=[], crs=wfs.crs)
    excluded_gapfill_council_sources = gpd.GeoDataFrame(geometry=[], crs=wfs.crs)

    gapfill_mask = _gapfill_council_mask(wfs)
    if bool(args.exclude_gapfill_council) and bool(gapfill_mask.any()):
        excluded_gapfill_council_sources = wfs[gapfill_mask].copy()
        excluded_gapfill_council_sources["exclude_reason"] = "gapfill_council_non_native_source"
        wfs = wfs[~gapfill_mask].copy()
        labels = labels[labels["source_fid"].astype(int).isin(set(wfs["source_fid"].astype(int)))].copy()
        _log(
            "[INFO] Excluded gapfill_council non-native sources="
            f"{len(excluded_gapfill_council_sources):,}; "
            f"area={float(excluded_gapfill_council_sources.geometry.area.sum()):.2f}"
        )

    if bool(args.exclude_problem_gapfill):
        shape = _shape_metrics(wfs.geometry.array)
        gapfill_mask = _gapfill_council_mask(wfs)
        problem_shape = (
            shape["mrr_ratio"].lt(float(args.problem_gapfill_min_mrr_ratio))
            | shape["hull_gap_ratio"].gt(float(args.problem_gapfill_max_hull_gap_ratio))
        )
        problem_gapfill = (
            gapfill_mask
            & wfs["source_uprn_count"].gt(int(args.problem_gapfill_max_uprn))
            & problem_shape.to_numpy()
        )
        if bool(problem_gapfill.any()):
            excluded_problem_sources = wfs[problem_gapfill].copy()
            excluded_problem_sources["exclude_reason"] = "problem_gapfill_multi_uprn_bad_shape"
            excluded_problem_sources["gapfill_mrr_ratio"] = shape.loc[problem_gapfill.to_numpy(), "mrr_ratio"].to_numpy()
            excluded_problem_sources["gapfill_hull_gap_ratio"] = shape.loc[
                problem_gapfill.to_numpy(), "hull_gap_ratio"
            ].to_numpy()
            wfs = wfs[~problem_gapfill].copy()
            labels = labels[labels["source_fid"].astype(int).isin(set(wfs["source_fid"].astype(int)))].copy()
            _log(f"[INFO] Excluded problem gapfill sources={len(excluded_problem_sources):,}")

    if args.comfort_zone == "reference-small-uprn":
        if not reference_enabled or semantic_reference.empty or labels.empty:
            raise ValueError("--comfort-zone reference-small-uprn requires --merge-gpkg/--reference-gpkg.")
        _log("[INFO] Applying reference-small-UPRN comfort-zone filter")
        labelled = wfs[wfs["reference_merge_fid"].notna()].copy()
        ref_uprn = labelled.groupby(labelled["reference_merge_fid"].astype(int))["source_uprn_count"].sum()
        role_counts = labelled.assign(
            is_building=labelled["role"].astype(str).str.lower().eq("building").astype(int),
            is_land=labelled["role"].astype(str).str.lower().eq("land").astype(int),
        ).groupby(labelled["reference_merge_fid"].astype(int))[["is_building", "is_land"]].sum()
        reference_lookup = semantic_reference.set_index("merge_fid")
        comfort_mask = (
            reference_lookup["merge_area"].astype(float).le(float(args.comfort_max_reference_area))
            & reference_lookup["merge_source_count"].astype(int).le(int(args.comfort_max_reference_source_count))
            & reference_lookup.index.to_series().map(ref_uprn).fillna(0).astype(int).between(
                int(args.comfort_min_uprn),
                int(args.comfort_max_uprn),
            )
        )
        if bool(args.comfort_require_building_land):
            comfort_mask &= (
                reference_lookup.index.to_series().map(role_counts["is_building"]).fillna(0).astype(int).gt(0)
                & reference_lookup.index.to_series().map(role_counts["is_land"]).fillna(0).astype(int).gt(0)
            )
        comfort_refs = set(reference_lookup.index[comfort_mask].astype(int))
        before_wfs = len(wfs)
        before_ref = len(semantic_reference)
        wfs = wfs[wfs["reference_merge_fid"].fillna(-1).astype(int).isin(comfort_refs)].copy()
        labels = labels[labels["merge_fid"].astype(int).isin(comfort_refs)].copy()
        semantic_reference = semantic_reference[semantic_reference["merge_fid"].astype(int).isin(comfort_refs)].copy()
        _log(
            "[INFO] Comfort zone kept "
            f"refs={len(semantic_reference):,}/{before_ref:,}; "
            f"sources={len(wfs):,}/{before_wfs:,}; labels={len(labels):,}"
        )

    pairs = _candidate_pairs(
        wfs,
        wfs,
        min_shared_edge=float(args.min_shared_edge),
        max_overlap_area=float(args.max_overlap_area),
        verbose=True,
    )
    if pairs.empty:
        raise RuntimeError("No edge candidates were generated.")
    _log("[INFO] Building edge feature matrix")
    edge_df = _add_pair_features_for_apply(pairs, wfs, labels)
    edge_df = _add_derived_features(edge_df)

    edge_model = joblib.load(edge_model_dir / "wfs_merge_edge_model_v1.joblib")
    edge_meta = json.loads((edge_model_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_cols = edge_meta["feature_columns"]
    missing = sorted(set(feature_cols) - set(edge_df.columns))
    if missing:
        raise RuntimeError(f"Edge candidates are missing model features: {missing}")
    edge_df["model_proba"] = edge_model.predict_proba(edge_df[feature_cols])[:, 1]
    edge_candidate_csv = Path(args.edge_candidate_csv)
    edge_candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[INFO] Writing full edge candidate CSV: {edge_candidate_csv}")
    edge_df.to_csv(edge_candidate_csv, index=False)
    initial_edges = edge_df[edge_df["model_proba"].ge(float(args.threshold))].copy()
    _log(f"[INFO] Candidate edges={len(edge_df):,}; initial positive edges={len(initial_edges):,}")

    if bool(args.no_shape_guard):
        selected_edges = initial_edges.copy()
        guard_summary = {"guarded_components": 0, "removed_low_confidence_edges_in_guarded_components": 0}
    else:
        selected_edges, guard_summary = _guard_edges(
            wfs,
            initial_edges,
            guard_threshold=float(args.guard_threshold),
            min_component_mrr_ratio=float(args.min_component_mrr_ratio),
            max_component_hull_gap_ratio=float(args.max_component_hull_gap_ratio),
        keep_high_support_edges=bool(args.guard_keep_high_support_edges),
        keep_min_proba=float(args.guard_keep_min_proba),
        keep_min_shared_edge=float(args.guard_keep_min_shared_edge),
        keep_same_ref_zero_uprn_edges=bool(args.guard_keep_same_ref_zero_uprn_edges),
        keep_same_ref_zero_uprn_min_proba=float(args.guard_keep_same_ref_zero_uprn_min_proba),
        keep_same_ref_zero_uprn_min_shared_edge=float(args.guard_keep_same_ref_zero_uprn_min_shared_edge),
        keep_same_ref_zero_uprn_max_small_area=float(args.guard_keep_same_ref_zero_uprn_max_small_area),
        keep_same_ref_zero_uprn_max_large_area=float(args.guard_keep_same_ref_zero_uprn_max_large_area),
    )
    _log(f"[INFO] Positive edges after guard={len(selected_edges):,}; guard={guard_summary}")

    component_ids = _component_ids(wfs["source_fid"], selected_edges)
    sources = wfs.copy()
    sources["pred_component_id"] = component_ids.to_numpy()
    selected_edges = selected_edges.copy()
    source_to_component = dict(zip(sources["source_fid"].astype(int), sources["pred_component_id"].astype(int)))
    selected_edges["pred_component_id"] = selected_edges["left_source_fid"].astype(int).map(source_to_component).astype(int)
    edges = _edge_lines(selected_edges, sources)
    predicted = _build_predicted_parcels(sources, edges)
    merged_only = predicted[predicted["source_count"].gt(1)].copy()
    possible_fp = predicted[predicted["possible_false_positive_cluster"].eq(1)].copy()
    possible_split = predicted[predicted["possible_split_reference"].eq(1)].copy()
    semantic_reference_output = _local_semantic_reference(semantic_reference, sources, wfs.crs)
    if len(semantic_reference_output) != len(semantic_reference):
        _log(
            "[INFO] Local semantic reference rows="
            f"{len(semantic_reference_output):,}/{len(semantic_reference):,}"
        )

    if output_gpkg.exists():
        output_gpkg.unlink()
    _log(f"[INFO] Writing output: {output_gpkg}")
    _write_layer(predicted.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels")
    _write_layer(merged_only.drop(columns=["pred_uprn_count"]), output_gpkg, "predicted_parcels_merged_only")
    _write_layer(semantic_reference_output, output_gpkg, "semantic_reference_parcels")
    _write_layer(sources, output_gpkg, "prediction_source_polygons")
    _write_layer(excluded_gapfill_council_sources, output_gpkg, "excluded_gapfill_council_sources")
    _write_layer(excluded_problem_sources, output_gpkg, "excluded_problem_sources")
    _write_layer(edges, output_gpkg, "predicted_positive_edges")
    _write_layer(possible_fp.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_false_positive_clusters")
    _write_layer(possible_split.drop(columns=["pred_uprn_count"]), output_gpkg, "possible_split_reference_clusters")
    _write_layer(predicted, output_gpkg, "predicted_parcels_with_uprn")
    _write_layer(merged_only, output_gpkg, "predicted_parcels_merged_only_with_uprn")
    _write_layer(possible_fp, output_gpkg, "possible_false_positive_clusters_with_uprn")
    _write_layer(possible_split, output_gpkg, "possible_split_reference_clusters_with_uprn")
    _write_layer(merged_only[merged_only["pred_uprn_count"].le(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_le1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].eq(1)].copy(), output_gpkg, "predicted_parcels_merged_only_uprn_eq1")
    _write_layer(merged_only[merged_only["pred_uprn_count"].gt(1)].copy(), output_gpkg, "predicted_parcels_merged_only_multi_uprn")

    summary = {
        "wfs_gpkg": str(args.wfs_gpkg),
        "wfs_layer": str(args.wfs_layer),
        "merge_gpkg": str(args.merge_gpkg),
        "merge_layer": str(args.merge_layer),
        "reference_enabled": bool(reference_enabled),
        "semantic_reference_rows": int(len(semantic_reference_output)),
        "semantic_reference_loaded_rows": int(len(semantic_reference)),
        "semantic_reference_source_labels": int(len(labels)),
        "output_gpkg": str(output_gpkg),
        "edge_candidate_csv": str(edge_candidate_csv),
        "comfort_zone": str(args.comfort_zone),
        "comfort_max_reference_area": float(args.comfort_max_reference_area),
        "comfort_max_reference_source_count": int(args.comfort_max_reference_source_count),
        "comfort_min_uprn": int(args.comfort_min_uprn),
        "comfort_max_uprn": int(args.comfort_max_uprn),
        "comfort_require_building_land": bool(args.comfort_require_building_land),
        "threshold": float(args.threshold),
        "guard_threshold": float(args.guard_threshold),
        "min_component_mrr_ratio": float(args.min_component_mrr_ratio),
        "max_component_hull_gap_ratio": float(args.max_component_hull_gap_ratio),
        "guard_keep_high_support_edges": bool(args.guard_keep_high_support_edges),
        "guard_keep_min_proba": float(args.guard_keep_min_proba),
        "guard_keep_min_shared_edge": float(args.guard_keep_min_shared_edge),
        "guard_keep_same_ref_zero_uprn_edges": bool(args.guard_keep_same_ref_zero_uprn_edges),
        "guard_keep_same_ref_zero_uprn_min_proba": float(args.guard_keep_same_ref_zero_uprn_min_proba),
        "guard_keep_same_ref_zero_uprn_min_shared_edge": float(
            args.guard_keep_same_ref_zero_uprn_min_shared_edge
        ),
        "guard_keep_same_ref_zero_uprn_max_small_area": float(
            args.guard_keep_same_ref_zero_uprn_max_small_area
        ),
        "guard_keep_same_ref_zero_uprn_max_large_area": float(
            args.guard_keep_same_ref_zero_uprn_max_large_area
        ),
        "exclude_gapfill_council": bool(args.exclude_gapfill_council),
        "excluded_gapfill_council_sources": int(len(excluded_gapfill_council_sources)),
        "excluded_gapfill_council_area_m2": float(excluded_gapfill_council_sources.geometry.area.sum())
        if len(excluded_gapfill_council_sources)
        else 0.0,
        "excluded_gapfill_council_uprn_count": int(
            excluded_gapfill_council_sources["source_uprn_count"].fillna(0).astype(int).sum()
        )
        if len(excluded_gapfill_council_sources) and "source_uprn_count" in excluded_gapfill_council_sources.columns
        else 0,
        "exclude_problem_gapfill": bool(args.exclude_problem_gapfill),
        "problem_gapfill_max_uprn": int(args.problem_gapfill_max_uprn),
        "problem_gapfill_min_mrr_ratio": float(args.problem_gapfill_min_mrr_ratio),
        "problem_gapfill_max_hull_gap_ratio": float(args.problem_gapfill_max_hull_gap_ratio),
        "excluded_problem_sources": int(len(excluded_problem_sources)),
        "candidate_edges": int(len(edge_df)),
        "initial_threshold_edges": int(len(initial_edges)),
        "predicted_positive_edges": int(len(selected_edges)),
        **guard_summary,
        "graph_nodes": int(len(sources)),
        "predicted_components": int(len(predicted)),
        "merged_only_polygons": int(len(merged_only)),
        "possible_false_positive_clusters": int(len(possible_fp)),
        "possible_split_reference_clusters": int(len(possible_split)),
        "merged_only_uprn_counts": merged_only["pred_uprn_count"].value_counts().sort_index().astype(int).to_dict(),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[DONE] Edge apply complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
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


DEFAULT_WFS_CLEAN_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_CLEAN_LAYER = "wfs_raw_clean"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_UPRN_ID_FIELD = "UPRN"
DEFAULT_TARGET_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_merged_council_train.gpkg"
DEFAULT_TARGET_LAYER = "wfs_raw_merged_council_train_merged_only"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_completeness_v2"

MODEL_FILE_NAME = "wfs_raw_anchor_group_model_v1.joblib"
CANDIDATES_FILE_NAME = "wfs_raw_anchor_group_candidates_v1.csv"
PREDICTIONS_FILE_NAME = "wfs_raw_anchor_group_predictions_v1.csv"
METRICS_FILE_NAME = "wfs_raw_anchor_group_metrics_v1.json"

TARGET_COL = "label"
CATEGORICAL_FEATURES = ["anchor_role", "role_signature"]
PROPOSAL_ROLE_ORDER = ("building", "land", "road", "gapfill", "other")
ID_COLUMNS = {
    "anchor_source_fid",
    "anchor_clean_fids",
    "candidate_clean_fids",
    "candidate_source_fids",
    "target_source_fids",
    "target_train_component_id",
    "label_source",
}
LABEL_DERIVED_MARKERS = ("target_", "source_target_", "label")
MODEL_OUTPUT_COLUMNS = {
    "raw_anchor_group_proba",
    "raw_anchor_group_pred_at_threshold",
}


@dataclass
class RawAnchorBuildContext:
    target: gpd.GeoDataFrame
    source_to_clean: dict[int, list[int]]
    source_by_clean: dict[int, int]
    eligible_clean_ids: set[int]
    adjacency: dict[int, list[tuple[int, float]]]
    adjacency_lookup: dict[int, dict[int, float]]
    shared_by_pair: dict[tuple[int, int], float]
    geom_by_clean: dict[int, Any]
    attrs_by_clean: dict[int, dict[str, Any]]
    shape_by_clean: dict[int, dict[str, float]]
    area_by_clean: dict[int, float]
    perimeter_by_clean: dict[int, float]
    cheap_attrs_by_clean: dict[int, tuple[int | None, float, float, int, str]]
    uprn_clean_ids: set[int]
    building_clean_ids: set[int]
    proposal_pipeline: Any | None = None
    proposal_feature_cols: list[str] | None = None


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _ids_text(values: set[int] | frozenset[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(value)) for value in sorted(int(v) for v in values))


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


def _parse_int_list(value: str) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _theme_text(rows: pd.DataFrame) -> pd.Series:
    pieces: list[pd.Series] = []
    for column in ["Theme", "DescriptiveGroup", "DescriptiveTerm", "raw_role"]:
        if column in rows.columns:
            pieces.append(rows[column].fillna("").astype(str))
    if not pieces:
        return pd.Series("", index=rows.index)
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.str.cat(piece, sep=" ")
    return out.str.lower()


def _theme_only_text(rows: pd.DataFrame) -> pd.Series:
    if "Theme" not in rows.columns:
        return pd.Series("", index=rows.index)
    return rows["Theme"].fillna("").astype(str).str.lower()


def _role_from_text(text: str) -> str:
    value = str(text or "").lower()
    if "building" in value:
        return "building"
    if "land" in value:
        return "land"
    if "road" in value or "track" in value or "path" in value:
        return "road"
    if "gap" in value or "hole" in value:
        return "gapfill"
    return "other"


def _as_valid(values: Any) -> Any:
    valid = shapely.is_valid(values)
    if bool(np.all(valid)):
        return values
    out = np.asarray(values, dtype=object).copy()
    out[~np.asarray(valid, dtype=bool)] = shapely.make_valid(out[~np.asarray(valid, dtype=bool)])
    return out


def _union(geoms: list[Any]) -> Any:
    if not geoms:
        return None
    if len(geoms) == 1:
        geom = geoms[0]
        if geom is None or geom.is_empty:
            return geom
        return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is None or geom.is_empty:
        return geom
    return shapely.make_valid(geom) if not bool(shapely.is_valid(geom)) else geom


def _shape_for_clean(
    clean_fid: int,
    *,
    geom_by_clean: dict[int, Any],
    shape_by_clean: dict[int, dict[str, float]],
) -> dict[str, float]:
    clean_fid = int(clean_fid)
    cached = shape_by_clean.get(clean_fid)
    if cached is not None:
        return cached
    geom = geom_by_clean.get(clean_fid)
    if geom is None or geom.is_empty:
        raise ValueError("Empty clean geometry")
    shape = _shape_metrics(geom)
    shape_by_clean[clean_fid] = shape
    return shape


def _shape_for_group(
    clean_ids: tuple[int, ...] | frozenset[int],
    *,
    geom_by_clean: dict[int, Any],
    shape_by_clean: dict[int, dict[str, float]],
    shape_by_group: dict[frozenset[int], dict[str, float]] | None = None,
) -> dict[str, float]:
    ids = tuple(sorted(int(v) for v in clean_ids))
    if not ids:
        raise ValueError("Empty candidate geometry")
    if len(ids) == 1:
        return _shape_for_clean(ids[0], geom_by_clean=geom_by_clean, shape_by_clean=shape_by_clean)
    key = frozenset(ids)
    if shape_by_group is not None:
        cached = shape_by_group.get(key)
        if cached is not None:
            return cached
    geom = _union([geom_by_clean[fid] for fid in ids if fid in geom_by_clean])
    if geom is None or geom.is_empty:
        raise ValueError("Empty candidate geometry")
    shape = _shape_metrics(geom)
    if shape_by_group is not None:
        shape_by_group[key] = shape
    return shape


def _build_proposal_cheap_context(
    *,
    source_by_clean: dict[int, int],
    attrs_by_clean: dict[int, dict[str, Any]],
    area_by_clean: dict[int, float],
    perimeter_by_clean: dict[int, float],
) -> tuple[dict[int, tuple[int | None, float, float, int, str]], set[int], set[int]]:
    cheap_attrs_by_clean: dict[int, tuple[int | None, float, float, int, str]] = {}
    uprn_clean_ids: set[int] = set()
    building_clean_ids: set[int] = set()
    all_ids = set(int(v) for v in area_by_clean)
    all_ids.update(int(v) for v in perimeter_by_clean)
    all_ids.update(int(v) for v in attrs_by_clean)
    all_ids.update(int(v) for v in source_by_clean)
    for clean_fid in all_ids:
        attrs = attrs_by_clean.get(int(clean_fid), {})
        role = str(attrs.get("anchor_role", "other") or "other")
        if role not in PROPOSAL_ROLE_ORDER:
            role = "other"
        uprn_count = int(attrs.get("uprn_count", 0) or 0)
        if uprn_count > 0:
            uprn_clean_ids.add(int(clean_fid))
        if role == "building":
            building_clean_ids.add(int(clean_fid))
        source_fid = source_by_clean.get(int(clean_fid))
        cheap_attrs_by_clean[int(clean_fid)] = (
            int(source_fid) if source_fid is not None else None,
            float(area_by_clean.get(int(clean_fid), 0.0)),
            float(perimeter_by_clean.get(int(clean_fid), 0.0)),
            int(uprn_count),
            role,
        )
    return cheap_attrs_by_clean, uprn_clean_ids, building_clean_ids


def _read_clean_wfs(path: Path, layer: str, bbox: tuple[float, float, float, float] | None) -> gpd.GeoDataFrame:
    kwargs: dict[str, Any] = {"layer": layer, "fid_as_index": True}
    if bbox is not None:
        kwargs["bbox"] = bbox
    _log(f"[INFO] Reading clean WFS: {path}:{layer}")
    wfs = pyogrio.read_dataframe(path, **kwargs)
    wfs = wfs[wfs.geometry.notna() & ~wfs.geometry.is_empty].copy()
    wfs.index = wfs.index.astype(int)
    wfs["clean_fid"] = wfs.index.astype("int64")
    if "source_fid" not in wfs.columns:
        wfs["source_fid"] = wfs["clean_fid"]
    wfs["source_fid"] = pd.to_numeric(wfs["source_fid"], errors="coerce")
    wfs = wfs[wfs["source_fid"].notna()].copy()
    wfs["source_fid"] = wfs["source_fid"].astype("int64")
    for column in ["Theme", "DescriptiveGroup", "DescriptiveTerm", "raw_role"]:
        if column not in wfs.columns:
            wfs[column] = ""
    wfs.geometry = _as_valid(wfs.geometry.to_numpy())
    wfs["area"] = wfs.geometry.area.astype("float64")
    wfs["perimeter"] = wfs.geometry.length.astype("float64")
    wfs = wfs[wfs["area"].gt(0.0)].copy()
    theme_text = _theme_only_text(wfs)
    wfs["anchor_role"] = theme_text.map(_role_from_text)
    hole_fill = wfs.get("is_polygon_hole_fill", pd.Series(0, index=wfs.index)).fillna(0).astype(int).gt(0)
    gap_fill = wfs.get("is_enclosed_gap_fill", pd.Series(0, index=wfs.index)).fillna(0).astype(int).gt(0)
    wfs["plot_eligible"] = theme_text.str.contains("building", regex=False) | theme_text.str.contains("land", regex=False)
    wfs["is_polygon_hole_fill"] = hole_fill.astype(int)
    wfs["is_enclosed_gap_fill"] = gap_fill.astype(int)
    _log(
        "[INFO] Clean WFS rows="
        f"{len(wfs):,}; plot_eligible={int(wfs['plot_eligible'].sum()):,}; "
        f"roles={wfs['anchor_role'].value_counts().to_dict()}"
    )
    return wfs


def _add_uprn_counts(
    wfs: gpd.GeoDataFrame,
    *,
    uprn_gpkg: Path,
    uprn_layer: str,
    uprn_id_field: str,
) -> gpd.GeoDataFrame:
    if wfs.empty:
        wfs["uprn_count"] = pd.Series(dtype="int64")
        return wfs
    bounds = tuple(float(v) for v in wfs.total_bounds)
    _log(f"[INFO] Reading UPRN points in clean WFS bounds: {uprn_gpkg}:{uprn_layer}")
    points = pyogrio.read_dataframe(uprn_gpkg, layer=uprn_layer, columns=[uprn_id_field], bbox=bounds)
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if points.crs != wfs.crs:
        points = points.to_crs(wfs.crs)
    _log(f"[INFO] UPRN points in bounds={len(points):,}")
    if points.empty:
        wfs["uprn_count"] = 0
        return wfs
    joined = gpd.sjoin(
        points[[uprn_id_field, "geometry"]],
        wfs[["clean_fid", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    counts = joined.groupby("clean_fid")[uprn_id_field].nunique() if not joined.empty else pd.Series(dtype="int64")
    out = wfs.copy()
    out["uprn_count"] = out["clean_fid"].map(counts).fillna(0).astype("int64")
    _log(
        "[INFO] Clean WFS UPRN intersections="
        f"{int(out['uprn_count'].gt(0).sum()):,} polygons; total_uprn={int(out['uprn_count'].sum()):,}"
    )
    return out


def _context_cache_signature(wfs: gpd.GeoDataFrame) -> dict[str, Any]:
    if wfs.empty:
        return {
            "rows": 0,
            "clean_fid_min": None,
            "clean_fid_max": None,
            "clean_fid_sum": 0,
            "bounds": [],
        }
    clean_fids = wfs["clean_fid"].astype("int64")
    return {
        "rows": int(len(wfs)),
        "clean_fid_min": int(clean_fids.min()),
        "clean_fid_max": int(clean_fids.max()),
        "clean_fid_sum": int(clean_fids.sum()),
        "bounds": [round(float(value), 3) for value in wfs.total_bounds],
    }


def _context_cache_path(cache_dir: str | Path, prefix: str, metadata: dict[str, Any]) -> Path:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    return Path(cache_dir) / f"{prefix}_{digest}.joblib"


def _add_uprn_counts_cached(
    wfs: gpd.GeoDataFrame,
    *,
    uprn_gpkg: Path,
    uprn_layer: str,
    uprn_id_field: str,
    context_cache_dir: str | Path = "",
) -> gpd.GeoDataFrame:
    if not str(context_cache_dir).strip():
        return _add_uprn_counts(
            wfs,
            uprn_gpkg=uprn_gpkg,
            uprn_layer=uprn_layer,
            uprn_id_field=uprn_id_field,
        )
    cache_dir = Path(context_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "uprn_counts",
        "wfs_signature": _context_cache_signature(wfs),
        "uprn_gpkg": str(uprn_gpkg),
        "uprn_layer": str(uprn_layer),
        "uprn_id_field": str(uprn_id_field),
    }
    cache_path = _context_cache_path(cache_dir, "uprn_counts", metadata)
    if cache_path.exists():
        payload = joblib.load(cache_path)
        if isinstance(payload, dict) and payload.get("metadata") == metadata:
            counts = {int(key): int(value) for key, value in dict(payload.get("counts", {})).items()}
            out = wfs.copy()
            out["uprn_count"] = out["clean_fid"].astype(int).map(counts).fillna(0).astype("int64")
            _log(f"[INFO] Loaded cached UPRN counts: path={cache_path}")
            return out
    out = _add_uprn_counts(
        wfs,
        uprn_gpkg=uprn_gpkg,
        uprn_layer=uprn_layer,
        uprn_id_field=uprn_id_field,
    )
    counts = {
        int(row.clean_fid): int(row.uprn_count)
        for row in out.loc[out["uprn_count"].astype(int).gt(0), ["clean_fid", "uprn_count"]].itertuples(index=False)
    }
    joblib.dump({"metadata": metadata, "counts": counts}, cache_path, compress=3)
    _log(f"[INFO] Wrote cached UPRN counts: path={cache_path}")
    return out


def _read_targets(
    path: Path,
    layer: str,
    bbox: tuple[float, float, float, float] | None,
    max_target_rows: int,
) -> gpd.GeoDataFrame:
    kwargs: dict[str, Any] = {"layer": layer, "fid_as_index": True}
    if bbox is not None:
        kwargs["bbox"] = bbox
    _log(f"[INFO] Reading source-target labels: {path}:{layer}")
    target = pyogrio.read_dataframe(path, **kwargs)
    target = target[target.geometry.notna() & ~target.geometry.is_empty].copy()
    target.index = target.index.astype(int)
    target["target_fid"] = target.index.astype("int64")
    if int(max_target_rows) > 0:
        target = target.head(int(max_target_rows)).copy()
        _log(f"[INFO] Max target rows applied: {len(target):,}")
    required = {"source_wfs_fids", "anchor_wfs_fid", "train_component_id"}
    missing = required - set(target.columns)
    if missing:
        raise RuntimeError(f"Target layer missing required columns: {sorted(missing)}")
    target["target_source_set"] = target["source_wfs_fids"].map(_parse_id_set)
    target["anchor_source_fid"] = pd.to_numeric(target["anchor_wfs_fid"], errors="coerce")
    target = target[target["anchor_source_fid"].notna()].copy()
    target["anchor_source_fid"] = target["anchor_source_fid"].astype("int64")
    target = target[target["target_source_set"].map(len).ge(2)].copy()
    _log(f"[INFO] Source-target rows after cleanup={len(target):,}")
    return target


def _build_source_indexes(wfs: gpd.GeoDataFrame) -> tuple[dict[int, list[int]], dict[int, int]]:
    source_to_clean: dict[int, list[int]] = {}
    source_by_clean: dict[int, int] = {}
    for row in wfs[["clean_fid", "source_fid"]].itertuples(index=False):
        clean_fid = int(row.clean_fid)
        source_fid = int(row.source_fid)
        source_to_clean.setdefault(source_fid, []).append(clean_fid)
        source_by_clean[clean_fid] = source_fid
    for source_fid, values in list(source_to_clean.items()):
        source_to_clean[int(source_fid)] = sorted(int(v) for v in values)
    return source_to_clean, source_by_clean


def _build_edges(
    nodes: gpd.GeoDataFrame,
    *,
    min_shared_edge: float,
    top_neighbors: int,
    query_chunk_size: int,
    edge_calc_chunk_size: int,
) -> tuple[pd.DataFrame, dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    if len(nodes) < 2:
        return (
            pd.DataFrame(columns=["left_clean_fid", "right_clean_fid", "shared_edge_len"]),
            {},
            {},
        )
    _log(f"[INFO] Building clean shared-edge graph for nodes={len(nodes):,}")
    nodes = nodes.reset_index(drop=True)
    geoms = nodes.geometry.reset_index(drop=True)
    clean_fids = nodes["clean_fid"].astype(int).to_numpy()
    sindex = nodes.sindex
    parts: list[pd.DataFrame] = []
    for start in range(0, len(nodes), int(query_chunk_size)):
        stop = min(start + int(query_chunk_size), len(nodes))
        query_geoms = geoms.iloc[start:stop]
        try:
            left_local, right_pos = sindex.query(query_geoms.geometry.array, predicate="intersects")
        except TypeError:
            left_local, right_pos = sindex.query(query_geoms.geometry.array)
        if len(left_local) == 0:
            continue
        left_pos = left_local.astype("int64") + int(start)
        right_pos = right_pos.astype("int64")
        keep = left_pos < right_pos
        if not bool(np.any(keep)):
            continue
        left_pos = left_pos[keep]
        right_pos = right_pos[keep]
        left_fids = clean_fids[left_pos]
        right_fids = clean_fids[right_pos]
        shared_values: list[float] = []
        for edge_start in range(0, len(left_pos), int(edge_calc_chunk_size)):
            edge_stop = edge_start + int(edge_calc_chunk_size)
            lp = left_pos[edge_start:edge_stop]
            rp = right_pos[edge_start:edge_stop]
            shared = shapely.length(
                shapely.intersection(
                    shapely.boundary(geoms.iloc[lp].array),
                    shapely.boundary(geoms.iloc[rp].array),
                )
            )
            shared_values.extend(float(v) for v in shared)
        chunk = pd.DataFrame(
            {
                "left_clean_fid": left_fids.astype("int64"),
                "right_clean_fid": right_fids.astype("int64"),
                "shared_edge_len": np.asarray(shared_values, dtype="float64"),
            }
        )
        chunk = chunk[chunk["shared_edge_len"].ge(float(min_shared_edge))].copy()
        if not chunk.empty:
            parts.append(chunk)
    edges = (
        pd.concat(parts, ignore_index=True).drop_duplicates(["left_clean_fid", "right_clean_fid"])
        if parts
        else pd.DataFrame(columns=["left_clean_fid", "right_clean_fid", "shared_edge_len"])
    )
    adjacency: dict[int, list[tuple[int, float]]] = {}
    shared_by_pair: dict[tuple[int, int], float] = {}
    for row in edges.itertuples(index=False):
        left = int(row.left_clean_fid)
        right = int(row.right_clean_fid)
        shared = float(row.shared_edge_len)
        adjacency.setdefault(left, []).append((right, shared))
        adjacency.setdefault(right, []).append((left, shared))
        shared_by_pair[(min(left, right), max(left, right))] = shared
    for clean_fid, values in list(adjacency.items()):
        adjacency[clean_fid] = sorted(values, key=lambda item: (-float(item[1]), int(item[0])))[: int(top_neighbors)]
    _log(f"[INFO] Shared-edge rows={len(edges):,}; adjacency_nodes={len(adjacency):,}")
    return edges.reset_index(drop=True), adjacency, shared_by_pair


def _build_edges_cached(
    nodes: gpd.GeoDataFrame,
    *,
    min_shared_edge: float,
    top_neighbors: int,
    query_chunk_size: int,
    edge_calc_chunk_size: int,
    context_cache_dir: str | Path = "",
) -> tuple[pd.DataFrame, dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    if not str(context_cache_dir).strip():
        return _build_edges(
            nodes,
            min_shared_edge=min_shared_edge,
            top_neighbors=top_neighbors,
            query_chunk_size=query_chunk_size,
            edge_calc_chunk_size=edge_calc_chunk_size,
        )
    cache_dir = Path(context_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "shared_edges",
        "nodes_signature": _context_cache_signature(nodes),
        "min_shared_edge": float(min_shared_edge),
        "top_neighbors": int(top_neighbors),
    }
    cache_path = _context_cache_path(cache_dir, "shared_edges", metadata)
    if cache_path.exists():
        payload = joblib.load(cache_path)
        if isinstance(payload, dict) and payload.get("metadata") == metadata:
            edges = pd.DataFrame(payload.get("edges", []))
            adjacency = {
                int(key): [(int(neighbor), float(shared)) for neighbor, shared in value]
                for key, value in dict(payload.get("adjacency", {})).items()
            }
            shared_by_pair = {
                tuple(int(part) for part in str(key).split("|")): float(value)
                for key, value in dict(payload.get("shared_by_pair", {})).items()
            }
            _log(
                "[INFO] Loaded cached shared-edge graph: "
                f"rows={len(edges):,}; adjacency_nodes={len(adjacency):,}; path={cache_path}"
            )
            return edges, adjacency, shared_by_pair
    edges, adjacency, shared_by_pair = _build_edges(
        nodes,
        min_shared_edge=min_shared_edge,
        top_neighbors=top_neighbors,
        query_chunk_size=query_chunk_size,
        edge_calc_chunk_size=edge_calc_chunk_size,
    )
    payload = {
        "metadata": metadata,
        "edges": edges,
        "adjacency": adjacency,
        "shared_by_pair": {f"{left}|{right}": value for (left, right), value in shared_by_pair.items()},
    }
    joblib.dump(payload, cache_path, compress=3)
    _log(f"[INFO] Wrote cached shared-edge graph: path={cache_path}")
    return edges, adjacency, shared_by_pair


def _target_clean_set(source_ids: set[int], source_to_clean: dict[int, list[int]]) -> tuple[frozenset[int], set[int]]:
    clean_ids: set[int] = set()
    missing: set[int] = set()
    for source_fid in source_ids:
        values = source_to_clean.get(int(source_fid))
        if not values:
            missing.add(int(source_fid))
            continue
        clean_ids.update(int(v) for v in values)
    return frozenset(clean_ids), missing


def _collect_anchor_pool(
    *,
    anchor_clean_ids: frozenset[int],
    positive_clean_ids: frozenset[int],
    adjacency: dict[int, list[tuple[int, float]]],
    eligible_clean_ids: set[int],
    max_depth: int,
    max_pool_size: int,
) -> set[int]:
    pool: set[int] = set(int(v) for v in anchor_clean_ids)
    pool.update(int(v) for v in positive_clean_ids)
    frontier = set(int(v) for v in anchor_clean_ids)
    seen_depth = {int(v): 0 for v in frontier}
    for depth in range(int(max_depth)):
        next_frontier: set[int] = set()
        candidates: list[tuple[float, int]] = []
        for clean_fid in sorted(frontier):
            for neighbor, shared in adjacency.get(int(clean_fid), []):
                if int(neighbor) in seen_depth:
                    continue
                if int(neighbor) not in eligible_clean_ids and int(neighbor) not in positive_clean_ids:
                    continue
                candidates.append((float(shared), int(neighbor)))
        for _shared, neighbor in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if neighbor in seen_depth:
                continue
            seen_depth[neighbor] = depth + 1
            next_frontier.add(neighbor)
            pool.add(neighbor)
            if len(pool) >= int(max_pool_size):
                break
        frontier = next_frontier
        if not frontier or len(pool) >= int(max_pool_size):
            break
    return pool


def _best_shared_to_group(
    clean_fid: int,
    group: frozenset[int] | set[int],
    adjacency: dict[int, list[tuple[int, float]]],
) -> float:
    group_set = set(int(v) for v in group)
    return max((float(shared) for neighbor, shared in adjacency.get(int(clean_fid), []) if int(neighbor) in group_set), default=0.0)


def _build_adjacency_lookup(adjacency: dict[int, list[tuple[int, float]]]) -> dict[int, dict[int, float]]:
    lookup: dict[int, dict[int, float]] = {}
    for clean_fid, neighbors in adjacency.items():
        neighbor_lookup: dict[int, float] = {}
        for neighbor, shared in neighbors:
            neighbor = int(neighbor)
            shared = float(shared)
            previous = neighbor_lookup.get(neighbor)
            if previous is None or shared > previous:
                neighbor_lookup[neighbor] = shared
        lookup[int(clean_fid)] = neighbor_lookup
    return lookup


def _group_search_priority(
    group: frozenset[int],
    *,
    anchor_clean_ids: frozenset[int],
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_clean: dict[int, float],
) -> tuple[int, float, float, str]:
    added_ids = sorted(int(fid) for fid in group if int(fid) not in anchor_clean_ids)
    shared_to_group = 0.0
    for fid in added_ids:
        shared_to_group += _best_shared_to_group(int(fid), set(group) - {int(fid)}, adjacency)
    area = sum(float(area_by_clean.get(int(fid), 0.0)) for fid in group)
    return (len(group), -float(shared_to_group), float(area), _ids_text(group))


def _group_fast_shape_score(
    group: frozenset[int],
    *,
    area_by_clean: dict[int, float],
    perimeter_by_clean: dict[int, float],
    shared_by_pair: dict[tuple[int, int], float],
) -> float:
    clean_ids = sorted(int(fid) for fid in group)
    area = sum(float(area_by_clean.get(fid, 0.0)) for fid in clean_ids)
    perimeter_sum = sum(float(perimeter_by_clean.get(fid, 0.0)) for fid in clean_ids)
    internal_shared = 0.0
    for idx, left in enumerate(clean_ids):
        for right in clean_ids[idx + 1 :]:
            internal_shared += float(shared_by_pair.get((left, right), 0.0))
    outer_perimeter = max(float(perimeter_sum) - (2.0 * float(internal_shared)), 1e-6)
    compactness = (4.0 * math.pi * float(area) / (outer_perimeter * outer_perimeter)) if area > 0.0 else 0.0
    boundary_simplification = _safe_ratio(float(perimeter_sum) - outer_perimeter, float(perimeter_sum))
    return (
        (2.5 * float(compactness))
        + (2.0 * float(boundary_simplification))
        + (0.03 * float(len(clean_ids)))
        - (0.00005 * abs(float(area) - 350.0))
    )


def _enumerate_anchor_groups_ordered(
    *,
    anchor_clean_ids: frozenset[int],
    pool: set[int],
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_clean: dict[int, float],
    max_group_size: int,
    max_candidate_area: float,
    per_anchor_limit: int,
    adjacency_lookup: dict[int, dict[int, float]] | None = None,
) -> list[frozenset[int]]:
    groups: list[frozenset[int]] = []
    start = frozenset(int(v) for v in anchor_clean_ids)
    if not start:
        return groups
    build_lookup_lazily = adjacency_lookup is None
    adjacency_lookup_cache: dict[int, dict[int, float]] = {} if adjacency_lookup is None else adjacency_lookup
    seen: set[frozenset[int]] = {start}
    current_level: list[frozenset[int]] = [start]
    beam_width = max(int(per_anchor_limit) * 3, int(per_anchor_limit), 1)
    area_cache: dict[frozenset[int], float] = {
        start: sum(float(area_by_clean.get(fid, 0.0)) for fid in start)
    }
    ids_text_cache: dict[frozenset[int], str] = {start: _ids_text(start)}
    best_shared_cache: dict[tuple[int, frozenset[int]], float] = {}
    priority_cache: dict[frozenset[int], tuple[int, float, float, str]] = {}

    def group_area(group: frozenset[int]) -> float:
        value = area_cache.get(group)
        if value is None:
            value = sum(float(area_by_clean.get(fid, 0.0)) for fid in group)
            area_cache[group] = float(value)
        return float(value)

    def group_ids_text(group: frozenset[int]) -> str:
        value = ids_text_cache.get(group)
        if value is None:
            value = _ids_text(group)
            ids_text_cache[group] = value
        return value

    def best_shared_to_cached(clean_fid: int, group: frozenset[int]) -> float:
        key = (int(clean_fid), group)
        cached = best_shared_cache.get(key)
        if cached is not None:
            return float(cached)
        neighbor_lookup = neighbor_lookup_for(clean_fid)
        value = 0.0
        for neighbor in group:
            shared = float(neighbor_lookup.get(int(neighbor), 0.0))
            if shared > value:
                value = shared
        best_shared_cache[key] = float(value)
        return float(value)

    def best_shared_to_group_excluding(clean_fid: int, group: frozenset[int]) -> float:
        key = (int(clean_fid), group)
        cached = best_shared_cache.get(key)
        if cached is not None:
            return float(cached)
        neighbor_lookup = neighbor_lookup_for(clean_fid)
        value = 0.0
        for neighbor in group:
            neighbor = int(neighbor)
            if neighbor == int(clean_fid):
                continue
            shared = float(neighbor_lookup.get(neighbor, 0.0))
            if shared > value:
                value = shared
        best_shared_cache[key] = float(value)
        return float(value)

    def neighbor_lookup_for(clean_fid: int) -> dict[int, float]:
        clean_fid = int(clean_fid)
        cached = adjacency_lookup_cache.get(clean_fid)
        if cached is not None:
            return cached
        if not build_lookup_lazily:
            return {}
        neighbor_lookup: dict[int, float] = {}
        for neighbor, shared in adjacency.get(clean_fid, []):
            neighbor = int(neighbor)
            shared = float(shared)
            previous = neighbor_lookup.get(neighbor)
            if previous is None or shared > previous:
                neighbor_lookup[neighbor] = shared
        adjacency_lookup_cache[clean_fid] = neighbor_lookup
        return neighbor_lookup

    def group_priority(group: frozenset[int]) -> tuple[int, float, float, str]:
        cached = priority_cache.get(group)
        if cached is not None:
            return cached
        added_ids = sorted(int(fid) for fid in group if int(fid) not in start)
        shared_to_group = 0.0
        for fid in added_ids:
            shared_to_group += best_shared_to_group_excluding(int(fid), group)
        priority = (len(group), -float(shared_to_group), group_area(group), group_ids_text(group))
        priority_cache[group] = priority
        return priority

    while current_level and len(groups) < int(per_anchor_limit):
        next_level_by_group: dict[frozenset[int], None] = {}
        for current in current_level:
            if len(current) >= int(max_group_size):
                continue
            frontier: set[int] = set()
            for clean_fid in current:
                for neighbor, _shared in adjacency.get(int(clean_fid), []):
                    if int(neighbor) not in current and int(neighbor) in pool:
                        frontier.add(int(neighbor))
            ordered_neighbors = sorted(
                frontier,
                key=lambda fid: (
                    -best_shared_to_cached(fid, current),
                    float(area_by_clean.get(fid, 0.0)),
                    fid,
                ),
            )
            for neighbor in ordered_neighbors:
                new_group = frozenset((*current, int(neighbor)))
                if new_group in seen:
                    continue
                area = group_area(current) + float(area_by_clean.get(int(neighbor), 0.0))
                if area > float(max_candidate_area):
                    continue
                area_cache[new_group] = float(area)
                seen.add(new_group)
                next_level_by_group[new_group] = None

        ordered_next = sorted(next_level_by_group, key=group_priority)
        if not ordered_next:
            break
        remaining = int(per_anchor_limit) - len(groups)
        groups.extend(ordered_next[:remaining])
        current_level = ordered_next[:beam_width]
    return groups


def _enumerate_anchor_groups_with_shape_supplement(
    *,
    anchor_clean_ids: frozenset[int],
    pool: set[int],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    area_by_clean: dict[int, float],
    perimeter_by_clean: dict[int, float],
    max_group_size: int,
    max_candidate_area: float,
    per_anchor_limit: int,
    shape_supplement_pool_limit: int,
    shape_supplement_keep: int,
    adjacency_lookup: dict[int, dict[int, float]] | None = None,
) -> list[frozenset[int]]:
    base = _enumerate_anchor_groups_ordered(
        anchor_clean_ids=anchor_clean_ids,
        pool=pool,
        adjacency=adjacency,
        area_by_clean=area_by_clean,
        max_group_size=max_group_size,
        max_candidate_area=max_candidate_area,
        per_anchor_limit=per_anchor_limit,
        adjacency_lookup=adjacency_lookup,
    )
    if (
        int(shape_supplement_pool_limit) <= int(per_anchor_limit)
        or int(shape_supplement_keep) <= 0
        or len(base) < int(per_anchor_limit)
    ):
        return base

    expanded = _enumerate_anchor_groups_ordered(
        anchor_clean_ids=anchor_clean_ids,
        pool=pool,
        adjacency=adjacency,
        area_by_clean=area_by_clean,
        max_group_size=max_group_size,
        max_candidate_area=max_candidate_area,
        per_anchor_limit=int(shape_supplement_pool_limit),
        adjacency_lookup=adjacency_lookup,
    )
    seen = set(base)
    tail = [group for group in expanded if group not in seen]
    if not tail:
        return base
    ranked_tail = sorted(
        tail,
        key=lambda group: (
            -_group_fast_shape_score(
                group,
                area_by_clean=area_by_clean,
                perimeter_by_clean=perimeter_by_clean,
                shared_by_pair=shared_by_pair,
            ),
            len(group),
            _ids_text(group),
        ),
    )
    out = list(base)
    for group in ranked_tail[: int(shape_supplement_keep)]:
        if group in seen:
            continue
        seen.add(group)
        out.append(group)
    return out


def _enumerate_anchor_groups(
    *,
    anchor_clean_ids: frozenset[int],
    pool: set[int],
    adjacency: dict[int, list[tuple[int, float]]],
    area_by_clean: dict[int, float],
    max_group_size: int,
    max_candidate_area: float,
    per_anchor_limit: int,
    adjacency_lookup: dict[int, dict[int, float]] | None = None,
) -> set[frozenset[int]]:
    return set(
        _enumerate_anchor_groups_ordered(
            anchor_clean_ids=anchor_clean_ids,
            pool=pool,
            adjacency=adjacency,
            area_by_clean=area_by_clean,
            max_group_size=max_group_size,
            max_candidate_area=max_candidate_area,
            per_anchor_limit=per_anchor_limit,
            adjacency_lookup=adjacency_lookup,
        )
    )


def _shape_prefixed(prefix: str, shape: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in shape.items()}


def _candidate_features(
    *,
    anchor_source_fid: int,
    anchor_clean_ids: frozenset[int],
    candidate_clean_ids: frozenset[int],
    target_source_ids: set[int],
    target_train_component_id: int,
    target_missing_source_count: int,
    geom_by_clean: dict[int, Any],
    attrs_by_clean: dict[int, dict[str, Any]],
    shape_by_clean: dict[int, dict[str, float]],
    source_by_clean: dict[int, int],
    adjacency: dict[int, list[tuple[int, float]]],
    shared_by_pair: dict[tuple[int, int], float],
    shape_by_group: dict[frozenset[int], dict[str, float]] | None = None,
) -> dict[str, Any]:
    clean_ids = tuple(sorted(int(v) for v in candidate_clean_ids))
    anchor_ids = frozenset(int(v) for v in anchor_clean_ids)
    added_ids = tuple(fid for fid in clean_ids if fid not in anchor_ids)
    source_ids = {int(source_by_clean[fid]) for fid in clean_ids if fid in source_by_clean}
    group_shape = _shape_for_group(
        clean_ids,
        geom_by_clean=geom_by_clean,
        shape_by_clean=shape_by_clean,
        shape_by_group=shape_by_group,
    )
    anchor_shape = _shape_for_group(
        anchor_ids,
        geom_by_clean=geom_by_clean,
        shape_by_clean=shape_by_clean,
        shape_by_group=shape_by_group,
    )

    known_roles = ("building", "land", "road", "gapfill", "other")
    added_id_set = set(added_ids)
    role_counts = {role: 0 for role in known_roles}
    added_role_counts = {role: 0 for role in known_roles}
    role_area = {role: 0.0 for role in known_roles}
    area_sum = 0.0
    area_min = 0.0
    area_max = 0.0
    added_area_sum = 0.0
    added_area_max = 0.0
    added_area_count = 0
    perimeter_sum = 0.0
    uprn_sum = 0
    anchor_uprn_sum = 0
    added_uprn_sum = 0
    added_uprn_polygon_count = 0
    weighted_part_regularity_num = 0.0
    weighted_part_regularity_den = 0.0
    part_shape_count = 0
    polygon_hole_fill_count = 0
    enclosed_gap_fill_count = 0

    for fid in clean_ids:
        attrs = attrs_by_clean[fid]
        area = float(attrs.get("area", 0.0) or 0.0)
        perimeter = float(attrs.get("perimeter", 0.0) or 0.0)
        role = str(attrs.get("anchor_role", "other") or "other")
        uprn_count = int(attrs.get("uprn_count", 0) or 0)
        shape = _shape_for_clean(fid, geom_by_clean=geom_by_clean, shape_by_clean=shape_by_clean)
        weight = max(area, 1e-6)

        area_sum += area
        area_min = area if part_shape_count == 0 else min(area_min, area)
        area_max = max(area_max, area)
        perimeter_sum += perimeter
        uprn_sum += uprn_count
        weighted_part_regularity_num += float(shape["regularity_score"]) * weight
        weighted_part_regularity_den += weight
        part_shape_count += 1
        polygon_hole_fill_count += int(attrs.get("is_polygon_hole_fill", 0) or 0)
        enclosed_gap_fill_count += int(attrs.get("is_enclosed_gap_fill", 0) or 0)

        if role in role_counts:
            role_counts[role] += 1
            role_area[role] += area
        if fid in anchor_ids:
            anchor_uprn_sum += uprn_count
        if fid in added_id_set:
            added_area_sum += area
            added_area_max = max(added_area_max, area)
            added_area_count += 1
            added_uprn_sum += uprn_count
            if uprn_count > 0:
                added_uprn_polygon_count += 1
            if role in added_role_counts:
                added_role_counts[role] += 1

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
    outside_zero_uprn_plot_neighbor_ids: set[int] = set()
    outside_zero_uprn_plot_area_by_clean: dict[int, float] = {}
    outside_zero_uprn_plot_shared_len = 0.0
    outside_zero_uprn_plot_best_shared_len = 0.0
    outside_zero_uprn_role_ids = {role: set() for role in known_roles}
    outside_zero_uprn_role_area_by_clean = {role: {} for role in known_roles}
    outside_zero_uprn_role_shared_len = {role: 0.0 for role in known_roles}
    contains_other_anchor_count = 0
    contains_other_anchor_area = 0.0
    for fid in clean_ids:
        if fid not in anchor_ids:
            attrs = attrs_by_clean.get(int(fid), {})
            if int(attrs.get("uprn_count", 0) or 0) > 0:
                contains_other_anchor_count += 1
                contains_other_anchor_area += float(attrs.get("area", 0.0) or 0.0)
        for neighbor, shared in adjacency.get(int(fid), []):
            if int(neighbor) in candidate_clean_ids:
                continue
            outside_neighbor_ids.add(int(neighbor))
            external_shared += float(shared)
            attrs = attrs_by_clean.get(int(neighbor), {})
            neighbor_uprn = int(attrs.get("uprn_count", 0) or 0)
            neighbor_role = str(attrs.get("anchor_role", "other") or "other")
            if neighbor_role not in outside_zero_uprn_role_ids:
                neighbor_role = "other"
            if neighbor_uprn > 0:
                outside_uprn_neighbor_ids.add(int(neighbor))
            if str(attrs.get("anchor_role", "") or "") == "building":
                outside_building_neighbor_ids.add(int(neighbor))
            plot_eligible = bool(attrs.get("plot_eligible", False))
            if neighbor_uprn == 0 and plot_eligible:
                neighbor = int(neighbor)
                neighbor_area = float(attrs.get("area", 0.0) or 0.0)
                outside_zero_uprn_plot_neighbor_ids.add(neighbor)
                outside_zero_uprn_plot_area_by_clean.setdefault(neighbor, neighbor_area)
                outside_zero_uprn_plot_shared_len += float(shared)
                outside_zero_uprn_plot_best_shared_len = max(
                    float(outside_zero_uprn_plot_best_shared_len),
                    float(shared),
                )
                outside_zero_uprn_role_ids[neighbor_role].add(neighbor)
                outside_zero_uprn_role_area_by_clean[neighbor_role].setdefault(neighbor, neighbor_area)
                outside_zero_uprn_role_shared_len[neighbor_role] += float(shared)

    candidate_area = float(group_shape["area"])
    anchor_area = float(anchor_shape["area"])
    outside_zero_uprn_plot_area = float(sum(outside_zero_uprn_plot_area_by_clean.values()))
    outside_zero_uprn_plot_max_area = (
        float(max(outside_zero_uprn_plot_area_by_clean.values())) if outside_zero_uprn_plot_area_by_clean else 0.0
    )
    outside_zero_uprn_plot_mean_area = _safe_ratio(
        outside_zero_uprn_plot_area,
        float(len(outside_zero_uprn_plot_area_by_clean)),
    ) if outside_zero_uprn_plot_area_by_clean else 0.0
    weighted_part_regularity = (
        _safe_ratio(weighted_part_regularity_num, weighted_part_regularity_den)
        if part_shape_count and area_sum > 0.0
        else 0.0
    )
    source_target_overlap = len(source_ids & target_source_ids)
    source_target_union = len(source_ids | target_source_ids)

    record: dict[str, Any] = {
        "anchor_source_fid": int(anchor_source_fid),
        "anchor_clean_fids": _ids_text(anchor_ids),
        "candidate_clean_fids": _ids_text(candidate_clean_ids),
        "candidate_source_fids": _ids_text(source_ids),
        "target_source_fids": _ids_text(target_source_ids),
        "target_train_component_id": int(target_train_component_id),
        "target_missing_source_count": int(target_missing_source_count),
        "candidate_clean_count": int(len(clean_ids)),
        "candidate_source_count": int(len(source_ids)),
        "anchor_clean_count": int(len(anchor_ids)),
        "added_clean_count": int(len(added_ids)),
        "added_source_count": int(max(len(source_ids) - 1, 0)),
        "candidate_area": candidate_area,
        "anchor_area": anchor_area,
        "added_area_sum": float(added_area_sum),
        "added_area_max": float(added_area_max) if added_area_count else 0.0,
        "added_area_mean": _safe_ratio(float(added_area_sum), float(added_area_count)) if added_area_count else 0.0,
        "candidate_area_to_anchor": _safe_ratio(candidate_area, anchor_area),
        "added_area_to_anchor": _safe_ratio(float(added_area_sum), anchor_area),
        "largest_part_area_ratio": _safe_ratio(float(area_max), float(area_sum)),
        "smallest_part_area_ratio": _safe_ratio(float(area_min), float(area_sum)),
        "internal_shared_len": float(internal_shared),
        "anchor_added_shared_len": float(anchor_added_shared),
        "external_shared_len": float(external_shared),
        "internal_shared_to_sqrt_area": _safe_ratio(float(internal_shared), math.sqrt(max(candidate_area, 0.0))),
        "anchor_added_shared_to_anchor_perimeter": _safe_ratio(float(anchor_added_shared), float(anchor_shape["perimeter"])),
        "shared_to_external_shared": _safe_ratio(float(internal_shared), float(external_shared)),
        "boundary_simplification": _safe_ratio(perimeter_sum - float(group_shape["perimeter"]), perimeter_sum),
        "uprn_count": int(uprn_sum),
        "anchor_uprn_count": int(anchor_uprn_sum),
        "added_uprn_count": int(added_uprn_sum),
        "added_uprn_polygon_count": int(added_uprn_polygon_count),
        "outside_neighbor_count": int(len(outside_neighbor_ids)),
        "outside_uprn_neighbor_count": int(len(outside_uprn_neighbor_ids)),
        "outside_building_neighbor_count": int(len(outside_building_neighbor_ids)),
        "outside_zero_uprn_plot_neighbor_count": int(len(outside_zero_uprn_plot_neighbor_ids)),
        "outside_zero_uprn_plot_shared_len": float(outside_zero_uprn_plot_shared_len),
        "outside_zero_uprn_plot_best_shared_len": float(outside_zero_uprn_plot_best_shared_len),
        "outside_zero_uprn_plot_area": float(outside_zero_uprn_plot_area),
        "outside_zero_uprn_plot_max_area": float(outside_zero_uprn_plot_max_area),
        "outside_zero_uprn_plot_mean_area": float(outside_zero_uprn_plot_mean_area),
        "outside_zero_uprn_plot_area_to_candidate": _safe_ratio(outside_zero_uprn_plot_area, candidate_area),
        "outside_zero_uprn_plot_shared_to_perimeter": _safe_ratio(
            float(outside_zero_uprn_plot_shared_len),
            float(group_shape["perimeter"]),
        ),
        "outside_zero_uprn_plot_shared_to_internal": _safe_ratio(
            float(outside_zero_uprn_plot_shared_len),
            float(internal_shared),
        ),
        "outside_zero_uprn_plot_shared_to_external": _safe_ratio(
            float(outside_zero_uprn_plot_shared_len),
            float(external_shared),
        ),
        "contains_other_anchor_count": int(contains_other_anchor_count),
        "contains_other_anchor_area": float(contains_other_anchor_area),
        "contains_other_anchor_area_fraction": _safe_ratio(float(contains_other_anchor_area), candidate_area),
        "weighted_part_regularity": float(weighted_part_regularity),
        "regularity_gain_vs_anchor": float(group_shape["regularity_score"] - anchor_shape["regularity_score"]),
        "hull_gap_reduction_vs_anchor": float(anchor_shape["hull_gap_ratio"] - group_shape["hull_gap_ratio"]),
        "mrr_gain_vs_anchor": float(group_shape["mrr_ratio"] - anchor_shape["mrr_ratio"]),
        "compactness_gain_vs_anchor": float(group_shape["compactness"] - anchor_shape["compactness"]),
        "regularity_gain_vs_parts": float(group_shape["regularity_score"] - weighted_part_regularity),
        "source_target_overlap_count": int(source_target_overlap),
        "source_target_jaccard": _safe_ratio(float(source_target_overlap), float(source_target_union)),
        "role_signature": "|".join(f"{role}:{role_counts[role]}" for role in sorted(role_counts) if role_counts[role]),
        "anchor_role": str(attrs_by_clean.get(next(iter(anchor_ids)), {}).get("anchor_role", "other") or "other"),
        "is_polygon_hole_fill_count": int(polygon_hole_fill_count),
        "is_enclosed_gap_fill_count": int(enclosed_gap_fill_count),
    }
    for role, count in role_counts.items():
        record[f"role_{role}_count"] = int(count)
        record[f"role_{role}_area"] = float(role_area[role])
        record[f"role_{role}_area_fraction"] = _safe_ratio(float(role_area[role]), candidate_area)
    for role, count in added_role_counts.items():
        record[f"added_role_{role}_count"] = int(count)
    for role in known_roles:
        role_area_value = float(sum(outside_zero_uprn_role_area_by_clean[role].values()))
        record[f"outside_zero_uprn_{role}_neighbor_count"] = int(len(outside_zero_uprn_role_ids[role]))
        record[f"outside_zero_uprn_{role}_area"] = role_area_value
        record[f"outside_zero_uprn_{role}_shared_len"] = float(outside_zero_uprn_role_shared_len[role])
        record[f"outside_zero_uprn_{role}_area_fraction"] = _safe_ratio(role_area_value, candidate_area)
    record.update(_shape_prefixed("group", group_shape))
    record.update(_shape_prefixed("anchor", anchor_shape))
    return record


def _positive_subset_hard_negative_groups(
    *,
    anchor_source_fid: int,
    anchor_clean_ids: frozenset[int],
    positive_clean_ids: frozenset[int],
    target_source_ids: set[int],
    source_to_clean: dict[int, list[int]],
    attrs_by_clean: dict[int, dict[str, Any]],
    max_remove: int,
    max_groups: int,
) -> set[frozenset[int]]:
    if int(max_remove) <= 0 or int(max_groups) <= 0:
        return set()
    removable_sources: list[int] = []
    for source_fid in sorted(int(v) for v in target_source_ids if int(v) != int(anchor_source_fid)):
        clean_ids = [int(v) for v in source_to_clean.get(int(source_fid), []) if int(v) in positive_clean_ids]
        if not clean_ids:
            continue
        has_uprn = any(int(attrs_by_clean.get(clean_id, {}).get("uprn_count", 0) or 0) > 0 for clean_id in clean_ids)
        plot_eligible = any(bool(attrs_by_clean.get(clean_id, {}).get("plot_eligible", False)) for clean_id in clean_ids)
        if has_uprn or not plot_eligible:
            continue
        removable_sources.append(int(source_fid))

    out: set[frozenset[int]] = set()
    if not removable_sources:
        return out
    for remove_count in range(1, min(int(max_remove), len(removable_sources)) + 1):
        for source_combo in combinations(removable_sources, remove_count):
            remove_clean: set[int] = set()
            for source_fid in source_combo:
                remove_clean.update(int(v) for v in source_to_clean.get(int(source_fid), []))
            group = frozenset(int(v) for v in positive_clean_ids if int(v) not in remove_clean)
            if not anchor_clean_ids.issubset(group) or group == positive_clean_ids or len(group) < len(anchor_clean_ids):
                continue
            out.add(group)
            if len(out) >= int(max_groups):
                return out
    return out


def _label_candidate(source_ids: set[int], target_source_ids: set[int], target_missing_source_count: int) -> tuple[int, str, float]:
    if int(target_missing_source_count) == 0 and source_ids == target_source_ids:
        return 1, "source_target_clean_complete_positive", 1.0
    if not source_ids & target_source_ids:
        return 0, "source_target_clean_unmatched_negative", 1.0
    if source_ids < target_source_ids:
        return 0, "source_target_clean_partial_negative", 1.0
    if target_source_ids < source_ids:
        return 0, "source_target_clean_overmerge_negative", 1.0
    return 0, "source_target_clean_mismatch_negative", 1.0


def _prepare_raw_anchor_context(args: argparse.Namespace) -> RawAnchorBuildContext:
    bbox = _parse_bbox(args.bbox)
    wfs = _read_clean_wfs(Path(args.wfs_clean_gpkg), str(args.wfs_clean_layer), bbox)
    wfs = _add_uprn_counts_cached(
        wfs,
        uprn_gpkg=Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
        context_cache_dir=str(getattr(args, "context_cache_dir", "") or ""),
    )
    target = _read_targets(Path(args.target_gpkg), str(args.target_layer), bbox, int(args.max_target_rows))
    if int(getattr(args, "target_id_mod", 0) or 0) > 0:
        remainders = _parse_int_list(str(getattr(args, "target_id_remainders", "") or ""))
        if not remainders:
            raise ValueError("--target-id-remainders is required when --target-id-mod is set")
        target = target[
            target["train_component_id"].astype(int).mod(int(args.target_id_mod)).isin(remainders)
        ].copy()
        _log(f"[INFO] Target id modulo filter applied: rows={len(target):,}")
    source_to_clean, source_by_clean = _build_source_indexes(wfs)

    all_target_sources: set[int] = set()
    for ids in target["target_source_set"]:
        all_target_sources.update(int(v) for v in ids)
    target_source_mask = wfs["source_fid"].astype(int).isin(all_target_sources)
    nodes = wfs[wfs["plot_eligible"].astype(bool) | target_source_mask].copy()
    eligible_clean_ids = set(nodes.loc[nodes["plot_eligible"].astype(bool), "clean_fid"].astype(int))

    _edges, adjacency, shared_by_pair = _build_edges_cached(
        nodes,
        min_shared_edge=float(args.min_shared_edge),
        top_neighbors=int(args.top_neighbors),
        query_chunk_size=int(args.edge_query_chunk_size),
        edge_calc_chunk_size=int(args.edge_calc_chunk_size),
        context_cache_dir=str(getattr(args, "context_cache_dir", "") or ""),
    )
    adjacency_lookup = _build_adjacency_lookup(adjacency)

    geom_by_clean = wfs.set_index("clean_fid").geometry.to_dict()
    attrs_by_clean = wfs.set_index("clean_fid").drop(columns="geometry").to_dict("index")
    shape_by_clean: dict[int, dict[str, float]] = {}
    area_by_clean = wfs.set_index("clean_fid")["area"].astype(float).to_dict()
    perimeter_by_clean = wfs.set_index("clean_fid")["perimeter"].astype(float).to_dict()
    cheap_attrs_by_clean, uprn_clean_ids, building_clean_ids = _build_proposal_cheap_context(
        source_by_clean=source_by_clean,
        attrs_by_clean=attrs_by_clean,
        area_by_clean=area_by_clean,
        perimeter_by_clean=perimeter_by_clean,
    )
    proposal_pipeline = None
    proposal_feature_cols = None
    if str(getattr(args, "proposal_model", "") or "").strip():
        proposal_payload = joblib.load(str(args.proposal_model))
        if not isinstance(proposal_payload, dict) or proposal_payload.get("model_kind") != "wfs_raw_anchor_candidate_proposal_ranker":
            raise RuntimeError("--proposal-model must be a wfs_raw_anchor_candidate_proposal_ranker payload.")
        proposal_pipeline = proposal_payload["pipeline"]
        proposal_feature_cols = list(proposal_payload["feature_cols"])
        _log(
            "[INFO] Proposal candidate mode enabled: "
            f"model={args.proposal_model}; expanded_limit={int(args.proposal_expanded_candidate_limit)}; "
            f"keep={int(args.proposal_keep_per_target)}"
        )
    return RawAnchorBuildContext(
        target=target.reset_index(drop=True),
        source_to_clean=source_to_clean,
        source_by_clean=source_by_clean,
        eligible_clean_ids=eligible_clean_ids,
        adjacency=adjacency,
        adjacency_lookup=adjacency_lookup,
        shared_by_pair=shared_by_pair,
        geom_by_clean=geom_by_clean,
        attrs_by_clean=attrs_by_clean,
        shape_by_clean=shape_by_clean,
        area_by_clean=area_by_clean,
        perimeter_by_clean=perimeter_by_clean,
        cheap_attrs_by_clean=cheap_attrs_by_clean,
        uprn_clean_ids=uprn_clean_ids,
        building_clean_ids=building_clean_ids,
        proposal_pipeline=proposal_pipeline,
        proposal_feature_cols=proposal_feature_cols,
    )


def _empty_build_stats() -> dict[str, Any]:
    return {
        "target_rows": 0,
        "targets_missing_sources": 0,
        "targets_without_positive_clean": 0,
        "trainable_target_ids": set(),
        "positive_target_ids": set(),
        "generated_candidate_count_before_dedupe": 0,
        "candidate_rows": 0,
        "positive_rows": 0,
        "subset_hard_negative_groups": 0,
        "proposal_expanded_group_count": 0,
        "proposal_selected_group_count": 0,
        "label_source_counts": {},
    }


def _finalize_build_stats(stats: dict[str, Any]) -> dict[str, Any]:
    label_counts = {
        str(key): int(value)
        for key, value in sorted(
            dict(stats.get("label_source_counts", {})).items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    }
    trainable_target_ids = set(stats.get("trainable_target_ids", set()))
    positive_target_ids = set(stats.get("positive_target_ids", set()))
    positive_rows = int(stats.get("positive_rows", 0))
    return {
        "target_rows": int(stats.get("target_rows", 0)),
        "targets_missing_sources": int(stats.get("targets_missing_sources", 0)),
        "targets_without_positive_clean": int(stats.get("targets_without_positive_clean", 0)),
        "trainable_target_rows": int(len(trainable_target_ids)),
        "positive_target_rows": int(len(positive_target_ids)),
        "duplicate_positive_rows": int(positive_rows - len(positive_target_ids)),
        "generated_candidate_count_before_dedupe": int(stats.get("generated_candidate_count_before_dedupe", 0)),
        "candidate_rows": int(stats.get("candidate_rows", 0)),
        "positive_rows": positive_rows,
        "subset_hard_negative_groups": int(stats.get("subset_hard_negative_groups", 0)),
        "proposal_expanded_group_count": int(stats.get("proposal_expanded_group_count", 0)),
        "proposal_selected_group_count": int(stats.get("proposal_selected_group_count", 0)),
        "positive_target_coverage_ratio": _safe_ratio(
            float(len(positive_target_ids)),
            float(len(trainable_target_ids)),
        ),
        "label_source_counts": label_counts,
    }


def _update_label_count(stats: dict[str, Any], label_source: str) -> None:
    counts = stats.setdefault("label_source_counts", {})
    counts[str(label_source)] = int(counts.get(str(label_source), 0)) + 1


def _candidate_records_for_targets(
    target: pd.DataFrame,
    *,
    args: argparse.Namespace,
    context: RawAnchorBuildContext,
    seen_keys: set[tuple[int, str]],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    shape_by_group: dict[frozenset[int], dict[str, float]] = {}

    for row in target.itertuples(index=False):
        stats["target_rows"] = int(stats.get("target_rows", 0)) + 1
        target_source_ids = set(int(v) for v in getattr(row, "target_source_set"))
        anchor_source_fid = int(row.anchor_source_fid)
        target_train_component_id = int(row.train_component_id)
        anchor_clean_ids = frozenset(context.source_to_clean.get(anchor_source_fid, []))
        positive_clean_ids, missing_source_ids = _target_clean_set(target_source_ids, context.source_to_clean)
        if missing_source_ids:
            stats["targets_missing_sources"] = int(stats.get("targets_missing_sources", 0)) + 1
        if not anchor_clean_ids or len(positive_clean_ids) < 2:
            stats["targets_without_positive_clean"] = int(stats.get("targets_without_positive_clean", 0)) + 1
            continue
        if not missing_source_ids:
            stats.setdefault("trainable_target_ids", set()).add(target_train_component_id)

        if context.proposal_pipeline is not None:
            from train_wfs_raw_anchor_candidate_proposal_model import cheap_candidate_features

            pool = _collect_anchor_pool(
                anchor_clean_ids=anchor_clean_ids,
                positive_clean_ids=frozenset(),
                adjacency=context.adjacency,
                eligible_clean_ids=context.eligible_clean_ids,
                max_depth=int(args.neighbor_depth),
                max_pool_size=int(args.max_pool_size),
            )
            expanded_groups = _enumerate_anchor_groups_ordered(
                anchor_clean_ids=anchor_clean_ids,
                pool=pool,
                adjacency=context.adjacency,
                area_by_clean=context.area_by_clean,
                max_group_size=int(args.max_group_size),
                max_candidate_area=float(args.max_candidate_area),
                per_anchor_limit=int(args.proposal_expanded_candidate_limit),
                adjacency_lookup=context.adjacency_lookup,
            )
            stats["proposal_expanded_group_count"] = int(stats.get("proposal_expanded_group_count", 0)) + len(expanded_groups)
            proposal_records = [
                cheap_candidate_features(
                    anchor_source_fid=anchor_source_fid,
                    anchor_clean_ids=anchor_clean_ids,
                    candidate_clean_ids=group,
                    enum_rank=rank,
                    target_train_component_id=target_train_component_id,
                    source_by_clean=context.source_by_clean,
                    attrs_by_clean=context.attrs_by_clean,
                    area_by_clean=context.area_by_clean,
                    perimeter_by_clean=context.perimeter_by_clean,
                    adjacency=context.adjacency,
                    shared_by_pair=context.shared_by_pair,
                    include_ids=False,
                    cheap_attrs_by_clean=context.cheap_attrs_by_clean,
                    uprn_clean_ids=context.uprn_clean_ids,
                    building_clean_ids=context.building_clean_ids,
                )
                for rank, group in enumerate(expanded_groups, start=1)
            ]
            if proposal_records:
                proposal_frame = pd.DataFrame.from_records(proposal_records)
                proposal_feature_cols = list(context.proposal_feature_cols or [])
                for column in proposal_feature_cols:
                    if column not in proposal_frame.columns:
                        proposal_frame[column] = np.nan
                proposal_scores = context.proposal_pipeline.predict_proba(proposal_frame[proposal_feature_cols])[:, 1]
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
                proposal_groups = [expanded_groups[idx] for idx in order[: int(args.proposal_keep_per_target)]]
            else:
                proposal_groups = []
            base_groups = []
            if bool(getattr(args, "proposal_include_base_candidates", False)):
                base_groups = list(expanded_groups[: int(args.per_anchor_candidate_limit)])
            groups = []
            seen_group: set[frozenset[int]] = set()
            for group in list(proposal_groups) + list(base_groups):
                if group in seen_group:
                    continue
                seen_group.add(group)
                groups.append(group)
            stats["proposal_selected_group_count"] = int(stats.get("proposal_selected_group_count", 0)) + len(groups)
        else:
            production_candidate_mode = bool(getattr(args, "production_candidate_mode", False))
            groups = set() if production_candidate_mode else {positive_clean_ids}
            pool = _collect_anchor_pool(
                anchor_clean_ids=anchor_clean_ids,
                positive_clean_ids=frozenset() if production_candidate_mode else positive_clean_ids,
                adjacency=context.adjacency,
                eligible_clean_ids=context.eligible_clean_ids,
                max_depth=int(args.neighbor_depth),
                max_pool_size=int(args.max_pool_size),
            )
            if int(getattr(args, "shape_supplement_pool_limit", 0) or 0) > int(args.per_anchor_candidate_limit):
                groups.update(
                    _enumerate_anchor_groups_with_shape_supplement(
                        anchor_clean_ids=anchor_clean_ids,
                        pool=pool,
                        adjacency=context.adjacency,
                        shared_by_pair=context.shared_by_pair,
                        area_by_clean=context.area_by_clean,
                        perimeter_by_clean=context.perimeter_by_clean,
                        max_group_size=int(args.max_group_size),
                        max_candidate_area=float(args.max_candidate_area),
                        per_anchor_limit=int(args.per_anchor_candidate_limit),
                        shape_supplement_pool_limit=int(args.shape_supplement_pool_limit),
                        shape_supplement_keep=int(args.shape_supplement_keep),
                        adjacency_lookup=context.adjacency_lookup,
                    )
                )
            else:
                groups.update(
                    _enumerate_anchor_groups(
                        anchor_clean_ids=anchor_clean_ids,
                        pool=pool,
                        adjacency=context.adjacency,
                        area_by_clean=context.area_by_clean,
                        max_group_size=int(args.max_group_size),
                        max_candidate_area=float(args.max_candidate_area),
                        per_anchor_limit=int(args.per_anchor_candidate_limit),
                        adjacency_lookup=context.adjacency_lookup,
                    )
                )
        subset_hard_negative_groups: set[frozenset[int]] = set()
        if bool(getattr(args, "positive_subset_hard_negatives", True)) and not missing_source_ids:
            subset_hard_negative_groups = _positive_subset_hard_negative_groups(
                anchor_source_fid=anchor_source_fid,
                anchor_clean_ids=anchor_clean_ids,
                positive_clean_ids=positive_clean_ids,
                target_source_ids=target_source_ids,
                source_to_clean=context.source_to_clean,
                attrs_by_clean=context.attrs_by_clean,
                max_remove=int(getattr(args, "positive_subset_hard_negative_max_remove", 1)),
                max_groups=int(getattr(args, "positive_subset_hard_negative_limit", 12)),
            )
            if subset_hard_negative_groups:
                groups = set(groups)
                groups.update(subset_hard_negative_groups)
                stats["subset_hard_negative_groups"] = (
                    int(stats.get("subset_hard_negative_groups", 0)) + len(subset_hard_negative_groups)
                )
        subset_hard_negative_keys = {_ids_text(group) for group in subset_hard_negative_groups}
        stats["generated_candidate_count_before_dedupe"] = (
            int(stats.get("generated_candidate_count_before_dedupe", 0)) + len(groups)
        )

        for group in groups:
            if not group or not anchor_clean_ids.issubset(group):
                continue
            key = (anchor_source_fid, _ids_text(group))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                rec = _candidate_features(
                    anchor_source_fid=anchor_source_fid,
                    anchor_clean_ids=anchor_clean_ids,
                    candidate_clean_ids=group,
                    target_source_ids=target_source_ids,
                    target_train_component_id=target_train_component_id,
                    target_missing_source_count=len(missing_source_ids),
                    geom_by_clean=context.geom_by_clean,
                    attrs_by_clean=context.attrs_by_clean,
                    shape_by_clean=context.shape_by_clean,
                    source_by_clean=context.source_by_clean,
                    adjacency=context.adjacency,
                    shared_by_pair=context.shared_by_pair,
                    shape_by_group=shape_by_group,
                )
            except Exception:
                continue
            candidate_sources = _parse_id_set(rec["candidate_source_fids"])
            label, label_source, sample_weight = _label_candidate(
                candidate_sources,
                target_source_ids,
                len(missing_source_ids),
            )
            if (
                key[1] in subset_hard_negative_keys
                and not label
                and bool(candidate_sources)
                and candidate_sources < target_source_ids
            ):
                label_source = "source_target_clean_subset_omission_hard_negative"
            rec[TARGET_COL] = int(label)
            rec["label_source"] = str(label_source)
            rec["sample_weight"] = float(sample_weight)
            if label:
                stats["positive_rows"] = int(stats.get("positive_rows", 0)) + 1
                stats.setdefault("positive_target_ids", set()).add(target_train_component_id)
            stats["candidate_rows"] = int(stats.get("candidate_rows", 0)) + 1
            _update_label_count(stats, str(label_source))
            records.append(rec)
    return records


def build_raw_anchor_group_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = _prepare_raw_anchor_context(args)
    stats = _empty_build_stats()
    seen_keys: set[tuple[int, str]] = set()
    records = _candidate_records_for_targets(
        context.target,
        args=args,
        context=context,
        seen_keys=seen_keys,
        stats=stats,
    )
    if not records:
        raise RuntimeError("No raw anchor group candidates were generated.")
    dataset = pd.DataFrame.from_records(records)
    summary = _finalize_build_stats(stats)
    _log("[INFO] Raw anchor candidate summary:")
    _log(json.dumps(summary, indent=2))
    return dataset, summary


def build_raw_anchor_group_candidate_cache(args: argparse.Namespace, cache_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    context = _prepare_raw_anchor_context(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("wfs_raw_anchor_group_candidates_part_*.csv"):
        stale.unlink()

    batch_size = int(args.target_batch_size)
    if batch_size <= 0:
        batch_size = len(context.target)
    stats = _empty_build_stats()
    seen_keys: set[tuple[int, str]] = set()
    part_paths: list[Path] = []
    total_targets = len(context.target)
    for batch_index, start in enumerate(range(0, total_targets, batch_size), start=1):
        stop = min(start + batch_size, total_targets)
        batch = context.target.iloc[start:stop].copy()
        _log(f"[INFO] Building candidate batch {batch_index}: target_rows={start:,}-{stop - 1:,}")
        records = _candidate_records_for_targets(
            batch,
            args=args,
            context=context,
            seen_keys=seen_keys,
            stats=stats,
        )
        if not records:
            _log(f"[WARN] Candidate batch {batch_index} produced no rows")
            continue
        part_path = cache_dir / f"wfs_raw_anchor_group_candidates_part_{batch_index:04d}.csv"
        pd.DataFrame.from_records(records).to_csv(part_path, index=False)
        part_paths.append(part_path)
        _log(f"[INFO] Wrote candidate part {batch_index}: rows={len(records):,}; path={part_path}")

    if not part_paths:
        raise RuntimeError("No raw anchor group candidate parts were generated.")
    summary = _finalize_build_stats(stats)
    summary["candidate_part_count"] = int(len(part_paths))
    summary["candidate_cache_dir"] = str(cache_dir)
    summary["candidate_part_paths"] = [str(path) for path in part_paths]
    (cache_dir / "wfs_raw_anchor_group_candidate_cache_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _log("[INFO] Raw anchor candidate cache summary:")
    _log(json.dumps(summary, indent=2))
    return part_paths, summary


def _candidate_input_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token)
        if path.is_dir():
            paths.extend(sorted(path.glob("wfs_raw_anchor_group_candidates_part_*.csv")))
            continue
        if any(ch in token for ch in "*?["):
            paths.extend(Path(match) for match in sorted(glob.glob(token)))
            continue
        paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Candidate input CSV not found: {missing[:5]}")
    if not unique:
        raise RuntimeError("No candidate input CSVs matched --candidate-input-csv")
    return unique


def read_candidate_inputs(value: str) -> pd.DataFrame:
    paths = _candidate_input_paths(value)
    _log(f"[INFO] Reading raw anchor candidate CSVs: files={len(paths):,}")
    frames: list[pd.DataFrame] = []
    dtype = {
        "anchor_clean_fids": "string",
        "candidate_clean_fids": "string",
        "candidate_source_fids": "string",
        "target_source_fids": "string",
        "role_signature": "string",
        "anchor_role": "string",
        "label_source": "string",
    }
    for idx, path in enumerate(paths, start=1):
        frame = pd.read_csv(path, dtype=dtype, low_memory=False)
        frames.append(frame)
        _log(f"[INFO] Read candidate CSV {idx}/{len(paths)}: rows={len(frame):,}; path={path}")
    dataset = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    _log(f"[INFO] Candidate rows loaded={len(dataset):,}")
    return dataset


def candidate_dataset_summary(dataset: pd.DataFrame) -> dict[str, Any]:
    positive = dataset[TARGET_COL].astype(int).eq(1)
    if "target_train_component_id" in dataset.columns:
        positive_target_ids = set(
            int(value) for value in dataset.loc[positive, "target_train_component_id"].dropna().astype(int)
        )
        target_ids = set(int(value) for value in dataset["target_train_component_id"].dropna().astype(int))
    else:
        positive_target_ids = set()
        target_ids = set()
    return {
        "candidate_rows": int(len(dataset)),
        "positive_rows": int(positive.sum()),
        "positive_target_rows": int(len(positive_target_ids)),
        "candidate_target_rows": int(len(target_ids)),
        "label_source_counts": dataset["label_source"].value_counts().to_dict()
        if "label_source" in dataset.columns
        else {},
    }


def _feature_columns(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = ID_COLUMNS | MODEL_OUTPUT_COLUMNS | {TARGET_COL, "sample_weight"}
    feature_cols = [
        column
        for column in dataset.columns
        if column not in excluded and not any(str(column).startswith(marker) for marker in LABEL_DERIVED_MARKERS)
    ]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_cols]
    numeric = [
        column
        for column in feature_cols
        if column not in categorical and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    return numeric + categorical, numeric, categorical


def _add_pool_rank_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    group_key = "anchor_source_fid"
    if group_key not in out.columns:
        return out

    desc_columns = [
        "candidate_clean_count",
        "candidate_source_count",
        "added_clean_count",
        "added_source_count",
        "candidate_area",
        "added_area_sum",
        "candidate_area_to_anchor",
        "internal_shared_len",
        "anchor_added_shared_len",
        "boundary_simplification",
        "group_regularity_score",
        "regularity_gain_vs_anchor",
        "hull_gap_reduction_vs_anchor",
    ]
    asc_columns = [
        "group_hull_gap_ratio",
        "group_notch_index",
        "outside_uprn_neighbor_count",
        "contains_other_anchor_count",
    ]

    for column in desc_columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        group_max = values.groupby(out[group_key]).transform("max").replace(0.0, np.nan)
        out[f"pool_{column}_to_max"] = (values / group_max).fillna(0.0)
        out[f"pool_{column}_rank_desc"] = values.groupby(out[group_key]).rank(method="average", ascending=False)

    for column in asc_columns:
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        group_min = values.groupby(out[group_key]).transform("min")
        out[f"pool_{column}_delta_min"] = values - group_min
        out[f"pool_{column}_rank_asc"] = values.groupby(out[group_key]).rank(method="average", ascending=True)

    if {"pool_candidate_area_rank_desc", "pool_candidate_source_count_rank_desc"}.issubset(out.columns):
        out["pool_area_source_rank_mean"] = (
            out["pool_candidate_area_rank_desc"].astype(float)
            + out["pool_candidate_source_count_rank_desc"].astype(float)
        ) / 2.0
    return out


def _sample_fit_rows(dataset: pd.DataFrame, *, max_negative_rows: int, random_state: int) -> pd.DataFrame:
    positives = dataset[dataset[TARGET_COL].astype(int).eq(1)]
    negatives = dataset[dataset[TARGET_COL].astype(int).eq(0)]
    if int(max_negative_rows) > 0 and len(negatives) > int(max_negative_rows):
        if "label_source" in negatives.columns:
            partial = negatives[negatives["label_source"].astype(str).str.contains("partial", case=False, na=False)]
            remaining = negatives.drop(index=partial.index)
            remaining_budget = max(int(max_negative_rows) - len(partial), 0)
            if remaining_budget > 0 and len(remaining) > remaining_budget:
                remaining = remaining.sample(n=int(remaining_budget), random_state=int(random_state))
            negatives = pd.concat([partial, remaining], ignore_index=False)
            if len(negatives) > int(max_negative_rows):
                negatives = negatives.sample(n=int(max_negative_rows), random_state=int(random_state))
        else:
            negatives = negatives.sample(n=int(max_negative_rows), random_state=int(random_state))
    out = pd.concat([positives, negatives], ignore_index=True)
    return out.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)


def _apply_hard_negative_weights(
    dataset: pd.DataFrame,
    *,
    partial_negative_weight: float,
    overmerge_negative_weight: float,
    subset_hard_negative_weight: float,
) -> pd.DataFrame:
    out = dataset.copy()
    out["sample_weight"] = pd.to_numeric(out.get("sample_weight", 1.0), errors="coerce").fillna(1.0).astype(float)
    if "label_source" not in out.columns:
        return out
    negative = out[TARGET_COL].astype(int).eq(0)
    label_text = out["label_source"].astype(str)
    partial = negative & label_text.str.contains("partial", case=False, na=False)
    overmerge = negative & label_text.str.contains("overmerge", case=False, na=False)
    subset = negative & label_text.str.contains("subset_omission", case=False, na=False)
    out.loc[partial, "sample_weight"] *= float(partial_negative_weight)
    out.loc[overmerge, "sample_weight"] *= float(overmerge_negative_weight)
    out.loc[subset, "sample_weight"] *= float(subset_hard_negative_weight)
    return out


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
        y_true,
        pred,
        labels=[1, 0],
        zero_division=0,
    )
    out: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "precision_positive": float(precision[0]),
        "recall_positive": float(recall[0]),
        "f1_positive": float(f1[0]),
        "support_positive": int(support[0]),
        "precision_negative": float(precision[1]),
        "recall_negative": float(recall[1]),
        "f1_negative": float(f1[1]),
        "support_negative": int(support[1]),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, pred, labels=[0, 1]).astype(int).tolist(),
        "thresholds_at_precision": _thresholds_at_precision(y_true, proba, [0.9, 0.95, 0.97]),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["roc_auc"] = None
    try:
        out["average_precision"] = float(average_precision_score(y_true, proba))
    except ValueError:
        out["average_precision"] = None
    return out


def _ranking_metrics(dataset: pd.DataFrame, proba: np.ndarray) -> dict[str, Any]:
    if dataset.empty or "target_train_component_id" not in dataset.columns:
        return {}
    work = dataset[["target_train_component_id", TARGET_COL]].copy()
    work["proba"] = np.asarray(proba, dtype="float64")
    ranks: list[float] = []
    top1 = 0
    top3 = 0
    top5 = 0
    positive_groups = 0
    for _target_id, group in work.groupby("target_train_component_id", sort=False):
        labels = group[TARGET_COL].astype(int).to_numpy()
        if not bool(np.any(labels == 1)):
            continue
        positive_groups += 1
        ordered = group.sort_values("proba", ascending=False).reset_index(drop=True)
        positive_positions = np.flatnonzero(ordered[TARGET_COL].astype(int).to_numpy() == 1)
        if len(positive_positions) == 0:
            continue
        rank = int(positive_positions[0]) + 1
        ranks.append(float(rank))
        if rank <= 1:
            top1 += 1
        if rank <= 3:
            top3 += 1
        if rank <= 5:
            top5 += 1
    if positive_groups == 0:
        return {"positive_groups": 0}
    rank_values = np.asarray(ranks, dtype="float64")
    return {
        "positive_groups": int(positive_groups),
        "top1_positive_group_accuracy": _safe_ratio(float(top1), float(positive_groups)),
        "top3_positive_group_accuracy": _safe_ratio(float(top3), float(positive_groups)),
        "top5_positive_group_accuracy": _safe_ratio(float(top5), float(positive_groups)),
        "mean_positive_rank": float(np.mean(rank_values)) if len(rank_values) else None,
        "median_positive_rank": float(np.median(rank_values)) if len(rank_values) else None,
        "p90_positive_rank": float(np.quantile(rank_values, 0.9)) if len(rank_values) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a raw-WFS-clean anchor group scorer.")
    parser.add_argument("--wfs-clean-gpkg", default=DEFAULT_WFS_CLEAN_GPKG)
    parser.add_argument("--wfs-clean-layer", default=DEFAULT_WFS_CLEAN_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--target-gpkg", default=DEFAULT_TARGET_GPKG)
    parser.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-input-csv", default="")
    parser.add_argument("--candidate-cache-dir", default="")
    parser.add_argument("--build-candidates-only", action="store_true")
    parser.add_argument("--production-candidate-mode", action="store_true")
    parser.add_argument("--bbox", default="")
    parser.add_argument("--max-target-rows", type=int, default=0)
    parser.add_argument("--target-id-mod", type=int, default=0)
    parser.add_argument("--target-id-remainders", default="")
    parser.add_argument("--target-batch-size", type=int, default=2000)
    parser.add_argument("--neighbor-depth", type=int, default=3)
    parser.add_argument("--max-pool-size", type=int, default=28)
    parser.add_argument("--max-group-size", type=int, default=10)
    parser.add_argument("--max-candidate-area", type=float, default=8000.0)
    parser.add_argument("--per-anchor-candidate-limit", type=int, default=160)
    parser.add_argument("--full-score-per-anchor-limit", type=int, default=160)
    parser.add_argument("--positive-subset-hard-negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positive-subset-hard-negative-max-remove", type=int, default=2)
    parser.add_argument("--positive-subset-hard-negative-limit", type=int, default=18)
    parser.add_argument("--shape-supplement-pool-limit", type=int, default=0)
    parser.add_argument("--shape-supplement-keep", type=int, default=0)
    parser.add_argument("--proposal-model", default="")
    parser.add_argument("--proposal-expanded-candidate-limit", type=int, default=3000)
    parser.add_argument("--proposal-keep-per-target", type=int, default=80)
    parser.add_argument("--proposal-include-base-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-neighbors", type=int, default=14)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--edge-query-chunk-size", type=int, default=20000)
    parser.add_argument("--edge-calc-chunk-size", type=int, default=50000)
    parser.add_argument("--context-cache-dir", default="")
    parser.add_argument("--max-negative-train-rows", type=int, default=240000)
    parser.add_argument("--partial-negative-weight", type=float, default=12.0)
    parser.add_argument("--overmerge-negative-weight", type=float, default=4.0)
    parser.add_argument("--subset-hard-negative-weight", type=float, default=20.0)
    parser.add_argument("--disable-pool-rank-features", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-candidate-output", action="store_true")
    parser.add_argument("--skip-predictions-output", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_summary: dict[str, Any] = {}
    if str(args.candidate_input_csv).strip():
        dataset = read_candidate_inputs(str(args.candidate_input_csv))
        build_summary = candidate_dataset_summary(dataset)
        _log(f"[INFO] Reusing raw anchor candidate input: {args.candidate_input_csv}")
    elif str(args.candidate_cache_dir).strip():
        part_paths, build_summary = build_raw_anchor_group_candidate_cache(args, Path(args.candidate_cache_dir))
        if bool(args.build_candidates_only):
            _log("[DONE] Raw anchor candidate cache build complete")
            _log(json.dumps(build_summary, indent=2))
            _log(f"[DONE] candidate_cache_dir={args.candidate_cache_dir}")
            return
        dataset = read_candidate_inputs(",".join(str(path) for path in part_paths))
    else:
        dataset, build_summary = build_raw_anchor_group_candidates(args)
        if bool(args.skip_candidate_output):
            _log("[INFO] Skipping raw anchor candidate CSV write")
        else:
            dataset.to_csv(output_dir / CANDIDATES_FILE_NAME, index=False)
            _log(f"[INFO] Wrote raw anchor candidate CSV: {output_dir / CANDIDATES_FILE_NAME}")
        if bool(args.build_candidates_only):
            _log("[DONE] Raw anchor candidate build complete")
            _log(json.dumps(build_summary, indent=2))
            _log(f"[DONE] outputs={output_dir}")
            return

    dataset[TARGET_COL] = dataset[TARGET_COL].astype(int)
    dataset = _apply_hard_negative_weights(
        dataset,
        partial_negative_weight=float(args.partial_negative_weight),
        overmerge_negative_weight=float(args.overmerge_negative_weight),
        subset_hard_negative_weight=float(args.subset_hard_negative_weight),
    )
    if int(dataset[TARGET_COL].sum()) == 0:
        raise RuntimeError("No positive rows are available for raw anchor group training.")

    if bool(args.disable_pool_rank_features):
        _log("[INFO] Pool rank features disabled")
    else:
        dataset = _add_pool_rank_features(dataset)
    feature_cols, numeric_cols, categorical_cols = _feature_columns(dataset)
    fit_dataset = _sample_fit_rows(
        dataset,
        max_negative_rows=int(args.max_negative_train_rows),
        random_state=int(args.random_state),
    )
    _log(
        "[INFO] Training rows="
        f"{len(fit_dataset):,}; fit_label_counts={fit_dataset[TARGET_COL].value_counts().to_dict()}; "
        f"all_label_counts={dataset[TARGET_COL].value_counts().to_dict()}"
    )
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
        max_iter=240,
        learning_rate=0.05,
        max_leaf_nodes=21,
        l2_regularization=0.08,
        random_state=int(args.random_state),
        early_stopping=True,
        n_iter_no_change=20,
        verbose=1,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    groups = fit_dataset["anchor_source_fid"].astype(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=int(args.random_state))
    train_idx, test_idx = next(splitter.split(fit_dataset, fit_dataset[TARGET_COL], groups=groups))
    train = fit_dataset.iloc[train_idx].copy()
    test = fit_dataset.iloc[test_idx].copy()

    _log("[INFO] Training raw anchor group scorer")
    pipeline.fit(
        train[feature_cols],
        train[TARGET_COL],
        model__sample_weight=train["sample_weight"].astype(float).to_numpy(),
    )
    test_proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    all_proba = pipeline.predict_proba(dataset[feature_cols])[:, 1]
    dataset = dataset.copy()
    dataset["raw_anchor_group_proba"] = all_proba
    dataset["raw_anchor_group_pred_at_threshold"] = dataset["raw_anchor_group_proba"].ge(float(args.threshold)).astype(int)
    test_metrics = _metrics(test[TARGET_COL].to_numpy(dtype=int), test_proba, float(args.threshold))
    all_metrics = _metrics(dataset[TARGET_COL].to_numpy(dtype=int), all_proba, float(args.threshold))
    test_ranking_metrics = _ranking_metrics(test, test_proba)
    all_ranking_metrics = _ranking_metrics(dataset, all_proba)

    _log("[INFO] Refitting final raw anchor group scorer")
    final_model = HistGradientBoostingClassifier(
        max_iter=240,
        learning_rate=0.05,
        max_leaf_nodes=21,
        l2_regularization=0.08,
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
        "model_kind": "wfs_raw_anchor_group_scorer",
        "pipeline": final_pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "training_params": {
            "wfs_clean_gpkg": str(args.wfs_clean_gpkg),
            "wfs_clean_layer": str(args.wfs_clean_layer),
            "uprn_gpkg": str(args.uprn_gpkg),
            "uprn_layer": str(args.uprn_layer),
            "target_gpkg": str(args.target_gpkg),
            "target_layer": str(args.target_layer),
            "candidate_input_csv": str(args.candidate_input_csv),
            "candidate_cache_dir": str(args.candidate_cache_dir),
            "production_candidate_mode": bool(args.production_candidate_mode),
            "bbox": str(args.bbox),
            "max_target_rows": int(args.max_target_rows),
            "target_id_mod": int(args.target_id_mod),
            "target_id_remainders": sorted(_parse_int_list(str(args.target_id_remainders))),
            "target_batch_size": int(args.target_batch_size),
            "neighbor_depth": int(args.neighbor_depth),
            "max_pool_size": int(args.max_pool_size),
            "max_group_size": int(args.max_group_size),
            "max_candidate_area": float(args.max_candidate_area),
            "per_anchor_candidate_limit": int(args.per_anchor_candidate_limit),
            "full_score_per_anchor_limit": int(args.full_score_per_anchor_limit),
            "positive_subset_hard_negatives": bool(args.positive_subset_hard_negatives),
            "positive_subset_hard_negative_max_remove": int(args.positive_subset_hard_negative_max_remove),
            "positive_subset_hard_negative_limit": int(args.positive_subset_hard_negative_limit),
            "shape_supplement_pool_limit": int(args.shape_supplement_pool_limit),
            "shape_supplement_keep": int(args.shape_supplement_keep),
            "proposal_model": str(args.proposal_model),
            "proposal_expanded_candidate_limit": int(args.proposal_expanded_candidate_limit),
            "proposal_keep_per_target": int(args.proposal_keep_per_target),
            "proposal_include_base_candidates": bool(args.proposal_include_base_candidates),
            "top_neighbors": int(args.top_neighbors),
            "min_shared_edge": float(args.min_shared_edge),
            "context_cache_dir": str(args.context_cache_dir),
            "max_negative_train_rows": int(args.max_negative_train_rows),
            "partial_negative_weight": float(args.partial_negative_weight),
            "overmerge_negative_weight": float(args.overmerge_negative_weight),
            "subset_hard_negative_weight": float(args.subset_hard_negative_weight),
            "disable_pool_rank_features": bool(args.disable_pool_rank_features),
            "threshold": float(args.threshold),
            "skip_predictions_output": bool(args.skip_predictions_output),
            "random_state": int(args.random_state),
        },
    }
    joblib.dump(payload, output_dir / MODEL_FILE_NAME)

    report_cols = [
        "anchor_source_fid",
        "anchor_clean_fids",
        "candidate_clean_fids",
        "candidate_source_fids",
        "target_source_fids",
        "target_train_component_id",
        "label",
        "label_source",
        "sample_weight",
        "raw_anchor_group_proba",
        "raw_anchor_group_pred_at_threshold",
        "candidate_clean_count",
        "candidate_source_count",
        "candidate_area",
        "anchor_area",
        "added_area_sum",
        "candidate_area_to_anchor",
        "uprn_count",
        "anchor_uprn_count",
        "added_uprn_count",
        "added_uprn_polygon_count",
        "outside_zero_uprn_plot_neighbor_count",
        "outside_zero_uprn_plot_shared_len",
        "outside_zero_uprn_plot_area",
        "outside_zero_uprn_plot_shared_to_perimeter",
        "contains_other_anchor_count",
        "internal_shared_len",
        "anchor_added_shared_len",
        "boundary_simplification",
        "source_target_jaccard",
        "group_regularity_score",
        "group_mrr_ratio",
        "group_hull_gap_ratio",
        "group_notch_index",
        "regularity_gain_vs_anchor",
        "hull_gap_reduction_vs_anchor",
        "role_signature",
    ]
    if bool(args.skip_predictions_output):
        _log("[INFO] Skipping raw anchor prediction CSV write")
    else:
        report_cols = [column for column in report_cols if column in dataset.columns]
        dataset[report_cols].sort_values("raw_anchor_group_proba", ascending=False).to_csv(
            output_dir / PREDICTIONS_FILE_NAME,
            index=False,
        )
    metrics = {
        "model_kind": "wfs_raw_anchor_group_scorer",
        "output_dir": str(output_dir),
        "model": str(output_dir / MODEL_FILE_NAME),
        "candidate_rows": int(len(dataset)),
        "label_counts": dataset[TARGET_COL].value_counts().sort_index().astype(int).to_dict(),
        "label_source_counts": dataset["label_source"].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "build_summary": build_summary,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "test_ranking_metrics": test_ranking_metrics,
        "all_ranking_metrics": all_ranking_metrics,
    }
    (output_dir / METRICS_FILE_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log("[DONE] Raw anchor group model training complete")
    _log(json.dumps(test_metrics, indent=2))
    _log(f"[DONE] outputs={output_dir}")


if __name__ == "__main__":
    main()

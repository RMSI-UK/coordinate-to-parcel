#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import Polygon


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg"
DEFAULT_WFS_LAYER = "wfs_raw_clean"
DEFAULT_ANCHOR_GPKG = "/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean_anchor.gpkg"
DEFAULT_ANCHOR_LAYER = "wfs_raw_clean_anchor"
DEFAULT_COUNCIL_SINGLE_ANCHOR_GPKG = (
    "/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_area05.gpkg"
)
DEFAULT_COUNCIL_SINGLE_ANCHOR_LAYER = "council_polygons_single_anchor_area05"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_fallback.gpkg"
DEFAULT_OUTPUT_LAYER = "council_polygons_single_anchor_fallback"
OWNED_COLUMNS = [
    "zero_clean_fid",
    "zero_source_fid",
    "owner_anchor_fid",
    "depth",
    "via_zero_clean_fid",
    "shared_edge_m",
    "attraction_ratio",
    "claim_rank",
]
FALLBACK_OUTPUT_COLUMNS = [
    "anchor_fid",
    "anchor_clean_fid",
    "anchor_source_fid",
    "anchor_theme",
    "anchor_uprn_count",
    "fallback_reason",
    "selection_method",
    "selection_policy",
    "candidate_pool_k0_count",
    "candidate_pool_k1_count",
    "evaluated_candidate_count",
    "added_clean_count",
    "added_k0_count",
    "added_k1_count",
    "added_clean_fids",
    "added_source_fids",
    "added_k0_clean_fids",
    "added_k1_clean_fids",
    "anchor_area",
    "anchor_regularity_score",
    "anchor_mrr_ratio",
    "anchor_hull_gap_ratio",
    "candidate_best_regularity_score",
    "min_band_regularity_score",
    "anchor_loss_regularity_floor",
    "fallback_area",
    "fallback_perimeter",
    "fallback_regularity_score",
    "fallback_mrr_ratio",
    "fallback_hull_gap_ratio",
    "fallback_compactness",
    "fallback_hole_count",
    "fallback_hole_area_ratio",
    "regularity_gain_vs_anchor",
    "mrr_gain_vs_anchor",
    "hull_gap_reduction_vs_anchor",
    "added_area_sum",
    "attraction_ratio_sum",
    "shared_edge_sum_m",
    "min_shared_edge_m",
    "min_completion_regularity",
    "max_completion_regularity_drop",
    "max_merge_regularity_loss_vs_anchor",
    "strong_completion_min_regularity",
    "strong_completion_max_hull_gap",
    "strong_completion_max_hole_area_ratio",
    "secondary_direct_min_ratio_of_best",
    "secondary_direct_min_attraction_ratio",
    "max_completion_hull_gap",
    "max_completion_hole_area_ratio",
    "large_anchor_area_threshold",
    "large_anchor_min_shared_edge",
    "large_anchor_min_attraction_ratio",
    "large_anchor_max_regularity_drop",
    "large_anchor_max_hull_gap_increase",
    "qualifying_council_min_anchor_overlap_m2",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / (float(den) if float(den) else 1.0)


def _ids_text(values: set[int] | list[int] | tuple[int, ...]) -> str:
    return "|".join(str(int(value)) for value in sorted(int(v) for v in values))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _empty_owned() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in OWNED_COLUMNS})


def _empty_fallback_gdf(crs: object | None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {column: pd.Series(dtype=object) for column in FALLBACK_OUTPUT_COLUMNS},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _iter_polygons(geom: Any):
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        yield geom
    elif geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _ring_area(ring: Any) -> float:
    try:
        return float(abs(Polygon(ring).area))
    except Exception:
        return 0.0


def _hole_metrics(geom: Any) -> dict[str, float]:
    hole_count = 0
    hole_area = 0.0
    for poly in _iter_polygons(geom) or []:
        for ring in poly.interiors:
            hole_count += 1
            hole_area += _ring_area(ring)
    return {"hole_count": float(hole_count), "hole_area": float(hole_area)}


def _mrr_dimensions(geom: Any) -> dict[str, float]:
    mrr = shapely.minimum_rotated_rectangle(geom)
    if not hasattr(mrr, "exterior"):
        return {
            "mrr_min_side": 0.0,
            "mrr_max_side": 0.0,
            "mrr_aspect_ratio": 1.0,
            "mrr_perimeter": 0.0,
            "mrr_orientation_deg": 0.0,
        }
    coords = list(mrr.exterior.coords)
    lengths: list[float] = []
    angles: list[float] = []
    for start, end in zip(coords, coords[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        lengths.append(length)
        angles.append(math.degrees(math.atan2(dy, dx)) % 180.0)
    if not lengths:
        return {
            "mrr_min_side": 0.0,
            "mrr_max_side": 0.0,
            "mrr_aspect_ratio": 1.0,
            "mrr_perimeter": 0.0,
            "mrr_orientation_deg": 0.0,
        }
    min_side = min(lengths)
    max_side = max(lengths)
    return {
        "mrr_min_side": float(min_side),
        "mrr_max_side": float(max_side),
        "mrr_aspect_ratio": _safe_ratio(max_side, min_side),
        "mrr_perimeter": float(sum(lengths)),
        "mrr_orientation_deg": float(angles[int(np.argmax(lengths))]),
    }


def _angle_delta_to_mrr_axis(angle: float, reference_angle: float) -> float:
    return abs(((angle - reference_angle + 45.0) % 90.0) - 45.0)


def _orthogonality_metrics(geom: Any, reference_angle: float) -> dict[str, float]:
    total = 0.0
    within_10 = 0.0
    within_20 = 0.0
    for poly in _iter_polygons(geom) or []:
        rings = [poly.exterior, *list(poly.interiors)]
        for ring in rings:
            coords = list(ring.coords)
            for start, end in zip(coords, coords[1:]):
                dx = float(end[0] - start[0])
                dy = float(end[1] - start[1])
                length = math.hypot(dx, dy)
                if length <= 1e-9:
                    continue
                total += length
                delta = _angle_delta_to_mrr_axis(math.degrees(math.atan2(dy, dx)) % 180.0, reference_angle)
                if delta <= 10.0:
                    within_10 += length
                if delta <= 20.0:
                    within_20 += length
    return {
        "orthogonal_len_ratio_10deg": _safe_ratio(within_10, total),
        "orthogonal_len_ratio_20deg": _safe_ratio(within_20, total),
    }


def _shape_metrics(geom: Any) -> dict[str, float]:
    if geom is None or geom.is_empty:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "mrr_ratio": 0.0,
            "hull_gap_ratio": 1.0,
            "compactness": 0.0,
            "regularity_score": 0.0,
            "hole_count": 0.0,
            "hole_area_ratio": 0.0,
        }
    area = float(shapely.area(geom))
    perimeter = float(shapely.length(geom))
    mrr_geom = shapely.minimum_rotated_rectangle(geom)
    mrr_area = float(shapely.area(mrr_geom)) or 1.0
    hull = shapely.convex_hull(geom)
    hull_area = float(shapely.area(hull)) or 1.0
    hull_perimeter = float(shapely.length(hull))
    envelope_area = float(shapely.area(shapely.envelope(geom))) or 1.0
    mrr = _mrr_dimensions(geom)
    holes = _hole_metrics(geom)
    orthogonal = _orthogonality_metrics(geom, mrr["mrr_orientation_deg"])
    perimeter_mrr_ratio = _safe_ratio(perimeter, mrr["mrr_perimeter"])
    perimeter_hull_ratio = _safe_ratio(perimeter, hull_perimeter)
    convexity = _safe_ratio(area, hull_area)
    mrr_ratio = _safe_ratio(area, mrr_area)
    perimeter_fit = min(_safe_ratio(1.0, perimeter_mrr_ratio), 1.0)
    regularity_score = (
        0.35 * mrr_ratio
        + 0.25 * convexity
        + 0.20 * perimeter_fit
        + 0.20 * orthogonal["orthogonal_len_ratio_10deg"]
    )
    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "mrr_ratio": float(mrr_ratio),
        "mrr_gap_ratio": float(max(mrr_area - area, 0.0) / (area or 1.0)),
        "hull_gap_ratio": float(max(hull_area - area, 0.0) / (area or 1.0)),
        "convexity": float(convexity),
        "bbox_fill_ratio": float(_safe_ratio(area, envelope_area)),
        "compactness": float(4.0 * math.pi * area / ((perimeter * perimeter) or 1.0)),
        "perimeter_mrr_ratio": float(perimeter_mrr_ratio),
        "perimeter_hull_ratio": float(perimeter_hull_ratio),
        "notch_index": float(max(hull_area - area, 0.0) / (area or 1.0) * max(perimeter_hull_ratio - 1.0, 0.0)),
        "hole_count": float(holes["hole_count"]),
        "hole_area_ratio": float(_safe_ratio(holes["hole_area"], area)),
        "regularity_score": float(regularity_score),
    }


def _plot_eligible(theme: object) -> bool:
    text = str(theme or "").lower()
    has_plot = "building" in text or "land" in text
    is_roadish = "road" in text or "track" in text or "path" in text
    return bool(has_plot and not is_roadish)


def _role(theme: object) -> str:
    text = str(theme or "").lower()
    if "building" in text:
        return "building"
    if "land" in text:
        return "land"
    return "other"


def _read_wfs(path: Path, layer: str) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading WFS clean: {path}:{layer}")
    cols = ["clean_fid", "source_fid", "Theme", "raw_role", "clean_area"]
    wfs = pyogrio.read_dataframe(path, layer=layer, columns=cols)
    wfs = wfs[wfs.geometry.notna() & ~wfs.geometry.is_empty].copy()
    wfs["clean_fid"] = wfs["clean_fid"].astype("int64")
    wfs["source_fid"] = wfs["source_fid"].fillna(wfs["clean_fid"]).astype("int64")
    wfs["plot_eligible"] = wfs["Theme"].map(_plot_eligible)
    wfs["anchor_role"] = wfs["Theme"].map(_role)
    _log(f"[INFO] WFS clean rows={len(wfs):,}; plot_eligible={int(wfs['plot_eligible'].sum()):,}")
    return wfs


def _read_anchors(path: Path, layer: str) -> gpd.GeoDataFrame:
    _log(f"[INFO] Reading anchors: {path}:{layer}")
    anchors = pyogrio.read_dataframe(
        path,
        layer=layer,
        columns=["clean_fid", "source_fid", "Theme", "anchor_uprn_count"],
        fid_as_index=True,
    )
    anchors = anchors[anchors.geometry.notna() & ~anchors.geometry.is_empty].copy()
    anchors.index.name = "anchor_fid"
    anchors = anchors.reset_index()
    anchors = anchors.rename(columns={"clean_fid": "anchor_clean_fid", "source_fid": "anchor_source_fid"})
    anchors["anchor_fid"] = anchors["anchor_fid"].astype("int64")
    anchors["anchor_clean_fid"] = anchors["anchor_clean_fid"].astype("int64")
    anchors["anchor_source_fid"] = anchors["anchor_source_fid"].astype("int64")
    anchors["anchor_perimeter"] = np.asarray(shapely.length(anchors.geometry.to_numpy()), dtype="float64")
    _log(f"[INFO] Anchors={len(anchors):,}")
    return anchors


def _read_covered_anchor_clean_ids(path: Path, layer: str) -> set[int]:
    _log(f"[INFO] Reading qualifying council single-anchor coverage: {path}:{layer}")
    if not path.exists():
        _log("[INFO] Council single-anchor coverage is missing; treating as empty.")
        return set()
    covered = pyogrio.read_dataframe(path, layer=layer, columns=["anchor_clean_fid"], read_geometry=False)
    if "anchor_clean_fid" not in covered.columns:
        _log("[INFO] Council single-anchor coverage has no anchor_clean_fid column; treating as empty.")
        return set()
    out = {int(value) for value in covered["anchor_clean_fid"].dropna().astype("int64").to_numpy()}
    _log(f"[INFO] Unique covered anchors={len(out):,}")
    return out


def _add_uprn_counts(wfs: gpd.GeoDataFrame, uprn_path: Path, uprn_layer: str) -> gpd.GeoDataFrame:
    if wfs.empty:
        out = wfs.copy()
        out["uprn_count"] = pd.Series(dtype="int64")
        return out
    bbox = tuple(float(v) for v in wfs.total_bounds)
    _log(f"[INFO] Reading UPRN points in WFS bbox: {bbox}")
    uprn = pyogrio.read_dataframe(uprn_path, layer=uprn_layer, columns=["UPRN"], bbox=bbox)
    uprn = uprn[uprn.geometry.notna() & ~uprn.geometry.is_empty].copy()
    if uprn.crs != wfs.crs:
        uprn = uprn.to_crs(wfs.crs)
    _log(f"[INFO] UPRN points in bbox={len(uprn):,}")
    _log("[INFO] Spatial join UPRN points to WFS clean polygons")
    joined = gpd.sjoin(
        uprn[["UPRN", "geometry"]],
        wfs[["clean_fid", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    counts = joined.groupby("clean_fid")["UPRN"].nunique().astype("int64")
    out = wfs.copy()
    out["uprn_count"] = out["clean_fid"].map(counts).fillna(0).astype("int64")
    _log(
        f"[INFO] WFS polygons with UPRN={int(out['uprn_count'].gt(0).sum()):,}; "
        f"zero UPRN={int(out['uprn_count'].eq(0).sum()):,}"
    )
    return out


def _shared_edge(left: Any, right: Any) -> float:
    try:
        return float(shapely.length(shapely.intersection(shapely.boundary(left), shapely.boundary(right))))
    except Exception:
        return 0.0


def _build_direct_owners(
    anchors: gpd.GeoDataFrame,
    zero: gpd.GeoDataFrame,
    *,
    min_shared_edge: float,
    secondary_owner_anchor_fids: set[int] | None = None,
    secondary_direct_min_ratio_of_best: float = 1.0,
    secondary_direct_min_attraction_ratio: float = 0.0,
) -> pd.DataFrame:
    _log("[INFO] Building k0 direct zero-UPRN ownership by anchor shared-edge attraction")
    if anchors.empty or zero.empty:
        return _empty_owned()
    joined = gpd.sjoin(
        zero[["clean_fid", "source_fid", "geometry"]],
        anchors[["anchor_fid", "anchor_clean_fid", "anchor_perimeter", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return _empty_owned()
    right_pos = joined["index_right"].to_numpy(dtype="int64")
    left_geoms = np.asarray(joined.geometry.to_numpy(), dtype=object)
    right_geoms = np.asarray(anchors.geometry.iloc[right_pos].to_numpy(), dtype=object)
    joined["shared_edge_m"] = np.asarray(
        shapely.length(shapely.intersection(shapely.boundary(left_geoms), shapely.boundary(right_geoms))),
        dtype="float64",
    )
    joined = joined[joined["shared_edge_m"].ge(float(min_shared_edge))].copy()
    if joined.empty:
        return _empty_owned()
    joined["anchor_perimeter"] = joined["index_right"].map(anchors["anchor_perimeter"])
    joined["attraction_ratio"] = joined["shared_edge_m"] / joined["anchor_perimeter"].replace(0.0, np.nan)
    joined["attraction_ratio"] = joined["attraction_ratio"].fillna(0.0)
    joined = joined.sort_values(
        ["clean_fid", "attraction_ratio", "shared_edge_m", "anchor_fid"],
        ascending=[True, False, False, True],
    )
    primary = joined.drop_duplicates("clean_fid", keep="first").copy()
    primary["claim_rank"] = "primary"
    claims = [primary]
    if secondary_owner_anchor_fids:
        best = primary[
            ["clean_fid", "anchor_fid", "attraction_ratio", "shared_edge_m"]
        ].rename(
            columns={
                "anchor_fid": "best_anchor_fid",
                "attraction_ratio": "best_attraction_ratio",
                "shared_edge_m": "best_shared_edge_m",
            }
        )
        secondary = joined.merge(best, on="clean_fid", how="inner", validate="many_to_one")
        secondary["ratio_of_best"] = secondary["attraction_ratio"] / secondary["best_attraction_ratio"].replace(
            0.0, np.nan
        )
        secondary["ratio_of_best"] = secondary["ratio_of_best"].fillna(0.0)
        secondary = secondary[
            secondary["anchor_fid"].astype(int).isin(secondary_owner_anchor_fids)
            & ~secondary["best_anchor_fid"].astype(int).isin(secondary_owner_anchor_fids)
            & secondary["anchor_fid"].astype(int).ne(secondary["best_anchor_fid"].astype(int))
            & secondary["ratio_of_best"].ge(float(secondary_direct_min_ratio_of_best))
            & secondary["attraction_ratio"].ge(float(secondary_direct_min_attraction_ratio))
        ].copy()
        if not secondary.empty:
            secondary = secondary.sort_values(
                ["clean_fid", "ratio_of_best", "attraction_ratio", "shared_edge_m", "anchor_fid"],
                ascending=[True, False, False, False, True],
            )
            secondary = secondary.drop_duplicates("clean_fid", keep="first").copy()
            secondary["claim_rank"] = "secondary"
            claims.append(secondary)
    owners = pd.concat(claims, ignore_index=True, sort=False)
    owners = owners.rename(
        columns={
            "clean_fid": "zero_clean_fid",
            "source_fid": "zero_source_fid",
            "anchor_fid": "owner_anchor_fid",
        }
    )
    owners["depth"] = 0
    owners["via_zero_clean_fid"] = -1
    secondary_count = int(owners["claim_rank"].eq("secondary").sum()) if "claim_rank" in owners else 0
    _log(
        f"[INFO] k0 edge pairs={len(joined):,}; owned k0 claims={len(owners):,}; "
        f"secondary claims={secondary_count:,}; owners={owners['owner_anchor_fid'].nunique():,}"
    )
    return owners[
        [
            "zero_clean_fid",
            "zero_source_fid",
            "owner_anchor_fid",
            "depth",
            "via_zero_clean_fid",
            "shared_edge_m",
            "attraction_ratio",
            "claim_rank",
        ]
    ].reset_index(drop=True)


def _build_indirect_owners(
    anchors: gpd.GeoDataFrame,
    zero: gpd.GeoDataFrame,
    direct: pd.DataFrame,
    *,
    min_shared_edge: float,
) -> pd.DataFrame:
    if direct.empty:
        return _empty_owned()
    _log("[INFO] Building k1 indirect zero-UPRN ownership through owned k0 polygons")
    direct_ids = set(direct["zero_clean_fid"].astype("int64").to_numpy())
    direct_zero = zero[zero["clean_fid"].isin(direct_ids)].copy().reset_index(drop=True)
    direct_zero = direct_zero.merge(
        direct[["zero_clean_fid", "owner_anchor_fid", "attraction_ratio"]].rename(
            columns={
                "zero_clean_fid": "parent_zero_clean_fid",
                "attraction_ratio": "parent_attraction_ratio",
            }
        ),
        left_on="clean_fid",
        right_on="parent_zero_clean_fid",
        how="inner",
        validate="one_to_many",
    )
    right_zero = zero[~zero["clean_fid"].isin(direct_ids)].copy().reset_index(drop=True)
    joined = gpd.sjoin(
        direct_zero[["clean_fid", "owner_anchor_fid", "parent_attraction_ratio", "geometry"]],
        right_zero[["clean_fid", "source_fid", "geometry"]],
        how="inner",
        predicate="intersects",
        lsuffix="parent",
        rsuffix="child",
    )
    if joined.empty:
        return _empty_owned()
    joined = joined.rename(
        columns={
            "clean_fid_parent": "via_zero_clean_fid",
            "clean_fid_child": "zero_clean_fid",
            "source_fid": "zero_source_fid",
        }
    )
    right_pos = joined["index_child"].to_numpy(dtype="int64")
    left_geoms = np.asarray(joined.geometry.to_numpy(), dtype=object)
    right_geoms = np.asarray(right_zero.geometry.iloc[right_pos].to_numpy(), dtype=object)
    joined["shared_edge_m"] = np.asarray(
        shapely.length(shapely.intersection(shapely.boundary(left_geoms), shapely.boundary(right_geoms))),
        dtype="float64",
    )
    joined = joined[joined["shared_edge_m"].ge(float(min_shared_edge))].copy()
    if joined.empty:
        return _empty_owned()
    perimeter_by_anchor = anchors.set_index("anchor_fid")["anchor_perimeter"].to_dict()
    joined["owner_anchor_perimeter"] = joined["owner_anchor_fid"].map(perimeter_by_anchor).fillna(0.0)
    joined["attraction_ratio"] = joined["shared_edge_m"] / joined["owner_anchor_perimeter"].replace(0.0, np.nan)
    joined["attraction_ratio"] = joined["attraction_ratio"].fillna(0.0)
    joined = joined.sort_values(
        [
            "zero_clean_fid",
            "attraction_ratio",
            "shared_edge_m",
            "parent_attraction_ratio",
            "owner_anchor_fid",
            "via_zero_clean_fid",
        ],
        ascending=[True, False, False, False, True, True],
    )
    owners = joined.drop_duplicates("zero_clean_fid", keep="first").copy()
    owners["depth"] = 1
    _log(
        f"[INFO] k1 edge pairs={len(joined):,}; owned k1 zero polygons={len(owners):,}; "
        f"owners={owners['owner_anchor_fid'].nunique():,}"
    )
    return owners[
        [
            "zero_clean_fid",
            "zero_source_fid",
            "owner_anchor_fid",
            "depth",
            "via_zero_clean_fid",
            "shared_edge_m",
            "attraction_ratio",
        ]
    ].reset_index(drop=True)


def _union_for_ids(anchor_geom: Any, ids: set[int], geom_by_clean: dict[int, Any]) -> Any:
    geoms = [anchor_geom]
    geoms.extend(geom_by_clean[int(clean_id)] for clean_id in sorted(ids) if int(clean_id) in geom_by_clean)
    geom = shapely.union_all(np.asarray(geoms, dtype=object))
    if geom is not None and not geom.is_empty and not bool(shapely.is_valid(geom)):
        geom = shapely.make_valid(geom)
    return geom


def _candidate_score(
    metrics: dict[str, float],
    *,
    attraction_sum: float,
    shared_edge_sum: float,
    added_count: int,
) -> tuple[float, float, float, float, float, int]:
    return (
        float(metrics.get("regularity_score", 0.0)),
        -float(metrics.get("hull_gap_ratio", 0.0)),
        float(metrics.get("mrr_ratio", 0.0)),
        float(attraction_sum),
        float(shared_edge_sum),
        -int(added_count),
    )


def _is_completion_acceptable(
    metrics: dict[str, float],
    *,
    min_regularity: float,
    max_hull_gap: float,
    max_hole_area_ratio: float,
) -> bool:
    return bool(
        float(metrics.get("regularity_score", 0.0)) >= float(min_regularity)
        and float(metrics.get("hull_gap_ratio", 0.0)) <= float(max_hull_gap)
        and float(metrics.get("hole_area_ratio", 0.0)) <= float(max_hole_area_ratio)
    )


def _is_strong_completion(
    metrics: dict[str, float],
    *,
    min_regularity: float,
    max_hull_gap: float,
    max_hole_area_ratio: float,
) -> bool:
    return bool(
        float(metrics.get("regularity_score", 0.0)) >= float(min_regularity)
        and float(metrics.get("hull_gap_ratio", 0.0)) <= float(max_hull_gap)
        and float(metrics.get("hole_area_ratio", 0.0)) <= float(max_hole_area_ratio)
    )


def _completion_score(
    metrics: dict[str, float],
    *,
    added_area_sum: float,
    attraction_sum: float,
    shared_edge_sum: float,
    added_count: int,
) -> tuple[float, int, float, float, float, float, float]:
    return (
        float(added_area_sum),
        int(added_count),
        float(metrics.get("regularity_score", 0.0)),
        -float(metrics.get("hull_gap_ratio", 0.0)),
        float(metrics.get("mrr_ratio", 0.0)),
        float(attraction_sum),
        float(shared_edge_sum),
    )


def _anchor_only_result(
    anchor_geom: Any,
    anchor_metrics: dict[str, float],
    *,
    selection_method: str,
    candidate_pool_k0_count: int = 0,
    candidate_pool_k1_count: int = 0,
    evaluated_candidate_count: int = 0,
    candidate_best_regularity_score: float | None = None,
    min_band_regularity_score: float = 0.0,
    anchor_loss_regularity_floor: float = 0.0,
) -> dict[str, Any]:
    return {
        "selected_clean_ids": set(),
        "selected_source_ids": set(),
        "selected_k0_ids": set(),
        "selected_k1_ids": set(),
        "geometry": anchor_geom,
        "selection_method": selection_method,
        "selection_policy": "anchor_only",
        "attraction_sum": 0.0,
        "shared_edge_sum": 0.0,
        "added_area_sum": 0.0,
        "candidate_pool_k0_count": int(candidate_pool_k0_count),
        "candidate_pool_k1_count": int(candidate_pool_k1_count),
        "evaluated_candidate_count": int(evaluated_candidate_count),
        "candidate_best_regularity_score": float(
            anchor_metrics.get("regularity_score", 0.0)
            if candidate_best_regularity_score is None
            else candidate_best_regularity_score
        ),
        "min_band_regularity_score": float(min_band_regularity_score),
        "anchor_loss_regularity_floor": float(anchor_loss_regularity_floor),
        "metrics": anchor_metrics,
        "anchor_metrics": anchor_metrics,
    }


def _large_anchor_merge_acceptable(
    item: dict[str, Any],
    anchor_metrics: dict[str, float],
    *,
    min_shared_edge: float,
    min_attraction_ratio: float,
    max_regularity_drop: float,
    max_hull_gap_increase: float,
) -> bool:
    metrics = item["metrics"]
    regularity_drop = float(anchor_metrics.get("regularity_score", 0.0)) - float(
        metrics.get("regularity_score", 0.0)
    )
    hull_gap_increase = float(metrics.get("hull_gap_ratio", 0.0)) - float(anchor_metrics.get("hull_gap_ratio", 0.0))
    improves_shape = regularity_drop <= 0.0 and hull_gap_increase <= 0.0
    has_strong_edge = (
        float(item.get("shared_edge_sum", 0.0)) >= float(min_shared_edge)
        and float(item.get("attraction_sum", 0.0)) >= float(min_attraction_ratio)
    )
    has_small_shape_loss = (
        regularity_drop <= float(max_regularity_drop)
        and hull_gap_increase <= float(max_hull_gap_increase)
    )
    return bool(improves_shape or (has_strong_edge and has_small_shape_loss))


def _best_merge_for_anchor(
    anchor_row: pd.Series,
    owned: pd.DataFrame,
    geom_by_clean: dict[int, Any],
    source_by_clean: dict[int, int],
    area_by_clean: dict[int, float],
    *,
    max_enum_nodes: int,
    top_direct: int,
    top_indirect_per_direct: int,
    min_completion_regularity: float,
    max_completion_regularity_drop: float,
    max_merge_regularity_loss_vs_anchor: float,
    max_completion_hull_gap: float,
    max_completion_hole_area_ratio: float,
    strong_completion_min_regularity: float,
    strong_completion_max_hull_gap: float,
    strong_completion_max_hole_area_ratio: float,
    large_anchor_area_threshold: float,
    large_anchor_min_shared_edge: float,
    large_anchor_min_attraction_ratio: float,
    large_anchor_max_regularity_drop: float,
    large_anchor_max_hull_gap_increase: float,
) -> dict[str, Any]:
    anchor_geom = anchor_row.geometry
    anchor_metrics = _shape_metrics(anchor_geom)
    anchor_fid = int(anchor_row.anchor_fid)
    owned = owned[owned["owner_anchor_fid"].astype(int).eq(anchor_fid)].copy()
    if owned.empty:
        return _anchor_only_result(anchor_geom, anchor_metrics, selection_method="anchor_only_no_candidates")

    direct = owned[owned["depth"].astype(int).eq(0)].copy()
    indirect = owned[owned["depth"].astype(int).eq(1)].copy()
    if not direct.empty:
        direct = direct.sort_values(["attraction_ratio", "shared_edge_m", "zero_clean_fid"], ascending=[False, False, True])
        direct = direct.head(int(top_direct)).copy()
    direct_ids = set(direct["zero_clean_fid"].astype(int).to_numpy())
    if not indirect.empty:
        indirect = indirect[indirect["via_zero_clean_fid"].astype(int).isin(direct_ids)].copy()
        indirect = (
            indirect.sort_values(["via_zero_clean_fid", "attraction_ratio", "shared_edge_m"], ascending=[True, False, False])
            .groupby("via_zero_clean_fid", sort=False)
            .head(int(top_indirect_per_direct))
            .copy()
        )
    indirect_ids = set(indirect["zero_clean_fid"].astype(int).to_numpy())

    owned_small = pd.concat([direct, indirect], ignore_index=True)
    attr_by_id = dict(zip(owned_small["zero_clean_fid"].astype(int), owned_small["attraction_ratio"].astype(float)))
    shared_by_id = dict(zip(owned_small["zero_clean_fid"].astype(int), owned_small["shared_edge_m"].astype(float)))
    parent_by_id = {
        int(row.zero_clean_fid): int(row.via_zero_clean_fid)
        for row in indirect[["zero_clean_fid", "via_zero_clean_fid"]].itertuples(index=False)
    }
    children_by_direct: dict[int, list[int]] = {}
    for child_id, parent_id in parent_by_id.items():
        children_by_direct.setdefault(int(parent_id), []).append(int(child_id))

    candidate_sets: list[tuple[str, set[int]]] = []
    for direct_id in sorted(direct_ids, key=lambda fid: (-attr_by_id.get(fid, 0.0), fid)):
        children = set(children_by_direct.get(int(direct_id), []))
        candidate_sets.append(("single_k0", {int(direct_id)}))
        if children:
            candidate_sets.append(("single_k0_all_k1", {int(direct_id), *children}))
            for child_id in sorted(children, key=lambda fid: (-attr_by_id.get(fid, 0.0), fid)):
                candidate_sets.append(("single_k0_single_k1", {int(direct_id), int(child_id)}))
    if direct_ids:
        candidate_sets.append(("all_k0", set(direct_ids)))
    if direct_ids or indirect_ids:
        candidate_sets.append(("all_k0_k1", set(direct_ids) | set(indirect_ids)))

    enum_nodes = list(sorted(direct_ids | indirect_ids, key=lambda fid: (-attr_by_id.get(fid, 0.0), fid)))
    if 0 < len(enum_nodes) <= int(max_enum_nodes):
        n = len(enum_nodes)
        for mask in range(1, 1 << n):
            selected = {enum_nodes[idx] for idx in range(n) if mask & (1 << idx)}
            if any(parent_by_id.get(child_id) not in selected for child_id in selected if child_id in parent_by_id):
                continue
            candidate_sets.append(("enum", selected))

    # Greedy regularity path. This gives a scalable fallback when there are many
    # nearby zero-UPRN polygons but exhaustive enumeration would be noisy.
    selected: set[int] = set()
    available_direct = set(direct_ids)
    greedy_method_added = False
    while available_direct or any(parent in selected for parent in parent_by_id.values()):
        available = set(available_direct)
        available.update(child for child, parent in parent_by_id.items() if parent in selected and child not in selected)
        if not available:
            break
        best_next: tuple[tuple[float, float, float, float, float, int], int] | None = None
        for clean_id in available:
            test_ids = set(selected)
            test_ids.add(int(clean_id))
            geom = _union_for_ids(anchor_geom, test_ids, geom_by_clean)
            metrics = _shape_metrics(geom)
            score = _candidate_score(
                metrics,
                attraction_sum=sum(attr_by_id.get(fid, 0.0) for fid in test_ids),
                shared_edge_sum=sum(shared_by_id.get(fid, 0.0) for fid in test_ids),
                added_count=len(test_ids),
            )
            if best_next is None or score > best_next[0]:
                best_next = (score, int(clean_id))
        if best_next is None:
            break
        _, chosen = best_next
        selected.add(chosen)
        available_direct.discard(chosen)
        greedy_method_added = True
        candidate_sets.append(("greedy", set(selected)))
        # Stop when every dependency-valid candidate has been considered.
        if len(selected) >= len(enum_nodes):
            break
    if greedy_method_added:
        candidate_sets.append(("greedy_final", set(selected)))

    deduped: dict[str, tuple[str, set[int]]] = {}
    for method, ids in candidate_sets:
        ids = {int(fid) for fid in ids if int(fid) in geom_by_clean}
        if not ids:
            continue
        if any(parent_by_id.get(child_id) not in ids for child_id in ids if child_id in parent_by_id):
            continue
        key = _ids_text(ids)
        deduped[key] = (method, ids)

    evaluated: list[dict[str, Any]] = []
    for method, ids in deduped.values():
        geom = _union_for_ids(anchor_geom, ids, geom_by_clean)
        metrics = _shape_metrics(geom)
        attraction_sum = sum(attr_by_id.get(fid, 0.0) for fid in ids)
        shared_edge_sum = sum(shared_by_id.get(fid, 0.0) for fid in ids)
        added_area_sum = sum(area_by_clean.get(fid, 0.0) for fid in ids)
        evaluated.append(
            {
                "method": method,
                "ids": set(ids),
                "geometry": geom,
                "metrics": metrics,
                "attraction_sum": float(attraction_sum),
                "shared_edge_sum": float(shared_edge_sum),
                "added_area_sum": float(added_area_sum),
                "regularity_score": _candidate_score(
                    metrics,
                    attraction_sum=attraction_sum,
                    shared_edge_sum=shared_edge_sum,
                    added_count=len(ids),
                ),
                "completion_score": _completion_score(
                    metrics,
                    added_area_sum=added_area_sum,
                    attraction_sum=attraction_sum,
                    shared_edge_sum=shared_edge_sum,
                    added_count=len(ids),
                ),
                "completion_acceptable": _is_completion_acceptable(
                    metrics,
                    min_regularity=float(min_completion_regularity),
                    max_hull_gap=float(max_completion_hull_gap),
                    max_hole_area_ratio=float(max_completion_hole_area_ratio),
                ),
                "strong_completion": _is_strong_completion(
                    metrics,
                    min_regularity=float(strong_completion_min_regularity),
                    max_hull_gap=float(strong_completion_max_hull_gap),
                    max_hole_area_ratio=float(strong_completion_max_hole_area_ratio),
                ),
            }
        )

    if not evaluated:
        return _anchor_only_result(
            anchor_geom,
            anchor_metrics,
            selection_method="anchor_only_no_valid_merge",
            candidate_pool_k0_count=len(direct_ids),
            candidate_pool_k1_count=len(indirect_ids),
        )

    best_regularity = max(float(item["metrics"].get("regularity_score", 0.0)) for item in evaluated)
    anchor_regularity = float(anchor_metrics.get("regularity_score", 0.0))
    anchor_loss_floor = anchor_regularity - float(max_merge_regularity_loss_vs_anchor)
    min_band_regularity = max(
        float(min_completion_regularity),
        float(best_regularity) - float(max_completion_regularity_drop),
    )
    acceptable = [
        item
        for item in evaluated
        if bool(item["completion_acceptable"])
        and float(item["metrics"].get("regularity_score", 0.0)) >= float(min_band_regularity)
        and (
            float(item["metrics"].get("regularity_score", 0.0)) >= float(anchor_loss_floor)
            or bool(item["strong_completion"])
        )
    ]
    if acceptable:
        large_anchor = float(anchor_metrics.get("area", 0.0)) >= float(large_anchor_area_threshold)
        if large_anchor:
            acceptable = [
                item
                for item in acceptable
                if _large_anchor_merge_acceptable(
                    item,
                    anchor_metrics,
                    min_shared_edge=float(large_anchor_min_shared_edge),
                    min_attraction_ratio=float(large_anchor_min_attraction_ratio),
                    max_regularity_drop=float(large_anchor_max_regularity_drop),
                    max_hull_gap_increase=float(large_anchor_max_hull_gap_increase),
                )
            ]
        if large_anchor and not acceptable:
            return _anchor_only_result(
                anchor_geom,
                anchor_metrics,
                selection_method="anchor_only_large_anchor_guard",
                candidate_pool_k0_count=len(direct_ids),
                candidate_pool_k1_count=len(indirect_ids),
                evaluated_candidate_count=len(evaluated),
                candidate_best_regularity_score=best_regularity,
                min_band_regularity_score=min_band_regularity,
                anchor_loss_regularity_floor=anchor_loss_floor,
            )
        chosen_item = max(acceptable, key=lambda item: item["completion_score"])
        if float(chosen_item["metrics"].get("regularity_score", 0.0)) >= float(anchor_loss_floor):
            selection_policy = "regularity_band_completion"
        else:
            selection_policy = "strong_completion"
    else:
        return _anchor_only_result(
            anchor_geom,
            anchor_metrics,
            selection_method="anchor_only_regularity_guard",
            candidate_pool_k0_count=len(direct_ids),
            candidate_pool_k1_count=len(indirect_ids),
            evaluated_candidate_count=len(evaluated),
            candidate_best_regularity_score=best_regularity,
            min_band_regularity_score=min_band_regularity,
            anchor_loss_regularity_floor=anchor_loss_floor,
        )

    method = str(chosen_item["method"])
    selected_ids = set(chosen_item["ids"])
    geom = chosen_item["geometry"]
    metrics = chosen_item["metrics"]
    selected_k0 = selected_ids & direct_ids
    selected_k1 = selected_ids & indirect_ids
    return {
        "selected_clean_ids": selected_ids,
        "selected_source_ids": {int(source_by_clean.get(fid, fid)) for fid in selected_ids},
        "selected_k0_ids": selected_k0,
        "selected_k1_ids": selected_k1,
        "geometry": geom,
        "selection_method": method,
        "selection_policy": selection_policy,
        "attraction_sum": float(chosen_item["attraction_sum"]),
        "shared_edge_sum": float(chosen_item["shared_edge_sum"]),
        "added_area_sum": float(chosen_item["added_area_sum"]),
        "candidate_pool_k0_count": int(len(direct_ids)),
        "candidate_pool_k1_count": int(len(indirect_ids)),
        "evaluated_candidate_count": int(len(evaluated)),
        "candidate_best_regularity_score": float(best_regularity),
        "min_band_regularity_score": float(min_band_regularity),
        "anchor_loss_regularity_floor": float(anchor_loss_floor),
        "metrics": metrics,
        "anchor_metrics": anchor_metrics,
    }


def build_fallback(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    wfs = _read_wfs(Path(args.wfs_gpkg), str(args.wfs_layer))
    anchors = _read_anchors(Path(args.anchor_gpkg), str(args.anchor_layer))
    covered_anchor_clean_ids = _read_covered_anchor_clean_ids(
        Path(args.council_single_anchor_gpkg),
        str(args.council_single_anchor_layer),
    )
    wfs = _add_uprn_counts(wfs, Path(args.uprn_gpkg), str(args.uprn_layer))

    fallback_scope = str(getattr(args, "fallback_scope", "uncovered") or "uncovered")
    if fallback_scope == "all":
        fallback_anchors = anchors.copy()
        _log(f"[INFO] Fallback scope=all; fallback anchors={len(fallback_anchors):,}")
    else:
        fallback_anchors = anchors[~anchors["anchor_clean_fid"].isin(covered_anchor_clean_ids)].copy()
        _log(f"[INFO] Fallback anchors without qualifying council polygon={len(fallback_anchors):,}")

    zero = wfs[wfs["uprn_count"].eq(0) & wfs["plot_eligible"]].copy()
    _log(f"[INFO] Plot-eligible zero-UPRN WFS polygons={len(zero):,}")

    # Let every anchor compete for zero-UPRN polygons, then only emit rows for
    # fallback anchors. Otherwise covered neighbouring anchors cannot protect
    # their own edge pieces from being pulled into a fallback merge.
    owner_anchors = anchors.reset_index(drop=True)
    direct = _build_direct_owners(
        owner_anchors,
        zero,
        min_shared_edge=float(args.min_shared_edge),
        secondary_owner_anchor_fids={int(value) for value in fallback_anchors["anchor_fid"].astype("int64")},
        secondary_direct_min_ratio_of_best=float(args.secondary_direct_min_ratio_of_best),
        secondary_direct_min_attraction_ratio=float(args.secondary_direct_min_attraction_ratio),
    )
    indirect = _build_indirect_owners(owner_anchors, zero, direct, min_shared_edge=float(args.min_shared_edge))
    owned = pd.concat([direct, indirect], ignore_index=True, sort=False) if not direct.empty or not indirect.empty else _empty_owned()
    _log(f"[INFO] Owned k0/k1 zero-UPRN polygons across all anchors={len(owned):,}")

    geom_by_clean = dict(zip(wfs["clean_fid"].astype(int), wfs.geometry.to_numpy()))
    source_by_clean = dict(zip(wfs["clean_fid"].astype(int), wfs["source_fid"].astype(int)))
    area_by_clean = dict(zip(wfs["clean_fid"].astype(int), wfs["clean_area"].fillna(0.0).astype(float)))

    records: list[dict[str, Any]] = []
    total = len(fallback_anchors)
    owned_by_anchor = {int(anchor_id): group.copy() for anchor_id, group in owned.groupby("owner_anchor_fid", sort=False)}
    empty_owned = owned.iloc[0:0].copy()
    for offset, row in enumerate(fallback_anchors.itertuples(index=False), start=1):
        if offset == 1 or offset % int(args.log_every) == 0:
            _log(f"[INFO] Selecting fallback merge {offset:,}/{total:,}; output_rows={len(records):,}")
        anchor_series = pd.Series(row._asdict())
        anchor_series.geometry = row.geometry
        local_owned = owned_by_anchor.get(int(row.anchor_fid), empty_owned)
        result = _best_merge_for_anchor(
            anchor_series,
            local_owned,
            geom_by_clean,
            source_by_clean,
            area_by_clean,
            max_enum_nodes=int(args.max_enum_nodes),
            top_direct=int(args.top_direct),
            top_indirect_per_direct=int(args.top_indirect_per_direct),
            min_completion_regularity=float(args.min_completion_regularity),
            max_completion_regularity_drop=float(args.max_completion_regularity_drop),
            max_merge_regularity_loss_vs_anchor=float(args.max_merge_regularity_loss_vs_anchor),
            max_completion_hull_gap=float(args.max_completion_hull_gap),
            max_completion_hole_area_ratio=float(args.max_completion_hole_area_ratio),
            strong_completion_min_regularity=float(args.strong_completion_min_regularity),
            strong_completion_max_hull_gap=float(args.strong_completion_max_hull_gap),
            strong_completion_max_hole_area_ratio=float(args.strong_completion_max_hole_area_ratio),
            large_anchor_area_threshold=float(args.large_anchor_area_threshold),
            large_anchor_min_shared_edge=float(args.large_anchor_min_shared_edge),
            large_anchor_min_attraction_ratio=float(args.large_anchor_min_attraction_ratio),
            large_anchor_max_regularity_drop=float(args.large_anchor_max_regularity_drop),
            large_anchor_max_hull_gap_increase=float(args.large_anchor_max_hull_gap_increase),
        )
        selected_ids = set(result.get("selected_clean_ids", set()))
        selected_source_ids = {int(source_by_clean.get(fid, fid)) for fid in selected_ids}
        selected_k0 = set(result.get("selected_k0_ids", set()))
        selected_k1 = set(result.get("selected_k1_ids", set()))
        metrics = result["metrics"]
        anchor_metrics = result["anchor_metrics"]
        records.append(
            {
                "anchor_fid": int(row.anchor_fid),
                "anchor_clean_fid": int(row.anchor_clean_fid),
                "anchor_source_fid": int(row.anchor_source_fid),
                "anchor_theme": str(row.Theme),
                "anchor_uprn_count": int(row.anchor_uprn_count),
                "fallback_reason": (
                    "all_scope_covered_by_council"
                    if fallback_scope == "all" and int(row.anchor_clean_fid) in covered_anchor_clean_ids
                    else "no_qualifying_council_single_anchor"
                ),
                "selection_method": str(result.get("selection_method", "")),
                "selection_policy": str(result.get("selection_policy", "")),
                "candidate_pool_k0_count": int(result.get("candidate_pool_k0_count", 0)),
                "candidate_pool_k1_count": int(result.get("candidate_pool_k1_count", 0)),
                "evaluated_candidate_count": int(result.get("evaluated_candidate_count", 0)),
                "added_clean_count": int(len(selected_ids)),
                "added_k0_count": int(len(selected_k0)),
                "added_k1_count": int(len(selected_k1)),
                "added_clean_fids": _ids_text(selected_ids),
                "added_source_fids": _ids_text(selected_source_ids),
                "added_k0_clean_fids": _ids_text(selected_k0),
                "added_k1_clean_fids": _ids_text(selected_k1),
                "anchor_area": float(anchor_metrics.get("area", 0.0)),
                "anchor_regularity_score": float(anchor_metrics.get("regularity_score", 0.0)),
                "anchor_mrr_ratio": float(anchor_metrics.get("mrr_ratio", 0.0)),
                "anchor_hull_gap_ratio": float(anchor_metrics.get("hull_gap_ratio", 0.0)),
                "candidate_best_regularity_score": float(result.get("candidate_best_regularity_score", 0.0)),
                "min_band_regularity_score": float(result.get("min_band_regularity_score", 0.0)),
                "anchor_loss_regularity_floor": float(result.get("anchor_loss_regularity_floor", 0.0)),
                "fallback_area": float(metrics.get("area", 0.0)),
                "fallback_perimeter": float(metrics.get("perimeter", 0.0)),
                "fallback_regularity_score": float(metrics.get("regularity_score", 0.0)),
                "fallback_mrr_ratio": float(metrics.get("mrr_ratio", 0.0)),
                "fallback_hull_gap_ratio": float(metrics.get("hull_gap_ratio", 0.0)),
                "fallback_compactness": float(metrics.get("compactness", 0.0)),
                "fallback_hole_count": float(metrics.get("hole_count", 0.0)),
                "fallback_hole_area_ratio": float(metrics.get("hole_area_ratio", 0.0)),
                "regularity_gain_vs_anchor": float(
                    metrics.get("regularity_score", 0.0) - anchor_metrics.get("regularity_score", 0.0)
                ),
                "mrr_gain_vs_anchor": float(metrics.get("mrr_ratio", 0.0) - anchor_metrics.get("mrr_ratio", 0.0)),
                "hull_gap_reduction_vs_anchor": float(
                    anchor_metrics.get("hull_gap_ratio", 0.0) - metrics.get("hull_gap_ratio", 0.0)
                ),
                "added_area_sum": float(result.get("added_area_sum", 0.0)),
                "attraction_ratio_sum": float(result.get("attraction_sum", 0.0)),
                "shared_edge_sum_m": float(result.get("shared_edge_sum", 0.0)),
                "min_shared_edge_m": float(args.min_shared_edge),
                "min_completion_regularity": float(args.min_completion_regularity),
                "max_completion_regularity_drop": float(args.max_completion_regularity_drop),
                "max_merge_regularity_loss_vs_anchor": float(args.max_merge_regularity_loss_vs_anchor),
                "strong_completion_min_regularity": float(args.strong_completion_min_regularity),
                "strong_completion_max_hull_gap": float(args.strong_completion_max_hull_gap),
                "strong_completion_max_hole_area_ratio": float(args.strong_completion_max_hole_area_ratio),
                "secondary_direct_min_ratio_of_best": float(args.secondary_direct_min_ratio_of_best),
                "secondary_direct_min_attraction_ratio": float(args.secondary_direct_min_attraction_ratio),
                "max_completion_hull_gap": float(args.max_completion_hull_gap),
                "max_completion_hole_area_ratio": float(args.max_completion_hole_area_ratio),
                "large_anchor_area_threshold": float(args.large_anchor_area_threshold),
                "large_anchor_min_shared_edge": float(args.large_anchor_min_shared_edge),
                "large_anchor_min_attraction_ratio": float(args.large_anchor_min_attraction_ratio),
                "large_anchor_max_regularity_drop": float(args.large_anchor_max_regularity_drop),
                "large_anchor_max_hull_gap_increase": float(args.large_anchor_max_hull_gap_increase),
                "qualifying_council_min_anchor_overlap_m2": float(args.council_min_anchor_overlap_area),
                "geometry": result["geometry"],
            }
        )

    out = (
        gpd.GeoDataFrame(records, geometry="geometry", crs=wfs.crs)
        if records
        else _empty_fallback_gdf(wfs.crs)
    )
    summary = {
        "wfs_gpkg": str(args.wfs_gpkg),
        "anchor_gpkg": str(args.anchor_gpkg),
        "council_single_anchor_gpkg": str(args.council_single_anchor_gpkg),
        "output_gpkg": str(args.output_gpkg),
        "output_layer": str(args.output_layer),
        "fallback_scope": fallback_scope,
        "total_anchors": int(len(anchors)),
        "covered_anchors": int(len(covered_anchor_clean_ids)),
        "fallback_anchor_rows": int(len(out)),
        "zero_plot_eligible_rows": int(len(zero)),
        "direct_owned_rows_all_anchors": int(len(direct)),
        "secondary_direct_claim_rows": (
            int(direct["claim_rank"].eq("secondary").sum()) if not direct.empty and "claim_rank" in direct else 0
        ),
        "indirect_owned_rows_all_anchors": int(len(indirect)),
        "owned_rows_all_anchors": int(len(owned)),
        "min_shared_edge": float(args.min_shared_edge),
        "min_completion_regularity": float(args.min_completion_regularity),
        "max_completion_regularity_drop": float(args.max_completion_regularity_drop),
        "max_merge_regularity_loss_vs_anchor": float(args.max_merge_regularity_loss_vs_anchor),
        "strong_completion_min_regularity": float(args.strong_completion_min_regularity),
        "strong_completion_max_hull_gap": float(args.strong_completion_max_hull_gap),
        "strong_completion_max_hole_area_ratio": float(args.strong_completion_max_hole_area_ratio),
        "secondary_direct_min_ratio_of_best": float(args.secondary_direct_min_ratio_of_best),
        "secondary_direct_min_attraction_ratio": float(args.secondary_direct_min_attraction_ratio),
        "max_completion_hull_gap": float(args.max_completion_hull_gap),
        "max_completion_hole_area_ratio": float(args.max_completion_hole_area_ratio),
        "large_anchor_area_threshold": float(args.large_anchor_area_threshold),
        "large_anchor_min_shared_edge": float(args.large_anchor_min_shared_edge),
        "large_anchor_min_attraction_ratio": float(args.large_anchor_min_attraction_ratio),
        "large_anchor_max_regularity_drop": float(args.large_anchor_max_regularity_drop),
        "large_anchor_max_hull_gap_increase": float(args.large_anchor_max_hull_gap_increase),
        "candidate_depth": "k0 direct anchor-neighbor and k1 one-hop through owned k0 zero-UPRN polygons",
        "assignment_rule": "all-anchor primary owner by largest shared_edge / anchor_perimeter; close secondary direct claims are allowed only for fallback anchors whose primary competitor is not a fallback output anchor",
        "selection_rule": "anchors below the large-anchor area threshold use regularity-first completion; large anchors keep anchor-only unless a merge improves shape or has strong shared-edge evidence with small shape loss",
        "plot_eligible_rule": "Theme contains building or land, excluding road/track/path",
        "added_count_distribution": {
            str(key): int(value)
            for key, value in out["added_clean_count"].value_counts().sort_index().items()
        },
        "selection_method_counts": {
            str(key): int(value)
            for key, value in out["selection_method"].value_counts().sort_index().items()
        },
        "selection_policy_counts": {
            str(key): int(value)
            for key, value in out["selection_policy"].value_counts().sort_index().items()
        },
        "fallback_regularity_mean": float(out["fallback_regularity_score"].mean()) if not out.empty else 0.0,
        "fallback_regularity_median": float(out["fallback_regularity_score"].median()) if not out.empty else 0.0,
        "rows_with_added_polygons": int(out["added_clean_count"].gt(0).sum()) if not out.empty else 0,
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic fallback parcels for anchors without a qualifying council polygon.")
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--anchor-gpkg", default=DEFAULT_ANCHOR_GPKG)
    parser.add_argument("--anchor-layer", default=DEFAULT_ANCHOR_LAYER)
    parser.add_argument("--council-single-anchor-gpkg", default=DEFAULT_COUNCIL_SINGLE_ANCHOR_GPKG)
    parser.add_argument("--council-single-anchor-layer", default=DEFAULT_COUNCIL_SINGLE_ANCHOR_LAYER)
    parser.add_argument("--council-min-anchor-overlap-area", type=float, default=0.5)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--output-layer", default=DEFAULT_OUTPUT_LAYER)
    parser.add_argument(
        "--fallback-scope",
        choices=["uncovered", "all"],
        default="uncovered",
        help="Use 'all' to emit fallback candidates for every anchor in the case.",
    )
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--top-direct", type=int, default=8)
    parser.add_argument("--top-indirect-per-direct", type=int, default=3)
    parser.add_argument("--max-enum-nodes", type=int, default=10)
    parser.add_argument("--min-completion-regularity", type=float, default=0.65)
    parser.add_argument("--max-completion-regularity-drop", type=float, default=0.05)
    parser.add_argument("--max-merge-regularity-loss-vs-anchor", type=float, default=0.05)
    parser.add_argument("--strong-completion-min-regularity", type=float, default=0.70)
    parser.add_argument("--strong-completion-max-hull-gap", type=float, default=0.05)
    parser.add_argument("--strong-completion-max-hole-area-ratio", type=float, default=0.005)
    parser.add_argument("--secondary-direct-min-ratio-of-best", type=float, default=0.75)
    parser.add_argument("--secondary-direct-min-attraction-ratio", type=float, default=0.15)
    parser.add_argument("--max-completion-hull-gap", type=float, default=0.60)
    parser.add_argument("--max-completion-hole-area-ratio", type=float, default=0.02)
    parser.add_argument("--large-anchor-area-threshold", type=float, default=500.0)
    parser.add_argument("--large-anchor-min-shared-edge", type=float, default=8.0)
    parser.add_argument("--large-anchor-min-attraction-ratio", type=float, default=0.08)
    parser.add_argument("--large-anchor-max-regularity-drop", type=float, default=0.01)
    parser.add_argument("--large-anchor-max-hull-gap-increase", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_gpkg = Path(args.output_gpkg)
    out, summary = build_fallback(args)
    if output_gpkg.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Output exists: {output_gpkg}")
        output_gpkg.unlink()
    _log(f"[INFO] Writing fallback output: {output_gpkg}:{args.output_layer}")
    write_kwargs = {"geometry_type": "MultiPolygon"} if out.empty else {}
    pyogrio.write_dataframe(out, output_gpkg, layer=str(args.output_layer), driver="GPKG", **write_kwargs)
    summary_path = output_gpkg.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _log("[DONE] Fallback build complete")
    _log(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

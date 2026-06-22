#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Set

import geopandas as gpd
from _core.centroid_combo import pick_centroid_aligned_wfs_combo
from _core.inline_merge import run_inline_merge_batch
from _core.polygonize_combo import (
    _polygonized_cells_multi,
    pick_polygonized_cell_centroid_combo,
    pick_polygonized_cell_centroid_combo_multi,
)
from _core.io import (
    apply_geometry_updates,
    load_layer,
    pick_nearest_polygon,
    pick_smallest_intersection_polygon,
    representative_points,
)
from _core.config import add_config_argument, get_config_section_from_argv, require_configured
from _core.wfs_merge import build_wfs_merge_gdf, filter_wfs_theme_features, resolve_theme_field
from shapely import set_precision
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import nearest_points, transform, unary_union
from shapely.validation import make_valid
from shapely.wkb import dumps as wkb_dumps, loads as wkb_loads


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv("capture", include_package_defaults=True)
    parser = argparse.ArgumentParser(
        description="Inline capture workflow (no external merge_demo6 dependency).",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)

    string_options = (
        "target-gpkg",
        "target-layer",
        "council-land-gpkg",
        "os-wfs-gpkg",
        "os-wfs-merge-gpkg",
        "save-built-os-wfs-merge",
        "save-built-os-wfs-merge-layer",
        "output-gpkg",
        "output-layer",
        "wfs-theme-include",
    )
    float_options = (
        "distance-tolerance",
        "area-tolerance",
        "min-iou",
        "window-area-scale",
        "point-centroid-tolerance",
        "point-combo-search-radius",
        "point-combo-max-area",
        "point-merge-combo-search-radius",
        "point-nearest-wfs-max-distance",
        "fallback-min-area",
        "fallback-max-aspect-ratio",
        "fallback-min-compactness",
        "point-contained-union-max-container-area",
        "council-driven-wfs-min-iou",
        "council-driven-wfs-min-seed-coverage",
        "council-driven-wfs-min-wfs-coverage",
        "council-driven-wfs-min-area-ratio",
        "council-driven-wfs-max-seed-area",
        "council-driven-wfs-max-candidate-seed-area-ratio",
        "council-driven-wfs-max-candidate-area-extra",
        "late-council-driven-wfs-min-iou",
        "late-council-driven-wfs-min-coverage",
        "late-council-driven-wfs-min-area-ratio",
        "late-council-driven-wfs-max-current-seed-iou",
        "late-council-driven-wfs-max-seed-area",
        "point-polygonize-search-radius",
        "point-polygonize-snap-refine-tolerance",
        "point-polygonize-snap-refine-grid",
        "point-polygonize-snap-refine-max-existing-symdiff",
        "point-polygonized-precision-model-grid",
        "point-polygonized-precision-model-tolerance",
        "point-polygonized-precision-model-max-existing-symdiff",
        "step2-output-precision-grid",
        "step3-intersection-output-precision-grid",
        "point-source-centroid-refine-grid",
        "point-source-centroid-refine-tolerance",
        "point-source-centroid-refine-max-area",
        "point-chargegeog-template-max-area",
        "council-seed-wfs-repair-search-radius",
        "council-seed-wfs-repair-min-cell-reference-coverage",
        "council-seed-wfs-repair-min-reference-iou",
        "council-seed-wfs-repair-centroid-tolerance",
        "council-seed-wfs-repair-centroid-improvement",
        "reference-qa-min-seed-area",
        "reference-qa-max-seed-area",
        "reference-qa-trim-max-seed-area",
        "reference-qa-trim-min-seed-coverage",
        "reference-qa-trim-min-current-coverage",
        "reference-qa-trim-min-outside-area",
        "reference-qa-trim-max-missing-ratio",
        "reference-qa-completion-min-seed-coverage",
        "reference-qa-completion-min-coverage-gain",
        "reference-qa-completion-min-area-ratio",
        "reference-qa-raw-union-max-outside-area",
        "single-polygon-bridge-width",
        "final-output-precision-grid",
    )
    int_options = (
        "max-candidates",
        "max-combo-size",
        "point-combo-max-candidates",
        "point-deep-combo-max-candidates",
        "point-polygonize-max-candidates",
        "council-seed-wfs-repair-max-cells",
        "reference-qa-completion-max-pieces",
    )
    bool_options = (
        "enable-slow-point-recovery",
        "enable-point-wfs-combo",
        "enable-point-deep-combo",
        "enable-point-merged-combo",
        "enable-point-contained-union",
        "enable-point-polygonized-combo",
        "enable-point-combined-polygonized-combo",
        "disable-reference-constrained-wfs-qa",
        "disable-point-chargegeog-template-refine",
        "quiet",
    )

    for option in string_options:
        parser.add_argument(f"--{option}")
    for option in float_options:
        parser.add_argument(f"--{option}", type=float)
    for option in int_options:
        parser.add_argument(f"--{option}", type=int)
    for option in bool_options:
        parser.add_argument(f"--{option}", action=argparse.BooleanOptionalAction)

    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(
        args,
        ("target_gpkg", "council_land_gpkg", "os_wfs_gpkg", "output_gpkg"),
        "capture",
    )
    return args


def _parse_wfs_theme_include(value: str) -> tuple[str, ...]:
    return tuple(term.strip().lower() for term in str(value).split(",") if term.strip())


def _geometry_aspect_ratio(geom) -> float:
    if geom is None or geom.is_empty:
        return float("inf")
    minx, miny, maxx, maxy = geom.bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    short_side = min(width, height)
    if short_side <= 0.0:
        return float("inf")
    return max(width, height) / short_side


def _geometry_compactness(geom) -> float:
    if geom is None or geom.is_empty or float(getattr(geom, "length", 0.0) or 0.0) <= 0.0:
        return 0.0
    return float(4.0 * math.pi * geom.area / (geom.length * geom.length))


def _filter_nearest_fallback_candidates(
    gdf: gpd.GeoDataFrame,
    *,
    min_area: float,
    max_aspect_ratio: float,
    min_compactness: float,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    geom = gdf.geometry
    mask = geom.apply(lambda value: value is not None and not value.is_empty)
    mask &= geom.apply(lambda value: "" if value is None else value.geom_type.upper()).isin(["POLYGON", "MULTIPOLYGON"])
    if float(min_area) > 0.0:
        mask &= gdf.geometry.area >= float(min_area)
    if float(max_aspect_ratio) > 0.0:
        mask &= gdf.geometry.apply(_geometry_aspect_ratio) <= float(max_aspect_ratio)
    if float(min_compactness) > 0.0:
        mask &= gdf.geometry.apply(_geometry_compactness) >= float(min_compactness)
    return gdf.loc[mask].copy()


def _nearest_wfs_info(
    point_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
) -> dict[int, tuple[float, str]]:
    if point_gdf.empty or polygon_gdf.empty:
        return {}
    try:
        theme_field = resolve_theme_field(polygon_gdf)
    except ValueError:
        theme_field = None

    nearest = gpd.sjoin_nearest(
        point_gdf[["capture_src_id", "geometry"]],
        polygon_gdf[["geometry"]],
        how="left",
        distance_col="dist_m",
    )
    nearest = nearest.dropna(subset=["index_right"]).copy()
    if nearest.empty:
        return {}
    nearest = nearest.sort_values(["capture_src_id", "dist_m"]).drop_duplicates(
        subset=["capture_src_id"],
        keep="first",
    )

    out: dict[int, tuple[float, str]] = {}
    for _, row in nearest.iterrows():
        src_id = int(row["capture_src_id"])
        right_idx = int(row["index_right"])
        theme_value = ""
        if theme_field is not None:
            theme_value = str(polygon_gdf.loc[right_idx, theme_field])
        out[src_id] = (float(row["dist_m"]), theme_value)
    return out


def _annotate_wfs_failure_diagnostics(
    result: gpd.GeoDataFrame,
    failed_points: gpd.GeoDataFrame,
    *,
    eligible_wfs: gpd.GeoDataFrame,
    any_wfs: gpd.GeoDataFrame,
) -> None:
    if failed_points.empty:
        return
    eligible_info = _nearest_wfs_info(failed_points, eligible_wfs)
    any_info = _nearest_wfs_info(failed_points, any_wfs)

    for src_id in failed_points["capture_src_id"].astype(int).tolist():
        mask = result["capture_src_id"].astype(int).eq(src_id)
        if src_id in eligible_info:
            dist, theme = eligible_info[src_id]
            result.loc[mask, "nearest_eligible_wfs_dist_m"] = dist
            result.loc[mask, "nearest_eligible_wfs_theme"] = theme
        if src_id in any_info:
            dist, theme = any_info[src_id]
            result.loc[mask, "nearest_any_wfs_dist_m"] = dist
            result.loc[mask, "nearest_any_wfs_theme"] = theme


def _filter_point_anchored_results(step_outputs: gpd.GeoDataFrame, point_reps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if step_outputs.empty:
        return step_outputs
    pts = point_reps[["capture_src_id", "geometry"]].copy()
    polys = step_outputs[["capture_src_id", "geometry"]].copy()
    # Use inner join so we only keep polygon rows that intersect at least one point.
    # This avoids depending on geopandas version-specific right-index column names.
    joined = gpd.sjoin(polys, pts, how="inner", predicate="intersects", lsuffix="poly", rsuffix="pt")
    if joined.empty:
        return step_outputs.iloc[0:0].copy()

    id_col = "capture_src_id_poly" if "capture_src_id_poly" in joined.columns else "capture_src_id"
    if id_col not in joined.columns:
        for candidate in ("capture_src_id_left", "capture_src_id_l", "capture_src_id_x"):
            if candidate in joined.columns:
                id_col = candidate
                break
    if id_col not in joined.columns:
        raise KeyError(f"capture_src_id column missing after sjoin. Columns: {list(joined.columns)}")
    keep_ids = joined[id_col].astype(int).unique().tolist()
    return step_outputs[step_outputs["capture_src_id"].astype(int).isin(set(keep_ids))].copy()


def _polygon_candidates(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g is not None and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for part in geom.geoms:
            out.extend(_polygon_candidates(part))
        return out
    return []


def _apply_final_precision_grid(result: gpd.GeoDataFrame, precision_grid: float) -> int:
    if precision_grid <= 0.0 or result.empty:
        return 0

    changed = 0
    rounded = []
    for geom in result.geometry:
        if geom is None or geom.is_empty:
            rounded.append(geom)
            continue
        snapped = _round_geometry_coordinates(geom, float(precision_grid))
        rounded.append(snapped)
        if not snapped.equals_exact(geom, 0.0):
            changed += 1
    result.geometry = rounded
    return changed


def _apply_stage_precision_grid(result: gpd.GeoDataFrame, stage: str, precision_grid: float) -> int:
    if precision_grid <= 0.0 or result.empty or "capture_stage" not in result.columns:
        return 0

    mask = result["capture_stage"].eq(stage)
    if not mask.any():
        return 0

    changed = 0
    rounded = []
    for geom in result.loc[mask, "geometry"]:
        if geom is None or geom.is_empty:
            rounded.append(geom)
            continue
        snapped = _round_geometry_coordinates(geom, float(precision_grid))
        rounded.append(snapped)
        if not snapped.equals_exact(geom, 0.0):
            changed += 1
    result.loc[mask, "geometry"] = rounded
    return changed


def _round_geometry_coordinates(geom, precision_grid: float):
    if geom is None or geom.is_empty or precision_grid <= 0.0:
        return geom

    inv_grid = 1.0 / float(precision_grid)

    def _round_xy(x, y, z=None):
        rx = [round(float(value) * inv_grid) / inv_grid for value in x]
        ry = [round(float(value) * inv_grid) / inv_grid for value in y]
        if z is None:
            return rx, ry
        return rx, ry, z

    rounded = transform(_round_xy, geom)
    if not rounded.is_valid:
        rounded = make_valid(rounded)
    return rounded


def _apply_point_source_centroid_precision_refinement(
    result: gpd.GeoDataFrame,
    point_reps: gpd.GeoDataFrame,
    polygon_sources: list[tuple[str, gpd.GeoDataFrame]],
    *,
    stages: set[str],
    precision_grid: float,
    tolerance: float,
    max_area: float,
) -> int:
    if (
        result.empty
        or point_reps.empty
        or not polygon_sources
        or float(precision_grid) < 0.0
        or float(tolerance) < 0.0
        or "capture_stage" not in result.columns
    ):
        return 0

    point_by_id = point_reps.set_index("capture_src_id")["geometry"].to_dict()
    mask = result["capture_stage"].isin(stages) & result["capture_src_id"].isin(point_by_id.keys())
    if not bool(mask.any()):
        return 0

    source_order = {name: order for order, (name, _) in enumerate(polygon_sources)}
    updated = 0
    geom_col = result.geometry.name
    for idx, row in result.loc[mask].iterrows():
        src_id = int(row["capture_src_id"])
        point_geom = point_by_id.get(src_id)
        if point_geom is None or point_geom.is_empty:
            continue

        best_key = None
        best_geom = None
        best_source = None
        for source_name, source_gdf in polygon_sources:
            if source_gdf.empty:
                continue
            sidx = source_gdf.sindex
            candidate_idxs = list(sidx.query(point_geom, predicate="intersects")) if sidx is not None else []
            for pos in candidate_idxs:
                geom = source_gdf.geometry.iloc[int(pos)]
                if geom is None or geom.is_empty:
                    continue
                area = float(geom.area)
                if area <= 0.01 or area > float(max_area):
                    continue
                snapped = _round_geometry_coordinates(geom, float(precision_grid)) if float(precision_grid) > 0.0 else geom
                if not _is_polygonal(snapped):
                    continue
                snapped_area = float(snapped.area)
                if snapped_area <= 0.01 or snapped_area > float(max_area):
                    continue
                centroid_dist = float(snapped.centroid.distance(point_geom))
                if centroid_dist > float(tolerance):
                    continue
                key = (centroid_dist, source_order[source_name], snapped_area)
                if best_key is None or key < best_key:
                    best_key = key
                    best_geom = snapped
                    best_source = source_name

        if best_geom is None or best_source is None:
            continue
        result.at[idx, geom_col] = best_geom
        result.at[idx, "capture_stage"] = f"point_source_centroid_precision_refine_{best_source}"
        result.at[idx, "capture_success"] = True
        updated += 1

    return updated


def _apply_chargegeog_template_translation_refinement(
    result: gpd.GeoDataFrame,
    point_targets: gpd.GeoDataFrame,
    *,
    stages: set[str],
    max_area: float,
) -> int:
    if (
        result.empty
        or point_targets.empty
        or "chargegeog" not in point_targets.columns
        or "capture_stage" not in result.columns
    ):
        return 0

    points = point_targets[["capture_src_id", "chargegeog", "geometry"]].copy()
    points = points[points["chargegeog"].notna()].copy()
    if points.empty:
        return 0

    point_by_id = points.set_index("capture_src_id")["geometry"].to_dict()
    charge_by_id = points.set_index("capture_src_id")["chargegeog"].to_dict()
    ids_by_charge = points.groupby("chargegeog")["capture_src_id"].apply(list).to_dict()

    result_by_id = result.set_index("capture_src_id")
    high_conf_stage = result_by_id["capture_stage"].astype(str).str.startswith(("point_centroid", "point_polygonized"))
    donor_ids = set(result_by_id.loc[high_conf_stage].index.astype(int).tolist())

    mask = result["capture_stage"].isin(stages) & result["capture_src_id"].isin(point_by_id.keys())
    if not bool(mask.any()):
        return 0

    updated = 0
    geom_col = result.geometry.name
    for idx, row in result.loc[mask].iterrows():
        src_id = int(row["capture_src_id"])
        charge = charge_by_id.get(src_id)
        if charge is None:
            continue
        same_ids = ids_by_charge.get(charge, [])
        donors = [
            int(donor_id)
            for donor_id in same_ids
            if int(donor_id) != src_id and int(donor_id) in donor_ids and int(donor_id) in result_by_id.index
        ]
        if not donors:
            continue

        point_geom = point_by_id.get(src_id)
        if point_geom is None or point_geom.is_empty:
            continue

        best_key = None
        best_geom = None
        current_area = float(row[geom_col].area) if row[geom_col] is not None and not row[geom_col].is_empty else 0.0
        for donor_id in donors:
            donor_geom = result_by_id.at[donor_id, geom_col]
            if not _is_polygonal(donor_geom):
                continue
            donor_area = float(donor_geom.area)
            if donor_area <= 0.01 or donor_area > float(max_area):
                continue
            donor_centroid = donor_geom.centroid
            translated = translate(
                donor_geom,
                xoff=float(point_geom.x - donor_centroid.x),
                yoff=float(point_geom.y - donor_centroid.y),
            )
            if not _is_polygonal(translated):
                continue
            translated_area = float(translated.area)
            if translated_area <= 0.01 or translated_area > float(max_area):
                continue
            key = (abs(translated_area - current_area), translated_area, donor_id)
            if best_key is None or key < best_key:
                best_key = key
                best_geom = translated

        if best_geom is None:
            continue
        result.at[idx, geom_col] = best_geom
        result.at[idx, "capture_stage"] = "chargegeog_template_translate"
        result.at[idx, "capture_success"] = True
        updated += 1

    return updated


def _pick_single_polygon_with_anchor(geom, anchor_geom):
    polys = _polygon_candidates(geom)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    if anchor_geom is None or anchor_geom.is_empty:
        return max(polys, key=lambda p: float(p.area))

    intersecting = [p for p in polys if p.intersects(anchor_geom)]
    if intersecting:
        def _inter_score(p):
            inter = p.intersection(anchor_geom)
            return float(inter.area), -float(p.distance(anchor_geom)), float(p.area)

        return max(intersecting, key=_inter_score)

    return min(polys, key=lambda p: (float(p.distance(anchor_geom)), -float(p.area)))


def _connect_polygon_parts(geom, *, bridge_width: float):
    polys = [poly for poly in _polygon_candidates(geom) if float(poly.area) > 0.0]
    if not polys:
        return geom
    if len(polys) == 1:
        return polys[0]

    connected = max(polys, key=lambda poly: float(poly.area))
    remaining = [poly for poly in polys if poly is not connected]
    width = max(float(bridge_width), 0.001)

    while remaining:
        next_poly = min(remaining, key=lambda poly: float(connected.distance(poly)))
        remaining.remove(next_poly)
        if connected.intersects(next_poly) or connected.touches(next_poly):
            bridge = None
        else:
            start, end = nearest_points(connected, next_poly)
            bridge_line = LineString([start, end])
            bridge = bridge_line.buffer(width, cap_style=1, join_style=1)

        pieces = [connected, next_poly]
        if bridge is not None and not bridge.is_empty:
            pieces.append(bridge)
        connected = unary_union(pieces)
        if not connected.is_valid:
            connected = make_valid(connected)
        connected = _extract_polygonal_geometry(connected)
        if not _is_polygonal(connected):
            return connected

    if isinstance(connected, MultiPolygon):
        polys = _polygon_candidates(connected)
        if len(polys) == 1:
            return polys[0]
    return connected


def _ensure_single_polygon_with_anchor(geom, anchor_geom, *, bridge_width: float = 0.5):
    polys = _polygon_candidates(geom)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]

    connected = _connect_polygon_parts(MultiPolygon(polys), bridge_width=bridge_width)
    if isinstance(connected, Polygon):
        return connected
    if isinstance(connected, MultiPolygon) and len(connected.geoms) == 1:
        return connected.geoms[0]
    return _pick_single_polygon_with_anchor(connected, anchor_geom)


def _force_single_polygon_per_row(
    result: gpd.GeoDataFrame,
    source_geom_by_id: dict[int, object],
    *,
    bridge_width: float = 0.5,
) -> gpd.GeoDataFrame:
    out = result.copy()
    new_geoms = []
    for _, row in out.iterrows():
        src_id = int(row["capture_src_id"])
        anchor = source_geom_by_id.get(src_id)
        single_geom = _ensure_single_polygon_with_anchor(
            row.geometry,
            anchor,
            bridge_width=bridge_width,
        )
        new_geoms.append(single_geom if single_geom is not None else row.geometry)
    out["geometry"] = new_geoms
    return out


def _extract_polygonal_geometry(geom):
    if geom is None or geom.is_empty or _is_polygonal(geom):
        return geom
    polys = _polygon_candidates(geom)
    if not polys:
        return geom
    merged = unary_union(polys)
    if _is_polygonal(merged):
        return merged
    return _pick_single_polygon_with_anchor(merged, geom)


def _extract_polygonal_geometries(result: gpd.GeoDataFrame) -> int:
    changed = 0
    geoms = []
    for geom in result.geometry:
        extracted = _extract_polygonal_geometry(geom)
        geoms.append(extracted)
        if extracted is not geom and (geom is None or not extracted.equals_exact(geom, 0.0)):
            changed += 1
    if changed:
        result.geometry = geoms
    return changed


def _remove_polygon_holes(geom):
    polys = _polygon_candidates(geom)
    if not polys:
        return geom, 0

    hole_count = sum(len(poly.interiors) for poly in polys)
    if hole_count == 0:
        return geom, 0

    cleaned = [Polygon(poly.exterior) for poly in polys if float(poly.area) > 0.0]
    if not cleaned:
        return geom, 0

    cleaned_geom = cleaned[0] if len(cleaned) == 1 else MultiPolygon(cleaned)
    if not cleaned_geom.is_valid:
        cleaned_geom = make_valid(cleaned_geom)
    cleaned_geom = _extract_polygonal_geometry(cleaned_geom)
    return cleaned_geom, hole_count


def _remove_holes_from_result(result: gpd.GeoDataFrame) -> tuple[int, int]:
    if result.empty:
        return 0, 0

    changed = 0
    removed = 0
    geoms = []
    for geom in result.geometry:
        cleaned, hole_count = _remove_polygon_holes(geom)
        geoms.append(cleaned)
        if hole_count:
            changed += 1
            removed += hole_count
    if changed:
        result.geometry = geoms
    return changed, removed


def _polygon_hole_count(geom) -> int:
    return sum(len(poly.interiors) for poly in _polygon_candidates(geom))


def _polygon_only_mask(gdf: gpd.GeoDataFrame) -> gpd.Series:
    geom = gdf.geometry
    not_missing = geom.apply(lambda value: value is not None)
    not_empty = geom.apply(lambda value: value is not None and not value.is_empty)
    geom_type = geom.apply(lambda value: "" if value is None else value.geom_type.upper())
    return not_missing & not_empty & geom_type.isin(["POLYGON", "MULTIPOLYGON"])


def _single_polygon_only_mask(gdf: gpd.GeoDataFrame) -> gpd.Series:
    geom = gdf.geometry
    not_missing = geom.apply(lambda value: value is not None)
    not_empty = geom.apply(lambda value: value is not None and not value.is_empty)
    geom_type = geom.apply(lambda value: "" if value is None else value.geom_type.upper())
    return not_missing & not_empty & geom_type.eq("POLYGON")


def _pick_centroid_aligned_polygon(
    point_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    *,
    tolerance: float,
) -> gpd.GeoDataFrame:
    if point_gdf.empty or polygon_gdf.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs="EPSG:27700")

    joined = gpd.sjoin(
        point_gdf[["capture_src_id", "geometry"]],
        polygon_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=polygon_gdf.crs)

    source_geoms = polygon_gdf.geometry
    joined["candidate_area"] = joined["index_right"].map(source_geoms.area)
    joined["centroid_dist"] = joined.apply(
        lambda row: float(source_geoms.loc[row["index_right"]].centroid.distance(row.geometry)),
        axis=1,
    )
    aligned = joined[joined["centroid_dist"] <= float(tolerance)].copy()
    if aligned.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=polygon_gdf.crs)

    chosen = (
        aligned.sort_values(
            ["capture_src_id", "centroid_dist", "candidate_area", "index_right"],
            ascending=[True, True, True, True],
        )
        .drop_duplicates(subset=["capture_src_id"], keep="first")
        .copy()
    )
    chosen["geometry"] = chosen["index_right"].map(source_geoms)
    return gpd.GeoDataFrame(chosen[["capture_src_id", "geometry"]], geometry="geometry", crs=polygon_gdf.crs)


def _pick_council_contained_wfs_union(
    point_gdf: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
    wfs_gdf: gpd.GeoDataFrame,
    *,
    tolerance: float,
    max_area: float,
    max_container_area: float,
    min_coverage_ratio: float = 0.995,
) -> gpd.GeoDataFrame:
    if point_gdf.empty or council_gdf.empty or wfs_gdf.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)

    containers = pick_smallest_intersection_polygon(point_gdf, council_gdf)
    if containers.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)

    container_by_id = dict(zip(containers["capture_src_id"].astype(int), containers.geometry))
    wfs_sidx = wfs_gdf.sindex
    rows = []
    for _, point_row in point_gdf.iterrows():
        src_id = int(point_row["capture_src_id"])
        point_geom = point_row.geometry
        container = container_by_id.get(src_id)
        if (
            container is None
            or bool(getattr(container, "is_empty", False))
            or float(container.area) > float(max_container_area)
        ):
            continue

        child_geoms = []
        for pos in wfs_sidx.query(container, predicate="intersects"):
            geom = wfs_gdf.geometry.iloc[int(pos)]
            if geom is None or geom.is_empty:
                continue
            area = float(geom.area)
            if area <= 0.01 or area > float(max_area):
                continue
            if not container.covers(geom.representative_point()):
                continue
            if float(geom.intersection(container).area) / max(area, 1e-9) < float(min_coverage_ratio):
                continue
            child_geoms.append(geom)

        if not child_geoms:
            continue
        union_geom = child_geoms[0] if len(child_geoms) == 1 else unary_union(child_geoms)
        if union_geom is None or union_geom.is_empty:
            continue
        if float(union_geom.area) > float(max_area):
            continue
        if float(union_geom.centroid.distance(point_geom)) <= float(tolerance):
            rows.append({"capture_src_id": src_id, "geometry": union_geom})

    if not rows:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=point_gdf.crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=point_gdf.crs)


def _pick_council_driven_wfs_overrides(
    result: gpd.GeoDataFrame,
    point_gdf: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
    wfs_sources: list[gpd.GeoDataFrame],
    *,
    stages: set[str],
    min_iou: float,
    min_seed_coverage: float,
    min_wfs_coverage: float,
    min_area_ratio: float,
    max_seed_area: float,
    max_candidate_seed_area_ratio: float,
    max_candidate_area_extra: float,
    max_current_seed_iou: float | None = None,
) -> gpd.GeoDataFrame:
    if result.empty or point_gdf.empty or council_gdf.empty or not wfs_sources:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=result.crs)

    stage_mask = result["capture_stage"].isin(stages)
    stage_mask &= result.geometry.notna() & ~result.geometry.is_empty
    if not bool(stage_mask.any()):
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=result.crs)

    eligible_points = point_gdf[point_gdf["capture_src_id"].isin(result.loc[stage_mask, "capture_src_id"])].copy()
    if eligible_points.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=result.crs)

    council_seeds = pick_smallest_intersection_polygon(eligible_points, council_gdf)
    if council_seeds.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=result.crs)

    seed_by_id = {
        int(src_id): geom
        for src_id, geom in zip(council_seeds["capture_src_id"], council_seeds.geometry)
        if _is_polygonal(geom)
    }
    point_by_id = {
        int(src_id): geom
        for src_id, geom in zip(eligible_points["capture_src_id"], eligible_points.geometry)
        if geom is not None and not geom.is_empty
    }
    current_by_id = {
        int(row["capture_src_id"]): row.geometry
        for _, row in result.loc[stage_mask, ["capture_src_id", result.geometry.name]].iterrows()
        if _is_polygonal(row.geometry)
    }
    usable_sources = [source for source in wfs_sources if source is not None and not source.empty]
    source_indexes = [source.sindex for source in usable_sources]

    rows = []
    for src_id, seed_geom in seed_by_id.items():
        point_geom = point_by_id.get(src_id)
        current_geom = current_by_id.get(src_id)
        if point_geom is None or current_geom is None:
            continue
        seed_area = float(seed_geom.area)
        current_area = float(current_geom.area)
        if seed_area <= 0.01 or current_area <= 0.01 or seed_area > float(max_seed_area):
            continue
        current_seed_iou = _geometry_iou(current_geom, seed_geom)
        if max_current_seed_iou is not None and current_seed_iou > float(max_current_seed_iou):
            continue

        max_candidate_area = max(
            seed_area * float(max_candidate_seed_area_ratio),
            seed_area + float(max_candidate_area_extra),
        )
        best = None
        best_key = None
        for source, source_sidx in zip(usable_sources, source_indexes):
            for pos in source_sidx.query(seed_geom, predicate="intersects"):
                candidate = source.geometry.iloc[int(pos)]
                if not _is_polygonal(candidate):
                    continue
                if not candidate.intersects(point_geom):
                    continue
                candidate = _pick_single_polygon_with_anchor(candidate, point_geom)
                if not _is_polygonal(candidate) or not candidate.intersects(point_geom):
                    continue

                candidate_area = float(candidate.area)
                if candidate_area <= current_area * float(min_area_ratio):
                    continue
                if candidate_area > max_candidate_area:
                    continue

                inter_area = float(candidate.intersection(seed_geom).area)
                if inter_area <= 0.0:
                    continue
                union_area = float(candidate.union(seed_geom).area)
                if union_area <= 0.0:
                    continue
                iou = inter_area / union_area
                seed_coverage = inter_area / seed_area
                wfs_coverage = inter_area / candidate_area
                if (
                    iou < float(min_iou)
                    or seed_coverage < float(min_seed_coverage)
                    or wfs_coverage < float(min_wfs_coverage)
                ):
                    continue

                key = (
                    iou,
                    seed_coverage,
                    wfs_coverage,
                    candidate_area / current_area,
                    -candidate_area,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "capture_src_id": src_id,
                        "geometry": candidate,
                        "council_seed_area_m2": seed_area,
                        "current_area_m2": current_area,
                        "candidate_area_m2": candidate_area,
                        "candidate_seed_iou": iou,
                        "candidate_seed_coverage": seed_coverage,
                        "candidate_wfs_coverage": wfs_coverage,
                        "current_seed_iou": current_seed_iou,
                    }

        if best is not None:
            rows.append(best)

    if not rows:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs=result.crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=result.crs)


def _clean_link_value(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value or "").strip()


def _is_polygonal(geom) -> bool:
    return (
        geom is not None
        and not bool(getattr(geom, "is_empty", False))
        and str(getattr(geom, "geom_type", "")).upper() in {"POLYGON", "MULTIPOLYGON"}
    )


def _geometry_iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter_area = float(a.intersection(b).area)
    if inter_area <= 0.0:
        return 0.0
    union_area = float(a.union(b).area)
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _find_first_column(gdf: gpd.GeoDataFrame, names: tuple[str, ...]) -> str | None:
    wanted = {name.lower().replace("_", "").replace(" ", "") for name in names}
    for column in gdf.columns:
        normalised = str(column).lower().replace("_", "").replace(" ", "")
        if normalised in wanted:
            return column
    return None


def _wfs_feature_signature(source: gpd.GeoDataFrame, pos: int) -> tuple[tuple[str, str], ...]:
    signature = []
    for label, names in (
        ("theme", ("Theme", "theme")),
        (
            "descriptive_group",
            ("DescriptiveGroup", "descriptive_group", "descriptive group", "descgroup"),
        ),
        (
            "descriptive_term",
            ("DescriptiveTerm", "descriptive_term", "descriptive term", "descterm"),
        ),
    ):
        column = _find_first_column(source, names)
        if column is None:
            continue
        value = _clean_link_value(source.iloc[int(pos)].get(column)).lower()
        if value:
            signature.append((label, value))
    return tuple(signature)


def _same_wfs_reference_union_for_seed(
    seed_geom,
    current_geom,
    point_geom,
    wfs_sources: list[gpd.GeoDataFrame],
):
    seed_area = float(seed_geom.area)
    best = None
    best_key = None

    for source in wfs_sources:
        if source is None or source.empty:
            continue
        source_sidx = source.sindex
        rows = []
        for pos in source_sidx.query(seed_geom, predicate="intersects"):
            pos = int(pos)
            geom = source.geometry.iloc[pos]
            if not _is_polygonal(geom):
                continue
            inter_seed = float(geom.intersection(seed_geom).area)
            if inter_seed <= 0.05:
                continue
            inter_current = float(geom.intersection(current_geom).area) if _is_polygonal(current_geom) else 0.0
            rows.append(
                {
                    "pos": pos,
                    "geometry": geom,
                    "signature": _wfs_feature_signature(source, pos),
                    "inter_seed": inter_seed,
                    "inter_current": inter_current,
                    "area": float(geom.area),
                }
            )

        if not rows:
            continue

        anchors = [row for row in rows if row["geometry"].intersects(point_geom)]
        if not anchors:
            anchors = [row for row in rows if row["inter_current"] > 0.05]
        if not anchors:
            continue

        anchor = max(
            anchors,
            key=lambda row: (
                float(row["inter_current"]),
                float(row["inter_seed"]),
                -float(row["area"]),
            ),
        )
        signature = anchor["signature"]
        same_rows = [row for row in rows if not signature or row["signature"] == signature]
        if not same_rows:
            continue

        seen_wkb = set()
        same_geoms = []
        for row in same_rows:
            geom = row["geometry"]
            geom_wkb = geom.wkb
            if geom_wkb in seen_wkb:
                continue
            seen_wkb.add(geom_wkb)
            same_geoms.append(geom)

        if not same_geoms:
            continue

        raw_union = same_geoms[0] if len(same_geoms) == 1 else unary_union(same_geoms)
        if not _is_polygonal(raw_union):
            continue

        constrained = raw_union.intersection(seed_geom)
        if not constrained.is_valid:
            constrained = make_valid(constrained)
        constrained = _extract_polygonal_geometry(constrained)
        if not _is_polygonal(constrained):
            continue

        constrained_inter_area = float(constrained.intersection(seed_geom).area)
        constrained_area = float(constrained.area)
        if constrained_area <= 0.01 or constrained_inter_area <= 0.01:
            continue

        raw_seed_coverage = float(raw_union.intersection(seed_geom).area) / max(seed_area, 1e-9)
        raw_outside_area = float(raw_union.difference(seed_geom).area)
        constrained_seed_coverage = constrained_inter_area / max(seed_area, 1e-9)

        key = (
            constrained_seed_coverage,
            constrained_inter_area,
            raw_seed_coverage,
            -raw_outside_area,
            -len(same_geoms),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "geometry": constrained,
                "piece_count": len(same_geoms),
                "raw_seed_coverage": raw_seed_coverage,
                "raw_outside_area": raw_outside_area,
                "constrained_seed_coverage": constrained_seed_coverage,
                "constrained_inter_area": constrained_inter_area,
                "constrained_area": constrained_area,
            }

    return best


def _prepare_reference_qa_candidate(geom, point_geom, *, bridge_width: float):
    if geom is None or bool(getattr(geom, "is_empty", False)):
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    geom = _extract_polygonal_geometry(geom)
    if not _is_polygonal(geom):
        return None
    geom = _ensure_single_polygon_with_anchor(geom, point_geom, bridge_width=bridge_width)
    if not isinstance(geom, Polygon) or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
        geom = _ensure_single_polygon_with_anchor(geom, point_geom, bridge_width=bridge_width)
    if not isinstance(geom, Polygon) or geom.is_empty:
        return None
    return geom


def _apply_reference_constrained_wfs_qa(
    result: gpd.GeoDataFrame,
    point_reps: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
    wfs_sources: list[gpd.GeoDataFrame],
    *,
    min_seed_area: float,
    max_seed_area: float,
    trim_max_seed_area: float,
    trim_min_seed_coverage: float,
    trim_min_current_coverage: float,
    trim_min_outside_area: float,
    trim_max_missing_ratio: float,
    completion_min_seed_coverage: float,
    completion_min_coverage_gain: float,
    completion_max_pieces: int,
    completion_min_area_ratio: float,
    raw_union_max_outside_area: float,
    bridge_width: float,
) -> dict[str, int]:
    if result.empty or point_reps.empty or council_gdf.empty or not wfs_sources:
        return {}
    if "capture_stage" not in result.columns or "capture_success" not in result.columns:
        return {}

    point_ids = set(point_reps["capture_src_id"].astype(int).tolist())
    eligible_mask = (
        result["capture_success"].fillna(False).astype(bool)
        & result["capture_src_id"].astype(int).isin(point_ids)
        & result.geometry.notna()
    )
    if not bool(eligible_mask.any()):
        return {}

    eligible_ids = set(result.loc[eligible_mask, "capture_src_id"].astype(int).tolist())
    eligible_points = point_reps[point_reps["capture_src_id"].astype(int).isin(eligible_ids)].copy()
    council_seeds = pick_smallest_intersection_polygon(eligible_points, council_gdf)
    if council_seeds.empty:
        return {}

    point_by_id = point_reps.set_index("capture_src_id")["geometry"].to_dict()
    seed_by_id = council_seeds.set_index("capture_src_id")["geometry"].to_dict()
    row_by_id = {int(src_id): idx for idx, src_id in zip(result.index, result["capture_src_id"])}
    geom_col = result.geometry.name
    counts: dict[str, int] = {}

    for src_id, seed_geom in seed_by_id.items():
        src_id = int(src_id)
        idx = row_by_id.get(src_id)
        point_geom = point_by_id.get(src_id)
        if idx is None or point_geom is None or point_geom.is_empty or not _is_polygonal(seed_geom):
            continue
        current_geom = result.at[idx, geom_col]
        if not _is_polygonal(current_geom):
            continue

        seed_area = float(seed_geom.area)
        current_area = float(current_geom.area)
        if seed_area < float(min_seed_area) or current_area <= 0.01:
            continue

        current_inter_area = float(current_geom.intersection(seed_geom).area)
        current_seed_coverage = current_inter_area / max(seed_area, 1e-9)
        current_coverage = current_inter_area / max(current_area, 1e-9)
        outside_area = float(current_geom.difference(seed_geom).area)
        missing_area = max(seed_area - current_inter_area, 0.0)

        best_stage = None
        best_geom = None
        best_key = None

        if (
            seed_area <= float(trim_max_seed_area)
            and current_seed_coverage >= float(trim_min_seed_coverage)
            and current_coverage >= float(trim_min_current_coverage)
            and outside_area >= float(trim_min_outside_area)
            and missing_area / max(seed_area, 1e-9) <= float(trim_max_missing_ratio)
        ):
            # Council land is a reference constraint here; the output remains the WFS geometry clipped
            # to the reference footprint rather than the council polygon by itself.
            trimmed = _prepare_reference_qa_candidate(
                current_geom.intersection(seed_geom),
                point_geom,
                bridge_width=bridge_width,
            )
            if trimmed is not None and float(trimmed.area) > 0.01:
                best_stage = "wfs_reference_constrained_council_reference_qa"
                best_geom = trimmed
                best_key = (10.0 + outside_area, current_seed_coverage, current_coverage)

        same_wfs = _same_wfs_reference_union_for_seed(
            seed_geom,
            current_geom,
            point_geom,
            wfs_sources,
        )
        if same_wfs is not None:
            coverage_gain = same_wfs["constrained_inter_area"] - current_inter_area
            completed_area_ratio = same_wfs["constrained_area"] / max(seed_area, 1e-9)
            completion_ok = (
                seed_area <= float(max_seed_area)
                and current_seed_coverage < 0.92
                and same_wfs["constrained_seed_coverage"] >= float(completion_min_seed_coverage)
                and coverage_gain >= float(completion_min_coverage_gain)
                and same_wfs["piece_count"] <= int(completion_max_pieces)
                and completed_area_ratio >= float(completion_min_area_ratio)
                and same_wfs["raw_outside_area"] <= float(raw_union_max_outside_area)
            )
            underfill_ok = (
                seed_area <= float(max_seed_area)
                and 0.50 <= current_seed_coverage < float(trim_min_seed_coverage)
                and same_wfs["raw_seed_coverage"] >= 0.95
                and coverage_gain >= float(completion_min_coverage_gain)
                and same_wfs["piece_count"] <= int(completion_max_pieces)
                and completed_area_ratio >= float(completion_min_area_ratio)
                and same_wfs["raw_outside_area"] <= float(raw_union_max_outside_area)
            )
            if completion_ok or underfill_ok:
                completed = _prepare_reference_qa_candidate(
                    same_wfs["geometry"],
                    point_geom,
                    bridge_width=bridge_width,
                )
                if completed is not None and float(completed.area) > 0.01:
                    stage = "wfs_reference_footprint_completion_qa"
                    key = (
                        100.0 + coverage_gain,
                        same_wfs["constrained_seed_coverage"],
                        -same_wfs["raw_outside_area"],
                    )
                    if best_key is None or key > best_key:
                        best_stage = stage
                        best_geom = completed
                        best_key = key

        if best_stage is None or best_geom is None:
            continue
        if best_geom.equals_exact(current_geom, 0.0) or float(best_geom.symmetric_difference(current_geom).area) < 1e-9:
            continue

        result.at[idx, geom_col] = best_geom
        result.at[idx, "capture_stage"] = best_stage
        result.at[idx, "capture_success"] = True
        counts[best_stage] = counts.get(best_stage, 0) + 1

    return counts


def _apply_council_seed_wfs_polygonized_repair(
    result: gpd.GeoDataFrame,
    point_reps: gpd.GeoDataFrame,
    council_gdf: gpd.GeoDataFrame,
    wfs_sources: list[gpd.GeoDataFrame],
    *,
    stages: set[str],
    search_radius: float,
    max_area: float,
    max_cells: int,
    min_cell_reference_coverage: float,
    min_reference_iou: float,
    centroid_tolerance: float,
    centroid_improvement: float,
) -> int:
    if result.empty or point_reps.empty or council_gdf.empty or not wfs_sources:
        return 0
    if max_cells <= 0 or float(search_radius) <= 0.0:
        return 0

    stage_mask = result["capture_stage"].isin(stages)
    if not bool(stage_mask.any()):
        return 0

    repair_points = point_reps[point_reps["capture_src_id"].isin(result.loc[stage_mask, "capture_src_id"])].copy()
    if repair_points.empty:
        return 0

    council_seeds = pick_smallest_intersection_polygon(repair_points, council_gdf)
    if council_seeds.empty:
        return 0

    point_by_id = repair_points.set_index("capture_src_id")["geometry"].to_dict()
    seed_by_id = council_seeds.set_index("capture_src_id")["geometry"].to_dict()
    row_by_id = {int(src_id): idx for idx, src_id in zip(result.index, result["capture_src_id"])}
    usable_sources = [source for source in wfs_sources if not source.empty]
    if not usable_sources:
        return 0

    updated = 0
    geom_col = result.geometry.name
    for src_id, seed_geom in seed_by_id.items():
        src_id = int(src_id)
        idx = row_by_id.get(src_id)
        point_geom = point_by_id.get(src_id)
        if idx is None or point_geom is None or point_geom.is_empty or seed_geom is None or seed_geom.is_empty:
            continue
        current_geom = result.at[idx, geom_col]
        if not _is_polygonal(current_geom):
            continue

        cells = _polygonized_cells_multi(
            point_geom,
            usable_sources,
            search_radius=float(search_radius),
            max_area=float(max_area),
        )
        if not cells:
            continue

        selected = []
        for cell in cells:
            inter_area = float(cell.intersection(seed_geom).area)
            if inter_area <= 0.0:
                continue
            cell_area = float(cell.area)
            if cell_area <= 0.0:
                continue
            if inter_area / cell_area >= float(min_cell_reference_coverage):
                selected.append(cell)
        if not selected or len(selected) > int(max_cells):
            continue

        candidate = selected[0] if len(selected) == 1 else unary_union(selected)
        if not _is_polygonal(candidate):
            continue
        candidate = _pick_single_polygon_with_anchor(candidate, point_geom)
        if not _is_polygonal(candidate):
            continue
        if float(candidate.area) <= 0.01 or float(candidate.area) > float(max_area):
            continue
        if not candidate.intersects(point_geom):
            continue

        candidate_centroid_dist = float(candidate.centroid.distance(point_geom))
        current_centroid_dist = float(current_geom.centroid.distance(point_geom))
        if candidate_centroid_dist > float(centroid_tolerance):
            continue
        if candidate_centroid_dist + float(centroid_improvement) >= current_centroid_dist:
            continue
        if _geometry_iou(candidate, seed_geom) < float(min_reference_iou):
            continue
        if candidate.equals_exact(current_geom, 0.0):
            continue

        result.at[idx, geom_col] = candidate
        result.at[idx, "capture_stage"] = "council_seed_wfs_polygonized_repair"
        result.at[idx, "capture_success"] = True
        updated += 1

    return updated


def _linked_union_is_spatially_coherent(
    child_geoms: list[object],
    *,
    max_child_area_m2: float = 5000.0,
    max_union_bbox_diag_m: float = 1000.0,
) -> bool:
    if not child_geoms:
        return False
    for geom in child_geoms:
        if float(getattr(geom, "area", 0.0) or 0.0) > max_child_area_m2:
            return False
    child_union = unary_union(child_geoms)
    if child_union is None or bool(getattr(child_union, "is_empty", False)):
        return False
    minx, miny, maxx, maxy = child_union.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    return bool(diag <= max_union_bbox_diag_m)


def _apply_linked_parent_unions(result: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int, int]:
    if "unique_key" not in result.columns or "variant_key" not in result.columns:
        return result, 0, 0
    out = result.copy()
    unioned = 0
    skipped = 0
    unique_values = out["unique_key"].map(_clean_link_value)
    variant_values = out["variant_key"].map(_clean_link_value)
    expanded_values = (
        out["_is_expanded_case"].map(_clean_link_value).str.lower()
        if "_is_expanded_case" in out.columns
        else None
    )
    for unique_key in unique_values[unique_values != ""].drop_duplicates().tolist():
        group_mask = unique_values.eq(unique_key)
        if expanded_values is not None:
            child_mask = group_mask & expanded_values.eq("yes")
            parent_mask = group_mask & ~expanded_values.eq("yes")
        else:
            child_mask = group_mask & variant_values.ne("") & variant_values.ne(unique_key)
            parent_mask = group_mask & (variant_values.eq(unique_key) | variant_values.eq(""))
        if not bool(child_mask.any()):
            continue
        if not bool(parent_mask.any()):
            skipped += 1
            continue
        child_geoms = list(out.loc[child_mask, out.geometry.name])
        if not child_geoms or not all(_is_polygonal(geom) for geom in child_geoms):
            skipped += 1
            continue
        if not _linked_union_is_spatially_coherent(child_geoms):
            skipped += 1
            continue
        parent_index = out.index[parent_mask][0]
        out.at[parent_index, out.geometry.name] = unary_union(child_geoms)
        if "capture_stage" in out.columns:
            out.at[parent_index, "capture_stage"] = "linked_parent_union"
        if "capture_success" in out.columns:
            out.at[parent_index, "capture_success"] = True
        unioned += 1
    return out, unioned, skipped


POINT_FALLBACK_STAGES = {
    "step1_council_seed_wfs_inline_no_move",
    "step2_os_wfs_merge_intersection",
    "fallback_nearest_os_wfs_merge",
    "fallback_nearest_council_land_inline_merge",
    "fallback_force_polygon_nearest_os_wfs_merge",
}


def _apply_snapped_polygonized_refinement(
    result: gpd.GeoDataFrame,
    point_reps: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    *,
    tolerance: float,
    search_radius: float,
    max_candidates: int,
    max_area: float,
    precision_grid: float,
    max_existing_symdiff: float,
) -> int:
    if result.empty or point_reps.empty or polygon_gdf.empty or float(precision_grid) <= 0.0:
        return 0

    fallback_ids = set(
        result.loc[
            result["capture_stage"].isin(POINT_FALLBACK_STAGES),
            "capture_src_id",
        ]
        .astype(int)
        .tolist()
    )
    if not fallback_ids:
        return 0

    refine_points = point_reps[point_reps["capture_src_id"].astype(int).isin(fallback_ids)].copy()
    if refine_points.empty:
        return 0

    picks = pick_polygonized_cell_centroid_combo(
        refine_points,
        polygon_gdf,
        tolerance=tolerance,
        search_radius=search_radius,
        max_candidates=max_candidates,
        max_area=max_area,
        desc="Point polygonized WFS cell snap refine",
    )
    if picks.empty:
        return 0

    row_by_id = {int(src_id): idx for idx, src_id in zip(result.index, result["capture_src_id"])}
    point_by_id = point_reps.set_index("capture_src_id")["geometry"].to_dict()
    updated = 0
    for pick in picks.itertuples():
        src_id = int(pick.capture_src_id)
        idx = row_by_id.get(src_id)
        if idx is None:
            continue
        current_geom = result.at[idx, result.geometry.name]
        if not _is_polygonal(current_geom):
            continue
        snapped = _round_geometry_coordinates(pick.geometry, float(precision_grid))
        snapped = _pick_single_polygon_with_anchor(snapped, point_by_id.get(src_id))
        if not _is_polygonal(snapped):
            continue
        if float(snapped.area) <= 0.01 or float(snapped.area) > float(max_area):
            continue
        existing_delta = float(snapped.symmetric_difference(current_geom).area)
        if existing_delta > float(max_existing_symdiff):
            continue
        result.at[idx, result.geometry.name] = snapped
        result.at[idx, "capture_stage"] = "point_polygonized_os_wfs_cell_snap_refine"
        result.at[idx, "capture_success"] = True
        updated += 1
    return updated


def _apply_point_polygonized_precision_model_refinement(
    result: gpd.GeoDataFrame,
    point_reps: gpd.GeoDataFrame,
    *,
    precision_grid: float,
    tolerance: float,
    max_area: float,
    max_existing_symdiff: float,
) -> int:
    if (
        result.empty
        or point_reps.empty
        or float(precision_grid) <= 0.0
        or float(tolerance) < 0.0
        or "capture_stage" not in result.columns
    ):
        return 0

    point_by_id = point_reps.set_index("capture_src_id")["geometry"].to_dict()
    stage_values = result["capture_stage"].astype(str)
    mask = stage_values.str.startswith("point_polygonized") & result["capture_src_id"].isin(point_by_id.keys())
    if not bool(mask.any()):
        return 0

    updated = 0
    geom_col = result.geometry.name
    for idx, row in result.loc[mask].iterrows():
        src_id = int(row["capture_src_id"])
        point_geom = point_by_id.get(src_id)
        geom = row[geom_col]
        if point_geom is None or point_geom.is_empty or not _is_polygonal(geom):
            continue

        snapped = wkb_loads(wkb_dumps(set_precision(geom, float(precision_grid))))
        if not _is_polygonal(snapped):
            continue
        existing_delta = float(snapped.symmetric_difference(geom).area)
        if existing_delta > float(max_existing_symdiff):
            continue
        snapped_area = float(snapped.area)
        if snapped_area <= 0.01 or snapped_area > float(max_area):
            continue
        if float(snapped.centroid.distance(point_geom)) > float(tolerance):
            continue
        if snapped.equals_exact(geom, 0.0):
            continue

        result.at[idx, geom_col] = snapped
        result.at[idx, "capture_stage"] = f"{row['capture_stage']}_precision_model"
        result.at[idx, "capture_success"] = True
        updated += 1

    return updated


def _recenter_point_fallback_geometries(
    result: gpd.GeoDataFrame,
    point_geom_by_id: dict[int, object],
    *,
    tolerance: float,
    max_area: float,
) -> int:
    fallback_stages = POINT_FALLBACK_STAGES
    shifted = 0
    for idx, row in result.iterrows():
        if row.get("capture_stage") not in fallback_stages:
            continue
        src_id = int(row["capture_src_id"])
        point_geom = point_geom_by_id.get(src_id)
        geom = row.geometry
        if (
            point_geom is None
            or bool(getattr(point_geom, "is_empty", False))
            or not _is_polygonal(geom)
            or float(geom.area) > float(max_area)
        ):
            continue
        centroid = geom.centroid
        dx = float(point_geom.x - centroid.x)
        dy = float(point_geom.y - centroid.y)
        if (dx * dx + dy * dy) ** 0.5 <= float(tolerance):
            continue
        result.at[idx, result.geometry.name] = translate(geom, xoff=dx, yoff=dy)
        shifted += 1
    return shifted


def main() -> None:
    args = parse_args()
    wfs_theme_include = _parse_wfs_theme_include(args.wfs_theme_include)

    target = load_layer(args.target_gpkg, args.target_layer).copy()
    target["capture_src_id"] = target.index + 1
    target_geom_type = target.geometry.geom_type.str.upper()
    point_mask = target_geom_type.isin(["POINT", "MULTIPOINT"])
    poly_mask = target_geom_type.isin(["POLYGON", "MULTIPOLYGON"])

    result = target.copy()
    result["capture_stage"] = "unprocessed"
    result["capture_success"] = False
    result["nearest_eligible_wfs_dist_m"] = None
    result["nearest_eligible_wfs_theme"] = None
    result["nearest_any_wfs_dist_m"] = None
    result["nearest_any_wfs_theme"] = None

    council_land: Optional[gpd.GeoDataFrame] = None
    os_wfs_raw_all: Optional[gpd.GeoDataFrame] = None
    os_wfs_basemap: Optional[gpd.GeoDataFrame] = None
    os_wfs_raw: Optional[gpd.GeoDataFrame] = None
    os_wfs_raw_dedup: Optional[gpd.GeoDataFrame] = None
    os_wfs_merge: Optional[gpd.GeoDataFrame] = None
    os_wfs_nearest_fallback: Optional[gpd.GeoDataFrame] = None

    def get_council_land() -> gpd.GeoDataFrame:
        nonlocal council_land
        if council_land is None:
            council_land = load_layer(args.council_land_gpkg)
            council_land = council_land[council_land.geometry.notna() & ~council_land.geometry.is_empty].copy()
        return council_land

    def get_os_wfs_raw_all() -> gpd.GeoDataFrame:
        nonlocal os_wfs_raw_all
        if os_wfs_raw_all is None:
            os_wfs_raw_all = load_layer(args.os_wfs_gpkg)
            os_wfs_raw_all = os_wfs_raw_all[
                os_wfs_raw_all.geometry.notna() & ~os_wfs_raw_all.geometry.is_empty
            ].copy()
        return os_wfs_raw_all

    def get_os_wfs_raw() -> gpd.GeoDataFrame:
        nonlocal os_wfs_raw
        if os_wfs_raw is None:
            raw = get_os_wfs_raw_all()
            os_wfs_raw = filter_wfs_theme_features(raw, include_terms=wfs_theme_include)
            if not args.quiet:
                print(
                    f"[INFO] WFS Theme filter {','.join(wfs_theme_include)}: "
                    f"{len(os_wfs_raw)} of {len(raw)} raw features kept"
                )
        return os_wfs_raw

    def get_os_wfs_raw_dedup() -> gpd.GeoDataFrame:
        nonlocal os_wfs_raw_dedup
        if os_wfs_raw_dedup is None:
            os_wfs_raw_dedup = get_os_wfs_raw().copy()
            os_wfs_raw_dedup["_capture_wkb"] = os_wfs_raw_dedup.geometry.apply(lambda geom: geom.wkb)
            os_wfs_raw_dedup = os_wfs_raw_dedup.drop_duplicates("_capture_wkb").drop(columns="_capture_wkb")
        return os_wfs_raw_dedup

    def get_os_wfs_basemap() -> gpd.GeoDataFrame:
        nonlocal os_wfs_basemap
        if os_wfs_basemap is None:
            os_wfs_basemap = get_os_wfs_raw()
        return os_wfs_basemap

    def get_os_wfs_merge() -> gpd.GeoDataFrame:
        nonlocal os_wfs_merge
        if os_wfs_merge is None:
            if args.os_wfs_merge_gpkg:
                loaded_merge = load_layer(args.os_wfs_merge_gpkg)
                loaded_merge = loaded_merge[
                    loaded_merge.geometry.notna() & ~loaded_merge.geometry.is_empty
                ].copy()
                os_wfs_merge = filter_wfs_theme_features(loaded_merge, include_terms=wfs_theme_include)
                if not args.quiet:
                    print(
                        f"[INFO] Using provided os_wfs_merge: {args.os_wfs_merge_gpkg}; "
                        f"WFS Theme filter {','.join(wfs_theme_include)} kept "
                        f"{len(os_wfs_merge)} of {len(loaded_merge)}"
                    )
            else:
                if not args.quiet:
                    print("[INFO] Building os_wfs_merge inline from os_wfs...")
                os_wfs_merge = build_wfs_merge_gdf(
                    get_os_wfs_basemap(),
                    get_council_land(),
                    include_terms=wfs_theme_include,
                )
                os_wfs_merge = os_wfs_merge[os_wfs_merge.geometry.notna() & ~os_wfs_merge.geometry.is_empty].copy()
                if args.save_built_os_wfs_merge:
                    save_path = Path(args.save_built_os_wfs_merge)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    if save_path.exists():
                        save_path.unlink()
                    os_wfs_merge.to_file(save_path, layer=args.save_built_os_wfs_merge_layer, driver="GPKG")
                    if not args.quiet:
                        print(f"[INFO] Saved inline-built os_wfs_merge to: {save_path}")
        return os_wfs_merge

    def get_os_wfs_nearest_fallback() -> gpd.GeoDataFrame:
        nonlocal os_wfs_nearest_fallback
        if os_wfs_nearest_fallback is None:
            source = get_os_wfs_merge()
            os_wfs_nearest_fallback = _filter_nearest_fallback_candidates(
                source,
                min_area=args.fallback_min_area,
                max_aspect_ratio=args.fallback_max_aspect_ratio,
                min_compactness=args.fallback_min_compactness,
            )
            if not args.quiet:
                print(
                    f"[INFO] Nearest fallback shape filter kept "
                    f"{len(os_wfs_nearest_fallback)} of {len(source)} WFS merge features"
                )
        return os_wfs_nearest_fallback

    if not args.quiet:
        print(f"[INFO] Target total: {len(target)}")
        print(f"[INFO] Point targets: {int(point_mask.sum())}")
        print(f"[INFO] Polygon targets: {int(poly_mask.sum())}")

    point_targets = target.loc[point_mask & target.geometry.notna()].copy()
    resolved_points: Set[int] = set()
    if not point_targets.empty:
        council_land = get_council_land()
        os_wfs_basemap = get_os_wfs_basemap()
        os_wfs_merge = get_os_wfs_merge()

        point_reps = representative_points(point_targets[["capture_src_id", "geometry"]].copy())

        centroid_sources = [
            ("point_centroid_os_wfs_merge", os_wfs_merge),
            ("point_centroid_os_wfs", os_wfs_basemap),
        ]
        centroid_candidate_maps: dict[str, dict[int, object]] = {}
        merge_centroid_candidate_ids: Set[int] = set()
        for stage, source_gdf in centroid_sources:
            remaining_candidate_ids = (
                set(point_targets["capture_src_id"].astype(int))
                - set().union(*(set(m.keys()) for m in centroid_candidate_maps.values()))
            )
            if not remaining_candidate_ids:
                break
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(remaining_candidate_ids)].copy()
            centroid_pick = _pick_centroid_aligned_polygon(
                unresolved_reps,
                source_gdf,
                tolerance=args.point_centroid_tolerance,
            )
            centroid_map = dict(zip(centroid_pick["capture_src_id"].astype(int), centroid_pick.geometry))
            centroid_candidate_maps[stage] = centroid_map
            if stage == "point_centroid_os_wfs_merge":
                merge_centroid_candidate_ids = set(centroid_map)
            if not args.quiet:
                print(f"[INFO] Point centroid WFS candidates from {stage}: {len(centroid_map)}")

        wfs_hit_ids: Set[int] = set()
        for source_gdf in (os_wfs_merge, os_wfs_basemap):
            remaining_hit_ids = set(point_targets["capture_src_id"].astype(int)) - wfs_hit_ids
            if not remaining_hit_ids:
                break
            hit_reps = point_reps[point_reps["capture_src_id"].isin(remaining_hit_ids)].copy()
            hit_pick = pick_smallest_intersection_polygon(hit_reps, source_gdf)
            wfs_hit_ids |= set(hit_pick["capture_src_id"].astype(int).tolist())
        wfs_hit_ids |= set().union(*(set(m.keys()) for m in centroid_candidate_maps.values()))

        council_seed_attempted_ids: Set[int] = set()
        if wfs_hit_ids:
            wfs_hit_reps = point_reps[point_reps["capture_src_id"].isin(wfs_hit_ids)].copy()
            wfs_hit_council_inputs = pick_smallest_intersection_polygon(wfs_hit_reps, council_land)
            council_seed_attempted_ids = set(wfs_hit_council_inputs["capture_src_id"].astype(int).tolist())
            if not args.quiet:
                print(
                    f"[INFO] WFS-hit council seed candidates: {len(wfs_hit_council_inputs)} "
                    f"(from {len(wfs_hit_ids)} WFS-hit points)"
                )
            wfs_hit_council_outputs = run_inline_merge_batch(
                input_polygons=wfs_hit_council_inputs,
                basemap_gdf=os_wfs_basemap,
                min_iou=args.min_iou,
                distance_tolerance=0.0,
                area_tolerance=args.area_tolerance,
                max_candidates=args.max_candidates,
                max_combo_size=args.max_combo_size,
                window_area_scale=args.window_area_scale,
                desc="WFS-hit council-seed inline-merge",
            )
            wfs_hit_council_outputs = _filter_point_anchored_results(wfs_hit_council_outputs, point_reps)
            wfs_hit_council_map = dict(
                zip(wfs_hit_council_outputs["capture_src_id"].astype(int), wfs_hit_council_outputs.geometry)
            )
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=wfs_hit_council_map,
                stage="step1_council_seed_wfs_inline_no_move",
            )
            if not args.quiet:
                print(f"[INFO] WFS-hit council-seed inline resolved: {len(wfs_hit_council_map)}")

        merge_centroid_ids: Set[int] = set()
        for stage, centroid_map in centroid_candidate_maps.items():
            unresolved_map = {
                src_id: geom
                for src_id, geom in centroid_map.items()
                if src_id not in resolved_points
            }
            updated_ids = apply_geometry_updates(result=result, geometry_by_id=unresolved_map, stage=stage)
            resolved_points |= updated_ids
            if stage == "point_centroid_os_wfs_merge":
                merge_centroid_ids = set(updated_ids) | (merge_centroid_candidate_ids & set(updated_ids))
            if not args.quiet:
                print(f"[INFO] Point centroid direct WFS fallback resolved from {stage}: {len(unresolved_map)}")

        enable_point_wfs_combo = args.enable_slow_point_recovery or args.enable_point_wfs_combo
        enable_point_deep_combo = args.enable_slow_point_recovery or args.enable_point_deep_combo
        enable_point_merged_combo = args.enable_slow_point_recovery or args.enable_point_merged_combo
        enable_point_contained_union = args.enable_slow_point_recovery or args.enable_point_contained_union
        enable_point_polygonized_combo = args.enable_slow_point_recovery or args.enable_point_polygonized_combo
        enable_point_combined_polygonized_combo = (
            args.enable_slow_point_recovery or args.enable_point_combined_polygonized_combo
        )
        if not args.quiet and not any(
            [
                enable_point_wfs_combo,
                enable_point_deep_combo,
                enable_point_merged_combo,
                enable_point_contained_union,
                enable_point_polygonized_combo,
                enable_point_combined_polygonized_combo,
            ]
        ):
            print(
                "[INFO] Slow point recovery passes disabled by default "
                "(use --enable-slow-point-recovery to run them)."
            )

        if merge_centroid_ids and enable_point_wfs_combo:
            merge_centroid_reps = point_reps[point_reps["capture_src_id"].isin(merge_centroid_ids)].copy()
            refined_combo_pick = pick_centroid_aligned_wfs_combo(
                merge_centroid_reps,
                get_os_wfs_raw(),
                tolerance=args.point_centroid_tolerance,
                search_radius=args.point_combo_search_radius,
                max_candidates=args.point_combo_max_candidates,
                max_area=args.point_combo_max_area,
                desc="Point centroid WFS combo refine merged centroid",
            )
            refined_combo_map = dict(zip(refined_combo_pick["capture_src_id"].astype(int), refined_combo_pick.geometry))
            apply_geometry_updates(
                result=result,
                geometry_by_id=refined_combo_map,
                stage="point_centroid_os_wfs_combo_refined_merge",
            )
            if not args.quiet:
                print(f"[INFO] Point centroid WFS combo refined merged centroid: {len(refined_combo_map)}")

        unresolved_after_centroid = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if unresolved_after_centroid and enable_point_wfs_combo:
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_centroid)].copy()
            combo_pick = pick_centroid_aligned_wfs_combo(
                unresolved_reps,
                get_os_wfs_raw_dedup(),
                tolerance=args.point_centroid_tolerance,
                search_radius=args.point_combo_search_radius,
                max_candidates=args.point_combo_max_candidates,
                max_area=args.point_combo_max_area,
                desc="Point centroid WFS combo",
            )
            combo_map = dict(zip(combo_pick["capture_src_id"].astype(int), combo_pick.geometry))
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=combo_map,
                stage="point_centroid_os_wfs_combo",
            )
            if not args.quiet:
                print(f"[INFO] Point centroid WFS combo resolved: {len(combo_map)}")

        unresolved_after_combo = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if (
            unresolved_after_combo
            and enable_point_deep_combo
            and args.point_deep_combo_max_candidates > args.point_combo_max_candidates
        ):
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_combo)].copy()
            deep_combo_pick = pick_centroid_aligned_wfs_combo(
                unresolved_reps,
                get_os_wfs_raw_dedup(),
                tolerance=args.point_centroid_tolerance,
                search_radius=args.point_combo_search_radius,
                max_candidates=args.point_deep_combo_max_candidates,
                max_area=args.point_combo_max_area,
                nearest_matches=1,
                desc="Point centroid WFS combo deep pass",
            )
            deep_combo_map = dict(zip(deep_combo_pick["capture_src_id"].astype(int), deep_combo_pick.geometry))
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=deep_combo_map,
                stage="point_centroid_os_wfs_combo_deep",
            )
            if not args.quiet:
                print(f"[INFO] Point centroid WFS combo deep pass resolved: {len(deep_combo_map)}")

        unresolved_after_combo = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if unresolved_after_combo and enable_point_merged_combo:
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_combo)].copy()
            merge_combo_pick = pick_centroid_aligned_wfs_combo(
                unresolved_reps,
                os_wfs_merge,
                tolerance=args.point_centroid_tolerance,
                search_radius=args.point_merge_combo_search_radius,
                max_candidates=args.point_combo_max_candidates,
                max_area=args.point_combo_max_area,
                desc="Point centroid merged WFS combo",
            )
            merge_combo_map = dict(zip(merge_combo_pick["capture_src_id"].astype(int), merge_combo_pick.geometry))
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=merge_combo_map,
                stage="point_centroid_os_wfs_merge_combo",
            )
            if not args.quiet:
                print(f"[INFO] Point centroid merged WFS combo resolved: {len(merge_combo_map)}")

        unresolved_after_combo = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if unresolved_after_combo and enable_point_contained_union:
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_combo)].copy()
            contained_union_pick = _pick_council_contained_wfs_union(
                unresolved_reps,
                council_land,
                get_os_wfs_raw(),
                tolerance=args.point_centroid_tolerance,
                max_area=args.point_combo_max_area,
                max_container_area=args.point_contained_union_max_container_area,
            )
            contained_union_map = dict(
                zip(contained_union_pick["capture_src_id"].astype(int), contained_union_pick.geometry)
            )
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=contained_union_map,
                stage="point_centroid_council_contained_wfs_union",
            )
            if not args.quiet:
                print(f"[INFO] Point centroid council-contained WFS union resolved: {len(contained_union_map)}")


        unresolved_after_combo = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if unresolved_after_combo and enable_point_polygonized_combo:
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_combo)].copy()
            polygonized_combo_pick = pick_polygonized_cell_centroid_combo(
                unresolved_reps,
                get_os_wfs_raw_dedup(),
                tolerance=args.point_centroid_tolerance,
                search_radius=args.point_polygonize_search_radius,
                max_candidates=args.point_polygonize_max_candidates,
                max_area=args.point_combo_max_area,
                desc="Point polygonized WFS cell combo",
            )
            polygonized_combo_map = dict(
                zip(polygonized_combo_pick["capture_src_id"].astype(int), polygonized_combo_pick.geometry)
            )
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=polygonized_combo_map,
                stage="point_polygonized_os_wfs_cell_combo",
            )
            if not args.quiet:
                print(f"[INFO] Point polygonized WFS cell combo resolved: {len(polygonized_combo_map)}")

        unresolved_after_combo = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if unresolved_after_combo and enable_point_combined_polygonized_combo:
            unresolved_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_combo)].copy()
            combined_polygonized_pick = pick_polygonized_cell_centroid_combo_multi(
                unresolved_reps,
                [get_os_wfs_raw_dedup(), os_wfs_merge],
                tolerance=args.point_centroid_tolerance,
                search_radius=args.point_polygonize_search_radius,
                max_candidates=args.point_polygonize_max_candidates,
                max_area=args.point_combo_max_area,
                desc="Point combined polygonized source cell combo",
            )
            combined_polygonized_map = dict(
                zip(combined_polygonized_pick["capture_src_id"].astype(int), combined_polygonized_pick.geometry)
            )
            resolved_points |= apply_geometry_updates(
                result=result,
                geometry_by_id=combined_polygonized_map,
                stage="point_polygonized_combined_source_cell_combo",
            )
            if not args.quiet:
                print(f"[INFO] Point combined polygonized source cell combo resolved: {len(combined_polygonized_map)}")

        unresolved_after_combo = (
            set(point_targets["capture_src_id"].astype(int)) - resolved_points - council_seed_attempted_ids
        )
        step1_inputs = pick_smallest_intersection_polygon(
            point_reps[point_reps["capture_src_id"].isin(unresolved_after_combo)].copy(),
            council_land,
        )
        if not args.quiet:
            print(f"[INFO] Step1 candidates from council land: {len(step1_inputs)}")

        # Keep merge combination search, but disallow move attempts (distance_tolerance=0).
        step1_outputs = run_inline_merge_batch(
            input_polygons=step1_inputs,
            basemap_gdf=os_wfs_basemap,
            min_iou=args.min_iou,
            distance_tolerance=0.0,
            area_tolerance=args.area_tolerance,
            max_candidates=args.max_candidates,
            max_combo_size=args.max_combo_size,
            window_area_scale=args.window_area_scale,
            desc="Step1 inline-merge",
        )
        step1_outputs = _filter_point_anchored_results(step1_outputs, point_reps)
        step1_map = dict(zip(step1_outputs["capture_src_id"].astype(int), step1_outputs.geometry))
        resolved_points |= apply_geometry_updates(
            result=result,
            geometry_by_id=step1_map,
            stage="step1_council_seed_wfs_inline_no_move",
        )
        if not args.quiet:
            print(f"[INFO] Step1 resolved: {len(step1_map)}")

        council_driven_outputs = _pick_council_driven_wfs_overrides(
            result,
            point_reps,
            get_council_land(),
            [get_os_wfs_raw_dedup(), os_wfs_merge],
            stages={
                "step1_council_seed_wfs_inline_no_move",
                "point_source_centroid_precision_refine_raw",
                "point_source_centroid_precision_refine_merge",
            },
            min_iou=args.council_driven_wfs_min_iou,
            min_seed_coverage=args.council_driven_wfs_min_seed_coverage,
            min_wfs_coverage=args.council_driven_wfs_min_wfs_coverage,
            min_area_ratio=args.council_driven_wfs_min_area_ratio,
            max_seed_area=args.council_driven_wfs_max_seed_area,
            max_candidate_seed_area_ratio=args.council_driven_wfs_max_candidate_seed_area_ratio,
            max_candidate_area_extra=args.council_driven_wfs_max_candidate_area_extra,
        )
        council_driven_map = dict(
            zip(council_driven_outputs["capture_src_id"].astype(int), council_driven_outputs.geometry)
        )
        council_driven_updated = apply_geometry_updates(
            result=result,
            geometry_by_id=council_driven_map,
            stage="council_driven_wfs_reference",
        )
        resolved_points |= council_driven_updated
        if not args.quiet:
            print(f"[INFO] Council-driven WFS reference overrides: {len(council_driven_updated)}")

        unresolved_after_step1 = set(point_targets["capture_src_id"].astype(int)) - resolved_points
        if unresolved_after_step1:
            unresolved_point_reps = point_reps[point_reps["capture_src_id"].isin(unresolved_after_step1)].copy()
            step2_pick = pick_smallest_intersection_polygon(unresolved_point_reps, os_wfs_merge)
            step2_map = dict(zip(step2_pick["capture_src_id"].astype(int), step2_pick.geometry))
            resolved_points |= apply_geometry_updates(
                result=result, geometry_by_id=step2_map, stage="step2_os_wfs_merge_intersection"
            )
            if not args.quiet:
                print(f"[INFO] Step2 resolved: {len(step2_map)}")

            council_driven_step2_outputs = _pick_council_driven_wfs_overrides(
                result,
                point_reps,
                get_council_land(),
                [get_os_wfs_raw_dedup(), os_wfs_merge],
                stages={
                    "step1_council_seed_wfs_inline_no_move",
                    "council_driven_wfs_reference",
                    "step2_os_wfs_merge_intersection",
                },
                min_iou=args.council_driven_wfs_min_iou,
                min_seed_coverage=args.council_driven_wfs_min_seed_coverage,
                min_wfs_coverage=args.council_driven_wfs_min_wfs_coverage,
                min_area_ratio=args.council_driven_wfs_min_area_ratio,
                max_seed_area=args.council_driven_wfs_max_seed_area,
                max_candidate_seed_area_ratio=args.council_driven_wfs_max_candidate_seed_area_ratio,
                max_candidate_area_extra=args.council_driven_wfs_max_candidate_area_extra,
            )
            council_driven_step2_map = dict(
                zip(council_driven_step2_outputs["capture_src_id"].astype(int), council_driven_step2_outputs.geometry)
            )
            council_driven_step2_updated = apply_geometry_updates(
                result=result,
                geometry_by_id=council_driven_step2_map,
                stage="council_driven_wfs_reference",
            )
            resolved_points |= council_driven_step2_updated
            if not args.quiet:
                print(f"[INFO] Council-driven WFS reference overrides after Step2: {len(council_driven_step2_updated)}")

            unresolved_after_step2 = unresolved_after_step1 - set(step2_map.keys())
            if unresolved_after_step2 and not args.quiet:
                print(
                    f"[INFO] Skipping direct council-land fallback for {len(unresolved_after_step2)} points; "
                    "final point polygons must be WFS-derived."
                )

    poly_targets = target.loc[poly_mask & target.geometry.notna()].copy()
    if not poly_targets.empty:
        os_wfs_basemap = get_os_wfs_basemap()
        poly_inputs = poly_targets[["capture_src_id", "geometry"]].copy()
        step4_outputs = run_inline_merge_batch(
            input_polygons=poly_inputs,
            basemap_gdf=os_wfs_basemap,
            min_iou=args.min_iou,
            distance_tolerance=args.distance_tolerance,
            area_tolerance=args.area_tolerance,
            max_candidates=args.max_candidates,
            max_combo_size=args.max_combo_size,
            window_area_scale=args.window_area_scale,
            desc="Step4 inline-merge",
        )
        step4_map = dict(zip(step4_outputs["capture_src_id"].astype(int), step4_outputs.geometry))
        resolved_polys = apply_geometry_updates(result=result, geometry_by_id=step4_map, stage="step4_polygon_inline")
        if not args.quiet:
            print(f"[INFO] Step4 resolved inline: {len(resolved_polys)}")

        poly_all_ids = set(poly_targets["capture_src_id"].astype(int).tolist())
        unresolved_polys = poly_all_ids - resolved_polys
        if unresolved_polys:
            mask = result["capture_src_id"].isin(unresolved_polys)
            result.loc[mask, "capture_stage"] = "step4_polygon_original_fallback"
            result.loc[mask, "capture_success"] = True
            if not args.quiet:
                print(f"[INFO] Step4 fallback to original polygon: {len(unresolved_polys)}")

    unresolved_mask = ~result["capture_success"]
    if unresolved_mask.any():
        unresolved = result.loc[unresolved_mask, ["capture_src_id", "geometry"]].copy()
        unresolved_reps = representative_points(unresolved)

        nearest_wfs_merge = pick_nearest_polygon(
            unresolved_reps,
            get_os_wfs_nearest_fallback(),
            max_distance=args.point_nearest_wfs_max_distance,
        )
        nearest_wfs_merge_map = dict(zip(nearest_wfs_merge["capture_src_id"].astype(int), nearest_wfs_merge.geometry))
        resolved_by_wfs_merge = apply_geometry_updates(
            result=result,
            geometry_by_id=nearest_wfs_merge_map,
            stage="fallback_nearest_os_wfs_merge",
        )
        if not args.quiet:
            print(f"[INFO] Final fallback resolved by nearest os_wfs_merge: {len(resolved_by_wfs_merge)}")

        unresolved_ids_after_wfs = set(result.loc[~result["capture_success"], "capture_src_id"].astype(int).tolist())
        if unresolved_ids_after_wfs:
            unresolved_reps2 = unresolved_reps[unresolved_reps["capture_src_id"].isin(unresolved_ids_after_wfs)].copy()
            nearest_council = pick_nearest_polygon(
                unresolved_reps2,
                get_council_land(),
                max_distance=args.point_nearest_wfs_max_distance,
            )
            merged_from_nearest_council = run_inline_merge_batch(
                input_polygons=nearest_council,
                basemap_gdf=get_os_wfs_basemap(),
                min_iou=args.min_iou,
                distance_tolerance=0.0,
                area_tolerance=args.area_tolerance,
                max_candidates=args.max_candidates,
                max_combo_size=args.max_combo_size,
                window_area_scale=args.window_area_scale,
                desc="Final fallback council+merge",
            )
            merged_from_council_map = dict(
                zip(merged_from_nearest_council["capture_src_id"].astype(int), merged_from_nearest_council.geometry)
            )
            resolved_by_council_merge = apply_geometry_updates(
                result=result,
                geometry_by_id=merged_from_council_map,
                stage="fallback_nearest_council_land_inline_merge",
            )
            if not args.quiet:
                print(f"[INFO] Final fallback resolved by nearest council land + merge: {len(resolved_by_council_merge)}")

    remaining_mask = ~result["capture_success"]
    if remaining_mask.any():
        result.loc[remaining_mask, "capture_stage"] = "original_geometry_fallback"
        result.loc[remaining_mask, "capture_success"] = True

    source_geom_by_id = target.set_index("capture_src_id")["geometry"].to_dict()
    result = _force_single_polygon_per_row(
        result,
        source_geom_by_id,
        bridge_width=args.single_polygon_bridge_width,
    )
    recentered_point_fallbacks = 0
    snapped_polygonized_refinements = 0
    if not point_targets.empty:
        snapped_polygonized_refinements = _apply_snapped_polygonized_refinement(
            result,
            point_reps,
            get_os_wfs_raw_dedup(),
            tolerance=args.point_polygonize_snap_refine_tolerance,
            search_radius=args.point_polygonize_search_radius,
            max_candidates=args.point_polygonize_max_candidates,
            max_area=args.point_combo_max_area,
            precision_grid=args.point_polygonize_snap_refine_grid,
            max_existing_symdiff=args.point_polygonize_snap_refine_max_existing_symdiff,
        )

    non_polygon_mask = ~_polygon_only_mask(result)
    if non_polygon_mask.any():
        unresolved_ids = set(result.loc[non_polygon_mask, "capture_src_id"].astype(int).tolist())
        fallback_inputs = target[target["capture_src_id"].isin(unresolved_ids)][["capture_src_id", "geometry"]].copy()
        fallback_points = representative_points(fallback_inputs)
        force_nearest = pick_nearest_polygon(
            fallback_points,
            get_os_wfs_nearest_fallback(),
            max_distance=args.point_nearest_wfs_max_distance,
        )
        force_nearest_map = dict(zip(force_nearest["capture_src_id"].astype(int), force_nearest.geometry))
        apply_geometry_updates(
            result=result,
            geometry_by_id=force_nearest_map,
            stage="fallback_force_polygon_nearest_os_wfs_merge",
        )
        result = _force_single_polygon_per_row(
            result,
            source_geom_by_id,
            bridge_width=args.single_polygon_bridge_width,
        )

    result, linked_parent_unions, linked_parent_union_skipped = _apply_linked_parent_unions(result)

    step2_precision_changed = _apply_stage_precision_grid(
        result,
        "step2_os_wfs_merge_intersection",
        args.step2_output_precision_grid,
    )
    step3_intersection_precision_changed = _apply_stage_precision_grid(
        result,
        "step3_council_land_intersection_fallback",
        args.step3_intersection_output_precision_grid,
    )

    source_centroid_precision_refinements = 0
    chargegeog_template_refinements = 0
    council_seed_wfs_repairs = 0
    if not point_targets.empty:
        source_centroid_precision_refinements = _apply_point_source_centroid_precision_refinement(
            result,
            point_reps,
            [
                ("merge", get_os_wfs_merge()),
                ("raw", get_os_wfs_raw()),
            ],
            stages=POINT_FALLBACK_STAGES,
            precision_grid=args.point_source_centroid_refine_grid,
            tolerance=args.point_source_centroid_refine_tolerance,
            max_area=args.point_source_centroid_refine_max_area,
        )
        chargegeog_template_refinements = 0
        council_seed_wfs_repairs = _apply_council_seed_wfs_polygonized_repair(
            result,
            point_reps,
            get_council_land(),
            [get_os_wfs_raw_dedup(), get_os_wfs_merge()],
            stages={
                "step1_council_seed_wfs_inline_no_move",
                "council_driven_wfs_reference",
                "step2_os_wfs_merge_intersection",
                "fallback_nearest_os_wfs_merge",
                "point_source_centroid_precision_refine_raw",
                "point_source_centroid_precision_refine_merge",
            },
            search_radius=args.council_seed_wfs_repair_search_radius,
            max_area=args.point_combo_max_area,
            max_cells=args.council_seed_wfs_repair_max_cells,
            min_cell_reference_coverage=args.council_seed_wfs_repair_min_cell_reference_coverage,
            min_reference_iou=args.council_seed_wfs_repair_min_reference_iou,
            centroid_tolerance=args.council_seed_wfs_repair_centroid_tolerance,
            centroid_improvement=args.council_seed_wfs_repair_centroid_improvement,
        )

    polygonized_precision_model_refinements = 0
    if not point_targets.empty:
        polygonized_precision_model_refinements = _apply_point_polygonized_precision_model_refinement(
            result,
            point_reps,
            precision_grid=args.point_polygonized_precision_model_grid,
            tolerance=args.point_polygonized_precision_model_tolerance,
            max_area=args.point_combo_max_area,
            max_existing_symdiff=args.point_polygonized_precision_model_max_existing_symdiff,
        )

    late_council_driven_updated: set[int] = set()
    if not point_targets.empty:
        late_council_driven_outputs = _pick_council_driven_wfs_overrides(
            result,
            point_reps,
            get_council_land(),
            [get_os_wfs_raw_dedup(), get_os_wfs_merge()],
            stages={
                "step1_council_seed_wfs_inline_no_move",
                "step2_os_wfs_merge_intersection",
                "fallback_nearest_os_wfs_merge",
                "linked_parent_union",
                "point_source_centroid_precision_refine_raw",
                "point_source_centroid_precision_refine_merge",
            },
            min_iou=args.late_council_driven_wfs_min_iou,
            min_seed_coverage=args.late_council_driven_wfs_min_coverage,
            min_wfs_coverage=args.late_council_driven_wfs_min_coverage,
            min_area_ratio=args.late_council_driven_wfs_min_area_ratio,
            max_seed_area=args.late_council_driven_wfs_max_seed_area,
            max_candidate_seed_area_ratio=args.council_driven_wfs_max_candidate_seed_area_ratio,
            max_candidate_area_extra=args.council_driven_wfs_max_candidate_area_extra,
            max_current_seed_iou=args.late_council_driven_wfs_max_current_seed_iou,
        )
        late_council_driven_map = dict(
            zip(late_council_driven_outputs["capture_src_id"].astype(int), late_council_driven_outputs.geometry)
        )
        late_council_driven_updated = apply_geometry_updates(
            result=result,
            geometry_by_id=late_council_driven_map,
            stage="council_driven_wfs_reference_late",
        )
        if not args.quiet:
            print(f"[INFO] Late council-driven WFS reference overrides: {len(late_council_driven_updated)}")

    reference_wfs_qa_counts: dict[str, int] = {}
    if not point_targets.empty and not args.disable_reference_constrained_wfs_qa:
        reference_wfs_qa_counts = _apply_reference_constrained_wfs_qa(
            result,
            point_reps,
            get_council_land(),
            [get_os_wfs_raw_dedup(), get_os_wfs_merge()],
            min_seed_area=args.reference_qa_min_seed_area,
            max_seed_area=args.reference_qa_max_seed_area,
            trim_max_seed_area=args.reference_qa_trim_max_seed_area,
            trim_min_seed_coverage=args.reference_qa_trim_min_seed_coverage,
            trim_min_current_coverage=args.reference_qa_trim_min_current_coverage,
            trim_min_outside_area=args.reference_qa_trim_min_outside_area,
            trim_max_missing_ratio=args.reference_qa_trim_max_missing_ratio,
            completion_min_seed_coverage=args.reference_qa_completion_min_seed_coverage,
            completion_min_coverage_gain=args.reference_qa_completion_min_coverage_gain,
            completion_max_pieces=args.reference_qa_completion_max_pieces,
            completion_min_area_ratio=args.reference_qa_completion_min_area_ratio,
            raw_union_max_outside_area=args.reference_qa_raw_union_max_outside_area,
            bridge_width=args.single_polygon_bridge_width,
        )
        if not args.quiet:
            total_reference_wfs_qa = sum(reference_wfs_qa_counts.values())
            print(f"[INFO] Reference-constrained WFS QA updates: {total_reference_wfs_qa}")

    final_polygonal_extractions = _extract_polygonal_geometries(result)

    point_target_ids = (
        set(point_targets["capture_src_id"].astype(int).tolist())
        if not point_targets.empty
        else set()
    )
    final_non_polygon_point_mask = (~_polygon_only_mask(result)) & result["capture_src_id"].isin(point_target_ids)
    if final_non_polygon_point_mask.any():
        failed_ids = set(result.loc[final_non_polygon_point_mask, "capture_src_id"].astype(int).tolist())
        failed_points = point_reps[point_reps["capture_src_id"].astype(int).isin(failed_ids)].copy()
        _annotate_wfs_failure_diagnostics(
            result,
            failed_points,
            eligible_wfs=get_os_wfs_raw(),
            any_wfs=get_os_wfs_raw_all(),
        )
        geom_col = result.geometry.name
        result.loc[final_non_polygon_point_mask, geom_col] = [
            Polygon() for _ in range(int(final_non_polygon_point_mask.sum()))
        ]
        result.loc[
            final_non_polygon_point_mask,
            "capture_stage",
        ] = "no_wfs_building_land_within_search_radius"
        result.loc[final_non_polygon_point_mask, "capture_success"] = False

    final_non_polygon_mask = (~_polygon_only_mask(result)) & result["capture_success"].fillna(False).astype(bool)
    if final_non_polygon_mask.any():
        bad_ids = result.loc[final_non_polygon_mask, "capture_src_id"].astype(int).tolist()
        raise RuntimeError(
            f"Final output contains non-polygon geometries for capture_src_id={bad_ids}. "
            "Please check os_wfs/os_wfs_merge availability."
        )

    result = _force_single_polygon_per_row(
        result,
        source_geom_by_id,
        bridge_width=args.single_polygon_bridge_width,
    )
    no_holes_rows_changed, no_holes_removed = _remove_holes_from_result(result)

    final_precision_changed = _apply_final_precision_grid(result, args.final_output_precision_grid)
    result = _force_single_polygon_per_row(
        result,
        source_geom_by_id,
        bridge_width=args.single_polygon_bridge_width,
    )
    post_precision_no_holes_rows_changed, post_precision_no_holes_removed = _remove_holes_from_result(result)

    final_non_polygon_mask = (~_polygon_only_mask(result)) & result["capture_success"].fillna(False).astype(bool)
    if final_non_polygon_mask.any():
        bad_ids = result.loc[final_non_polygon_mask, "capture_src_id"].astype(int).tolist()
        raise RuntimeError(
            f"Final precision grid produced non-polygon geometries for capture_src_id={bad_ids}. "
            "Set --final-output-precision-grid 0 to disable output rounding."
        )
    final_non_single_polygon_mask = (
        ~_single_polygon_only_mask(result)
    ) & result["capture_success"].fillna(False).astype(bool)
    if final_non_single_polygon_mask.any():
        bad_ids = result.loc[final_non_single_polygon_mask, "capture_src_id"].astype(int).tolist()
        raise RuntimeError(
            f"Final output contains non-single Polygon geometries for capture_src_id={bad_ids}."
        )
    final_holes = int(result.geometry.apply(_polygon_hole_count).sum())
    if final_holes:
        raise RuntimeError(f"Final output contains {final_holes} polygon holes after no-hole enforcement.")

    output_path = Path(args.output_gpkg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    result.to_file(output_path, layer=args.output_layer, driver="GPKG")

    if not args.quiet:
        stage_counts = result["capture_stage"].value_counts(dropna=False)
        print("[INFO] Stage counts:")
        for stage, cnt in stage_counts.items():
            print(f"  - {stage}: {cnt}")
        if linked_parent_unions or linked_parent_union_skipped:
            print(
                f"[INFO] Linked parent unions: "
                f"applied={linked_parent_unions} skipped={linked_parent_union_skipped}"
            )
        if recentered_point_fallbacks:
            print(f"[INFO] Recentered point fallback geometries: {recentered_point_fallbacks}")
        if snapped_polygonized_refinements:
            print(f"[INFO] Snapped polygonized fallback refinements: {snapped_polygonized_refinements}")
        if step2_precision_changed:
            print(
                f"[INFO] Step2 output precision grid "
                f"{args.step2_output_precision_grid} changed {step2_precision_changed} geometries"
            )
        if step3_intersection_precision_changed:
            print(
                f"[INFO] Step3 intersection output precision grid "
                f"{args.step3_intersection_output_precision_grid} changed "
                f"{step3_intersection_precision_changed} geometries"
            )
        if source_centroid_precision_refinements:
            print(
                f"[INFO] Source centroid precision fallback refinements: "
                f"{source_centroid_precision_refinements}"
            )
        if chargegeog_template_refinements:
            print(f"[INFO] Chargegeog template fallback refinements: {chargegeog_template_refinements}")
        if council_seed_wfs_repairs:
            print(f"[INFO] Council-seed WFS polygonized repairs: {council_seed_wfs_repairs}")
        if polygonized_precision_model_refinements:
            print(f"[INFO] Point polygonized precision-model refinements: {polygonized_precision_model_refinements}")
        if late_council_driven_updated:
            print(f"[INFO] Late council-driven WFS reference overrides: {len(late_council_driven_updated)}")
        if reference_wfs_qa_counts:
            for stage, cnt in sorted(reference_wfs_qa_counts.items()):
                print(f"[INFO] {stage}: {cnt}")
        if final_polygonal_extractions:
            print(f"[INFO] Final polygonal geometry extractions: {final_polygonal_extractions}")
        total_no_holes_rows_changed = no_holes_rows_changed + post_precision_no_holes_rows_changed
        total_no_holes_removed = no_holes_removed + post_precision_no_holes_removed
        if total_no_holes_removed:
            print(
                f"[INFO] Removed polygon holes: "
                f"{total_no_holes_removed} holes across {total_no_holes_rows_changed} rows"
            )
        if final_precision_changed:
            print(
                f"[INFO] Final output precision grid "
                f"{args.final_output_precision_grid} changed {final_precision_changed} geometries"
            )
    print(f"[DONE] Wrote {len(result)} features to: {output_path} (layer={args.output_layer})")


if __name__ == "__main__":
    main()

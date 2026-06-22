from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from _core.config import add_config_argument, get_config_section_from_argv, require_configured
from _core.io import load_layer
from _core.wfs_merge import filter_wfs_theme_features, resolve_theme_field


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv(
        "postprocess_existing_wfs_theme",
        include_package_defaults=True,
    )
    parser = argparse.ArgumentParser(
        description=(
            "Post-process an existing capture output so retained/replacement polygons "
            "are constrained to WFS Theme building/land features."
        ),
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--input-gpkg", help="Existing capture output GPKG.")
    parser.add_argument("--input-layer", help="Existing capture layer. Defaults to first layer.")
    parser.add_argument("--point-gpkg", help="Original point GPKG used as fallback anchor.")
    parser.add_argument("--point-layer", help="Original point layer. Defaults to first layer.")
    parser.add_argument("--os-wfs-gpkg", help="Raw OS WFS polygon GPKG.")
    parser.add_argument("--os-wfs-merge-gpkg", help="Merged OS WFS polygon GPKG.")
    parser.add_argument("--output-gpkg", help="Post-processed output GPKG.")
    parser.add_argument("--output-layer", help="Output layer name.")
    parser.add_argument("--wfs-theme-include", help="Comma-separated Theme substrings.")
    parser.add_argument("--min-current-wfs-coverage", type=float)
    parser.add_argument("--nearest-wfs-max-distance", type=float)
    parser.add_argument(
        "--road-point-nearest-wfs-max-distance",
        type=float,
        help="Stricter replacement distance when the point is on a non-building/land WFS feature.",
    )
    parser.add_argument("--fallback-min-area", type=float)
    parser.add_argument("--fallback-max-aspect-ratio", type=float)
    parser.add_argument("--fallback-min-compactness", type=float)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(
        args,
        ("input_gpkg", "point_gpkg", "os_wfs_gpkg", "os_wfs_merge_gpkg", "output_gpkg"),
        "postprocess_existing_wfs_theme",
    )
    return args


def _parse_terms(value: str) -> tuple[str, ...]:
    return tuple(term.strip().lower() for term in str(value).split(",") if term.strip())


def _is_allowed_theme(theme: object, terms: Sequence[str]) -> bool:
    text = str(theme or "").lower()
    return any(term in text for term in terms)


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
    return float(4.0 * 3.141592653589793 * geom.area / (geom.length * geom.length))


def _filter_fallback_candidates(
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
        mask &= geom.area >= float(min_area)
    if float(max_aspect_ratio) > 0.0:
        mask &= geom.apply(_geometry_aspect_ratio) <= float(max_aspect_ratio)
    if float(min_compactness) > 0.0:
        mask &= geom.apply(_geometry_compactness) >= float(min_compactness)
    return gdf.loc[mask].copy()


def _ensure_capture_src_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if "capture_src_id" not in out.columns:
        out["capture_src_id"] = out.index + 1
    out["capture_src_id"] = out["capture_src_id"].astype(int)
    return out


def _align_points_to_result(result: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = _ensure_capture_src_id(points)
    if "unique_key" in result.columns and "unique_key" in points.columns:
        keys = result["unique_key"]
        if keys.notna().all() and keys.is_unique and points["unique_key"].is_unique:
            aligned = result[["capture_src_id", "unique_key"]].merge(
                points.drop(columns=["capture_src_id"], errors="ignore"),
                on="unique_key",
                how="left",
            )
            return gpd.GeoDataFrame(aligned, geometry="geometry", crs=points.crs)
    return points[points["capture_src_id"].isin(result["capture_src_id"])].copy()


def _coverage_by_wfs(result: gpd.GeoDataFrame, eligible_wfs: gpd.GeoDataFrame, quiet: bool) -> pd.Series:
    sidx = eligible_wfs.sindex
    coverage: dict[int, float] = {}
    total = len(result)
    for pos, row in enumerate(result[["capture_src_id", "geometry"]].itertuples(index=False), start=1):
        src_id = int(row.capture_src_id)
        geom = row.geometry
        if geom is None or geom.is_empty or float(geom.area) <= 0.0:
            coverage[src_id] = 0.0
        else:
            candidate_idx = list(sidx.query(geom, predicate="intersects"))
            if not candidate_idx:
                coverage[src_id] = 0.0
            else:
                union_geom = unary_union([eligible_wfs.geometry.iloc[int(idx)] for idx in candidate_idx])
                coverage[src_id] = float(geom.intersection(union_geom).area) / max(float(geom.area), 1e-9)
        if not quiet and pos % 500 == 0:
            print(f"[INFO] Coverage checked {pos}/{total}")
    return result["capture_src_id"].astype(int).map(coverage).fillna(0.0)


def _point_output_distance(result: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> pd.Series:
    point_by_id = points.set_index("capture_src_id")["geometry"].to_dict()
    distances: dict[int, float] = {}
    for row in result[["capture_src_id", "geometry"]].itertuples(index=False):
        src_id = int(row.capture_src_id)
        point = point_by_id.get(src_id)
        geom = row.geometry
        if point is None or geom is None or point.is_empty or geom.is_empty:
            distances[src_id] = float("inf")
        else:
            distances[src_id] = float(point.distance(geom))
    return result["capture_src_id"].astype(int).map(distances)


def _nearest_info(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    max_distance: Optional[float] = None,
) -> pd.DataFrame:
    if points.empty or polygons.empty:
        return pd.DataFrame(columns=["capture_src_id", "dist_m", "index_right", "nearest_theme"])
    kwargs = {}
    if max_distance is not None and float(max_distance) > 0.0:
        kwargs["max_distance"] = float(max_distance)
    nearest = gpd.sjoin_nearest(
        points[["capture_src_id", "geometry"]],
        polygons[["geometry"]],
        how="left",
        distance_col="dist_m",
        **kwargs,
    )
    nearest = nearest.dropna(subset=["index_right"]).copy()
    if nearest.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "dist_m", "index_right", "nearest_theme"], crs=points.crs)
    nearest["index_right"] = nearest["index_right"].astype(int)
    nearest = nearest.sort_values(["capture_src_id", "dist_m"]).drop_duplicates("capture_src_id", keep="first")
    try:
        theme_field = resolve_theme_field(polygons)
        nearest["nearest_theme"] = nearest["index_right"].map(polygons[theme_field].to_dict())
    except ValueError:
        nearest["nearest_theme"] = ""
    return nearest[["capture_src_id", "dist_m", "index_right", "nearest_theme"]].copy()


def main() -> None:
    args = parse_args()
    terms = _parse_terms(args.wfs_theme_include)

    result = _ensure_capture_src_id(load_layer(args.input_gpkg, args.input_layer))
    points = _align_points_to_result(result, load_layer(args.point_gpkg, args.point_layer))
    points = _ensure_capture_src_id(points)
    point_reps = points[["capture_src_id", "geometry"]].copy()
    point_reps = point_reps[point_reps.geometry.apply(lambda value: value is not None and not value.is_empty)].copy()

    raw_wfs_all = load_layer(args.os_wfs_gpkg)
    raw_wfs_all = raw_wfs_all[raw_wfs_all.geometry.apply(lambda value: value is not None and not value.is_empty)].copy()
    raw_wfs = filter_wfs_theme_features(raw_wfs_all, include_terms=terms)

    merged_wfs_all = load_layer(args.os_wfs_merge_gpkg)
    merged_wfs_all = merged_wfs_all[
        merged_wfs_all.geometry.apply(lambda value: value is not None and not value.is_empty)
    ].copy()
    merged_wfs = filter_wfs_theme_features(merged_wfs_all, include_terms=terms)
    fallback_wfs = _filter_fallback_candidates(
        merged_wfs,
        min_area=args.fallback_min_area,
        max_aspect_ratio=args.fallback_max_aspect_ratio,
        min_compactness=args.fallback_min_compactness,
    )

    if not args.quiet:
        print(f"[INFO] Rows: {len(result)}")
        print(f"[INFO] Raw WFS eligible: {len(raw_wfs)} / {len(raw_wfs_all)}")
        print(f"[INFO] Merged WFS eligible: {len(merged_wfs)} / {len(merged_wfs_all)}")
        print(f"[INFO] Nearest fallback candidates after shape filters: {len(fallback_wfs)}")

    result["postprocess_wfs_coverage"] = _coverage_by_wfs(result, merged_wfs, args.quiet)
    result["postprocess_point_output_dist_m"] = _point_output_distance(result, point_reps)
    result["postprocess_action"] = "keep"
    result["postprocess_reason"] = ""
    result["nearest_eligible_wfs_dist_m"] = None
    result["nearest_eligible_wfs_theme"] = None
    result["nearest_any_wfs_dist_m"] = None
    result["nearest_any_wfs_theme"] = None

    nearest_any_all = _nearest_info(point_reps, raw_wfs_all)
    nearest_any_all_by_id = nearest_any_all.set_index("capture_src_id").to_dict("index") if not nearest_any_all.empty else {}
    road_limited_ids: set[int] = set()
    for src_id, info in nearest_any_all_by_id.items():
        if float(info["dist_m"]) <= 0.25 and not _is_allowed_theme(info["nearest_theme"], terms):
            road_limited_ids.add(int(src_id))

    coverage_suspect_mask = result["postprocess_wfs_coverage"] < float(args.min_current_wfs_coverage)
    road_too_far_mask = result["capture_src_id"].astype(int).isin(road_limited_ids) & (
        result["postprocess_point_output_dist_m"] > float(args.road_point_nearest_wfs_max_distance)
    )
    suspect_mask = coverage_suspect_mask | road_too_far_mask
    suspect_ids = set(result.loc[suspect_mask, "capture_src_id"].astype(int).tolist())
    suspect_points = point_reps[point_reps["capture_src_id"].isin(suspect_ids)].copy()

    nearest_eligible = _nearest_info(
        suspect_points,
        fallback_wfs,
        max_distance=args.nearest_wfs_max_distance,
    )
    nearest_any = nearest_any_all[nearest_any_all["capture_src_id"].isin(suspect_ids)].copy()
    eligible_by_id = nearest_eligible.set_index("capture_src_id").to_dict("index") if not nearest_eligible.empty else {}
    any_by_id = nearest_any.set_index("capture_src_id").to_dict("index") if not nearest_any.empty else {}

    fallback_geom = fallback_wfs.geometry.to_dict()
    geom_col = result.geometry.name
    replaced = 0
    failed = 0
    road_limited = 0
    for idx, row in result.loc[suspect_mask].iterrows():
        src_id = int(row["capture_src_id"])
        any_info = any_by_id.get(src_id)
        if any_info:
            result.at[idx, "nearest_any_wfs_dist_m"] = float(any_info["dist_m"])
            result.at[idx, "nearest_any_wfs_theme"] = str(any_info["nearest_theme"])

        elig = eligible_by_id.get(src_id)
        if elig:
            result.at[idx, "nearest_eligible_wfs_dist_m"] = float(elig["dist_m"])
            result.at[idx, "nearest_eligible_wfs_theme"] = str(elig["nearest_theme"])

        point_on_noneligible_wfs = src_id in road_limited_ids
        max_dist = (
            float(args.road_point_nearest_wfs_max_distance)
            if point_on_noneligible_wfs
            else float(args.nearest_wfs_max_distance)
        )
        if point_on_noneligible_wfs:
            road_limited += 1

        if elig and float(elig["dist_m"]) <= max_dist and int(elig["index_right"]) in fallback_geom:
            result.at[idx, geom_col] = fallback_geom[int(elig["index_right"])]
            result.at[idx, "capture_stage"] = "postprocess_nearest_building_land_wfs"
            result.at[idx, "capture_success"] = True
            result.at[idx, "postprocess_action"] = "replace"
            result.at[idx, "postprocess_reason"] = "coverage_below_threshold"
            replaced += 1
        else:
            result.at[idx, geom_col] = Polygon()
            result.at[idx, "capture_stage"] = "postprocess_no_wfs_building_land_within_search_radius"
            result.at[idx, "capture_success"] = False
            result.at[idx, "postprocess_action"] = "fail"
            result.at[idx, "postprocess_reason"] = (
                "point_on_noneligible_wfs_no_close_building_land"
                if point_on_noneligible_wfs
                else "no_building_land_wfs_within_search_radius"
            )
            failed += 1

    out_path = Path(args.output_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    result.to_file(out_path, layer=args.output_layer, driver="GPKG")

    kept = int((result["postprocess_action"] == "keep").sum())
    if not args.quiet:
        print(f"[INFO] Suspect below coverage threshold: {int(coverage_suspect_mask.sum())}")
        print(f"[INFO] Suspect by road point distance: {int(road_too_far_mask.sum())}")
        print(f"[INFO] Total suspects: {len(suspect_ids)}")
        print(f"[INFO] Points on non-building/land WFS: {len(road_limited_ids)}")
        print(f"[INFO] Suspect road/noneligible-limited: {road_limited}")
        print(f"[INFO] Kept: {kept}; replaced: {replaced}; failed: {failed}")
    print(f"[DONE] Wrote {len(result)} features to: {out_path} (layer={args.output_layer})")


if __name__ == "__main__":
    main()

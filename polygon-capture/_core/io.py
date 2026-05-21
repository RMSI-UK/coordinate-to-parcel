from __future__ import annotations

from typing import Dict, Optional, Set

import geopandas as gpd


def load_layer(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:27700")
    elif str(gdf.crs).upper() != "EPSG:27700":
        gdf = gdf.to_crs("EPSG:27700")
    return gdf


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    reps = gdf.copy()
    reps = reps[reps.geometry.notna()].copy()
    reps = reps[~reps.geometry.is_empty].copy()
    reps["geometry"] = reps.geometry.representative_point()
    return reps


def pick_smallest_intersection_polygon(
    point_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
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
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs="EPSG:27700")

    areas = polygon_gdf.geometry.area
    joined["poly_area"] = joined["index_right"].map(areas)
    chosen = (
        joined.sort_values(["capture_src_id", "poly_area", "index_right"], ascending=[True, True, True])
        .drop_duplicates(subset=["capture_src_id"], keep="first")
        .copy()
    )
    chosen["geometry"] = chosen["index_right"].map(polygon_gdf.geometry)
    return gpd.GeoDataFrame(chosen[["capture_src_id", "geometry"]], geometry="geometry", crs=polygon_gdf.crs)


def pick_nearest_polygon(
    point_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    max_distance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    if point_gdf.empty or polygon_gdf.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs="EPSG:27700")
    nearest_kwargs = {}
    if max_distance is not None and float(max_distance) > 0.0:
        nearest_kwargs["max_distance"] = float(max_distance)
    nearest = gpd.sjoin_nearest(
        point_gdf[["capture_src_id", "geometry"]],
        polygon_gdf[["geometry"]],
        how="left",
        distance_col="dist_m",
        **nearest_kwargs,
    )
    nearest = nearest.dropna(subset=["index_right"]).copy()
    if nearest.empty:
        return gpd.GeoDataFrame(columns=["capture_src_id", "geometry"], geometry="geometry", crs="EPSG:27700")
    nearest["geometry"] = nearest["index_right"].astype(int).map(polygon_gdf.geometry)
    out = nearest.sort_values(["capture_src_id", "dist_m"]).drop_duplicates(subset=["capture_src_id"], keep="first")
    return gpd.GeoDataFrame(out[["capture_src_id", "geometry"]], geometry="geometry", crs=polygon_gdf.crs)


def apply_geometry_updates(
    result: gpd.GeoDataFrame,
    geometry_by_id: Dict[int, object],
    stage: str,
) -> Set[int]:
    if not geometry_by_id:
        return set()
    mask = result["capture_src_id"].isin(geometry_by_id.keys())
    if not mask.any():
        return set()
    result.loc[mask, "geometry"] = result.loc[mask, "capture_src_id"].map(geometry_by_id)
    result.loc[mask, "capture_stage"] = stage
    result.loc[mask, "capture_success"] = True
    return set(result.loc[mask, "capture_src_id"].astype(int).tolist())

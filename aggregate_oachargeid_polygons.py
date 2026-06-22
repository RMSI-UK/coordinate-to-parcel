#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.ops import unary_union


TARGET_CRS = "EPSG:27700"
EPS = 1e-9


def choose_layer(path: str, layer: Optional[str]) -> str:
    if layer:
        return layer
    layers = pyogrio.list_layers(path)
    if len(layers) == 0:
        raise ValueError(f"No layers found in {path}")
    return str(layers[0][0])


def load_layer(path: str, layer: Optional[str] = None, bbox=None) -> gpd.GeoDataFrame:
    layer_name = choose_layer(path, layer)
    kwargs = {"layer": layer_name, "engine": "pyogrio"}
    if bbox is not None:
        kwargs["bbox"] = bbox
    gdf = gpd.read_file(path, **kwargs)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def polygon_parts(geom) -> list[object]:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return []
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom_type == "GeometryCollection":
        parts: list[object] = []
        for child in geom.geoms:
            parts.extend(polygon_parts(child))
        return parts
    return []


def polygon_part_count(geom) -> int:
    return len(polygon_parts(geom))


def add_parent_keys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["_parent_key"] = out["oachargeid"].map(clean_key)
    out["_variant_key"] = out["oachargeid_sub"].map(clean_key)
    out["_is_child"] = out["_variant_key"].ne(out["_parent_key"])
    return out


def raw_connector_mask(gdf: gpd.GeoDataFrame, mode: str) -> pd.Series:
    theme = gdf["Theme"].fillna("").astype(str).str.lower()
    no_water_rail = ~(theme.str.contains("water", regex=False) | theme.str.contains("rail", regex=False))
    land_building = theme.str.contains("building", regex=False) | theme.str.contains("land", regex=False)
    if mode == "land_building":
        return no_water_rail & land_building
    if mode == "non_water_rail":
        return no_water_rail
    raise ValueError(f"Unknown connector mode: {mode}")


def bbox_with_pad(geom, pad: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return (float(minx - pad), float(miny - pad), float(maxx + pad), float(maxy + pad))


def shortest_connector_path(start_geom, target_geom, candidates: gpd.GeoDataFrame, current_union):
    if start_geom.intersects(target_geom):
        return []
    if candidates.empty:
        return None

    geoms = [start_geom, target_geom] + list(candidates.geometry)
    graph_gdf = gpd.GeoDataFrame({"node": range(len(geoms)), "geometry": geoms}, geometry="geometry", crs=TARGET_CRS)
    sindex = graph_gdf.sindex
    adjacency = [set() for _ in geoms]

    for idx, geom in enumerate(geoms):
        for other_idx in sindex.query(geom, predicate="intersects"):
            other_idx = int(other_idx)
            if other_idx <= idx:
                continue
            adjacency[idx].add(other_idx)
            adjacency[other_idx].add(idx)

    costs = [0.0] * len(geoms)
    for idx, geom in enumerate(geoms[2:], start=2):
        costs[idx] = max(float(geom.difference(current_union).area), 0.0)

    distances = [float("inf")] * len(geoms)
    previous: list[Optional[int]] = [None] * len(geoms)
    distances[0] = 0.0
    queue: list[tuple[float, int]] = [(0.0, 0)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == 1:
            path: list[int] = []
            cursor: Optional[int] = node
            while cursor is not None:
                path.append(cursor)
                cursor = previous[cursor]
            path.reverse()
            return [item - 2 for item in path if item >= 2]
        for neighbor in adjacency[node]:
            step_cost = costs[neighbor]
            new_distance = distance + step_cost
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    return None


def read_connector_candidates(
    raw_wfs_path: str,
    raw_wfs_layer: Optional[str],
    current_union,
    island,
    *,
    bbox_pad: float,
    hull_buffer: float,
    mode: str,
) -> gpd.GeoDataFrame:
    local_union = unary_union([current_union, island])
    raw = load_layer(raw_wfs_path, raw_wfs_layer, bbox=bbox_with_pad(local_union, bbox_pad))
    if raw.empty:
        return raw
    raw = raw[raw_connector_mask(raw, mode)].copy()
    if raw.empty:
        return raw
    hull = local_union.convex_hull.buffer(float(hull_buffer))
    raw = raw[raw.geometry.intersects(hull)].copy().reset_index(drop=True)
    raw["__connector_area__"] = raw.geometry.area.astype(float)
    return raw


def try_connect_island(
    raw_wfs_path: str,
    raw_wfs_layer: Optional[str],
    current_union,
    island,
    original_base,
    args: argparse.Namespace,
) -> tuple[bool, list[dict], float, str]:
    for mode in ("land_building", "non_water_rail"):
        candidates = read_connector_candidates(
            raw_wfs_path,
            raw_wfs_layer,
            current_union,
            island,
            bbox_pad=float(args.bbox_pad),
            hull_buffer=float(args.hull_buffer),
            mode=mode,
        )
        path = shortest_connector_path(current_union, island, candidates, current_union)
        if path is None:
            continue

        selected = candidates.iloc[path].copy() if path else candidates.iloc[0:0].copy()
        connector_geoms = list(selected.geometry)
        final = unary_union([current_union, island] + connector_geoms)
        if polygon_part_count(final) != 1:
            continue

        connector_area = sum(max(float(geom.difference(current_union).area), 0.0) for geom in connector_geoms)
        max_connector_area = min(
            float(args.max_connector_area),
            max(float(island.area) * float(args.max_connector_island_ratio), float(original_base.area) * float(args.max_connector_base_ratio)),
        )
        if connector_area > max_connector_area + EPS:
            continue

        connector_rows: list[dict] = []
        for connector_index, row in selected.reset_index(drop=True).iterrows():
            theme = row.get("Theme", "")
            connector_rows.append(
                {
                    "connector_index": int(connector_index),
                    "connector_mode": mode,
                    "connector_theme": theme,
                    "connector_area_m2": float(row.geometry.area),
                    "connector_added_area_m2": max(float(row.geometry.difference(current_union).area), 0.0),
                    "geometry": row.geometry,
                }
            )
        return True, connector_rows, float(connector_area), mode
    return False, [], 0.0, ""


def is_droppable_island(island, original_base, args: argparse.Namespace) -> bool:
    area = float(island.area)
    return area <= float(args.drop_island_area) or (
        area <= float(original_base.area) * float(args.drop_island_area_ratio)
        and area <= float(args.drop_island_area) * 4.0
    )


def empty_polygon_layer(columns: Iterable[str]) -> gpd.GeoDataFrame:
    data = {column: pd.Series(dtype="object") for column in columns}
    data["geometry"] = gpd.GeoSeries([], crs=TARGET_CRS)
    return gpd.GeoDataFrame(data, geometry="geometry", crs=TARGET_CRS)


def aggregate_parent(parent_key: str, point_group: gpd.GeoDataFrame, cap_group: gpd.GeoDataFrame, args: argparse.Namespace):
    child_count = int(point_group["_is_child"].sum())
    child_variants = set(point_group.loc[point_group["_is_child"], "_variant_key"].tolist())
    matched_child_variants = set(cap_group.loc[cap_group["_is_child"], "_variant_key"].tolist()) if not cap_group.empty else set()
    missing_child_count = int(len(child_variants - matched_child_variants))

    use_group = cap_group
    source_basis = "all_rows"
    if child_count > 0:
        child_cap = cap_group[cap_group["_is_child"]].copy() if not cap_group.empty else cap_group
        if not child_cap.empty:
            use_group = child_cap
            source_basis = "child_rows"
        elif not cap_group.empty:
            use_group = cap_group
            source_basis = "parent_fallback"

    common = {
        "oachargeid": parent_key,
        "point_rows": int(len(point_group)),
        "child_rows": child_count,
        "matched_polygon_rows": int(len(cap_group)),
        "used_polygon_rows": int(len(use_group)),
        "missing_child_count": missing_child_count,
        "address_clarity": ",".join(sorted(point_group["address_clarity"].fillna("").astype(str).unique())),
        "source_basis": source_basis,
    }

    if use_group.empty:
        return None, {
            **common,
            "aggregation_status": "no_polygon",
            "aggregation_method": "no_polygon",
            "unresolved_reason": "no_matched_wfs_polygon",
            "geometry": point_group.geometry.iloc[0],
        }, [], []

    base = unary_union(list(use_group.geometry))
    original_parts = sorted(polygon_parts(base), key=lambda geom: float(geom.area), reverse=True)
    if not original_parts:
        return None, {
            **common,
            "aggregation_status": "no_polygon",
            "aggregation_method": "empty_polygon",
            "unresolved_reason": "empty_matched_polygon",
            "geometry": point_group.geometry.iloc[0],
        }, [], []

    qa_flags: list[str] = []
    if missing_child_count:
        qa_flags.append("missing_child_polygons")
    if source_basis == "parent_fallback":
        qa_flags.append("parent_fallback")

    if len(original_parts) == 1:
        method = "child_union_single" if source_basis == "child_rows" else "single_polygon"
        return {
            **common,
            "aggregation_status": "accepted",
            "aggregation_method": method,
            "base_parts": 1,
            "final_parts": 1,
            "dropped_island_count": 0,
            "dropped_island_area_m2": 0.0,
            "connector_count": 0,
            "connector_area_m2": 0.0,
            "base_area_m2": float(base.area),
            "final_area_m2": float(base.area),
            "qa_flags": ",".join(qa_flags),
            "geometry": base,
        }, None, [], []

    kept_geoms: list[object] = [original_parts[0]]
    remaining = list(original_parts[1:])
    dropped_rows: list[dict] = []
    connector_rows: list[dict] = []
    unresolved_islands: list[object] = []
    connector_area_total = 0.0

    while remaining:
        current_union = unary_union(kept_geoms)
        nearest_pos = min(range(len(remaining)), key=lambda idx: float(current_union.distance(remaining[idx])))
        island = remaining.pop(nearest_pos)
        distance = float(current_union.distance(island))

        if distance > float(args.connect_distance):
            if is_droppable_island(island, base, args):
                dropped_rows.append(
                    {
                        **common,
                        "drop_reason": "far_small_island",
                        "island_area_m2": float(island.area),
                        "island_distance_m": distance,
                        "geometry": island,
                    }
                )
                continue
            unresolved_islands.append(island)
            continue

        connected, new_connectors, connector_area, connector_mode = try_connect_island(
            args.raw_wfs_gpkg,
            args.raw_wfs_layer,
            current_union,
            island,
            base,
            args,
        )
        if connected:
            kept_geoms.append(island)
            kept_geoms.extend(row["geometry"] for row in new_connectors)
            connector_area_total += connector_area
            for row in new_connectors:
                connector_rows.append({**common, **row})
            if connector_mode == "non_water_rail":
                qa_flags.append("road_path_connector")
            elif connector_mode == "land_building":
                qa_flags.append("land_building_connector")
            continue

        if is_droppable_island(island, base, args):
            dropped_rows.append(
                {
                    **common,
                    "drop_reason": "unconnected_small_island",
                    "island_area_m2": float(island.area),
                    "island_distance_m": distance,
                    "geometry": island,
                }
            )
            continue
        unresolved_islands.append(island)

    if dropped_rows:
        qa_flags.append("dropped_small_islands")
    if unresolved_islands:
        unresolved_union = unary_union(unresolved_islands)
        return None, {
            **common,
            "aggregation_status": "unresolved",
            "aggregation_method": "multipart_unresolved",
            "unresolved_reason": "large_or_unconnected_islands",
            "base_parts": int(len(original_parts)),
            "unresolved_island_count": int(len(unresolved_islands)),
            "unresolved_island_area_m2": float(unresolved_union.area),
            "dropped_island_count": int(len(dropped_rows)),
            "dropped_island_area_m2": float(sum(row["island_area_m2"] for row in dropped_rows)),
            "connector_count": int(len(connector_rows)),
            "connector_area_m2": float(connector_area_total),
            "base_area_m2": float(base.area),
            "qa_flags": ",".join(sorted(set(qa_flags))),
            "geometry": point_group.geometry.iloc[0],
        }, dropped_rows, connector_rows

    final = unary_union(kept_geoms)
    final_parts = polygon_part_count(final)
    if final_parts != 1:
        return None, {
            **common,
            "aggregation_status": "unresolved",
            "aggregation_method": "multipart_after_connect",
            "unresolved_reason": "final_geometry_still_multipart",
            "base_parts": int(len(original_parts)),
            "final_parts": int(final_parts),
            "dropped_island_count": int(len(dropped_rows)),
            "dropped_island_area_m2": float(sum(row["island_area_m2"] for row in dropped_rows)),
            "connector_count": int(len(connector_rows)),
            "connector_area_m2": float(connector_area_total),
            "base_area_m2": float(base.area),
            "final_area_m2": float(final.area),
            "qa_flags": ",".join(sorted(set(qa_flags))),
            "geometry": point_group.geometry.iloc[0],
        }, dropped_rows, connector_rows

    method = "connect_prune_union"
    if connector_rows and not dropped_rows:
        method = "wfs_connector_union"
    elif dropped_rows and not connector_rows:
        method = "pruned_small_island_union"
    return {
        **common,
        "aggregation_status": "accepted",
        "aggregation_method": method,
        "base_parts": int(len(original_parts)),
        "final_parts": int(final_parts),
        "dropped_island_count": int(len(dropped_rows)),
        "dropped_island_area_m2": float(sum(row["island_area_m2"] for row in dropped_rows)),
        "connector_count": int(len(connector_rows)),
        "connector_area_m2": float(connector_area_total),
        "base_area_m2": float(base.area),
        "final_area_m2": float(final.area),
        "qa_flags": ",".join(sorted(set(qa_flags))),
        "geometry": final,
    }, None, dropped_rows, connector_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate point-captured polygons back to one row per oachargeid.")
    parser.add_argument("--point-gpkg", required=True)
    parser.add_argument("--point-layer")
    parser.add_argument("--capture-gpkg", required=True)
    parser.add_argument("--capture-layer", default="capture_polygons")
    parser.add_argument("--raw-wfs-gpkg", required=True)
    parser.add_argument("--raw-wfs-layer")
    parser.add_argument("--output-gpkg", required=True)
    parser.add_argument("--output-layer", default="parent_polygons")
    parser.add_argument("--unresolved-layer", default="unresolved_parents")
    parser.add_argument("--dropped-layer", default="dropped_islands")
    parser.add_argument("--connector-layer", default="wfs_connectors")
    parser.add_argument("--diagnostics-csv")
    parser.add_argument("--connect-distance", type=float, default=50.0)
    parser.add_argument("--drop-island-area", type=float, default=500.0)
    parser.add_argument("--drop-island-area-ratio", type=float, default=0.08)
    parser.add_argument("--bbox-pad", type=float, default=80.0)
    parser.add_argument("--hull-buffer", type=float, default=25.0)
    parser.add_argument("--max-connector-island-ratio", type=float, default=4.0)
    parser.add_argument("--max-connector-base-ratio", type=float, default=0.75)
    parser.add_argument("--max-connector-area", type=float, default=10000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    point_layer = choose_layer(args.point_gpkg, args.point_layer)
    capture_layer = choose_layer(args.capture_gpkg, args.capture_layer)
    raw_layer = choose_layer(args.raw_wfs_gpkg, args.raw_wfs_layer)
    args.raw_wfs_layer = raw_layer

    print(f"[INFO] Reading points: {args.point_gpkg} (layer={point_layer})")
    points = add_parent_keys(load_layer(args.point_gpkg, point_layer))
    print(f"[INFO] points={len(points)} parents={points['_parent_key'].nunique()}")

    print(f"[INFO] Reading capture polygons: {args.capture_gpkg} (layer={capture_layer})")
    capture = add_parent_keys(load_layer(args.capture_gpkg, capture_layer))
    print(f"[INFO] capture polygons={len(capture)}")

    cap_groups = {key: group.copy() for key, group in capture.groupby("_parent_key", sort=False)}
    accepted_rows: list[dict] = []
    unresolved_rows: list[dict] = []
    dropped_rows: list[dict] = []
    connector_rows: list[dict] = []

    for parent_key, point_group in points.groupby("_parent_key", sort=False):
        cap_group = cap_groups.get(parent_key, capture.iloc[0:0]).copy()
        accepted, unresolved, dropped, connectors = aggregate_parent(parent_key, point_group.copy(), cap_group, args)
        if accepted is not None:
            accepted_rows.append(accepted)
        if unresolved is not None:
            unresolved_rows.append(unresolved)
        dropped_rows.extend(dropped)
        connector_rows.extend(connectors)

    parent_gdf = gpd.GeoDataFrame(accepted_rows, geometry="geometry", crs=TARGET_CRS)
    unresolved_gdf = gpd.GeoDataFrame(unresolved_rows, geometry="geometry", crs=TARGET_CRS)
    dropped_gdf = (
        gpd.GeoDataFrame(dropped_rows, geometry="geometry", crs=TARGET_CRS)
        if dropped_rows
        else empty_polygon_layer(["oachargeid", "drop_reason", "island_area_m2", "island_distance_m"])
    )
    connector_gdf = (
        gpd.GeoDataFrame(connector_rows, geometry="geometry", crs=TARGET_CRS)
        if connector_rows
        else empty_polygon_layer(["oachargeid", "connector_mode", "connector_theme", "connector_area_m2"])
    )

    output_path = Path(args.output_gpkg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    parent_gdf.to_file(output_path, layer=args.output_layer, driver="GPKG", engine="pyogrio")
    unresolved_gdf.to_file(output_path, layer=args.unresolved_layer, driver="GPKG", engine="pyogrio")
    dropped_gdf.to_file(output_path, layer=args.dropped_layer, driver="GPKG", engine="pyogrio")
    connector_gdf.to_file(output_path, layer=args.connector_layer, driver="GPKG", engine="pyogrio")

    diagnostics = pd.concat(
        [
            parent_gdf.drop(columns="geometry").assign(has_output_polygon=True),
            unresolved_gdf.drop(columns="geometry").assign(has_output_polygon=False),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics_csv = args.diagnostics_csv or str(output_path.with_suffix(".csv"))
    diagnostics.to_csv(diagnostics_csv, index=False)

    print(f"[DONE] Wrote {output_path}")
    print(f"[INFO] parent polygons={len(parent_gdf)} unresolved={len(unresolved_gdf)}")
    print(f"[INFO] dropped islands={len(dropped_gdf)} connectors={len(connector_gdf)}")
    if not parent_gdf.empty:
        print("[INFO] aggregation methods:")
        print(parent_gdf["aggregation_method"].value_counts(dropna=False).to_string())
        flags = parent_gdf["qa_flags"].fillna("").astype(str)
        print(f"[INFO] accepted with QA flags={int(flags.ne('').sum())}")
    if not unresolved_gdf.empty:
        print("[INFO] unresolved reasons:")
        print(unresolved_gdf["unresolved_reason"].value_counts(dropna=False).to_string())
    print(f"[INFO] diagnostics_csv={diagnostics_csv}")


if __name__ == "__main__":
    main()

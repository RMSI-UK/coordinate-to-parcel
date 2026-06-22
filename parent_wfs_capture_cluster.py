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


def choose_layer(path: str, layer: str | None) -> str:
    if layer:
        return layer
    layers = pyogrio.list_layers(path)
    if len(layers) == 0:
        raise ValueError(f"No layers found in {path}")
    return str(layers[0][0])


def to_target_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(TARGET_CRS)
    if str(gdf.crs).upper() != TARGET_CRS:
        return gdf.to_crs(TARGET_CRS)
    return gdf


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def parse_terms(value: str) -> tuple[str, ...]:
    return tuple(term.strip().lower() for term in value.split(",") if term.strip())


def csv_unique(values: Iterable[object]) -> str:
    items: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        for item in str(value).split(","):
            item = item.strip()
            if item:
                items.add(item)
    return ",".join(sorted(items))


def read_points(path: str, layer: str | None) -> gpd.GeoDataFrame:
    layer_name = choose_layer(path, layer)
    points = gpd.read_file(path, layer=layer_name, engine="pyogrio")
    points = to_target_crs(points)
    points = points[points.geometry.notna()].copy()
    points = points[~points.geometry.is_empty].copy()
    points = points.reset_index(drop=True)
    points["point_row_id"] = points.index.astype("int64")
    if "oachargeid" not in points.columns:
        raise ValueError("Point layer must contain oachargeid")
    if "oachargeid_sub" not in points.columns:
        points["oachargeid_sub"] = points["oachargeid"]
    points["_parent_key"] = points["oachargeid"].map(clean_key)
    points["_sub_key"] = points["oachargeid_sub"].map(clean_key)
    points["_is_child"] = points["_sub_key"].ne(points["_parent_key"])
    points["_similarity"] = pd.to_numeric(points.get("api_address_similarity", 0), errors="coerce").fillna(0.0)
    points["_parity"] = points.get("address_range_parity", "").fillna("").astype(str).str.strip().str.lower()
    return points


def read_polygons(path: str, layer: str | None, *, source: str) -> gpd.GeoDataFrame:
    layer_name = choose_layer(path, layer)
    preferred_columns = [
        "GmlID",
        "GML_ID",
        "TOID",
        "Theme",
        "DescriptiveGroup",
        "DescriptiveTerm",
        "source_fid",
        "CalculatedAreaValue",
        "Shape_Area",
    ]
    try:
        gdf = gpd.read_file(
            path,
            layer=layer_name,
            columns=preferred_columns,
            engine="pyogrio",
            fid_as_index=True,
        )
    except Exception:
        gdf = gpd.read_file(path, layer=layer_name, engine="pyogrio", fid_as_index=True)
    gdf = to_target_crs(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf[f"{source}_fid"] = gdf.index.astype(str)
    gdf = gdf.reset_index(drop=True)
    gdf[f"{source}_row_id"] = gdf.index.astype("int64")
    gdf[f"{source}_area_m2"] = gdf.geometry.area.astype(float)
    return gdf


def polygon_parts(geom) -> list[object]:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return []
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts: list[object] = []
        for child in geom.geoms:
            parts.extend(polygon_parts(child))
        return parts
    return []


def part_count(geom) -> int:
    return len(polygon_parts(geom))


def intersect_points_with_merge(points: gpd.GeoDataFrame, merge: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        points[["point_row_id", "geometry"]],
        merge[["merge_row_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return gpd.GeoDataFrame(columns=["point_row_id", "merge_row_id", "geometry"], geometry="geometry", crs=TARGET_CRS)

    point_rows = points.set_index("point_row_id", drop=False).loc[joined["point_row_id"].astype(int)].copy()
    point_rows.index = joined.index
    merge_rows = merge.set_index("merge_row_id", drop=False).loc[joined["merge_row_id"].astype(int)].copy()
    merge_rows.index = joined.index

    out = pd.DataFrame(
        {
            "point_row_id": point_rows["point_row_id"].astype(int).values,
            "oachargeid": point_rows["_parent_key"].astype(str).values,
            "oachargeid_sub": point_rows["_sub_key"].astype(str).values,
            "is_child": point_rows["_is_child"].astype(bool).values,
            "similarity": point_rows["_similarity"].astype(float).values,
            "parity": point_rows["_parity"].astype(str).values,
            "merge_row_id": merge_rows["merge_row_id"].astype(int).values,
            "merge_fid": merge_rows["merge_fid"].astype(str).values,
            "merge_source_fid": merge_rows["source_fid"].fillna("").astype(str).values if "source_fid" in merge_rows.columns else "",
            "merge_theme": merge_rows["Theme"].fillna("").astype(str).values if "Theme" in merge_rows.columns else "",
            "merge_area_m2": merge_rows["merge_area_m2"].astype(float).values,
        }
    )
    out["geometry"] = merge_rows.geometry.values
    return gpd.GeoDataFrame(out, geometry="geometry", crs=merge.crs)


def component_indices(gdf: gpd.GeoDataFrame, *, gap: float = 0.0) -> list[list[int]]:
    if gdf.empty:
        return []
    gap = max(float(gap), 0.0)
    neighbors: dict[int, list[int]] = {int(idx): [] for idx in gdf.index}
    sindex = gdf.sindex
    for pos, idx in enumerate(gdf.index):
        idx = int(idx)
        geom = gdf.geometry.iloc[pos]
        query_geom = geom.buffer(gap) if gap > 0 else geom
        for other_pos in sindex.query(query_geom, predicate="intersects"):
            other_idx = int(gdf.index[int(other_pos)])
            if other_idx == idx:
                continue
            if gap > 0 and float(geom.distance(gdf.at[other_idx, "geometry"])) > gap:
                continue
            neighbors[idx].append(other_idx)

    components: list[list[int]] = []
    seen: set[int] = set()
    for idx in gdf.index:
        idx = int(idx)
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    return components


def make_cluster_record(parent_key: str, cluster_id: int, rows: gpd.GeoDataFrame, bridge_rows: gpd.GeoDataFrame | None = None) -> dict:
    bridge_rows = bridge_rows if bridge_rows is not None else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)
    merge_unique = rows.drop_duplicates("merge_row_id")
    point_unique = rows.drop_duplicates("point_row_id")
    geom = unary_union(list(merge_unique.geometry) + list(bridge_rows.geometry))
    return {
        "oachargeid": str(parent_key),
        "cluster_id": int(cluster_id),
        "point_rows": int(point_unique["point_row_id"].nunique()),
        "child_rows": int(point_unique["is_child"].astype(bool).sum()),
        "merge_feature_count": int(len(merge_unique)),
        "merge_fids": csv_unique(merge_unique["merge_fid"]),
        "merge_source_fids": csv_unique(merge_unique["merge_source_fid"]) if "merge_source_fid" in merge_unique.columns else "",
        "merge_themes": csv_unique(merge_unique["merge_theme"]) if "merge_theme" in merge_unique.columns else "",
        "mean_similarity": float(point_unique["similarity"].mean()) if not point_unique.empty else 0.0,
        "same_parity_rows": int(point_unique["parity"].eq("same").sum()) if "parity" in point_unique.columns else 0,
        "bridge_feature_count": int(len(bridge_rows)),
        "bridge_fids": csv_unique(bridge_rows["raw_fid"]) if "raw_fid" in bridge_rows.columns else "",
        "bridge_themes": csv_unique(bridge_rows["Theme"]) if "Theme" in bridge_rows.columns else "",
        "bridge_added_area_m2": float(bridge_rows["bridge_added_area_m2"].sum()) if "bridge_added_area_m2" in bridge_rows.columns else 0.0,
        "area_m2": float(geom.area),
        "part_count": int(part_count(geom)),
        "geometry": geom,
    }


def initial_clusters(intersections: gpd.GeoDataFrame) -> list[dict]:
    records: list[dict] = []
    for parent_key, parent_rows in intersections.groupby("oachargeid", sort=False):
        merge_rows = parent_rows.drop_duplicates("merge_row_id").reset_index(drop=True)
        components = component_indices(merge_rows, gap=0.0)
        for cluster_id, component in enumerate(components, start=1):
            merge_ids = set(merge_rows.loc[component, "merge_row_id"].astype(int).tolist())
            component_rows = parent_rows[parent_rows["merge_row_id"].astype(int).isin(merge_ids)].copy()
            records.append(make_cluster_record(str(parent_key), cluster_id, component_rows))
    return records


def bridge_candidate_mask(raw: gpd.GeoDataFrame, exclude_terms: tuple[str, ...]) -> pd.Series:
    if not exclude_terms:
        return pd.Series(True, index=raw.index)
    text = pd.Series("", index=raw.index)
    for column in ("Theme", "DescriptiveGroup", "DescriptiveTerm"):
        if column in raw.columns:
            text = text + " " + raw[column].fillna("").astype(str).str.lower()
    mask = pd.Series(True, index=raw.index)
    for term in exclude_terms:
        mask = mask & ~text.str.contains(term, regex=False)
    return mask


def shortest_bridge_path(start_geom, target_geom, candidates: gpd.GeoDataFrame, current_union):
    if start_geom.intersects(target_geom):
        return []
    if candidates.empty:
        return None

    geoms = [start_geom, target_geom] + list(candidates.geometry)
    graph = gpd.GeoDataFrame({"node": range(len(geoms)), "geometry": geoms}, geometry="geometry", crs=TARGET_CRS)
    sindex = graph.sindex
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
            new_distance = distance + costs[neighbor]
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    return None


def candidate_raw_for_pair(raw: gpd.GeoDataFrame, geom_a, geom_b, args: argparse.Namespace) -> gpd.GeoDataFrame:
    base = unary_union([geom_a, geom_b])
    hull = base.convex_hull.buffer(float(args.bridge_hull_buffer))
    idx = list(raw.sindex.query(hull, predicate="intersects"))
    if not idx:
        return raw.iloc[0:0].copy()
    candidates = raw.iloc[idx].copy()
    candidates = candidates[candidates.geometry.intersects(hull)].copy()
    candidates = candidates[candidates["raw_area_m2"].astype(float).le(float(args.max_bridge_feature_area))].copy()
    if candidates.empty:
        return candidates
    candidates["bridge_added_area_m2"] = candidates.geometry.apply(lambda geom: max(float(geom.difference(base).area), 0.0))
    candidates = candidates[candidates["bridge_added_area_m2"].gt(float(args.min_bridge_added_area))].copy()
    return candidates.sort_values(["bridge_added_area_m2", "raw_area_m2"])


def bridge_close_clusters(cluster_records: list[dict], raw: gpd.GeoDataFrame, args: argparse.Namespace) -> gpd.GeoDataFrame:
    if not cluster_records:
        return empty_output()

    output_records: list[dict] = []
    max_gap = float(args.bridge_gap)
    max_total_bridge_area = float(args.max_bridge_total_area)

    by_parent: dict[str, list[dict]] = {}
    for record in cluster_records:
        by_parent.setdefault(str(record["oachargeid"]), []).append(record)

    for parent_key, records in by_parent.items():
        records = [dict(record) for record in records]
        changed = True
        while changed and len(records) > 1:
            changed = False
            best = None
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    left = records[i]
                    right = records[j]
                    distance = float(left["geometry"].distance(right["geometry"]))
                    if distance > max_gap:
                        continue
                    candidates = candidate_raw_for_pair(raw, left["geometry"], right["geometry"], args)
                    if candidates.empty:
                        continue
                    current_union = unary_union([left["geometry"], right["geometry"]])
                    path = shortest_bridge_path(left["geometry"], right["geometry"], candidates, current_union)
                    if path is None:
                        continue
                    selected = candidates.iloc[path].copy() if path else candidates.iloc[0:0].copy()
                    bridge_area = float(selected["bridge_added_area_m2"].sum()) if not selected.empty else 0.0
                    if bridge_area > max_total_bridge_area:
                        continue
                    merged_geom = unary_union([left["geometry"], right["geometry"]] + list(selected.geometry))
                    if part_count(merged_geom) != 1:
                        continue
                    score = (bridge_area, distance, int(len(selected)))
                    if best is None or score < best[0]:
                        best = (score, i, j, selected)

            if best is None:
                break

            _, i, j, bridge_rows = best
            left = records[i]
            right = records[j]
            merged_geom = unary_union([left["geometry"], right["geometry"]] + list(bridge_rows.geometry))
            merged = {
                "oachargeid": parent_key,
                "cluster_id": -1,
                "point_rows": int(left["point_rows"]) + int(right["point_rows"]),
                "child_rows": int(left["child_rows"]) + int(right["child_rows"]),
                "merge_feature_count": int(left["merge_feature_count"]) + int(right["merge_feature_count"]),
                "merge_fids": csv_unique([left["merge_fids"], right["merge_fids"]]),
                "merge_source_fids": csv_unique([left["merge_source_fids"], right["merge_source_fids"]]),
                "merge_themes": csv_unique([left["merge_themes"], right["merge_themes"]]),
                "mean_similarity": (
                    float(left["mean_similarity"]) * int(left["point_rows"])
                    + float(right["mean_similarity"]) * int(right["point_rows"])
                )
                / max(int(left["point_rows"]) + int(right["point_rows"]), 1),
                "same_parity_rows": int(left["same_parity_rows"]) + int(right["same_parity_rows"]),
                "bridge_feature_count": int(left["bridge_feature_count"]) + int(right["bridge_feature_count"]) + int(len(bridge_rows)),
                "bridge_fids": csv_unique([left["bridge_fids"], right["bridge_fids"], csv_unique(bridge_rows["raw_fid"]) if "raw_fid" in bridge_rows.columns else ""]),
                "bridge_themes": csv_unique([left["bridge_themes"], right["bridge_themes"], csv_unique(bridge_rows["Theme"]) if "Theme" in bridge_rows.columns else ""]),
                "bridge_added_area_m2": float(left["bridge_added_area_m2"])
                + float(right["bridge_added_area_m2"])
                + (float(bridge_rows["bridge_added_area_m2"].sum()) if "bridge_added_area_m2" in bridge_rows.columns else 0.0),
                "area_m2": float(merged_geom.area),
                "part_count": int(part_count(merged_geom)),
                "geometry": merged_geom,
            }
            records = [record for pos, record in enumerate(records) if pos not in {i, j}]
            records.append(merged)
            changed = True

        for cluster_id, record in enumerate(records, start=1):
            record["cluster_id"] = int(cluster_id)
            record["area_m2"] = float(record["geometry"].area)
            record["part_count"] = int(part_count(record["geometry"]))
            output_records.append(record)

    return gpd.GeoDataFrame(output_records, geometry="geometry", crs=TARGET_CRS)


def empty_output() -> gpd.GeoDataFrame:
    columns = [
        "oachargeid",
        "cluster_id",
        "point_rows",
        "child_rows",
        "merge_feature_count",
        "merge_fids",
        "merge_source_fids",
        "merge_themes",
        "mean_similarity",
        "same_parity_rows",
        "bridge_feature_count",
        "bridge_fids",
        "bridge_themes",
        "bridge_added_area_m2",
        "area_m2",
        "part_count",
    ]
    data = {column: pd.Series(dtype="object") for column in columns}
    data["geometry"] = gpd.GeoSeries([], crs=TARGET_CRS)
    return gpd.GeoDataFrame(data, geometry="geometry", crs=TARGET_CRS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture points to WFS merge, merge touching islands, then bridge very close islands with raw WFS polygons.",
    )
    parser.add_argument("--point-gpkg", required=True)
    parser.add_argument("--point-layer")
    parser.add_argument("--wfs-merge-gpkg", required=True)
    parser.add_argument("--wfs-merge-layer")
    parser.add_argument("--wfs-gpkg", required=True)
    parser.add_argument("--wfs-layer")
    parser.add_argument("--output-gpkg", required=True)
    parser.add_argument("--output-layer", default="parent_wfs_bridge_clusters")
    parser.add_argument("--bridge-gap", type=float, default=5.0)
    parser.add_argument("--bridge-hull-buffer", type=float, default=8.0)
    parser.add_argument("--bridge-exclude-terms", default="water,rail")
    parser.add_argument("--max-bridge-feature-area", type=float, default=1200.0)
    parser.add_argument("--max-bridge-total-area", type=float, default=2000.0)
    parser.add_argument("--min-bridge-added-area", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(f"[INFO] Reading points: {args.point_gpkg}")
    points = read_points(args.point_gpkg, args.point_layer)
    print(f"[INFO] points={len(points)}")

    print(f"[INFO] Reading WFS merge: {args.wfs_merge_gpkg}")
    merge = read_polygons(args.wfs_merge_gpkg, args.wfs_merge_layer, source="merge")
    print(f"[INFO] wfs_merge_features={len(merge)}")

    print(f"[INFO] Intersecting points with WFS merge")
    intersections = intersect_points_with_merge(points, merge)
    print(
        "[INFO] intersections="
        f"{len(intersections)} matched_points={intersections['point_row_id'].nunique() if not intersections.empty else 0} "
        f"parents={intersections['oachargeid'].nunique() if not intersections.empty else 0}"
    )

    print("[INFO] Merging touching WFS merge polygons per oachargeid")
    cluster_records = initial_clusters(intersections)
    print(f"[INFO] initial_clusters={len(cluster_records)}")

    print(f"[INFO] Reading raw WFS bridge polygons: {args.wfs_gpkg}")
    raw = read_polygons(args.wfs_gpkg, args.wfs_layer, source="raw")
    exclude_terms = parse_terms(args.bridge_exclude_terms)
    raw = raw[bridge_candidate_mask(raw, exclude_terms)].copy().reset_index(drop=True)
    print(f"[INFO] raw_bridge_candidates={len(raw)} exclude_terms={args.bridge_exclude_terms!r}")

    print(f"[INFO] Bridging clusters within {float(args.bridge_gap):.2f}m")
    output = bridge_close_clusters(cluster_records, raw, args)
    print(
        "[INFO] final_clusters="
        f"{len(output)} parents={output['oachargeid'].nunique() if not output.empty else 0} "
        f"bridged_clusters={int(output['bridge_feature_count'].astype(int).gt(0).sum()) if not output.empty else 0}"
    )

    output_path = Path(args.output_gpkg)
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(output_path, layer=args.output_layer, driver="GPKG", engine="pyogrio")
    print(f"[DONE] Wrote {output_path} layer={args.output_layer}")


if __name__ == "__main__":
    main()

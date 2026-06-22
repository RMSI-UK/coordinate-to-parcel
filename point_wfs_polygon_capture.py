#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.ops import unary_union


TARGET_CRS = "EPSG:27700"
DEFAULT_TERMS = ("building", "land")


def choose_layer(path: str, layer: Optional[str]) -> str:
    if layer:
        return layer
    layers = pyogrio.list_layers(path)
    if len(layers) == 0:
        raise ValueError(f"No layers found in {path}")
    return str(layers[0][0])


def load_layer(
    path: str,
    layer: Optional[str] = None,
    columns: Optional[list[str]] = None,
) -> gpd.GeoDataFrame:
    layer_name = choose_layer(path, layer)
    kwargs = {"layer": layer_name, "engine": "pyogrio"}
    if columns is not None:
        kwargs["columns"] = columns
    gdf = gpd.read_file(path, **kwargs)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def parse_terms(value: str) -> tuple[str, ...]:
    terms = tuple(term.strip().lower() for term in value.split(",") if term.strip())
    if not terms:
        raise ValueError("--theme-include must include at least one term")
    return terms


def filter_theme(gdf: gpd.GeoDataFrame, theme_field: str, terms: Iterable[str]) -> gpd.GeoDataFrame:
    if theme_field not in gdf.columns:
        raise ValueError(f"Theme field {theme_field!r} not found")
    theme = gdf[theme_field].fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=gdf.index)
    for term in terms:
        mask = mask | theme.str.contains(term, regex=False)
    return gdf[mask].copy()


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def add_parent_keys(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.copy()
    if "oachargeid" not in out.columns:
        raise ValueError("Point layer must contain oachargeid")
    if "oachargeid_sub" not in out.columns:
        out["oachargeid_sub"] = out["oachargeid"]
    out["_parent_key"] = out["oachargeid"].map(clean_key)
    out["_variant_key"] = out["oachargeid_sub"].map(clean_key)
    out["_is_child"] = out["_variant_key"].ne(out["_parent_key"])
    out["_similarity"] = pd.to_numeric(out.get("api_address_similarity", 0), errors="coerce").fillna(0.0)
    out["_parity"] = out.get("address_range_parity", "").fillna("").astype(str).str.strip().str.lower()
    return out


def expanded_parent_mask(points: gpd.GeoDataFrame) -> pd.Series:
    child_counts = points.groupby("_parent_key")["_is_child"].sum()
    expanded_ids = set(child_counts[child_counts > 0].index.astype(str))
    return points["_parent_key"].isin(expanded_ids)


def first_existing(gdf: gpd.GeoDataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in gdf.columns:
            return name
    return None


def prepare_polygons(
    path: str,
    layer: Optional[str],
    theme_field: str,
    terms: Optional[tuple[str, ...]],
) -> gpd.GeoDataFrame:
    preferred_columns = [
        theme_field,
        "GmlID",
        "GML_ID",
        "TOID",
        "source_fid",
        "CalculatedAreaValue",
        "Shape_Area",
    ]
    try:
        polygons = load_layer(path, layer, columns=preferred_columns)
    except Exception:
        polygons = load_layer(path, layer)
    if terms is not None:
        polygons = filter_theme(polygons, theme_field, terms)
    polygons["__poly_area__"] = polygons.geometry.area.astype(float)
    return polygons


def empty_chosen() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "capture_src_id": pd.Series(dtype="int64"),
            "index_right": pd.Series(dtype="int64"),
            "capture_source": pd.Series(dtype="object"),
            "geometry": gpd.GeoSeries([], crs=TARGET_CRS),
        },
        geometry="geometry",
        crs=TARGET_CRS,
    )


def empty_capture_rows(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.iloc[0:0].copy()
    for name, dtype in {
        "capture_source": "object",
        "capture_polygon_id": "object",
        "capture_polygon_area_m2": "float64",
        "capture_polygon_toid": "object",
        "capture_polygon_gmlid": "object",
        "capture_polygon_source_fid": "object",
        "capture_polygon_theme": "object",
        "capture_polygon_calculated_area": "object",
        "capture_polygon_shape_area": "object",
    }.items():
        out[name] = pd.Series(dtype=dtype)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=points.crs)


def pick_smallest_intersections(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    source: str,
) -> gpd.GeoDataFrame:
    if points.empty or polygons.empty:
        return empty_chosen()
    joined = gpd.sjoin(
        points[["capture_src_id", "geometry"]],
        polygons[["__poly_area__", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return empty_chosen()
    right_index_col = "index_right"
    if right_index_col not in joined.columns:
        right_index_col = polygons.index.name or "index_right"
    if right_index_col not in joined.columns:
        candidates = [col for col in joined.columns if col not in {"capture_src_id", "geometry", "__poly_area__"}]
        if not candidates:
            raise ValueError("Spatial join did not return a right polygon index column")
        right_index_col = candidates[0]
    chosen = (
        joined.sort_values(["capture_src_id", "__poly_area__", right_index_col], ascending=[True, True, True])
        .drop_duplicates("capture_src_id", keep="first")
        .copy()
    )
    chosen = chosen.rename(columns={right_index_col: "index_right"})
    chosen["capture_source"] = source
    return chosen


def build_capture_rows(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    chosen: gpd.GeoDataFrame,
    source: str,
    theme_field: str,
) -> gpd.GeoDataFrame:
    if chosen.empty:
        return empty_capture_rows(points)
    poly = polygons.loc[chosen["index_right"].astype(int)].copy()
    poly.index = chosen.index
    rows = points.set_index("capture_src_id", drop=False).loc[chosen["capture_src_id"].astype(int)].copy()
    rows.index = chosen.index
    rows["capture_source"] = source
    rows["capture_polygon_id"] = chosen["index_right"].astype(str).values
    rows["capture_polygon_area_m2"] = poly.geometry.area.astype(float).values

    output_columns = [
        ("capture_polygon_toid", ("TOID", "toid")),
        ("capture_polygon_gmlid", ("GmlID", "GML_ID", "gml_id", "gmlid")),
        ("capture_polygon_source_fid", ("source_fid",)),
        ("capture_polygon_theme", (theme_field,)),
        ("capture_polygon_calculated_area", ("CalculatedAreaValue", "calculatedareavalue")),
        ("capture_polygon_shape_area", ("Shape_Area", "shape_area")),
    ]
    for output_column, candidates in output_columns:
        source_column = first_existing(poly, candidates)
        rows[output_column] = poly[source_column].values if source_column else pd.NA

    rows["geometry"] = poly.geometry.values
    return gpd.GeoDataFrame(rows.reset_index(drop=True), geometry="geometry", crs=polygons.crs)


def build_capture(
    points: gpd.GeoDataFrame,
    merged_wfs: gpd.GeoDataFrame,
    raw_wfs: gpd.GeoDataFrame,
    theme_field: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, int]]:
    points = points.reset_index(drop=True).copy()
    points["capture_src_id"] = points.index.astype("int64")

    merge_chosen = pick_smallest_intersections(points, merged_wfs, "wfs_merge")
    merge_rows = build_capture_rows(points, merged_wfs, merge_chosen, "wfs_merge", theme_field)
    matched_ids = set(merge_chosen["capture_src_id"].astype(int).tolist()) if not merge_chosen.empty else set()

    remaining = points[~points["capture_src_id"].isin(matched_ids)].copy()
    raw_chosen = pick_smallest_intersections(remaining, raw_wfs, "wfs_raw")
    raw_rows = build_capture_rows(points, raw_wfs, raw_chosen, "wfs_raw", theme_field)
    raw_ids = set(raw_chosen["capture_src_id"].astype(int).tolist()) if not raw_chosen.empty else set()

    capture = pd.concat([merge_rows, raw_rows], ignore_index=True)
    capture = gpd.GeoDataFrame(capture, geometry="geometry", crs=TARGET_CRS)
    if not capture.empty:
        capture = capture.sort_values(["capture_src_id", "capture_source"]).reset_index(drop=True)

    unmatched = points[~points["capture_src_id"].isin(matched_ids | raw_ids)].copy()
    unmatched["capture_source"] = "unmatched"
    unmatched = unmatched.sort_values("capture_src_id").reset_index(drop=True)

    counts = {
        "points": int(len(points)),
        "merge_eligible": int(len(merged_wfs)),
        "matched_by_merge": int(len(matched_ids)),
        "remaining_after_merge": int(len(remaining)),
        "raw_eligible": int(len(raw_wfs)),
        "matched_by_raw": int(len(raw_ids)),
        "capture_polygons": int(len(capture)),
        "unmatched_points": int(len(unmatched)),
    }
    return capture, unmatched, counts


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


def empty_cluster_rows() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "oachargeid": pd.Series(dtype="object"),
            "cluster_id": pd.Series(dtype="int64"),
            "capture_rows": pd.Series(dtype="int64"),
            "point_rows": pd.Series(dtype="int64"),
            "child_rows": pd.Series(dtype="int64"),
            "source_mix": pd.Series(dtype="object"),
            "capture_polygon_ids": pd.Series(dtype="object"),
            "source_fids": pd.Series(dtype="object"),
            "themes": pd.Series(dtype="object"),
            "mean_similarity": pd.Series(dtype="float64"),
            "same_parity_rows": pd.Series(dtype="int64"),
            "area_m2": pd.Series(dtype="float64"),
            "part_count": pd.Series(dtype="int64"),
            "geometry": gpd.GeoSeries([], crs=TARGET_CRS),
        },
        geometry="geometry",
        crs=TARGET_CRS,
    )


def build_capture_clusters(
    capture: gpd.GeoDataFrame,
    *,
    cluster_gap: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    capture_out = capture.copy()
    capture_out["_cluster_id"] = -1
    if capture_out.empty:
        return capture_out, empty_cluster_rows()

    gap = max(float(cluster_gap), 0.0)
    cluster_rows: list[dict] = []

    for parent_key, group in capture_out.groupby("_parent_key", sort=False):
        valid = group[group.geometry.notna() & ~group.geometry.is_empty].copy()
        if valid.empty:
            continue

        neighbors: dict[int, list[int]] = {int(idx): [] for idx in valid.index}
        sindex = valid.sindex
        for pos, idx in enumerate(valid.index):
            geom = valid.geometry.iloc[pos]
            query_geom = geom.buffer(gap) if gap > 0 else geom
            for other_pos in sindex.query(query_geom, predicate="intersects"):
                other_idx = int(valid.index[int(other_pos)])
                if other_idx == int(idx):
                    continue
                other_geom = valid.at[other_idx, "geometry"]
                if gap > 0 and float(geom.distance(other_geom)) > gap:
                    continue
                neighbors[int(idx)].append(other_idx)

        seen: set[int] = set()
        cluster_id = 0
        for start_idx in valid.index:
            start_idx = int(start_idx)
            if start_idx in seen:
                continue
            cluster_id += 1
            stack = [start_idx]
            component: list[int] = []
            seen.add(start_idx)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in neighbors[current]:
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    stack.append(neighbor)

            component_rows = capture_out.loc[component].copy()
            capture_out.loc[component, "_cluster_id"] = int(cluster_id)
            geom = unary_union(list(component_rows.geometry))
            polygon_ids = ",".join(
                sorted(
                    {
                        str(value)
                        for value in component_rows.get("capture_polygon_id", pd.Series(dtype="object")).dropna()
                        if str(value).strip()
                    }
                )
            )
            source_fids = ",".join(
                sorted(
                    {
                        str(value)
                        for value in component_rows.get("capture_polygon_source_fid", pd.Series(dtype="object")).dropna()
                        if str(value).strip()
                    }
                )
            )
            themes = ",".join(
                sorted(
                    {
                        str(value)
                        for value in component_rows.get("capture_polygon_theme", pd.Series(dtype="object")).dropna()
                        if str(value).strip()
                    }
                )
            )
            source_mix = ",".join(sorted(component_rows["capture_source"].fillna("").astype(str).unique()))
            point_rows = component_rows.drop_duplicates("capture_src_id")
            cluster_rows.append(
                {
                    "oachargeid": parent_key,
                    "cluster_id": int(cluster_id),
                    "capture_rows": int(len(component_rows)),
                    "point_rows": int(point_rows["capture_src_id"].nunique()),
                    "child_rows": int(point_rows["_is_child"].astype(bool).sum()) if "_is_child" in point_rows.columns else 0,
                    "source_mix": source_mix,
                    "capture_polygon_ids": polygon_ids,
                    "source_fids": source_fids,
                    "themes": themes,
                    "mean_similarity": float(point_rows["_similarity"].mean()) if "_similarity" in point_rows.columns else 0.0,
                    "same_parity_rows": int(point_rows["_parity"].eq("same").sum()) if "_parity" in point_rows.columns else 0,
                    "area_m2": float(geom.area),
                    "part_count": int(part_count(geom)),
                    "geometry": geom,
                }
            )

    cluster_gdf = (
        gpd.GeoDataFrame(cluster_rows, geometry="geometry", crs=TARGET_CRS)
        if cluster_rows
        else empty_cluster_rows()
    )
    return capture_out, cluster_gdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture points to WFS polygons, then cluster/merge captured polygons by oachargeid.",
    )
    parser.add_argument("--point-gpkg", required=True)
    parser.add_argument("--point-layer")
    parser.add_argument("--wfs-merge-gpkg", required=True)
    parser.add_argument("--wfs-merge-layer")
    parser.add_argument("--wfs-gpkg", required=True)
    parser.add_argument("--wfs-layer")
    parser.add_argument("--output-gpkg", required=True)
    parser.add_argument("--output-layer", default="capture_polygons")
    parser.add_argument("--cluster-layer", default="capture_clusters")
    parser.add_argument("--unmatched-layer", default="unmatched_points")
    parser.add_argument("--theme-field", default="Theme")
    parser.add_argument("--theme-include", default="building,land")
    parser.add_argument("--cluster-gap", type=float, default=0.0)
    parser.add_argument("--expanded-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    terms = parse_terms(args.theme_include)

    point_layer = choose_layer(args.point_gpkg, args.point_layer)
    merge_layer = choose_layer(args.wfs_merge_gpkg, args.wfs_merge_layer)
    raw_layer = choose_layer(args.wfs_gpkg, args.wfs_layer)

    print(f"[INFO] Reading points: {args.point_gpkg} (layer={point_layer})")
    points = load_layer(args.point_gpkg, point_layer)
    points = add_parent_keys(points)
    if args.expanded_only:
        before = len(points)
        points = points[expanded_parent_mask(points)].copy()
        print(f"[INFO] expanded-only filter: before={before} after={len(points)}")
    print(f"[INFO] points={len(points)}")

    print(f"[INFO] Reading merged WFS: {args.wfs_merge_gpkg} (layer={merge_layer})")
    merged_wfs = prepare_polygons(args.wfs_merge_gpkg, merge_layer, args.theme_field, None)
    print(f"[INFO] merge eligible={len(merged_wfs)}")

    print(f"[INFO] Reading raw WFS: {args.wfs_gpkg} (layer={raw_layer})")
    raw_wfs = prepare_polygons(args.wfs_gpkg, raw_layer, args.theme_field, terms)
    print(f"[INFO] raw eligible={len(raw_wfs)}")

    capture, unmatched, counts = build_capture(points, merged_wfs, raw_wfs, args.theme_field)
    for key, value in counts.items():
        print(f"[INFO] {key}={value}")
    if not capture.empty:
        print(f"[INFO] source_counts={capture['capture_source'].value_counts(dropna=False).to_dict()}")

    capture, clusters = build_capture_clusters(capture, cluster_gap=args.cluster_gap)
    print(
        "[INFO] capture_clusters="
        f"{len(clusters)} parents_with_clusters={clusters['oachargeid'].nunique() if not clusters.empty else 0} "
        f"gap={float(args.cluster_gap):.2f}m"
    )

    output_path = Path(args.output_gpkg)
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture.to_file(args.output_gpkg, layer=args.output_layer, driver="GPKG", engine="pyogrio")
    clusters.to_file(args.output_gpkg, layer=args.cluster_layer, driver="GPKG", engine="pyogrio")
    unmatched.to_file(args.output_gpkg, layer=args.unmatched_layer, driver="GPKG", engine="pyogrio")
    print(f"[INFO] Wrote {args.output_gpkg}")


if __name__ == "__main__":
    main()

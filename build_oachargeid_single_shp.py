#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box
from shapely.ops import unary_union


TARGET_CRS = "EPSG:27700"
ROADLIKE_CLARITY = {"road", "offset", "named"}


def choose_layer(path: str, layer: Optional[str]) -> str:
    if layer:
        return layer
    layers = pyogrio.list_layers(path)
    if len(layers) == 0:
        raise ValueError(f"No layers found in {path}")
    return str(layers[0][0])


def load_layer(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    layer_name = choose_layer(path, layer)
    gdf = gpd.read_file(path, layer=layer_name, engine="pyogrio")
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


def add_parent_keys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["_parent_key"] = out["oachargeid"].map(clean_key)
    out["_variant_key"] = out["oachargeid_sub"].map(clean_key)
    out["_is_child"] = out["_variant_key"].ne(out["_parent_key"])
    return out


def polygon_parts(geom) -> list[object]:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return []
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


def largest_polygon(geom):
    parts = polygon_parts(geom)
    if not parts:
        return None
    return max(parts, key=lambda part: float(part.area))


def single_polygon(geom):
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    return largest_polygon(geom)


def square_around_point(point, size_m: float):
    half = float(size_m) / 2.0
    return box(point.x - half, point.y - half, point.x + half, point.y + half)


def confidence_rank(value: object) -> int:
    text = str(value or "").strip().lower()
    if text == "high":
        return 3
    if text == "medium":
        return 2
    if text == "low":
        return 1
    return 0


def is_roadlike_group(point_group: gpd.GeoDataFrame) -> bool:
    clarity = {str(value).strip().lower() for value in point_group["address_clarity"].dropna().tolist()}
    if "road" in clarity:
        return True
    child_group = point_group[point_group["_is_child"]].copy()
    if child_group.empty:
        return bool(clarity & {"offset", "named"})
    number_values = child_group.get("address_number", pd.Series([], dtype=object)).fillna("").astype(str).str.strip()
    numbered_ratio = float(number_values.ne("").mean()) if len(number_values) else 0.0
    unique_roads = child_group.get("address_road", pd.Series([], dtype=object)).dropna().astype(str).str.strip()
    unique_roads = unique_roads[unique_roads.ne("")].nunique()
    return bool(clarity <= ROADLIKE_CLARITY and (numbered_ratio < 0.5 or unique_roads > 1))


def best_polygon_row(candidates: gpd.GeoDataFrame):
    if candidates.empty:
        return None
    ranked = candidates.copy()
    ranked["_sim"] = pd.to_numeric(ranked.get("api_address_similarity", pd.Series(index=ranked.index)), errors="coerce").fillna(0)
    ranked["_geo_conf_rank"] = ranked.get("geocoding_confidence", pd.Series(index=ranked.index)).map(confidence_rank).fillna(0)
    ranked["_has_number"] = ranked.get("address_number", pd.Series(index=ranked.index)).fillna("").astype(str).str.strip().ne("")
    ranked["_is_merge"] = ranked.get("capture_source", pd.Series(index=ranked.index)).fillna("").astype(str).eq("wfs_merge")
    ranked["_area"] = ranked.geometry.area.astype(float)
    ranked = ranked.sort_values(
        ["_sim", "_geo_conf_rank", "_has_number", "_is_merge", "_area", "_variant_key"],
        ascending=[False, False, False, False, False, True],
    )
    return ranked.iloc[0]


def representative_point_row(point_group: gpd.GeoDataFrame):
    parent_row = point_group[~point_group["_is_child"]]
    if not parent_row.empty:
        return parent_row.iloc[0]
    return point_group.iloc[0]


def make_output_row(parent_key: str, point_group: gpd.GeoDataFrame, capture_group: gpd.GeoDataFrame, square_size: float):
    point_rows = int(len(point_group))
    child_rows = int(point_group["_is_child"].sum())
    matched_rows = int(len(capture_group))
    child_variants = set(point_group.loc[point_group["_is_child"], "_variant_key"].tolist())
    matched_child_variants = set(capture_group.loc[capture_group["_is_child"], "_variant_key"].tolist()) if not capture_group.empty else set()
    missing_child_count = int(len(child_variants - matched_child_variants))
    address_clarity = ",".join(sorted(point_group["address_clarity"].fillna("").astype(str).unique()))
    description = " | ".join(
        point_group["charge-geographic-description"].dropna().astype(str).drop_duplicates().head(2).tolist()
    )

    common = {
        "oachargeid": parent_key,
        "point_rows": point_rows,
        "child_rows": child_rows,
        "matched_rows": matched_rows,
        "missing_child_count": missing_child_count,
        "address_clarity": address_clarity,
        "source_description": description,
    }

    if capture_group.empty:
        source_point = representative_point_row(point_group)
        geom = square_around_point(source_point.geometry, square_size)
        return {
            **common,
            "auto_polygon_confidence": "low",
            "aggregation_method": "point_square_10m",
            "selected_oachargeid_sub": clean_key(source_point.get("oachargeid_sub")),
            "selected_similarity": float(pd.to_numeric(source_point.get("api_address_similarity"), errors="coerce") or 0.0),
            "base_parts": 0,
            "omitted_island_count": 0,
            "omitted_island_area_m2": 0.0,
            "omitted_island_area_ratio": 0.0,
            "final_area_m2": float(geom.area),
            "geometry": geom,
        }

    child_capture = capture_group[capture_group["_is_child"]].copy()
    use_group = child_capture if child_rows > 0 and not child_capture.empty else capture_group.copy()

    if is_roadlike_group(point_group):
        row = best_polygon_row(use_group)
        geom = single_polygon(row.geometry)
        selected_similarity = float(pd.to_numeric(row.get("api_address_similarity"), errors="coerce") or 0.0)
        if selected_similarity >= 90 and missing_child_count == 0:
            confidence = "medium"
        else:
            confidence = "low"
        return {
            **common,
            "auto_polygon_confidence": confidence,
            "aggregation_method": "roadlike_best_polygon",
            "selected_oachargeid_sub": clean_key(row.get("oachargeid_sub")),
            "selected_similarity": selected_similarity,
            "base_parts": int(len(polygon_parts(unary_union(list(use_group.geometry))))),
            "omitted_island_count": max(int(len(polygon_parts(unary_union(list(use_group.geometry))))) - 1, 0),
            "omitted_island_area_m2": float(max(float(unary_union(list(use_group.geometry)).area) - float(geom.area), 0.0)),
            "omitted_island_area_ratio": float(max(float(unary_union(list(use_group.geometry)).area) - float(geom.area), 0.0) / max(float(unary_union(list(use_group.geometry)).area), 1e-9)),
            "final_area_m2": float(geom.area),
            "geometry": geom,
        }

    base = unary_union(list(use_group.geometry))
    parts = sorted(polygon_parts(base), key=lambda geom: float(geom.area), reverse=True)
    if not parts:
        source_point = representative_point_row(point_group)
        geom = square_around_point(source_point.geometry, square_size)
        return {
            **common,
            "auto_polygon_confidence": "low",
            "aggregation_method": "point_square_10m_empty_capture",
            "selected_oachargeid_sub": clean_key(source_point.get("oachargeid_sub")),
            "selected_similarity": float(pd.to_numeric(source_point.get("api_address_similarity"), errors="coerce") or 0.0),
            "base_parts": 0,
            "omitted_island_count": 0,
            "omitted_island_area_m2": 0.0,
            "omitted_island_area_ratio": 0.0,
            "final_area_m2": float(geom.area),
            "geometry": geom,
        }

    selected = parts[0]
    omitted_area = float(sum(part.area for part in parts[1:]))
    total_area = float(sum(part.area for part in parts))
    omitted_ratio = omitted_area / max(total_area, 1e-9)
    if len(parts) == 1 and missing_child_count == 0:
        confidence = "high"
        method = "child_union_single" if child_rows > 0 else "single_polygon"
    elif len(parts) == 1:
        confidence = "medium"
        method = "single_polygon_missing_child"
    else:
        method = "largest_island_from_multipart"
        if missing_child_count > 0 or omitted_ratio > 0.25 or omitted_area > 1000.0:
            confidence = "low"
        else:
            confidence = "medium"

    best_row = best_polygon_row(use_group)
    selected_similarity = float(pd.to_numeric(best_row.get("api_address_similarity"), errors="coerce") or 0.0) if best_row is not None else 0.0
    return {
        **common,
        "auto_polygon_confidence": confidence,
        "aggregation_method": method,
        "selected_oachargeid_sub": clean_key(best_row.get("oachargeid_sub")) if best_row is not None else "",
        "selected_similarity": selected_similarity,
        "base_parts": int(len(parts)),
        "omitted_island_count": int(max(len(parts) - 1, 0)),
        "omitted_island_area_m2": omitted_area,
        "omitted_island_area_ratio": omitted_ratio,
        "final_area_m2": float(selected.area),
        "geometry": selected,
    }


def remove_shapefile(path: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]:
        target = path.with_suffix(suffix)
        if target.exists():
            target.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final single-polygon-per-oachargeid Shapefile.")
    parser.add_argument("--point-gpkg", required=True)
    parser.add_argument("--point-layer")
    parser.add_argument("--capture-gpkg", required=True)
    parser.add_argument("--capture-layer", default="capture_polygons")
    parser.add_argument("--output-shp", required=True)
    parser.add_argument("--output-gpkg")
    parser.add_argument("--output-layer", default="oachargeid_single_polygons")
    parser.add_argument("--square-size", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    point_layer = choose_layer(args.point_gpkg, args.point_layer)
    capture_layer = choose_layer(args.capture_gpkg, args.capture_layer)

    print(f"[INFO] Reading points: {args.point_gpkg} (layer={point_layer})")
    points = add_parent_keys(load_layer(args.point_gpkg, point_layer))
    print(f"[INFO] points={len(points)} parents={points['_parent_key'].nunique()}")

    print(f"[INFO] Reading capture polygons: {args.capture_gpkg} (layer={capture_layer})")
    capture = add_parent_keys(load_layer(args.capture_gpkg, capture_layer))
    print(f"[INFO] capture polygons={len(capture)}")

    capture_groups = {key: group.copy() for key, group in capture.groupby("_parent_key", sort=False)}
    rows = []
    for parent_key, point_group in points.groupby("_parent_key", sort=False):
        capture_group = capture_groups.get(parent_key, capture.iloc[0:0]).copy()
        rows.append(make_output_row(parent_key, point_group.copy(), capture_group, float(args.square_size)))

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)
    out = out.sort_values("oachargeid").reset_index(drop=True)

    output_gpkg = Path(args.output_gpkg) if args.output_gpkg else Path(args.output_shp).with_suffix(".gpkg")
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()
    out.to_file(output_gpkg, layer=args.output_layer, driver="GPKG", engine="pyogrio")

    shp_path = Path(args.output_shp)
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    remove_shapefile(shp_path)
    shp = out[
        [
            "oachargeid",
            "auto_polygon_confidence",
            "aggregation_method",
            "point_rows",
            "child_rows",
            "matched_rows",
            "missing_child_count",
            "address_clarity",
            "selected_oachargeid_sub",
            "selected_similarity",
            "base_parts",
            "omitted_island_count",
            "omitted_island_area_m2",
            "omitted_island_area_ratio",
            "final_area_m2",
            "geometry",
        ]
    ].copy()
    shp["auto_conf"] = shp["auto_polygon_confidence"]
    shp.to_file(shp_path, driver="ESRI Shapefile", engine="pyogrio", encoding="UTF-8")

    print(f"[DONE] Wrote GPKG: {output_gpkg}")
    print(f"[DONE] Wrote SHP: {shp_path}")
    print(f"[INFO] rows={len(out)}")
    print("[INFO] confidence counts:")
    print(out["auto_polygon_confidence"].value_counts(dropna=False).to_string())
    print("[INFO] aggregation methods:")
    print(out["aggregation_method"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

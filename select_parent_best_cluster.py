#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio


TARGET_CRS = "EPSG:27700"


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


def numeric(gdf: gpd.GeoDataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in gdf.columns:
        return pd.Series(default, index=gdf.index, dtype="float64")
    return pd.to_numeric(gdf[column], errors="coerce").fillna(default)


def select_best_clusters(clusters: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    clusters = clusters.copy()
    clusters["oachargeid"] = clusters["oachargeid"].astype(str).str.replace(r"\.0$", "", regex=True)
    clusters["point_rows"] = numeric(clusters, "point_rows").astype(int)
    clusters["same_parity_rows"] = numeric(clusters, "same_parity_rows").astype(int)
    clusters["mean_similarity"] = numeric(clusters, "mean_similarity")
    clusters["area_m2"] = numeric(clusters, "area_m2")
    clusters["same_ratio"] = (
        clusters["same_parity_rows"].astype(float) / clusters["point_rows"].replace(0, pd.NA).astype(float)
    ).fillna(0.0)

    group_stats = (
        clusters.groupby("oachargeid", dropna=False)
        .agg(
            cluster_count=("oachargeid", "size"),
            total_cluster_point_rows=("point_rows", "sum"),
            total_cluster_area_m2=("area_m2", "sum"),
        )
        .reset_index()
    )

    ranked = clusters.sort_values(
        ["oachargeid", "mean_similarity", "same_ratio", "point_rows", "area_m2"],
        ascending=[True, False, False, False, False],
    ).copy()
    ranked["selection_rank"] = ranked.groupby("oachargeid", dropna=False).cumcount() + 1

    selected = ranked[ranked["selection_rank"].eq(1)].copy()
    selected = selected.merge(group_stats, on="oachargeid", how="left", suffixes=("", "_all"))
    selected["omitted_cluster_count"] = selected["cluster_count"] - 1
    selected["omitted_point_rows"] = selected["total_cluster_point_rows"] - selected["point_rows"]
    selected["omitted_area_m2"] = selected["total_cluster_area_m2"] - selected["area_m2"]
    selected["selection_rule"] = "mean_similarity_desc,same_ratio_desc,point_rows_desc,area_m2_desc"
    selected["auto_polygon_confidence"] = "medium"
    selected.loc[
        (selected["mean_similarity"].ge(90))
        & (selected["same_ratio"].ge(0.75))
        & (selected["omitted_cluster_count"].eq(0)),
        "auto_polygon_confidence",
    ] = "high"
    selected.loc[
        (selected["mean_similarity"].lt(75)) | (selected["omitted_point_rows"].gt(selected["point_rows"])),
        "auto_polygon_confidence",
    ] = "low"
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=clusters.crs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select one best WFS cluster per oachargeid.")
    parser.add_argument("--input-gpkg", required=True)
    parser.add_argument("--input-layer")
    parser.add_argument("--output-gpkg", required=True)
    parser.add_argument("--output-layer", default="parent_wfs_selected")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_layer = choose_layer(args.input_gpkg, args.input_layer)
    print(f"[INFO] Reading clusters: {args.input_gpkg} layer={input_layer}")
    clusters = gpd.read_file(args.input_gpkg, layer=input_layer, engine="pyogrio")
    clusters = to_target_crs(clusters)
    clusters = clusters[clusters.geometry.notna()].copy()
    clusters = clusters[~clusters.geometry.is_empty].copy()
    print(f"[INFO] cluster_rows={len(clusters)} parents={clusters['oachargeid'].nunique()}")

    selected = select_best_clusters(clusters)
    print(f"[INFO] selected_rows={len(selected)} parents={selected['oachargeid'].nunique()}")
    print(
        "[INFO] multi_cluster_selected="
        f"{int(selected['omitted_cluster_count'].astype(int).gt(0).sum())} "
        f"low_confidence={int(selected['auto_polygon_confidence'].eq('low').sum())}"
    )

    output_path = Path(args.output_gpkg)
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_file(output_path, layer=args.output_layer, driver="GPKG", engine="pyogrio")
    print(f"[DONE] Wrote {output_path} layer={args.output_layer}")


if __name__ == "__main__":
    main()

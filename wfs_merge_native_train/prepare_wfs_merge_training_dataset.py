#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import shapely


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wp5_wfs_polygons.gpkg"
DEFAULT_WFS_LAYER = "polygons_in_buffers"
DEFAULT_MERGE_GPKG = "/data/sheffield/spatial/base-map/sheffield_wp5_os_wfs_merge.gpkg"
DEFAULT_MERGE_LAYER = "os_wfs_merge"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_OUTPUT_CSV = "/data/sheffield/spatial/base-map/sheffield_wp5_wfs_merge_training_small_edges.csv"


WFS_COLUMNS = [
    "GmlID",
    "TOID",
    "Theme",
    "DescriptiveGroup",
    "DescriptiveTerm",
    "Make",
    "PhysicalLevel",
    "Shape_Area",
    "Shape_Length",
]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _parts(value: object) -> list[int]:
    out: list[int] = []
    for part in str(value or "").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _role(theme: object) -> str:
    text = str(theme or "").lower()
    if "building" in text:
        return "building"
    if "land" in text:
        return "land"
    return "other"


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, 1.0)


def _shape_metrics(geoms) -> pd.DataFrame:
    area = pd.Series(shapely.area(geoms), dtype="float64")
    perimeter = pd.Series(shapely.length(geoms), dtype="float64")
    hull_area = pd.Series(shapely.area(shapely.convex_hull(geoms)), dtype="float64")
    mrr_area = pd.Series(shapely.area(shapely.minimum_rotated_rectangle(geoms)), dtype="float64")
    compactness = 4.0 * math.pi * area / (perimeter * perimeter).replace(0.0, 1.0)
    return pd.DataFrame(
        {
            "area": area,
            "perimeter": perimeter,
            "mrr_ratio": _ratio(area, mrr_area),
            "hull_gap_ratio": (hull_area - area).clip(lower=0.0) / area.replace(0.0, 1.0),
            "compactness": compactness,
        }
    )


def _load_merge_labels(path: str, layer: str, verbose: bool) -> gpd.GeoDataFrame:
    _log(verbose, f"[INFO] Reading merge labels: {path} ({layer})")
    merge = gpd.read_file(
        path,
        layer=layer,
        engine="pyogrio",
        columns=["merge_source_fids", "merge_source_count", "merge_stage"],
        fid_as_index=True,
    )
    merge.index.name = "merge_fid"
    merge = merge[merge.geometry.notna() & ~merge.geometry.is_empty].copy()
    merge["merge_area"] = merge.geometry.area.astype(float)
    merge["merge_source_count"] = merge["merge_source_count"].fillna(1).astype(int)
    merge["merge_stage"] = merge["merge_stage"].fillna("").astype(str)
    _log(verbose, f"[INFO] Merge rows: {len(merge)}")
    return merge


def _source_to_merge_table(merge: gpd.GeoDataFrame) -> pd.DataFrame:
    records = []
    for merge_fid, row in merge.iterrows():
        source_fids = _parts(row.get("merge_source_fids"))
        if not source_fids and pd.notna(row.get("source_fid")):
            source_fids = _parts(row.get("source_fid"))
        for source_fid in source_fids:
            records.append(
                {
                    "source_fid": int(source_fid),
                    "merge_fid": int(merge_fid),
                    "merge_area": float(row["merge_area"]),
                    "merge_source_count": int(row["merge_source_count"]),
                    "merge_stage": str(row["merge_stage"]),
                }
            )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    table = table.sort_values(["source_fid", "merge_source_count", "merge_fid"])
    return table.drop_duplicates("source_fid", keep="first").reset_index(drop=True)


def _read_wfs(path: str, layer: str, include_terms: set[str], verbose: bool) -> gpd.GeoDataFrame:
    _log(verbose, f"[INFO] Reading WFS candidates: {path} ({layer})")
    wfs = gpd.read_file(path, layer=layer, engine="pyogrio", columns=WFS_COLUMNS, fid_as_index=True)
    wfs.index.name = "source_fid"
    wfs = wfs[wfs.geometry.notna() & ~wfs.geometry.is_empty].copy()
    wfs["source_fid"] = wfs.index.astype(int)
    if include_terms:
        theme = wfs["Theme"].fillna("").astype(str).str.lower()
        mask = pd.Series(False, index=wfs.index)
        for term in include_terms:
            mask |= theme.str.contains(term.lower(), regex=False)
        wfs = wfs[mask].copy()
    wfs["role"] = wfs["Theme"].map(_role)
    wfs["raw_area"] = wfs.geometry.area.astype(float)
    wfs["raw_perimeter"] = wfs.geometry.length.astype(float)
    _log(verbose, f"[INFO] WFS candidate rows after theme filter: {len(wfs)}")
    return wfs


def _load_uprn_counts(
    uprn_gpkg: str | None,
    uprn_layer: str,
    uprn_id_field: str,
    polygons: gpd.GeoDataFrame,
    verbose: bool,
) -> pd.Series:
    if not uprn_gpkg or polygons.empty:
        return pd.Series(0, index=polygons["source_fid"], dtype="int64")

    bounds = tuple(float(v) for v in polygons.total_bounds)
    _log(verbose, f"[INFO] Reading UPRNs in candidate bounds: {uprn_gpkg} ({uprn_layer})")
    points = gpd.read_file(
        uprn_gpkg,
        layer=uprn_layer,
        engine="pyogrio",
        bbox=bounds,
        columns=[uprn_id_field],
    )
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if points.empty:
        return pd.Series(0, index=polygons["source_fid"], dtype="int64")
    if points.crs is None and polygons.crs is not None:
        points = points.set_crs(polygons.crs)
    elif polygons.crs is not None and points.crs != polygons.crs:
        points = points.to_crs(polygons.crs)

    _log(verbose, f"[INFO] UPRN points in bounds: {len(points)}")
    polygon_ref = polygons[["source_fid", "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(
        points[[uprn_id_field, "geometry"]],
        polygon_ref,
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return pd.Series(0, index=polygons["source_fid"], dtype="int64")
    counts = joined.groupby("source_fid")[uprn_id_field].nunique()
    return polygons["source_fid"].map(counts).fillna(0).astype(int)


def _candidate_pairs(
    selected: gpd.GeoDataFrame,
    pool: gpd.GeoDataFrame,
    *,
    min_shared_edge: float,
    max_overlap_area: float,
    verbose: bool,
) -> pd.DataFrame:
    _log(verbose, f"[INFO] Spatial join for adjacency: selected={len(selected)} pool={len(pool)}")
    selected_ref = selected[["source_fid", "geometry"]].reset_index(drop=True)
    pool_ref = pool[["source_fid", "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(
        selected_ref,
        pool_ref,
        how="inner",
        predicate="intersects",
        lsuffix="left",
        rsuffix="right",
    )
    if joined.empty:
        return pd.DataFrame()

    joined = joined.rename(columns={"source_fid_left": "left_source_fid", "source_fid_right": "right_source_fid"})
    joined = joined[joined["left_source_fid"].ne(joined["right_source_fid"])].copy()
    if joined.empty:
        return pd.DataFrame()

    left_id = joined["left_source_fid"].astype(int)
    right_id = joined["right_source_fid"].astype(int)
    joined["pair_a"] = left_id.where(left_id < right_id, right_id)
    joined["pair_b"] = right_id.where(left_id < right_id, left_id)
    joined = joined.drop_duplicates(["pair_a", "pair_b"]).reset_index(drop=True)

    geom_by_source = pool.set_index("source_fid")["geometry"]
    left_geoms = gpd.GeoSeries(joined["pair_a"].map(geom_by_source).to_list(), index=joined.index, crs=pool.crs)
    right_geoms = gpd.GeoSeries(joined["pair_b"].map(geom_by_source).to_list(), index=joined.index, crs=pool.crs)

    boundary_intersections = shapely.intersection(shapely.boundary(left_geoms.array), shapely.boundary(right_geoms.array))
    joined["shared_edge_len"] = shapely.length(boundary_intersections)
    joined["overlap_area"] = shapely.area(shapely.intersection(left_geoms.array, right_geoms.array))
    joined = joined[
        (joined["shared_edge_len"] >= float(min_shared_edge))
        & (joined["overlap_area"] <= float(max_overlap_area))
    ].copy()
    _log(verbose, f"[INFO] Shared-edge candidate pairs: {len(joined)}")
    return joined[["pair_a", "pair_b", "shared_edge_len", "overlap_area"]].reset_index(drop=True)


def _add_pair_features(pairs: pd.DataFrame, pool: gpd.GeoDataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    attrs = pool.set_index("source_fid")
    label_attrs = labels.set_index("source_fid")

    left = attrs.loc[pairs["pair_a"].to_numpy()]
    right = attrs.loc[pairs["pair_b"].to_numpy()]
    left_labels = label_attrs.loc[pairs["pair_a"].to_numpy()]
    right_labels = label_attrs.loc[pairs["pair_b"].to_numpy()]

    left_geoms = left.geometry.array
    right_geoms = right.geometry.array
    union_geoms = shapely.union(left_geoms, right_geoms)

    left_shape = _shape_metrics(left_geoms).add_prefix("left_")
    right_shape = _shape_metrics(right_geoms).add_prefix("right_")
    union_shape = _shape_metrics(union_geoms).add_prefix("union_")

    out = pd.DataFrame(
        {
            "left_source_fid": pairs["pair_a"].astype(int).to_numpy(),
            "right_source_fid": pairs["pair_b"].astype(int).to_numpy(),
            "label": (left_labels["merge_fid"].to_numpy() == right_labels["merge_fid"].to_numpy()).astype(int),
            "left_merge_fid": left_labels["merge_fid"].astype(int).to_numpy(),
            "right_merge_fid": right_labels["merge_fid"].astype(int).to_numpy(),
            "left_merge_area": left_labels["merge_area"].astype(float).to_numpy(),
            "right_merge_area": right_labels["merge_area"].astype(float).to_numpy(),
            "left_merge_source_count": left_labels["merge_source_count"].astype(int).to_numpy(),
            "right_merge_source_count": right_labels["merge_source_count"].astype(int).to_numpy(),
            "left_merge_stage": left_labels["merge_stage"].astype(str).to_numpy(),
            "right_merge_stage": right_labels["merge_stage"].astype(str).to_numpy(),
            "left_theme": left["Theme"].fillna("").astype(str).to_numpy(),
            "right_theme": right["Theme"].fillna("").astype(str).to_numpy(),
            "left_role": left["role"].astype(str).to_numpy(),
            "right_role": right["role"].astype(str).to_numpy(),
            "left_descriptive_group": left["DescriptiveGroup"].fillna("").astype(str).to_numpy(),
            "right_descriptive_group": right["DescriptiveGroup"].fillna("").astype(str).to_numpy(),
            "left_descriptive_term": left["DescriptiveTerm"].fillna("").astype(str).to_numpy(),
            "right_descriptive_term": right["DescriptiveTerm"].fillna("").astype(str).to_numpy(),
            "left_make": left["Make"].fillna("").astype(str).to_numpy(),
            "right_make": right["Make"].fillna("").astype(str).to_numpy(),
            "left_physical_level": left["PhysicalLevel"].fillna(-9999).astype(int).to_numpy(),
            "right_physical_level": right["PhysicalLevel"].fillna(-9999).astype(int).to_numpy(),
            "left_uprn_count": left["uprn_count"].fillna(0).astype(int).to_numpy(),
            "right_uprn_count": right["uprn_count"].fillna(0).astype(int).to_numpy(),
            "shared_edge_len": pairs["shared_edge_len"].astype(float).to_numpy(),
            "overlap_area": pairs["overlap_area"].astype(float).to_numpy(),
        }
    )
    out = pd.concat([out, left_shape, right_shape, union_shape], axis=1)
    small_area = out[["left_area", "right_area"]].min(axis=1)
    large_area = out[["left_area", "right_area"]].max(axis=1)
    small_perimeter = out[["left_perimeter", "right_perimeter"]].min(axis=1)
    out["small_area"] = small_area
    out["large_area"] = large_area
    out["small_large_area_ratio"] = small_area / large_area.replace(0.0, 1.0)
    out["shared_ratio_small_perimeter"] = out["shared_edge_len"] / small_perimeter.replace(0.0, 1.0)
    out["role_pair"] = out["left_role"] + "__" + out["right_role"]
    out["same_descriptive_group"] = (
        out["left_descriptive_group"].ne("") & out["left_descriptive_group"].eq(out["right_descriptive_group"])
    ).astype(int)
    out["same_descriptive_term"] = (
        out["left_descriptive_term"].ne("") & out["left_descriptive_term"].eq(out["right_descriptive_term"])
    ).astype(int)
    out["same_make"] = (out["left_make"].ne("") & out["left_make"].eq(out["right_make"])).astype(int)
    out["same_physical_level"] = out["left_physical_level"].eq(out["right_physical_level"]).astype(int)
    out["uprn_count_sum"] = out["left_uprn_count"] + out["right_uprn_count"]
    out["both_have_uprn"] = ((out["left_uprn_count"] > 0) & (out["right_uprn_count"] > 0)).astype(int)
    out["one_has_uprn"] = ((out["left_uprn_count"] > 0) ^ (out["right_uprn_count"] > 0)).astype(int)
    out["neither_has_uprn"] = ((out["left_uprn_count"] == 0) & (out["right_uprn_count"] == 0)).astype(int)
    out["mid_x"] = (shapely.get_x(shapely.centroid(left_geoms)) + shapely.get_x(shapely.centroid(right_geoms))) / 2.0
    out["mid_y"] = (shapely.get_y(shapely.centroid(left_geoms)) + shapely.get_y(shapely.centroid(right_geoms))) / 2.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WFS merge edge-classifier training data from small parcels.")
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--merge-gpkg", default=DEFAULT_MERGE_GPKG)
    parser.add_argument("--merge-layer", default=DEFAULT_MERGE_LAYER)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default="UPRN")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-parcel-area", type=float, default=2000.0)
    parser.add_argument("--max-merge-source-count", type=int, default=20)
    parser.add_argument("--max-parcels", type=int, default=20000)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--max-pairs", type=int, default=250000)
    parser.add_argument("--min-shared-edge", type=float, default=0.05)
    parser.add_argument("--max-overlap-area", type=float, default=1e-6)
    parser.add_argument("--include-terms", default="building,land")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-uprn", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not bool(args.quiet)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    merge = _load_merge_labels(args.merge_gpkg, args.merge_layer, verbose)
    source_labels = _source_to_merge_table(merge)
    if source_labels.empty:
        raise RuntimeError("No source_fid labels could be parsed from merge_source_fids.")
    _log(verbose, f"[INFO] Parsed source labels: {len(source_labels)}")

    small = merge[
        (merge["merge_area"] <= float(args.max_parcel_area))
        & (merge["merge_source_count"] <= int(args.max_merge_source_count))
    ].copy()
    small = small.sort_values(["merge_area", "merge_source_count"], ascending=[True, True])
    if args.max_parcels and len(small) > args.max_parcels:
        small = small.sample(n=int(args.max_parcels), random_state=int(args.random_seed)).sort_index()
    selected_source_fids = sorted({fid for value in small["merge_source_fids"] for fid in _parts(value)})
    _log(
        verbose,
        f"[INFO] Selected small parcels: {len(small)}; selected source_fids: {len(selected_source_fids)}",
    )

    include_terms = {term.strip().lower() for term in str(args.include_terms or "").split(",") if term.strip()}
    wfs = _read_wfs(args.wfs_gpkg, args.wfs_layer, include_terms, verbose)
    label_ids = set(source_labels["source_fid"].astype(int))
    wfs = wfs[wfs["source_fid"].isin(label_ids)].copy()
    selected = wfs[wfs["source_fid"].isin(selected_source_fids)].copy()
    if selected.empty:
        raise RuntimeError("No selected WFS source features were found.")

    pairs = _candidate_pairs(
        selected,
        wfs,
        min_shared_edge=float(args.min_shared_edge),
        max_overlap_area=float(args.max_overlap_area),
        verbose=verbose,
    )
    if pairs.empty:
        raise RuntimeError("No shared-edge candidate pairs were found.")

    pair_source_ids = sorted(set(pairs["pair_a"].astype(int)) | set(pairs["pair_b"].astype(int)))
    feature_pool = wfs[wfs["source_fid"].isin(pair_source_ids)].copy()
    if args.no_uprn:
        feature_pool["uprn_count"] = 0
    else:
        feature_pool["uprn_count"] = _load_uprn_counts(
            args.uprn_gpkg,
            args.uprn_layer,
            args.uprn_id_field,
            feature_pool,
            verbose,
        ).to_numpy()

    dataset = _add_pair_features(pairs, feature_pool, source_labels)
    positives = dataset[dataset["label"].eq(1)]
    negatives = dataset[dataset["label"].eq(0)]
    if not positives.empty and args.negative_ratio >= 0:
        max_negatives = int(math.ceil(len(positives) * float(args.negative_ratio)))
        if len(negatives) > max_negatives:
            negatives = negatives.sample(n=max_negatives, random_state=int(args.random_seed))
        dataset = pd.concat([positives, negatives], ignore_index=True)
    if args.max_pairs and len(dataset) > args.max_pairs:
        positives = dataset[dataset["label"].eq(1)]
        negatives = dataset[dataset["label"].eq(0)]
        remaining = max(0, int(args.max_pairs) - len(positives))
        if len(negatives) > remaining:
            negatives = negatives.sample(n=remaining, random_state=int(args.random_seed))
        dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=int(args.random_seed)).reset_index(drop=True)

    dataset.to_csv(output_csv, index=False)
    meta = {
        "wfs_gpkg": args.wfs_gpkg,
        "wfs_layer": args.wfs_layer,
        "merge_gpkg": args.merge_gpkg,
        "merge_layer": args.merge_layer,
        "uprn_gpkg": None if args.no_uprn else args.uprn_gpkg,
        "output_csv": str(output_csv),
        "max_parcel_area": args.max_parcel_area,
        "max_merge_source_count": args.max_merge_source_count,
        "max_parcels": args.max_parcels,
        "selected_small_parcels": int(len(small)),
        "selected_source_fids": int(len(selected_source_fids)),
        "candidate_pairs_before_sampling": int(len(pairs)),
        "rows": int(len(dataset)),
        "positive_rows": int(dataset["label"].sum()),
        "negative_rows": int((dataset["label"] == 0).sum()),
        "columns": list(dataset.columns),
    }
    meta_path = output_csv.with_suffix(output_csv.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(verbose, f"[DONE] Wrote training CSV: {output_csv} ({len(dataset)} rows)")
    _log(verbose, f"[DONE] Wrote metadata: {meta_path}")
    _log(verbose, f"[INFO] Label counts: {dataset['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

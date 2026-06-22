from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from _core.config import add_config_argument, get_config_section_from_argv, require_configured
from _core.io import load_layer


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv("qa_parent_unique", include_package_defaults=True)
    parser = argparse.ArgumentParser(
        description="Build QA review layers for parent unique capture output.",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--capture-gpkg")
    parser.add_argument("--capture-layer")
    parser.add_argument("--point-gpkg")
    parser.add_argument("--point-layer")
    parser.add_argument("--raw-wfs-gpkg")
    parser.add_argument("--output-gpkg")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-polygons-layer")
    parser.add_argument("--output-points-layer")
    parser.add_argument("--road-anchor-distance", type=float)
    parser.add_argument("--large-area", type=float)
    parser.add_argument("--linked-parent-distance", type=float)
    parser.add_argument("--multipart-threshold", type=int)
    parser.add_argument("--low-compactness", type=float)
    parser.add_argument("--min-low-compactness-area", type=float)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(
        args,
        ("capture_gpkg", "point_gpkg", "raw_wfs_gpkg", "output_gpkg", "output_csv"),
        "qa_parent_unique",
    )
    return args


def _threshold_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _parts(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    return 1


def _compactness(geom) -> float:
    if geom is None or geom.is_empty or float(geom.length) <= 0.0:
        return 0.0
    return float(4.0 * 3.141592653589793 * geom.area / (geom.length * geom.length))


def _point_distance(row) -> float:
    point = row["point_geom"]
    geom = row["geometry"]
    if point is None or geom is None or point.is_empty or geom.is_empty:
        return float("nan")
    return float(point.distance(geom))


def main() -> None:
    args = parse_args()
    capture = load_layer(args.capture_gpkg, args.capture_layer)
    points = load_layer(args.point_gpkg, args.point_layer).copy()
    points["capture_src_id"] = points.index + 1
    raw_wfs = load_layer(args.raw_wfs_gpkg)

    nearest_under = gpd.sjoin_nearest(
        points[["capture_src_id", "geometry"]],
        raw_wfs[["Theme", "geometry"]],
        how="left",
        distance_col="underlying_wfs_dist_m",
    )
    nearest_under = (
        nearest_under.sort_values(["capture_src_id", "underlying_wfs_dist_m"])
        .drop_duplicates("capture_src_id")
        .copy()
    )

    qa_base = capture.merge(
        points[["capture_src_id", "geometry"]].rename(columns={"geometry": "point_geom"}),
        on="capture_src_id",
        how="left",
    ).merge(
        nearest_under[["capture_src_id", "Theme", "underlying_wfs_dist_m"]],
        on="capture_src_id",
        how="left",
    )
    qa_base = gpd.GeoDataFrame(qa_base, geometry="geometry", crs=capture.crs)
    qa_base["area_m2"] = qa_base.geometry.area
    qa_base["parts"] = qa_base.geometry.apply(_parts)
    qa_base["compactness"] = qa_base.geometry.apply(_compactness)
    qa_base["point_output_dist_m"] = qa_base.apply(_point_distance, axis=1)

    theme_text = qa_base["Theme"].fillna("").astype(str).str.lower()
    road_anchor = (
        (qa_base["underlying_wfs_dist_m"] <= 0.25)
        & ~theme_text.str.contains("building|land", regex=True)
        & (qa_base["point_output_dist_m"] > float(args.road_anchor_distance))
        & qa_base["capture_success"].fillna(False).astype(bool)
    )
    linked_far = qa_base["capture_stage"].eq("linked_parent_union") & (
        qa_base["point_output_dist_m"] > float(args.linked_parent_distance)
    )
    large_area = qa_base["area_m2"] > float(args.large_area)
    multipart = qa_base["parts"] >= int(args.multipart_threshold)
    low_compactness = (
        (qa_base["compactness"] < float(args.low_compactness))
        & (qa_base["area_m2"] > float(args.min_low_compactness_area))
    )
    failed = ~qa_base["capture_success"].fillna(False).astype(bool)

    masks = [
        (f"road_anchor_point_on_noneligible_wfs_gt{_threshold_label(args.road_anchor_distance)}m", road_anchor),
        (f"linked_parent_union_point_dist_gt{_threshold_label(args.linked_parent_distance)}m", linked_far),
        (f"large_area_gt{_threshold_label(args.large_area)}m2", large_area),
        (f"multipart_ge{int(args.multipart_threshold)}", multipart),
        (f"low_compactness_gt{_threshold_label(args.min_low_compactness_area)}m2", low_compactness),
        ("capture_failed", failed),
    ]

    any_mask = masks[0][1].copy()
    for _, mask in masks[1:]:
        any_mask = any_mask | mask
    qa = qa_base.loc[any_mask].copy()

    reason_map: dict[object, list[str]] = {}
    for name, mask in masks:
        for unique_key in qa_base.loc[mask, "unique_key"].tolist():
            reason_map.setdefault(unique_key, []).append(name)
    qa["qa_reason"] = qa["unique_key"].map(lambda key: ";".join(reason_map.get(key, [])))
    qa["qa_priority"] = qa["qa_reason"].apply(
        lambda value: 1
        if "capture_failed" in value or "road_anchor" in value
        else (2 if "linked_parent_union" in value else 3)
    )
    qa = qa.sort_values(["qa_priority", "point_output_dist_m", "area_m2"], ascending=[True, False, False])

    out_path = Path(args.output_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    qa.drop(columns=["point_geom"]).to_file(out_path, layer=args.output_polygons_layer, driver="GPKG")

    qa_points = points[points["capture_src_id"].isin(qa["capture_src_id"])].merge(
        qa.drop(columns=["geometry", "point_geom"]),
        on="capture_src_id",
        how="left",
        suffixes=("", "_capture"),
    )
    qa_points.to_file(out_path, layer=args.output_points_layer, driver="GPKG")

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    qa_points.drop(columns="geometry").to_csv(csv_path, index=False)

    print(f"[DONE] Wrote QA package: {out_path}")
    print(f"[DONE] Wrote QA CSV: {csv_path}")
    print(f"[INFO] QA features: {len(qa)}")
    print("[INFO] QA reasons:")
    print(qa["qa_reason"].value_counts(dropna=False).head(30).to_string())
    print("[INFO] Top QA rows:")
    cols = [
        "unique_key",
        "original_address",
        "capture_stage",
        "qa_reason",
        "point_output_dist_m",
        "area_m2",
        "parts",
        "compactness",
        "Theme",
    ]
    print(qa[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()

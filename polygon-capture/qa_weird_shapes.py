from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from _core.config import add_config_argument, get_config_section_from_argv, require_configured
from _core.io import load_layer


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv("qa_weird_shapes", include_package_defaults=True)
    parser = argparse.ArgumentParser(
        description="Flag unusual polygon shapes for manual QA.",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--capture-gpkg")
    parser.add_argument("--capture-layer")
    parser.add_argument("--output-gpkg")
    parser.add_argument("--output-layer")
    parser.add_argument("--output-csv")
    parser.add_argument("--large-area", type=float)
    parser.add_argument("--huge-area", type=float)
    parser.add_argument("--max-aspect", type=float)
    parser.add_argument("--low-compactness", type=float)
    parser.add_argument("--min-low-compactness-area", type=float)
    parser.add_argument("--multipart-threshold", type=int)
    parser.add_argument("--hole-threshold", type=int)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(args, ("capture_gpkg", "output_gpkg", "output_csv"), "qa_weird_shapes")
    return args


def _parts(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    return 1


def _holes(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return len(geom.interiors)
    if geom.geom_type == "MultiPolygon":
        return sum(len(poly.interiors) for poly in geom.geoms)
    return 0


def _aspect(geom) -> float:
    if geom is None or geom.is_empty:
        return float("inf")
    minx, miny, maxx, maxy = geom.bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    short = min(width, height)
    if short <= 0.0:
        return float("inf")
    return max(width, height) / short


def _compactness(geom) -> float:
    if geom is None or geom.is_empty or float(geom.length) <= 0.0:
        return 0.0
    return float(4.0 * 3.141592653589793 * geom.area / (geom.length * geom.length))


def main() -> None:
    args = parse_args()
    gdf = load_layer(args.capture_gpkg, args.capture_layer).copy()
    gdf["qa_area_m2"] = gdf.geometry.area
    gdf["qa_parts"] = gdf.geometry.apply(_parts)
    gdf["qa_holes"] = gdf.geometry.apply(_holes)
    gdf["qa_aspect"] = gdf.geometry.apply(_aspect)
    gdf["qa_compactness"] = gdf.geometry.apply(_compactness)

    flags = []
    for _, row in gdf.iterrows():
        row_flags = []
        if float(row["qa_area_m2"]) >= float(args.huge_area):
            row_flags.append("huge_area")
        elif float(row["qa_area_m2"]) >= float(args.large_area):
            row_flags.append("large_area")
        if int(row["qa_parts"]) >= int(args.multipart_threshold):
            row_flags.append("multipart")
        if int(row["qa_holes"]) >= int(args.hole_threshold):
            row_flags.append("many_holes")
        if float(row["qa_aspect"]) >= float(args.max_aspect):
            row_flags.append("high_aspect")
        if (
            float(row["qa_area_m2"]) >= float(args.min_low_compactness_area)
            and float(row["qa_compactness"]) <= float(args.low_compactness)
        ):
            row_flags.append("low_compactness")
        flags.append(";".join(row_flags))
    gdf["shape_qa_flags"] = flags

    review = gdf[gdf["shape_qa_flags"].astype(bool)].copy()
    severity_rank = {
        "huge_area": 1,
        "high_aspect": 1,
        "low_compactness": 2,
        "many_holes": 2,
        "multipart": 3,
        "large_area": 3,
    }

    def rank(value: str) -> int:
        parts = [severity_rank.get(flag, 9) for flag in value.split(";") if flag]
        return min(parts) if parts else 9

    review["shape_qa_priority"] = review["shape_qa_flags"].apply(rank)
    review = review.sort_values(
        ["shape_qa_priority", "qa_area_m2", "qa_aspect", "qa_parts"],
        ascending=[True, False, False, False],
    )

    out_path = Path(args.output_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    review.to_file(out_path, layer=args.output_layer, driver="GPKG")

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    review.drop(columns="geometry").to_csv(csv_path, index=False)

    print(f"[DONE] Wrote weird-shape QA: {out_path}")
    print(f"[DONE] Wrote weird-shape CSV: {csv_path}")
    print(f"[INFO] Review count: {len(review)}")
    print("[INFO] Flag counts:")
    exploded = review["shape_qa_flags"].str.split(";").explode()
    print(exploded.value_counts().to_string())
    print("[INFO] Top rows:")
    cols = [
        "unique_key",
        "original_address",
        "capture_stage",
        "shape_qa_flags",
        "qa_area_m2",
        "qa_parts",
        "qa_holes",
        "qa_aspect",
        "qa_compactness",
    ]
    print(review[cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()

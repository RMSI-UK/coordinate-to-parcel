#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from point_large_parcel_features import (
    boundary_neighbors_for_group,
    build_point_large_parcel_candidates,
    log,
    parse_fid_groups,
    read_seed_by_point,
)


DEFAULT_WFS_GPKG = "/data/sheffield/spatial/base-map/sheffield_wp5_wfs_polygons.gpkg"
DEFAULT_WFS_LAYER = "polygons_in_buffers"
DEFAULT_OUTPUT_DIR = "/data/sheffield/spatial/base-map/tmp/wfs_merge_point_large_parcel_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare seed-point large parcel assembly candidates and QA layers."
    )
    parser.add_argument("--wfs-gpkg", default=DEFAULT_WFS_GPKG)
    parser.add_argument("--wfs-layer", default=DEFAULT_WFS_LAYER)
    parser.add_argument("--seed-fid", type=int, default=None)
    parser.add_argument("--point-x", type=float, default=None)
    parser.add_argument("--point-y", type=float, default=None)
    parser.add_argument("--point-buffer", type=float, default=2.0)
    parser.add_argument("--local-buffer", type=float, default=180.0)
    parser.add_argument("--min-shared-edge", type=float, default=0.2)
    parser.add_argument("--max-group-size", type=int, default=30)
    parser.add_argument("--max-group-area", type=float, default=12000.0)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--top-frontier", type=int, default=18)
    parser.add_argument("--max-candidates", type=int, default=30000)
    parser.add_argument("--refine-top-groups", type=int, default=40)
    parser.add_argument("--refine-max-steps", type=int, default=24)
    parser.add_argument("--refine-frontier", type=int, default=24)
    parser.add_argument("--pocket-max-area", type=float, default=100.0)
    parser.add_argument("--pocket-min-shared", type=float, default=1.0)
    parser.add_argument("--pocket-max-frontier", type=int, default=18)
    parser.add_argument("--pocket-exclusion-depth", type=int, default=2)
    parser.add_argument("--boundary-exclusion-depth", type=int, default=1)
    parser.add_argument("--manual-positive-fid-groups", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--top-candidates", type=int, default=500)
    return parser.parse_args()


def _write_layer(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    if gdf.empty:
        return
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")


def main() -> None:
    args = parse_args()
    wfs_gpkg = Path(args.wfs_gpkg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_fid is None:
        if args.point_x is None or args.point_y is None:
            raise SystemExit("Provide either --seed-fid or both --point-x/--point-y.")
        seed_fid = read_seed_by_point(
            wfs_gpkg=wfs_gpkg,
            wfs_layer=str(args.wfs_layer),
            point_x=float(args.point_x),
            point_y=float(args.point_y),
            point_buffer=float(args.point_buffer),
        )
    else:
        seed_fid = int(args.seed_fid)

    manual_groups = parse_fid_groups(args.manual_positive_fid_groups)
    output_name = args.output_name or f"point_large_parcel_seed_{seed_fid}.gpkg"
    output_gpkg = output_dir / output_name
    output_csv = output_gpkg.with_suffix(".candidates.csv")
    summary_path = output_gpkg.with_suffix(".summary.json")
    if output_gpkg.exists():
        output_gpkg.unlink()

    candidates, context = build_point_large_parcel_candidates(
        wfs_gpkg=wfs_gpkg,
        wfs_layer=str(args.wfs_layer),
        seed_fid=seed_fid,
        local_buffer=float(args.local_buffer),
        min_shared_edge=float(args.min_shared_edge),
        max_group_size=int(args.max_group_size),
        max_group_area=float(args.max_group_area),
        beam_width=int(args.beam_width),
        top_frontier=int(args.top_frontier),
        max_candidates=int(args.max_candidates),
        refine_top_groups=int(args.refine_top_groups),
        refine_max_steps=int(args.refine_max_steps),
        refine_frontier=int(args.refine_frontier),
        pocket_max_area=float(args.pocket_max_area),
        pocket_min_shared=float(args.pocket_min_shared),
        pocket_max_frontier=int(args.pocket_max_frontier),
        pocket_exclusion_depth=int(args.pocket_exclusion_depth),
        boundary_exclusion_depth=int(args.boundary_exclusion_depth),
        manual_positive_groups=manual_groups,
        include_labels=True,
    )
    if candidates.empty:
        raise RuntimeError("No point-large-parcel candidates were generated.")

    log(f"[INFO] Writing {output_gpkg}")
    candidates.drop(columns="geometry").to_csv(output_csv, index=False)

    seed = context.local[context.local["wfs_fid"].eq(seed_fid)].copy()
    _write_layer(seed, output_gpkg, "seed_polygon")
    _write_layer(context.local.sort_values("wfs_fid"), output_gpkg, "local_polygons")

    edges = context.edges.copy()
    if not edges.empty:
        edges_gdf = gpd.GeoDataFrame(edges, geometry=[context.seed_geometry] * len(edges), crs=context.local.crs)
        _write_layer(edges_gdf, output_gpkg, "local_shared_edges_table")

    top = candidates.sort_values(["label", "heuristic_score"], ascending=[False, False]).head(
        int(args.top_candidates)
    )
    _write_layer(top, output_gpkg, "candidate_groups_top")

    positives = candidates[candidates["label"].eq(1)].copy()
    _write_layer(positives, output_gpkg, "manual_positive_groups")
    if manual_groups:
        first_manual = set(int(fid) for fid in manual_groups[0])
        boundary = boundary_neighbors_for_group(first_manual, context)
        _write_layer(boundary, output_gpkg, "manual_positive_boundary_neighbors")

    summary = {
        "wfs_gpkg": str(wfs_gpkg),
        "wfs_layer": str(args.wfs_layer),
        "seed_fid": int(seed_fid),
        "local_buffer": float(args.local_buffer),
        "min_shared_edge": float(args.min_shared_edge),
        "max_group_size": int(args.max_group_size),
        "max_group_area": float(args.max_group_area),
        "beam_width": int(args.beam_width),
        "top_frontier": int(args.top_frontier),
        "max_candidates": int(args.max_candidates),
        "refine_top_groups": int(args.refine_top_groups),
        "refine_max_steps": int(args.refine_max_steps),
        "refine_frontier": int(args.refine_frontier),
        "pocket_max_area": float(args.pocket_max_area),
        "pocket_min_shared": float(args.pocket_min_shared),
        "pocket_max_frontier": int(args.pocket_max_frontier),
        "pocket_exclusion_depth": int(args.pocket_exclusion_depth),
        "boundary_exclusion_depth": int(args.boundary_exclusion_depth),
        "local_polygons": int(len(context.local)),
        "local_shared_edges": int(len(context.edges)),
        "candidate_groups": int(len(candidates)),
        "label_counts": {str(k): int(v) for k, v in candidates["label"].value_counts().to_dict().items()},
        "label_source_counts": {
            str(k): int(v) for k, v in candidates["label_source"].value_counts().to_dict().items()
        },
        "top_10_candidates": candidates.sort_values("heuristic_score", ascending=False)
        .head(10)[
            [
                "candidate_fids",
                "group_size",
                "group_area",
                "heuristic_score",
                "regularity_gain_vs_seed",
                "hull_gap_reduction_vs_seed",
                "boundary_simplification",
                "hard_boundary_ratio",
                "label",
                "label_source",
            ]
        ]
        .to_dict("records"),
        "output_gpkg": str(output_gpkg),
        "output_csv": str(output_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[INFO] Wrote candidates CSV {output_csv}")
    log(f"[INFO] Wrote summary {summary_path}")


if __name__ == "__main__":
    main()

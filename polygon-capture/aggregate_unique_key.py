from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

from _core.config import add_config_argument, get_config_section_from_argv, require_configured
from _core.io import load_layer


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv("aggregate_unique_key", include_package_defaults=True)
    parser = argparse.ArgumentParser(
        description="Collapse capture output to one parent record per unique_key.",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--input-gpkg")
    parser.add_argument("--input-layer")
    parser.add_argument("--output-gpkg")
    parser.add_argument("--output-layer")
    parser.add_argument(
        "--union-children-when-parent-missing-or-failed",
        action=argparse.BooleanOptionalAction,
        help="If a multi-record group has no successful parent row, use successful child geometry union.",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(args, ("input_gpkg", "output_gpkg"), "aggregate_unique_key")
    return args


def _is_success(row) -> bool:
    value = row.get("capture_success", False)
    try:
        return bool(value)
    except Exception:
        return False


def _valid_geom(geom) -> bool:
    return geom is not None and not geom.is_empty


def _parent_mask(group: gpd.GeoDataFrame):
    return group["variant_key"].astype(str).eq(group["unique_key"].astype(str))


def _child_variant_keys(group: gpd.GeoDataFrame) -> str:
    mask = ~_parent_mask(group)
    return ",".join(group.loc[mask, "variant_key"].astype(str).tolist())


def _successful_child_union(group: gpd.GeoDataFrame):
    children = group.loc[~_parent_mask(group)].copy()
    children = children[children.apply(_is_success, axis=1)]
    geoms = [geom for geom in children.geometry.tolist() if _valid_geom(geom)]
    if not geoms:
        return None
    return geoms[0] if len(geoms) == 1 else unary_union(geoms)


def aggregate(gdf: gpd.GeoDataFrame, *, union_children_when_parent_missing_or_failed: bool) -> gpd.GeoDataFrame:
    rows = []
    geom_name = gdf.geometry.name
    for unique_key, group in gdf.groupby("unique_key", sort=False):
        group = group.sort_values("capture_src_id").copy()
        group_size = len(group)
        parent_rows = group.loc[_parent_mask(group)].copy()
        child_success_count = int((~_parent_mask(group) & group.apply(_is_success, axis=1)).sum())

        if group_size == 1:
            selected = group.iloc[0].copy()
            method = "single_record"
        elif not parent_rows.empty:
            selected = parent_rows.iloc[0].copy()
            method = "parent_record"
            if "capture_stage" in selected and "linked_parent_union" in str(selected["capture_stage"]):
                method = "parent_linked_union_record"
            if (
                union_children_when_parent_missing_or_failed
                and (not _is_success(selected) or not _valid_geom(selected[geom_name]))
            ):
                child_union = _successful_child_union(group)
                if _valid_geom(child_union):
                    selected[geom_name] = child_union
                    selected["capture_stage"] = "aggregate_successful_child_union"
                    selected["capture_success"] = True
                    method = "successful_child_union_parent_failed"
        else:
            successful = group[group.apply(_is_success, axis=1)].copy()
            selected = (successful if not successful.empty else group).iloc[0].copy()
            method = "first_success_record_no_parent" if not successful.empty else "first_record_no_parent"
            if union_children_when_parent_missing_or_failed:
                child_union = _successful_child_union(group)
                if _valid_geom(child_union):
                    selected[geom_name] = child_union
                    selected["capture_stage"] = "aggregate_successful_child_union"
                    selected["capture_success"] = True
                    method = "successful_child_union_no_parent"

        selected["aggregation_method"] = method
        selected["aggregation_group_size"] = int(group_size)
        selected["aggregation_child_count"] = int(max(group_size - len(parent_rows), 0))
        selected["aggregation_child_success_count"] = child_success_count
        selected["aggregation_child_variant_keys"] = _child_variant_keys(group)
        rows.append(selected)

    out = gpd.GeoDataFrame(rows, geometry=geom_name, crs=gdf.crs)
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    gdf = load_layer(args.input_gpkg, args.input_layer)
    if "unique_key" not in gdf.columns:
        raise ValueError("Input must contain unique_key.")
    if "variant_key" not in gdf.columns:
        raise ValueError("Input must contain variant_key.")
    if "capture_src_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["capture_src_id"] = gdf.index + 1

    out = aggregate(
        gdf,
        union_children_when_parent_missing_or_failed=args.union_children_when_parent_missing_or_failed,
    )
    out_path = Path(args.output_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    out.to_file(out_path, layer=args.output_layer, driver="GPKG")

    print(f"[DONE] Wrote {len(out)} parent records to: {out_path} (layer={args.output_layer})")
    print(f"[INFO] Input rows: {len(gdf)}")
    print(f"[INFO] Unique keys: {gdf['unique_key'].nunique()}")
    print("[INFO] Aggregation methods:")
    print(out["aggregation_method"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

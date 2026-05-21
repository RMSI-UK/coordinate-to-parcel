from __future__ import annotations

from typing import Dict, Sequence, Set, Tuple

import geopandas as gpd
from shapely.ops import unary_union

MAX_MERGE_AREA_M2 = 2000.0


def _validate_input(gdf: gpd.GeoDataFrame, theme_field: str) -> None:
    if theme_field not in gdf.columns:
        raise ValueError(f"Theme field not found: {theme_field}")
    if gdf.geometry.name not in gdf.columns:
        raise ValueError("Geometry column is missing.")


def resolve_theme_field(gdf: gpd.GeoDataFrame, preferred: str = "Theme") -> str:
    if preferred in gdf.columns:
        return preferred
    lower_to_original = {str(col).lower(): str(col) for col in gdf.columns}
    for candidate in ("theme", preferred.lower()):
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(f"Theme field not found. Available columns: {list(gdf.columns)}")


def filter_wfs_theme_features(
    gdf: gpd.GeoDataFrame,
    theme_field: str | None = None,
    include_terms: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    use_theme_field = theme_field or resolve_theme_field(gdf)
    terms = tuple(term.strip().lower() for term in (include_terms or ("land", "building")) if term.strip())
    if not terms:
        return gdf.iloc[0:0].copy()
    theme = gdf[use_theme_field].fillna("").astype(str).str.lower()
    keep_mask = theme.eq("__capture_no_theme_match__")
    for term in terms:
        keep_mask = keep_mask | theme.str.contains(term, regex=False)
    return gdf.loc[keep_mask].copy()


def _select_buildings_and_lands(
    gdf: gpd.GeoDataFrame, theme_field: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    theme = gdf[theme_field].fillna("").astype(str)
    building_mask = theme.str.contains("building", case=False, regex=False)
    land_mask = theme.str.contains("land", case=False, regex=False) & (~building_mask)

    buildings = gdf.loc[building_mask, ["__fid", "geometry"]].copy().rename(columns={"__fid": "building_fid"})
    lands = gdf.loc[land_mask, ["__fid", "geometry"]].copy().rename(columns={"__fid": "land_fid"})
    return buildings, lands


def _filter_merge_candidates_by_area(gdf: gpd.GeoDataFrame, max_area_m2: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    is_polygon = gdf.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
    return gdf.loc[is_polygon & (gdf.geometry.area <= float(max_area_m2))].copy()


def _edge_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    gtype = geom.geom_type
    if gtype in {"LineString", "LinearRing"}:
        return 1 if geom.length > 0 else 0
    if gtype == "MultiLineString":
        return sum(1 for g in geom.geoms if g.length > 0)
    if gtype == "GeometryCollection":
        return sum(_edge_count(g) for g in geom.geoms)
    return 0


def _assign_lands_to_buildings(lands: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if lands.empty or buildings.empty:
        return gpd.GeoDataFrame(
            columns=["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]
        )

    candidates = gpd.sjoin(lands, buildings, how="left", predicate="intersects")
    candidates = candidates.dropna(subset=["building_fid"]).copy()
    if candidates.empty:
        return gpd.GeoDataFrame(
            columns=["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]
        )

    candidates["building_fid"] = candidates["building_fid"].astype(int)
    land_geom: Dict[int, object] = lands.set_index("land_fid")["geometry"].to_dict()
    building_geom: Dict[int, object] = buildings.set_index("building_fid")["geometry"].to_dict()

    def shared_metrics(row: gpd.pd.Series) -> Tuple[float, int]:
        lg = land_geom[int(row["land_fid"])]
        bg = building_geom[int(row["building_fid"])]
        inter = lg.boundary.intersection(bg.boundary)
        return inter.length, _edge_count(inter)

    metrics = candidates.apply(shared_metrics, axis=1, result_type="expand")
    metrics.columns = ["shared_edge_len", "shared_edge_count"]
    candidates[["shared_edge_len", "shared_edge_count"]] = metrics
    candidates = candidates[candidates["shared_edge_len"] > 0].copy()
    if candidates.empty:
        return gpd.GeoDataFrame(
            columns=["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]
        )

    candidates["len_max"] = candidates.groupby("land_fid")["shared_edge_len"].transform("max")
    candidates["cnt_max"] = candidates.groupby("land_fid")["shared_edge_count"].transform("max")
    candidates["len_norm"] = candidates["shared_edge_len"] / candidates["len_max"].replace(0, 1)
    candidates["cnt_norm"] = candidates["shared_edge_count"] / candidates["cnt_max"].replace(0, 1)
    candidates["score"] = 0.5 * candidates["len_norm"] + 0.5 * candidates["cnt_norm"]

    assigned = (
        candidates.sort_values(
            ["land_fid", "score", "shared_edge_len", "shared_edge_count", "building_fid"],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates(subset=["land_fid"], keep="first")
        [["land_fid", "building_fid", "shared_edge_len", "shared_edge_count", "score"]]
    )
    return assigned


def _merge_geometries(gdf: gpd.GeoDataFrame, assigned: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, int]:
    if assigned.empty:
        return gdf.copy(), 0

    grouped_land_ids = assigned.groupby("building_fid")["land_fid"].apply(list)
    all_geoms = gdf.set_index("__fid")["geometry"]
    merged_building_geoms: Dict[int, object] = {}
    for bfid, lfids in grouped_land_ids.items():
        geoms = [all_geoms.loc[bfid]] + [all_geoms.loc[lid] for lid in lfids]
        merged_building_geoms[int(bfid)] = unary_union(geoms)

    assigned_land_ids = set(int(v) for v in assigned["land_fid"].tolist())
    out = gdf[~gdf["__fid"].isin(assigned_land_ids)].copy()
    update_mask = out["__fid"].isin(merged_building_geoms.keys())
    out.loc[update_mask, "geometry"] = out.loc[update_mask, "__fid"].map(merged_building_geoms)
    return out, len(merged_building_geoms)


def _assign_independent_buildings_to_original_lands(
    lands: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    assigned_land_to_building: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if lands.empty or buildings.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    building_ids_all: Set[int] = set(int(v) for v in buildings["building_fid"].tolist())
    building_ids_updated: Set[int] = set()
    if not assigned_land_to_building.empty:
        building_ids_updated = set(int(v) for v in assigned_land_to_building["building_fid"].tolist())

    independent_building_ids = building_ids_all - building_ids_updated
    if not independent_building_ids:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    independent_buildings = buildings[buildings["building_fid"].isin(independent_building_ids)][
        ["building_fid", "geometry"]
    ].copy()
    if independent_buildings.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    candidates = gpd.sjoin(independent_buildings, lands, how="left", predicate="intersects")
    candidates = candidates.dropna(subset=["land_fid"]).copy()
    if candidates.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    candidates["land_fid"] = candidates["land_fid"].astype(int)
    bgeom = independent_buildings.set_index("building_fid")["geometry"].to_dict()
    lgeom = lands.set_index("land_fid")["geometry"].to_dict()

    def shared_edge_count(row: gpd.pd.Series) -> int:
        bg = bgeom[int(row["building_fid"])]
        lg = lgeom[int(row["land_fid"])]
        inter = bg.boundary.intersection(lg.boundary)
        return _edge_count(inter)

    candidates["shared_edge_count"] = candidates.apply(shared_edge_count, axis=1)
    candidates = candidates[candidates["shared_edge_count"] > 0].copy()
    if candidates.empty:
        return gpd.GeoDataFrame(columns=["building_fid", "land_fid", "shared_edge_count"])

    return (
        candidates.sort_values(["building_fid", "shared_edge_count", "land_fid"], ascending=[True, False, True])
        .drop_duplicates(subset=["building_fid"], keep="first")[["building_fid", "land_fid", "shared_edge_count"]]
    )


def _merge_independent_buildings_using_original_land_targets(
    gdf_after_step1: gpd.GeoDataFrame,
    assign_building_to_land_original: gpd.GeoDataFrame,
    assigned_land_to_building_step1: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, int, int]:
    if gdf_after_step1.empty or assign_building_to_land_original.empty:
        return gdf_after_step1, 0, 0

    out = gdf_after_step1.copy()
    out_ids = set(int(v) for v in out["__fid"].tolist())
    land_to_building_step1: Dict[int, int] = {}
    if not assigned_land_to_building_step1.empty:
        land_to_building_step1 = {
            int(row["land_fid"]): int(row["building_fid"])
            for _, row in assigned_land_to_building_step1[["land_fid", "building_fid"]].iterrows()
        }

    mappings = []
    for _, row in assign_building_to_land_original.iterrows():
        building_fid = int(row["building_fid"])
        land_fid = int(row["land_fid"])
        if building_fid not in out_ids:
            continue

        target_fid = None
        if land_fid in out_ids:
            target_fid = land_fid
        elif land_fid in land_to_building_step1 and land_to_building_step1[land_fid] in out_ids:
            target_fid = land_to_building_step1[land_fid]
        if target_fid is None or target_fid == building_fid:
            continue
        mappings.append((building_fid, target_fid))

    if not mappings:
        return out, 0, 0

    map_df = gpd.pd.DataFrame(mappings, columns=["building_fid", "target_fid"]).drop_duplicates()
    grouped_buildings = map_df.groupby("target_fid")["building_fid"].apply(list)
    all_geoms = out.set_index("__fid")["geometry"]

    new_target_geoms: Dict[int, object] = {}
    for target_fid, bfids in grouped_buildings.items():
        geoms = [all_geoms.loc[int(target_fid)]] + [all_geoms.loc[int(b)] for b in bfids]
        new_target_geoms[int(target_fid)] = unary_union(geoms)

    removed_building_ids = set(int(v) for v in map_df["building_fid"].tolist())
    out = out[~out["__fid"].isin(removed_building_ids)].copy()
    update_mask = out["__fid"].isin(new_target_geoms.keys())
    out.loc[update_mask, "geometry"] = out.loc[update_mask, "__fid"].map(new_target_geoms)
    return out, len(removed_building_ids), len(new_target_geoms)


def build_wfs_merge_gdf(
    os_wfs_gdf: gpd.GeoDataFrame,
    theme_field: str = "Theme",
    include_terms: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    gdf = os_wfs_gdf.copy()
    use_theme_field = resolve_theme_field(gdf, preferred=theme_field)
    _validate_input(gdf, use_theme_field)
    gdf = filter_wfs_theme_features(gdf, use_theme_field, include_terms=include_terms)
    gdf["__fid"] = gdf.index
    buildings, lands = _select_buildings_and_lands(gdf, use_theme_field)
    buildings = _filter_merge_candidates_by_area(buildings, MAX_MERGE_AREA_M2)
    lands = _filter_merge_candidates_by_area(lands, MAX_MERGE_AREA_M2)
    assigned = _assign_lands_to_buildings(lands, buildings)
    out, _ = _merge_geometries(gdf, assigned)
    independent = _assign_independent_buildings_to_original_lands(lands, buildings, assigned)
    out, _, _ = _merge_independent_buildings_using_original_land_targets(out, independent, assigned)
    out = out.drop(columns=["__fid"])
    return out

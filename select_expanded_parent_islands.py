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
EPS = 1e-9


def choose_layer(path: str, layer: Optional[str]) -> str:
    if layer:
        return layer
    layers = pyogrio.list_layers(path)
    if len(layers) == 0:
        raise ValueError(f"No layers found in {path}")
    return str(layers[0][0])


def clean_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_layer(
    path: str,
    layer: Optional[str] = None,
    columns: Optional[list[str]] = None,
    fid_as_index: bool = False,
) -> gpd.GeoDataFrame:
    layer_name = choose_layer(path, layer)
    kwargs = {"layer": layer_name, "engine": "pyogrio", "fid_as_index": fid_as_index}
    if columns is not None:
        kwargs["columns"] = columns
    gdf = gpd.read_file(path, **kwargs)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs).upper() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
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


def load_wfs_polygons(
    path: str,
    layer: Optional[str],
    theme_field: str,
    terms: Optional[tuple[str, ...]] = None,
) -> gpd.GeoDataFrame:
    preferred_columns = [
        "GmlID",
        "OBJECTID",
        "TOID",
        theme_field,
        "CalculatedAreaValue",
        "Shape_Area",
    ]
    try:
        gdf = load_layer(path, layer, columns=preferred_columns)
    except Exception:
        gdf = load_layer(path, layer)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if terms is not None:
        gdf = filter_theme(gdf, theme_field, terms)
    gdf = gdf.reset_index(drop=True)
    gdf["__poly_id__"] = gdf.index.astype("int64")
    gdf["__poly_area__"] = gdf.geometry.area.astype(float)
    return gdf


def load_raw_supplement_polygons(
    path: str,
    layer: Optional[str],
    theme_field: str,
) -> gpd.GeoDataFrame:
    preferred_columns = [
        "GmlID",
        "OBJECTID",
        "TOID",
        theme_field,
        "CalculatedAreaValue",
        "Shape_Area",
        "DescriptiveGroup",
    ]
    try:
        gdf = load_layer(path, layer, columns=preferred_columns, fid_as_index=True)
    except Exception:
        gdf = load_layer(path, layer, fid_as_index=True)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["raw_fid"] = gdf.index.astype("int64")
    gdf = gdf.reset_index(drop=True)
    gdf["__raw_area__"] = gdf.geometry.area.astype(float)
    return gdf


def polygon_parts(geom) -> list[object]:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return []
    if not geom.is_valid:
        geom = geom.buffer(0)
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom_type == "GeometryCollection":
        parts: list[object] = []
        for child in geom.geoms:
            parts.extend(polygon_parts(child))
        return parts
    return []


def add_point_keys(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.copy()
    out["_parent_key"] = out["oachargeid"].map(clean_key)
    out["_variant_key"] = out["oachargeid_sub"].map(clean_key)
    out["_is_child"] = out["_variant_key"].ne(out["_parent_key"])
    out["_point_id"] = range(len(out))
    out["_similarity"] = pd.to_numeric(out.get("api_address_similarity", 0), errors="coerce").fillna(0.0)
    out["_parity"] = out.get("address_range_parity", "").fillna("").astype(str).str.strip().str.lower()
    out["_cluster_id"] = -1
    return out


def expanded_parent_ids(points: gpd.GeoDataFrame) -> list[str]:
    keyed = add_point_keys(points)
    child_counts = keyed.groupby("_parent_key")["_is_child"].sum()
    return child_counts[child_counts > 0].index.astype(str).tolist()


def assign_point_clusters(point_group: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    out = point_group.copy()
    out["_cluster_id"] = -1
    valid = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if valid.empty:
        return out

    cluster_source = valid[valid["_is_child"]].copy()
    if cluster_source.empty:
        cluster_source = valid

    source_indices = list(cluster_source.index)
    coords = [(cluster_source.at[idx, "geometry"].x, cluster_source.at[idx, "geometry"].y) for idx in source_indices]
    neighbors: dict[int, list[int]] = {idx: [] for idx in source_indices}
    threshold_sq = float(distance) * float(distance)
    for pos, idx in enumerate(source_indices):
        x1, y1 = coords[pos]
        for other_pos in range(pos + 1, len(source_indices)):
            other_idx = source_indices[other_pos]
            x2, y2 = coords[other_pos]
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= threshold_sq:
                neighbors[idx].append(other_idx)
                neighbors[other_idx].append(idx)

    cluster_id = 0
    seen: set[int] = set()
    labels: dict[int, int] = {}
    for idx in source_indices:
        if idx in seen:
            continue
        cluster_id += 1
        stack = [idx]
        seen.add(idx)
        labels[idx] = cluster_id
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                labels[neighbor] = cluster_id
                stack.append(neighbor)

    for idx, label in labels.items():
        out.at[idx, "_cluster_id"] = int(label)

    labelled = out[out["_cluster_id"].astype(int).gt(0)].copy()
    if labelled.empty:
        return out
    for idx, row in valid[~valid.index.isin(labelled.index)].iterrows():
        nearest_label = -1
        nearest_distance = float("inf")
        point = row.geometry
        for labelled_idx, labelled_row in labelled.iterrows():
            distance_m = float(point.distance(labelled_row.geometry))
            if distance_m < nearest_distance:
                nearest_distance = distance_m
                nearest_label = int(labelled_row["_cluster_id"])
        out.at[idx, "_cluster_id"] = nearest_label
    return out


def pick_smallest_intersections(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    source: str,
    theme_field: str,
) -> gpd.GeoDataFrame:
    if points.empty or polygons.empty:
        return gpd.GeoDataFrame(columns=list(points.columns) + ["capture_source"], geometry="geometry", crs=TARGET_CRS)

    joined = gpd.sjoin(
        points[["_point_id", "geometry"]],
        polygons[["__poly_area__", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return gpd.GeoDataFrame(columns=list(points.columns) + ["capture_source"], geometry="geometry", crs=TARGET_CRS)

    chosen = (
        joined.sort_values(["_point_id", "__poly_area__", "index_right"], ascending=[True, True, True])
        .drop_duplicates("_point_id", keep="first")
        .copy()
    )
    source_rows = points.set_index("_point_id", drop=False).loc[chosen["_point_id"].astype(int)].copy()
    source_rows.index = chosen.index
    poly = polygons.loc[chosen["index_right"].astype(int)].copy()
    poly.index = chosen.index

    source_rows["capture_source"] = source
    source_rows["capture_polygon_area_m2"] = poly.geometry.area.astype(float).values
    source_rows["capture_polygon_id"] = poly["__poly_id__"].astype(str).values
    source_rows["capture_polygon_toid"] = poly["TOID"].fillna("").astype(str).values if "TOID" in poly.columns else ""
    source_rows["capture_polygon_gmlid"] = poly["GmlID"].fillna("").astype(str).values if "GmlID" in poly.columns else ""
    source_rows["capture_polygon_theme"] = poly[theme_field].fillna("").astype(str).values if theme_field in poly.columns else ""
    source_rows["geometry"] = poly.geometry.values
    return gpd.GeoDataFrame(source_rows.reset_index(drop=True), geometry="geometry", crs=polygons.crs)


def capture_points(
    points: gpd.GeoDataFrame,
    merge_wfs: gpd.GeoDataFrame,
    raw_wfs: Optional[gpd.GeoDataFrame],
    theme_field: str,
) -> tuple[gpd.GeoDataFrame, set[int]]:
    merge_rows = pick_smallest_intersections(points, merge_wfs, "wfs_merge", theme_field)
    matched = set(merge_rows["_point_id"].astype(int).tolist()) if not merge_rows.empty else set()

    rows = [merge_rows]
    if raw_wfs is not None:
        remaining = points[~points["_point_id"].isin(matched)].copy()
        raw_rows = pick_smallest_intersections(remaining, raw_wfs, "wfs_raw", theme_field)
        raw_matched = set(raw_rows["_point_id"].astype(int).tolist()) if not raw_rows.empty else set()
        matched |= raw_matched
        rows.append(raw_rows)

    capture = pd.concat(rows, ignore_index=True)
    capture = gpd.GeoDataFrame(capture, geometry="geometry", crs=TARGET_CRS)
    if not capture.empty:
        capture = capture.sort_values(["_parent_key", "_point_id", "capture_source"]).reset_index(drop=True)
    return capture, matched


def assign_capture_clusters(
    points: gpd.GeoDataFrame,
    capture: gpd.GeoDataFrame,
    cluster_gap: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    point_out = points.copy()
    capture_out = capture.copy()
    point_out["_cluster_id"] = -1
    if capture_out.empty:
        return (
            point_out,
            capture_out,
            empty_layer(
                [
                    "oachargeid",
                    "cluster_id",
                    "capture_rows",
                    "point_rows",
                    "child_rows",
                    "source_mix",
                    "capture_polygon_ids",
                    "area_m2",
                    "part_count",
                ]
            ),
        )

    capture_out["_cluster_id"] = -1
    cluster_rows: list[dict] = []
    gap = max(float(cluster_gap), 0.0)

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
                if gap > 0:
                    if float(geom.distance(other_geom)) > gap:
                        continue
                elif not geom.intersects(other_geom):
                    continue
                neighbors[int(idx)].append(other_idx)

        seen: set[int] = set()
        cluster_id = 0
        for idx in valid.index:
            idx = int(idx)
            if idx in seen:
                continue
            cluster_id += 1
            stack = [idx]
            component: list[int] = []
            seen.add(idx)
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
            point_ids = component_rows["_point_id"].astype(int).tolist()
            point_out.loc[point_out["_point_id"].astype(int).isin(point_ids), "_cluster_id"] = int(cluster_id)
            geom = unary_union(list(component_rows.geometry))
            polygon_ids = ",".join(sorted(component_rows["capture_polygon_id"].fillna("").astype(str).unique()))
            source_mix = ",".join(sorted(component_rows["capture_source"].fillna("").astype(str).unique()))
            cluster_rows.append(
                {
                    "oachargeid": parent_key,
                    "cluster_id": int(cluster_id),
                    "capture_rows": int(len(component_rows)),
                    "point_rows": int(component_rows["_point_id"].nunique()),
                    "child_rows": int(component_rows.drop_duplicates("_point_id")["_is_child"].astype(bool).sum()),
                    "source_mix": source_mix,
                    "capture_polygon_ids": polygon_ids,
                    "area_m2": float(geom.area),
                    "part_count": int(part_count(geom)),
                    "geometry": geom,
                }
            )

    cluster_gdf = (
        gpd.GeoDataFrame(cluster_rows, geometry="geometry", crs=TARGET_CRS)
        if cluster_rows
        else empty_layer(
            [
                "oachargeid",
                "cluster_id",
                "capture_rows",
                "point_rows",
                "child_rows",
                "source_mix",
                "capture_polygon_ids",
                "area_m2",
                "part_count",
            ]
        )
    )
    return point_out, capture_out, cluster_gdf


def point_table_for_output(points: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [
        "_point_id",
        "_parent_key",
        "_variant_key",
        "_is_child",
        "_similarity",
        "_parity",
        "address_clarity",
        "address_number",
        "address_road",
        "postcode",
        "geocoding_confidence",
    ]
    existing = [col for col in cols if col in points.columns]
    return pd.DataFrame(points[existing].drop(columns=[], errors="ignore"))


def explode_capture_parts(capture_group: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows: list[dict] = []
    for capture_index, row in capture_group.reset_index(drop=True).iterrows():
        for part_index, part in enumerate(polygon_parts(row.geometry)):
            record = row.drop(labels=["geometry"]).to_dict()
            record["_capture_index"] = int(capture_index)
            record["_capture_part_index"] = int(part_index)
            record["geometry"] = part
            rows.append(record)
    if not rows:
        return gpd.GeoDataFrame(columns=list(capture_group.columns) + ["_capture_index"], geometry="geometry", crs=TARGET_CRS)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)


def part_count(geom) -> int:
    return len(polygon_parts(geom))


def parse_exclude_terms(value: str) -> tuple[str, ...]:
    return tuple(term.strip().lower() for term in value.split(",") if term.strip())


def supplement_excluded(row: pd.Series, theme_field: str, exclude_terms: tuple[str, ...]) -> bool:
    if not exclude_terms:
        return False
    haystack = " ".join(
        str(row.get(column, "") or "").lower()
        for column in [theme_field, "DescriptiveGroup", "DescriptiveTerm"]
    )
    return any(term in haystack for term in exclude_terms)


def raw_candidates_for_geom(raw_supplement: gpd.GeoDataFrame, geom, hull_buffer: float) -> gpd.GeoDataFrame:
    if raw_supplement is None or raw_supplement.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=TARGET_CRS)
    hull = geom.convex_hull.buffer(float(hull_buffer))
    idx = list(raw_supplement.sindex.query(hull, predicate="intersects"))
    if not idx:
        return raw_supplement.iloc[0:0].copy()
    candidates = raw_supplement.iloc[idx].copy()
    candidates = candidates[candidates.geometry.intersects(hull)].copy()
    return candidates


def candidate_metrics(candidate_geom, current_geom, initial_hull) -> dict[str, float]:
    if not candidate_geom.is_valid:
        candidate_geom = candidate_geom.buffer(0)
    raw_area = float(candidate_geom.area)
    if raw_area <= EPS:
        return {
            "raw_area": 0.0,
            "new_area": 0.0,
            "contact_ratio": 0.0,
            "inside_hull_ratio": 0.0,
            "touched_parts": 0,
            "new_part_count": part_count(current_geom),
        }
    new_area = float(candidate_geom.difference(current_geom).area)
    boundary_len = max(float(candidate_geom.boundary.length), EPS)
    shared_len = float(candidate_geom.boundary.intersection(current_geom.boundary).length)
    current_parts = polygon_parts(current_geom)
    touched_parts = sum(1 for part in current_parts if candidate_geom.buffer(0.05).intersects(part))
    inside_hull_ratio = float(candidate_geom.intersection(initial_hull).area / raw_area)
    new_part_count = part_count(unary_union([current_geom, candidate_geom]))
    distance_to_current = float(candidate_geom.distance(current_geom))
    return {
        "raw_area": raw_area,
        "new_area": new_area,
        "contact_ratio": shared_len / boundary_len,
        "inside_hull_ratio": inside_hull_ratio,
        "touched_parts": float(touched_parts),
        "new_part_count": float(new_part_count),
        "distance_to_current": distance_to_current,
    }


def supplement_cluster_with_raw(
    base_geom,
    raw_supplement: Optional[gpd.GeoDataFrame],
    parent_key: str,
    cluster_id: int,
    args: argparse.Namespace,
) -> tuple[object, list[dict]]:
    if raw_supplement is None or raw_supplement.empty or base_geom is None or base_geom.is_empty:
        return base_geom, []

    initial_hull = base_geom.convex_hull.buffer(float(args.supplement_hull_buffer))
    candidates = raw_candidates_for_geom(raw_supplement, base_geom, float(args.supplement_hull_buffer))
    if candidates.empty:
        return base_geom, []

    exclude_terms = parse_exclude_terms(args.supplement_exclude_terms)
    current = base_geom
    used_raw_fids: set[int] = set()
    supplement_rows: list[dict] = []
    total_new_area = 0.0
    max_total_area = min(
        float(args.max_supplement_total_area),
        max(float(base_geom.area) * float(args.max_supplement_total_area_ratio), float(args.max_supplement_area)),
    )

    for _ in range(int(args.max_supplements_per_cluster)):
        current_part_count = part_count(current)
        best: Optional[tuple[float, int, object, dict[str, float], pd.Series]] = None
        for candidate_index, row in candidates.iterrows():
            raw_fid = int(row.get("raw_fid", candidate_index))
            if raw_fid in used_raw_fids:
                continue
            if supplement_excluded(row, args.theme_field, exclude_terms):
                continue
            candidate_geom = row.geometry
            if candidate_geom is None or candidate_geom.is_empty:
                continue
            if not candidate_geom.is_valid:
                candidate_geom = candidate_geom.buffer(0)
            metrics = candidate_metrics(candidate_geom, current, initial_hull)
            if metrics["new_area"] < float(args.min_supplement_new_area):
                continue
            if metrics["raw_area"] > float(args.max_supplement_area):
                continue
            if total_new_area + metrics["new_area"] > max_total_area:
                continue
            if metrics["inside_hull_ratio"] < float(args.min_supplement_hull_ratio):
                continue
            if metrics["new_part_count"] > current_part_count:
                continue

            connects = metrics["new_part_count"] < current_part_count or metrics["touched_parts"] >= 2
            enclosed = metrics["contact_ratio"] >= float(args.min_supplement_contact_ratio)
            near_gap = (
                metrics["distance_to_current"] <= float(args.max_supplement_gap)
                and metrics["inside_hull_ratio"] >= 0.95
                and metrics["new_part_count"] <= current_part_count
            )
            if not connects and not enclosed:
                continue

            score = (
                (current_part_count - metrics["new_part_count"]) * 1000.0
                + metrics["touched_parts"] * 100.0
                + metrics["contact_ratio"] * 75.0
                + metrics["inside_hull_ratio"] * 25.0
                + min(metrics["new_area"], 400.0) * 0.5
            )
            if best is None or score > best[0]:
                best = (score, raw_fid, candidate_geom, metrics, row)

        if best is None:
            break

        score, raw_fid, candidate_geom, metrics, row = best
        reason_bits = []
        if metrics["new_part_count"] < current_part_count:
            reason_bits.append("connects_parts")
        if metrics["touched_parts"] >= 2:
            reason_bits.append("touches_multiple_parts")
        if metrics["contact_ratio"] >= float(args.min_supplement_contact_ratio):
            reason_bits.append("enclosed_contact")
        if metrics["distance_to_current"] <= float(args.max_supplement_gap):
            reason_bits.append("near_gap")
        supplement_rows.append(
            {
                "oachargeid": parent_key,
                "cluster_id": int(cluster_id),
                "raw_fid": int(raw_fid),
                "raw_theme": str(row.get(args.theme_field, "") or ""),
                "raw_descriptive_group": str(row.get("DescriptiveGroup", "") or ""),
                "raw_area_m2": float(metrics["raw_area"]),
                "new_area_m2": float(metrics["new_area"]),
                "contact_ratio": float(metrics["contact_ratio"]),
                "inside_hull_ratio": float(metrics["inside_hull_ratio"]),
                "distance_to_current": float(metrics["distance_to_current"]),
                "touched_parts": int(metrics["touched_parts"]),
                "part_count_before": int(current_part_count),
                "part_count_after": int(metrics["new_part_count"]),
                "selection_score": float(score),
                "reason": ",".join(reason_bits),
                "geometry": candidate_geom,
            }
        )
        used_raw_fids.add(raw_fid)
        total_new_area += float(metrics["new_area"])
        current = unary_union([current, candidate_geom])

    return current, supplement_rows


def metric_value(row: pd.Series, name: str, default=0):
    return row[name] if name in row.index else default


def confidence_for(row: pd.Series, missing_child_count: int) -> str:
    island_child_rows = int(metric_value(row, "island_child_rows", 0))
    matched_child_total = int(metric_value(row, "matched_child_total", 0))
    island_count = int(metric_value(row, "island_count", 0))
    coverage_ratio = float(island_child_rows / max(matched_child_total, 1))

    if island_child_rows == 0:
        return "low"
    if (
        float(metric_value(row, "same_ratio", 0.0)) >= 0.75
        and float(metric_value(row, "mean_similarity", 0.0)) >= 90
        and missing_child_count == 0
        and (island_count <= 1 or coverage_ratio >= 0.7)
    ):
        return "high"
    if (
        float(metric_value(row, "same_ratio", 0.0)) >= 0.5
        and float(metric_value(row, "mean_similarity", 0.0)) >= 80
        and (island_count <= 1 or coverage_ratio >= 0.3)
    ):
        return "medium"
    if (
        int(metric_value(row, "same_count", 0)) > int(metric_value(row, "opposite_count", 0))
        and float(metric_value(row, "mean_similarity", 0.0)) >= 75
        and coverage_ratio >= 0.3
    ):
        return "medium"
    return "low"


def build_island_rows(
    parent_key: str,
    point_group: gpd.GeoDataFrame,
    capture_group: gpd.GeoDataFrame,
    raw_supplement: Optional[gpd.GeoDataFrame],
    args: argparse.Namespace,
) -> tuple[list[dict], Optional[dict], list[dict]]:
    child_ids = set(point_group.loc[point_group["_is_child"], "_point_id"].astype(int).tolist())
    matched_ids = set(capture_group["_point_id"].astype(int).tolist()) if not capture_group.empty else set()
    matched_child_ids = child_ids & matched_ids
    missing_child_count = int(len(child_ids - matched_child_ids))

    if capture_group.empty:
        return [], None, []

    parent_row = point_group[~point_group["_is_child"]].head(1)
    parent_address_number = str(parent_row.iloc[0].get("address_number", "")) if not parent_row.empty else ""
    parent_address_clarity = str(parent_row.iloc[0].get("address_clarity", "")) if not parent_row.empty else ""
    description = str(parent_row.iloc[0].get("charge_geographic_description", "")) if not parent_row.empty else ""

    rows: list[dict] = []
    all_supplement_rows: list[dict] = []
    cluster_ids = sorted(
        int(value)
        for value in capture_group["_cluster_id"].dropna().unique().tolist()
        if int(value) > 0
    )
    if not cluster_ids:
        cluster_ids = [0]

    for cluster_id in cluster_ids:
        cluster_capture = capture_group[capture_group["_cluster_id"].astype(int).eq(cluster_id)].copy()
        if cluster_id == 0:
            cluster_capture = capture_group.copy()
        if cluster_capture.empty:
            continue

        cap_parts = explode_capture_parts(cluster_capture)
        if cap_parts.empty:
            continue

        base = unary_union(list(cap_parts.geometry))
        base_part_count = part_count(base)
        if base_part_count == 0:
            continue
        if base_part_count > 1:
            final_geom, supplement_rows = supplement_cluster_with_raw(base, raw_supplement, parent_key, cluster_id, args)
        else:
            final_geom, supplement_rows = base, []
        all_supplement_rows.extend(supplement_rows)
        islands = sorted(polygon_parts(final_geom), key=lambda geom: float(geom.area), reverse=True)
        if not islands:
            continue

        for cluster_island_id, island in enumerate(islands, start=1):
            assigned_mask = []
            for geom in cap_parts.geometry:
                intersection_area = float(geom.intersection(island).area)
                assigned_mask.append(intersection_area > EPS or geom.representative_point().intersects(island))
            island_parts = cap_parts.loc[assigned_mask].copy()
            island_points = island_parts.drop_duplicates("_point_id").copy()

            cluster_supplements = [
                item
                for item in supplement_rows
                if item["geometry"].intersects(island) or item["geometry"].representative_point().intersects(island)
            ]
            supplement_fids = ",".join(str(item["raw_fid"]) for item in cluster_supplements)
            supplement_themes = ",".join(sorted({str(item["raw_theme"]) for item in cluster_supplements if item.get("raw_theme")}))
            supplement_area = float(sum(item["new_area_m2"] for item in cluster_supplements))

            parity = island_points["_parity"].fillna("").astype(str).str.lower()
            same_count = int(parity.eq("same").sum())
            opposite_count = int(parity.eq("opposite").sum())
            blank_parity_count = int(parity.eq("").sum())
            parity_known = same_count + opposite_count
            same_ratio = float(same_count / parity_known) if parity_known else 0.0
            same_share_all = float(same_count / max(len(island_points), 1))

            similarity = pd.to_numeric(island_points["_similarity"], errors="coerce").fillna(0.0)
            mean_similarity = float(similarity.mean()) if len(similarity) else 0.0
            max_similarity = float(similarity.max()) if len(similarity) else 0.0
            matched_child_rows = int(island_points["_is_child"].astype(bool).sum())
            matched_parent_rows = int((~island_points["_is_child"].astype(bool)).sum())

            score = (
                mean_similarity
                + float(args.same_ratio_weight) * same_ratio
                + float(args.same_count_weight) * min(same_count, int(args.same_count_cap))
                + float(args.child_count_weight) * matched_child_rows
                + float(args.supplement_score_weight) * min(len(cluster_supplements), int(args.supplement_score_cap))
            )

            source_mix = ",".join(sorted(island_points["capture_source"].fillna("").astype(str).unique()))
            selected_examples = ",".join(island_points["_variant_key"].fillna("").astype(str).head(12).tolist())
            row = {
                "oachargeid": parent_key,
                "cluster_id": int(cluster_id),
                "cluster_count": int(len(cluster_ids)),
                "cluster_island_id": int(cluster_island_id),
                "base_part_count": int(base_part_count),
                "final_part_count": int(len(islands)),
                "supplement_count": int(len(cluster_supplements)),
                "supplement_area_m2": supplement_area,
                "supplement_fids": supplement_fids,
                "supplement_themes": supplement_themes,
                "selection_score": float(score),
                "point_rows": int(len(point_group)),
                "child_rows": int(len(child_ids)),
                "matched_rows_total": int(len(matched_ids)),
                "matched_child_total": int(len(matched_child_ids)),
                "missing_child_count": missing_child_count,
                "island_point_rows": int(len(island_points)),
                "island_child_rows": matched_child_rows,
                "island_parent_rows": matched_parent_rows,
                "same_count": same_count,
                "opposite_count": opposite_count,
                "blank_parity_count": blank_parity_count,
                "same_ratio": same_ratio,
                "same_share_all": same_share_all,
                "mean_similarity": mean_similarity,
                "max_similarity": max_similarity,
                "area_m2": float(island.area),
                "source_mix": source_mix,
                "variant_examples": selected_examples,
                "parent_address_number": parent_address_number,
                "parent_address_clarity": parent_address_clarity,
                "description": description,
                "geometry": island,
            }
            rows.append(row)

    if not rows:
        return [], None, all_supplement_rows

    for island_id, row in enumerate(rows, start=1):
        row["island_id"] = int(island_id)
        row["island_count"] = int(len(rows))

    ranked = sorted(
        rows,
        key=lambda item: (
            item["selection_score"],
            item["mean_similarity"],
            item["same_ratio"],
            item["same_count"],
            item["island_child_rows"],
            item["supplement_count"],
            item["area_m2"],
        ),
        reverse=True,
    )
    selected = ranked[0].copy()
    selected["selected"] = 1
    selected["auto_polygon_confidence"] = confidence_for(pd.Series(selected), missing_child_count)
    for row in rows:
        row["selected"] = 1 if row["island_id"] == selected["island_id"] else 0
        row["auto_polygon_confidence"] = confidence_for(pd.Series(row), missing_child_count)
    return rows, selected, all_supplement_rows


def empty_layer(columns: list[str], crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    data = {column: pd.Series(dtype="object") for column in columns}
    data["geometry"] = gpd.GeoSeries([], crs=crs)
    return gpd.GeoDataFrame(data, geometry="geometry", crs=crs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one WFS island for each expanded oachargeid using similarity and address_range_parity.",
    )
    parser.add_argument("--geocoding-gpkg", required=True)
    parser.add_argument("--geocoding-layer")
    parser.add_argument("--wfs-merge-gpkg", required=True)
    parser.add_argument("--wfs-merge-layer")
    parser.add_argument("--raw-wfs-gpkg")
    parser.add_argument("--raw-wfs-layer")
    parser.add_argument("--raw-supplement-gpkg")
    parser.add_argument("--raw-supplement-layer")
    parser.add_argument("--output-gpkg", required=True)
    parser.add_argument("--selected-layer", default="expanded_parent_selected_islands")
    parser.add_argument("--candidate-layer", default="expanded_parent_candidate_islands")
    parser.add_argument("--capture-layer", default="expanded_parent_point_captures")
    parser.add_argument("--capture-cluster-layer", default="expanded_parent_capture_clusters")
    parser.add_argument("--unmatched-points-layer", default="expanded_parent_unmatched_points")
    parser.add_argument("--unresolved-layer", default="expanded_parent_unresolved")
    parser.add_argument("--raw-supplement-output-layer", default="raw_supplements")
    parser.add_argument("--theme-field", default="Theme")
    parser.add_argument("--theme-include", default="building,land")
    parser.add_argument("--cluster-distance", type=float, default=35.0)
    parser.add_argument("--capture-cluster-gap", type=float, default=0.0)
    parser.add_argument("--stop-after-capture-clusters", action="store_true")
    parser.add_argument("--same-ratio-weight", type=float, default=25.0)
    parser.add_argument("--same-count-weight", type=float, default=0.5)
    parser.add_argument("--same-count-cap", type=int, default=20)
    parser.add_argument("--child-count-weight", type=float, default=0.05)
    parser.add_argument("--supplement-score-weight", type=float, default=1.0)
    parser.add_argument("--supplement-score-cap", type=int, default=8)
    parser.add_argument("--supplement-hull-buffer", type=float, default=12.0)
    parser.add_argument("--max-supplement-area", type=float, default=1200.0)
    parser.add_argument("--max-supplement-total-area", type=float, default=3000.0)
    parser.add_argument("--max-supplement-total-area-ratio", type=float, default=0.6)
    parser.add_argument("--max-supplements-per-cluster", type=int, default=12)
    parser.add_argument("--min-supplement-new-area", type=float, default=1.0)
    parser.add_argument("--min-supplement-contact-ratio", type=float, default=0.28)
    parser.add_argument("--min-supplement-hull-ratio", type=float, default=0.75)
    parser.add_argument("--max-supplement-gap", type=float, default=2.5)
    parser.add_argument("--supplement-exclude-terms", default="water,rail")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    terms = parse_terms(args.theme_include)

    geocoding_layer = choose_layer(args.geocoding_gpkg, args.geocoding_layer)
    print(f"[INFO] Reading geocoding: {args.geocoding_gpkg} (layer={geocoding_layer})")
    points = load_layer(args.geocoding_gpkg, geocoding_layer)
    points = add_point_keys(points)
    expanded_ids = expanded_parent_ids(points)
    expanded_set = set(expanded_ids)
    expanded_points = points[points["_parent_key"].isin(expanded_set)].copy()
    valid_points = expanded_points[expanded_points.geometry.notna() & ~expanded_points.geometry.is_empty].copy()
    print(
        "[INFO] expanded parents="
        f"{len(expanded_ids)} expanded rows={len(expanded_points)} valid point rows={len(valid_points)}"
    )

    merge_layer = choose_layer(args.wfs_merge_gpkg, args.wfs_merge_layer)
    print(f"[INFO] Reading merged WFS: {args.wfs_merge_gpkg} (layer={merge_layer})")
    merge_wfs = load_wfs_polygons(args.wfs_merge_gpkg, merge_layer, args.theme_field)
    print(f"[INFO] merged WFS features={len(merge_wfs)}")

    raw_wfs = None
    if args.raw_wfs_gpkg:
        raw_layer = choose_layer(args.raw_wfs_gpkg, args.raw_wfs_layer)
        print(f"[INFO] Reading raw WFS fallback: {args.raw_wfs_gpkg} (layer={raw_layer})")
        raw_wfs = load_wfs_polygons(args.raw_wfs_gpkg, raw_layer, args.theme_field, terms)
        print(f"[INFO] raw WFS eligible={len(raw_wfs)}")

    raw_supplement = None
    if args.raw_supplement_gpkg:
        raw_supplement_layer = choose_layer(args.raw_supplement_gpkg, args.raw_supplement_layer)
        print(f"[INFO] Reading raw WFS supplements: {args.raw_supplement_gpkg} (layer={raw_supplement_layer})")
        raw_supplement = load_raw_supplement_polygons(args.raw_supplement_gpkg, raw_supplement_layer, args.theme_field)
        print(f"[INFO] raw supplement candidates={len(raw_supplement)}")

    print("[INFO] Capturing expanded parent/child points to WFS polygons")
    capture, matched = capture_points(valid_points, merge_wfs, raw_wfs, args.theme_field)
    print(f"[INFO] captured rows={len(capture)} matched point rows={len(matched)}")

    expanded_points, capture, capture_clusters = assign_capture_clusters(
        expanded_points,
        capture,
        float(args.capture_cluster_gap),
    )
    cluster_count = int(capture_clusters[["oachargeid", "cluster_id"]].drop_duplicates().shape[0])
    unmatched_points = expanded_points[~expanded_points["_point_id"].astype(int).isin(matched)].copy()
    print(
        "[INFO] capture-based clusters="
        f"{cluster_count} unmatched point rows={len(unmatched_points)} gap={float(args.capture_cluster_gap):.2f}m"
    )

    output_path = Path(args.output_gpkg)
    if args.stop_after_capture_clusters:
        if output_path.exists():
            output_path.unlink()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Writing capture cluster check layers: {output_path}")
        capture.to_file(output_path, layer=args.capture_layer, driver="GPKG", engine="pyogrio")
        capture_clusters.to_file(output_path, layer=args.capture_cluster_layer, driver="GPKG", engine="pyogrio")
        unmatched_points.to_file(output_path, layer=args.unmatched_points_layer, driver="GPKG", engine="pyogrio")
        print(
            "[DONE] captures="
            f"{len(capture)} capture_clusters={len(capture_clusters)} unmatched_points={len(unmatched_points)}"
        )
        return

    selected_rows: list[dict] = []
    island_rows: list[dict] = []
    supplement_rows: list[dict] = []
    unresolved_rows: list[dict] = []
    for parent_key in expanded_ids:
        point_group = expanded_points[expanded_points["_parent_key"].eq(parent_key)].copy()
        capture_group = capture[capture["_parent_key"].eq(parent_key)].copy() if not capture.empty else capture
        rows, selected, supplements = build_island_rows(parent_key, point_group, capture_group, raw_supplement, args)
        island_rows.extend(rows)
        supplement_rows.extend(supplements)
        if selected is None:
            parent_point = point_group[~point_group["_is_child"]].head(1)
            source = parent_point if not parent_point.empty else point_group.head(1)
            geometry = source.geometry.iloc[0] if not source.empty else None
            unresolved_rows.append(
                {
                    "oachargeid": parent_key,
                    "point_rows": int(len(point_group)),
                    "child_rows": int(point_group["_is_child"].sum()),
                    "valid_point_rows": int(point_group.geometry.notna().sum()),
                    "reason": "no_wfs_island",
                    "geometry": geometry,
                }
            )
        else:
            selected_rows.append(selected)

    selected_columns = [
        "oachargeid",
        "selected",
        "auto_polygon_confidence",
        "island_id",
        "island_count",
        "cluster_id",
        "cluster_count",
        "cluster_island_id",
        "base_part_count",
        "final_part_count",
        "supplement_count",
        "supplement_area_m2",
        "supplement_fids",
        "supplement_themes",
        "selection_score",
        "point_rows",
        "child_rows",
        "matched_rows_total",
        "matched_child_total",
        "missing_child_count",
        "island_point_rows",
        "island_child_rows",
        "island_parent_rows",
        "same_count",
        "opposite_count",
        "blank_parity_count",
        "same_ratio",
        "same_share_all",
        "mean_similarity",
        "max_similarity",
        "area_m2",
        "source_mix",
        "variant_examples",
        "parent_address_number",
        "parent_address_clarity",
        "description",
    ]
    selected_gdf = (
        gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=TARGET_CRS)
        if selected_rows
        else empty_layer(selected_columns)
    )
    island_gdf = (
        gpd.GeoDataFrame(island_rows, geometry="geometry", crs=TARGET_CRS)
        if island_rows
        else empty_layer(selected_columns)
    )
    unresolved_gdf = (
        gpd.GeoDataFrame(unresolved_rows, geometry="geometry", crs=TARGET_CRS)
        if unresolved_rows
        else empty_layer(["oachargeid", "point_rows", "child_rows", "valid_point_rows", "reason"])
    )
    supplement_gdf = (
        gpd.GeoDataFrame(supplement_rows, geometry="geometry", crs=TARGET_CRS)
        if supplement_rows
        else empty_layer(
            [
                "oachargeid",
                "cluster_id",
                "raw_fid",
                "raw_theme",
                "raw_descriptive_group",
                "raw_area_m2",
                "new_area_m2",
                "contact_ratio",
                "inside_hull_ratio",
                "distance_to_current",
                "touched_parts",
                "part_count_before",
                "part_count_after",
                "selection_score",
                "reason",
            ]
        )
    )

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing {output_path}")
    capture.to_file(output_path, layer=args.capture_layer, driver="GPKG", engine="pyogrio")
    capture_clusters.to_file(output_path, layer=args.capture_cluster_layer, driver="GPKG", engine="pyogrio")
    unmatched_points.to_file(output_path, layer=args.unmatched_points_layer, driver="GPKG", engine="pyogrio")
    selected_gdf.to_file(output_path, layer=args.selected_layer, driver="GPKG", engine="pyogrio")
    island_gdf.to_file(output_path, layer=args.candidate_layer, driver="GPKG", engine="pyogrio")
    supplement_gdf.to_file(output_path, layer=args.raw_supplement_output_layer, driver="GPKG", engine="pyogrio")
    unresolved_gdf.to_file(output_path, layer=args.unresolved_layer, driver="GPKG", engine="pyogrio")

    print(
        "[DONE] selected="
        f"{len(selected_gdf)} candidate_islands={len(island_gdf)} raw_supplements={len(supplement_gdf)} unresolved={len(unresolved_gdf)}"
    )


if __name__ == "__main__":
    main()

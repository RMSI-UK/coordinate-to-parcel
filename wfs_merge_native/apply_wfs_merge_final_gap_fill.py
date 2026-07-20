#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from apply_wfs_merge_completion_model import _write_layer
from train_wfs_merge_completion_model import _shape_metrics


DEFAULT_INPUT_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/04_anchor_group_repaired.gpkg"
DEFAULT_OUTPUT_GPKG = "/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_merged_native.gpkg"
DEFAULT_UPRN_GPKG = "/data/base-data/osopenuprn_202602.gpkg"
DEFAULT_UPRN_LAYER = "osopenuprn_address"
DEFAULT_UPRN_ID_FIELD = "UPRN"
PREDICTED_LAYER = "predicted_parcels_with_uprn"
FINAL_ALIAS_LAYER = "wfs_merged_parcels"


@dataclass(frozen=True)
class FinalGapFillConfig:
    fill_output_holes: bool = True
    fill_enclosed_gap_holes: bool = True
    enclosed_gap_max_area_m2: float = 250.0
    enclosed_gap_min_shared_edge_m: float = 0.05
    min_polygon_part_area_m2: float = 0.01
    guard_enclosed_gaps_with_uprn: bool = False
    guard_output_holes_with_uprn: bool = False
    skip_occupied_output_holes: bool = False
    recompute_uprn_count_from_final_geometry: bool = True


@dataclass(frozen=True)
class UprnContext:
    points: gpd.GeoDataFrame
    id_field: str
    sindex: Any


def _log(message: str) -> None:
    print(message, flush=True)


def _make_valid_geometry(geom):
    if geom is None or shapely.is_empty(geom):
        return geom
    try:
        if shapely.is_valid(geom):
            return geom
    except Exception:
        pass
    try:
        fixed = make_valid(geom)
    except Exception:
        fixed = geom.buffer(0)
    if fixed is None or shapely.is_empty(fixed):
        return fixed
    try:
        if shapely.is_valid(fixed):
            return fixed
    except Exception:
        return fixed
    try:
        return fixed.buffer(0)
    except Exception:
        return fixed


def _extract_polygonal_geometry(geom):
    if geom is None or shapely.is_empty(geom):
        return Polygon()
    geom_type = getattr(geom, "geom_type", "")
    if geom_type in {"Polygon", "MultiPolygon"}:
        return geom
    if geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            extracted = _extract_polygonal_geometry(part)
            if extracted is None or shapely.is_empty(extracted):
                continue
            if extracted.geom_type == "Polygon":
                parts.append(extracted)
            elif extracted.geom_type == "MultiPolygon":
                parts.extend(list(extracted.geoms))
        if not parts:
            return Polygon()
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)
    return Polygon()


def _polygon_part_count(geom) -> int:
    if geom is None or shapely.is_empty(geom):
        return 0
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        return 1
    if geom_type == "MultiPolygon":
        return len(geom.geoms)
    if geom_type == "GeometryCollection":
        return sum(_polygon_part_count(part) for part in geom.geoms)
    return 0


def _drop_tiny_polygon_parts(geom, min_area: float):
    if geom is None or shapely.is_empty(geom):
        return geom
    geom = _extract_polygonal_geometry(_make_valid_geometry(geom))
    if geom is None or shapely.is_empty(geom):
        return Polygon()
    if geom.geom_type == "Polygon":
        return geom if float(geom.area) >= float(min_area) else Polygon()
    if geom.geom_type == "MultiPolygon":
        parts = [part for part in geom.geoms if float(part.area) >= float(min_area)]
        if not parts:
            return Polygon()
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)
    return Polygon()


def _as_multi_polygon(geom):
    geom = _extract_polygonal_geometry(geom)
    if geom is None or shapely.is_empty(geom):
        return Polygon()
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    return Polygon()


def _iter_polygon_parts(geom):
    if geom is None or shapely.is_empty(geom):
        return
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "Polygon":
        yield geom
    elif geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geom.geoms:
            yield from _iter_polygon_parts(part)


def _collect_interior_hole_polygons(
    geom,
    *,
    min_area: float,
    max_area: float | None = None,
) -> list[Polygon]:
    holes: list[Polygon] = []
    for part in _iter_polygon_parts(geom):
        for ring in part.interiors:
            hole = Polygon(ring)
            if hole.is_empty:
                continue
            hole = _drop_tiny_polygon_parts(hole, min_area)
            if hole is None or shapely.is_empty(hole) or hole.geom_type != "Polygon":
                continue
            area = float(hole.area)
            if area < float(min_area):
                continue
            if max_area is not None and area > float(max_area):
                continue
            holes.append(hole)
    return holes


def _fill_polygon_holes(geom, min_area: float):
    if geom is None or shapely.is_empty(geom):
        return geom
    geom = _extract_polygonal_geometry(_make_valid_geometry(geom))
    if geom is None or shapely.is_empty(geom):
        return Polygon()
    if geom.geom_type == "Polygon":
        return _drop_tiny_polygon_parts(Polygon(geom.exterior), min_area)
    if geom.geom_type == "MultiPolygon":
        polygons = [Polygon(part.exterior) for part in geom.geoms if not part.is_empty]
        return _drop_tiny_polygon_parts(MultiPolygon(polygons), min_area)
    if geom.geom_type == "GeometryCollection":
        polygons = []
        for part in geom.geoms:
            filled = _fill_polygon_holes(part, min_area)
            if filled is None or shapely.is_empty(filled):
                continue
            if filled.geom_type == "Polygon":
                polygons.append(filled)
            elif filled.geom_type == "MultiPolygon":
                polygons.extend(list(filled.geoms))
        return _drop_tiny_polygon_parts(MultiPolygon(polygons), min_area)
    return geom


def _coverage_union(geometries: list[Any]):
    try:
        return shapely.coverage_union_all(geometries)
    except Exception:
        return shapely.union_all(geometries)


def _load_uprn_context(
    uprn_gpkg: Path | None,
    uprn_layer: str,
    uprn_id_field: str,
    predicted: gpd.GeoDataFrame,
) -> UprnContext | None:
    if uprn_gpkg is None or not str(uprn_gpkg).strip():
        return None
    path = Path(uprn_gpkg)
    if not path.exists():
        raise FileNotFoundError(f"UPRN GeoPackage does not exist: {path}")
    if predicted.empty:
        return None

    bounds = tuple(float(value) for value in predicted.total_bounds)
    if len(bounds) != 4 or any(not math.isfinite(value) for value in bounds):
        return None

    _log(f"[INFO] Reading UPRN points for gap-fill guard: {path} ({uprn_layer})")
    try:
        uprn = pyogrio.read_dataframe(path, layer=uprn_layer, bbox=bounds, columns=[uprn_id_field])
    except ValueError:
        uprn = pyogrio.read_dataframe(path, layer=uprn_layer, bbox=bounds)
    if uprn.empty:
        return None
    if uprn_id_field not in uprn.columns:
        uprn[uprn_id_field] = uprn.index.astype(str)
    if predicted.crs is not None and uprn.crs is not None and str(uprn.crs) != str(predicted.crs):
        uprn = uprn.to_crs(predicted.crs)
    uprn = uprn[uprn.geometry.notna() & ~uprn.geometry.is_empty].copy()
    if uprn.empty:
        return None
    _log(f"[INFO] Loaded UPRN guard points={len(uprn):,}")
    return UprnContext(points=uprn, id_field=uprn_id_field, sindex=uprn.sindex)


def _uprn_ids_intersecting(geom, uprn: UprnContext | None) -> set[Any]:
    if uprn is None or geom is None or shapely.is_empty(geom) or uprn.points.empty:
        return set()
    try:
        positions = list(uprn.sindex.query(geom, predicate="intersects"))
    except TypeError:
        positions = list(uprn.sindex.query(geom))
    ids: set[Any] = set()
    for pos in positions:
        point = uprn.points.geometry.iloc[int(pos)]
        if point is None or shapely.is_empty(point):
            continue
        try:
            intersects = bool(geom.intersects(point))
        except Exception:
            intersects = bool(shapely.intersects(geom, point))
        if not intersects:
            continue
        ids.add(uprn.points.iloc[int(pos)][uprn.id_field])
    return ids


def _recompute_uprn_counts(predicted: gpd.GeoDataFrame, uprn: UprnContext | None) -> gpd.GeoDataFrame:
    if uprn is None or predicted.empty:
        return predicted
    out = predicted.copy()
    counts = []
    for geom in out.geometry.array:
        counts.append(len(_uprn_ids_intersecting(geom, uprn)))
    out["pred_uprn_count"] = counts
    return out


def _refresh_shape_metrics(predicted: gpd.GeoDataFrame, changed_indices: set[Any]) -> gpd.GeoDataFrame:
    if not changed_indices:
        return predicted
    out = predicted.copy()
    for idx in sorted(changed_indices):
        if idx not in out.index:
            continue
        shape = _shape_metrics(out.at[idx, out.geometry.name])
        for name, value in shape.items():
            column = f"pred_{name}"
            if column in out.columns:
                out.at[idx, column] = float(value)
    return out


def _fill_output_holes(
    predicted: gpd.GeoDataFrame,
    cfg: FinalGapFillConfig,
    *,
    uprn: UprnContext | None = None,
    stage: str = "output_hole_fill",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, set[Any]]:
    if not cfg.fill_output_holes or predicted.empty:
        return predicted, _empty_gap_debug(predicted.crs), _empty_gap_debug(predicted.crs), set()

    out = predicted.copy()
    geom_col = out.geometry.name
    records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    changed: set[Any] = set()

    occupied_ref = gpd.GeoDataFrame(columns=[geom_col], geometry=geom_col, crs=out.crs)
    occupied_sindex = None
    if cfg.skip_occupied_output_holes:
        valid_mask = out.geometry.notna() & ~out.geometry.is_empty
        polygon_mask = valid_mask & out.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
        occupied_ref = out.loc[polygon_mask, [geom_col]].copy()
        occupied_sindex = occupied_ref.sindex if len(occupied_ref) else None

    for idx, row in out.iterrows():
        geom = row[geom_col]
        if geom is None or shapely.is_empty(geom):
            continue
        holes = _collect_interior_hole_polygons(
            geom,
            min_area=0.0,
            max_area=None,
        )
        if not holes:
            continue

        fillable_holes = []
        component_id = int(row["pred_component_id"]) if "pred_component_id" in row else int(idx)
        for hole_idx, hole in enumerate(holes, start=1):
            uprn_ids = _uprn_ids_intersecting(hole, uprn) if cfg.guard_output_holes_with_uprn else set()
            if cfg.guard_output_holes_with_uprn and uprn_ids:
                skipped_records.append(
                    {
                        "pred_component_id": component_id,
                        "fill_stage": stage,
                        "hole_index": hole_idx,
                        "hole_area": float(hole.area),
                        "shared_edge_m": 0.0,
                        "target_area_before": float(shapely.area(geom)),
                        "target_area_after": float(shapely.area(geom)),
                        "skipped_reason": "contains_uprn",
                        "uprn_count": int(len(uprn_ids)),
                        "geometry": hole,
                    }
                )
                continue
            occupied = False
            if cfg.skip_occupied_output_holes and occupied_sindex is not None:
                try:
                    positions = list(occupied_sindex.query(hole, predicate="intersects"))
                except TypeError:
                    positions = list(occupied_sindex.query(hole))
                for pos in positions:
                    other_idx = occupied_ref.index[int(pos)]
                    if other_idx == idx:
                        continue
                    other_geom = occupied_ref.at[other_idx, geom_col]
                    if other_geom is None or shapely.is_empty(other_geom):
                        continue
                    if float(hole.intersection(other_geom).area) >= float(cfg.min_polygon_part_area_m2):
                        occupied = True
                        break
            if not occupied:
                fillable_holes.append(hole)

        if not fillable_holes:
            continue

        filled = _make_valid_geometry(geom)
        if filled is None or shapely.is_empty(filled):
            continue
        for hole in fillable_holes:
            filled = shapely.union_all([filled, hole])
        filled = _drop_tiny_polygon_parts(filled, cfg.min_polygon_part_area_m2)
        if filled is None or shapely.is_empty(filled):
            continue
        out.at[idx, geom_col] = _as_multi_polygon(filled)
        changed.add(idx)
        for hole_idx, hole in enumerate(fillable_holes, start=1):
            records.append(
                {
                    "pred_component_id": component_id,
                    "fill_stage": stage,
                    "hole_index": hole_idx,
                    "hole_area": float(hole.area),
                    "shared_edge_m": 0.0,
                    "target_area_before": float(shapely.area(geom)),
                    "target_area_after": float(shapely.area(filled)),
                    "skipped_reason": "",
                    "uprn_count": 0,
                    "geometry": hole,
                }
            )

    debug = _gap_debug_from_records(records, out.crs)
    skipped = _gap_debug_from_records(skipped_records, out.crs)
    return out, debug, skipped, changed


def _merge_enclosed_gap_holes(
    predicted: gpd.GeoDataFrame,
    cfg: FinalGapFillConfig,
    *,
    uprn: UprnContext | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, set[Any]]:
    if not cfg.fill_enclosed_gap_holes or len(predicted) < 2:
        return predicted, _empty_gap_debug(predicted.crs), _empty_gap_debug(predicted.crs), set()

    out = predicted.copy()
    geom_col = out.geometry.name
    valid_mask = out.geometry.notna() & ~out.geometry.is_empty
    polygon_mask = valid_mask & out.geometry.geom_type.str.upper().isin(["POLYGON", "MULTIPOLYGON"])
    target_ref = out.loc[polygon_mask, [geom_col, "pred_component_id"]].copy()
    if len(target_ref) < 2:
        return out, _empty_gap_debug(out.crs), _empty_gap_debug(out.crs), set()

    _log("[INFO] Building coverage union for enclosed gap fill")
    coverage = _make_valid_geometry(_coverage_union(list(target_ref.geometry.array)))
    holes = _collect_interior_hole_polygons(
        coverage,
        min_area=cfg.min_polygon_part_area_m2,
        max_area=cfg.enclosed_gap_max_area_m2,
    )
    if not holes:
        return out, _empty_gap_debug(out.crs), _empty_gap_debug(out.crs), set()

    _log(f"[INFO] Enclosed gap candidates={len(holes):,}")
    sindex = target_ref.sindex
    target_area = target_ref.geometry.area.astype(float).to_dict()
    replacements: dict[Any, Any] = {}
    records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    changed: set[Any] = set()

    for gap_id, hole in enumerate(sorted(holes, key=lambda item: float(item.area)), start=1):
        uprn_ids = _uprn_ids_intersecting(hole, uprn)
        if cfg.guard_enclosed_gaps_with_uprn and uprn_ids:
            skipped_records.append(
                {
                    "pred_component_id": -1,
                    "fill_stage": "enclosed_gap_fill",
                    "hole_index": gap_id,
                    "hole_area": float(hole.area),
                    "shared_edge_m": 0.0,
                    "target_area_before": 0.0,
                    "target_area_after": 0.0,
                    "skipped_reason": "contains_uprn",
                    "uprn_count": int(len(uprn_ids)),
                    "geometry": hole,
                }
            )
            continue
        try:
            positions = list(sindex.query(hole, predicate="intersects"))
        except TypeError:
            positions = list(sindex.query(hole))
        if not positions:
            continue

        best_idx = None
        best_score = None
        best_geom = None
        best_shared = 0.0
        for pos in positions:
            idx = target_ref.index[int(pos)]
            target_geom = replacements.get(idx, out.at[idx, geom_col])
            if target_geom is None or shapely.is_empty(target_geom):
                continue

            shared_len = float(hole.boundary.intersection(target_geom.boundary).length)
            if shared_len < cfg.enclosed_gap_min_shared_edge_m:
                continue

            merged_geom = _drop_tiny_polygon_parts(
                shapely.union_all([target_geom, hole]),
                cfg.min_polygon_part_area_m2,
            )
            if merged_geom is None or shapely.is_empty(merged_geom):
                continue
            if _polygon_part_count(merged_geom) > _polygon_part_count(target_geom):
                continue

            score = (shared_len, float(target_area.get(idx, 0.0)), -float(hole.area), str(idx))
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
                best_geom = merged_geom
                best_shared = shared_len

        if best_idx is None or best_geom is None:
            continue

        old_geom = replacements.get(best_idx, out.at[best_idx, geom_col])
        replacements[best_idx] = _as_multi_polygon(best_geom)
        changed.add(best_idx)
        component_id = int(out.at[best_idx, "pred_component_id"]) if "pred_component_id" in out.columns else int(best_idx)
        records.append(
            {
                "pred_component_id": component_id,
                "fill_stage": "enclosed_gap_fill",
                "hole_index": gap_id,
                "hole_area": float(hole.area),
                "shared_edge_m": float(best_shared),
                "target_area_before": float(shapely.area(old_geom)),
                "target_area_after": float(shapely.area(best_geom)),
                "skipped_reason": "",
                "uprn_count": int(len(uprn_ids)),
                "geometry": hole,
            }
        )

    if replacements:
        new_geometry = out.geometry.copy()
        for idx, geom in replacements.items():
            if idx in new_geometry.index:
                new_geometry.loc[idx] = geom
        out = out.set_geometry(new_geometry)

    debug = _gap_debug_from_records(records, out.crs)
    skipped = _gap_debug_from_records(skipped_records, out.crs)
    return out, debug, skipped, changed


def _empty_gap_debug(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "pred_component_id",
            "fill_stage",
            "hole_index",
            "hole_area",
            "shared_edge_m",
            "target_area_before",
            "target_area_after",
            "skipped_reason",
            "uprn_count",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def _gap_debug_from_records(records: list[dict[str, Any]], crs) -> gpd.GeoDataFrame:
    if not records:
        return _empty_gap_debug(crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _changed_parcels_layer(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    changed_indices: set[Any],
) -> gpd.GeoDataFrame:
    if not changed_indices:
        return gpd.GeoDataFrame(columns=list(after.columns) + ["area_delta"], geometry="geometry", crs=after.crs)
    rows = after.loc[sorted(idx for idx in changed_indices if idx in after.index)].copy()
    before_area = before.loc[rows.index].geometry.area.astype(float)
    rows["area_delta"] = rows.geometry.area.astype(float).to_numpy() - before_area.to_numpy()
    return rows


def _related_prediction_layers(predicted: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    merged_only = predicted[predicted["source_count"].fillna(0).astype(int).gt(1)].copy()
    possible_fp = predicted[predicted["possible_false_positive_cluster"].fillna(0).astype(int).eq(1)].copy()
    possible_split = predicted[predicted["possible_split_reference"].fillna(0).astype(int).eq(1)].copy()
    no_uprn = predicted.drop(columns=["pred_uprn_count"], errors="ignore")
    layers = {
        "predicted_parcels_with_uprn": predicted,
        "predicted_parcels": no_uprn,
        "predicted_parcels_merged_only_with_uprn": merged_only,
        "predicted_parcels_merged_only": merged_only.drop(columns=["pred_uprn_count"], errors="ignore"),
        "possible_false_positive_clusters_with_uprn": possible_fp,
        "possible_false_positive_clusters": possible_fp.drop(columns=["pred_uprn_count"], errors="ignore"),
        "possible_split_reference_clusters_with_uprn": possible_split,
        "possible_split_reference_clusters": possible_split.drop(columns=["pred_uprn_count"], errors="ignore"),
        "predicted_parcels_merged_only_uprn_le1": merged_only[
            merged_only["pred_uprn_count"].fillna(0).astype(int).le(1)
        ].copy(),
        "predicted_parcels_merged_only_uprn_eq1": merged_only[
            merged_only["pred_uprn_count"].fillna(0).astype(int).eq(1)
        ].copy(),
        "predicted_parcels_merged_only_multi_uprn": merged_only[
            merged_only["pred_uprn_count"].fillna(0).astype(int).gt(1)
        ].copy(),
    }
    return layers


def _write_output_gpkg(
    input_gpkg: Path,
    output_gpkg: Path,
    replacements: dict[str, gpd.GeoDataFrame],
) -> None:
    input_gpkg = input_gpkg.resolve()
    output_gpkg = output_gpkg.resolve()
    same_path = input_gpkg == output_gpkg
    target = output_gpkg.with_suffix(".tmp_final_gap_fill.gpkg") if same_path else output_gpkg
    if target.exists():
        target.unlink()

    original_layers = [str(row[0]) for row in pyogrio.list_layers(input_gpkg)]
    written: set[str] = set()
    for layer in original_layers:
        if layer in replacements:
            gdf = replacements[layer]
        else:
            gdf = pyogrio.read_dataframe(input_gpkg, layer=layer)
        _write_layer(gdf, target, layer)
        written.add(layer)

    for layer, gdf in replacements.items():
        if layer in written:
            continue
        _write_layer(gdf, target, layer)

    if same_path:
        os.replace(target, output_gpkg)


def apply_final_gap_fill(
    input_gpkg: Path,
    output_gpkg: Path,
    cfg: FinalGapFillConfig,
    *,
    final_layer: str = FINAL_ALIAS_LAYER,
    uprn_gpkg: Path | None = None,
    uprn_layer: str = DEFAULT_UPRN_LAYER,
    uprn_id_field: str = DEFAULT_UPRN_ID_FIELD,
) -> dict[str, Any]:
    _log(f"[INFO] Reading final parcels: {input_gpkg} ({PREDICTED_LAYER})")
    predicted_before = pyogrio.read_dataframe(input_gpkg, layer=PREDICTED_LAYER)
    predicted_before = predicted_before[
        predicted_before.geometry.notna() & ~predicted_before.geometry.is_empty
    ].copy()
    predicted_before = predicted_before.reset_index(drop=True)
    predicted = predicted_before.copy()
    uprn = _load_uprn_context(uprn_gpkg, uprn_layer, uprn_id_field, predicted_before)

    _log("[INFO] Applying enclosed gap fill")
    predicted, enclosed_gaps, skipped_enclosed, changed_enclosed = _merge_enclosed_gap_holes(
        predicted,
        cfg,
        uprn=uprn,
    )

    _log("[INFO] Applying final output hole fill")
    predicted, output_holes, skipped_output_holes, changed_output_holes = _fill_output_holes(
        predicted,
        cfg,
        uprn=uprn,
        stage="final_output_hole_fill",
    )

    changed_indices = set(changed_enclosed) | set(changed_output_holes)
    predicted = _refresh_shape_metrics(predicted, changed_indices)
    predicted = predicted.set_geometry(
        predicted.geometry.apply(lambda geom: _as_multi_polygon(_drop_tiny_polygon_parts(geom, cfg.min_polygon_part_area_m2)))
    )
    predicted = predicted[predicted.geometry.notna() & ~predicted.geometry.is_empty].copy()
    if cfg.recompute_uprn_count_from_final_geometry:
        predicted = _recompute_uprn_counts(predicted, uprn)

    changed_parcels = _changed_parcels_layer(predicted_before, predicted, changed_indices)
    replacements = _related_prediction_layers(predicted)
    replacements["final_gap_fill_output_holes"] = output_holes
    replacements["final_gap_fill_enclosed_gaps"] = enclosed_gaps
    skipped_uprn_records = list(skipped_enclosed.to_dict("records")) + list(skipped_output_holes.to_dict("records"))
    if skipped_uprn_records:
        skipped_uprn_holes = gpd.GeoDataFrame(
            skipped_uprn_records,
            geometry="geometry",
            crs=predicted.crs,
        )
    else:
        skipped_uprn_holes = _empty_gap_debug(predicted.crs)
    replacements["final_gap_fill_skipped_uprn_holes"] = skipped_uprn_holes
    replacements["final_gap_fill_changed_parcels"] = changed_parcels

    if final_layer:
        replacements[final_layer] = predicted

    _log(f"[INFO] Writing gap-filled output: {output_gpkg}")
    _write_output_gpkg(input_gpkg, output_gpkg, replacements)

    summary = {
        "input_gpkg": str(input_gpkg),
        "output_gpkg": str(output_gpkg),
        "predicted_layer": PREDICTED_LAYER,
        "final_layer": final_layer,
        "fill_output_holes": bool(cfg.fill_output_holes),
        "output_hole_fill_order": "after_enclosed_gap_fill",
        "fill_enclosed_gap_holes": bool(cfg.fill_enclosed_gap_holes),
        "guard_enclosed_gaps_with_uprn": bool(cfg.guard_enclosed_gaps_with_uprn),
        "guard_output_holes_with_uprn": bool(cfg.guard_output_holes_with_uprn),
        "skip_occupied_output_holes": bool(cfg.skip_occupied_output_holes),
        "recompute_uprn_count_from_final_geometry": bool(cfg.recompute_uprn_count_from_final_geometry),
        "enclosed_gap_max_area_m2": float(cfg.enclosed_gap_max_area_m2),
        "enclosed_gap_min_shared_edge_m": float(cfg.enclosed_gap_min_shared_edge_m),
        "min_polygon_part_area_m2": float(cfg.min_polygon_part_area_m2),
        "input_features": int(len(predicted_before)),
        "output_features": int(len(predicted)),
        "output_hole_rows": int(len(output_holes)),
        "output_hole_area_m2": float(output_holes["hole_area"].sum()) if "hole_area" in output_holes else 0.0,
        "enclosed_gap_rows": int(len(enclosed_gaps)),
        "enclosed_gap_area_m2": float(enclosed_gaps["hole_area"].sum()) if "hole_area" in enclosed_gaps else 0.0,
        "uprn_guard_skipped_hole_rows": int(len(skipped_uprn_holes)),
        "uprn_guard_skipped_hole_area_m2": float(skipped_uprn_holes["hole_area"].sum())
        if "hole_area" in skipped_uprn_holes
        else 0.0,
        "changed_parcel_rows": int(len(changed_parcels)),
    }
    output_gpkg.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply final fallback hole and enclosed-small-gap fill to native WFS merge output.",
    )
    parser.add_argument("--input-gpkg", default=DEFAULT_INPUT_GPKG)
    parser.add_argument("--output-gpkg", default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--final-layer", default=FINAL_ALIAS_LAYER)
    parser.add_argument("--keep-output-holes", action="store_true")
    parser.add_argument("--disable-enclosed-gap-fill", action="store_true")
    parser.add_argument("--enclosed-gap-max-area", type=float, default=250.0)
    parser.add_argument("--enclosed-gap-min-shared-edge", type=float, default=0.05)
    parser.add_argument("--min-polygon-part-area", type=float, default=0.01)
    parser.add_argument("--uprn-gpkg", default=DEFAULT_UPRN_GPKG)
    parser.add_argument("--uprn-layer", default=DEFAULT_UPRN_LAYER)
    parser.add_argument("--uprn-id-field", default=DEFAULT_UPRN_ID_FIELD)
    parser.add_argument("--disable-uprn-gap-guard", action="store_true")
    parser.add_argument("--guard-enclosed-gaps-with-uprn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guard-output-holes-with-uprn", action="store_true")
    parser.add_argument("--skip-occupied-output-holes", action="store_true")
    parser.add_argument(
        "--recompute-uprn-count-from-final-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_final_gap_fill(
        Path(args.input_gpkg),
        Path(args.output_gpkg),
        FinalGapFillConfig(
            fill_output_holes=not bool(args.keep_output_holes),
            fill_enclosed_gap_holes=not bool(args.disable_enclosed_gap_fill),
            enclosed_gap_max_area_m2=float(args.enclosed_gap_max_area),
            enclosed_gap_min_shared_edge_m=float(args.enclosed_gap_min_shared_edge),
            min_polygon_part_area_m2=float(args.min_polygon_part_area),
            guard_enclosed_gaps_with_uprn=bool(args.guard_enclosed_gaps_with_uprn)
            and not bool(args.disable_uprn_gap_guard),
            guard_output_holes_with_uprn=bool(args.guard_output_holes_with_uprn),
            skip_occupied_output_holes=bool(args.skip_occupied_output_holes),
            recompute_uprn_count_from_final_geometry=bool(args.recompute_uprn_count_from_final_geometry),
        ),
        final_layer=str(args.final_layer),
        uprn_gpkg=None if bool(args.disable_uprn_gap_guard) else Path(args.uprn_gpkg),
        uprn_layer=str(args.uprn_layer),
        uprn_id_field=str(args.uprn_id_field),
    )
    _log("[DONE] Final gap fill complete")
    _log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

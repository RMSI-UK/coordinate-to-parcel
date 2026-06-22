from __future__ import annotations

import argparse
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from _core.config import add_config_argument, get_config_section_from_argv, require_configured


SEVERE_WEIRD_LABELS = {
    "review_likely_geocode_mismatch",
    "review_out_of_area_or_geocode",
    "review_possible_overcapture",
    "review_low_compactness",
    "review_shape",
}


GEOMETRY_CLEAN_NEAREST_NO_INTERSECTION = "multipart_nearest_part_no_point_intersection"


CONFIDENCE_ORDER = {"missing_input": 0, "failed": 1, "low": 2, "medium": 3, "high": 4}

ROAD_WORDS = {
    "avenue",
    "close",
    "court",
    "crescent",
    "drive",
    "gardens",
    "gate",
    "green",
    "grove",
    "hill",
    "lane",
    "mount",
    "park",
    "place",
    "road",
    "row",
    "square",
    "street",
    "terrace",
    "view",
    "walk",
    "way",
}

GENERIC_ADDRESS_WORDS = {
    "and",
    "at",
    "by",
    "flat",
    "land",
    "no",
    "nos",
    "of",
    "off",
    "plot",
    "plots",
    "rear",
    "sheffield",
    "site",
    "the",
    "to",
    "unit",
}

HIGH_DISQUALIFY_PATTERNS = [
    r"\bland\b",
    r"\bplot\b",
    r"\bplots\b",
    r"\badjoining\b",
    r"\badjacent\b",
    r"\brear\b",
    r"\bcurtilage\b",
    r"\bsite\b",
    r"\bcompound\b",
    r"\bgarage\b",
    r"\bcar\s+park\b",
    r"\bbetween\b",
    r"\boff\b",
    r"\bwithin\b",
]

NUMBER_CONTEXT_EXCLUDE = {"unit", "flat", "plot", "plots", "block"}
STREET_PREFIX_EXCLUDE = {
    "and",
    "at",
    "by",
    "land",
    "no",
    "nos",
    "of",
    "off",
    "plot",
    "plots",
    "rear",
    "sheffield",
    "site",
    "the",
    "to",
    "unit",
}
SPECIFIC_SINGLE_STREET_WORDS = {"gate", "green", "grove", "hill", "mount", "row", "square", "view", "walk"}


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv(
        "build_final_parent_table",
        include_package_defaults=True,
    )
    parser = argparse.ArgumentParser(
        description="Assemble the final parent-unique delivery table.",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--capture-gpkg")
    parser.add_argument("--capture-layer")
    parser.add_argument("--source-xlsx")
    parser.add_argument("--qa-gpkg")
    parser.add_argument("--qa-layer")
    parser.add_argument("--weird-gpkg")
    parser.add_argument("--weird-layer")
    parser.add_argument("--point-gpkg")
    parser.add_argument("--point-layer")
    parser.add_argument("--output-gpkg")
    parser.add_argument("--output-layer")
    parser.add_argument("--output-xlsx")
    parser.add_argument("--output-csv")
    parser.add_argument("--missing-xlsx")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(
        args,
        ("capture_gpkg", "source_xlsx", "output_gpkg", "output_xlsx", "output_csv", "missing_xlsx"),
        "build_final_parent_table",
    )
    return args


def _lower_confidence(current: str, candidate: str) -> str:
    return current if CONFIDENCE_ORDER[current] <= CONFIDENCE_ORDER[candidate] else candidate


def _read_optional_gpkg(path: str | None, layer: str | None, cols: list[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["unique_key", *cols])
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["unique_key", *cols])
    gdf = gpd.read_file(path, layer=layer)
    keep = ["unique_key", *[c for c in cols if c in gdf.columns]]
    return pd.DataFrame(gdf[keep]).drop_duplicates("unique_key")


def _polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return [part for part in getattr(geom, "geoms", []) if part.geom_type == "Polygon"]


def _fill_polygon_holes(poly: Polygon) -> Polygon:
    return Polygon(poly.exterior)


def _point_from_row(row):
    try:
        x = row.get("geocoding_final_easting_27700")
        y = row.get("geocoding_final_northing_27700")
        if pd.notna(x) and pd.notna(y):
            return Point(float(x), float(y))
    except Exception:
        pass
    return None


def _load_point_lookup(point_gpkg: str | None, point_layer: str | None) -> dict[str, Point]:
    if not point_gpkg:
        return {}
    p = Path(point_gpkg)
    if not p.exists():
        return {}
    points = gpd.read_file(point_gpkg, layer=point_layer)
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    parent = points[points["variant_key"].astype(str).eq(points["unique_key"].astype(str))].drop_duplicates("unique_key")
    first = points.drop_duplicates("unique_key")
    lookup = {str(key): geom for key, geom in zip(parent["unique_key"], parent.geometry)}
    for key, geom in zip(first["unique_key"], first.geometry):
        lookup.setdefault(str(key), geom)
    return lookup


def _clean_single_geometry(row, point_lookup: dict[str, Point]):
    geom = row.geometry
    parts = _polygon_parts(geom)
    holes_before = sum(len(part.interiors) for part in parts)
    parts_before = len(parts)
    if not parts:
        return None, "missing_or_empty_geometry", parts_before, holes_before, 0, False, None

    point = point_lookup.get(str(row.get("unique_key"))) or _point_from_row(row)
    selected = parts[0]
    method = "single_part"
    point_intersects_selected = False
    selected_distance = None

    if len(parts) > 1:
        intersecting = [part for part in parts if point is not None and part.intersects(point)]
        if intersecting:
            selected = max(intersecting, key=lambda part: part.area)
            method = "multipart_point_intersect_part"
            point_intersects_selected = True
            selected_distance = 0.0
        elif point is not None:
            selected = min(parts, key=lambda part: part.distance(point))
            selected_distance = float(selected.distance(point))
            method = GEOMETRY_CLEAN_NEAREST_NO_INTERSECTION
        else:
            selected = max(parts, key=lambda part: part.area)
            method = "multipart_largest_part_no_point"
    elif point is not None:
        point_intersects_selected = bool(selected.intersects(point))
        selected_distance = 0.0 if point_intersects_selected else float(selected.distance(point))

    cleaned = _fill_polygon_holes(selected)
    if not cleaned.is_valid:
        repaired = cleaned.buffer(0)
        repaired_parts = _polygon_parts(repaired)
        if repaired_parts:
            cleaned = max(repaired_parts, key=lambda part: part.area)
            method = f"{method}_buffer0_repair"

    if holes_before:
        method = f"{method}_holes_filled"
    return cleaned, method, parts_before, holes_before, 1, point_intersects_selected, selected_distance


def _clean_output_geometries(
    out: gpd.GeoDataFrame,
    *,
    point_gpkg: str | None,
    point_layer: str | None,
) -> gpd.GeoDataFrame:
    point_lookup = _load_point_lookup(point_gpkg, point_layer)
    cleaned_rows = out.apply(
        lambda row: _clean_single_geometry(row, point_lookup),
        axis=1,
        result_type="expand",
    )
    cleaned_rows.columns = [
        "geometry_clean_geometry",
        "geometry_clean_method",
        "geometry_parts_before",
        "geometry_holes_before",
        "geometry_parts_after",
        "geometry_selected_part_intersects_point",
        "geometry_selected_part_point_distance_m",
    ]
    out = out.copy()
    out["geometry_clean_method"] = cleaned_rows["geometry_clean_method"]
    out["geometry_parts_before"] = cleaned_rows["geometry_parts_before"].astype("Int64")
    out["geometry_holes_before"] = cleaned_rows["geometry_holes_before"].astype("Int64")
    out["geometry_parts_after"] = cleaned_rows["geometry_parts_after"].astype("Int64")
    out["geometry_selected_part_intersects_point"] = cleaned_rows["geometry_selected_part_intersects_point"]
    out["geometry_selected_part_point_distance_m"] = cleaned_rows["geometry_selected_part_point_distance_m"]
    out[out.geometry.name] = cleaned_rows["geometry_clean_geometry"]
    return gpd.GeoDataFrame(out, geometry=out.geometry.name, crs=out.crs)


def _string(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _address_tokens(value: object) -> set[str]:
    text = _string(value).lower()
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text):
        if _is_postcode_token(token):
            continue
        tokens.add(token)
    return tokens


def _is_postcode_token(token: str) -> bool:
    return bool(
        re.fullmatch(r"[a-z]{1,2}\d+[a-z]?", token)
        or re.fullmatch(r"\d[a-z]{2}", token)
    )


def _door_number_tokens(value: object) -> list[str]:
    text = _string(value).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    numbers: list[str] = []
    for idx, token in enumerate(tokens):
        if _is_postcode_token(token):
            continue
        if not re.fullmatch(r"\d+[a-z]?", token):
            continue
        prev = tokens[idx - 1] if idx else ""
        if prev in NUMBER_CONTEXT_EXCLUDE:
            continue
        if _is_postcode_token(prev):
            continue
        numbers.append(token)

    for match in re.finditer(r"\b\d+[a-z]?\s*[-/]\s*\d+[a-z]?\b", text):
        prefix_tokens = re.findall(r"[a-z0-9]+", text[: match.start()])
        prev = prefix_tokens[-1] if prefix_tokens else ""
        if prev in NUMBER_CONTEXT_EXCLUDE:
            continue
        numbers.extend(re.findall(r"\d+[a-z]?", match.group(0)))

    return list(dict.fromkeys(numbers))


def _has_door_number(value: object) -> bool:
    return bool(_door_number_tokens(value))


def _street_signatures(value: object) -> set[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", _string(value).lower()) if not _is_postcode_token(token)]
    signatures: set[str] = set()
    for idx, token in enumerate(tokens):
        if token not in ROAD_WORDS:
            continue
        prev = tokens[idx - 1] if idx else ""
        if prev and prev not in STREET_PREFIX_EXCLUDE and not re.fullmatch(r"\d+[a-z]?", prev):
            signatures.add(f"{prev} {token}")
        elif token in SPECIFIC_SINGLE_STREET_WORDS:
            signatures.add(token)
    return signatures


def _postcode_outcodes(value: object) -> set[str]:
    return set(re.findall(r"\b[a-z]{1,2}\d+[a-z]?\b", _string(value).lower()))


def _has_explicit_road_name(value: object) -> bool:
    tokens = _address_tokens(value)
    return bool(tokens & ROAD_WORDS)


def _address_similarity(a: object, b: object) -> float:
    left = _address_tokens(a) - GENERIC_ADDRESS_WORDS
    right = _address_tokens(b) - GENERIC_ADDRESS_WORDS
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _high_confidence_address_gate(row) -> tuple[bool, str]:
    original = row.get("original_address")
    geocoded = row.get("geocoding_final_output_address")
    original_text = _string(original).lower()
    disqualified = any(re.search(pattern, original_text) for pattern in HIGH_DISQUALIFY_PATTERNS)
    original_numbers = set(_door_number_tokens(original))
    geocoded_numbers = set(_door_number_tokens(geocoded))
    number_match = bool(original_numbers & geocoded_numbers)
    has_number = bool(original_numbers)
    has_road = _has_explicit_road_name(original)
    original_streets = _street_signatures(original)
    geocoded_streets = _street_signatures(geocoded)
    street_match = bool(original_streets & geocoded_streets)
    original_outcodes = _postcode_outcodes(original)
    geocoded_outcodes = _postcode_outcodes(geocoded)
    outcode_match = bool(original_outcodes & geocoded_outcodes)
    similarity = _address_similarity(original, geocoded)
    component_close = number_match and street_match and (outcode_match or similarity >= 0.55)
    ok = (not disqualified) and has_number and has_road and (similarity >= 0.85 or component_close)
    note = (
        "High-confidence address gate: "
        f"door_number={has_number}, explicit_road={has_road}, "
        f"disqualified_address_terms={disqualified}, "
        f"door_match={number_match}, street_match={street_match}, "
        f"postcode_outcode_match={outcode_match}, "
        f"original/geocode_similarity={similarity:.2f}."
    )
    return ok, note


def _confidence_and_notes(row) -> tuple[str, str]:
    notes: list[str] = []
    stage = _string(row.get("capture_stage"))
    aggregation = _string(row.get("aggregation_method"))
    qa_reason = _string(row.get("qa_reason"))
    weird_label = _string(row.get("manual_shape_qa_label"))
    weird_note = _string(row.get("manual_shape_qa_note"))

    if not bool(row.get("capture_success", False)):
        dist = _string(row.get("nearest_eligible_wfs_dist_m"))
        theme = _string(row.get("nearest_any_wfs_theme"))
        notes.append(f"FAILED: no eligible building/land WFS polygon accepted; nearest eligible distance={dist}; point-under-theme={theme}.")
        return "failed", " ".join(notes)

    high_address_ok, high_address_note = _high_confidence_address_gate(row)
    confidence = "high" if high_address_ok else "medium"
    notes.append(high_address_note)

    if stage == "fallback_nearest_os_wfs_merge":
        confidence = "medium"
        notes.append("Uses nearest eligible WFS fallback rather than direct point intersection.")
    elif stage == "linked_parent_union":
        notes.append("Parent record uses linked child polygon union.")
    elif stage == "step1_council_seed_wfs_inline_no_move":
        notes.append("Captured by council seed only as WFS merge reference; output boundary remains WFS-derived.")
    elif stage in {"council_driven_wfs_reference", "council_driven_wfs_reference_late"}:
        notes.append("Council-driven WFS selection: containing council polygon used only as reference; output boundary remains WFS-derived.")
    elif stage == "step2_os_wfs_merge_intersection":
        notes.append("Captured by direct point intersection with eligible merged WFS polygon.")
    else:
        notes.append(f"Captured by {stage}.")

    if aggregation and aggregation != "single_record":
        notes.append(f"Aggregated method={aggregation}; group_size={int(row.get('aggregation_group_size', 1) or 1)}.")

    if qa_reason:
        notes.append(f"QA flag: {qa_reason}.")
        if "road_anchor" in qa_reason:
            point_dist = row.get("point_output_dist_m")
            theme = _string(row.get("Theme"))
            try:
                point_dist_text = f"{float(point_dist):.2f}m"
            except Exception:
                point_dist_text = _string(point_dist)
            notes.append(f"Point lies on non-building/land WFS ({theme}) and output is {point_dist_text} from point.")
            if float(row.get("point_output_dist_m", 0.0) or 0.0) > 15.0:
                confidence = _lower_confidence(confidence, "low")
            else:
                confidence = _lower_confidence(confidence, "medium")
        if "linked_parent_union_point_dist" in qa_reason and confidence != "low":
            confidence = _lower_confidence(confidence, "medium")

    if weird_label:
        notes.append(f"Shape QA: {weird_label}.")
        if weird_note:
            notes.append(weird_note)
        if weird_label in SEVERE_WEIRD_LABELS:
            confidence = _lower_confidence(confidence, "low")
        elif weird_label == "review_fragmented_plot_union":
            confidence = _lower_confidence(confidence, "medium")
        elif weird_label == "likely_ok_named_large_or_complex_site" and confidence == "high":
            confidence = _lower_confidence(confidence, "medium")

    if not qa_reason and not weird_label and confidence == "high":
        notes.append("No automatic QA flags.")

    return confidence, " ".join(notes)


def main() -> None:
    args = parse_args()
    capture = gpd.read_file(args.capture_gpkg, layer=args.capture_layer)
    excel = pd.read_excel(args.source_xlsx)
    excel = excel.copy()
    excel["oachargeid"] = excel["originating-authority-charge-identifier"].astype("Int64")
    excel_join = excel[["oachargeid"]].drop_duplicates("oachargeid")

    qa = _read_optional_gpkg(
        args.qa_gpkg,
        args.qa_layer,
        ["qa_reason", "qa_priority", "point_output_dist_m", "Theme", "underlying_wfs_dist_m", "area_m2", "parts", "compactness"],
    )
    weird = _read_optional_gpkg(
        args.weird_gpkg,
        args.weird_layer,
        [
            "shape_qa_flags",
            "shape_qa_priority",
            "manual_shape_qa_label",
            "manual_shape_qa_note",
            "qa_area_m2",
            "qa_parts",
            "qa_holes",
            "qa_aspect",
            "qa_compactness",
        ],
    )

    out = capture.copy()
    out["unique_key_int"] = out["unique_key"].astype(int)
    out = out.merge(excel_join, left_on="unique_key_int", right_on="oachargeid", how="left")
    out = out.merge(qa, on="unique_key", how="left")
    out = out.merge(weird, on="unique_key", how="left")
    out = out.drop(columns=["unique_key_int"])

    confidence_notes = out.apply(_confidence_and_notes, axis=1, result_type="expand")
    confidence_notes.columns = ["auto_polygon_confidence", "auto_polygon_notes"]
    out[["auto_polygon_confidence", "auto_polygon_notes"]] = confidence_notes

    # Put the new business fields near the front while preserving original geometry output.
    preferred = [
        "oachargeid",
        "unique_key",
        "variant_key",
        "original_address",
        "auto_polygon_confidence",
        "auto_polygon_notes",
        "capture_success",
        "capture_stage",
        "aggregation_method",
        "aggregation_group_size",
        "aggregation_child_count",
        "aggregation_child_success_count",
        "qa_reason",
        "manual_shape_qa_label",
        "shape_qa_flags",
    ]
    remaining = [c for c in out.columns if c not in preferred and c != out.geometry.name]
    out = out[[*preferred, *remaining, out.geometry.name]]
    out = gpd.GeoDataFrame(out, geometry=out.geometry.name, crs=capture.crs)

    capture_keys = set(capture["unique_key"].astype(int).tolist())
    missing = excel[~excel["oachargeid"].astype(int).isin(capture_keys)].copy()
    if not missing.empty:
        missing_rows = []
        template_columns = [col for col in out.columns if col != out.geometry.name]
        for _, source_row in missing.iterrows():
            record = {col: None for col in template_columns}
            oachargeid = int(source_row["oachargeid"])
            address = source_row.get("Changed Address")
            if pd.isna(address) or not str(address).strip():
                address = source_row.get("charge-geographic-description")
            record["oachargeid"] = oachargeid
            record["unique_key"] = oachargeid
            record["variant_key"] = str(oachargeid)
            record["original_address"] = address
            record["auto_polygon_confidence"] = "missing_input"
            record["auto_polygon_notes"] = (
                "No polygon generated because this oachargeid exists in the source workbook "
                "but is missing from the configured point input."
            )
            record["capture_success"] = False
            record["capture_stage"] = "missing_from_point_input"
            record["aggregation_method"] = "missing_from_point_input"
            record["aggregation_group_size"] = 0
            record["aggregation_child_count"] = 0
            record["aggregation_child_success_count"] = 0
            record["geocoding_final_status"] = "missing_from_point_input"
            record["geocoding_final_source"] = "missing_from_point_input"
            record["geocoding_final_output_address"] = address
            missing_rows.append(record)
        missing_gdf = gpd.GeoDataFrame(
            missing_rows,
            geometry=[None for _ in missing_rows],
            crs=out.crs,
        )
        out = gpd.GeoDataFrame(
            pd.concat([out, missing_gdf], ignore_index=True),
            geometry=out.geometry.name,
            crs=out.crs,
        )
        out = out.sort_values("oachargeid").reset_index(drop=True)

    out = _clean_output_geometries(out, point_gpkg=args.point_gpkg, point_layer=args.point_layer)
    no_intersection_mask = out["geometry_clean_method"].astype(str).str.contains(
        GEOMETRY_CLEAN_NEAREST_NO_INTERSECTION,
        regex=False,
        na=False,
    )
    if no_intersection_mask.any():
        distance_text = out.loc[no_intersection_mask, "geometry_selected_part_point_distance_m"].map(
            lambda value: "" if pd.isna(value) else f" nearest_part_distance={float(value):.2f}m."
        )
        out.loc[no_intersection_mask, "auto_polygon_notes"] = (
            out.loc[no_intersection_mask, "auto_polygon_notes"].fillna("")
            + " Geometry clean: multipart had no part intersecting the point; kept nearest part."
            + distance_text
        )
        high_no_intersection_mask = no_intersection_mask & out["auto_polygon_confidence"].eq("high")
        out.loc[high_no_intersection_mask, "auto_polygon_confidence"] = "medium"
        out.loc[high_no_intersection_mask, "auto_polygon_notes"] = (
            out.loc[high_no_intersection_mask, "auto_polygon_notes"].fillna("")
            + " Confidence lowered to medium because multipart selection used nearest part rather than a point-intersecting part."
        )

    output_gpkg = Path(args.output_gpkg)
    output_xlsx = Path(args.output_xlsx)
    output_csv = Path(args.output_csv)
    missing_xlsx = Path(args.missing_xlsx)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    missing_xlsx.parent.mkdir(parents=True, exist_ok=True)

    if output_gpkg.exists():
        output_gpkg.unlink()
    out.to_file(output_gpkg, layer=args.output_layer, driver="GPKG")

    attr = pd.DataFrame(out.drop(columns=out.geometry.name))
    attr.to_csv(output_csv, index=False)
    attr.to_excel(output_xlsx, index=False)

    missing.to_excel(missing_xlsx, index=False)

    print(f"[DONE] Wrote spatial final table: {output_gpkg}")
    print(f"[DONE] Wrote attribute XLSX: {output_xlsx}")
    print(f"[DONE] Wrote attribute CSV: {output_csv}")
    print(f"[DONE] Wrote Excel rows missing from capture: {missing_xlsx}")
    print(f"[INFO] Output rows: {len(out)}")
    print(f"[INFO] oachargeid populated: {int(out['oachargeid'].notna().sum())}/{len(out)}")
    print("[INFO] auto_polygon_confidence counts:")
    print(out["auto_polygon_confidence"].value_counts(dropna=False).to_string())
    print("[INFO] Geometry clean methods:")
    print(out["geometry_clean_method"].value_counts(dropna=False).to_string())
    print("[INFO] Missing Excel oachargeid count:")
    print(len(missing))
    if not missing.empty:
        print(missing[["oachargeid", "charge-geographic-description", "Changed Address"]].to_string(index=False))


if __name__ == "__main__":
    main()

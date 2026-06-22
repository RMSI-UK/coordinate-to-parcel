#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent / "polygon-capture"))
from _core.config import add_config_argument, get_config_section_from_argv, require_configured  # noqa: E402


ID_JOIN_CANDIDATES: List[Tuple[str, ...]] = [
    ("unique_id",),
    ("queryid",),
    ("geomid", "oachargeid", "lafilereference"),
    ("geomid", "oachargeid"),
    ("oachargeid",),
    ("geomid",),
    ("lafilereference",),
    ("unique_key",),
]

POINT_X_CANDIDATES = [
    "api_easting_27700",
    "top1_easting_27700",
    "os_easting_27700",
    "easting_27700",
    "easting",
    "x",
]

POINT_Y_CANDIDATES = [
    "api_northing_27700",
    "top1_northing_27700",
    "os_northing_27700",
    "northing_27700",
    "northing",
    "y",
]

NOTE_CANDIDATES = ["note", "notes", "qa_note"]


@dataclass
class JoinResult:
    mapping: pd.Series
    strategy: str
    matched: int


def _canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _norm_key_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip().str.lower()


def _build_col_map(columns: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in columns:
        k = _canon(c)
        if k and k not in out:
            out[k] = c
    return out


def _slug_token(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return s.strip("_") or "unknown"


def _dedupe_output_tokens(council: str, location: str) -> tuple[str, str]:
    c = _slug_token(council)
    l = _slug_token(location)

    # If both tokens are effectively the same context (e.g. crawley_batch1 + crawley_batch1),
    # keep a short council token and use generic "location".
    if c == l or c.startswith(l) or l.startswith(c) or c.endswith(f"_{l}") or l.endswith(f"_{c}"):
        short_c = c.split("_")[0] if "_" in c else c
        return (short_c or c, "location")
    return (c, l)


def _pick_col(col_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if _canon(c) in col_map:
            return col_map[_canon(c)]
    return None


def _choose_layer(gpkg_path: str, layer: Optional[str]) -> str:
    if layer:
        return layer
    layers = gpd.list_layers(gpkg_path)
    if layers.empty:
        raise ValueError(f"No layers found in: {gpkg_path}")
    return str(layers.iloc[0]["name"])


def _build_id_join(addr: pd.DataFrame, qa: gpd.GeoDataFrame) -> Optional[JoinResult]:
    addr_map = _build_col_map(addr.columns)
    qa_map = _build_col_map(qa.columns)

    best: Optional[JoinResult] = None
    for keyset in ID_JOIN_CANDIDATES:
        if not all(_canon(k) in addr_map and _canon(k) in qa_map for k in keyset):
            continue
        a_cols = [addr_map[_canon(k)] for k in keyset]
        q_cols = [qa_map[_canon(k)] for k in keyset]

        a_key = pd.Series(list(zip(*[_norm_key_series(addr[c]) for c in a_cols])), index=addr.index)
        q_key = pd.Series(list(zip(*[_norm_key_series(qa[c]) for c in q_cols])), index=qa.index)

        a_nonempty = a_key.apply(lambda t: any(x != "" for x in t))
        q_nonempty = q_key.apply(lambda t: any(x != "" for x in t))

        a_sub = pd.DataFrame({"k": a_key[a_nonempty], "addr_idx": a_key[a_nonempty].index})
        q_sub = pd.DataFrame({"k": q_key[q_nonempty], "qa_idx": q_key[q_nonempty].index})

        if a_sub.empty or q_sub.empty:
            continue

        # strongest id join should be one-to-one to avoid exploding duplicates
        if a_sub["k"].duplicated().any() or q_sub["k"].duplicated().any():
            continue

        merged = a_sub.merge(q_sub, on="k", how="left")
        mapping = pd.Series(index=addr.index, dtype="float")
        mapping.loc[merged["addr_idx"]] = merged["qa_idx"].to_numpy()
        matched = int(mapping.notna().sum())

        jr = JoinResult(mapping=mapping, strategy=f"id:{'+'.join(a_cols)}", matched=matched)
        if best is None or jr.matched > best.matched:
            best = jr

    return best


def _build_text_occurrence_join(addr: pd.DataFrame, qa: gpd.GeoDataFrame) -> Optional[JoinResult]:
    addr_map = _build_col_map(addr.columns)
    qa_map = _build_col_map(qa.columns)

    # Known equivalent text fields across files
    addr_text_col = None
    qa_text_col = None

    for cand in ["charge-geographic-description", "chargegeographicdescription", "chargegeog"]:
        c = _canon(cand)
        if c in addr_map:
            addr_text_col = addr_map[c]
            break

    for cand in ["chargegeographicdescription", "charge-geographic-description", "chargegeog"]:
        c = _canon(cand)
        if c in qa_map:
            qa_text_col = qa_map[c]
            break

    if not addr_text_col or not qa_text_col:
        return None

    a_val = _normalize_text(addr[addr_text_col])
    q_val = _normalize_text(qa[qa_text_col])

    a_occ = a_val.groupby(a_val, dropna=False).cumcount()
    q_occ = q_val.groupby(q_val, dropna=False).cumcount()

    a_df = pd.DataFrame({"key": a_val, "occ": a_occ, "addr_idx": addr.index})
    q_df = pd.DataFrame({"key": q_val, "occ": q_occ, "qa_idx": qa.index})

    # Skip blank text keys
    a_df = a_df[a_df["key"] != ""]
    q_df = q_df[q_df["key"] != ""]

    if a_df.empty or q_df.empty:
        return None

    merged = a_df.merge(q_df, on=["key", "occ"], how="left")
    mapping = pd.Series(index=addr.index, dtype="float")
    mapping.loc[merged["addr_idx"]] = merged["qa_idx"].to_numpy()
    matched = int(mapping.notna().sum())

    return JoinResult(
        mapping=mapping,
        strategy=f"text_occurrence:{addr_text_col}->{qa_text_col}",
        matched=matched,
    )


def _build_row_order_join(addr: pd.DataFrame, qa: gpd.GeoDataFrame) -> Optional[JoinResult]:
    if len(addr) != len(qa):
        return None
    mapping = pd.Series(index=addr.index, dtype="float")
    mapping.loc[:] = np.arange(len(addr), dtype=float)
    return JoinResult(mapping=mapping, strategy="row_order_fallback", matched=len(addr))


def _resolve_join(addr: pd.DataFrame, qa: gpd.GeoDataFrame) -> JoinResult:
    candidates: List[JoinResult] = []

    id_join = _build_id_join(addr, qa)
    if id_join:
        candidates.append(id_join)

    txt_join = _build_text_occurrence_join(addr, qa)
    if txt_join:
        candidates.append(txt_join)

    # prefer highest matched count; tie prefers id join then text join by sort key
    if candidates:
        def _rank(j: JoinResult) -> Tuple[int, int]:
            pref = 2
            if j.strategy.startswith("id:"):
                pref = 0
            elif j.strategy.startswith("text_occurrence:"):
                pref = 1
            return (-j.matched, pref)

        return sorted(candidates, key=_rank)[0]

    row_join = _build_row_order_join(addr, qa)
    if row_join:
        return row_join

    raise RuntimeError(
        "Unable to auto-join CSV and GPKG. Please add a stable shared key (e.g. unique_id) or ensure equal row counts."
    )


def _compute_offset_m(
    addr: pd.DataFrame,
    qa: gpd.GeoDataFrame,
    mapping: pd.Series,
    x_col: str,
    y_col: str,
) -> pd.Series:
    x = pd.to_numeric(addr[x_col], errors="coerce")
    y = pd.to_numeric(addr[y_col], errors="coerce")

    out = pd.Series(np.nan, index=addr.index, dtype="float")
    valid_rows = mapping.dropna().index
    if len(valid_rows) == 0:
        return out

    qa_idx = mapping.loc[valid_rows].astype(int)

    for a_idx, q_idx in zip(valid_rows, qa_idx):
        xv = x.loc[a_idx]
        yv = y.loc[a_idx]
        if pd.isna(xv) or pd.isna(yv):
            continue
        geom = qa.geometry.iloc[q_idx]
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    geom = shapely.make_valid(geom)
            except Exception:
                try:
                    geom = geom.buffer(0)
                except Exception:
                    continue
            if geom is None or geom.is_empty:
                continue
        try:
            d = float(geom.distance(Point(float(xv), float(yv))))
        except Exception:
            continue
        if np.isfinite(d):
            out.loc[a_idx] = d

    return out


def _derive_spatial_issue(offset_m: pd.Series, note: pd.Series, tol: float) -> pd.Series:
    note_present = note.astype("string").fillna("").str.strip().ne("")

    # if offset is NaN, treat as unmatched address geocode
    has_offset = offset_m.notna()
    outside = has_offset & (offset_m > tol)
    inside_or_touch = has_offset & (offset_m <= tol)

    out = pd.Series("", index=offset_m.index, dtype="string")

    out.loc[inside_or_touch & ~note_present] = "high confident"
    out.loc[inside_or_touch & note_present] = "address issue"
    out.loc[outside & ~note_present] = "polygon issue"
    out.loc[outside & note_present] = "polygon and address issue"

    # no offset distance available
    out.loc[~has_offset] = "address unmatched"
    return out


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv(
        "spatial_qa",
        allow_top_level=True,
        include_package_defaults=True,
    )
    parser = argparse.ArgumentParser(
        description="Compute offset_m + spatial_issue by auto-joining address CSV with QA polygon GPKG.",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--address-csv", help="Address result CSV path.")
    parser.add_argument("--qa-gpkg", help="QA polygon GPKG path.")
    parser.add_argument("--qa-layer", help="Layer name inside QA GPKG (optional).")
    parser.add_argument("--output-csv", help="Output CSV path. Default: <council>_<location>_qa_<rows>.csv (with dedupe).")
    parser.add_argument("--output-gpkg", help="Optional: output GPKG path with offset_m/spatial_issue appended.")
    parser.add_argument("--offset-tolerance", type=float, help="Distance <= tolerance considered inside/touching (meters).")
    parser.add_argument("--force-27700", action=argparse.BooleanOptionalAction, help="If input GPKG has no CRS, force EPSG:27700.")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(args, ("address_csv", "qa_gpkg"), "spatial_qa")
    return args


def main() -> None:
    args = parse_args()
    address_csv = args.address_csv
    qa_gpkg = args.qa_gpkg
    qa_layer = args.qa_layer
    output_csv = args.output_csv
    output_gpkg = args.output_gpkg
    offset_tolerance = float(args.offset_tolerance)
    force_27700 = bool(args.force_27700)

    addr = pd.read_csv(address_csv, dtype=str)
    layer = _choose_layer(qa_gpkg, qa_layer)
    qa = gpd.read_file(qa_gpkg, layer=layer)

    if output_csv is None:
        location_name_raw = os.path.basename(os.path.dirname(os.path.abspath(qa_gpkg)))
        council_name_raw = layer
        council_name, location_name = _dedupe_output_tokens(council_name_raw, location_name_raw)
        output_name = f"{council_name}_{location_name}_qa_{len(addr)}.csv"
        output_csv = os.path.join(os.path.dirname(os.path.abspath(address_csv)), output_name)

    if qa.crs is None:
        if force_27700:
            qa = qa.set_crs(epsg=27700)
        else:
            raise ValueError("QA GPKG layer has no CRS. Use --force-27700 if geometry coordinates are EPSG:27700.")
    elif qa.crs.to_epsg() != 27700:
        qa = qa.to_crs(epsg=27700)

    addr_cols = _build_col_map(addr.columns)
    x_col = _pick_col(addr_cols, POINT_X_CANDIDATES)
    y_col = _pick_col(addr_cols, POINT_Y_CANDIDATES)
    note_col = _pick_col(addr_cols, NOTE_CANDIDATES)

    if not x_col or not y_col:
        raise ValueError(
            "Cannot find point coordinates in address CSV. Expected one of: "
            f"x={POINT_X_CANDIDATES}, y={POINT_Y_CANDIDATES}"
        )

    if not note_col:
        note_col = "note"
        addr[note_col] = ""

    join_result = _resolve_join(addr, qa)
    offset_m = _compute_offset_m(addr, qa, join_result.mapping, x_col=x_col, y_col=y_col)
    spatial_issue = _derive_spatial_issue(offset_m, addr[note_col], tol=offset_tolerance)

    out_df = addr.copy()
    out_df["offset_m"] = offset_m
    out_df["spatial_issue"] = spatial_issue

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    if output_gpkg:
        mapping = join_result.mapping
        qa_out = qa.copy()
        qa_out["offset_m"] = np.nan
        qa_out["spatial_issue"] = ""
        qa_out["note"] = ""

        matched = mapping.dropna().astype(int)
        for addr_idx, qa_idx in zip(matched.index, matched.values):
            qa_out.at[qa_idx, "offset_m"] = float(offset_m.loc[addr_idx]) if pd.notna(offset_m.loc[addr_idx]) else np.nan
            qa_out.at[qa_idx, "spatial_issue"] = str(spatial_issue.loc[addr_idx])
            qa_out.at[qa_idx, "note"] = str(addr.loc[addr_idx, note_col]) if pd.notna(addr.loc[addr_idx, note_col]) else ""

        os.makedirs(os.path.dirname(os.path.abspath(output_gpkg)), exist_ok=True)
        qa_out.to_file(output_gpkg, layer=layer, driver="GPKG")

    matched_count = int(join_result.mapping.notna().sum())
    print(f"join_strategy={join_result.strategy}")
    print(f"matched_rows={matched_count}/{len(addr)}")
    print(f"coordinate_columns={x_col},{y_col}")
    print(f"note_column={note_col}")
    print(f"output_csv={output_csv}")
    if output_gpkg:
        print(f"output_gpkg={output_gpkg}")


if __name__ == "__main__":
    main()

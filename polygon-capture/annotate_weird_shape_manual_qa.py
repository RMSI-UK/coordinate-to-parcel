from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from _core.config import add_config_argument, get_config_section_from_argv, require_configured


LABELS = {
    95640: (
        "fail_keep_failed",
        "Empty failure output; nearest eligible WFS is too far.",
    ),
    90512: (
        "review_likely_geocode_mismatch",
        "Original says High Street Beighton S19, but geocoder output is High Street Meadowhall Centre S9 and polygon is huge Meadowhall.",
    ),
    112190: (
        "review_likely_geocode_mismatch",
        "Original Corporation/Bridge Street, but geocoder output is Sheffield City Council Licensing Team Staniforth Road.",
    ),
    93405: (
        "review_out_of_area_or_geocode",
        "Original/geocoder mentions Rotherham; low-compact fallback polygon should be checked.",
    ),
    103087: (
        "review_possible_overcapture",
        "8 Charles Street captured the larger Cambridge/Division/Carver land block; could be overcapture unless application site is the block.",
    ),
}

PLOT_KEYS = {125140, 108954, 95392, 120873, 122218, 118757, 109001}

LIKELY_TERMS = [
    "golf",
    "park",
    "woods",
    "meadowhall",
    "norfolk",
    "recreation ground",
    "playing field",
    "land bordered",
    "land bounded",
    "land between",
    "land adjacent",
    "land opposite",
    "driving range",
    "sports club",
    "ski village",
    "crystal peaks",
    "barclays bank wholesale",
    "parkway drive",
    "kilner way",
    "don road",
    "river don",
    "attercliffe",
    "thorncliffe estate",
    "firth brown",
    "hague plant",
    "scott street",
    "petre street",
    "effingham",
    "whirlow",
    "graves",
]


def parse_args() -> argparse.Namespace:
    config_defaults, _ = get_config_section_from_argv(
        "annotate_weird_shape_manual_qa",
        include_package_defaults=True,
    )
    parser = argparse.ArgumentParser(
        description="Annotate weird-shape QA rows with manual labels.",
        argument_default=argparse.SUPPRESS,
    )
    add_config_argument(parser)
    parser.add_argument("--input-gpkg")
    parser.add_argument("--input-layer")
    parser.add_argument("--output-gpkg")
    parser.add_argument("--output-layer")
    parser.add_argument("--output-csv")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    require_configured(
        args,
        ("input_gpkg", "output_gpkg", "output_csv"),
        "annotate_weird_shape_manual_qa",
    )
    return args


def classify(row) -> tuple[str, str]:
    unique_key = int(row["unique_key"])
    if unique_key in LABELS:
        return LABELS[unique_key]
    if unique_key in PLOT_KEYS:
        return (
            "review_fragmented_plot_union",
            "Multipart linked parent union for plot range; likely plausible but should be map-checked against listed plot ranges.",
        )

    address = str(row.get("original_address", "") or "").lower()
    geocode = str(row.get("geocoding_final_output_address", "") or "").lower()
    flags = str(row.get("shape_qa_flags", "") or "")

    if any(term in address or term in geocode for term in LIKELY_TERMS):
        return (
            "likely_ok_named_large_or_complex_site",
            "Large/complex shape matches named site or land description.",
        )
    if "many_holes" in flags and int(row.get("qa_holes", 0) or 0) >= 20:
        return (
            "review_many_holes_complex_site",
            "Many holes; usually a retail/industrial complex but worth visual check.",
        )
    if "low_compactness" in flags:
        return (
            "review_low_compactness",
            "Low compactness shape; check for strip/road-like overcapture.",
        )
    return ("review_shape", "Flagged by shape metrics; visual check recommended.")


def main() -> None:
    args = parse_args()
    q = gpd.read_file(args.input_gpkg, layer=args.input_layer)
    labels = q.apply(classify, axis=1, result_type="expand")
    labels.columns = ["manual_shape_qa_label", "manual_shape_qa_note"]
    q[["manual_shape_qa_label", "manual_shape_qa_note"]] = labels

    out = Path(args.output_gpkg)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    q.to_file(out, layer=args.output_layer, driver="GPKG")

    csv = Path(args.output_csv)
    csv.parent.mkdir(parents=True, exist_ok=True)
    q.drop(columns="geometry").to_csv(csv, index=False)

    print(f"[DONE] Wrote {out}")
    print(f"[DONE] Wrote {csv}")
    print(q["manual_shape_qa_label"].value_counts().to_string())
    cols = [
        "unique_key",
        "original_address",
        "shape_qa_flags",
        "manual_shape_qa_label",
        "manual_shape_qa_note",
    ]
    print(q[cols].sort_values(["manual_shape_qa_label", "unique_key"]).to_string(index=False))


if __name__ == "__main__":
    main()

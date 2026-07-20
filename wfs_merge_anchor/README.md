# wfs_merge_anchor

`wfs_merge_anchor` is the anchor-first rewrite of the native WFS merge workflow.

The first stage is deliberately independent from ML and council land. It turns
the raw WFS layer into a deterministic one-layer clean base map:

- read `sheffield_wfs_raw.gpkg:polygons_in_buffers`;
- keep `building`, `land`, and small `Roads Tracks And Paths` polygons;
- remove large road polygons while retaining narrow paths, tracks, and small road
  pieces;
- make geometries valid and keep polygonal parts only;
- explode multipart geometries into stable single parts;
- drop empty, unrecoverable, and tiny polygon slivers below `0.25 m²` by default;
- remove overlaps by priority: building first, then smaller polygons first, then
  original `source_fid` as the stable tie-breaker.
- fill small polygon-internal holes and enclosed gaps as synthetic polygons with
  `Theme=building_or_land`.

The de-overlap step cuts lower-priority polygons by the already accepted
higher-priority geometry. This is different from the old cleanup, which mostly
dropped near-duplicates and could leave partial overlaps in the base layer.

Default road filtering uses a simple width proxy, `2 * area / perimeter`, after
multipart geometries have been exploded. The default retained road thresholds
are:

- path/steps/footbridges: width `<= 4.5m`, area `<= 5000m²`;
- tracks: width `<= 6m`, area `<= 5000m²`;
- ordinary road-or-track pieces: width `<= 5m`, minimum-rotated-rectangle
  width `<= 8m`, area `<= 1200m²`;
- roadside pieces: width `<= 3m`, area `<= 800m²`.

Pass `--keep-large-roads` to disable this filter.

After de-overlap, small holes are filled in two passes:

1. Polygon-internal holes are filled when the hole is not occupied by another
   polygon.
2. Holes in the combined coverage are filled when they are bounded by at least
   two existing polygons.

The default fill rule is:

- area `>= 0.25m²` and `<= 250m²`;
- at least two surrounding polygons;
- each surrounding polygon must share at least `0.05m` of boundary.

The surrounding-polygon rule applies to the combined-coverage pass. Polygon
internal holes only need to be empty of other polygon area.

Pass `--skip-polygon-hole-fill` to disable the polygon-internal pass.
Pass `--skip-enclosed-gap-fill` to disable this stage. Gap-fill polygons use
negative `source_fid` values and `GmlID=gapfill_polygon_hole_*` or
`GmlID=gapfill_enclosed_*`.

Example smoke run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python wfs_merge_anchor/preprocess_wfs_raw.py \
  --max-features 5000 \
  --validate-overlaps \
  --overwrite \
  --output-gpkg /data/sheffield/spatial/base-map/tmp/wfs_merge_anchor_preprocess_smoke/sheffield_wfs_raw_clean_smoke.gpkg
```

Default production output:

```text
/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg
```

By default the output GPKG contains only `wfs_raw_clean`. Pass
`--write-debug-layers` when diagnostic excluded/overlap layers are needed.

## End-To-End Workflow

The current production workflow is driven by:

```bash
python wfs_merge_anchor/run_workflow.py
```

By default this reuses existing outputs and only builds missing stages. Use
`--force` to rebuild stage outputs, and `--run-preprocess --force` to rebuild
from raw WFS.

```bash
# Reuse existing clean WFS, rebuild anchor/council/fallback outputs.
python wfs_merge_anchor/run_workflow.py --skip-preprocess --force

# Full raw-WFS-to-fallback rebuild.
python wfs_merge_anchor/run_workflow.py --run-preprocess --force

# Show commands without running them.
python wfs_merge_anchor/run_workflow.py --dry-run --force
```

Default outputs:

```text
/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg
/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean_anchor.gpkg
/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_area05.gpkg
/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_fallback.gpkg
/data/sheffield/spatial/base-map/wfs_merge_anchor_workflow.summary.json
```

### Stage 1: Clean WFS

Script:

```bash
python wfs_merge_anchor/preprocess_wfs_raw.py
```

This creates `sheffield_wfs_raw_clean.gpkg:wfs_raw_clean`.

### Stage 2: Anchor Layer

Script:

```bash
python wfs_merge_anchor/build_anchor_layer.py
```

Anchor rule:

- `Theme` contains `building` or `land`;
- polygon area is at or below `4000 m²` by default;
- polygon intersects one or more distinct UPRN points.

Use `--max-anchor-area` to override the area gate, or pass `0` to disable it.
The area gate only controls which polygons can become anchors. If a geocoded
point has no eligible anchor, the end-to-end selector falls back to the
intersecting `wfs_raw_clean` polygon for that point. If a selected parcel still
does not intersect the geocoded point, the selector switches to the fallback
merge for the preprocessed `wfs_raw_clean` polygon intersecting the point; when
that fallback row is unavailable, it emits the raw clean polygon itself.

Default output:

```text
/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean_anchor.gpkg:wfs_raw_clean_anchor
```

The production run currently produces `142,657` anchors.

When an anchor output already exists, `build_anchor_layer.py` preserves the
existing `clean_fid -> GeoPackage FID` order by default. This keeps QGIS/manual
debug references stable across `--force` rebuilds. Pass
`--no-preserve-existing-fid-order` only when you intentionally want a fresh
output order.

### Stage 3: Council Single-Anchor Reference

Script:

```bash
python wfs_merge_anchor/build_council_single_anchor.py
```

Council polygon rule:

- intersecting anchor overlap area must be at least `0.5 m2`;
- exactly one anchor may satisfy that threshold.
- after the council single-anchor set is selected, the stage looks for a
  regular union of one or more `wfs_raw_clean` building/land polygons whose IoU
  with the council polygon is at least `0.90`;
- when such a WFS union exists, the output geometry is replaced by that union
  and the row is marked with `output_geometry_source=wfs_clean_iou_match`;
- otherwise the original council geometry is retained with
  `output_geometry_source=council`.

Default output:

```text
/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_area05.gpkg:council_polygons_single_anchor_area05
```

The production run currently produces `90,302` council single-anchor polygons
covering `86,296` unique anchors. In the current output, `56,177` rows use a
WFS-clean IoU geometry match and `34,125` retain the council geometry.
The point selector only treats the WFS-clean IoU matched rows as usable council
outputs. Retained raw council geometries are kept for diagnostics and cause the
case to use the WFS fallback merge instead.

### Stage 4: Fallback Merge

Script:

```bash
python wfs_merge_anchor/build_single_anchor_fallback.py
```

Fallback applies to anchors that have no qualifying council single-anchor
polygon. It merges eligible zero-UPRN WFS polygons using:

- all-anchor primary ownership by `shared_edge / anchor_perimeter`;
- close secondary direct claims for fallback anchors when a non-fallback anchor
  is only slightly more attractive;
- k0 direct and k1 one-hop indirect candidates;
- regularity-first candidate selection;
- strong completion for low-hull-gap, no-hole parcel completions that are
  visually regular even when the single building anchor is more rectangular.

Fallback is area-adaptive. Anchors below `500 m2` keep the active completion
behaviour because they are often only the building core of the final parcel.
Anchors at or above `500 m2` keep the anchor-only geometry unless a merge either
improves the shape or has strong shared-edge evidence with only small shape
loss. This prevents already substantial anchors from absorbing small edge
fragments that do not materially improve the parcel.

Default output:

```text
/data/sheffield/spatial/base-map/sheffield_council_polygons_single_anchor_fallback.gpkg:council_polygons_single_anchor_fallback
```

The current production run produces:

- fallback anchor rows: `56,361`;
- rows with added polygons: `41,219`;
- `regularity_band_completion`: `37,073`;
- `strong_completion`: `4,146`;
- `anchor_only`: `15,142`.

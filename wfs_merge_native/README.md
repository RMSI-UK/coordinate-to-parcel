# Native WFS Merge Pipeline

This folder contains the current production native-WFS merge workflow.

Production inference code lives here. Model-training and feature-building scripts live in:

```text
wfs_merge_native_train/
```

Production inference uses only raw WFS polygons, UPRN points, and trained model
artifacts. Council land, manual labels, and old result layers belong in
`wfs_merge_native_train/` for training/evaluation label construction, not in
the production geometry path.

## Current Raw-Anchor Model Workflow

The current model-first production path is:

```bash
/env/venv/textual/bin/python wfs_merge_native/run_raw_anchor_group_pipeline.py \
  --bbox 431000,386000,431200,386200 \
  --output-gpkg /data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_preview.gpkg
```

This runner performs two steps:

1. Preprocess raw WFS into a clean, de-overlapped WFS layer.
2. Apply the anchor candidate proposal model plus the anchor group scorer.

It uses raw WFS + UPRN + model artifacts only. It has no council/reference
input. A bbox is required by default so raw full-area runs do not happen
accidentally; pass `--allow-full-raw` only for an intentional full production
run. The current default apply path uses `--anchor-workers 16` on this
machine.

Output layers:

```text
predicted_parcels
predicted_parcels_merged_only
```

Validated default model artifacts:

```text
/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/wfs_raw_anchor_group_model_v1.joblib
/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_proposal_model_full_sampled_v1/wfs_raw_anchor_candidate_proposal_model_v1.joblib
```

Heldout evidence and speed smoke results are tracked in:

```text
wfs_merge_native_train/ANCHOR_GROUP_MODEL_STATUS.md
```

## Legacy Multi-Stage Pipeline

`run_pipeline.py` is the older multi-stage native pipeline. Keep it for
comparison and for historical QA, but do not treat it as the current
raw-anchor model workflow.

The pipeline runs:

1. Edge model merge
2. Completion model
3. Operation prune-only cleanup
4. Local-mode overmerge split
5. Anchor gate + anchor group repair
6. Final output-hole + enclosed-small-gap fallback fill

The final cleanup assigns enclosed gaps first, then runs output-hole fill last.
Small enclosed coverage gaps are filled even when they contain UPRN points, so
address points do not remain in uncovered voids between otherwise complete
source polygons. Output-hole fill then removes all final polygon interior rings
to match the no-hole native output.
`pred_uprn_count` is recomputed from the final display geometry so coverage-gap
fills that capture UPRN points are reflected in the final QA fields.

Native geometry is built from the raw WFS source layer. If a reference layer is
provided explicitly, it is carried only as QA/debug labels; it does not provide
copied final geometry and it is not required for the production run.

It writes two GeoPackages:

```text
wfs_raw_merged_native.gpkg      clean delivery output, one layer only
wfs_raw_merged_native_log.gpkg  full QA/debug output with all intermediate layers
```

The clean delivery layer is:

```text
wfs_raw_merged_native
```

The same final geometries are also available in the log GeoPackage's internal
QA layer:

```text
predicted_parcels_with_uprn
```

## Run

```bash
/env/venv/textual/bin/python wfs_merge_native/run_pipeline.py \
  --wfs-gpkg /data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg \
  --wfs-layer polygons_in_buffers \
  --output-gpkg /data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native.gpkg \
  --log-gpkg /data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native_log.gpkg
```

Default Sheffield inputs:

```text
WFS:       /data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg
WFS layer: polygons_in_buffers
Reference: disabled by default
UPRN:      /data/base-data/osopenuprn_202602.gpkg
```

Default model artifacts:

```text
/data/sheffield/spatial/base-map/tmp/wfs_merge_edge_model_v1
/data/sheffield/spatial/base-map/tmp/wfs_merge_completion_model_v3
/data/sheffield/spatial/base-map/tmp/wfs_merge_operation_models_v1
/data/sheffield/spatial/base-map/tmp/wfs_merge_full_ml_v2_user_feedback/anchor_problem_detection_model_probe_v1.joblib
/data/sheffield/spatial/base-map/tmp/wfs_merge_native_train_anchor_group_final_selector_light/anchor_group_repair_model_v1.joblib
```

## Output

The delivery GeoPackage contains exactly one layer by default:

```text
wfs_raw_merged_native.gpkg
└── wfs_raw_merged_native
```

The log GeoPackage contains the original multi-layer pipeline output:

```text
wfs_raw_merged_native
predicted_parcels_with_uprn
anchor_group_repair_selected
anchor_group_repair_review_candidates
anchor_need_repair_candidates_top
neighborhood_overmerge_split_components
neighborhood_overmerge_split_removed_edges
neighborhood_overmerge_split_results
final_gap_fill_output_holes
final_gap_fill_enclosed_gaps
final_gap_fill_skipped_uprn_holes
final_gap_fill_changed_parcels
possible_false_positive_clusters_with_uprn
possible_split_reference_clusters_with_uprn
prediction_source_polygons
```

A pipeline summary is written next to the output:

```text
<log-output>.pipeline_summary.json
```

## Current Production Thresholds

```text
edge threshold:              0.90
edge shape guard threshold:  0.95
completion threshold:        0.90
prune threshold:             0.80
local-mode split threshold:  0.76
anchor group threshold:      model default, 0.94 for final-selector
anchor need gate threshold:  legacy only, disabled for final-selector
anchor group candidate mode: light
anchor group enclosure:      pair-level proxy
residual fallback:           on, pair >= 0.90 and shared edge >= 6 m
complete-pool gate bypass:   legacy only, disabled for final-selector
UPRN skeleton gate bypass:   legacy only, disabled for final-selector
UPRN skeleton pair override: legacy only, disabled for final-selector
enclosed gap max area:       250 m2
enclosed gap min shared edge: 0.05 m
enclosed gap UPRN guard:     off by default
final UPRN count:            recomputed from final geometry
```

The operation stage deliberately disables the older zero-UPRN attachment and parcel completion layers. The current production fix for the under-merge cases is the anchor gate + group repair layer.
The edge shape guard keeps narrow same-reference zero-UPRN building-land edges when both polygons are WFS source polygons, the edge model score is at least 0.90, the shared edge is at least 6 m, and the pair remains in small/medium parcel scale.
The local-mode split layer handles small multi-UPRN overmerge cases whose area and UPRN count are out of line with the surrounding run of regular one-UPRN parcels.
The final fallback mirrors the legacy `wfs_merge.py` output cleanup for output holes: it fills interior rings in final parcel geometries. Small enclosed coverage gaps are assigned to the adjacent parcel with the strongest shared boundary, including UPRN-bearing gaps that would otherwise leave address points in uncovered voids. Final QA layers recompute `pred_uprn_count` from the final geometry.

## Hybrid Native + Council Final

When both native and council merge outputs are available, build the final hybrid layer with:

```bash
/env/venv/textual/bin/python wfs_merge_native/build_hybrid_wfs_merge.py \
  --raw-gpkg /data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg \
  --raw-layer polygons_in_buffers \
  --native-gpkg /data/sheffield/spatial/base-map/sheffield_wfs_merge_native.gpkg \
  --native-layer predicted_parcels_with_uprn \
  --council-gpkg /data/sheffield/spatial/base-map/sheffield_wfs_merge_council.gpkg \
  --council-layer os_wfs_merge \
  --output-gpkg /data/sheffield/spatial/base-map/sheffield_wfs_merge_final.gpkg \
  --overwrite
```

Stable final layer:

```text
wfs_merged_final
```

QA layers:

```text
native_retained
council_group_patches_accepted
council_group_patches_review
council_group_patches_rejected
```

The hybrid builder keeps native as the base layer. Council is only used as a grouping patch when it almost exactly covers multiple native fragments and does not look like it is over-merging already complete native parcels. A council patch is rejected when most of its covered native parcels are already complete-looking 1-UPRN parcels, even if the council/native coverage is exact.

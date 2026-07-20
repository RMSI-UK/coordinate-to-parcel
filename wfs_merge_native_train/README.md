# Native WFS Merge Training

This folder contains model-training and feature-building scripts for the native WFS merge pipeline.

Production inference lives in:

```text
wfs_merge_native/
```

Training and modeling code lives here:

```text
wfs_merge_native_train/
```

Council land, manual labels, and previous outputs may be used here to create
labels and evaluation slices. Runtime feature columns should remain derivable
from raw WFS + UPRN so the production pipeline can run without council land.

Current raw-anchor model scripts:

```text
preprocess_wfs_raw.py
build_wfs_raw_merged_council_train.py
train_wfs_raw_anchor_candidate_proposal_model.py
train_wfs_raw_anchor_group_model.py
score_wfs_raw_anchor_group_candidates.py
simulate_wfs_raw_anchor_candidate_budget.py
evaluate_wfs_raw_anchor_group_apply.py
audit_wfs_raw_anchor_group_splits.py
verify_wfs_raw_anchor_group_model.py
estimate_wfs_raw_anchor_runtime.py
verify_wfs_raw_anchor_workflow.py
```

The production pipeline imports a few feature functions from this folder so that training and inference use the same feature definitions. The runnable production entrypoint remains:

```bash
/env/venv/textual/bin/python wfs_merge_native/run_raw_anchor_group_pipeline.py --bbox minx,miny,maxx,maxy
```

The old parcel-completion training branch is not part of the current model-first production workflow. The current production path is one candidate proposal model plus one anchor group scorer and a deterministic conflict selector.

Current model status, heldout precision/recall, and bbox speed evidence live in:

```text
ANCHOR_GROUP_MODEL_STATUS.md
```

Run the current 95/95 gate with:

```bash
/env/venv/textual/bin/python wfs_merge_native_train/verify_wfs_raw_anchor_group_model.py
```

Audit train/heldout target-id disjointness with:

```bash
/env/venv/textual/bin/python wfs_merge_native_train/audit_wfs_raw_anchor_group_splits.py
```

Verify the production workflow evidence chain with:

```bash
/env/venv/textual/bin/python wfs_merge_native_train/verify_wfs_raw_anchor_workflow.py
```

Estimate apply-only full clean-WFS runtime from the current bbox speed evidence
with:

```bash
/env/venv/textual/bin/python wfs_merge_native_train/estimate_wfs_raw_anchor_runtime.py
```

## Raw WFS Preprocess

`preprocess_wfs_raw.py` prepares the clean WFS base map used by training and
QA experiments. It reads raw WFS polygons, keeps building/land/small-road
themes, removes coarse vehicle-road polygons, de-overlaps the coverage, and
fills polygon-internal holes plus enclosed coverage gaps.

Default output:

```text
/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg
```

Smoke example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/env/venv/textual/bin/python wfs_merge_native_train/preprocess_wfs_raw.py \
  --max-features 5000 \
  --validate-overlaps \
  --overwrite \
  --output-gpkg /data/sheffield/spatial/base-map/tmp/wfs_merge_native_train_preprocess_smoke/sheffield_wfs_raw_clean_smoke.gpkg
```

## Point Large Parcel Prototype

This branch prepares candidates for point-query large parcel assembly. It starts from a seed WFS polygon, builds a local graph, expands/refines candidate groups, completes small pocket fragments, and exports QA layers plus a candidate CSV.

Example seed:

```bash
/env/venv/textual/bin/python wfs_merge_native_train/prepare_point_large_parcel_training_dataset.py \
  --seed-fid 141507 \
  --manual-positive-fid-groups '140604|141507|141605|142154|167415|167589|167590|167591|167592|167615|167620|168172|168173|168297|168364|168578|168589|168604|168961|169121' \
  --output-dir /data/sheffield/spatial/base-map/tmp/wfs_merge_point_large_parcel_v1 \
  --output-name point_large_parcel_seed_141507_v2.gpkg \
  --max-group-area 12000 \
  --max-candidates 20000
```

The default `--max-group-area` is intentionally capped at `12000` square metres so query-time candidates cannot over-expand into very large surrounding land blocks. Raise it only for known campuses or estates that genuinely exceed that size.

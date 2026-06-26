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

Current training scripts:

```text
prepare_wfs_merge_training_dataset.py
train_wfs_merge_edge_model.py
train_wfs_merge_completion_model.py
train_wfs_merge_prune_model.py
train_wfs_merge_anchor_group_repair_model.py
prepare_point_large_parcel_training_dataset.py
train_point_large_parcel_model.py
```

The production pipeline imports a few feature functions from this folder so that training and inference use the same feature definitions. The runnable production entrypoint remains:

```bash
/env/venv/textual/bin/python wfs_merge_native/run_pipeline.py
```

The old parcel-completion training branch is not part of the current production workflow. The current production under-merge fix is the anchor need gate plus anchor group repair model.

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

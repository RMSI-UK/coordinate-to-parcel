# Native WFS Merge Production Input Config

This file records the production input data paths used by the native WFS merge
pipeline. Production inference uses raw WFS + UPRN + model artifacts. Reference
layers are optional QA labels only.

The pipeline entrypoint is:

```bash
/env/venv/textual/bin/python wfs_merge_native/run_pipeline.py
```

## sheffield_raw

Current production Sheffield raw-WFS configuration.

| Field | Value |
|---|---|
| council_key | `sheffield_raw` |
| WFS GPKG | `/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg` |
| WFS layer | `polygons_in_buffers` |
| Reference GPKG | disabled by default |
| Reference layer | `os_wfs_merge` if explicitly provided |
| UPRN GPKG | `/data/base-data/osopenuprn_202602.gpkg` |
| UPRN layer | `osopenuprn_address` |
| UPRN id field | `UPRN` |
| Default work dir | `/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline` |
| Default clean output GPKG | `/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native.gpkg` |
| Default log output GPKG | `/data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native_log.gpkg` |
| Final output layer | `wfs_raw_merged_native` |

Run command:

```bash
/env/venv/textual/bin/python wfs_merge_native/run_pipeline.py \
  --wfs-gpkg /data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg \
  --wfs-layer polygons_in_buffers \
  --uprn-gpkg /data/base-data/osopenuprn_202602.gpkg \
  --uprn-layer osopenuprn_address \
  --uprn-id-field UPRN \
  --work-dir /data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline \
  --output-gpkg /data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native.gpkg \
  --log-gpkg /data/sheffield/spatial/base-map/tmp/wfs_merge_native_pipeline/wfs_raw_merged_native_log.gpkg
```

## Add A New Council

Copy this block and fill in the paths.

```text
## <council_key>

| Field | Value |
|---|---|
| council_key | `<council_key>` |
| WFS GPKG | `<path-to-wfs-polygons.gpkg>` |
| WFS layer | `<wfs-layer-name>` |
| Reference GPKG | optional QA/reference GPKG |
| Reference layer | optional reference layer |
| UPRN GPKG | `<path-to-uprn.gpkg>` |
| UPRN layer | `<uprn-layer-name>` |
| UPRN id field | `UPRN` |
| Default work dir | `<path-to-temp-work-dir>` |
| Default clean output GPKG | `<path-to-output-dir>/wfs_raw_merged_native.gpkg` |
| Default log output GPKG | `<path-to-output-dir>/wfs_raw_merged_native_log.gpkg` |
| Final output layer | `wfs_raw_merged_native` |
```

Run command template:

```bash
/env/venv/textual/bin/python wfs_merge_native/run_pipeline.py \
  --wfs-gpkg <path-to-wfs-polygons.gpkg> \
  --wfs-layer <wfs-layer-name> \
  --uprn-gpkg <path-to-uprn.gpkg> \
  --uprn-layer <uprn-layer-name> \
  --uprn-id-field UPRN \
  --work-dir <path-to-temp-work-dir> \
  --output-gpkg <path-to-output-dir>/wfs_raw_merged_native.gpkg \
  --log-gpkg <path-to-output-dir>/wfs_raw_merged_native_log.gpkg
```

## Notes

- The final geometry is built from raw WFS source polygon unions.
- The council/reference layer is optional and only populates QA/debug reference IDs when explicitly passed.
- The UPRN layer is required for the residential anchor logic and multi-UPRN safety checks.
- `wfs_raw_merged_native.gpkg` is a clean one-layer delivery file. `wfs_raw_merged_native_log.gpkg` keeps the full original multi-layer QA/debug output.
- Final cleanup fills small enclosed coverage gaps even when they contain UPRN points, then fills output interior rings last.
- Final QA fields recompute `pred_uprn_count` from the final geometry so coverage-gap fills are reflected in UPRN counts.
- Keep intermediate files under a council-specific `tmp` directory so large CSV/GPKG artifacts do not mix between councils.

## sheffield_final_hybrid

Final hybrid merge built from the raw WFS, native merge output, and council merge output.

| Field | Value |
|---|---|
| council_key | `sheffield_final_hybrid` |
| Raw WFS GPKG | `/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg` |
| Raw WFS layer | `polygons_in_buffers` |
| Native merge GPKG | `/data/sheffield/spatial/base-map/sheffield_wfs_merge_native.gpkg` |
| Native merge layer | `predicted_parcels_with_uprn` |
| Council merge GPKG | `/data/sheffield/spatial/base-map/sheffield_wfs_merge_council.gpkg` |
| Council merge layer | `os_wfs_merge` |
| Default output GPKG | `/data/sheffield/spatial/base-map/sheffield_wfs_merge_final.gpkg` |
| Final output layer | `wfs_merged_final` |

Run command:

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

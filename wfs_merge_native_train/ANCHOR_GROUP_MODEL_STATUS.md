# WFS Raw Anchor Group Model Status

Last verified: 2026-07-02

Verified production apply entry:

```bash
python wfs_merge_native/apply_wfs_raw_anchor_group_model.py
```

Verified raw-WFS production wrapper:

```bash
python wfs_merge_native/run_raw_anchor_group_pipeline.py --bbox minx,miny,maxx,maxy
```

Verified 95/95 gate:

```bash
python wfs_merge_native_train/verify_wfs_raw_anchor_group_model.py
```

Verified split disjointness:

```bash
python wfs_merge_native_train/audit_wfs_raw_anchor_group_splits.py
```

Verified production workflow evidence:

```bash
python wfs_merge_native_train/verify_wfs_raw_anchor_workflow.py
```

Current default workflow:

- WFS input: `/data/sheffield/spatial/base-map/sheffield_wfs_raw_clean.gpkg:wfs_raw_clean`
- UPRN input: `/data/base-data/osopenuprn_202602.gpkg:osopenuprn_address`
- Scorer model: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/wfs_raw_anchor_group_model_v1.joblib`
- Proposal model: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_candidate_proposal_model_full_sampled_v1/wfs_raw_anchor_candidate_proposal_model_v1.joblib`
- Threshold: `0.005`
- Proposal expanded candidates: `3000`
- Proposal keep per anchor: `80`
- Include base candidates: `true`
- Full-score candidates per anchor: `96`
- Anchor workers: `16` by default on this machine, with automatic fallback to `1` if multiprocessing `fork` is unavailable

Heldout evidence against `sheffield_wfs_raw_merged_council_train.gpkg`:

| Split | Candidate budget | Threshold | Precision | Available recall | Evidence CSV |
| --- | ---: | ---: | ---: | ---: | --- |
| `train_component_id % 20 == 0` | 96 | 0.005 | 0.952964 | 0.959073 | `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/heldout_mod20r0_base_candidate_budget_sweep.csv` |
| `train_component_id % 20 == 5` | 96 | 0.005 | 0.950369 | 0.951772 | `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/heldout_mod20r5_base_candidate_budget_sweep.csv` |

BBox smoke evidence:

- BBox: `431000,386000,432000,387000`
- Current default parallel output: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/default_full_bbox_parallel16_431000_386000_432000_387000.gpkg`
- Layers: `predicted_parcels`, `predicted_parcels_merged_only`
- Runtime: `0:10.32` wall time, internal `elapsed_seconds=9.388303685002029`, for this bbox with `--debug-layer-limit 0` and `anchor_workers=16`
- The `anchor_workers=16` selected CSV matches the `anchor_workers=8` selected CSV byte-for-byte.
- Smaller pure-model preview bbox `431000,386000,431500,386500`: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/pure_model_preview_current_bbox_431000_386000_431500_386500.gpkg`; raw-wrapper runtime `3.08s`, apply runtime `1.45s`, selected groups `105`, output parcels `629`.
- PNG preview: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/pure_model_preview_current_bbox_431000_386000_431500_386500.png`
- Raw-to-model wrapper smoke bbox `431000,386000,431200,386200`: runtime `0:03.60`, raw WFS rows `196`, clean rows `186`, anchors `43`, scored candidates `3,025`, selected groups `24`, output parcels `144`.
- Raw-to-model wrapper smoke output: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/raw_anchor_pipeline_smoke_bbox_431000_386000_431200_386200.gpkg`
- Raw-to-model wrapper smoke with internal elapsed fields: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/raw_anchor_pipeline_smoke_elapsed_bbox_431000_386000_431200_386200.raw_anchor_pipeline_summary.json`; wrapper elapsed `2.59s`, apply elapsed `1.25s`.
- Apply-only full clean-WFS runtime estimate: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/wfs_raw_anchor_runtime_estimate.json`; estimated `18.74471383551106` minutes from clean WFS rows and the 16-worker 1km bbox evidence. This does not include full raw-WFS preprocessing time.

Verification gate output:

- `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/wfs_raw_anchor_group_95_95_verification.json`
- Gate: threshold `0.005`, candidate budget `96`, min precision `0.95`, min available recall `0.95`
- Result: pass. Minimum observed precision `0.9503685503685504`; minimum observed available recall `0.9517716535433072`.
- Verification report now records model provenance and file hashes. Current scorer model sha256: `2004517fb12a6f0b15ce1d68a05dd5990251cae2602afbf187d2886cbe8eeee3`; feature count: `142`.
- Scorer training candidate caches in the model payload used `proposal_expanded_candidate_limit=1500`; production/default candidate generation uses `proposal_expanded_candidate_limit=3000` to improve candidate coverage before the same scorer and conflict selector run.
- Split audit output: `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/wfs_raw_anchor_group_split_audit.json`
- Split audit result: pass. Training candidate target ids use `train_component_id % 20 in {1,2,3,4,6,7,8,9}`; heldout target ids use `{0,5}`; overlap target id count is `0`.
- The 95/95 verification JSON embeds the split audit result, so the pass report now covers both metric threshold and train/heldout disjointness.

Production workflow verification:

- `/data/sheffield/spatial/base-map/tmp/wfs_raw_anchor_group_model_proposal_mod20r1_4_6_9_v1_hardneg_p8_o4/wfs_raw_anchor_workflow_verification.json`
- Result: pass.
- Checks include: quality gate pass, split audit pass, production scripts contain no `council`/`reference`/`gapfill_council` tokens, raw wrapper refuses no-bbox runs, raw smoke uses `sheffield_wfs_raw.gpkg`, output layers are `predicted_parcels` and `predicted_parcels_merged_only`, raw smoke elapsed is under `10s`, and the 1km bbox output uses the current default model.
- The workflow check also requires the apply-only runtime estimate to be under the configured `30` minute target.

Notes:

- Do not run `/data/sheffield/spatial/base-map/sheffield_wfs_raw.gpkg` full during modeling.
- The current validated production path uses `sheffield_wfs_raw_clean.gpkg`, UPRN, the proposal model, and the scorer model.
- The raw-WFS wrapper preprocesses raw WFS to clean WFS and then runs the same model path. It has no council/reference input and refuses raw full-area runs unless `--allow-full-raw` is explicitly passed.
- Base-only production-style candidates are too narrow for r5: candidate positive-target coverage was about `0.9286`, below the `0.95` recall target before scoring.
- Narrow production top80 without base candidates did not pass r5 heldout: at threshold `0.005`, precision was about `0.9474` and available recall about `0.9488`.
- Lowering proposal expanded candidates to `1500`, even with base candidates, did not pass r5 heldout: at budget `96` and threshold `0.005`, precision was about `0.9484` and available recall about `0.9493`.
- Lowering proposal expanded candidates to `2500`, even with base candidates, was close but still did not pass r5 heldout at budget `96` and threshold `0.005`: precision was about `0.9499` and available recall about `0.9513`.
- The apply/train candidate generation reuses the expanded candidate prefix for base candidates, avoiding a second equivalent enumeration pass.
- The production apply and training candidate generation paths precompute cheap per-clean attributes for proposal scoring. This preserves selected candidates byte-for-byte on the bbox smoke while reducing runtime from about `1:36` to about `1:33`.
- The production apply path now parallelizes anchor scoring with `--anchor-workers` and defaults to `16` workers on this machine. This preserves selected candidates byte-for-byte on the same bbox while reducing runtime from about `1:33` to `0:10.32`.
- Candidate enumeration now uses an adjacency lookup for repeated shared-edge queries. The apply path also reuses one lookup across anchors; training/proposal scripts share the same optimized enumerator.
- Apply and raw-wrapper summaries now include `elapsed_seconds` so speed checks do not depend on shell timing logs.

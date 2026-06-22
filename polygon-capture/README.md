# Polygon Capture

This folder contains the polygon-capture stage of the point-to-polygon workflow.

## Public scripts

- `capture.py` - thin entrypoint for the main capture workflow. Reads target points/polygons, council land, and OS WFS polygons; writes the capture result GPKG.
- `postprocess_existing_wfs_theme.py` - post-process an existing capture output against WFS theme filters.
- `aggregate_unique_key.py` - collapse capture output to one parent record per `unique_key`.
- `qa_parent_unique.py` - build QA review layers for parent-unique capture output.
- `qa_weird_shapes.py` - flag unusual polygon shapes for manual QA.
- `annotate_weird_shape_manual_qa.py` - annotate weird-shape QA rows with manual labels.
- `build_final_parent_table.py` - assemble the final parent-unique delivery table.

## Internal helpers

The `_core/` package holds shared implementation modules used by the public scripts:

- `_core/io.py`
- `_core/inline_merge.py`
- `_core/workflow.py`
- `_core/wfs_merge.py`
- `_core/centroid_combo.py`
- `_core/polygonize_combo.py`

Path-like inputs and outputs should be supplied by config or command line, not by editing script defaults.
Shared defaults for layers, thresholds, and feature switches live in `_core/defaults.json`.
See `../CONFIG.md` and `../config.example.json`.

Run scripts with a config file, for example:

```bash
python3 polygon-capture/capture.py --config config.local.json
```

Individual CLI flags still override config values when needed.

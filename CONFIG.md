# Configuration

All public scripts accept a shared JSON config via `--config`.

```bash
python3 polygon-capture/capture.py --config config.local.json
python3 2_point_surrounding.py --config config.local.json
```

Lookup order:

1. `--config path/to/config.json`
2. `COORDINATE_TO_PARCEL_CONFIG`
3. `SPATIAL_CAPTURE_CONFIG`
4. `coordinate_to_parcel.config.json`, `spatial_capture_config.json`, or `config.json` in the current directory

Command-line flags override config values. Config keys use the same names as CLI flags, with either
dashes or underscores accepted. For example, `target_gpkg` and `target-gpkg` are equivalent.

Runtime paths belong in your local config. Algorithm defaults, layer defaults, thresholds, and boolean
switch defaults live in `polygon-capture/_core/defaults.json`, so the public scripts no longer carry
business-specific default paths or output names in code. Override any of those keys in your local config
when a council/batch needs different behavior.

Start from `config.example.json` and keep real council/batch paths in an untracked local config file.

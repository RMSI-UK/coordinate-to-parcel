#!/usr/bin/env python3
"""
Build 100m buffers for many EPSG:27700 points and download OS WFS polygons
intersecting the merged buffer layer.

Workflow:
1) Read many points from CSV/XLSX.
2) Build per-point buffer (default radius=100m).
3) Merge all buffers into one geometry layer.
4) Query OS Features WFS by sliding windows over merged buffer bbox.
5) Keep only polygons intersecting merged buffer geometry.
6) Save merged buffer layer + filtered polygon layer as GPKG (or GeoJSON by extension).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

WFS_URL = "https://api.os.uk/features/v1/wfs"
CONFIG_ENV_VARS = ("COORDINATE_TO_PARCEL_CONFIG", "SPATIAL_CAPTURE_CONFIG")
CONFIG_CANDIDATES = (
    "coordinate_to_parcel.config.json",
    "spatial_capture_config.json",
    "config.json",
)
POINT_SURROUNDING_DEFAULTS: Dict[str, Any] = {
    "api_key": None,
    "keys_file": None,
    "output_dir": None,
    "x_col": "",
    "y_col": "",
    "radius": 100.0,
    "type_names": "Topography_TopographicArea",
    "page_size": 100,
    "max_pages": 5000,
    "window_size": 800.0,
    "window_overlap": 100.0,
    "checkpoint_every_pages": 50,
    "timeout": 60,
    "resume_dir": None,
    "buffers_output": None,
    "polygons_output": None,
    "existing_polygons_input": None,
    "existing_polygons_layer": None,
    "coverage_input": None,
    "coverage_layer": None,
    "coverage_missing_output": None,
    "coverage_topup_rounds": 2,
    "coverage_topup_cell_size": 500.0,
    "coverage_topup_expand": 150.0,
    "coverage_topup_page_size": 100,
    "coverage_topup_max_pages": 30,
    "coverage_buffer_radius": None,
    "coverage_min_ratio": 0.995,
    "coverage_gap_min_area": 1.0,
    "disable_coverage_topup": False,
    "local_existing_fast_path": False,
    "local_existing_min_features": 1,
    "existing_read_bbox_expand": 0.0,
}
DEFAULT_PROFILE_JOB_ROOT = "/data/file-browser-data/spatial-jobs"


def _normalise_config_key(value: str) -> str:
    return str(value).strip().replace("-", "_")


def _normalise_config_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {_normalise_config_key(str(key)): item for key, item in value.items()}


def _discover_config_path(explicit_path: Optional[str] = None) -> Optional[Path]:
    if explicit_path:
        return Path(explicit_path).expanduser()

    for env_name in CONFIG_ENV_VARS:
        env_path = os.getenv(env_name)
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                return path

    for name in CONFIG_CANDIDATES:
        path = Path.cwd() / name
        if path.exists():
            return path
    return None


def _load_config(explicit_path: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    path = _discover_config_path(explicit_path)
    if path is None:
        return {}, None
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload, path


def _script_config(payload: Dict[str, Any], section: str) -> Dict[str, Any]:
    defaults = _normalise_config_dict(payload.get("defaults", {}))
    section_data = payload.get(section)
    if isinstance(section_data, dict):
        out = defaults
        out.update(_normalise_config_dict(section_data))
        return out
    return defaults


def _get_config_section_from_argv(section: str) -> Tuple[Dict[str, Any], Optional[Path]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    namespace, _ = parser.parse_known_args()
    payload, path = _load_config(getattr(namespace, "config", None))
    config = dict(POINT_SURROUNDING_DEFAULTS)
    config.update(_script_config(payload, section))
    return config, path


def _require_configured(args: argparse.Namespace, names: Tuple[str, ...], section: str) -> None:
    missing = []
    for name in names:
        value = getattr(args, name, None)
        if value is None or str(value).strip() == "":
            missing.append(name.replace("_", "-"))
    if missing:
        joined = ", ".join(f"--{name}" for name in missing)
        raise ValueError(f"Missing required {section} setting(s): {joined}. Provide them via --config or CLI.")


def _load_os_api_key_from_keys_file(keys_file: Optional[str]) -> Optional[str]:
    if not keys_file:
        return None
    root = Path(keys_file).expanduser()
    if not root.exists():
        return None

    patterns = [
        r"(?im)^\s*Ordnance\s+Survey\s+key\s*=\s*(\S+)\s*$",
        r"(?im)^\s*os\s*map\s*[:=]\s*(\S+)\s*$",
        r"(?im)^\s*os\s*[:=]\s*(\S+)\s*$",
    ]
    paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in patterns:
            m = re.search(pat, content)
            if m:
                return m.group(1).strip()
    return None


def parse_args() -> argparse.Namespace:
    config_defaults, _ = _get_config_section_from_argv("point_surrounding")
    parser = argparse.ArgumentParser(
        description="Create merged point buffers and download intersecting OS WFS polygons.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--area-profile",
        help="Optional AreaProfile JSON. Fills WFS coverage defaults without passing a council name.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to JSON config. If omitted, uses COORDINATE_TO_PARCEL_CONFIG/"
            "SPATIAL_CAPTURE_CONFIG or a config JSON in the current working directory."
        ),
    )
    parser.add_argument(
        "--input",
        help="Input CSV/XLSX/GeoJSON/SHP/GPKG file containing EPSG:27700 points.",
    )
    parser.add_argument(
        "--api-key",
        help="OS Data Hub API key. Default reads OS_API_KEY env var, then --keys-file.",
    )
    parser.add_argument(
        "--keys-file",
        help="Optional local key file path containing key entries (used only if env vars / --api-key are absent).",
    )
    parser.add_argument("--output-dir", help="Directory for inferred outputs when explicit output paths are omitted.")
    parser.add_argument("--x-col", help="Easting column name (optional for CSV/XLSX: auto-detect if omitted).")
    parser.add_argument("--y-col", help="Northing column name (optional for CSV/XLSX: auto-detect if omitted).")
    parser.add_argument("--radius", type=float, help="Buffer radius in meters.")
    parser.add_argument("--type-names", help="WFS layer typeNames.")
    parser.add_argument("--page-size", type=int, help="WFS page size (count parameter).")
    parser.add_argument("--max-pages", type=int, help="Safety cap for max pages per window.")
    parser.add_argument("--window-size", type=float, help="Sliding window size in meters.")
    parser.add_argument("--window-overlap", type=float, help="Window overlap in meters.")
    parser.add_argument(
        "--checkpoint-every-pages",
        type=int,
        help="Write checkpoint every N pages during download.",
    )
    parser.add_argument("--timeout", type=int, help="HTTP timeout seconds.")
    parser.add_argument(
        "--resume-dir",
        help="Resume working directory. Defaults to <output-dir>/resume.",
    )
    parser.add_argument(
        "--buffers-output",
        help="Output path for merged buffers (GPKG/GeoJSON by extension).",
    )
    parser.add_argument(
        "--polygons-output",
        help="Output path for polygons intersecting merged buffers (GPKG/GeoJSON by extension).",
    )
    parser.add_argument(
        "--existing-polygons-input",
        help=(
            "Optional existing WFS polygons GPKG/SHP/GeoJSON used as the starting coverage set. "
            "If omitted, --polygons-output is reused when it already exists."
        ),
    )
    parser.add_argument(
        "--existing-polygons-layer",
        help="Layer name for --existing-polygons-input when it is a multi-layer datasource.",
    )
    parser.add_argument(
        "--coverage-input",
        help="Optional vector input used for coverage check (point buffers / polygons must be covered by downloaded polygons).",
    )
    parser.add_argument(
        "--coverage-layer",
        help="Layer name for --coverage-input when it is a multi-layer datasource (e.g. GPKG).",
    )
    parser.add_argument(
        "--coverage-missing-output",
        help="Output path for coverage-check misses after top-up (GPKG/GeoJSON by extension).",
    )
    parser.add_argument(
        "--coverage-topup-rounds",
        type=int,
        help="Max rounds for coverage top-up download (default: 2).",
    )
    parser.add_argument(
        "--coverage-topup-cell-size",
        type=float,
        help="Grid cell size (m) used to cluster missing points for top-up requests.",
    )
    parser.add_argument(
        "--coverage-topup-expand",
        type=float,
        help="Expand distance (m) around each top-up request bbox.",
    )
    parser.add_argument(
        "--coverage-topup-page-size",
        type=int,
        help="WFS page size used in top-up requests (default: 1000).",
    )
    parser.add_argument(
        "--coverage-topup-max-pages",
        type=int,
        help="Max pages per top-up bbox request (default: 30).",
    )
    parser.add_argument(
        "--coverage-buffer-radius",
        type=float,
        help="Buffer radius (m) for point coverage checks. Defaults to --radius.",
    )
    parser.add_argument(
        "--coverage-min-ratio",
        type=float,
        help="Minimum WFS area coverage ratio for each point buffer / polygon before it is considered complete.",
    )
    parser.add_argument(
        "--coverage-gap-min-area",
        type=float,
        help="Ignore uncovered slivers smaller than this area in square metres.",
    )
    parser.add_argument(
        "--disable-coverage-topup",
        action=argparse.BooleanOptionalAction,
        help="Disable auto top-up when coverage check finds missing intersections.",
    )
    parser.add_argument(
        "--local-existing-fast-path",
        action=argparse.BooleanOptionalAction,
        help=(
            "If existing WFS polygons intersect the target buffer, emit that local subset immediately "
            "without strict coverage top-up. Intended for per-case file-browser runs."
        ),
    )
    parser.add_argument(
        "--local-existing-min-features",
        type=int,
        help="Minimum local existing polygons required for --local-existing-fast-path.",
    )
    parser.add_argument(
        "--existing-read-bbox-expand",
        type=float,
        help="Extra metres to expand the target bbox when reading existing polygons.",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    _apply_area_profile_defaults(args)
    _require_configured(args, ("input",), "point_surrounding")
    return args


def _load_area_profile(path: str) -> Any:
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from spatial_pipeline.area_profile import load_area_profile

    return load_area_profile(path)


def _has_value(args: argparse.Namespace, name: str) -> bool:
    value = getattr(args, name, None)
    return value is not None and str(value).strip() != ""


def _apply_area_profile_defaults(args: argparse.Namespace) -> None:
    profile_path = getattr(args, "area_profile", None)
    if not profile_path:
        return
    profile = _load_area_profile(str(profile_path))
    if not _has_value(args, "output_dir"):
        root = Path(os.environ.get("SPATIAL_PIPELINE_JOB_ROOT", DEFAULT_PROFILE_JOB_ROOT))
        args.output_dir = str(root / "manual" / profile.area_key / "point_surrounding")
    if not _has_value(args, "existing_polygons_input"):
        args.existing_polygons_input = profile.wfs_raw.path
    if not _has_value(args, "existing_polygons_layer"):
        args.existing_polygons_layer = profile.wfs_raw.layer
    if not _has_value(args, "coverage_buffer_radius"):
        args.coverage_buffer_radius = float(getattr(args, "radius", 100.0) or 100.0)


def _read_points_from_geojson(input_path: str) -> List[Tuple[float, float]]:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    features = payload.get("features", [])
    points: List[Tuple[float, float]] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            points.append((float(coords[0]), float(coords[1])))
        except (TypeError, ValueError):
            continue
    return points


def _read_points_from_vector(input_path: str) -> List[Tuple[float, float]]:
    gdf = _read_vector_gdf(input_path)
    points: List[Tuple[float, float]] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            points.append((float(geom.x), float(geom.y)))
        elif geom.geom_type == "MultiPoint":
            for p in geom.geoms:
                points.append((float(p.x), float(p.y)))
    return points


def _read_vector_gdf(
    input_path: str,
    layer: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Reading SHP/GPKG requires geopandas. Install with: pip install geopandas") from exc

    kwargs = {}
    if layer:
        kwargs["layer"] = layer
    if bbox is not None:
        kwargs["bbox"] = bbox
    try:
        return gpd.read_file(input_path, **kwargs)
    except TypeError:
        if bbox is None:
            raise
        kwargs.pop("bbox", None)
        return gpd.read_file(input_path, **kwargs)


def _read_points(input_path: str, x_col: str, y_col: str) -> List[Tuple[float, float]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext in {".shp", ".gpkg"}:
        points = _read_points_from_vector(input_path)
        if not points:
            raise ValueError("No valid Point geometries found in vector input.")
        return points
    if ext in {".geojson", ".json"}:
        points = _read_points_from_geojson(input_path)
        if not points:
            raise ValueError("No valid Point geometries found in GeoJSON input.")
        return points
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Only CSV/XLSX/GeoJSON/SHP/GPKG input is supported.")

    use_x = str(x_col or "").strip()
    use_y = str(y_col or "").strip()
    if use_x not in df.columns or use_y not in df.columns:
        x_candidates = [
            "api_x_27700",
            "api_easting_27700",
            "os_easting_27700",
            "top1_easting_27700",
            "gazetteer_x_27700",
            "easting_27700",
            "easting",
            "x",
        ]
        y_candidates = [
            "api_y_27700",
            "api_northing_27700",
            "os_northing_27700",
            "top1_northing_27700",
            "gazetteer_y_27700",
            "northing_27700",
            "northing",
            "y",
        ]
        use_x = next((c for c in x_candidates if c in df.columns), "")
        use_y = next((c for c in y_candidates if c in df.columns), "")
        if not use_x or not use_y:
            raise ValueError(
                f"Missing coordinate columns. Provided x/y=({x_col}, {y_col}). "
                f"Available columns: {list(df.columns)}"
            )
        print(f"Auto-detected coordinate columns: x={use_x}, y={use_y}")

    points: List[Tuple[float, float]] = []
    for x, y in zip(df[use_x], df[use_y]):
        if pd.isna(x) or pd.isna(y):
            continue
        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue

    if not points:
        raise ValueError("No valid points found in input.")

    return points


def _resolve_preferred_input(input_path: str) -> str:
    raw = str(input_path or "").strip()
    if not raw:
        raise ValueError("Input path is empty.")

    root, ext = os.path.splitext(raw)
    ext_l = ext.lower()

    if ext_l in {".csv", ".gpkg"}:
        preferred = f"{root}.gpkg"
        fallback = f"{root}.csv"
        if os.path.exists(preferred):
            return preferred
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"Input file not found (checked): {preferred}, {fallback}")

    if os.path.exists(raw):
        return raw

    # no extension given: prefer gpkg then csv
    preferred = f"{raw}.gpkg"
    fallback = f"{raw}.csv"
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Input file not found: {raw} (also checked {preferred}, {fallback})")


def _infer_council_base_map_dir(input_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "base-map")


def _infer_council_slug(input_path: str) -> str:
    normalized = os.path.abspath(input_path).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "data":
        return re.sub(r"[^a-z0-9]+", "", parts[1].strip().lower()) or "council"
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return re.sub(r"[^a-z0-9]+", "", stem.strip().lower()) or "council"


def _build_merged_buffers(points: Iterable[Tuple[float, float]], radius: float):
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Missing dependency 'shapely'. Install with: pip install shapely") from exc

    geoms = [Point(x, y).buffer(radius) for x, y in points]
    merged = unary_union(geoms)
    if merged.is_empty:
        raise ValueError("Merged buffer geometry is empty.")
    return geoms, merged


def _points_to_gdf(points: List[Tuple[float, float]], crs: str = "EPSG:27700"):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError("Point coverage checks require geopandas + shapely.") from exc
    geoms = [Point(x, y) for x, y in points]
    return gpd.GeoDataFrame({"_point_idx": list(range(len(points)))}, geometry=geoms, crs=crs)


def _coverage_buffer_radius(args: argparse.Namespace) -> float:
    radius = getattr(args, "coverage_buffer_radius", None)
    if radius is None:
        radius = args.radius
    radius = float(radius)
    if radius <= 0:
        raise ValueError("coverage-buffer-radius must be > 0")
    return radius


def _points_to_buffer_gdf(
    points: List[Tuple[float, float]],
    radius: float,
    crs: str = "EPSG:27700",
):
    pts_gdf = _points_to_gdf(points, crs=crs)
    out = pts_gdf.copy()
    out.geometry = out.geometry.buffer(radius)
    out["_coverage_source_geom"] = "Point"
    out["_coverage_buffer_radius_m"] = radius
    return out


def _target_bbox(target_gdf, expand_m: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
    if target_gdf is None or target_gdf.empty:
        return None
    targets = target_gdf.loc[target_gdf.geometry.notna()].copy()
    targets = targets.loc[~targets.geometry.is_empty].copy()
    if targets.empty:
        return None
    minx, miny, maxx, maxy = [float(value) for value in targets.total_bounds]
    expand = max(float(expand_m or 0.0), 0.0)
    return (minx - expand, miny - expand, maxx + expand, maxy + expand)


def _load_existing_polygon_output(
    path: str,
    layer: Optional[str] = None,
    target_gdf=None,
    bbox_expand_m: float = 0.0,
):
    if not path or not os.path.exists(path):
        return _features_to_gdf([])
    bbox = _target_bbox(target_gdf, bbox_expand_m)
    try:
        gdf = _read_vector_gdf(path, layer=layer, bbox=bbox)
    except Exception:
        return _features_to_gdf([])
    if gdf.empty:
        return gdf
    return _dedupe_polygon_gdf(gdf)


def _split_points_by_existing_coverage(
    points: List[Tuple[float, float]],
    polygon_gdf,
    radius: float,
    min_ratio: float,
    gap_min_area: float,
) -> Tuple[List[Tuple[float, float]], int]:
    if not points or polygon_gdf is None or polygon_gdf.empty:
        return points, 0

    buffer_gdf = _points_to_buffer_gdf(points, radius)
    covered_gdf = _annotate_wfs_area_coverage(
        check_gdf=buffer_gdf,
        polygon_gdf=polygon_gdf,
        min_ratio=min_ratio,
        gap_min_area=gap_min_area,
    )
    covered_idx = set(
        covered_gdf.loc[~covered_gdf["_coverage_missing"], "_point_idx"]
        .astype(int)
        .tolist()
    )
    pending = [pt for i, pt in enumerate(points) if i not in covered_idx]
    return pending, len(covered_idx)


def _safe_get_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception as exc:
        content_type = str(resp.headers.get("Content-Type", "")).strip()
        preview = (resp.text or "").strip().replace("\n", " ")[:160]
        raise RuntimeError(
            f"Invalid JSON response: {exc}; status={resp.status_code}; "
            f"content_type={content_type or 'unknown'}; preview={preview!r}"
        ) from exc


def _fetch_wfs_page(
    api_key: str,
    type_names: str,
    bbox: Tuple[float, float, float, float],
    start_index: int,
    page_size: int,
    timeout: int,
    max_retries: int = 6,
) -> Dict[str, Any]:
    minx, miny, maxx, maxy = bbox

    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": type_names,
        "outputFormat": "GeoJSON",
        "srsName": "EPSG:27700",
        "key": api_key,
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:27700",
        "count": page_size,
        "startIndex": start_index,
    }

    last_error = ""
    last_status: Optional[int] = None
    last_preview = ""
    for attempt in range(max_retries):
        try:
            resp = requests.get(WFS_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                last_status = resp.status_code
                last_preview = (resp.text or "").strip().replace("\n", " ")[:160]
                retry_after_raw = str(resp.headers.get("Retry-After", "2")).strip()
                try:
                    retry_after = int(retry_after_raw)
                except (TypeError, ValueError):
                    retry_after = min(2 ** attempt, 8)
                time.sleep(max(retry_after, 1))
                continue
            if 500 <= resp.status_code < 600:
                last_status = resp.status_code
                last_preview = (resp.text or "").strip().replace("\n", " ")[:160]
                time.sleep(min(2 ** attempt, 8))
                continue
            resp.raise_for_status()
            return _safe_get_json(resp)
        except RuntimeError as exc:
            # Upstream occasionally returns HTML/empty content with HTTP 200.
            last_error = str(exc)
            if attempt < max_retries - 1:
                wait_s = min(2 ** attempt, 8)
                print(
                    f"WFS non-JSON response at startIndex={start_index}; "
                    f"retry {attempt + 1}/{max_retries - 1} in {wait_s}s"
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"WFS request failed after JSON retries: {last_error}") from exc
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
            else:
                raise RuntimeError(f"WFS request failed: {last_error}") from exc

    if last_error:
        raise RuntimeError(f"WFS request failed: {last_error}")
    if last_status is not None:
        raise RuntimeError(
            f"WFS request failed after retries: last_status={last_status}; "
            f"response_preview={last_preview!r}"
        )
    raise RuntimeError("WFS request failed: no response/error captured")


def _write_page_geojson(
    pages_dir: str,
    signature_tag: str,
    window_idx: int,
    start_index: int,
    features: List[Dict[str, Any]],
) -> None:
    os.makedirs(pages_dir, exist_ok=True)
    out_path = os.path.join(pages_dir, f"page_{signature_tag}_w{window_idx}_s{start_index}.geojson")
    fc = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "EPSG:27700"},
        },
        "features": features,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def _preferred_page_path(pages_dir: str, signature_tag: str, window_idx: int, start_index: int) -> str:
    return os.path.join(pages_dir, f"page_{signature_tag}_w{window_idx}_s{start_index}.geojson")


def _legacy_page_path(pages_dir: str, window_idx: int, start_index: int) -> str:
    return os.path.join(pages_dir, f"page_w{window_idx}_s{start_index}.geojson")


def _find_existing_page_path(
    pages_dir: str,
    signature_tag: str,
    window_idx: int,
    start_index: int,
) -> Optional[str]:
    preferred = _preferred_page_path(pages_dir, signature_tag, window_idx, start_index)
    if os.path.exists(preferred):
        return preferred

    legacy = _legacy_page_path(pages_dir, window_idx, start_index)
    if os.path.exists(legacy):
        return legacy

    # Reuse previously downloaded accumulated batches if same window/start page exists.
    pattern = os.path.join(pages_dir, f"page_*_w{window_idx}_s{start_index}.geojson")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    return None


def _load_page_feature_count(page_path: str) -> int:
    try:
        with open(page_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        feats = payload.get("features", [])
        return len(feats) if isinstance(feats, list) else 0
    except Exception:
        return -1


def _load_page_features(pages_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(pages_dir):
        return []

    page_files = [
        f for f in os.listdir(pages_dir)
        if f.startswith("page_") and f.endswith(".geojson")
    ]
    page_files.sort()

    unique: Dict[str, Dict[str, Any]] = {}
    all_features: List[Dict[str, Any]] = []
    for page_file in page_files:
        page_path = os.path.join(pages_dir, page_file)
        with open(page_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for feat in payload.get("features", []):
            key = _feature_key(feat)
            if key in unique:
                continue
            unique[key] = feat
            all_features.append(feat)
    return all_features


def _feature_key(feat: Dict[str, Any]) -> str:
    fid = feat.get("id")
    if fid:
        return f"id:{fid}"
    geom = feat.get("geometry")
    if geom is not None:
        raw = json.dumps(geom, sort_keys=True, ensure_ascii=True)
        return "geom:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    props = feat.get("properties") or {}
    raw = json.dumps(props, sort_keys=True, ensure_ascii=True)
    return "prop:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_input_signature(
    input_path: str,
    x_col: str,
    y_col: str,
    radius: float,
    type_names: str,
    page_size: int,
    window_size: float,
    window_overlap: float,
    bbox: Tuple[float, float, float, float],
    point_count: int,
) -> str:
    payload = {
        "input_path": os.path.abspath(input_path),
        "x_col": x_col,
        "y_col": y_col,
        "radius": radius,
        "type_names": type_names,
        "page_size": page_size,
        "window_size": window_size,
        "window_overlap": window_overlap,
        "point_count": point_count,
        "bbox": [round(v, 3) for v in bbox],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_checkpoint(
    checkpoint_path: str,
    checkpoint_payload: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, ensure_ascii=False, indent=2)


def _new_checkpoint(input_signature: str, page_size: int) -> Dict[str, Any]:
    return {
        "input_signature": input_signature,
        "page_size": page_size,
        "completed_windows": [],
        "current_window_idx": None,
        "next_start_index": 0,
        "downloaded_count": 0,
        "completed": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _update_checkpoint_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def _window_grid(
    bbox: Tuple[float, float, float, float],
    window_size: float,
    overlap: float,
) -> List[Tuple[float, float, float, float]]:
    if window_size <= 0:
        raise ValueError("window-size must be > 0")
    if overlap < 0:
        raise ValueError("window-overlap must be >= 0")
    if overlap >= window_size:
        raise ValueError("window-overlap must be smaller than window-size")

    minx, miny, maxx, maxy = bbox
    step = window_size - overlap
    windows: List[Tuple[float, float, float, float]] = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            windows.append((x, y, min(x + window_size, maxx), min(y + window_size, maxy)))
            y += step
        x += step
    return windows


def _download_wfs_by_windows(
    api_key: str,
    type_names: str,
    windows: List[Tuple[float, float, float, float]],
    mask_geom,
    page_size: int,
    max_pages: int,
    timeout: int,
    pages_dir: str,
    checkpoint_path: str,
    input_signature: str,
    checkpoint_every_pages: int,
) -> bool:
    try:
        from shapely.geometry import box
    except ImportError as exc:
        raise ImportError("Missing dependency 'shapely'. Install with: pip install shapely") from exc

    checkpoint = _load_checkpoint(checkpoint_path)
    signature_tag = input_signature[:10]
    if checkpoint.get("input_signature") != input_signature:
        checkpoint = _new_checkpoint(input_signature, page_size)
        _save_checkpoint(checkpoint_path, checkpoint)

    completed_windows = set(int(i) for i in checkpoint.get("completed_windows", []))
    current_window_idx = checkpoint.get("current_window_idx")
    next_start_index = int(checkpoint.get("next_start_index", 0))
    downloaded_count = int(checkpoint.get("downloaded_count", 0))

    if completed_windows:
        print(f"Resume detected. Completed windows: {len(completed_windows)}/{len(windows)}")

    pages_since_checkpoint = 0
    for window_idx, window_bbox in enumerate(windows):
        if window_idx in completed_windows:
            continue

        if not box(*window_bbox).intersects(mask_geom):
            completed_windows.add(window_idx)
            checkpoint["completed_windows"] = sorted(completed_windows)
            checkpoint["current_window_idx"] = None
            checkpoint["next_start_index"] = 0
            _save_checkpoint(checkpoint_path, _update_checkpoint_timestamp(checkpoint))
            continue

        start_index = next_start_index if current_window_idx == window_idx else 0
        pages_done = 0
        while pages_done < max_pages:
            existing_page = _find_existing_page_path(
                pages_dir=pages_dir,
                signature_tag=signature_tag,
                window_idx=window_idx,
                start_index=start_index,
            )
            reused = False
            if existing_page:
                feature_count = _load_page_feature_count(existing_page)
                if feature_count >= 0:
                    page_features_count = feature_count
                    reused = True
                else:
                    existing_page = None

            if not existing_page:
                payload = _fetch_wfs_page(
                    api_key=api_key,
                    type_names=type_names,
                    bbox=window_bbox,
                    start_index=start_index,
                    page_size=page_size,
                    timeout=timeout,
                )
                page_features = payload.get("features", [])
                page_features_count = len(page_features)
                _write_page_geojson(pages_dir, signature_tag, window_idx, start_index, page_features)
                downloaded_count += page_features_count

            pages_done += 1

            checkpoint["completed_windows"] = sorted(completed_windows)
            checkpoint["current_window_idx"] = window_idx
            checkpoint["next_start_index"] = start_index + page_features_count
            checkpoint["downloaded_count"] = downloaded_count
            checkpoint["completed"] = False
            pages_since_checkpoint += 1
            if pages_since_checkpoint >= checkpoint_every_pages:
                _save_checkpoint(checkpoint_path, _update_checkpoint_timestamp(checkpoint))
                pages_since_checkpoint = 0

            print(
                f"Window {window_idx + 1}/{len(windows)} startIndex={start_index}: "
                f"{page_features_count} features "
                f"({'reused' if reused else 'downloaded'}, total downloaded {downloaded_count})"
            )

            if page_features_count == 0:
                completed_windows.add(window_idx)
                checkpoint["completed_windows"] = sorted(completed_windows)
                checkpoint["current_window_idx"] = None
                checkpoint["next_start_index"] = 0
                _save_checkpoint(checkpoint_path, _update_checkpoint_timestamp(checkpoint))
                pages_since_checkpoint = 0
                break

            start_index += page_features_count
            time.sleep(0.12)
        else:
            _save_checkpoint(checkpoint_path, _update_checkpoint_timestamp(checkpoint))
            print(f"Window {window_idx + 1} hit max-pages={max_pages}, will continue on next run.")
            return False

    checkpoint["completed_windows"] = sorted(completed_windows)
    checkpoint["current_window_idx"] = None
    checkpoint["next_start_index"] = 0
    checkpoint["completed"] = True
    _save_checkpoint(checkpoint_path, _update_checkpoint_timestamp(checkpoint))
    return True


def _reset_resume_state(resume_dir: str) -> None:
    pages_dir = os.path.join(resume_dir, "raw_pages")
    checkpoint_path = os.path.join(resume_dir, "checkpoint.json")
    if os.path.isdir(pages_dir):
        for name in os.listdir(pages_dir):
            if name.startswith("page_") and name.endswith(".geojson"):
                os.remove(os.path.join(pages_dir, name))
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def _filter_intersecting_polygons(features: List[Dict[str, Any]], mask_geom) -> List[Dict[str, Any]]:
    try:
        from shapely.geometry import shape
    except ImportError as exc:
        raise ImportError("Missing dependency 'shapely'. Install with: pip install shapely") from exc

    kept: List[Dict[str, Any]] = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            shp = shape(geom)
        except Exception:
            continue
        if shp.is_empty:
            continue
        if shp.intersects(mask_geom):
            kept.append(feat)
    return kept


def _write_geojson(path: str, features: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "EPSG:27700"},
        },
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def _write_features(path: str, features: List[Dict[str, Any]], gpkg_layer: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gpkg":
        gdf = _features_to_gdf(features)
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        gdf.to_file(path, layer=gpkg_layer, driver="GPKG")
        return
    _write_geojson(path, features)


def _features_to_gdf(features: List[Dict[str, Any]], crs: str = "EPSG:27700"):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Coverage check requires geopandas. Install with: pip install geopandas") from exc

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame.from_features(features, crs=crs)


def _gdf_to_features(gdf) -> List[Dict[str, Any]]:
    if gdf.empty:
        return []
    payload = json.loads(gdf.to_json(drop_id=True))
    return payload.get("features", [])


def _dedupe_polygon_gdf(gdf):
    if gdf.empty:
        return gdf
    out = gdf[gdf.geometry.notna()].copy()
    out = out[~out.geometry.is_empty].copy()
    out["_dedupe_key"] = None

    for col in ("GmlID", "TOID", "OBJECTID"):
        if col in out.columns:
            vals = out[col].astype("string")
            mask = out["_dedupe_key"].isna() & vals.notna() & (vals != "")
            out.loc[mask, "_dedupe_key"] = f"{col.lower()}:" + vals[mask]

    mask = out["_dedupe_key"].isna()
    if mask.any():
        out.loc[mask, "_dedupe_key"] = "geom:" + out.loc[mask, "geometry"].to_wkb(hex=True)

    out = out.drop_duplicates(subset=["_dedupe_key"]).drop(columns=["_dedupe_key"])
    return out


def _read_coverage_gdf(input_path: str, layer: Optional[str]):
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".gpkg", ".shp", ".geojson", ".json"}:
        raise ValueError("Coverage input must be one of GPKG/SHP/GeoJSON/JSON.")
    return _read_vector_gdf(input_path, layer=layer)


def _select_checkable_features(gdf):
    geom_type = gdf.geometry.geom_type.fillna("").str.upper()
    mask = gdf.geometry.notna() & geom_type.isin(["POINT", "MULTIPOINT", "POLYGON", "MULTIPOLYGON"])
    return gdf.loc[mask].copy()


def _prepare_coverage_check_gdf(gdf, point_buffer_radius: float):
    check_gdf = _select_checkable_features(gdf)
    if check_gdf.empty:
        return check_gdf

    if check_gdf.crs is None:
        check_gdf = check_gdf.set_crs("EPSG:27700")

    geom_type = check_gdf.geometry.geom_type.fillna("")
    check_gdf["_coverage_source_geom"] = geom_type
    check_gdf["_coverage_buffer_radius_m"] = None

    point_mask = geom_type.str.upper().isin(["POINT", "MULTIPOINT"])
    if point_mask.any():
        check_gdf.loc[point_mask, check_gdf.geometry.name] = (
            check_gdf.loc[point_mask].geometry.buffer(point_buffer_radius)
        )
        check_gdf.loc[point_mask, "_coverage_buffer_radius_m"] = point_buffer_radius

    return check_gdf


def _clean_area_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        return None
    if geom is None or geom.is_empty:
        return None
    return geom


def _annotate_wfs_area_coverage(
    check_gdf,
    polygon_gdf,
    min_ratio: float,
    gap_min_area: float,
):
    try:
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Coverage check requires shapely. Install with: pip install shapely") from exc

    if check_gdf.empty:
        return check_gdf
    if not (0.0 < float(min_ratio) <= 1.0):
        raise ValueError("coverage-min-ratio must be > 0 and <= 1")
    if float(gap_min_area) < 0.0:
        raise ValueError("coverage-gap-min-area must be >= 0")

    annotated = check_gdf.copy()
    if annotated.crs is None:
        annotated = annotated.set_crs("EPSG:27700")

    target_geoms = [_clean_area_geometry(geom) for geom in annotated.geometry]
    target_areas = [float(geom.area) if geom is not None else 0.0 for geom in target_geoms]

    covered_areas = [0.0 for _ in target_geoms]
    if polygon_gdf is not None and not polygon_gdf.empty:
        polygons = polygon_gdf.loc[polygon_gdf.geometry.notna()].copy()
        polygons = polygons.loc[~polygons.geometry.is_empty].copy()
        if annotated.crs and polygons.crs and annotated.crs != polygons.crs:
            polygons = polygons.to_crs(annotated.crs)
        elif polygons.crs is None and annotated.crs is not None:
            polygons = polygons.set_crs(annotated.crs)

        if not polygons.empty:
            sindex = polygons.sindex
            poly_geoms = list(polygons.geometry)
            total = len(target_geoms)
            for pos, target in enumerate(target_geoms):
                if target is None or target_areas[pos] <= 0.0:
                    continue
                try:
                    candidate_positions = list(sindex.query(target, predicate="intersects"))
                except TypeError:
                    candidate_positions = list(sindex.query(target))

                candidates = []
                for candidate_pos in candidate_positions:
                    candidate = _clean_area_geometry(poly_geoms[int(candidate_pos)])
                    if candidate is None:
                        continue
                    try:
                        if candidate.intersects(target):
                            candidates.append(candidate)
                    except Exception:
                        continue
                if not candidates:
                    continue

                try:
                    coverage_geom = candidates[0] if len(candidates) == 1 else unary_union(candidates)
                    covered_areas[pos] = float(coverage_geom.intersection(target).area)
                except Exception:
                    fixed = [_clean_area_geometry(candidate) for candidate in candidates]
                    fixed = [candidate for candidate in fixed if candidate is not None]
                    if fixed:
                        try:
                            coverage_geom = fixed[0] if len(fixed) == 1 else unary_union(fixed)
                            covered_areas[pos] = float(coverage_geom.intersection(target).area)
                        except Exception:
                            covered_areas[pos] = 0.0

                if total >= 5000 and (pos + 1) % 5000 == 0:
                    print(f"Coverage area check processed {pos + 1}/{total} targets")

    ratios = []
    missing_areas = []
    missing_flags = []
    for target_area, covered_area in zip(target_areas, covered_areas):
        if target_area <= 0.0:
            ratio = 1.0
            missing_area = 0.0
        else:
            covered_area = min(max(float(covered_area), 0.0), target_area)
            ratio = covered_area / target_area
            missing_area = max(0.0, target_area - covered_area)
        ratios.append(ratio)
        missing_areas.append(missing_area)
        missing_flags.append(ratio < min_ratio and missing_area > gap_min_area)

    annotated["_coverage_area"] = target_areas
    annotated["_covered_area"] = covered_areas
    annotated["_missing_area"] = missing_areas
    annotated["_coverage_ratio"] = ratios
    annotated["_coverage_missing"] = missing_flags
    return annotated


def _find_missing_coverage(
    check_gdf,
    polygon_gdf,
    min_ratio: float,
    gap_min_area: float,
):
    annotated = _annotate_wfs_area_coverage(
        check_gdf=check_gdf,
        polygon_gdf=polygon_gdf,
        min_ratio=min_ratio,
        gap_min_area=gap_min_area,
    )
    if annotated.empty:
        return annotated
    return annotated.loc[annotated["_coverage_missing"]].copy()


def _subset_polygons_to_targets(polygon_gdf, target_gdf):
    if polygon_gdf is None or polygon_gdf.empty or target_gdf is None or target_gdf.empty:
        return polygon_gdf
    try:
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Polygon subsetting requires shapely. Install with: pip install shapely") from exc

    targets = target_gdf.loc[target_gdf.geometry.notna()].copy()
    targets = targets.loc[~targets.geometry.is_empty].copy()
    if targets.empty:
        return polygon_gdf.iloc[0:0].copy()
    polygons = polygon_gdf.loc[polygon_gdf.geometry.notna()].copy()
    polygons = polygons.loc[~polygons.geometry.is_empty].copy()
    if polygons.empty:
        return polygons
    if targets.crs and polygons.crs and targets.crs != polygons.crs:
        polygons = polygons.to_crs(targets.crs)
    elif polygons.crs is None and targets.crs is not None:
        polygons = polygons.set_crs(targets.crs)

    mask_geom = unary_union([geom for geom in targets.geometry if geom is not None and not geom.is_empty])
    if mask_geom is None or mask_geom.is_empty:
        return polygons.iloc[0:0].copy()
    try:
        positions = list(polygons.sindex.query(mask_geom, predicate="intersects"))
    except TypeError:
        positions = list(polygons.sindex.query(mask_geom))
    if not positions:
        return polygons.iloc[0:0].copy()
    subset = polygons.iloc[sorted({int(pos) for pos in positions})].copy()
    subset = subset.loc[subset.geometry.intersects(mask_geom)].copy()
    return _dedupe_polygon_gdf(subset)


def _write_coverage_missing(path: str, missing_gdf) -> None:
    missing_features = _gdf_to_features(missing_gdf)
    _write_features(path, missing_features, gpkg_layer="missing_after_topup")
    print(f"Coverage missing features saved: {path} ({len(missing_features)})")


def _write_local_fast_path_outputs(args: argparse.Namespace, check_gdf, polygon_gdf) -> None:
    try:
        from shapely.geometry import mapping
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Local fast path requires shapely. Install with: pip install shapely") from exc

    target_geoms = [
        geom for geom in (_clean_area_geometry(geom) for geom in check_gdf.geometry)
        if geom is not None
    ]
    if target_geoms:
        merged = target_geoms[0] if len(target_geoms) == 1 else unary_union(target_geoms)
        buffer_features = [
            {
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "point_count": len(check_gdf),
                    "radius_m": float(args.coverage_buffer_radius or args.radius),
                    "source": "local_existing_fast_path",
                },
            }
        ]
        _write_features(args.buffers_output, buffer_features, gpkg_layer="merged_buffers")
        print(f"Saved merged buffers: {args.buffers_output}")

    final_gdf = _subset_polygons_to_targets(polygon_gdf, check_gdf)
    final_features = _gdf_to_features(final_gdf)
    _write_features(args.polygons_output, final_features, gpkg_layer="polygons_in_buffers")
    _write_coverage_missing(args.coverage_missing_output, check_gdf.iloc[0:0].copy())
    print(f"Local existing WFS fast path used; saved polygons: {args.polygons_output} ({len(final_features)})")


def _fetch_wfs_bbox_all_pages(
    api_key: str,
    type_names: str,
    bbox: Tuple[float, float, float, float],
    page_size: int,
    max_pages: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    all_features: List[Dict[str, Any]] = []
    start_index = 0
    for page in range(max_pages):
        payload = _fetch_wfs_page(
            api_key=api_key,
            type_names=type_names,
            bbox=bbox,
            start_index=start_index,
            page_size=page_size,
            timeout=timeout,
        )
        page_features = payload.get("features", [])
        returned = len(page_features)
        all_features.extend(page_features)
        if returned == 0:
            break
        start_index += returned
        time.sleep(0.05)
    return all_features


def _build_topup_boxes(missing_gdf, cell_size: float, expand_m: float) -> List[Tuple[float, float, float, float]]:
    if cell_size <= 0:
        raise ValueError("coverage-topup-cell-size must be > 0")
    if expand_m < 0:
        raise ValueError("coverage-topup-expand must be >= 0")

    boxes: List[Tuple[float, float, float, float]] = []
    geom_type = missing_gdf.geometry.geom_type.str.upper()

    point_like = missing_gdf.loc[geom_type.isin(["POINT", "MULTIPOINT"])].copy()
    if not point_like.empty:
        anchors = point_like.geometry.representative_point()
        point_like["_x"] = anchors.x
        point_like["_y"] = anchors.y
        point_like["_cx"] = (point_like["_x"] // cell_size).astype(int)
        point_like["_cy"] = (point_like["_y"] // cell_size).astype(int)
        grouped = point_like.groupby(["_cx", "_cy"], as_index=False)
        for _, grp in grouped:
            minx = float(grp["_x"].min() - expand_m)
            miny = float(grp["_y"].min() - expand_m)
            maxx = float(grp["_x"].max() + expand_m)
            maxy = float(grp["_y"].max() + expand_m)
            boxes.append((minx, miny, maxx, maxy))

    polygon_like = missing_gdf.loc[geom_type.isin(["POLYGON", "MULTIPOLYGON"])].copy()
    for geom in polygon_like.geometry:
        minx, miny, maxx, maxy = geom.bounds
        boxes.append((minx - expand_m, miny - expand_m, maxx + expand_m, maxy + expand_m))

    return boxes


def _run_coverage_topup(
    args: argparse.Namespace,
    check_gdf,
    polygon_gdf,
) -> Tuple[Any, Any]:
    min_ratio = float(args.coverage_min_ratio)
    gap_min_area = float(args.coverage_gap_min_area)
    missing = _find_missing_coverage(check_gdf, polygon_gdf, min_ratio, gap_min_area)
    if missing.empty:
        print(
            "Coverage check: all targets meet WFS area coverage "
            f"(min_ratio={min_ratio:.4f}, gap_min_area={gap_min_area:.2f} sqm)."
        )
        return polygon_gdf, missing

    print(
        f"Coverage check: initial under-covered targets = {len(missing)} "
        f"(min_ratio={min_ratio:.4f}, gap_min_area={gap_min_area:.2f} sqm)"
    )
    if args.disable_coverage_topup:
        print("Coverage top-up disabled by --disable-coverage-topup.")
        return polygon_gdf, missing

    seen_bboxes: set[Tuple[float, float, float, float]] = set()
    for round_idx in range(1, max(args.coverage_topup_rounds, 0) + 1):
        try:
            from shapely.ops import unary_union
        except ImportError as exc:
            raise ImportError("Coverage top-up requires shapely. Install with: pip install shapely") from exc

        missing_parts = [
            geom for geom in (_clean_area_geometry(geom) for geom in missing.geometry)
            if geom is not None
        ]
        if not missing_parts:
            print(f"Coverage top-up round {round_idx}: no missing buffer geometry generated.")
            break

        missing_mask_geom = unary_union(missing_parts)
        if missing_mask_geom is None or missing_mask_geom.is_empty:
            print(f"Coverage top-up round {round_idx}: merged missing buffer is empty.")
            break

        minx, miny, maxx, maxy = missing_mask_geom.bounds
        expand_m = float(args.coverage_topup_expand)
        merged_bbox = (minx - expand_m, miny - expand_m, maxx + expand_m, maxy + expand_m)
        bbox_key = tuple(round(float(value), 2) for value in merged_bbox)
        if bbox_key in seen_bboxes:
            print(
                f"Coverage top-up round {round_idx}: repeated bbox with no previous improvement; stop top-up."
            )
            break
        seen_bboxes.add(bbox_key)
        before_missing_count = len(missing)
        before_missing_area = (
            float(missing["_missing_area"].fillna(0).sum())
            if "_missing_area" in missing.columns
            else 0.0
        )
        windows = _window_grid(merged_bbox, args.window_size, args.window_overlap)
        topup_signature = _build_input_signature(
            input_path=f"{args.input}#coverage-topup-round-{round_idx}",
            x_col=args.x_col,
            y_col=args.y_col,
            radius=float(args.coverage_buffer_radius or args.radius),
            type_names=args.type_names,
            page_size=args.coverage_topup_page_size,
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            bbox=merged_bbox,
            point_count=len(missing),
        )
        topup_resume_dir = os.path.join(
            args.resume_dir,
            "coverage_topup",
            f"round_{round_idx}_{topup_signature[:10]}",
        )
        pages_dir = os.path.join(topup_resume_dir, "raw_pages")
        checkpoint_path = os.path.join(topup_resume_dir, "checkpoint.json")

        print(
            f"Coverage top-up round {round_idx}: "
            f"missing={len(missing)} merged_bbox="
            f"({merged_bbox[0]:.2f}, {merged_bbox[1]:.2f}, {merged_bbox[2]:.2f}, {merged_bbox[3]:.2f}) "
            f"windows={len(windows)}"
        )

        complete = _download_wfs_by_windows(
            api_key=args.api_key,
            type_names=args.type_names,
            windows=windows,
            mask_geom=missing_mask_geom,
            page_size=args.coverage_topup_page_size,
            max_pages=args.coverage_topup_max_pages,
            timeout=args.timeout,
            pages_dir=pages_dir,
            checkpoint_path=checkpoint_path,
            input_signature=topup_signature,
            checkpoint_every_pages=args.checkpoint_every_pages,
        )
        if not complete:
            raise RuntimeError(
                "Coverage top-up download not complete (hit max-pages on at least one window). "
                "Rerun to continue resume, or increase --coverage-topup-max-pages."
            )

        new_features = _load_page_features(pages_dir)
        if not new_features:
            print(f"Coverage top-up round {round_idx}: no new features fetched.")
            break

        before_filter = len(new_features)
        new_features = _filter_intersecting_polygons(new_features, missing_mask_geom)
        print(
            f"Coverage top-up round {round_idx}: "
            f"kept {len(new_features)}/{before_filter} fetched features intersecting merged missing buffers"
        )
        if not new_features:
            break

        new_gdf = _features_to_gdf(new_features)
        before = len(polygon_gdf)
        polygon_gdf = _dedupe_polygon_gdf(
            _features_to_gdf(_gdf_to_features(polygon_gdf) + _gdf_to_features(new_gdf))
        )
        added = max(0, len(polygon_gdf) - before)
        missing = _find_missing_coverage(check_gdf, polygon_gdf, min_ratio, gap_min_area)
        after_missing_area = (
            float(missing["_missing_area"].fillna(0).sum())
            if "_missing_area" in missing.columns
            else 0.0
        )
        print(
            f"Coverage top-up round {round_idx}: added={added} "
            f"total_polygons={len(polygon_gdf)} remaining_undercovered={len(missing)}"
        )
        if missing.empty:
            break
        if added == 0 and len(missing) >= before_missing_count and after_missing_area >= before_missing_area - 1e-6:
            print(
                f"Coverage top-up round {round_idx}: no new polygons and no coverage improvement; stop top-up."
            )
            break

    return polygon_gdf, missing


def main() -> None:
    args = parse_args()
    try:
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError("Missing dependency 'shapely'. Install with: pip install shapely") from exc

    api_key = args.api_key or _load_os_api_key_from_keys_file(args.keys_file)
    args.api_key = api_key

    resolved_input = _resolve_preferred_input(args.input)
    if resolved_input != args.input:
        print(f"Resolved --input: {args.input} -> {resolved_input}")
    args.input = resolved_input

    inferred_output_dir = args.output_dir or _infer_council_base_map_dir(args.input)
    council_slug = _infer_council_slug(args.input)
    if not hasattr(args, "coverage_min_ratio"):
        args.coverage_min_ratio = 0.995
    if not hasattr(args, "coverage_gap_min_area"):
        args.coverage_gap_min_area = 1.0
    coverage_radius = _coverage_buffer_radius(args)
    if not args.resume_dir:
        args.resume_dir = f"{inferred_output_dir}/resume"
    if not args.buffers_output:
        args.buffers_output = f"{inferred_output_dir}/{council_slug}_100m_buffers_merged.gpkg"
    if not args.polygons_output:
        args.polygons_output = f"{inferred_output_dir}/{council_slug}_polygons_in_buffers.gpkg"
    if not args.coverage_missing_output:
        args.coverage_missing_output = f"{inferred_output_dir}/{council_slug}_missing_after_topup.gpkg"

    all_points = _read_points(args.input, args.x_col, args.y_col)
    print(f"Valid input points: {len(all_points)}")

    coverage_source = args.coverage_input
    if not coverage_source:
        input_ext = os.path.splitext(args.input)[1].lower()
        if input_ext in {".gpkg", ".shp", ".geojson", ".json"}:
            coverage_source = args.input

    if coverage_source:
        coverage_gdf = _read_coverage_gdf(coverage_source, args.coverage_layer)
        check_gdf = _prepare_coverage_check_gdf(coverage_gdf, coverage_radius)
        print(
            f"Coverage check target buffers/polygons: {len(check_gdf)} "
            f"(from {coverage_source}; point_buffer_radius={coverage_radius}m)"
        )
    else:
        check_gdf = _points_to_buffer_gdf(all_points, coverage_radius)
        print(
            f"Coverage check target point buffers: {len(check_gdf)} "
            f"(from input points; radius={coverage_radius}m)"
        )

    existing_polygons_input = getattr(args, "existing_polygons_input", None) or args.polygons_output
    existing_polygons_layer = getattr(args, "existing_polygons_layer", None)
    existing_polygon_gdf = _load_existing_polygon_output(
        existing_polygons_input,
        layer=existing_polygons_layer,
        target_gdf=check_gdf,
        bbox_expand_m=float(getattr(args, "existing_read_bbox_expand", 0.0) or 0.0),
    )
    if existing_polygon_gdf is not None and not existing_polygon_gdf.empty:
        print(
            f"Existing polygons detected near target: {existing_polygons_input} "
            f"({len(existing_polygon_gdf)} features)"
        )
    if (
        bool(getattr(args, "local_existing_fast_path", False))
        and existing_polygon_gdf is not None
        and len(existing_polygon_gdf) >= max(int(getattr(args, "local_existing_min_features", 1) or 1), 1)
    ):
        local_subset = _subset_polygons_to_targets(existing_polygon_gdf, check_gdf)
        if local_subset is not None and len(local_subset) >= max(
            int(getattr(args, "local_existing_min_features", 1) or 1),
            1,
        ):
            _write_local_fast_path_outputs(args, check_gdf, local_subset)
            return

    pending_points, skipped_covered = _split_points_by_existing_coverage(
        points=all_points,
        polygon_gdf=existing_polygon_gdf,
        radius=coverage_radius,
        min_ratio=float(args.coverage_min_ratio),
        gap_min_area=float(args.coverage_gap_min_area),
    )
    if skipped_covered > 0:
        print(
            f"Pre-check fully covered point buffers skipped: {skipped_covered} "
            f"(radius={coverage_radius}m, min_ratio={float(args.coverage_min_ratio):.4f})"
        )
    print(f"Pending points for new WFS fetch: {len(pending_points)}")

    polygon_gdf = existing_polygon_gdf
    if pending_points:
        if not args.api_key:
            raise ValueError(
                "Missing API key. Use --api-key, set OS_API_KEY, or provide --keys-file with 'os map = ...'."
            )
        _, merged = _build_merged_buffers(pending_points, args.radius)
        merged_bbox = merged.bounds
        print(
            "Merged buffer bbox (EPSG:27700): "
            f"{merged_bbox[0]:.2f}, {merged_bbox[1]:.2f}, {merged_bbox[2]:.2f}, {merged_bbox[3]:.2f}"
        )

        buffer_features = [
            {
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "point_count": len(pending_points),
                    "radius_m": args.radius,
                },
            }
        ]
        _write_features(args.buffers_output, buffer_features, gpkg_layer="merged_buffers")
        print(f"Saved merged buffers: {args.buffers_output}")

        resume_dir = args.resume_dir
        pages_dir = os.path.join(resume_dir, "raw_pages")
        checkpoint_path = os.path.join(resume_dir, "checkpoint.json")
        input_signature = _build_input_signature(
            input_path=args.input,
            x_col=args.x_col,
            y_col=args.y_col,
            radius=args.radius,
            type_names=args.type_names,
            page_size=args.page_size,
            window_size=args.window_size,
            window_overlap=args.window_overlap,
            bbox=merged_bbox,
            point_count=len(pending_points),
        )
        windows = _window_grid(merged_bbox, args.window_size, args.window_overlap)
        print(f"Sliding windows: {len(windows)} (size={args.window_size}m overlap={args.window_overlap}m)")

        checkpoint = _load_checkpoint(checkpoint_path)
        existing_signature = checkpoint.get("input_signature")
        if existing_signature and existing_signature != input_signature:
            print("Input signature changed. Keep existing pages and start a new accumulated batch.")

        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.get("completed") and checkpoint.get("input_signature") == input_signature:
            print("Checkpoint says completed; skip download and reuse saved pages.")
        else:
            complete = _download_wfs_by_windows(
                api_key=args.api_key,
                type_names=args.type_names,
                windows=windows,
                mask_geom=merged,
                page_size=args.page_size,
                max_pages=args.max_pages,
                timeout=args.timeout,
                pages_dir=pages_dir,
                checkpoint_path=checkpoint_path,
                input_signature=input_signature,
                checkpoint_every_pages=args.checkpoint_every_pages,
            )
            if not complete:
                raise RuntimeError(
                    "Download not complete (hit max-pages on at least one window). "
                    "Rerun to continue resume, or increase --max-pages."
                )

        downloaded = _load_page_features(pages_dir)
        print(f"Total downloaded by windows (deduplicated): {len(downloaded)}")

        kept = _filter_intersecting_polygons(downloaded, merged)
        new_polygon_gdf = _dedupe_polygon_gdf(_features_to_gdf(kept))
        print(f"Polygons intersecting merged buffers (new): {len(new_polygon_gdf)}")
        if polygon_gdf is None or polygon_gdf.empty:
            polygon_gdf = new_polygon_gdf
        else:
            polygon_gdf = _dedupe_polygon_gdf(
                _features_to_gdf(_gdf_to_features(polygon_gdf) + _gdf_to_features(new_polygon_gdf))
            )
            print(f"Polygons after merge with existing output: {len(polygon_gdf)}")
    else:
        print("All points already covered by existing polygons output. Skip WFS fetch.")

    if not args.api_key and (polygon_gdf is None or polygon_gdf.empty):
        raise ValueError(
            "Missing API key. Use --api-key, set OS_API_KEY, or provide --keys-file with 'os map = ...'."
        )
    polygon_gdf, missing_gdf = _run_coverage_topup(args, check_gdf, polygon_gdf)
    _write_coverage_missing(args.coverage_missing_output, missing_gdf)

    polygon_gdf = _subset_polygons_to_targets(polygon_gdf, check_gdf)
    final_features = _gdf_to_features(polygon_gdf)
    _write_features(args.polygons_output, final_features, gpkg_layer="polygons_in_buffers")
    print(f"Saved polygons: {args.polygons_output} ({len(final_features)})")


if __name__ == "__main__":
    main()

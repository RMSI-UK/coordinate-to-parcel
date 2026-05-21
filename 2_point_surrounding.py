#!/usr/bin/env python3
"""
Build 50m buffers for many EPSG:27700 points and download OS WFS polygons
intersecting the merged buffer layer.

Workflow:
1) Read many points from CSV/XLSX.
2) Build per-point buffer (default radius=50m).
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

WFS_URL = "https://api.os.uk/features/v1/wfs"
DEFAULT_OUTPUT_DIR = "/data/south holland/spatial/base-map"
DEFAULT_INPUT = "/data/south holland/qa/Southholland/southholland_location_qa_9509.gpkg"
DEFAULT_COVERAGE_INPUT = None
DEFAULT_COVERAGE_LAYER = "southholland_location_qa_9509"
DEFAULT_KEYS_FILE = "/env/key/spatial_capture.keys.md"


def _load_os_api_key_from_keys_file(keys_file: Optional[str]) -> Optional[str]:
    if not keys_file or not os.path.exists(keys_file):
        return None
    try:
        with open(keys_file, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None

    patterns = [
        r"(?im)^\s*Ordnance\s+Survey\s+key\s*=\s*(\S+)\s*$",
        r"(?im)^\s*os\s*map\s*[:=]\s*(\S+)\s*$",
        r"(?im)^\s*os\s*[:=]\s*(\S+)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, content)
        if m:
            return m.group(1).strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create merged point buffers and download intersecting OS WFS polygons."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV/XLSX/GeoJSON/SHP/GPKG file containing EPSG:27700 points. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OS_API_KEY"),
        help="OS Data Hub API key. Default reads OS_API_KEY env var, then --keys-file.",
    )
    parser.add_argument(
        "--keys-file",
        default=DEFAULT_KEYS_FILE,
        help="Optional local key file path containing key entries (used only if env vars / --api-key are absent).",
    )
    parser.add_argument("--x-col", default="", help="Easting column name (optional for CSV/XLSX: auto-detect if omitted).")
    parser.add_argument("--y-col", default="", help="Northing column name (optional for CSV/XLSX: auto-detect if omitted).")
    parser.add_argument("--radius", type=float, default=50.0, help="Buffer radius in meters (default: 50).")
    parser.add_argument("--type-names", default="Topography_TopographicArea", help="WFS layer typeNames.")
    parser.add_argument("--page-size", type=int, default=100, help="WFS page size (count parameter).")
    parser.add_argument("--max-pages", type=int, default=5000, help="Safety cap for max pages per window.")
    parser.add_argument("--window-size", type=float, default=800.0, help="Sliding window size in meters.")
    parser.add_argument("--window-overlap", type=float, default=100.0, help="Window overlap in meters.")
    parser.add_argument(
        "--checkpoint-every-pages",
        type=int,
        default=50,
        help="Write checkpoint every N pages during download.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help=f"Resume working directory. Default: {DEFAULT_OUTPUT_DIR}/resume",
    )
    parser.add_argument(
        "--buffers-output",
        default=f"{DEFAULT_OUTPUT_DIR}/southholland_50m_buffers_merged.gpkg",
        help="Output path for merged buffers (GPKG/GeoJSON by extension).",
    )
    parser.add_argument(
        "--polygons-output",
        default=f"{DEFAULT_OUTPUT_DIR}/southholland_polygons_in_buffers.gpkg",
        help="Output path for polygons intersecting merged buffers (GPKG/GeoJSON by extension).",
    )
    parser.add_argument(
        "--coverage-input",
        default=DEFAULT_COVERAGE_INPUT,
        help="Optional vector input used for coverage check (Point/Polygon must intersect downloaded polygons).",
    )
    parser.add_argument(
        "--coverage-layer",
        default=DEFAULT_COVERAGE_LAYER,
        help="Layer name for --coverage-input when it is a multi-layer datasource (e.g. GPKG).",
    )
    parser.add_argument(
        "--coverage-missing-output",
        default=f"{DEFAULT_OUTPUT_DIR}/southholland_missing_after_topup.gpkg",
        help="Output path for coverage-check misses after top-up (GPKG/GeoJSON by extension).",
    )
    parser.add_argument(
        "--coverage-topup-rounds",
        type=int,
        default=2,
        help="Max rounds for coverage top-up download (default: 2).",
    )
    parser.add_argument(
        "--coverage-topup-cell-size",
        type=float,
        default=500.0,
        help="Grid cell size (m) used to cluster missing points for top-up requests.",
    )
    parser.add_argument(
        "--coverage-topup-expand",
        type=float,
        default=150.0,
        help="Expand distance (m) around each top-up request bbox.",
    )
    parser.add_argument(
        "--coverage-topup-page-size",
        type=int,
        default=1000,
        help="WFS page size used in top-up requests (default: 1000).",
    )
    parser.add_argument(
        "--coverage-topup-max-pages",
        type=int,
        default=30,
        help="Max pages per top-up bbox request (default: 30).",
    )
    parser.add_argument(
        "--disable-coverage-topup",
        action="store_true",
        help="Disable auto top-up when coverage check finds missing intersections.",
    )
    return parser.parse_args()


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


def _read_vector_gdf(input_path: str, layer: Optional[str] = None):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Reading SHP/GPKG requires geopandas. Install with: pip install geopandas") from exc

    if layer:
        return gpd.read_file(input_path, layer=layer)
    return gpd.read_file(input_path)


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
            "api_easting_27700",
            "os_easting_27700",
            "top1_easting_27700",
            "easting_27700",
            "easting",
            "x",
        ]
        y_candidates = [
            "api_northing_27700",
            "os_northing_27700",
            "top1_northing_27700",
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
    normalized = os.path.abspath(input_path).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    # expected shape like: /data/<council>/...
    if len(parts) >= 2 and parts[0].lower() == "data":
        council = parts[1]
        return f"/data/{council}/spatial/base-map"
    return DEFAULT_OUTPUT_DIR


def _infer_council_slug(input_path: str) -> str:
    normalized = os.path.abspath(input_path).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "data":
        # Keep slug compact and stable across naming variants (e.g. "south holland" -> "southholland").
        return re.sub(r"[^a-z0-9]+", "", parts[1].strip().lower()) or "council"
    return "council"


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


def _load_existing_polygon_output(path: str):
    if not path or not os.path.exists(path):
        return _features_to_gdf([])
    try:
        gdf = _read_vector_gdf(path, layer=None)
    except Exception:
        return _features_to_gdf([])
    if gdf.empty:
        return gdf
    return _dedupe_polygon_gdf(gdf)


def _split_points_by_existing_coverage(
    points: List[Tuple[float, float]],
    polygon_gdf,
) -> Tuple[List[Tuple[float, float]], int]:
    if not points or polygon_gdf is None or polygon_gdf.empty:
        return points, 0
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Coverage pre-check requires geopandas.") from exc

    pts_gdf = _points_to_gdf(points)
    if pts_gdf.crs and polygon_gdf.crs and str(pts_gdf.crs) != str(polygon_gdf.crs):
        pts_gdf = pts_gdf.to_crs(polygon_gdf.crs)
    joined = gpd.sjoin(pts_gdf[["_point_idx", "geometry"]], polygon_gdf[["geometry"]], how="left", predicate="intersects")
    covered_idx = set(joined.loc[joined["index_right"].notna(), "_point_idx"].astype(int).tolist())
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
            checkpoint["next_start_index"] = start_index + page_size
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

            start_index += page_size
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
    geom_type = gdf.geometry.geom_type.str.upper()
    mask = gdf.geometry.notna() & geom_type.isin(["POINT", "MULTIPOINT", "POLYGON", "MULTIPOLYGON"])
    return gdf.loc[mask].copy()


def _find_missing_intersections(check_gdf, polygon_gdf):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Coverage check requires geopandas. Install with: pip install geopandas") from exc

    if check_gdf.empty:
        return check_gdf
    if polygon_gdf.empty:
        return check_gdf

    if check_gdf.crs and polygon_gdf.crs and check_gdf.crs != polygon_gdf.crs:
        check_gdf = check_gdf.to_crs(polygon_gdf.crs)

    joined = gpd.sjoin(check_gdf[["geometry"]], polygon_gdf[["geometry"]], how="left", predicate="intersects")
    missing_idx = joined.index[joined["index_right"].isna()].unique()
    return check_gdf.loc[missing_idx].copy()


def _fetch_wfs_bbox_all_pages(
    api_key: str,
    type_names: str,
    bbox: Tuple[float, float, float, float],
    page_size: int,
    max_pages: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    all_features: List[Dict[str, Any]] = []
    for page in range(max_pages):
        start_index = page * page_size
        payload = _fetch_wfs_page(
            api_key=api_key,
            type_names=type_names,
            bbox=bbox,
            start_index=start_index,
            page_size=page_size,
            timeout=timeout,
        )
        page_features = payload.get("features", [])
        all_features.extend(page_features)
        if len(page_features) < page_size:
            break
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
    missing = _find_missing_intersections(check_gdf, polygon_gdf)
    if missing.empty:
        print("Coverage check: no missing intersections before top-up.")
        return polygon_gdf, missing

    print(f"Coverage check: initial missing intersections = {len(missing)}")
    if args.disable_coverage_topup:
        print("Coverage top-up disabled by --disable-coverage-topup.")
        return polygon_gdf, missing

    for round_idx in range(1, max(args.coverage_topup_rounds, 0) + 1):
        boxes = _build_topup_boxes(
            missing_gdf=missing,
            cell_size=args.coverage_topup_cell_size,
            expand_m=args.coverage_topup_expand,
        )
        if not boxes:
            print(f"Coverage top-up round {round_idx}: no request boxes generated.")
            break

        print(
            f"Coverage top-up round {round_idx}: "
            f"missing={len(missing)} request_boxes={len(boxes)}"
        )

        new_features: List[Dict[str, Any]] = []
        for box_idx, bbox in enumerate(boxes, start=1):
            fetched = _fetch_wfs_bbox_all_pages(
                api_key=args.api_key,
                type_names=args.type_names,
                bbox=bbox,
                page_size=args.coverage_topup_page_size,
                max_pages=args.coverage_topup_max_pages,
                timeout=args.timeout,
            )
            new_features.extend(fetched)
            if box_idx % 25 == 0 or box_idx == len(boxes):
                print(
                    f"  Top-up boxes processed {box_idx}/{len(boxes)}; "
                    f"fetched_raw={len(new_features)}"
                )

        if not new_features:
            print(f"Coverage top-up round {round_idx}: no new features fetched.")
            break

        new_gdf = _features_to_gdf(new_features)
        before = len(polygon_gdf)
        polygon_gdf = _dedupe_polygon_gdf(
            _features_to_gdf(_gdf_to_features(polygon_gdf) + _gdf_to_features(new_gdf))
        )
        added = max(0, len(polygon_gdf) - before)
        missing = _find_missing_intersections(check_gdf, polygon_gdf)
        print(
            f"Coverage top-up round {round_idx}: added={added} "
            f"total_polygons={len(polygon_gdf)} remaining_missing={len(missing)}"
        )
        if missing.empty:
            break

    return polygon_gdf, missing


def main() -> None:
    args = parse_args()
    try:
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError("Missing dependency 'shapely'. Install with: pip install shapely") from exc

    api_key = args.api_key or _load_os_api_key_from_keys_file(args.keys_file)
    if not api_key:
        raise ValueError(
            "Missing API key. Use --api-key, set OS_API_KEY, or provide --keys-file with 'os map = ...'."
        )
    args.api_key = api_key

    resolved_input = _resolve_preferred_input(args.input)
    if resolved_input != args.input:
        print(f"Resolved --input: {args.input} -> {resolved_input}")
    args.input = resolved_input

    inferred_output_dir = _infer_council_base_map_dir(args.input)
    council_slug = _infer_council_slug(args.input)
    arg_set = set(sys.argv[1:])
    if "--resume-dir" not in arg_set:
        args.resume_dir = f"{inferred_output_dir}/resume"
    if "--buffers-output" not in arg_set:
        args.buffers_output = f"{inferred_output_dir}/{council_slug}_50m_buffers_merged.gpkg"
    if "--polygons-output" not in arg_set:
        args.polygons_output = f"{inferred_output_dir}/{council_slug}_polygons_in_buffers.gpkg"
    if "--coverage-missing-output" not in arg_set:
        args.coverage_missing_output = f"{inferred_output_dir}/{council_slug}_missing_after_topup.gpkg"

    all_points = _read_points(args.input, args.x_col, args.y_col)
    print(f"Valid input points: {len(all_points)}")

    existing_polygon_gdf = _load_existing_polygon_output(args.polygons_output)
    if existing_polygon_gdf is not None and not existing_polygon_gdf.empty:
        print(f"Existing polygons output detected: {args.polygons_output} ({len(existing_polygon_gdf)} features)")
    pending_points, skipped_covered = _split_points_by_existing_coverage(all_points, existing_polygon_gdf)
    if skipped_covered > 0:
        print(f"Pre-check covered points skipped: {skipped_covered}")
    print(f"Pending points for new WFS fetch: {len(pending_points)}")

    polygon_gdf = existing_polygon_gdf
    if pending_points:
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

        resume_dir = args.resume_dir or f"{DEFAULT_OUTPUT_DIR}/resume"
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

    coverage_source = args.coverage_input
    if not coverage_source:
        input_ext = os.path.splitext(args.input)[1].lower()
        if input_ext in {".gpkg", ".shp", ".geojson", ".json"}:
            coverage_source = args.input

    if coverage_source:
        coverage_gdf = _read_coverage_gdf(coverage_source, args.coverage_layer)
        check_gdf = _select_checkable_features(coverage_gdf)
        print(
            f"Coverage check target features: {len(check_gdf)} "
            f"(from {coverage_source})"
        )
        polygon_gdf, missing_gdf = _run_coverage_topup(args, check_gdf, polygon_gdf)
        missing_features = _gdf_to_features(missing_gdf)
        _write_features(args.coverage_missing_output, missing_features, gpkg_layer="missing_after_topup")
        print(f"Coverage missing features saved: {args.coverage_missing_output} ({len(missing_features)})")
    else:
        check_gdf = _points_to_gdf(all_points)
        print(
            f"Coverage check target features: {len(check_gdf)} "
            "(from input points)"
        )
        polygon_gdf, missing_gdf = _run_coverage_topup(args, check_gdf, polygon_gdf)
        missing_features = _gdf_to_features(missing_gdf)
        _write_features(args.coverage_missing_output, missing_features, gpkg_layer="missing_after_topup")
        print(f"Coverage missing features saved: {args.coverage_missing_output} ({len(missing_features)})")

    final_features = _gdf_to_features(polygon_gdf)
    _write_features(args.polygons_output, final_features, gpkg_layer="polygons_in_buffers")
    print(f"Saved polygons: {args.polygons_output} ({len(final_features)})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


CONFIG_ENV_VARS = ("COORDINATE_TO_PARCEL_CONFIG", "SPATIAL_CAPTURE_CONFIG")
CONFIG_CANDIDATES = (
    "coordinate_to_parcel.config.json",
    "spatial_capture_config.json",
    "config.json",
)
PACKAGE_DEFAULTS = Path(__file__).with_name("defaults.json")


def _normalise_key(value: str) -> str:
    return str(value).strip().replace("-", "_")


def _normalise_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {_normalise_key(key): item for key, item in value.items()}


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help=(
            "Path to JSON config. If omitted, uses COORDINATE_TO_PARCEL_CONFIG/"
            "SPATIAL_CAPTURE_CONFIG or a config JSON in the current working directory."
        ),
    )


def discover_config_path(explicit_path: str | None = None) -> Path | None:
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


def load_config(explicit_path: str | None = None) -> tuple[dict[str, Any], Path | None]:
    path = discover_config_path(explicit_path)
    if path is None:
        return {}, None
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload, path


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload


def script_config(
    payload: dict[str, Any],
    section: str,
    *,
    allow_top_level: bool = False,
) -> dict[str, Any]:
    defaults = _normalise_dict(payload.get("defaults", {}))
    section_data = payload.get(section)
    if isinstance(section_data, dict):
        out = defaults
        out.update(_normalise_dict(section_data))
        return out
    if allow_top_level:
        out = _normalise_dict(payload)
        out.pop("defaults", None)
        return out
    return defaults


def get_config_section_from_argv(
    section: str,
    *,
    allow_top_level: bool = False,
    include_package_defaults: bool = False,
) -> tuple[dict[str, Any], Path | None]:
    parser = argparse.ArgumentParser(add_help=False)
    add_config_argument(parser)
    namespace, _ = parser.parse_known_args()
    payload, path = load_config(getattr(namespace, "config", None))
    config = {}
    if include_package_defaults:
        config.update(script_config(load_config_file(PACKAGE_DEFAULTS), section))
    config.update(script_config(payload, section, allow_top_level=allow_top_level))
    return config, path


def config_value(config: dict[str, Any], key: str, fallback: Any = None) -> Any:
    return config.get(_normalise_key(key), fallback)


def require_configured(args: argparse.Namespace, names: tuple[str, ...], section: str) -> None:
    missing = []
    for name in names:
        value = getattr(args, name, None)
        if value is None or str(value).strip() == "":
            missing.append(name.replace("_", "-"))
    if missing:
        joined = ", ".join(f"--{name}" for name in missing)
        raise ValueError(f"Missing required {section} setting(s): {joined}. Provide them via --config or CLI.")

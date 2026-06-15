# vfmgeom/config/utils.py

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in config:
        raise KeyError(f"Missing required config section: {section}")

    value = config[section]
    if not isinstance(value, dict):
        raise TypeError(f"Config section {section!r} must be a mapping.")

    return value


def get_required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]


def get_optional(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def make_experiment_output_dir(config: dict[str, Any]) -> Path:
    experiment = require_section(config, "experiment")
    paths = require_section(config, "paths")
    model = get_required(config, "model")

    name = get_required(experiment, "name")
    output_root = as_path(get_required(paths, "output_root"))
    encoder_id = model.get("encoder_id", "no_encoder_id")
    token_mode = model.get("token_mode", "no_token_mode")

    return output_root / str(name) / f"{encoder_id}_{token_mode}"

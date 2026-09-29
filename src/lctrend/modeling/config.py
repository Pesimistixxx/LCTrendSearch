"""Configuration of the modeling pipeline: defaults plus an override."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.config import load_catalog
from .dataset.labels import TRAINING_TARGETS


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: Optional[Path] = None, **overrides: Any
) -> Dict[str, Any]:
    """resources/training.json, then ``path``, then keyword overrides."""
    config = load_catalog("training")
    if path:
        config = _merge(
            config, json.loads(Path(path).read_text(encoding="utf-8-sig"))
        )
    config = _merge(
        config, {key: value for key, value in overrides.items() if value}
    )
    validate(config)
    return config


def validate(config: Dict[str, Any]) -> None:
    if config["target"] not in TRAINING_TARGETS:
        raise ValueError(f"target must be one of {list(TRAINING_TARGETS)}")
    if config["labels"].get("source", "auto") not in ("auto", "llm"):
        raise ValueError("labels.source must be 'auto' or 'llm'")
    fractions = config["split"]["fractions"]
    if (
        len(fractions) != 3
        or any(value <= 0 for value in fractions)
        or abs(sum(fractions) - 1) > 1e-6
    ):
        raise ValueError("split.fractions: three positive shares summing to 1")
    if config["split"].get("embargo", "none") not in ("none", "strict"):
        raise ValueError("split.embargo must be 'none' or 'strict'")
    if not 0 < config["features"]["max_abs_correlation"] <= 1:
        raise ValueError("features.max_abs_correlation must be in (0, 1]")
    if config["stacking"]["enabled"] and not config["hgt"]["enabled"]:
        raise ValueError("stacking needs hgt.enabled")
    if config["stacking"]["folds"] < 2:
        raise ValueError("stacking.folds must be at least 2")

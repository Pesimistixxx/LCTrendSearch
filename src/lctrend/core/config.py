"""Editable catalogs, kept separate from processing code and source
evidence.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"


def resource_path(name: str, suffix: str = ".json") -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or suffix not in (
        ".json",
        ".cypher",
    ):
        raise ValueError("Invalid catalog name")
    override = os.getenv("LCTREND_CONFIG_DIR")
    if override:
        candidate = Path(override).expanduser() / (name + suffix)
        if candidate.is_file():
            return candidate
    return RESOURCE_DIR / (name + suffix)


def load_catalog(name: str) -> Dict[str, Any]:
    """Load a complete JSON catalog; invalid overrides fail explicitly."""
    path = resource_path(name)
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load catalog {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Catalog {name} must contain a JSON object")
    return value


def load_environment(path: Optional[Path] = None) -> None:
    """Read the project's .env without overriding exported environment
    variables.
    """
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=path or Path.cwd() / ".env", override=False)
    # The container's env_file is fixed at creation. UI settings live on
    # persistent storage and must also be applied after a container restart.
    settings_file = os.getenv("LCTREND_SETTINGS_FILE")
    if settings_file:
        allowed = {
            "LLM_PROVIDER",
            "LLM_MODEL",
            "LLM_EXTRACT_MODEL",
            "LLM_REVIEW_MODEL",
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "GIGACHAT_BASE_URL",
            "GIGACHAT_CREDENTIALS",
        }
        for key, value in dotenv_values(settings_file).items():
            if key in allowed and value is not None:
                os.environ[key] = value


def cypher_identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", value
    ):
        raise ValueError("Invalid graph schema identifier")
    return value

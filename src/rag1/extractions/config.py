"""Load repository extraction settings from YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "extraction"


@lru_cache(maxsize=None)
def load_extraction_config(name: str) -> dict[str, Any]:
    """Load one named extraction configuration as a YAML mapping."""
    if name not in {"layouts", "ocr_text"}:
        raise ValueError(f"Unknown extraction configuration: {name!r}")
    path = _CONFIG_DIR / f"{name}.yml"
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Extraction configuration must be a mapping: {path}")
    return config

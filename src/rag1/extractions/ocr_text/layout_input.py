"""Load versioned layout JSON for OCR enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag1.extractions.layouts.contracts import CURRENT_SCHEMA_VERSION, LayoutDocument


class LayoutInputError(ValueError):
    """Raised when a layout JSON file is malformed or unsupported."""


def load_layout_document(path: str | Path) -> LayoutDocument:
    """Parse a supported layout document while rejecting duplicate proposal IDs."""
    source_path = Path(path)
    try:
        raw: Any = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LayoutInputError(f"cannot read layout JSON from {source_path}: {error}") from error
    if not isinstance(raw, dict):
        raise LayoutInputError("layout JSON root must be an object")
    version = raw.get("schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        raise LayoutInputError(
            f"unsupported layout schema version {version!r}; "
            f"expected {CURRENT_SCHEMA_VERSION!r}"
        )
    try:
        document = LayoutDocument.model_validate(raw)
    except ValidationError as error:
        raise LayoutInputError(f"invalid layout document: {error}") from error
    region_ids = [region.id for region in document.regions]
    if len(region_ids) != len(set(region_ids)):
        raise LayoutInputError("layout region IDs must be unique")
    return document

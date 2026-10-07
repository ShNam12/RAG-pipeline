"""Load versioned layout JSON for OCR enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag1.extractions.layouts.contracts import (
    CURRENT_SCHEMA_VERSION,
    LayoutDocument,
    VisualLocation,
)
from rag1.extractions.ocr_text.contracts import OcrPageMetadata


class LayoutInputError(ValueError):
    """Raised when a layout JSON file is malformed or unsupported."""


class _ManifestPage(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    page_number: int = Field(ge=1)
    rotation_degrees: int | None = None
    pixel_width: int | None = Field(default=None, gt=0)
    pixel_height: int | None = Field(default=None, gt=0)
    image: str | None = None
    bbox_image: str | None = None
    region_ids: list[str] | None = None


class _UpstreamManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    source: str = Field(min_length=1)
    dpi: int | None = Field(default=None, gt=0)
    pages: list[_ManifestPage] = Field(default_factory=list)


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


def load_upstream_manifest(
    path: str | Path,
    layout: LayoutDocument,
) -> tuple[int | None, list[OcrPageMetadata]]:
    """Load optional manifest metadata and validate its source and proposal links."""
    manifest_path = Path(path)
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _UpstreamManifest.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise LayoutInputError(f"invalid upstream manifest {manifest_path}: {error}") from error
    if manifest.source != layout.source:
        raise LayoutInputError("upstream manifest source does not match layout source")

    pages_by_number: dict[int, _ManifestPage] = {}
    regions_by_id = {region.id: region for region in layout.regions}
    for page in manifest.pages:
        if page.page_number in pages_by_number:
            raise LayoutInputError("upstream manifest page numbers must be unique")
        pages_by_number[page.page_number] = page
        if page.region_ids is not None:
            if len(page.region_ids) != len(set(page.region_ids)):
                raise LayoutInputError("upstream manifest region IDs must be unique per page")
            for region_id in page.region_ids:
                region = regions_by_id.get(region_id)
                if region is None:
                    raise LayoutInputError(
                        f"upstream manifest references unknown region ID {region_id!r}"
                    )
                location = region.location
                if not isinstance(location, VisualLocation):
                    raise LayoutInputError("upstream manifest region references must be visual")
                if location.page_number != page.page_number:
                    raise LayoutInputError(
                        f"upstream manifest region {region_id!r} has a mismatched page number"
                    )
        page_regions = [
            region for region in layout.regions
            if isinstance(region.location, VisualLocation)
            and region.location.page_number == page.page_number
        ]
        if page.region_ids is not None and set(page.region_ids) != {
            region.id for region in page_regions
        }:
            raise LayoutInputError(
                f"upstream manifest region references do not match page {page.page_number}"
            )
        for region in page_regions:
            location = region.location
            if page.pixel_width is not None and page.pixel_width != location.page_width:
                raise LayoutInputError(
                    f"upstream manifest width conflicts with proposal page {page.page_number}"
                )
            if page.pixel_height is not None and page.pixel_height != location.page_height:
                raise LayoutInputError(
                    f"upstream manifest height conflicts with proposal page {page.page_number}"
                )

    page_numbers = {
        region.location.page_number
        for region in layout.regions
        if isinstance(region.location, VisualLocation)
    }
    metadata = [
        OcrPageMetadata.model_validate(
            pages_by_number[page_number].model_dump()
            if page_number in pages_by_number
            else {"page_number": page_number}
        )
        for page_number in sorted(page_numbers | pages_by_number.keys())
    ]
    return manifest.dpi, metadata

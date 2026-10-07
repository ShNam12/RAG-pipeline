"""Resolve rendered page image files to 1-based page numbers."""

from __future__ import annotations

import re
from pathlib import Path


_PAGE_IMAGE = re.compile(r"^page-(\d+)\.png$")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def resolve_page_images(image_directory: str | Path) -> dict[int, Path]:
    """Return clean ``page-NNNN.png`` images keyed by their 1-based page number."""
    directory = Path(image_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"page image directory does not exist: {directory}")
    resolved: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.name.endswith("_bbox.png"):
            continue
        match = _PAGE_IMAGE.fullmatch(path.name)
        if match is None:
            continue
        page_number = int(match.group(1))
        if page_number < 1:
            raise ValueError(f"page image number must be 1-based: {path.name}")
        if page_number in resolved:
            raise ValueError(f"multiple page images resolve to page {page_number}")
        resolved[page_number] = path
    return dict(sorted(resolved.items()))


def resolve_direct_images(input_path: str | Path) -> dict[int, Path]:
    """Resolve one image or a directory of clean page images for direct OCR."""
    path = Path(input_path)
    if path.is_dir():
        images = resolve_page_images(path)
        if not images:
            raise FileNotFoundError(f"no page-NNNN.png images were found in {path}")
        return images
    if not path.is_file():
        raise FileNotFoundError(f"OCR image input does not exist: {path}")
    if path.suffix.lower() not in _IMAGE_SUFFIXES or path.stem.endswith("_bbox"):
        raise ValueError(f"direct OCR requires a clean image file: {path}")
    match = _PAGE_IMAGE.fullmatch(path.name)
    page_number = int(match.group(1)) if match else 1
    if page_number < 1:
        raise ValueError(f"page image number must be 1-based: {path.name}")
    return {page_number: path}

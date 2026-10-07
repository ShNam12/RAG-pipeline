"""Write document-scoped layout extraction artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rag1.extractions.layouts.contracts import LayoutDocument, VisualLocation
from rag1.extractions.layouts.paddle import RenderedPage


def document_output_path(output_root: str | Path, source: str | Path) -> Path:
    """Return a stable, source-specific output path under the given root."""
    source_path = Path(source)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._-")
    safe_stem = safe_stem or "document"
    source_key = hashlib.sha256(
        str(source_path.resolve()).casefold().encode("utf-8")
    ).hexdigest()[:12]
    return Path(output_root) / f"{safe_stem}-{source_key}"


@dataclass(frozen=True)
class LayoutArtifacts:
    """Paths written for one layout extraction run."""

    output_dir: Path
    manifest_path: Path
    layout_path: Path
    page_images: tuple[Path, ...]
    bbox_images: tuple[Path, ...]
    status: str


class LayoutArtifactWriter:
    """Create a document directory and serialize layout output and metadata."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def create_document_directory(self, source: str | Path) -> Path:
        """Create and return the stable output directory for a source document."""
        document_dir = document_output_path(self.output_root, source)
        document_dir.mkdir(parents=True, exist_ok=True)
        return document_dir

    def write(
        self,
        *,
        source: str | Path,
        document_dir: str | Path,
        document: LayoutDocument,
        dpi: int,
        pages: Sequence[RenderedPage],
        bbox_images_by_page: Mapping[int, str | Path],
    ) -> LayoutArtifacts:
        """Write the layout JSON and a manifest describing all page artifacts."""
        output_dir = Path(document_dir)
        expected_output_dir = document_output_path(self.output_root, source)
        if output_dir.resolve() != expected_output_dir.resolve():
            raise ValueError(
                "document_dir must be the source-specific directory under output_root"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
            raise ValueError("dpi must be a positive integer")

        page_numbers = {page.page_number for page in pages}
        if set(bbox_images_by_page) != page_numbers:
            raise ValueError("bbox image references must match rendered page numbers")

        page_images = tuple(
            self._checked_artifact_path(page.image, output_dir)
            for page in pages
        )
        bbox_images = tuple(
            self._checked_artifact_path(bbox_images_by_page[page.page_number], output_dir)
            for page in pages
        )
        regions_by_page: dict[int, list[str]] = {}
        for region in document.regions:
            if isinstance(region.location, VisualLocation):
                regions_by_page.setdefault(region.location.page_number, []).append(
                    region.id
                )

        page_entries = [
            {
                "page_number": page.page_number,
                "rotation_degrees": page.rotation_degrees,
                "pixel_width": page.page_width,
                "pixel_height": page.page_height,
                "image": image.relative_to(output_dir).as_posix(),
                "bbox_image": bbox_image.relative_to(output_dir).as_posix(),
                "region_count": len(regions_by_page.get(page.page_number, [])),
                "region_ids": regions_by_page.get(page.page_number, []),
            }
            for page, image, bbox_image in zip(pages, page_images, bbox_images, strict=True)
        ]

        layout_path = output_dir / "layout.json"
        layout_path.write_text(
            document.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "manifest_version": "1.0",
            "source": str(Path(source).resolve()),
            "source_format": document.source_format.value,
            "status": document.status.value,
            "dpi": dpi,
            "layout": layout_path.name,
            "page_count": len(pages),
            "region_count": len(document.regions),
            "error_count": len(document.errors),
            "pages": page_entries,
            "errors": [error.model_dump(mode="json") for error in document.errors],
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return LayoutArtifacts(
            output_dir=output_dir,
            manifest_path=manifest_path,
            layout_path=layout_path,
            page_images=page_images,
            bbox_images=bbox_images,
            status=document.status.value,
        )

    @staticmethod
    def _checked_artifact_path(path: str | Path, output_dir: Path) -> Path:
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Expected layout artifact does not exist: {artifact_path}")
        try:
            artifact_path.resolve().relative_to(output_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Layout artifact must be inside the document output directory: {artifact_path}"
            ) from error
        return artifact_path

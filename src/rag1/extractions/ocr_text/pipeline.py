"""Run text detection and recognition over versioned layout proposals."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from PIL import Image

from rag1.extractions.layouts.contracts import (
    DocumentStatus,
    PageFailure,
    Region,
    RegionKind,
    VisualLocation,
)
from rag1.extractions.layouts.writer import document_output_path
from rag1.extractions.ocr_text.contracts import (
    OcrCrop,
    OcrDocument,
    OcrError,
    OcrLine,
    OcrPageMetadata,
    OcrRegion,
    TableMetadata,
)
from rag1.extractions.ocr_text.image_paths import resolve_page_images
from rag1.extractions.ocr_text.layout_input import (
    load_layout_document,
    load_upstream_manifest,
)
from rag1.extractions.ocr_text.paddle import PaddleOcrV6Adapter


class OcrAdapter(Protocol):
    def initialize(self) -> None: ...
    def detect(self, image_path: str) -> tuple[list[list[list[float]]], list[float]]: ...
    def recognize(self, image_path: str) -> tuple[str, float]: ...


def run_ocr(
    layout_path: str | Path,
    *,
    image_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path = Path("data/ocr"),
    device: str = "cpu",
    adapter: OcrAdapter | None = None,
) -> Path:
    """Enrich layout proposals and write ``ocr.json`` plus referenced crop images."""
    layout_path = Path(layout_path)
    if layout_path.is_dir():
        artifact_directory = layout_path
        layout_path = artifact_directory / "layout.json"
        if manifest_path is None and (artifact_directory / "manifest.json").is_file():
            manifest_path = artifact_directory / "manifest.json"
    layout = load_layout_document(layout_path)
    for region in layout.regions:
        if not isinstance(region.location, VisualLocation):
            raise ValueError("OCR input requires visual page proposals")
    if image_dir is not None:
        image_directory = Path(image_dir)
        if not image_directory.is_absolute():
            image_directory = layout_path.parent / image_directory
        try:
            page_images = resolve_page_images(image_directory)
            image_directory_error = None
        except FileNotFoundError as error:
            page_images = {}
            image_directory_error = error
    else:
        page_images = {}
        image_directory_error = None
        image_directory = layout_path.parent / "pages"
        for candidate in (
            layout_path.parent,
            layout_path.parent / "pages",
            layout_path.parent / "region",
        ):
            try:
                candidate_images = resolve_page_images(candidate)
            except FileNotFoundError:
                continue
            if candidate_images:
                image_directory = candidate
                page_images = candidate_images
                break
    if manifest_path is None:
        dpi = None
        page_numbers = sorted({
            region.location.page_number
            for region in layout.regions
            if isinstance(region.location, VisualLocation)
        })
        page_metadata = [OcrPageMetadata(page_number=number) for number in page_numbers]
    else:
        dpi, page_metadata = load_upstream_manifest(manifest_path, layout)
    page_proposals: dict[int, list[Region]] = {}
    for region in layout.regions:
        location = region.location
        assert isinstance(location, VisualLocation)
        page_proposals.setdefault(location.page_number, []).append(region)
    page_image_errors: dict[int, Exception] = {}
    for page_number, proposals in page_proposals.items():
        page_path = page_images.get(page_number)
        try:
            if image_directory_error is not None:
                raise image_directory_error
            if page_path is None:
                raise FileNotFoundError(f"no page-{page_number:04d}.png image was found")
            expected_sizes = {
                (proposal.location.page_width, proposal.location.page_height)
                for proposal in proposals
                if isinstance(proposal.location, VisualLocation)
            }
            if len(expected_sizes) != 1:
                raise ValueError("layout proposals disagree on page dimensions")
            expected_size = next(iter(expected_sizes))
            with Image.open(page_path) as image:
                image.load()
                actual_size = image.size
            if actual_size != expected_size:
                raise ValueError(
                    f"page image dimensions {actual_size} do not match layout "
                    f"dimensions {expected_size}; coordinates were not rescaled"
                )
        except Exception as error:
            page_image_errors[page_number] = error

    result_dir = document_output_path(output_dir, layout.source)
    crop_dir = result_dir / "crops"
    region_crop_dir = crop_dir / "regions"
    line_crop_dir = crop_dir / "lines"
    region_crop_dir.mkdir(parents=True, exist_ok=True)
    line_crop_dir.mkdir(parents=True, exist_ok=True)

    ocr_adapter = adapter or PaddleOcrV6Adapter(device=device)
    initialization_error: Exception | None = None
    if layout.regions and len(page_image_errors) < len(page_proposals):
        try:
            ocr_adapter.initialize()
        except Exception as error:
            initialization_error = error

    output_regions: list[OcrRegion] = []
    errors: list[OcrError] = []
    page_failures = [
        PageFailure.from_exception(page_number, error)
        for page_number, error in sorted(page_image_errors.items())
    ]
    for index, proposal in enumerate(layout.regions, start=1):
        location = proposal.location
        assert isinstance(location, VisualLocation)
        left = max(0, math.floor(location.bbox[0]))
        top = max(0, math.floor(location.bbox[1]))
        right = math.ceil(location.bbox[2])
        bottom = math.ceil(location.bbox[3])
        page_bbox = [left, top, right, bottom]
        crop_width, crop_height = right - left, bottom - top
        crop_path = region_crop_dir / f"region-{index:05d}.png"
        crop_ref = crop_path.relative_to(result_dir).as_posix()
        table = TableMetadata() if proposal.kind is RegionKind.TABLE else None
        page_image_path = page_images.get(location.page_number)
        if location.page_number in page_image_errors or page_image_path is None:
            output_regions.append(
                _failed_region(
                    proposal,
                    location.bbox,
                    page_bbox,
                    crop_width,
                    crop_height,
                    None,
                    table,
                )
            )
            continue

        try:
            with Image.open(page_image_path) as opened:
                page_image = opened.convert("RGB")
            region_image = page_image.crop((left, top, right, bottom))
            if region_image.width <= 0 or region_image.height <= 0:
                raise ValueError("proposal produces an empty page crop")
            region_image.save(crop_path, format="PNG")
            crop_model = OcrCrop(
                proposal_bbox=location.bbox,
                page_bbox=page_bbox,
                page_origin=[left, top],
                width=crop_width,
                height=crop_height,
                path=crop_ref,
            )
            if initialization_error is not None:
                raise initialization_error
            polygons, detection_scores = ocr_adapter.detect(str(crop_path))
            lines: list[OcrLine] = []
            for detector_index, (polygon, detection_score) in enumerate(
                zip(polygons, detection_scores, strict=True)
            ):
                line_bounds = _polygon_bounds(polygon, region_image.width, region_image.height)
                line_image = region_image.crop(line_bounds)
                if line_image.width <= 0 or line_image.height <= 0:
                    raise ValueError("detected text region produces an empty crop")
                line_path = line_crop_dir / f"region-{index:05d}-line-{detector_index:05d}.png"
                line_image.save(line_path, format="PNG")
                text, recognition_score = ocr_adapter.recognize(str(line_path))
                page_polygon = [[x + left, y + top] for x, y in polygon]
                lines.append(
                    OcrLine(
                        id=f"{proposal.id}-line-{detector_index + 1:04d}",
                        region_id=proposal.id,
                        detector_index=detector_index,
                        crop_path=line_path.relative_to(result_dir).as_posix(),
                        text=text,
                        detection_confidence=detection_score,
                        recognition_confidence=recognition_score,
                        region_quad=polygon,
                        page_quad=page_polygon,
                    )
                )
            paragraph = None if table is not None else "\n".join(line.text for line in lines)
            output_regions.append(
                OcrRegion(
                    proposal=proposal,
                    status="complete",
                    crop=crop_model,
                    lines=lines,
                    text=paragraph,
                    table=table,
                )
            )
        except Exception as error:
            errors.append(_error("detection_or_recognition", location.page_number, proposal, error))
            output_regions.append(
                _failed_region(
                    proposal,
                    location.bbox,
                    page_bbox,
                    crop_width,
                    crop_height,
                    crop_ref if crop_path.is_file() else None,
                    table,
                )
            )

    complete_regions = sum(region.status == "complete" for region in output_regions)
    if errors or page_failures or layout.errors:
        status = DocumentStatus.PARTIAL if complete_regions else DocumentStatus.FAILED
    else:
        status = DocumentStatus.COMPLETE
    document = OcrDocument(
        source=layout.source,
        source_format=layout.source_format,
        upstream_schema_version=layout.schema_version,
        upstream_status=layout.status,
        upstream_errors=layout.errors,
        dpi=dpi,
        pages=page_metadata,
        status=status,
        regions=output_regions,
        errors=errors,
        page_failures=page_failures,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "ocr.json"
    output_path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path


def _polygon_bounds(
    polygon: list[list[float]], width: int, height: int
) -> tuple[int, int, int, int]:
    if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
        raise ValueError("detected text polygon must contain four x/y vertices")
    if any(not math.isfinite(value) for point in polygon for value in point):
        raise ValueError("detected text polygon coordinates must be finite")
    left = max(0, math.floor(min(point[0] for point in polygon)))
    top = max(0, math.floor(min(point[1] for point in polygon)))
    right = min(width, math.ceil(max(point[0] for point in polygon)))
    bottom = min(height, math.ceil(max(point[1] for point in polygon)))
    if left >= right or top >= bottom:
        raise ValueError("detected text polygon does not overlap its proposal crop")
    return left, top, right, bottom


def _failed_region(
    proposal: Region,
    proposal_bbox: list[float],
    page_bbox: list[int],
    width: int,
    height: int,
    path: str | None,
    table: TableMetadata | None,
) -> OcrRegion:
    return OcrRegion(
        proposal=proposal,
        status="failed",
        crop=OcrCrop(
            proposal_bbox=proposal_bbox,
            page_bbox=page_bbox,
            page_origin=page_bbox[:2],
            width=width,
            height=height,
            path=path,
        ),
        table=table,
    )


def _error(stage: str, page_number: int, proposal: Region, error: Exception) -> OcrError:
    return OcrError(
        stage=stage,
        page_number=page_number,
        region_id=proposal.id,
        exception_type=type(error).__name__,
        message=str(error),
    )

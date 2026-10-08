"""Run text detection and recognition over versioned layout proposals."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Literal, cast

from PIL import Image, ImageDraw

from rag1.extractions.layouts.contracts import (
    DocumentFormat,
    DocumentStatus,
    LayoutDocument,
    PageFailure,
    Region,
    RegionKind,
    VisualLocation,
)
from rag1.extractions.layouts.writer import document_output_path
from rag1.extractions.layouts.pdf_renderer import DEFAULT_PAGE_IMAGE_DIR
from rag1.extractions.ocr_text.contracts import (
    OcrBlock,
    OcrCrop,
    OcrDocument,
    OcrError,
    OcrLine,
    OcrPageMetadata,
    OcrRegion,
    TableMetadata,
)
from rag1.extractions.ocr_text.image_paths import resolve_direct_images, resolve_page_images
from rag1.extractions.ocr_text.layout_input import (
    load_layout_document,
    load_upstream_manifest,
)
from rag1.extractions.ocr_text.adapters import OcrAdapter, RecognizedBlock, RegionOcrAdapter
from rag1.extractions.ocr_text.paddle import (
    PaddleOcrV6Adapter,
    TEXT_RECOGNITION_MODEL_DIR,
)
from rag1.extractions.ocr_text.paddle_vl import PaddleOcrVlAdapter
from rag1.extractions.ocr_text.text_output import render_ocr_text, write_ocr_text
from rag1.extractions.ocr_text.vietocr import VietOcrAdapter
from rag1.extractions.config import load_extraction_config


_CONFIG = load_extraction_config("ocr_text")
DEFAULT_OCR_OUTPUT_DIR = Path(_CONFIG["output_dir"])
DEFAULT_OCR_DEVICE = _CONFIG["device"]
DEFAULT_OCR_MODEL = _CONFIG["model"]
logger = logging.getLogger(__name__)


@contextmanager
def _log_phase(name: str) -> Iterator[None]:
    logger.info("Phase started: %s", name)
    started_at = perf_counter()
    try:
        yield
    except Exception:
        logger.error(
            "Phase failed: %s (%.2f seconds)",
            name,
            perf_counter() - started_at,
        )
        raise
    else:
        logger.info(
            "Phase completed: %s (%.2f seconds)",
            name,
            perf_counter() - started_at,
        )


def _assign_page_blocks(
    blocks: list[RecognizedBlock], proposals: list[Region], page_size: tuple[int, int]
) -> tuple[dict[str, list[RecognizedBlock]], list[RecognizedBlock]]:
    """Assign each page block once, keeping unmatched blocks for a page region."""
    assigned: dict[str, list[RecognizedBlock]] = {}
    unmatched: list[RecognizedBlock] = []
    tables = [region for region in proposals if region.kind is RegionKind.TABLE]
    non_tables = [region for region in proposals if region.kind is not RegionKind.TABLE]
    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        if x0 < 0 or y0 < 0 or x1 > page_size[0] or y1 > page_size[1]:
            raise ValueError("PaddleOCR-VL page block falls outside the page image")
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        if any(
            region.location.bbox[0] <= center_x <= region.location.bbox[2]
            and region.location.bbox[1] <= center_y <= region.location.bbox[3]
            for region in tables
        ):
            continue
        candidates: list[tuple[int, int, Region]] = []
        for index, region in enumerate(non_tables):
            location = region.location
            assert isinstance(location, VisualLocation)
            left, top = math.floor(location.bbox[0]), math.floor(location.bbox[1])
            right, bottom = math.ceil(location.bbox[2]), math.ceil(location.bbox[3])
            if left <= x0 and top <= y0 and x1 <= right and y1 <= bottom:
                candidates.append(((right - left) * (bottom - top), index, region))
        if candidates:
            region = min(candidates, key=lambda item: (item[0], item[1]))[2]
            assigned.setdefault(region.id, []).append(block)
        else:
            unmatched.append(block)
    return assigned, unmatched


def run_ocr(
    layout_path: str | Path,
    *,
    direct: bool = False,
    image_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OCR_OUTPUT_DIR,
    device: str = DEFAULT_OCR_DEVICE,
    model: Literal[
        "paddleocr-v6", "pp-ocrv6-medium-rec-vietnamese", "paddleocr-vl"
    ] = DEFAULT_OCR_MODEL,
    adapter: OcrAdapter | RegionOcrAdapter | None = None,
) -> Path:
    """Run OCR on layout proposals or directly on full page images."""
    input_started_at = perf_counter()
    logger.info("Phase started: load OCR input")
    if model not in {
        "paddleocr-v6",
        "pp-ocrv6-medium-rec-vietnamese",
        "paddleocr-vl",
    }:
        raise ValueError(
            "model must be paddleocr-v6, pp-ocrv6-medium-rec-vietnamese, or paddleocr-vl"
        )
    if adapter is None and model == "pp-ocrv6-medium-rec-vietnamese":
        if not TEXT_RECOGNITION_MODEL_DIR:
            raise ValueError("recognition_model_dir must be configured for Vietnamese OCR")
        if not Path(TEXT_RECOGNITION_MODEL_DIR).is_dir():
            raise FileNotFoundError(
                "Vietnamese recognition model directory does not exist: "
                f"{TEXT_RECOGNITION_MODEL_DIR}"
            )
    layout_path = Path(layout_path)
    if direct:
        if image_dir is not None or manifest_path is not None:
            raise ValueError("--image-dir and --manifest cannot be used with direct OCR")
        layout, page_images, page_metadata, direct_page_failures = _load_direct_pages(layout_path)
        dpi = None
        image_directory_error = None
    else:
        direct_page_failures = []
        if layout_path.is_dir():
            artifact_directory = layout_path
            layout_path = artifact_directory / "layout.json"
            if manifest_path is None and (artifact_directory / "manifest.json").is_file():
                manifest_path = artifact_directory / "manifest.json"
        elif manifest_path is None and (layout_path.parent / "manifest.json").is_file():
            manifest_path = layout_path.parent / "manifest.json"
        layout = load_layout_document(layout_path)
        for region in layout.regions:
            if not isinstance(region.location, VisualLocation):
                raise ValueError(
                    "OCR schema version 1.0 requires visual page geometry; "
                    f"region {region.id!r} has an unsupported structural location"
                )
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
            image_directory = layout_path.parent / DEFAULT_PAGE_IMAGE_DIR
            for candidate in (
                layout_path.parent,
                layout_path.parent / DEFAULT_PAGE_IMAGE_DIR,
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
    logger.info(
        "Phase completed: load OCR input (%.2f seconds)",
        perf_counter() - input_started_at,
    )
    processable_kinds = (
        set(RegionKind) if model == "paddleocr-vl" else {RegionKind.TEXT, RegionKind.TABLE}
    )
    processable_regions = [
        region for region in layout.regions if region.kind in processable_kinds
    ]
    page_proposals: dict[int, list[Region]] = {}
    for region in layout.regions:
        if region.kind not in processable_kinds:
            continue
        location = region.location
        assert isinstance(location, VisualLocation)
        page_proposals.setdefault(location.page_number, []).append(region)
    logger.info(
        "%s OCR with %s for %s (%d page image(s), %d processable region(s))",
        "direct" if direct else "layout-based",
        model,
        layout.source,
        len(page_images),
        len(processable_regions),
    )
    validation_started_at = perf_counter()
    logger.info("Phase started: validate page images")
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
            logger.warning("Page %d image is unavailable: %s", page_number, error)
    logger.info(
        "Phase completed: validate page images (%.2f seconds)",
        perf_counter() - validation_started_at,
    )

    result_dir = document_output_path(output_dir, layout.source)
    formatted_path = result_dir / "ocr.formatted.md"
    formatted_path.unlink(missing_ok=True)
    crop_dir = result_dir / "crops"
    region_crop_dir = crop_dir / "regions"
    line_crop_dir = crop_dir / "lines"
    region_crop_dir.mkdir(parents=True, exist_ok=True)
    line_crop_dir.mkdir(parents=True, exist_ok=True)

    if adapter is not None:
        ocr_adapter = adapter
    elif model == "paddleocr-vl":
        ocr_adapter = PaddleOcrVlAdapter(device=device)
    elif model == "paddleocr-v6":
        ocr_adapter = PaddleOcrV6Adapter(
            device=device,
            recognizer=VietOcrAdapter(device=device),
        )
    else:
        ocr_adapter = PaddleOcrV6Adapter(device=device)
    initialization_error: Exception | None = None
    if processable_regions and len(page_image_errors) < len(page_proposals):
        try:
            with _log_phase(f"initialize {model} OCR model on {device}"):
                ocr_adapter.initialize()
        except Exception as error:
            initialization_error = error
            logger.warning("OCR model initialization failed: %s", error)

    output_regions: list[OcrRegion] = []
    errors: list[OcrError] = []
    formatted_pages: dict[int, str] = {}
    page_blocks: dict[str, list[RecognizedBlock]] = {}
    unmatched_blocks: dict[int, list[RecognizedBlock]] = {}
    page_inputs: dict[int, Path] = {}
    vl_page_failures: dict[int, Exception] = {}
    if model == "paddleocr-vl" and not direct:
        region_adapter = cast(RegionOcrAdapter, ocr_adapter)
        for page_number, proposals in sorted(page_proposals.items()):
            non_tables = [region for region in proposals if region.kind is not RegionKind.TABLE]
            if not non_tables or page_number in page_image_errors:
                continue
            page_path = page_images[page_number]
            try:
                if initialization_error is not None:
                    raise initialization_error
                with Image.open(page_path) as opened:
                    page_image = opened.convert("RGB")
                table_proposals = [
                    region for region in proposals if region.kind is RegionKind.TABLE
                ]
                input_path = page_path
                if table_proposals:
                    masked = page_image.copy()
                    draw = ImageDraw.Draw(masked)
                    for region in table_proposals:
                        location = region.location
                        assert isinstance(location, VisualLocation)
                        x0, y0, x1, y1 = location.bbox
                        draw.rectangle(
                            (math.floor(x0), math.floor(y0),
                             math.ceil(x1) - 1, math.ceil(y1) - 1),
                            fill="white",
                        )
                    masked_dir = result_dir / "pages"
                    masked_dir.mkdir(parents=True, exist_ok=True)
                    input_path = masked_dir / f"page-{page_number:04d}-non-table.png"
                    masked.save(input_path, format="PNG")
                page_inputs[page_number] = input_path
                with _log_phase(f"PaddleOCR-VL page {page_number} non-table inference"):
                    parsed = region_adapter.parse_region(str(input_path))
                assigned, unmatched = _assign_page_blocks(
                    parsed, proposals, page_image.size
                )
                page_blocks.update(assigned)
                unmatched_blocks[page_number] = unmatched
            except Exception as error:
                vl_page_failures[page_number] = error
                errors.append(OcrError(
                    stage="page_detection_or_recognition",
                    page_number=page_number,
                    region_id=None,
                    exception_type=type(error).__name__,
                    message=str(error),
                ))
                logger.warning("Page %d non-table OCR failed: %s", page_number, error)
    page_failures = direct_page_failures + [
        PageFailure.from_exception(page_number, error)
        for page_number, error in sorted(page_image_errors.items())
    ]
    regions_started_at = perf_counter()
    logger.info("Phase started: process OCR regions")
    for index, proposal in enumerate(layout.regions, start=1):
        region_started_at = perf_counter()
        location = proposal.location
        assert isinstance(location, VisualLocation)
        logger.info(
            "OCR region %d/%d: %s on page %d (%s)",
            index,
            len(layout.regions),
            proposal.id,
            location.page_number,
            proposal.kind.value,
        )
        left = max(0, math.floor(location.bbox[0]))
        top = max(0, math.floor(location.bbox[1]))
        right = math.ceil(location.bbox[2])
        bottom = math.ceil(location.bbox[3])
        page_bbox = [left, top, right, bottom]
        crop_width, crop_height = right - left, bottom - top
        if proposal.kind not in processable_kinds:
            output_regions.append(
                _skipped_region(
                    proposal,
                    location.bbox,
                    page_bbox,
                    crop_width,
                    crop_height,
                )
            )
            logger.info(
                "Region %s skipped (%.2f seconds)",
                proposal.id,
                perf_counter() - region_started_at,
            )
            continue
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
            logger.warning(
                "Region %s failed (%.2f seconds)",
                proposal.id,
                perf_counter() - region_started_at,
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
            if (
                model == "paddleocr-vl" and not direct
                and proposal.kind is not RegionKind.TABLE
                and location.page_number in vl_page_failures
            ):
                output_regions.append(_failed_region(
                    proposal, location.bbox, page_bbox,
                    crop_width, crop_height, crop_ref, table,
                ))
                continue
            if initialization_error is not None:
                raise initialization_error
            model_input_path = str(page_image_path if direct else crop_path)
            if model == "paddleocr-vl":
                region_adapter = cast(RegionOcrAdapter, ocr_adapter)
                blocks: list[OcrBlock] = []
                page_mode = not direct and proposal.kind is not RegionKind.TABLE
                recognized = (
                    page_blocks.get(proposal.id, [])
                    if page_mode else region_adapter.parse_region(model_input_path)
                )
                if direct and isinstance(ocr_adapter, PaddleOcrVlAdapter):
                    try:
                        formatted_pages[location.page_number] = _save_native_markdown_page(
                            ocr_adapter.last_markdown(), result_dir, location.page_number
                        )
                    except Exception as error:
                        errors.append(_formatted_markdown_error(location.page_number, error))
                        logger.warning("Page %d formatted Markdown failed: %s", location.page_number, error)
                for block in recognized:
                    x0, y0, x1, y1 = block.bbox
                    blocks.append(
                        OcrBlock(
                            label=block.label,
                            content=block.content,
                            region_bbox=(
                                [x0 - left, y0 - top, x1 - left, y1 - top]
                                if page_mode else block.bbox
                            ),
                            page_bbox=(
                                block.bbox if page_mode
                                else [x0 + left, y0 + top, x1 + left, y1 + top]
                            ),
                        )
                    )
                paragraph = (
                    None if table is not None
                    else "\n".join(block.content for block in blocks)
                )
                output_regions.append(
                    OcrRegion(
                        proposal=proposal,
                        status="complete" if blocks else "empty",
                        crop=crop_model,
                        blocks=blocks,
                        text=paragraph,
                        table=table,
                    )
                )
                logger.info(
                    "Region %s %s (%.2f seconds)",
                    proposal.id,
                    "completed" if blocks else "empty",
                    perf_counter() - region_started_at,
                )
                continue
            line_adapter = cast(OcrAdapter, ocr_adapter)
            detections = line_adapter.detect(model_input_path)
            lines: list[OcrLine] = []
            region_error_count = len(errors)
            for detector_index, detection in enumerate(detections):
                polygon = detection.polygon
                line_path = line_crop_dir / f"region-{index:05d}-line-{detector_index:05d}.png"
                try:
                    line_image = _rectify_polygon(region_image, polygon)
                    line_image.save(line_path, format="PNG")
                except Exception as error:
                    errors.append(
                        _error(
                            f"line_{detector_index + 1}_rectification",
                            location.page_number,
                            proposal,
                            error,
                        )
                    )
                    continue
                try:
                    text, recognition_score = line_adapter.recognize(str(line_path))
                except Exception as error:
                    text = None
                    recognition_score = None
                    errors.append(
                        _error(
                            f"line_{detector_index + 1}_recognition",
                            location.page_number,
                            proposal,
                            error,
                        )
                    )
                page_polygon = [[x + left, y + top] for x, y in polygon]
                lines.append(
                    OcrLine(
                        id=f"{proposal.id}-line-{detector_index + 1:04d}",
                        region_id=proposal.id,
                        detector_index=detector_index,
                        crop_path=line_path.relative_to(result_dir).as_posix(),
                        text=text,
                        detection_confidence=detection.confidence,
                        recognition_confidence=recognition_score,
                        region_quad=polygon,
                        page_quad=page_polygon,
                    )
                )
            lines.sort(
                key=lambda line: (
                    min(point[1] for point in line.region_quad),
                    min(point[0] for point in line.region_quad),
                    line.detector_index,
                )
            )
            paragraph = (
                None
                if table is not None
                else "\n".join(line.text for line in lines if line.text is not None)
            )
            successful_lines = sum(line.text is not None for line in lines)
            line_errors = len(errors) - region_error_count
            if line_errors:
                region_status = "partial" if successful_lines else "failed"
            elif not detections:
                region_status = "empty"
            else:
                region_status = "complete"
            output_regions.append(
                OcrRegion(
                    proposal=proposal,
                    status=region_status,
                    crop=crop_model,
                    lines=lines,
                    text=paragraph,
                    table=table,
                )
            )
            logger.info(
                "Region %s %s (%.2f seconds)",
                proposal.id,
                region_status,
                perf_counter() - region_started_at,
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
            logger.warning(
                "Region %s failed (%.2f seconds)",
                proposal.id,
                perf_counter() - region_started_at,
            )

    used_ids = {region.proposal.id for region in output_regions}
    for page_number, blocks in sorted(unmatched_blocks.items()):
        if not blocks:
            continue
        location = page_proposals[page_number][0].location
        assert isinstance(location, VisualLocation)
        width, height = int(location.page_width), int(location.page_height)
        fallback_id = f"p{page_number:04d}-ocr-unassigned"
        suffix = 2
        while fallback_id in used_ids:
            fallback_id = f"p{page_number:04d}-ocr-unassigned-{suffix}"
            suffix += 1
        used_ids.add(fallback_id)
        for page in page_metadata:
            if page.page_number == page_number and page.region_ids is not None:
                page.region_ids.append(fallback_id)
                break
        fallback_path = region_crop_dir / f"page-{page_number:04d}-unassigned.png"
        with Image.open(page_inputs[page_number]) as opened:
            opened.convert("RGB").save(fallback_path, format="PNG")
        fallback = OcrRegion(
            proposal=Region(
                id=fallback_id,
                kind=RegionKind.OTHER,
                raw_label="unassigned_ocr",
                location=VisualLocation(
                    page_number=page_number,
                    bbox=[0, 0, width, height],
                    page_width=width,
                    page_height=height,
                ),
            ),
            status="complete",
            crop=OcrCrop(
                proposal_bbox=[0, 0, width, height],
                page_bbox=[0, 0, width, height],
                page_origin=[0, 0],
                width=width,
                height=height,
                path=fallback_path.relative_to(result_dir).as_posix(),
            ),
            blocks=[OcrBlock(
                label=block.label,
                content=block.content,
                region_bbox=block.bbox,
                page_bbox=block.bbox,
            ) for block in blocks],
            text="\n".join(block.content for block in blocks),
        )
        insert_at = max(
            index for index, region in enumerate(output_regions)
            if isinstance(region.proposal.location, VisualLocation)
            and region.proposal.location.page_number == page_number
        ) + 1
        output_regions.insert(insert_at, fallback)

    logger.info(
        "Phase completed: process OCR regions (%.2f seconds)",
        perf_counter() - regions_started_at,
    )

    if model == "paddleocr-vl" and not direct and isinstance(ocr_adapter, PaddleOcrVlAdapter):
        for page_number, page_path in sorted(page_images.items()):
            if page_number in page_image_errors:
                continue
            try:
                if initialization_error is not None:
                    raise initialization_error
                with _log_phase(f"PaddleOCR-VL page {page_number} formatted Markdown inference"):
                    ocr_adapter.parse_region(str(page_path))
                formatted_pages[page_number] = _save_native_markdown_page(
                    ocr_adapter.last_markdown(), result_dir, page_number
                )
            except Exception as error:
                errors.append(_formatted_markdown_error(page_number, error))
                logger.warning("Page %d formatted Markdown failed: %s", page_number, error)

    has_errors = bool(errors or page_failures or layout.errors)
    has_successful_processing = any(
        region.status in {"complete", "empty", "partial"}
        for region in output_regions
    )
    if has_errors:
        status = (
            DocumentStatus.PARTIAL
            if has_successful_processing
            else DocumentStatus.FAILED
        )
    else:
        status = DocumentStatus.COMPLETE
    document = OcrDocument(
        input_mode="direct" if direct else "layout",
        model=model,
        source=layout.source,
        source_format=layout.source_format,
        upstream_schema_version=None if direct else layout.schema_version,
        upstream_status=None if direct else layout.status,
        upstream_errors=[] if direct else layout.errors,
        dpi=dpi,
        pages=page_metadata,
        status=status,
        regions=output_regions,
        errors=errors,
        page_failures=page_failures,
    )
    with _log_phase("write OCR artifacts"):
        document.text = render_ocr_text(document)
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / "ocr.json"
        if not direct:
            exclude = {"input_mode": True}
        else:
            exclude = {}
        if model in {"paddleocr-v6", "pp-ocrv6-medium-rec-vietnamese"}:
            if model == "paddleocr-v6" and not direct:
                exclude["model"] = True
            exclude["regions"] = {"__all__": {"blocks"}}
        _write_text_atomically(
            output_path,
            document.model_dump_json(indent=2, exclude=exclude or None) + "\n",
        )
        write_ocr_text(document, result_dir / "ocr.md")
        if formatted_pages:
            formatted_text = "\n\n".join(
                formatted_pages[page_number].strip()
                for page_number in sorted(formatted_pages)
                if formatted_pages[page_number].strip()
            )
            _write_text_atomically(
                formatted_path, formatted_text + "\n" if formatted_text else ""
            )
    logger.info(
        "OCR status: %s; %d region(s), %d error(s); artifacts: %s",
        status.value,
        len(output_regions),
        len(errors) + len(page_failures),
        result_dir,
    )
    return output_path


def _load_direct_pages(
    input_path: Path,
) -> tuple[LayoutDocument, dict[int, Path], list[OcrPageMetadata], list[PageFailure]]:
    """Create full-page scopes from images without running layout extraction."""
    input_images = resolve_direct_images(input_path)
    page_images: dict[int, Path] = {}
    regions: list[Region] = []
    metadata: list[OcrPageMetadata] = []
    failures: list[PageFailure] = []
    for page_number, page_path in input_images.items():
        try:
            with Image.open(page_path) as image:
                image.load()
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"page image has invalid dimensions: {page_path}")
        except Exception as error:
            failures.append(PageFailure.from_exception(page_number, error))
            metadata.append(
                OcrPageMetadata(page_number=page_number, image=str(page_path.resolve()))
            )
            continue
        page_images[page_number] = page_path
        region_id = f"p{page_number:04d}-full-page"
        regions.append(
            Region(
                id=region_id,
                kind=RegionKind.TEXT,
                raw_label="full_page",
                location=VisualLocation(
                    page_number=page_number,
                    bbox=[0, 0, width, height],
                    page_width=width,
                    page_height=height,
                ),
            )
        )
        metadata.append(
            OcrPageMetadata(
                page_number=page_number,
                pixel_width=width,
                pixel_height=height,
                image=str(page_path.resolve()),
                region_ids=[region_id],
            )
        )
    layout = LayoutDocument(
        source=str(input_path.resolve()),
        source_format=DocumentFormat.IMAGE,
        status=DocumentStatus.COMPLETE,
        regions=regions,
    )
    return layout, page_images, metadata, failures


def _rectify_polygon(image: Image.Image, polygon: list[list[float]]) -> Image.Image:
    """Perspective rectify a detector quadrilateral without changing its geometry."""
    if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
        raise ValueError("detected text polygon must contain four x/y vertices")
    if any(not math.isfinite(value) for point in polygon for value in point):
        raise ValueError("detected text polygon coordinates must be finite")
    if any(
        point[0] < 0 or point[0] > image.width or point[1] < 0 or point[1] > image.height
        for point in polygon
    ):
        raise ValueError("detected text polygon must fit within its proposal crop")

    edges = [
        math.dist(polygon[index], polygon[(index + 1) % 4])
        for index in range(4)
    ]
    cross_products = [
        (polygon[(index + 1) % 4][0] - polygon[index][0])
        * (polygon[(index + 2) % 4][1] - polygon[(index + 1) % 4][1])
        - (polygon[(index + 1) % 4][1] - polygon[index][1])
        * (polygon[(index + 2) % 4][0] - polygon[(index + 1) % 4][0])
        for index in range(4)
    ]
    area = abs(sum(
        polygon[index][0] * polygon[(index + 1) % 4][1]
        - polygon[(index + 1) % 4][0] * polygon[index][1]
        for index in range(4)
    )) / 2
    if min(edges) <= 0 or area <= 0 or not (
        all(value > 0 for value in cross_products)
        or all(value < 0 for value in cross_products)
    ):
        raise ValueError("detected text polygon is degenerate or not convex")

    width = max(1, round(max(edges[0], edges[2])))
    height = max(1, round(max(edges[1], edges[3])))
    if width <= 0 or height <= 0:
        raise ValueError("detected text polygon does not overlap its proposal crop")
    # Pillow QUAD expects upper-left, lower-left, lower-right, upper-right.
    quad = [polygon[index] for index in (0, 3, 2, 1)]
    data = tuple(coordinate for point in quad for coordinate in point)
    return image.transform(
        (width, height),
        Image.Transform.QUAD,
        data,
        resample=Image.Resampling.BICUBIC,
    )


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


def _skipped_region(
    proposal: Region,
    proposal_bbox: list[float],
    page_bbox: list[int],
    width: int,
    height: int,
) -> OcrRegion:
    return OcrRegion(
        proposal=proposal,
        status="skipped",
        crop=OcrCrop(
            proposal_bbox=proposal_bbox,
            page_bbox=page_bbox,
            page_origin=page_bbox[:2],
            width=width,
            height=height,
            path=None,
        ),
    )


def _error(stage: str, page_number: int, proposal: Region, error: Exception) -> OcrError:
    return OcrError(
        stage=stage,
        page_number=page_number,
        region_id=proposal.id,
        exception_type=type(error).__name__,
        message=str(error),
    )


def _formatted_markdown_error(page_number: int, error: Exception) -> OcrError:
    return OcrError(
        stage="formatted_markdown",
        page_number=page_number,
        region_id=None,
        exception_type=type(error).__name__,
        message=str(error),
    )


def _save_native_markdown_page(
    markdown: Mapping[str, Any], result_dir: Path, page_number: int
) -> str:
    """Save a VL page's image assets and point its native Markdown at them."""
    text = markdown["markdown_texts"]
    images = markdown["markdown_images"]
    assert isinstance(text, str)
    assert isinstance(images, Mapping)
    replacements: dict[str, str] = {}
    for index, (source, image) in enumerate(images.items(), start=1):
        if not isinstance(source, str) or not source:
            raise ValueError("PaddleOCR-VL Markdown image path must be a non-empty string")
        if not isinstance(image, Image.Image):
            raise ValueError("PaddleOCR-VL Markdown image must be a Pillow image")
        relative_path = Path("formatted-images") / f"page-{page_number:04d}" / f"image-{index:04d}.png"
        image_path = result_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, format="PNG")
        replacements[source] = relative_path.as_posix()
    for source in sorted(replacements, key=len, reverse=True):
        text = text.replace(source, replacements[source])
    return text


def _write_text_atomically(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

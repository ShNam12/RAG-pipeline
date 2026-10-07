"""Run text detection and recognition over versioned layout proposals."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Literal, cast

from PIL import Image

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
from rag1.extractions.ocr_text.adapters import OcrAdapter, RegionOcrAdapter
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
    processable_kinds = {RegionKind.TEXT, RegionKind.TABLE}
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
            with _log_phase(f"initialize OCR model on {device}"):
                ocr_adapter.initialize()
        except Exception as error:
            initialization_error = error
            logger.warning("OCR model initialization failed: %s", error)

    output_regions: list[OcrRegion] = []
    errors: list[OcrError] = []
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
            if initialization_error is not None:
                raise initialization_error
            model_input_path = str(page_image_path if direct else crop_path)
            if model == "paddleocr-vl":
                region_adapter = cast(RegionOcrAdapter, ocr_adapter)
                blocks: list[OcrBlock] = []
                for block in region_adapter.parse_region(model_input_path):
                    x0, y0, x1, y1 = block.bbox
                    blocks.append(
                        OcrBlock(
                            label=block.label,
                            content=block.content,
                            region_bbox=block.bbox,
                            page_bbox=[x0 + left, y0 + top, x1 + left, y1 + top],
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

    logger.info(
        "Phase completed: process OCR regions (%.2f seconds)",
        perf_counter() - regions_started_at,
    )

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
        _write_json_atomically(
            output_path,
            document.model_dump_json(indent=2, exclude=exclude or None) + "\n",
        )
        write_ocr_text(document, result_dir / "ocr.md")
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


def _write_json_atomically(path: Path, content: str) -> None:
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

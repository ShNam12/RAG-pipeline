"""Run the PDF layout extraction phase and persist its artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
from time import perf_counter

from rag1.extractions.layouts.contracts import DocumentFormat
from rag1.extractions.layouts.extractors import (
    ExtractorNotRegisteredError,
    detect_document_format,
)
from rag1.extractions.layouts.paddle import ModelFactory, PaddleLayoutAdapter
from rag1.extractions.layouts.pdf_renderer import (
    DEFAULT_LAYOUT_OUTPUT_DIR,
    DEFAULT_PAGE_IMAGE_DIR,
    DEFAULT_RENDER_DPI,
    PDFPageRenderer,
)
from rag1.extractions.layouts.writer import LayoutArtifactWriter, LayoutArtifacts
from rag1.extractions.layouts.paddle import DEFAULT_LAYOUT_DEVICE


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


def run_pdf_layout(
    source: str | Path,
    *,
    output_dir: str | Path = DEFAULT_LAYOUT_OUTPUT_DIR,
    image_dir: str | Path = DEFAULT_PAGE_IMAGE_DIR,
    dpi: int = DEFAULT_RENDER_DPI,
    device: str = DEFAULT_LAYOUT_DEVICE,
    model_factory: ModelFactory | None = None,
) -> LayoutArtifacts:
    """Render a PDF, detect page regions, and write one document artifact set."""
    source_path = Path(source)
    document_format = detect_document_format(source_path)
    if document_format is not DocumentFormat.PDF:
        raise ExtractorNotRegisteredError(
            "The .docx format is recognized, but this layout command currently "
            "processes PDF input only."
        )

    if not source_path.is_file():
        raise FileNotFoundError(f"Layout input does not exist: {source_path}")

    renderer = PDFPageRenderer(
        dpi=dpi,
        output_dir=output_dir,
        image_dir=image_dir,
    )
    adapter = PaddleLayoutAdapter(model_factory=model_factory, device=device)
    writer = LayoutArtifactWriter(output_dir)
    with _log_phase("prepare layout output directory"):
        document_dir = writer.create_document_directory(source_path)
    with _log_phase(f"render PDF pages at {dpi} DPI"):
        pages = renderer.render(source_path, document_output_dir=document_dir)
    with _log_phase(f"detect layout regions across {len(pages)} page(s)"):
        document = adapter.extract_document(
            source=str(source_path.resolve()),
            source_format=document_format,
            pages=pages,
        )
    logger.info("Detected %d region(s)", len(document.regions))
    with _log_phase("export page overlays"):
        bbox_images = renderer.export_bbox_overlays(
            pages,
            document.regions,
            document_output_dir=document_dir,
        )
    bbox_images_by_page = {
        page.page_number: bbox_image
        for page, bbox_image in zip(pages, bbox_images, strict=True)
    }
    with _log_phase("write layout artifacts"):
        artifacts = writer.write(
            source=source_path,
            document_dir=document_dir,
            document=document,
            dpi=dpi,
            pages=pages,
            bbox_images_by_page=bbox_images_by_page,
        )
    logger.info("Layout status: %s; artifacts: %s", artifacts.status, artifacts.output_dir)
    return artifacts

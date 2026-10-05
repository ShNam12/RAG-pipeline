"""Run the PDF layout extraction phase and persist its artifacts."""

from __future__ import annotations

from pathlib import Path

from rag1.extractions.layouts.contracts import DocumentFormat
from rag1.extractions.layouts.extractors import (
    ExtractorNotRegisteredError,
    detect_document_format,
)
from rag1.extractions.layouts.paddle import ModelFactory, PaddleLayoutAdapter
from rag1.extractions.layouts.pdf_renderer import DEFAULT_RENDER_DPI, PDFPageRenderer
from rag1.extractions.layouts.writer import LayoutArtifactWriter, LayoutArtifacts


def run_pdf_layout(
    source: str | Path,
    *,
    output_dir: str | Path = Path("data/extraction"),
    image_dir: str | Path = Path("pages"),
    dpi: int = DEFAULT_RENDER_DPI,
    device: str = "cpu",
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
    document_dir = writer.create_document_directory(source_path)
    pages = renderer.render(source_path, document_output_dir=document_dir)
    document = adapter.extract_document(
        source=str(source_path.resolve()),
        source_format=document_format,
        pages=pages,
    )
    bbox_images = renderer.export_bbox_overlays(
        pages,
        document.regions,
        document_output_dir=document_dir,
    )
    bbox_images_by_page = {
        page.page_number: bbox_image
        for page, bbox_image in zip(pages, bbox_images, strict=True)
    }
    return writer.write(
        source=source_path,
        document_dir=document_dir,
        document=document,
        dpi=dpi,
        pages=pages,
        bbox_images_by_page=bbox_images_by_page,
    )

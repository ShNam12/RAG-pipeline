"""Command-line interface for the rag1 extraction package."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    from rag1.extractions.layouts.pdf_renderer import (
        DEFAULT_LAYOUT_OUTPUT_DIR,
        DEFAULT_PAGE_IMAGE_DIR,
        DEFAULT_RENDER_DPI,
    )
    from rag1.extractions.layouts.paddle import DEFAULT_LAYOUT_DEVICE
    from rag1.extractions.ocr_text.pipeline import (
        DEFAULT_OCR_DEVICE,
        DEFAULT_OCR_MODEL,
        DEFAULT_OCR_OUTPUT_DIR,
    )

    parser = argparse.ArgumentParser(prog="rag1")
    commands = parser.add_subparsers(dest="command")
    layout = commands.add_parser(
        "layout",
        help="detect PDF layout regions and write page artifacts",
        description="Render a PDF and detect text, picture, and table regions.",
    )
    layout.add_argument("input", type=Path, help="PDF document to process")
    layout.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_LAYOUT_OUTPUT_DIR,
        help=f"root directory for document-specific artifacts (default: {DEFAULT_LAYOUT_OUTPUT_DIR})",
    )
    layout.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_PAGE_IMAGE_DIR,
        help=f"relative subdirectory for unannotated page images (default: {DEFAULT_PAGE_IMAGE_DIR})",
    )
    layout.add_argument(
        "--dpi",
        type=_positive_int,
        default=DEFAULT_RENDER_DPI,
        help=f"PDF page render resolution (default: {DEFAULT_RENDER_DPI})",
    )
    layout.add_argument(
        "--device",
        default=DEFAULT_LAYOUT_DEVICE,
        help=f"Paddle device such as cpu or gpu:0 (default: {DEFAULT_LAYOUT_DEVICE})",
    )
    ocr = commands.add_parser(
        "ocr",
        help="recognize text from layout proposals or directly from page images",
        description="Run line OCR or PaddleOCR-VL on layout proposals or page images.",
    )
    ocr.add_argument(
        "layout_json",
        type=Path,
        help="layout JSON or region directory; with --direct, an image or page image directory",
    )
    ocr.add_argument(
        "--direct",
        action="store_true",
        help="OCR full page images without reading layout proposals",
    )
    ocr.add_argument(
        "--manifest",
        type=Path,
        help="optional upstream manifest.json with source and page metadata",
    )
    ocr.add_argument(
        "--image-dir",
        type=Path,
        help="directory containing page-NNNN.png images (default: infer from layout artifacts)",
    )
    ocr.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OCR_OUTPUT_DIR,
        help=f"root directory for OCR JSON and crop artifacts (default: {DEFAULT_OCR_OUTPUT_DIR})",
    )
    ocr.add_argument(
        "--device",
        default=DEFAULT_OCR_DEVICE,
        help=f"Paddle device such as cpu or gpu:0 (default: {DEFAULT_OCR_DEVICE})",
    )
    ocr.add_argument(
        "--model",
        choices=("paddleocr-v6", "pp-ocrv6-medium-rec-vietnamese", "paddleocr-vl"),
        default=DEFAULT_OCR_MODEL,
        help=f"OCR model path (default: {DEFAULT_OCR_MODEL})",
    )
    ocr_text = commands.add_parser(
        "ocr-text",
        help="export Markdown from an existing OCR JSON file",
    )
    ocr_text.add_argument("ocr_json", type=Path, help="OCR JSON file to export")
    ocr_text.add_argument(
        "--output",
        type=Path,
        help="output Markdown path (default: ocr.md beside the JSON file)",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("rag1").setLevel(logging.INFO)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command not in {"layout", "ocr", "ocr-text"}:
        parser.print_help()
        return
    _configure_logging()

    if arguments.command == "ocr-text":
        from rag1.extractions.ocr_text.contracts import OcrDocument
        from rag1.extractions.ocr_text.text_output import write_ocr_text

        try:
            document = OcrDocument.model_validate_json(
                arguments.ocr_json.read_text(encoding="utf-8")
            )
            output_path = arguments.output or arguments.ocr_json.with_name("ocr.md")
            write_ocr_text(document, output_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"OCR Markdown: {output_path}")
        return

    if arguments.command == "ocr":
        from rag1.extractions.ocr_text.contracts import OcrDocument
        from rag1.extractions.ocr_text.pipeline import run_ocr

        try:
            output_path = run_ocr(
                arguments.layout_json,
                direct=arguments.direct,
                image_dir=arguments.image_dir,
                manifest_path=arguments.manifest,
                output_dir=arguments.output_dir,
                device=arguments.device,
                model=arguments.model,
            )
        except (LookupError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"OCR JSON: {output_path}")
        print(f"OCR Markdown: {output_path.with_name('ocr.md')}")
        output_document = OcrDocument.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        if output_document.status.value == "failed":
            raise SystemExit(1)
        return

    from rag1.extractions.layouts.pipeline import run_pdf_layout
    from rag1.extractions.layouts.pdf_renderer import BBOX_IMAGE_DIR

    try:
        artifacts = run_pdf_layout(
            arguments.input,
            output_dir=arguments.output_dir,
            image_dir=arguments.image_dir,
            dpi=arguments.dpi,
            device=arguments.device,
        )
    except (LookupError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Status: {artifacts.status}")
    print(f"Output directory: {artifacts.output_dir}")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Layout: {artifacts.layout_path}")
    page_image_dir = (
        artifacts.page_images[0].parent
        if artifacts.page_images
        else artifacts.output_dir / arguments.image_dir
    )
    bbox_image_dir = (
        artifacts.bbox_images[0].parent
        if artifacts.bbox_images
        else artifacts.output_dir / BBOX_IMAGE_DIR
    )
    print(f"Page images ({len(artifacts.page_images)}): {page_image_dir}")
    print(f"Highlighted page images ({len(artifacts.bbox_images)}): {bbox_image_dir}")
    if artifacts.status == "failed":
        raise SystemExit(1)

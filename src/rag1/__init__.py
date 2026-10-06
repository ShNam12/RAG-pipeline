"""Command-line interface for the rag1 extraction package."""

from __future__ import annotations

import argparse
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
        default=Path("data/extraction"),
        help="root directory for document-specific artifacts (default: data/extraction)",
    )
    layout.add_argument(
        "--image-dir",
        type=Path,
        default=Path("pages"),
        help="relative subdirectory for unannotated page images (default: pages)",
    )
    layout.add_argument(
        "--dpi",
        type=_positive_int,
        default=200,
        help="PDF page render resolution (default: 200)",
    )
    layout.add_argument(
        "--device",
        default="cpu",
        help="Paddle device such as cpu or gpu:0 (default: cpu)",
    )
    ocr = commands.add_parser(
        "ocr",
        help="recognize text inside a layout proposal JSON",
        description="Run PP-OCRv6 text detection and recognition for layout proposals.",
    )
    ocr.add_argument(
        "layout_json",
        type=Path,
        help="version 1.0 layout JSON or its region artifact directory",
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
        default=Path("data/ocr"),
        help="root directory for OCR JSON and crop artifacts (default: data/ocr)",
    )
    ocr.add_argument(
        "--device",
        default="cpu",
        help="Paddle device such as cpu or gpu:0 (default: cpu)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command not in {"layout", "ocr"}:
        parser.print_help()
        return

    if arguments.command == "ocr":
        from rag1.extractions.ocr_text.contracts import OcrDocument
        from rag1.extractions.ocr_text.pipeline import run_ocr

        try:
            output_path = run_ocr(
                arguments.layout_json,
                image_dir=arguments.image_dir,
                manifest_path=arguments.manifest,
                output_dir=arguments.output_dir,
                device=arguments.device,
            )
        except (LookupError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"OCR JSON: {output_path}")
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

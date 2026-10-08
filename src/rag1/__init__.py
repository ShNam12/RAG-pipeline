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
    from rag1.extractions.refinement.openai_ocr import (
        DEFAULT_OPENAI_MODEL,
        DEFAULT_OCR_ROOT,
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
    tables = commands.add_parser("tables", help="reconstruct data tables from layout and OCR")
    tables.add_argument("layout_json", type=Path, help="layout JSON with table coordinates")
    tables.add_argument("--ocr-json", required=True, type=Path, help="positioned OCR JSON")
    tables.add_argument("--manifest", type=Path, help="manifest JSON (default: beside layout JSON)")
    tables.add_argument("--output-dir", type=Path, default=Path("data/tables"))
    tables.add_argument("--model", default="PaddlePaddle/SLANet_plus_safetensors")
    tables.add_argument("--device", default="cpu")
    tables.add_argument("--region-id", help="one layout table ID or unique region-NNNNN page alias")
    tables_flat = commands.add_parser(
        "tables-flat", help="normalize tables.json or tables.html into flat row records"
    )
    tables_flat.add_argument("tables_json", type=Path, help="reconstructed tables JSON or HTML")
    tables_flat.add_argument(
        "--output", type=Path, help="output path (default: tables.flat.json or tables.from-html.json beside input)"
    )
    refine_ocr = commands.add_parser(
        "refine-ocr",
        help="rewrite OCR Markdown with GPT-5.4 mini",
    )
    refine_ocr.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help="an ocr.md file or directory to search recursively (default: data/ocr)",
    )
    refine_ocr.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model ID (default: gpt-5.4-mini)",
    )
    refine_ocr.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing ocr.refined.md files",
    )
    chunk = commands.add_parser(
        "chunk", help="build section-aware parent and child chunks from OCR JSON"
    )
    chunk.add_argument("ocr_json", type=Path, help="structured OCR JSON file")
    index = commands.add_parser(
        "index", help="embed child chunks and upsert the hierarchy into Qdrant"
    )
    index.add_argument("chunks_json", type=Path, help="chunks.json produced by rag1 chunk")
    index.add_argument(
        "--tables-json", type=Path,
        help="reconstructed tables.json whose nonempty cells should also be embedded",
    )
    index.add_argument(
        "--collection",
        help="Qdrant collection (default: rag1_hybrid_v1)",
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
    if arguments.command not in {"layout", "ocr", "ocr-text", "refine-ocr", "tables", "tables-flat", "chunk", "index"}:
        parser.print_help()
        return
    _configure_logging()

    if arguments.command == "tables-flat":
        from rag1.extractions.tabular.flat_output import write_flat_tables

        try:
            output_path = write_flat_tables(arguments.tables_json, arguments.output)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"Flat tables JSON: {output_path}")
        return

    if arguments.command == "chunk":
        from rag1.chunking import write_chunks

        try:
            output_path = write_chunks(arguments.ocr_json)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Chunks JSON: {output_path}")
        return

    if arguments.command == "index":
        from qdrant_client.http.exceptions import UnexpectedResponse

        from rag1.indexing import index_chunk_artifact

        try:
            result = index_chunk_artifact(
                arguments.chunks_json, collection_name=arguments.collection,
                tables_path=arguments.tables_json,
            )
        except (OSError, RuntimeError, UnexpectedResponse, ValueError) as error:
            parser.error(str(error))
        print(
            f"Qdrant collection: {result['collection']}; "
            f"{result['children']} child vectors; {result['nodes']} total chunks; "
            f"{result.get('table_cells', 0)} table cells"
        )
        return

    if arguments.command == "tables":
        from rag1.extractions.tabular.pipeline import run_tables

        try:
            output_path = run_tables(
                arguments.layout_json,
                ocr_json=arguments.ocr_json,
                manifest_path=arguments.manifest,
                output_dir=arguments.output_dir,
                model=arguments.model,
                device=arguments.device,
                region_id=arguments.region_id,
            )
        except (LookupError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Tables JSON: {output_path}")
        print(f"Tables HTML: {output_path.with_suffix('.html')}")
        return

    if arguments.command == "refine-ocr":
        import os

        from rag1.extractions.refinement.openai_ocr import (
            collect_ocr_markdown,
            refine_ocr_file,
        )

        try:
            sources = collect_ocr_markdown(arguments.source)
            if not arguments.overwrite:
                existing = next(
                    (
                        path.with_name("ocr.refined.md")
                        for path in sources
                        if path.with_name("ocr.refined.md").exists()
                    ),
                    None,
                )
                if existing is not None:
                    raise FileExistsError(f"Refined OCR output already exists: {existing}")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required")
            for source in sources:
                output = refine_ocr_file(
                    source,
                    api_key=api_key,
                    model=arguments.model,
                    overwrite=arguments.overwrite,
                )
                print(f"Refined OCR Markdown: {output}")
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        return

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
        formatted_path = output_path.with_name("ocr.formatted.md")
        if arguments.model == "paddleocr-vl" and formatted_path.is_file():
            print(f"OCR formatted Markdown: {formatted_path}")
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

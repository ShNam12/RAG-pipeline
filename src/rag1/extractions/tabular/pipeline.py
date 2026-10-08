"""Run SLANet on layout crops and align its structure with positioned OCR."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image

from rag1.extractions.layouts.contracts import Region, VisualLocation
from rag1.extractions.ocr_text.contracts import OcrDocument
from rag1.extractions.ocr_text.layout_input import load_layout_document, load_upstream_manifest
from rag1.extractions.tabular.flat_output import write_flat_tables
from rag1.extractions.tabular.geometry import TextBox, build_geometry_grid, group_rows, infer_column_anchors
from rag1.extractions.tabular.structure import Grid, parse_structure


DEFAULT_TABLE_OUTPUT_DIR = Path("data/tables")
DEFAULT_TABLE_MODEL = "PaddlePaddle/SLANet_plus_safetensors"
DEFAULT_TABLE_DEVICE = "cpu"
logger = logging.getLogger(__name__)


class SLANetPredictor:
    """Reuse one downloaded Transformers model for all table crops in a run."""

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForTableRecognition

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForTableRecognition.from_pretrained(model_name).to(device).eval()

    def predict(self, image: Image.Image) -> tuple[list[str], float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            result = self.processor.post_process_table_recognition(self.model(**inputs))
        return result["structure"], float(result["structure_score"])


def _image_path(root: Path, image: str) -> Path:
    path = Path(image)
    if path.is_absolute():
        raise ValueError("manifest page images must be relative to the layout directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("manifest page image escapes the layout directory")
    return resolved


def _line_box(line: Any) -> TextBox:
    xs = [point[0] for point in line.page_quad]
    ys = [point[1] for point in line.page_quad]
    return TextBox(line.id, line.text, (min(xs), min(ys), max(xs), max(ys)))


def _ocr_boxes(document: OcrDocument, region: Region) -> list[TextBox]:
    location = region.location
    assert isinstance(location, VisualLocation)
    x0, y0, x1, y1 = location.bbox
    boxes: list[TextBox] = []
    for ocr_region in document.regions:
        if not isinstance(ocr_region.proposal.location, VisualLocation):
            continue
        if ocr_region.proposal.location.page_number != location.page_number:
            continue
        if document.input_mode == "layout" and ocr_region.proposal.id != region.id:
            continue
        for line in ocr_region.lines:
            if not line.text or not line.text.strip():
                continue
            box = _line_box(line)
            center_x = (box.bbox[0] + box.bbox[2]) / 2
            center_y = (box.bbox[1] + box.bbox[3]) / 2
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                boxes.append(box)
        for index, block in enumerate(ocr_region.blocks):
            if not block.content.strip():
                continue
            bx0, by0, bx1, by1 = block.page_bbox
            if x0 <= (bx0 + bx1) / 2 <= x1 and y0 <= (by0 + by1) / 2 <= y1:
                boxes.append(TextBox(f"{ocr_region.proposal.id}:block-{index}", block.content, (bx0, by0, bx1, by1)))
    return boxes


def _align_model_grid(model_grid: Grid, boxes: list[TextBox], anchors: list[float]) -> tuple[Grid, list[TextBox]]:
    rows = group_rows(boxes)
    if len(rows) != len(model_grid.rows):
        raise ValueError(f"SLANet has {len(model_grid.rows)} rows but OCR has {len(rows)}")
    geometry, unassigned = build_geometry_grid(boxes, anchors)
    used: set[str] = set()
    for model_row in model_grid.rows:
        for cell in model_row:
            for row_number in range(cell.row, cell.row + cell.rowspan):
                for source in geometry.rows[row_number][cell.column:cell.column + cell.colspan]:
                    if source.text:
                        cell.text = f"{cell.text} {source.text}".strip()
                        cell.ocr_refs.extend(source.ocr_refs)
                        used.update(source.ocr_refs)
    unassigned.extend(box for box in boxes if box.ref not in used and box not in unassigned)
    return model_grid, unassigned


def _render_html(tables: list[dict[str, Any]]) -> str:
    parts = ["<!doctype html><html lang=\"vi\"><head><meta charset=\"utf-8\"><title>Extracted tables</title></head><body>"]
    for table in tables:
        parts.append(f"<section><h2>{escape(table['region_id'])} - page {table['page_number']}</h2>")
        parts.append(f"<p>Status: {escape(table['status'])}; method: {escape(table['method'])}</p><table border=\"1\">")
        for row in table["rows"]:
            parts.append("<tr>")
            for cell in row:
                parts.append(
                    f'<td rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">{escape(cell["text"])}</td>'
                )
            parts.append("</tr>")
        parts.append("</table></section>")
    return "".join(parts) + "</body></html>\n"


def run_tables(
    layout_json: Path,
    *,
    ocr_json: Path,
    manifest_path: Path | None = None,
    output_dir: Path = DEFAULT_TABLE_OUTPUT_DIR,
    model: str = DEFAULT_TABLE_MODEL,
    device: str = DEFAULT_TABLE_DEVICE,
    region_id: str | None = None,
    predictor: Any | None = None,
) -> Path:
    """Reconstruct visual data tables and return the written JSON path."""
    layout_json = Path(layout_json)
    manifest_path = Path(manifest_path) if manifest_path is not None else layout_json.with_name("manifest.json")
    layout = load_layout_document(layout_json)
    _, pages = load_upstream_manifest(manifest_path, layout)
    try:
        ocr = OcrDocument.model_validate_json(Path(ocr_json).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid OCR JSON {ocr_json}: {error}") from error
    if ocr.input_mode == "layout" and ocr.source != layout.source:
        raise ValueError("OCR source does not match layout source")
    page_map = {page.page_number: page for page in pages}
    ocr_pages = {page.page_number: page for page in ocr.pages}
    table_regions = [
        region for region in layout.regions
        if region.raw_label.strip().lower() == "table" and isinstance(region.location, VisualLocation)
    ]
    table_regions.sort(key=lambda region: (region.location.page_number, region.location.bbox[1], region.location.bbox[0]))
    skipped = [
        {"region_id": region.id, "reason": "document_index is not a data table"}
        for region in layout.regions if region.raw_label.strip().lower() == "document_index"
    ]
    if region_id is not None:
        selected = [region for region in table_regions if region.id == region_id]
        alias = re.fullmatch(r"region-(\d{5})", region_id)
        if not selected and alias:
            page_number = int(alias.group(1))
            selected = [region for region in table_regions if region.location.page_number == page_number]
        if len(selected) != 1:
            raise ValueError(f"--region-id {region_id!r} must identify exactly one data table")
        table_regions = selected
    if not table_regions:
        raise ValueError("layout contains no selected visual data tables")
    logger.info(
        "Table reconstruction started: regions=%d model=%s device=%s",
        len(table_regions), model, device,
    )
    output_path = Path(output_dir) / layout_json.parent.name / "tables.json"
    results: list[dict[str, Any]] = []
    for region in table_regions:
        logger.info("Table region started: id=%s page=%d", region.id,
                    region.location.page_number)
        location = region.location
        assert isinstance(location, VisualLocation)
        page = page_map.get(location.page_number)
        ocr_page = ocr_pages.get(location.page_number)
        if page is None or page.image is None or page.pixel_width is None or page.pixel_height is None:
            raise ValueError(f"missing manifest image or dimensions for page {location.page_number}")
        if (page.pixel_width, page.pixel_height) != (location.page_width, location.page_height):
            raise ValueError(f"layout and manifest dimensions disagree on page {location.page_number}")
        if ocr_page is None or (ocr_page.pixel_width, ocr_page.pixel_height) != (page.pixel_width, page.pixel_height):
            print (ocr_page, ocr_page.pixel_width, ocr_page.pixel_height, page.pixel_width, page.pixel_height)
            raise ValueError(f"OCR and manifest dimensions disagree on page {location.page_number}")
        image_path = _image_path(layout_json.parent, page.image)
        if ocr.input_mode == "direct" and (ocr_page.image is None or Path(ocr_page.image).resolve() != image_path):
            raise ValueError(f"direct OCR image does not match manifest on page {location.page_number}")
        with Image.open(image_path) as image:
            if image.size != (page.pixel_width, page.pixel_height):
                raise ValueError(f"page image dimensions disagree with manifest on page {location.page_number}")
            x0, y0, x1, y1 = location.bbox
            crop = image.crop((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))).convert("RGB")
        boxes = _ocr_boxes(ocr, region)
        anchors = infer_column_anchors(boxes, tuple(location.bbox))
        warnings: list[str] = []
        if predictor is None:
            predictor = SLANetPredictor(model, device)
        try:
            tokens, score = predictor.predict(crop)
            model_grid = parse_structure(tokens)
        except (OSError, RuntimeError, ValueError) as error:
            score = None
            model_grid = None
            warnings.append(f"SLANet structure unavailable: {error}")
        if not boxes:
            warnings.append("No OCR text falls inside this table region")
        if model_grid is not None and anchors and model_grid.column_count == len(anchors):
            try:
                grid, unassigned = _align_model_grid(model_grid, boxes, anchors)
                method = "slanet_ocr"
            except ValueError as error:
                warnings.append(str(error))
                grid, unassigned = build_geometry_grid(boxes, anchors)
                method = "ocr_geometry"
        elif anchors:
            if model_grid is not None:
                warnings.append(f"SLANet has {model_grid.column_count} columns but OCR alignment has {len(anchors)}")
            grid, unassigned = build_geometry_grid(boxes, anchors)
            method = "ocr_geometry"
        elif model_grid is not None:
            grid, unassigned = model_grid, boxes
            method = "slanet"
        else:
            grid, unassigned = Grid([], 0), boxes
            method = "unresolved"
        if unassigned:
            warnings.append(f"{len(unassigned)} OCR items could not be placed confidently")
        status = "complete" if method == "slanet_ocr" and not warnings else "partial"
        logger.info(
            "Table region completed: id=%s status=%s method=%s rows=%d warnings=%d",
            region.id, status, method, len(grid.rows), len(warnings),
        )
        for warning in warnings:
            logger.warning("Table region %s: %s", region.id, warning)
        results.append({
            "region_id": region.id,
            "page_number": location.page_number,
            "bbox": location.bbox,
            "status": status,
            "method": method,
            "structure_score": score,
            "model_column_count": model_grid.column_count if model_grid else None,
            "column_count": grid.column_count,
            "warnings": warnings,
            "rows": [[asdict(cell) for cell in row] for row in grid.rows],
            "unassigned_ocr": [asdict(box) for box in unassigned],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": layout.source,
        "layout_json": str(layout_json.resolve()),
        "manifest": str(manifest_path.resolve()),
        "ocr_json": str(Path(ocr_json).resolve()),
        "model": model,
        "status": "complete" if all(table["status"] == "complete" for table in results) else "partial",
        "tables": results,
        "skipped": skipped,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path.with_suffix(".html").write_text(_render_html(results), encoding="utf-8")
    write_flat_tables(output_path)
    logger.info("Table reconstruction completed: status=%s output=%s",
                payload["status"], output_path)
    return output_path

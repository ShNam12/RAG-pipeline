"""Export reconstructed tables as flat, source-linked row records."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from rag1.extractions.tabular.html_input import parse_table_html


_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$")


def _text_value(raw: str) -> str | None:
    value = unicodedata.normalize("NFC", " ".join(raw.split()))
    return value or None


def _numeric_value(value: str | None) -> str | None:
    if value is None:
        return None
    negative = value.startswith("(") and value.endswith(")")
    candidate = value[1:-1] if negative else value
    if not _NUMBER.fullmatch(candidate):
        return None
    candidate = candidate.replace(".", "").replace(",", ".")
    return f"-{candidate}" if negative else candidate


def flatten_tables(document: dict[str, Any], source_path: str | None = None) -> dict[str, Any]:
    """Create one wide record per table row without guessing header semantics."""
    if document.get("schema_version") != "1.0" or not isinstance(document.get("tables"), list):
        raise ValueError("expected a version 1.0 tables.json document")
    records: list[dict[str, Any]] = []
    table_metadata: list[dict[str, Any]] = []
    unassigned_ocr: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for table in document["tables"]:
        if not isinstance(table, dict):
            raise ValueError("table entries must be objects")
        table_id = table.get("region_id")
        width = table.get("column_count")
        rows = table.get("rows")
        if not isinstance(table_id, str) or not table_id or table_id in seen_ids:
            raise ValueError("table region IDs must be unique nonempty strings")
        seen_ids.add(table_id)
        if type(width) is not int or width < 0 or not isinstance(rows, list):
            raise ValueError(f"invalid column count or rows for table {table_id}")
        table_metadata.append({
            key: table.get(key) for key in (
                "region_id", "page_number", "bbox", "status", "method", "structure_score",
                "model_column_count", "column_count", "warnings",
            )
        })
        for item in table.get("unassigned_ocr", []):
            if not isinstance(item, dict):
                raise ValueError(f"invalid unassigned OCR item in table {table_id}")
            unassigned_ocr.append({"table_id": table_id, **item})
        occupied: dict[tuple[int, int], str] = {}
        for row_index, cells in enumerate(rows):
            if not isinstance(cells, list):
                raise ValueError(f"invalid row {row_index} in table {table_id}")
            for cell in cells:
                if not isinstance(cell, dict):
                    raise ValueError(f"invalid cell in table {table_id}")
                row = cell.get("row")
                column = cell.get("column")
                rowspan = cell.get("rowspan")
                colspan = cell.get("colspan")
                if (
                    type(row) is not int or row != row_index or type(column) is not int
                    or type(rowspan) is not int or type(colspan) is not int
                    or column < 0 or rowspan < 1 or colspan < 1
                    or row + rowspan > len(rows) or column + colspan > width
                ):
                    raise ValueError(f"invalid cell coordinates or span in table {table_id}")
                if not isinstance(cell.get("text"), str) or not isinstance(cell.get("ocr_refs"), list):
                    raise ValueError(f"invalid cell text or OCR references in table {table_id}")
                anchor = f"r{row}c{column}"
                for covered_row in range(row, row + rowspan):
                    for covered_column in range(column, column + colspan):
                        position = (covered_row, covered_column)
                        if position in occupied:
                            raise ValueError(f"table cells overlap in {table_id}")
                        occupied[position] = anchor
        for row_index, cells in enumerate(rows):
            record: dict[str, Any] = {
                "table_id": table_id,
                "page_number": table.get("page_number"),
                "row_index": row_index,
                "record_id": f"{table_id}:r{row_index}",
                "table_status": table.get("status"),
            }
            for column in range(width):
                name = f"column_{column + 1}"
                record[name] = None
                record[f"{name}_raw"] = None
                record[f"{name}_number"] = None
                record[f"{name}_ocr_refs"] = []
                record[f"{name}_rowspan"] = None
                record[f"{name}_colspan"] = None
                record[f"{name}_covered_by"] = occupied.get((row_index, column))
            for cell in cells:
                name = f"column_{cell['column'] + 1}"
                raw = cell["text"]
                value = _text_value(raw)
                record[name] = value
                record[f"{name}_raw"] = raw
                record[f"{name}_number"] = _numeric_value(value)
                record[f"{name}_ocr_refs"] = cell["ocr_refs"].copy()
                record[f"{name}_rowspan"] = cell["rowspan"]
                record[f"{name}_colspan"] = cell["colspan"]
                record[f"{name}_covered_by"] = None
            records.append(record)
    return {
        "schema_version": "1.0",
        "source_format": "json",
        "source": document.get("source"),
        "source_tables_json": source_path,
        "source_tables_html": None,
        "status": document.get("status"),
        "tables": table_metadata,
        "records": records,
        "unassigned_ocr": unassigned_ocr,
        "skipped": document.get("skipped", []),
    }


def flatten_html_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Use each HTML table's first row as keys for its subsequent data rows."""
    tables = document["tables"]
    multiple_tables = len(tables) > 1
    records: list[dict[str, Any]] = []
    for table in tables:
        width = table["column_count"]
        rows = table["rows"]
        if not rows or width == 0:
            continue
        header: list[str | None] = [None] * width
        for cell in rows[0]:
            header[cell["column"]] = _text_value(cell["text"])
        names: list[str] = []
        used = {"_table_id", "_page_number"} if multiple_tables else set()
        for column, candidate in enumerate(header, start=1):
            base = candidate or f"column_{column}"
            name = base
            suffix = 2
            while name in used:
                name = f"{base}__{suffix}"
                suffix += 1
            names.append(name)
            used.add(name)
        for row in rows[1:]:
            values: list[str | None] = [None] * width
            for cell in row:
                values[cell["column"]] = _text_value(cell["text"])
            if not any(value is not None for value in values):
                continue
            record: dict[str, Any] = {}
            if multiple_tables:
                record["_table_id"] = table["region_id"]
                record["_page_number"] = table["page_number"]
            record.update(zip(names, values, strict=True))
            records.append(record)
    return records


def write_flat_tables(source: Path, output: Path | None = None) -> Path:
    """Flatten an existing tables.json or tables.html without model inference."""
    source = Path(source)
    source_format = "html" if source.suffix.lower() in {".html", ".htm"} else "json"
    default_name = "tables.from-html.json" if source_format == "html" else "tables.flat.json"
    output = Path(output) if output is not None else source.with_name(default_name)
    if source.resolve() == output.resolve():
        raise ValueError("flat output must not overwrite its source")
    try:
        contents = source.read_text(encoding="utf-8")
        document = parse_table_html(contents) if source_format == "html" else json.loads(contents)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read table artifact {source}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("table artifact root must be an object")
    flattened = (
        flatten_html_rows(document)
        if source_format == "html"
        else flatten_tables(document, str(source.resolve()))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(flattened, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output

"""Read the table HTML artifact emitted by the tabular pipeline."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any


_HEADING = re.compile(r"^(.+?) - page (\d+)$")
_DETAILS = re.compile(r"^Status: ([a-z]+); method: ([a-z_]+)$")


class _RenderedTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self.section: dict[str, Any] | None = None
        self.capture: str | None = None
        self.parts: list[str] = []
        self.in_table = False
        self.in_row = False
        self.cell: dict[str, Any] | None = None
        self.occupied: set[tuple[int, int]] = set()
        self.table_seen = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            if self.section is not None:
                raise ValueError("nested table sections are unsupported")
            self.section = {"rows": []}
            self.occupied = set()
            self.table_seen = False
        elif tag in {"h2", "p"} and self.section is not None and not self.in_table:
            self.capture = tag
            self.parts = []
        elif tag == "table" and self.section is not None:
            if self.table_seen:
                raise ValueError("each HTML section must contain one table")
            self.in_table = True
            self.table_seen = True
        elif tag == "tr" and self.in_table:
            if self.in_row:
                raise ValueError("nested table rows are unsupported")
            self.in_row = True
            self.section["rows"].append([])
        elif tag in {"td", "th"} and self.in_row:
            if self.cell is not None:
                raise ValueError("nested table cells are unsupported")
            attributes = dict(attrs)
            try:
                rowspan = int(attributes.get("rowspan") or 1)
                colspan = int(attributes.get("colspan") or 1)
            except ValueError as error:
                raise ValueError("invalid HTML table cell span") from error
            if rowspan < 1 or colspan < 1:
                raise ValueError("HTML table cell spans must be positive")
            row = len(self.section["rows"]) - 1
            column = 0
            while any((row, index) in self.occupied for index in range(column, column + colspan)):
                column += 1
            for covered_row in range(row, row + rowspan):
                for covered_column in range(column, column + colspan):
                    self.occupied.add((covered_row, covered_column))
            self.cell = {
                "row": row, "column": column, "rowspan": rowspan,
                "colspan": colspan, "ocr_refs": None,
            }
            self.capture = "cell"
            self.parts = []
        elif tag == "br" and self.capture == "cell":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.capture is not None:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None:
            self.cell["text"] = "".join(self.parts)
            self.section["rows"][-1].append(self.cell)
            self.cell = None
            self.capture = None
            self.parts = []
        elif tag in {"h2", "p"} and self.capture == tag:
            self.section[tag] = "".join(self.parts).strip()
            self.capture = None
            self.parts = []
        elif tag == "tr" and self.in_row:
            if self.cell is not None:
                raise ValueError("HTML table cell was not closed")
            self.in_row = False
        elif tag == "table" and self.in_table:
            if self.in_row:
                raise ValueError("HTML table row was not closed")
            self.in_table = False
        elif tag == "section" and self.section is not None:
            if self.in_table or not self.table_seen:
                raise ValueError("HTML table section has no closed table")
            heading = _HEADING.fullmatch(self.section.get("h2", ""))
            if heading is None:
                raise ValueError("HTML table section has no valid heading")
            details = _DETAILS.fullmatch(self.section.get("p", ""))
            if details is None:
                raise ValueError("HTML table section has no valid status and method")
            rows = self.section["rows"]
            if any(row >= len(rows) for row, _ in self.occupied):
                raise ValueError("HTML table rowspan exceeds its row count")
            width = max((column for _, column in self.occupied), default=-1) + 1
            self.tables.append({
                "region_id": heading.group(1),
                "page_number": int(heading.group(2)),
                "bbox": None,
                "status": details.group(1),
                "method": details.group(2),
                "structure_score": None,
                "model_column_count": None,
                "column_count": width,
                "warnings": None,
                "rows": rows,
                "unassigned_ocr": None,
            })
            self.section = None


def parse_table_html(html: str) -> dict[str, Any]:
    """Recover table rows and spans from the generated HTML artifact."""
    parser = _RenderedTableParser()
    parser.feed(html)
    parser.close()
    if parser.section is not None or parser.cell is not None:
        raise ValueError("HTML table document is incomplete")
    if not parser.tables:
        raise ValueError("HTML document contains no table sections")
    return {
        "schema_version": "1.0",
        "source": None,
        "status": "partial" if any(table["status"] != "complete" for table in parser.tables) else "complete",
        "tables": parser.tables,
        "skipped": None,
    }

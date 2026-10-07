"""Parse SLANet HTML structure tokens into a validated cell grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class Cell:
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    ocr_refs: list[str] = field(default_factory=list)


@dataclass
class Grid:
    rows: list[list[Cell]]
    column_count: int


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[Cell]] = []
        self.spans: dict[int, int] = {}
        self.in_table = False
        self.in_row = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.rows.append([])
        elif tag in {"td", "th"} and self.in_row:
            attributes = dict(attrs)
            try:
                rowspan = int(attributes.get("rowspan") or 1)
                colspan = int(attributes.get("colspan") or 1)
            except ValueError as error:
                raise ValueError("invalid table cell span") from error
            if rowspan < 1 or colspan < 1:
                raise ValueError("table cell span must be positive")
            occupied = set(self.spans)
            occupied.update(
                column
                for cell in self.rows[-1]
                for column in range(cell.column, cell.column + cell.colspan)
            )
            column = 0
            while column in occupied:
                column += 1
            if any(index in occupied for index in range(column, column + colspan)):
                raise ValueError("overlapping table cell spans")
            cell = Cell(len(self.rows) - 1, column, rowspan, colspan)
            self.rows[-1].append(cell)
            if rowspan > 1:
                for index in range(column, column + colspan):
                    self.spans[index] = rowspan

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self.in_row:
            self.in_row = False
            self.spans = {column: count - 1 for column, count in self.spans.items() if count > 1}
        elif tag == "table":
            self.in_table = False


def parse_structure(tokens: list[str]) -> Grid:
    parser = _TableParser()
    parser.feed("".join(tokens))
    if not parser.rows:
        raise ValueError("SLANet returned no table rows")
    widths: list[int] = []
    occupied: dict[int, int] = {}
    for row_number, row in enumerate(parser.rows):
        taken = {column for column, end in occupied.items() if end > row_number}
        for cell in row:
            for column in range(cell.column, cell.column + cell.colspan):
                if column in taken:
                    raise ValueError("overlapping table cell spans")
                taken.add(column)
                if cell.rowspan > 1:
                    occupied[column] = row_number + cell.rowspan
        widths.append(max(taken) + 1 if taken else 0)
    if len(set(widths)) != 1 or widths[0] == 0:
        raise ValueError("SLANet returned inconsistent row widths")
    if any(end > len(parser.rows) for end in occupied.values()):
        raise ValueError("table cell rowspan exceeds row count")
    return Grid(parser.rows, widths[0])

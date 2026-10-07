"""Infer table rows and columns from page-positioned OCR lines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from rag1.extractions.tabular.structure import Cell, Grid


@dataclass(frozen=True)
class TextBox:
    ref: str
    text: str
    bbox: tuple[float, float, float, float]


def group_rows(boxes: list[TextBox]) -> list[list[TextBox]]:
    if not boxes:
        return []
    height = median(box.bbox[3] - box.bbox[1] for box in boxes)
    tolerance = max(8.0, height * 0.8)
    rows: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0])):
        center = (box.bbox[1] + box.bbox[3]) / 2
        if rows and abs(center - median((item.bbox[1] + item.bbox[3]) / 2 for item in rows[-1])) <= tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [sorted(row, key=lambda item: item.bbox[0]) for row in rows]


def infer_column_anchors(
    boxes: list[TextBox], bbox: tuple[float, float, float, float]
) -> list[float]:
    if not boxes:
        return []
    rows = group_rows(boxes)
    candidates: list[tuple[float, float]] = []
    x0, _, x1, _ = bbox
    for x in range(int(x0), int(x1) + 1, 2):
        score = sum(max(0.0, 1.0 - abs(box.bbox[0] - x) / 24.0) for box in boxes)
        candidates.append((score, float(x)))
    candidates.sort(reverse=True)
    minimum = max(min(3.0, float(len(rows))), candidates[0][0] * 0.18)
    anchors: list[float] = []
    for score, x in candidates:
        if score < minimum:
            break
        if any(abs(x - existing) < 60 for existing in anchors):
            continue
        support = sum(any(abs(item.bbox[0] - x) < 24 for item in row) for row in rows)
        if support < min(3, len(rows)):
            continue
        anchors.append(x)
    return sorted(anchors)


def build_geometry_grid(
    boxes: list[TextBox], anchors: list[float]
) -> tuple[Grid, list[TextBox]]:
    if not anchors:
        return Grid([], 0), boxes.copy()
    grid_rows: list[list[Cell]] = []
    unassigned: list[TextBox] = []
    for row_number, row in enumerate(group_rows(boxes)):
        cells = [Cell(row_number, column) for column in range(len(anchors))]
        for box in row:
            nearest = min(range(len(anchors)), key=lambda index: abs(box.bbox[0] - anchors[index]))
            gap = min(
                (abs(anchors[nearest] - anchor) for index, anchor in enumerate(anchors) if index != nearest),
                default=100.0,
            )
            if abs(box.bbox[0] - anchors[nearest]) > max(35.0, gap * 0.45):
                unassigned.append(box)
                continue
            cell = cells[nearest]
            cell.text = f"{cell.text} {box.text}".strip()
            cell.ocr_refs.append(box.ref)
        grid_rows.append(cells)
    return Grid(grid_rows, len(anchors)), unassigned

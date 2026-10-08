"""Module định dạng và chuyển đổi dữ liệu bảng biểu sang Markdown."""

from __future__ import annotations

from typing import Any


def format_table_to_markdown(table_data: Any, raw_text: str = "") -> str:
    """Chuyển đổi dữ liệu bảng biểu (JSON hoặc raw text) thành định dạng Markdown dễ đọc.

    Tham số:
        table_data: Cấu trúc JSON của bảng (nếu có từ partner).
        raw_text: Chuỗi văn bản các dòng OCR từ payload["text"].

    Trả về:
        Chuỗi Markdown biểu diễn bảng biểu.
    """
    # Trường hợp 1: Nếu có cấu trúc headers và rows
    if isinstance(table_data, dict):
        headers = table_data.get("headers")
        rows = table_data.get("rows")
        if headers and isinstance(headers, list) and rows and isinstance(rows, list):
            header_line = "| " + " | ".join(str(h) for h in headers) + " |"
            separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
            row_lines = [
                "| " + " | ".join(str(cell) for cell in row) + " |"
                for row in rows
                if isinstance(row, list)
            ]
            return "\n".join([header_line, separator_line] + row_lines)

    # Trường hợp 2: Nếu có dạng danh sách dicts
    if isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
        headers = list(table_data[0].keys())
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = [
            "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
            for row in table_data
        ]
        return "\n".join([header_line, separator_line] + row_lines)

    # Trường hợp 3: Khi cấu trúc lưới chưa có (pending), dùng trực tiếp raw_text từ OCR
    if raw_text and raw_text.strip():
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(f"- {line}" for line in lines)

    return ""

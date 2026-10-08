"""Render OCR document text for Markdown-compatible output."""

from __future__ import annotations

from pathlib import Path

from rag1.extractions.layouts.contracts import RegionKind
from rag1.extractions.ocr_text.contracts import OcrDocument


def render_ocr_text(document: OcrDocument) -> str:
    """Return recognized text in proposal order, including raw table line text."""
    if document.text is not None:
        return document.text

    blocks: list[str] = []
    for region in document.regions:
        if region.proposal.kind is RegionKind.TABLE:
            if region.blocks:
                content = "\n".join(block.content for block in region.blocks)
            else:
                content = "\n".join(
                    line.text for line in region.lines if line.text is not None
                )
        elif region.proposal.kind in {RegionKind.TEXT, RegionKind.PICTURE, RegionKind.OTHER}:
            content = region.text or ""
        else:
            continue
        if content:
            blocks.append(content)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_ocr_text(document: OcrDocument, output_path: str | Path) -> Path:
    """Write a UTF-8 Markdown-compatible text view of an OCR document."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ocr_text(document), encoding="utf-8", newline="\n")
    return path

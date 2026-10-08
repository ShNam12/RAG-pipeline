"""Shared contracts for replaceable OCR detector and recognizer adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DetectedText:
    """One detector polygon and its confidence score."""

    polygon: list[list[float]]
    confidence: float


class TextRecognizer(Protocol):
    """Recognize one rectified text line and return text with confidence."""

    def recognize(self, image_path: str) -> tuple[str, float]: ...


class OcrAdapter(TextRecognizer, Protocol):
    """Provide the detection and recognition operations used by the pipeline."""

    def initialize(self) -> None: ...

    def detect(self, image_path: str) -> list[DetectedText]: ...


@dataclass(frozen=True)
class RecognizedBlock:
    """One document element in crop coordinates and reading order."""

    bbox: list[float]
    label: str
    content: str


class RegionOcrAdapter(Protocol):
    """Parse complete page images or proposal crops into ordered document elements."""

    def initialize(self) -> None: ...

    def parse_region(self, image_path: str) -> list[RecognizedBlock]: ...

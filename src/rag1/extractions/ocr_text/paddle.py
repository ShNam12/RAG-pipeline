"""Lazy PaddleOCR adapters for text detection and recognition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol


TEXT_DETECTION_MODEL = "PP-OCRv6_small_det"
TEXT_RECOGNITION_MODEL = "PP-OCRv6_small_rec"


class PaddleResult(Protocol):
    @property
    def json(self) -> Mapping[str, Any]: ...


class PaddleVisionModel(Protocol):
    def predict(self, input: Any, *, batch_size: int) -> Iterable[PaddleResult]: ...


def _create_models(*, device: str) -> tuple[PaddleVisionModel, PaddleVisionModel]:
    from paddleocr import TextDetection, TextRecognition

    return (
        TextDetection(model_name=TEXT_DETECTION_MODEL, device=device),
        TextRecognition(model_name=TEXT_RECOGNITION_MODEL, device=device),
    )


class PaddleOcrV6Adapter:
    """Initialize the two requested PaddleOCR models once and reuse them."""

    def __init__(self, *, device: str = "cpu", model_factory=None) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self.device = device.strip()
        self._model_factory = model_factory or _create_models
        self._detector: PaddleVisionModel | None = None
        self._recognizer: PaddleVisionModel | None = None

    def initialize(self) -> None:
        if self._detector is None or self._recognizer is None:
            self._detector, self._recognizer = self._model_factory(device=self.device)

    def detect(self, image_path: str) -> tuple[list[list[list[float]]], list[float]]:
        self.initialize()
        assert self._detector is not None
        results = list(self._detector.predict(input=image_path, batch_size=1))
        if len(results) != 1:
            raise ValueError("text detector must return exactly one result per crop")
        result = results[0].json
        payload = result.get("res", result)
        if not isinstance(payload, Mapping):
            raise ValueError("text detector result must contain a mapping")
        polygons = _as_polygons(payload.get("dt_polys", []))
        scores = _as_scores(payload.get("dt_scores", []))
        if len(polygons) != len(scores):
            raise ValueError("text detector polygons and scores must have equal lengths")
        return polygons, scores

    def recognize(self, image_path: str) -> tuple[str, float]:
        self.initialize()
        assert self._recognizer is not None
        results = list(self._recognizer.predict(input=image_path, batch_size=1))
        if len(results) != 1:
            raise ValueError("text recognizer must return exactly one result per crop")
        result = results[0].json
        payload = result.get("res", result)
        if not isinstance(payload, Mapping):
            raise ValueError("text recognizer result must contain a mapping")
        text = payload.get("rec_text")
        score = payload.get("rec_score")
        if not isinstance(text, str):
            raise ValueError("text recognizer result must contain rec_text")
        return text, _as_scores([score])[0]


def _as_polygons(value: Any) -> list[list[list[float]]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("text detector dt_polys must be a sequence")
    polygons = []
    for polygon in value:
        if hasattr(polygon, "tolist"):
            polygon = polygon.tolist()
        if not isinstance(polygon, Sequence) or len(polygon) != 4:
            raise ValueError("each detected text polygon must contain four vertices")
        vertices = []
        for vertex in polygon:
            if hasattr(vertex, "tolist"):
                vertex = vertex.tolist()
            if not isinstance(vertex, Sequence) or len(vertex) != 2:
                raise ValueError("each text polygon vertex must contain two coordinates")
            vertices.append([float(vertex[0]), float(vertex[1])])
        polygons.append(vertices)
    return polygons


def _as_scores(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("OCR scores must be a sequence")
    scores = [float(score) for score in value]
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError("OCR scores must be finite values in [0, 1]")
    return scores

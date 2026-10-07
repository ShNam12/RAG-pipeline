"""Lazy PaddleOCR adapters for text detection and recognition."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, Protocol

from rag1.extractions.ocr_text.adapters import DetectedText, TextRecognizer
from rag1.extractions.config import load_extraction_config


_CONFIG = load_extraction_config("ocr_text")["models"]["paddleocr_v6"]
TEXT_DETECTION_MODEL = _CONFIG["detection_model"]
TEXT_RECOGNITION_MODEL = _CONFIG["recognition_model"]
TEXT_RECOGNITION_MODEL_DIR = _CONFIG.get("recognition_model_dir")
PADDLE_OCR_BATCH_SIZE = _CONFIG["batch_size"]
PADDLE_OCR_ENABLE_MKLDNN = _CONFIG["enable_mkldnn"]
DEFAULT_OCR_DEVICE = load_extraction_config("ocr_text")["device"]


class PaddleResult(Protocol):
    @property
    def json(self) -> Mapping[str, Any]: ...


class PaddleVisionModel(Protocol):
    def predict(self, input: Any, *, batch_size: int) -> Iterable[PaddleResult]: ...


DetectorFactory = Callable[..., PaddleVisionModel]
RecognizerFactory = Callable[..., PaddleVisionModel]


def _create_detector(*, device: str) -> PaddleVisionModel:
    from paddleocr import TextDetection

    return TextDetection(
        model_name=TEXT_DETECTION_MODEL,
        device=device,
        enable_mkldnn=PADDLE_OCR_ENABLE_MKLDNN,
    )


def _create_recognizer(*, device: str) -> PaddleVisionModel:
    from paddleocr import TextRecognition

    if TEXT_RECOGNITION_MODEL_DIR:
        return TextRecognition(
            model_dir=TEXT_RECOGNITION_MODEL_DIR,
            device=device,
            enable_mkldnn=PADDLE_OCR_ENABLE_MKLDNN,
        )
    return TextRecognition(
        model_name=TEXT_RECOGNITION_MODEL,
        device=device,
        enable_mkldnn=PADDLE_OCR_ENABLE_MKLDNN,
    )


class PaddleOcrV6Adapter:
    """Use PaddleOCR v6 detection with an optional replaceable recognizer."""

    def __init__(
        self,
        *,
        device: str = DEFAULT_OCR_DEVICE,
        detector_factory: DetectorFactory | None = None,
        recognizer_factory: RecognizerFactory | None = None,
        recognizer: TextRecognizer | None = None,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self.device = device.strip()
        self._detector_factory = detector_factory or _create_detector
        self._recognizer_factory = recognizer_factory or _create_recognizer
        self._recognizer_adapter = recognizer
        self._detector: PaddleVisionModel | None = None
        self._recognizer: PaddleVisionModel | None = None

    def initialize(self) -> None:
        """Initialize the detector only; recognition remains lazy until needed."""
        if self._detector is None:
            self._detector = self._detector_factory(device=self.device)

    def detect(self, image_path: str) -> list[DetectedText]:
        self.initialize()
        assert self._detector is not None
        results = list(
            self._detector.predict(input=image_path, batch_size=PADDLE_OCR_BATCH_SIZE)
        )
        if len(results) != 1:
            raise ValueError("text detector must return exactly one result per crop")
        result = results[0].json
        payload = result.get("res", result)
        if not isinstance(payload, Mapping):
            raise ValueError("text detector result must contain a mapping")
        if "dt_polys" not in payload or "dt_scores" not in payload:
            raise ValueError("text detector result must contain dt_polys and dt_scores")
        polygons = _as_polygons(payload["dt_polys"])
        scores = _as_scores(payload["dt_scores"])
        if len(polygons) != len(scores):
            raise ValueError("text detector polygons and scores must have equal lengths")
        return [
            DetectedText(polygon=polygon, confidence=score)
            for polygon, score in zip(polygons, scores, strict=True)
        ]

    def recognize(self, image_path: str) -> tuple[str, float]:
        if self._recognizer_adapter is not None:
            return self._recognizer_adapter.recognize(image_path)
        if self._recognizer is None:
            self._recognizer = self._recognizer_factory(device=self.device)
        results = list(
            self._recognizer.predict(input=image_path, batch_size=PADDLE_OCR_BATCH_SIZE)
        )
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
    polygons: list[list[list[float]]] = []
    for polygon in value:
        if hasattr(polygon, "tolist"):
            polygon = polygon.tolist()
        if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)) or len(polygon) != 4:
            raise ValueError("each detected text polygon must contain four vertices")
        vertices: list[list[float]] = []
        for vertex in polygon:
            if hasattr(vertex, "tolist"):
                vertex = vertex.tolist()
            if not isinstance(vertex, Sequence) or isinstance(vertex, (str, bytes)) or len(vertex) != 2:
                raise ValueError("each text polygon vertex must contain two coordinates")
            vertices.append([_finite_number(value, "polygon coordinate") for value in vertex])
        area = abs(sum(
            vertices[index][0] * vertices[(index + 1) % 4][1]
            - vertices[(index + 1) % 4][0] * vertices[index][1]
            for index in range(4)
        )) / 2
        if area <= 0:
            raise ValueError("detected text polygon must have positive area")
        polygons.append(vertices)
    return polygons


def _as_scores(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("OCR scores must be a sequence")
    return [_bounded_score(score) for score in value]


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_score(value: Any) -> float:
    score = _finite_number(value, "OCR score")
    if not 0 <= score <= 1:
        raise ValueError("OCR scores must be in [0, 1]")
    return score

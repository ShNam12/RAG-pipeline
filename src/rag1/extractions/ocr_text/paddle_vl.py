"""PaddleOCR-VL adapter for parsing complete proposal crops."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, Protocol

from rag1.extractions.ocr_text.adapters import RecognizedBlock
from rag1.extractions.config import load_extraction_config


_OCR_CONFIG = load_extraction_config("ocr_text")
DEFAULT_OCR_DEVICE = _OCR_CONFIG["device"]
PADDLE_OCR_VL_PIPELINE_VERSION = _OCR_CONFIG["models"]["paddleocr_vl"]["pipeline_version"]


class PaddleResult(Protocol):
    @property
    def json(self) -> Mapping[str, Any]: ...


class PaddleVlModel(Protocol):
    def predict(self, input: str) -> Iterable[PaddleResult]: ...


ModelFactory = Callable[..., PaddleVlModel]


def _create_model(*, device: str) -> PaddleVlModel:
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(
        device=device,
        pipeline_version=PADDLE_OCR_VL_PIPELINE_VERSION,
    )


class PaddleOcrVlAdapter:
    """Create one VL pipeline and parse each proposal crop in reading order."""

    def __init__(
        self,
        *,
        device: str = DEFAULT_OCR_DEVICE,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self.device = device.strip()
        self._model_factory = model_factory or _create_model
        self._model: PaddleVlModel | None = None

    def initialize(self) -> None:
        if self._model is None:
            self._model = self._model_factory(device=self.device)

    def parse_region(self, image_path: str) -> list[RecognizedBlock]:
        self.initialize()
        assert self._model is not None
        results = list(self._model.predict(input=image_path))
        if len(results) != 1:
            raise ValueError("PaddleOCR-VL must return exactly one result per crop")
        result = results[0].json
        if not isinstance(result, Mapping):
            raise ValueError("PaddleOCR-VL result must be a mapping")
        payload = result.get("res", result)
        if not isinstance(payload, Mapping):
            raise ValueError("PaddleOCR-VL result must contain a mapping")
        entries = payload.get("parsing_res_list")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("PaddleOCR-VL result must contain parsing_res_list")
        blocks: list[RecognizedBlock] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("PaddleOCR-VL blocks must be mappings")
            label = entry.get("block_label")
            content = entry.get("block_content")
            if not isinstance(label, str) or not label:
                raise ValueError("PaddleOCR-VL block_label must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError("PaddleOCR-VL block_content must be a string")
            blocks.append(
                RecognizedBlock(
                    bbox=_as_bbox(entry.get("block_bbox")),
                    label=label,
                    content=content,
                )
            )
        return blocks


def _as_bbox(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("PaddleOCR-VL block_bbox must contain four coordinates")
    bbox: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise ValueError("PaddleOCR-VL block_bbox coordinates must be real numbers")
        numeric = float(coordinate)
        if not math.isfinite(numeric):
            raise ValueError("PaddleOCR-VL block_bbox coordinates must be finite")
        bbox.append(numeric)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("PaddleOCR-VL block_bbox must have positive area")
    return bbox

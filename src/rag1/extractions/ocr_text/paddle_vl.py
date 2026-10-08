"""PaddleOCR-VL adapter for parsing page images and proposal crops."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from time import perf_counter
from typing import Any, Protocol

from rag1.extractions.ocr_text.adapters import RecognizedBlock
from rag1.extractions.config import load_extraction_config


_OCR_CONFIG = load_extraction_config("ocr_text")
DEFAULT_OCR_DEVICE = _OCR_CONFIG["device"]
PADDLE_OCR_VL_PIPELINE_VERSION = _OCR_CONFIG["models"]["paddleocr_vl"]["pipeline_version"]
logger = logging.getLogger(__name__)


class PaddleResult(Protocol):
    @property
    def json(self) -> Mapping[str, Any]: ...

    @property
    def markdown(self) -> Mapping[str, Any]: ...


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
    """Create one VL pipeline and parse each input image in reading order."""

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
        self._last_result: PaddleResult | None = None

    def initialize(self) -> None:
        if self._model is None:
            self._model = self._model_factory(device=self.device)

    def parse_region(self, image_path: str) -> list[RecognizedBlock]:
        self.initialize()
        assert self._model is not None
        self._last_result = None
        started_at = perf_counter()
        logger.info(
            "PaddleOCR-VL predict started: input=%s device=%s pipeline_version=%s",
            image_path,
            self.device,
            PADDLE_OCR_VL_PIPELINE_VERSION,
        )
        try:
            results = list(self._model.predict(input=image_path))
        except Exception:
            logger.exception(
                "PaddleOCR-VL predict failed: input=%s elapsed=%.2f seconds",
                image_path,
                perf_counter() - started_at,
            )
            raise
        logger.info(
            "PaddleOCR-VL predict completed: input=%s results=%d elapsed=%.2f seconds",
            image_path,
            len(results),
            perf_counter() - started_at,
        )
        if len(results) != 1:
            raise ValueError("PaddleOCR-VL must return exactly one result per image")
        self._last_result = results[0]
        result = self._last_result.json
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
        logger.info("PaddleOCR-VL parsed blocks: input=%s blocks=%d", image_path, len(blocks))
        return blocks

    def last_markdown(self) -> Mapping[str, Any]:
        """Return the native Markdown from the most recent successful prediction."""
        if self._last_result is None:
            raise RuntimeError("PaddleOCR-VL has no prediction result for Markdown")
        markdown = self._last_result.markdown
        if not isinstance(markdown, Mapping):
            raise ValueError("PaddleOCR-VL Markdown result must be a mapping")
        if not isinstance(markdown.get("markdown_texts"), str):
            raise ValueError("PaddleOCR-VL Markdown result must contain markdown_texts")
        if not isinstance(markdown.get("markdown_images"), Mapping):
            raise ValueError("PaddleOCR-VL Markdown result must contain markdown_images")
        return markdown


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

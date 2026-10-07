"""VietOCR text-line recognition adapter."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any, Protocol

from PIL import Image

from rag1.extractions.config import load_extraction_config


VIETOCR_CONFIG = load_extraction_config("ocr_text")["models"]["vietocr"]["config_name"]
DEFAULT_OCR_DEVICE = load_extraction_config("ocr_text")["device"]


class VietOcrPredictor(Protocol):
    def predict(self, img: Image.Image, return_prob: bool = False) -> Any: ...


PredictorFactory = Callable[..., VietOcrPredictor]


def _torch_device(device: str) -> str:
    if device == "gpu":
        return "cuda"
    if device.startswith("gpu:"):
        return "cuda:" + device.removeprefix("gpu:")
    return device


def _create_predictor(*, device: str, config_name: str) -> VietOcrPredictor:
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    config = Cfg.load_config_from_name(config_name)
    config["device"] = _torch_device(device)
    return Predictor(config)


class VietOcrAdapter:
    """Load one VietOCR predictor lazily and reuse it for text-line crops."""

    def __init__(
        self,
        *,
        device: str = DEFAULT_OCR_DEVICE,
        config_name: str = VIETOCR_CONFIG,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(config_name, str) or not config_name.strip():
            raise ValueError("config_name must be a non-empty string")
        self.device = device.strip()
        self.config_name = config_name.strip()
        self._predictor_factory = predictor_factory or _create_predictor
        self._predictor: VietOcrPredictor | None = None

    def initialize(self) -> None:
        """Create the predictor once when recognition is first needed."""
        if self._predictor is None:
            self._predictor = self._predictor_factory(
                device=self.device,
                config_name=self.config_name,
            )

    def recognize(self, image_path: str) -> tuple[str, float]:
        self.initialize()
        assert self._predictor is not None
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            text, confidence = self._predictor.predict(image, return_prob=True)
        if not isinstance(text, str):
            raise ValueError("VietOCR must return recognized text as a string")
        if hasattr(confidence, "item"):
            confidence = confidence.item()
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError("VietOCR confidence must be a real number")
        score = float(confidence)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("VietOCR confidence must be finite and in [0, 1]")
        return text, score

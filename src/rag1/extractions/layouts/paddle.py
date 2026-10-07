"""PaddleOCR layout detector adapter."""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from rag1.extractions.layouts.contracts import (
    LayoutDocument,
    PageFailure,
    Region,
    VisualLocation,
    normalize_detector_label,
)
from rag1.extractions.config import load_extraction_config


_CONFIG = load_extraction_config("layouts")
_MODEL_CONFIG = _CONFIG["model"]
PADDLE_LAYOUT_MODEL_NAME = _MODEL_CONFIG["name"]
DEFAULT_LAYOUT_DEVICE = _MODEL_CONFIG["device"]
PADDLE_LAYOUT_BATCH_SIZE = _MODEL_CONFIG["batch_size"]
PADDLE_LAYOUT_ENABLE_MKLDNN = _MODEL_CONFIG["enable_mkldnn"]


@dataclass(frozen=True)
class RenderedPage:
    """A rendered page image and its dimensions in pixels."""

    page_number: int
    image: Any
    page_width: int
    page_height: int
    rotation_degrees: int = 0


class PaddleResult(Protocol):
    @property
    def json(self) -> Mapping[str, Any]: ...


class PaddleLayoutModel(Protocol):
    def predict(self, input: Any, *, batch_size: int) -> Iterable[PaddleResult]: ...


ModelFactory = Callable[..., PaddleLayoutModel]


def _clamp_visual_box(
    coordinate: Any,
    *,
    page_width: int,
    page_height: int,
) -> list[float]:
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page dimensions must be positive")
    if (
        isinstance(coordinate, (str, bytes))
        or not isinstance(coordinate, Sequence)
        or len(coordinate) != 4
    ):
        raise ValueError("bounding box must contain four coordinates")

    values: list[float] = []
    for value in coordinate:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("bounding box coordinates must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("bounding box coordinates must be finite")
        values.append(numeric_value)

    x0, y0, x1, y1 = values
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bounding box must have positive width and height")

    clamped = [
        min(max(x0, 0.0), float(page_width)),
        min(max(y0, 0.0), float(page_height)),
        min(max(x1, 0.0), float(page_width)),
        min(max(y1, 0.0), float(page_height)),
    ]
    if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
        raise ValueError("bounding box does not overlap the page")
    return clamped


def _create_model(*, model_name: str, device: str) -> PaddleLayoutModel:
    from paddleocr import LayoutDetection

    # Pinned PaddlePaddle 3.3.0 CPU inference fails in its oneDNN PIR conversion.
    return LayoutDetection(
        model_name=model_name,
        device=device,
        enable_mkldnn=PADDLE_LAYOUT_ENABLE_MKLDNN,
    )


class PaddleLayoutAdapter:
    """Create one Paddle layout model and reuse it for every page in a document."""

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        *,
        device: str = _MODEL_CONFIG["device"],
        batch_size: int = PADDLE_LAYOUT_BATCH_SIZE,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self._model_factory = model_factory or _create_model
        self.device = device.strip()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

    def extract_document(
        self,
        *,
        source: str,
        source_format: str,
        pages: Sequence[RenderedPage],
    ) -> LayoutDocument:
        regions: list[Region] = []
        errors: list[PageFailure] = []

        model = None
        if pages:
            try:
                model = self._model_factory(
                    model_name=PADDLE_LAYOUT_MODEL_NAME,
                    device=self.device,
                )
            except Exception as error:
                errors.extend(
                    PageFailure.from_exception(page_number=page.page_number, error=error)
                    for page in pages
                )

        for page in pages:
            try:
                if model is None:
                    continue
                regions.extend(self._detect_page(model, page))
            except Exception as error:
                errors.append(
                    PageFailure.from_exception(page_number=page.page_number, error=error)
                )

        status = "complete"
        if errors:
            status = "partial" if regions else "failed"

        return LayoutDocument(
            schema_version="1.0",
            source=source,
            source_format=source_format,
            status=status,
            regions=regions,
            errors=errors,
        )

    def _detect_page(self, model: PaddleLayoutModel, page: RenderedPage) -> list[Region]:
        regions: list[Region] = []
        region_index = 0

        for prediction in model.predict(page.image, batch_size=self.batch_size):
            prediction_data = prediction.json
            result_data = prediction_data.get("res", prediction_data)
            boxes = result_data.get("boxes", [])

            for box in boxes:
                region_index += 1
                if not isinstance(box, Mapping):
                    raise ValueError("detector box must be a mapping")
                label = box.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("detector box label must be a non-empty string")
                kind, role = normalize_detector_label(label)
                regions.append(
                    Region(
                        id=f"p{page.page_number:04d}-r{region_index:04d}",
                        kind=kind,
                        role=role,
                        raw_label=label,
                        confidence=float(box["score"]),
                        location=VisualLocation(
                            page_number=page.page_number,
                            bbox=_clamp_visual_box(
                                box.get("coordinate"),
                                page_width=page.page_width,
                                page_height=page.page_height,
                            ),
                            page_width=page.page_width,
                            page_height=page.page_height,
                        ),
                    )
                )

        return regions

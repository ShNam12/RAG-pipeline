"""Tests for the PaddleOCR layout adapter."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from rag1.extractions.layouts.contracts import RegionKind, RegionRole, VisualLocation
from rag1.extractions.layouts.paddle import (
    PADDLE_LAYOUT_MODEL_NAME,
    PaddleLayoutAdapter,
    RenderedPage,
    _create_model,
)


class FakePrediction:
    def __init__(self, boxes: list[dict[str, object]]) -> None:
        self.json = {"res": {"boxes": boxes}}


class FakeModel:
    def __init__(self) -> None:
        self.inputs: list[object] = []
        self.batch_sizes: list[int] = []

    def predict(self, input: object, *, batch_size: int) -> list[FakePrediction]:
        self.inputs.append(input)
        self.batch_sizes.append(batch_size)
        if input == "page-2.png":
            raise RuntimeError("layout inference failed")
        return [
            FakePrediction(
                [
                    {
                        "label": "table",
                        "score": 0.91,
                        "coordinate": [10, 20, 90, 120],
                    }
                ]
            )
        ]


class PaddleLayoutAdapterTests(unittest.TestCase):
    def test_model_factory_disables_mkldnn_for_cpu_runtime(self) -> None:
        paddleocr_module = ModuleType("paddleocr")
        layout_detection = Mock(return_value=object())
        paddleocr_module.LayoutDetection = layout_detection

        with patch.dict(sys.modules, {"paddleocr": paddleocr_module}):
            model = _create_model(model_name=PADDLE_LAYOUT_MODEL_NAME, device="cpu")

        self.assertIs(model, layout_detection.return_value)
        layout_detection.assert_called_once_with(
            model_name=PADDLE_LAYOUT_MODEL_NAME,
            device="cpu",
            enable_mkldnn=False,
        )

    def test_model_is_initialized_once_and_regions_use_shared_contract(self) -> None:
        model = FakeModel()
        factory_calls: list[dict[str, object]] = []

        def create_model(**kwargs: object) -> FakeModel:
            factory_calls.append(kwargs)
            return model

        adapter = PaddleLayoutAdapter(model_factory=create_model)
        document = adapter.extract_document(
            source="report.pdf",
            source_format="pdf",
            pages=[
                RenderedPage(1, "page-1.png", 100, 130),
                RenderedPage(2, "page-2.png", 100, 130),
                RenderedPage(3, "page-3.png", 100, 130),
            ],
        )

        self.assertEqual(
            factory_calls,
            [{"model_name": PADDLE_LAYOUT_MODEL_NAME, "device": "cpu"}],
        )
        self.assertEqual(model.inputs, ["page-1.png", "page-2.png", "page-3.png"])
        self.assertEqual(model.batch_sizes, [1, 1, 1])
        self.assertEqual(document.status, "partial")
        self.assertEqual(len(document.regions), 2)
        region = document.regions[0]
        self.assertEqual(region.kind, RegionKind.TABLE)
        self.assertEqual(region.role, RegionRole.BODY)
        self.assertEqual(region.raw_label, "table")
        self.assertEqual(region.confidence, 0.91)
        self.assertEqual(
            region.location,
            VisualLocation(
                page_number=1,
                bbox=[10, 20, 90, 120],
                page_width=100,
                page_height=130,
            ),
        )
        self.assertEqual(document.errors[0].page_number, 2)
        self.assertEqual(document.errors[0].message, "layout inference failed")
        self.assertEqual(document.regions[1].location.page_number, 3)
        self.assertNotEqual(document.regions[0].id, document.regions[1].id)

    def test_picture_labels_are_preserved_and_boxes_are_clamped(self) -> None:
        class PictureModel:
            def predict(self, input: object, *, batch_size: int) -> list[FakePrediction]:
                return [
                    FakePrediction(
                        [
                            {
                                "label": "chart",
                                "score": 0.8,
                                "coordinate": [-5, 10, 105, 80],
                            },
                            {
                                "label": "seal",
                                "score": 0.7,
                                "coordinate": [20, 20, 40, 40],
                            },
                        ]
                    )
                ]

        adapter = PaddleLayoutAdapter(model_factory=lambda **kwargs: PictureModel())
        document = adapter.extract_document(
            source="images.pdf",
            source_format="pdf",
            pages=[RenderedPage(4, "page-4.png", 100, 90)],
        )

        self.assertEqual(
            [region.raw_label for region in document.regions],
            ["chart", "seal"],
        )
        self.assertTrue(
            all(region.kind is RegionKind.PICTURE for region in document.regions)
        )
        self.assertEqual(document.regions[0].location.page_number, 4)
        self.assertEqual(document.regions[0].location.bbox, [0, 10, 100, 80])

    def test_malformed_box_records_page_failure_and_discards_that_page(self) -> None:
        class MalformedBoxModel:
            def predict(self, input: object, *, batch_size: int) -> list[FakePrediction]:
                if input == "page-2.png":
                    return [
                        FakePrediction(
                            [
                                {
                                    "label": "text",
                                    "score": 0.9,
                                    "coordinate": [10, 10, 20, 20],
                                },
                                {
                                    "label": "text",
                                    "score": 0.8,
                                    "coordinate": [1, 2, 3],
                                },
                            ]
                        )
                    ]
                return [
                    FakePrediction(
                        [
                            {
                                "label": "text",
                                "score": 0.9,
                                "coordinate": [10, 10, 20, 20],
                            }
                        ]
                    )
                ]

        adapter = PaddleLayoutAdapter(model_factory=lambda **kwargs: MalformedBoxModel())
        document = adapter.extract_document(
            source="report.pdf",
            source_format="pdf",
            pages=[
                RenderedPage(1, "page-1.png", 100, 130),
                RenderedPage(2, "page-2.png", 100, 130),
                RenderedPage(3, "page-3.png", 100, 130),
            ],
        )

        self.assertEqual(document.status, "partial")
        self.assertEqual(
            [region.location.page_number for region in document.regions],
            [1, 3],
        )
        self.assertEqual([error.page_number for error in document.errors], [2])
        self.assertIn("four coordinates", document.errors[0].message)

    def test_no_pages_does_not_initialize_a_model(self) -> None:
        factory_calls: list[dict[str, object]] = []

        def create_model(**kwargs: object) -> FakeModel:
            factory_calls.append(kwargs)
            return FakeModel()

        adapter = PaddleLayoutAdapter(model_factory=create_model)

        document = adapter.extract_document(
            source="empty.pdf",
            source_format="pdf",
            pages=[],
        )

        self.assertEqual(factory_calls, [])
        self.assertEqual(document.status, "complete")
        self.assertEqual(document.regions, [])

    def test_configured_device_is_passed_to_the_model_factory(self) -> None:
        factory_calls: list[dict[str, object]] = []

        def create_model(**kwargs: object) -> FakeModel:
            factory_calls.append(kwargs)
            return FakeModel()

        adapter = PaddleLayoutAdapter(model_factory=create_model, device="gpu:0")
        adapter.extract_document(
            source="report.pdf",
            source_format="pdf",
            pages=[RenderedPage(1, "page-1.png", 100, 130)],
        )

        self.assertEqual(factory_calls[0]["device"], "gpu:0")

    def test_model_initialization_failure_marks_every_page_failed(self) -> None:
        def fail_to_create_model(**kwargs: object) -> FakeModel:
            raise RuntimeError("model weights are unavailable")

        adapter = PaddleLayoutAdapter(model_factory=fail_to_create_model)
        document = adapter.extract_document(
            source="report.pdf",
            source_format="pdf",
            pages=[
                RenderedPage(1, "page-1.png", 100, 130),
                RenderedPage(2, "page-2.png", 100, 130),
            ],
        )

        self.assertEqual(document.status, "failed")
        self.assertEqual([error.page_number for error in document.errors], [1, 2])
        self.assertTrue(
            all(
                error.message == "model weights are unavailable"
                for error in document.errors
            )
        )


if __name__ == "__main__":
    unittest.main()

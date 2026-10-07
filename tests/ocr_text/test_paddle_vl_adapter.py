"""PaddleOCR-VL result mapping without loading model weights."""

import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from rag1.extractions.ocr_text.paddle_vl import PaddleOcrVlAdapter, _create_model


class FakeResult:
    def __init__(self, payload):
        self.json = {"res": payload}


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.inputs = []

    def predict(self, input):
        self.inputs.append(input)
        return [FakeResult(self.payload)]


class PaddleVlAdapterTests(unittest.TestCase):
    def test_factory_uses_paddleocr_vl_pipeline(self):
        paddleocr = ModuleType("paddleocr")
        paddleocr.PaddleOCRVL = Mock(return_value=object())
        with patch.dict("sys.modules", {"paddleocr": paddleocr}):
            _create_model(device="cpu")
        paddleocr.PaddleOCRVL.assert_called_once_with(
            device="cpu",
            pipeline_version="v1.6",
        )

    def test_maps_ordered_blocks_and_reuses_pipeline(self):
        model = FakeModel({
            "parsing_res_list": [
                {"block_bbox": [1, 2, 11, 12], "block_label": "text", "block_content": "First"},
                {"block_bbox": [2, 14, 25, 30], "block_label": "table", "block_content": "| A | B |"},
            ]
        })
        created = []

        def create_model(*, device):
            created.append(device)
            return model

        adapter = PaddleOcrVlAdapter(device="cpu", model_factory=create_model)
        first = adapter.parse_region("first.png")
        second = adapter.parse_region("second.png")

        self.assertEqual(created, ["cpu"])
        self.assertEqual(model.inputs, ["first.png", "second.png"])
        self.assertEqual([(block.label, block.content) for block in first],
                         [("text", "First"), ("table", "| A | B |")])
        self.assertEqual(second[0].bbox, [1.0, 2.0, 11.0, 12.0])

    def test_rejects_missing_or_invalid_block_geometry(self):
        for bbox in (None, [1, 2, 3], [5, 2, 1, 12]):
            with self.subTest(bbox=bbox):
                model = FakeModel({
                    "parsing_res_list": [
                        {"block_bbox": bbox, "block_label": "text", "block_content": "A"}
                    ]
                })
                adapter = PaddleOcrVlAdapter(model_factory=lambda *, device: model)
                with self.assertRaises(ValueError):
                    adapter.parse_region("region.png")


if __name__ == "__main__":
    unittest.main()

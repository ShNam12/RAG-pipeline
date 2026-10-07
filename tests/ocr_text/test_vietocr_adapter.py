"""VietOCR recognizer adapter lifecycle and result contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from rag1.extractions.ocr_text.vietocr import VietOcrAdapter, _torch_device


class VietOcrAdapterTests(unittest.TestCase):
    def test_predictor_is_created_lazily_and_reused(self) -> None:
        predictor = Mock()
        predictor.predict.return_value = ("Xin chào", 0.91)
        factory = Mock(return_value=predictor)
        adapter = VietOcrAdapter(device="cpu", predictor_factory=factory)

        factory.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "line.png"
            Image.new("RGB", (20, 10), "white").save(image_path)

            self.assertEqual(adapter.recognize(str(image_path)), ("Xin chào", 0.91))
            self.assertEqual(adapter.recognize(str(image_path)), ("Xin chào", 0.91))

        factory.assert_called_once_with(device="cpu", config_name="vgg_transformer")
        self.assertEqual(predictor.predict.call_count, 2)
        self.assertTrue(predictor.predict.call_args.kwargs["return_prob"])

    def test_rejects_invalid_confidence(self) -> None:
        predictor = Mock()
        predictor.predict.return_value = ("text", float("nan"))
        adapter = VietOcrAdapter(
            predictor_factory=Mock(return_value=predictor),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "line.png"
            Image.new("RGB", (20, 10), "white").save(image_path)

            with self.assertRaises(ValueError):
                adapter.recognize(str(image_path))

    def test_maps_paddle_gpu_device_names_to_torch_names(self) -> None:
        self.assertEqual(_torch_device("gpu"), "cuda")
        self.assertEqual(_torch_device("gpu:0"), "cuda:0")
        self.assertEqual(_torch_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()

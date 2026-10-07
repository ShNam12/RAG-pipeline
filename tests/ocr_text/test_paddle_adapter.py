"""PaddleOCR result decoding and adapter lifecycle contracts."""

import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from rag1.extractions.ocr_text.paddle import (
    TEXT_DETECTION_MODEL,
    TEXT_RECOGNITION_MODEL,
    DetectedText,
    PaddleOcrV6Adapter,
    _create_detector,
    _create_recognizer,
)


class FakeResult:
    def __init__(self, result):
        self.json = {"res": result}


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.inputs = []

    def predict(self, *, input, batch_size):
        self.inputs.append((input, batch_size))
        return iter(self.results)


class PaddleAdapterTests(unittest.TestCase):
    def test_detector_factory_disables_mkldnn_for_cpu_compatibility(self):
        paddleocr = ModuleType("paddleocr")
        paddleocr.TextDetection = Mock(return_value=object())

        with patch.dict("sys.modules", {"paddleocr": paddleocr}):
            _create_detector(device="cpu")

        paddleocr.TextDetection.assert_called_once_with(
            model_name=TEXT_DETECTION_MODEL,
            device="cpu",
            enable_mkldnn=False,
        )

    def test_recognizer_factory_uses_checkpoint_cpu_default_and_mkldnn_compatibility(self):
        paddleocr = ModuleType("paddleocr")
        paddleocr.TextRecognition = Mock(return_value=object())

        with patch.dict("sys.modules", {"paddleocr": paddleocr}):
            _create_recognizer(device="cpu")

        paddleocr.TextRecognition.assert_called_once_with(
            model_name=TEXT_RECOGNITION_MODEL,
            device="cpu",
            enable_mkldnn=False,
        )

    def test_detection_is_lazy_singleton_and_recognition_is_initialized_on_demand(self):
        detector = FakeModel([
            FakeResult({
                "dt_polys": [[[1, 2], [11, 2], [11, 12], [1, 12]]],
                "dt_scores": [0.87],
            }),
            FakeResult({
                "dt_polys": [[[2, 3], [12, 3], [12, 13], [2, 13]]],
                "dt_scores": [0.72],
            }),
        ])
        recognizer = FakeModel([
            FakeResult({"rec_text": " text\t", "rec_score": 0.61}),
        ])
        detector_calls = []
        recognizer_calls = []

        def create_detector(*, device):
            detector_calls.append(device)
            return detector

        def create_recognizer(*, device):
            recognizer_calls.append(device)
            return recognizer

        adapter = PaddleOcrV6Adapter(
            device="cpu",
            detector_factory=create_detector,
            recognizer_factory=create_recognizer,
        )
        self.assertEqual(detector_calls, [])
        self.assertEqual(recognizer_calls, [])

        detections = adapter.detect("region.png")
        repeated_detection = adapter.detect("region-2.png")
        self.assertEqual(detector_calls, ["cpu"])
        self.assertEqual(recognizer_calls, [])
        text, recognition_score = adapter.recognize("line.png")

        self.assertEqual(recognizer_calls, ["cpu"])
        self.assertEqual(
            detections,
            [DetectedText(
                polygon=[[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]],
                confidence=0.87,
            )],
        )
        self.assertEqual(repeated_detection[0].confidence, 0.72)
        self.assertEqual((text, recognition_score), (" text\t", 0.61))
        self.assertEqual(detector.inputs, [("region.png", 1), ("region-2.png", 1)])
        self.assertEqual(recognizer.inputs, [("line.png", 1)])

    def test_recognition_backend_can_be_replaced(self):
        recognition_backend = Mock()
        recognition_backend.recognize.return_value = ("Xin chào", 0.94)
        adapter = PaddleOcrV6Adapter(recognizer=recognition_backend)

        self.assertEqual(adapter.recognize("line.png"), ("Xin chào", 0.94))
        recognition_backend.recognize.assert_called_once_with("line.png")

    def test_rejects_mismatched_counts_and_malformed_predictions(self):
        invalid_results = [
            {"dt_polys": [], "dt_scores": [0.5]},
            {"dt_polys": []},
            {"dt_scores": []},
            {"dt_polys": [], "dt_scores": 0.5},
            {"dt_polys": [[[1, 2], [11, 2], [11, 12]]], "dt_scores": [0.5]},
            {"dt_polys": [[[1, 2], [5, 2], [9, 2], [1, 2]]], "dt_scores": [0.5]},
            {
                "dt_polys": [[[1, 2], [float("inf"), 2], [11, 12], [1, 12]]],
                "dt_scores": [0.5],
            },
            {"dt_polys": [[[1, 2], [11, 2], [11, 12], [1, 12]]], "dt_scores": [1.1]},
            {"dt_polys": [[[1, 2], [11, 2], [11, 12], [1, 12]]], "dt_scores": [float("nan")]},
        ]
        for result in invalid_results:
            with self.subTest(result=result):
                detector = FakeModel([FakeResult(result)])
                adapter = PaddleOcrV6Adapter(
                    detector_factory=lambda *, device: detector,
                )
                with self.assertRaises(ValueError):
                    adapter.detect("region.png")


if __name__ == "__main__":
    unittest.main()

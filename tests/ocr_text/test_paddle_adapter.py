"""PaddleOCR result decoding and adapter lifecycle contracts."""

import unittest

from rag1.extractions.ocr_text.paddle import PaddleOcrV6Adapter


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
    def test_models_are_created_once_and_detection_and_recognition_are_decoded(self):
        detector = FakeModel([
            FakeResult({
                "dt_polys": [[[1, 2], [11, 2], [11, 12], [1, 12]]],
                "dt_scores": [0.87],
            })
        ])
        recognizer = FakeModel([FakeResult({"rec_text": " text\t", "rec_score": 0.61})])
        factory_calls = []

        def create_models(*, device):
            factory_calls.append(device)
            return detector, recognizer

        adapter = PaddleOcrV6Adapter(device="cpu", model_factory=create_models)

        polygons, scores = adapter.detect("region.png")
        text, recognition_score = adapter.recognize("line.png")

        self.assertEqual(factory_calls, ["cpu"])
        self.assertEqual(polygons, [[[1.0, 2.0], [11.0, 2.0], [11.0, 12.0], [1.0, 12.0]]])
        self.assertEqual(scores, [0.87])
        self.assertEqual((text, recognition_score), (" text\t", 0.61))
        self.assertEqual(detector.inputs, [("region.png", 1)])
        self.assertEqual(recognizer.inputs, [("line.png", 1)])

    def test_rejects_mismatched_detection_polygon_and_score_counts(self):
        detector = FakeModel([FakeResult({"dt_polys": [], "dt_scores": [0.5]})])
        recognizer = FakeModel([])
        adapter = PaddleOcrV6Adapter(
            model_factory=lambda *, device: (detector, recognizer)
        )

        with self.assertRaises(ValueError):
            adapter.detect("region.png")


if __name__ == "__main__":
    unittest.main()

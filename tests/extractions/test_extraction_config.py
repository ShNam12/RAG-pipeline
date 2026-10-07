"""Tests for YAML-backed extraction defaults."""

import unittest

from rag1.extractions.config import load_extraction_config
from rag1.extractions.layouts.paddle import PADDLE_LAYOUT_MODEL_NAME
from rag1.extractions.layouts.pdf_renderer import DEFAULT_RENDER_DPI
from rag1.extractions.ocr_text.paddle import TEXT_DETECTION_MODEL
from rag1.extractions.ocr_text.paddle_vl import PADDLE_OCR_VL_PIPELINE_VERSION


class ExtractionConfigTests(unittest.TestCase):
    def test_layout_defaults_come_from_yaml(self) -> None:
        config = load_extraction_config("layouts")

        self.assertEqual(PADDLE_LAYOUT_MODEL_NAME, config["model"]["name"])
        self.assertEqual(DEFAULT_RENDER_DPI, config["render"]["dpi"])

    def test_ocr_model_defaults_come_from_yaml(self) -> None:
        config = load_extraction_config("ocr_text")

        self.assertEqual(
            TEXT_DETECTION_MODEL,
            config["models"]["paddleocr_v6"]["detection_model"],
        )
        self.assertEqual(
            PADDLE_OCR_VL_PIPELINE_VERSION,
            config["models"]["paddleocr_vl"]["pipeline_version"],
        )

    def test_rejects_unknown_configuration_name(self) -> None:
        with self.assertRaises(ValueError):
            load_extraction_config("unknown")


if __name__ == "__main__":
    unittest.main()

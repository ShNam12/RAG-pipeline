"""OCR command options for layout and direct image inputs."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag1 import _build_parser, main


class OcrCliTests(unittest.TestCase):
    def test_vietnamese_recognizer_is_default_model(self) -> None:
        arguments = _build_parser().parse_args([
            "ocr", "region/page-0001.png", "--direct"
        ])

        self.assertEqual(arguments.model, "pp-ocrv6-medium-rec-vietnamese")

    def test_direct_option_accepts_page_image_and_model(self) -> None:
        arguments = _build_parser().parse_args([
            "ocr", "region/page-0001.png", "--direct", "--model", "paddleocr-vl"
        ])

        self.assertEqual(arguments.layout_json, Path("region/page-0001.png"))
        self.assertTrue(arguments.direct)
        self.assertEqual(arguments.model, "paddleocr-vl")

    def test_vietnamese_recognizer_option_is_forwarded_to_pipeline(self) -> None:
        model = "pp-ocrv6-medium-rec-vietnamese"
        arguments = _build_parser().parse_args(["ocr", "region", "--model", model])
        self.assertEqual(arguments.model, model)

        with patch(
            "rag1.extractions.ocr_text.pipeline.run_ocr",
            side_effect=ValueError("stop before inference"),
        ) as run_ocr:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["ocr", "region", "--model", model])

        self.assertEqual(run_ocr.call_args.kwargs["model"], model)

    def test_direct_option_is_forwarded_to_pipeline(self) -> None:
        with patch(
            "rag1.extractions.ocr_text.pipeline.run_ocr",
            side_effect=ValueError("stop before inference"),
        ) as run_ocr:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["ocr", "region/page-0001.png", "--direct"])

        run_ocr.assert_called_once_with(
            Path("region/page-0001.png"),
            direct=True,
            image_dir=None,
            manifest_path=None,
            output_dir=Path("data/ocr"),
            device="cpu",
            model="pp-ocrv6-medium-rec-vietnamese",
        )

    def test_ocr_text_defaults_to_markdown_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ocr_json = Path(directory) / "ocr.json"
            ocr_json.write_text("{}", encoding="utf-8")
            with patch(
                "rag1.extractions.ocr_text.contracts.OcrDocument.model_validate_json",
                return_value=object(),
            ):
                with patch(
                    "rag1.extractions.ocr_text.text_output.write_ocr_text"
                ) as write_text:
                    main(["ocr-text", str(ocr_json)])

        self.assertEqual(write_text.call_args.args[1], ocr_json.with_name("ocr.md"))


if __name__ == "__main__":
    unittest.main()

"""Page-level PaddleOCR-VL keeps layout regions and table OCR separate."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rag1.chunking import sections_from_ocr
from rag1.extractions.ocr_text.adapters import RecognizedBlock
from rag1.extractions.ocr_text.contracts import OcrDocument
from rag1.extractions.ocr_text.pipeline import run_ocr
from rag1.extractions.ocr_text.paddle_vl import PaddleOcrVlAdapter


def _proposal(region_id, kind, bbox):
    return {
        "id": region_id,
        "kind": kind,
        "role": "body",
        "raw_label": kind,
        "location": {
            "type": "visual",
            "page_number": 1,
            "bbox": bbox,
            "page_width": 100,
            "page_height": 100,
            "coordinate_origin": "top_left",
        },
    }


class PageAdapter:
    def __init__(self):
        self.paths = []
        self.table_mask_pixel = None

    def initialize(self):
        pass

    def parse_region(self, image_path):
        path = Path(image_path)
        self.paths.append(path)
        if path.name.startswith("region-"):
            return [RecognizedBlock([1, 1, 5, 5], "table", "Table value")]
        with Image.open(path) as image:
            self.table_mask_pixel = image.getpixel((75, 5))
        return [
            RecognizedBlock([2, 2, 8, 8], "text", "Text value"),
            RecognizedBlock([22, 2, 28, 8], "image", "Picture value"),
            RecognizedBlock([42, 2, 48, 8], "text", "Other value"),
            RecognizedBlock([72, 2, 78, 8], "table", "Masked table value"),
            RecognizedBlock([92, 2, 98, 8], "text", "Unassigned value"),
        ]


class PageVlPipelineTests(unittest.TestCase):
    def _layout_with_page(self, root):
        layout_path = root / "layout.json"
        layout_path.write_text(json.dumps({
            "schema_version": "1.0",
            "source": "report.pdf",
            "source_format": "pdf",
            "status": "complete",
            "regions": [
                _proposal("text", "text", [0, 0, 20, 20]),
                _proposal("picture", "picture", [20, 0, 40, 20]),
                _proposal("other", "other", [40, 0, 60, 20]),
                _proposal("table", "table", [70, 0, 90, 20]),
            ],
            "errors": [],
        }), encoding="utf-8")
        pages = root / "pages"
        pages.mkdir()
        Image.new("RGB", (100, 100), "black").save(pages / "page-0001.png")
        return layout_path

    def test_one_page_call_covers_non_table_kinds_and_preserves_table_ocr(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path = self._layout_with_page(root)
            adapter = PageAdapter()

            output_path = run_ocr(
                layout_path, output_dir=root / "ocr", model="paddleocr-vl", adapter=adapter
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            regions = {region.proposal.id: region for region in result.regions}
            self.assertEqual(len(adapter.paths), 2)
            self.assertEqual(adapter.table_mask_pixel, (255, 255, 255))
            self.assertEqual(regions["text"].text, "Text value")
            self.assertEqual(regions["picture"].text, "Picture value")
            self.assertEqual(regions["other"].text, "Other value")
            self.assertEqual(regions["table"].blocks[0].content, "Table value")
            self.assertNotIn("Masked table value", result.text)
            self.assertEqual(
                regions["p0001-ocr-unassigned"].text, "Unassigned value"
            )
            self.assertIn("Picture value", result.text)
            self.assertIn("Other value", result.text)
            self.assertIn("Unassigned value", result.text)
            section_text = "\n".join(section["text"] for section in sections_from_ocr(result))
            self.assertIn("Picture value", section_text)
            self.assertIn("Other value", section_text)
            self.assertIn("Unassigned value", section_text)

    def test_page_failure_keeps_table_result_and_marks_non_tables_failed(self):
        class FailingPageAdapter(PageAdapter):
            def parse_region(self, image_path):
                if not Path(image_path).name.startswith("region-"):
                    raise RuntimeError("page inference failed")
                return super().parse_region(image_path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path = self._layout_with_page(root)
            adapter = FailingPageAdapter()

            output_path = run_ocr(
                layout_path, output_dir=root / "ocr", model="paddleocr-vl", adapter=adapter
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status.value, "partial")
            self.assertEqual([region.status for region in result.regions],
                             ["failed", "failed", "failed", "complete"])
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(result.errors[0].stage, "page_detection_or_recognition")
            self.assertEqual(result.regions[-1].blocks[0].content, "Table value")

    def test_native_markdown_uses_an_additional_unmasked_page_prediction(self):
        class Result:
            def __init__(self, block, markdown):
                self.json = {"res": {"parsing_res_list": [block]}}
                self.markdown = markdown

        class Model:
            def __init__(self):
                self.paths = []

            def predict(self, input):
                path = Path(input)
                self.paths.append(path)
                if path.name.startswith("region-"):
                    block = {"block_bbox": [1, 1, 5, 5], "block_label": "table", "block_content": "Cell"}
                else:
                    block = {"block_bbox": [2, 2, 8, 8], "block_label": "text", "block_content": "Plain text"}
                markdown = {
                    "markdown_texts": "# Native heading\n\n| A |\n| - |\n| Cell |\n\n![](imgs/figure.png)",
                    "markdown_images": {"imgs/figure.png": Image.new("RGB", (2, 2), "red")},
                }
                return [Result(block, markdown)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path = self._layout_with_page(root)
            model = Model()
            adapter = PaddleOcrVlAdapter(model_factory=lambda *, device: model)
            output_path = run_ocr(
                layout_path, output_dir=root / "ocr", model="paddleocr-vl", adapter=adapter
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status.value, "complete")
            self.assertEqual(len(model.paths), 3)
            self.assertEqual(model.paths[-1], root / "pages" / "page-0001.png")
            self.assertIn("Plain text", output_path.with_name("ocr.md").read_text(encoding="utf-8"))
            formatted = output_path.with_name("ocr.formatted.md").read_text(encoding="utf-8")
            self.assertIn("# Native heading", formatted)
            self.assertIn("| Cell |", formatted)
            self.assertIn("![](formatted-images/page-0001/image-0001.png)", formatted)
            self.assertTrue((output_path.parent / "formatted-images/page-0001/image-0001.png").is_file())

    def test_direct_mode_reuses_full_page_prediction_for_markdown(self):
        class Model:
            def __init__(self):
                self.paths = []

            def predict(self, input):
                self.paths.append(input)
                result = type("Result", (), {})()
                result.json = {"res": {"parsing_res_list": [
                    {"block_bbox": [2, 2, 8, 8], "block_label": "text", "block_content": "Plain text"}
                ]}}
                result.markdown = {"markdown_texts": "# Native heading", "markdown_images": {}}
                return [result]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page_path = root / "page-0001.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            model = Model()
            adapter = PaddleOcrVlAdapter(model_factory=lambda *, device: model)
            output_path = run_ocr(
                page_path, direct=True, output_dir=root / "ocr",
                model="paddleocr-vl", adapter=adapter
            )
            self.assertEqual(model.paths, [str(page_path)])
            self.assertEqual(
                output_path.with_name("ocr.formatted.md").read_text(encoding="utf-8"),
                "# Native heading\n",
            )

    def test_markdown_failure_preserves_structured_ocr_and_reports_partial(self):
        class Model:
            def predict(self, input):
                result = type("Result", (), {})()
                result.json = {"res": {"parsing_res_list": [
                    {"block_bbox": [2, 2, 8, 8], "block_label": "text", "block_content": "Plain text"}
                ]}}
                result.markdown = {"markdown_texts": "# Heading", "markdown_images": {
                    "figure.png": object()
                }}
                return [result]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page_path = root / "page-0001.png"
            Image.new("RGB", (100, 100), "white").save(page_path)
            adapter = PaddleOcrVlAdapter(model_factory=lambda *, device: Model())
            output_path = run_ocr(
                page_path, direct=True, output_dir=root / "ocr",
                model="paddleocr-vl", adapter=adapter
            )
            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status.value, "partial")
            self.assertEqual(result.regions[0].text, "Plain text")
            self.assertEqual(result.errors[0].stage, "formatted_markdown")
            self.assertFalse(output_path.with_name("ocr.formatted.md").exists())


if __name__ == "__main__":
    unittest.main()

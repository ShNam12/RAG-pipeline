"""Standalone OCR pipeline contracts using an in-memory model adapter."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rag1.extractions.ocr_text.adapters import DetectedText, RecognizedBlock
from rag1.extractions.ocr_text.contracts import OcrDocument
from rag1.extractions.ocr_text.pipeline import _rectify_polygon, run_ocr
from rag1.extractions.ocr_text.text_output import render_ocr_text


def proposal(region_id, kind):
    return {
        "id": region_id,
        "kind": kind,
        "role": "body",
        "raw_label": kind,
        "confidence": 0.94,
        "location": {
            "type": "visual",
            "page_number": 1,
            "bbox": [10.25, 20.75, 90.5, 120.25],
            "page_width": 100,
            "page_height": 140,
            "coordinate_origin": "top_left",
        },
    }


class FakeAdapter:
    def __init__(self):
        self.initialized = False
        self.detected = []
        self.recognized = []

    def initialize(self):
        self.initialized = True

    def detect(self, image_path):
        self.detected.append(Path(image_path))
        return [DetectedText(
            polygon=[[1, 2], [11, 2], [11, 12], [1, 12]],
            confidence=0.87,
        )]

    def recognize(self, image_path):
        self.recognized.append(Path(image_path))
        return ("  Doanh thu qu\u00fd III\t1.250.000\n", 0.61)


class FakeVlAdapter:
    def __init__(self):
        self.initialized = False
        self.parsed = []

    def initialize(self):
        self.initialized = True

    def parse_region(self, image_path):
        self.parsed.append(Path(image_path))
        if "region-00001" in image_path:
            return [
                RecognizedBlock(
                    bbox=[1, 2, 11, 12],
                    label="text",
                    content="Doanh thu Q3",
                )
            ]
        return [
            RecognizedBlock(
                bbox=[3, 4, 30, 40],
                label="table",
                content="| Chi tieu | VND |\n|---|---|\n| Doanh thu | 100 |",
            )
        ]


class OcrPipelineTests(unittest.TestCase):
    def create_layout_and_page(self, root, *, images_in_layout_dir=False):
        layout_path = root / "layout.json"
        layout_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "source": "quarterly-report.pdf",
                    "source_format": "pdf",
                    "status": "complete",
                    "regions": [
                        proposal("p0001-r0001", "text"),
                        proposal("p0001-r0002", "table"),
                    ],
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        pages = root if images_in_layout_dir else root / "pages"
        pages.mkdir(exist_ok=True)
        Image.new("RGB", (100, 140), "white").save(pages / "page-0001.png")
        Image.new("RGB", (100, 140), "red").save(pages / "page-0001_bbox.png")
        return layout_path, pages

    def test_runs_detection_crop_recognition_and_saves_enriched_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, pages = self.create_layout_and_page(root)
            adapter = FakeAdapter()

            output_path = run_ocr(
                layout_path,
                output_dir=root / "ocr",
                model="paddleocr-v6",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            raw_result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertNotIn("blocks", raw_result["regions"][0])
            self.assertNotIn("model", raw_result)
            self.assertEqual(raw_result["text"], result.text)
            self.assertEqual([region.proposal.id for region in result.regions],
                             ["p0001-r0001", "p0001-r0002"])
            self.assertEqual(len(adapter.detected), 2)
            self.assertEqual(len(adapter.recognized), 2)
            text_region, table_region = result.regions
            self.assertIsNone(result.dpi)
            self.assertIsNone(result.pages[0].rotation_degrees)
            self.assertIsNone(result.pages[0].pixel_width)
            self.assertIsNone(result.pages[0].region_ids)
            self.assertEqual(text_region.text, "  Doanh thu qu\u00fd III\t1.250.000\n")
            self.assertEqual(text_region.lines[0].detection_confidence, 0.87)
            self.assertEqual(text_region.lines[0].recognition_confidence, 0.61)
            self.assertEqual(text_region.crop.proposal_bbox,
                             [10.25, 20.75, 90.5, 120.25])
            self.assertEqual(text_region.crop.page_bbox, [10, 20, 91, 121])
            self.assertTrue((output_path.parent / text_region.crop.path).is_file())
            self.assertTrue((output_path.parent / text_region.lines[0].crop_path).is_file())
            self.assertEqual(text_region.lines[0].region_quad,
                             [[1, 2], [11, 2], [11, 12], [1, 12]])
            self.assertEqual(text_region.lines[0].page_quad,
                             [[11, 22], [21, 22], [21, 32], [11, 32]])
            self.assertIsNone(table_region.text)
            self.assertEqual(table_region.table.structure_status, "pending")
            self.assertEqual(len(table_region.lines), 1)

    def test_vietnamese_model_uses_paddle_recognizer_and_records_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, _ = self.create_layout_and_page(root)
            adapter = FakeAdapter()
            with patch(
                "rag1.extractions.ocr_text.pipeline.TEXT_RECOGNITION_MODEL_DIR",
                str(root),
            ):
                with patch(
                    "rag1.extractions.ocr_text.pipeline.PaddleOcrV6Adapter",
                    return_value=adapter,
                ) as adapter_factory:
                    output_path = run_ocr(
                        layout_path,
                        output_dir=root / "ocr",
                        model="pp-ocrv6-medium-rec-vietnamese",
                    )

            adapter_factory.assert_called_once_with(device="cpu")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "pp-ocrv6-medium-rec-vietnamese")
            self.assertNotIn("blocks", result["regions"][0])
            self.assertEqual(len(adapter.recognized), 2)

    def test_vietnamese_model_requires_downloaded_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_dir = Path(temporary_directory) / "missing-model"
            with patch(
                "rag1.extractions.ocr_text.pipeline.TEXT_RECOGNITION_MODEL_DIR",
                str(missing_dir),
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "model directory does not exist"
                ):
                    run_ocr("unused-layout.json", model="pp-ocrv6-medium-rec-vietnamese")

    def test_vl_blocks_preserve_layout_geometry_and_table_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, _ = self.create_layout_and_page(root)
            adapter = FakeVlAdapter()

            output_path = run_ocr(
                layout_path,
                output_dir=root / "ocr",
                model="paddleocr-vl",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            text_region, table_region = result.regions
            self.assertEqual(result.model, "paddleocr-vl")
            self.assertTrue(adapter.initialized)
            self.assertEqual(len(adapter.parsed), 2)
            self.assertEqual(text_region.lines, [])
            self.assertEqual(text_region.text, "Doanh thu Q3")
            self.assertEqual(text_region.blocks[0].region_bbox, [1, 2, 11, 12])
            self.assertEqual(text_region.blocks[0].page_bbox, [11, 22, 21, 32])
            self.assertIsNone(table_region.text)
            self.assertEqual(table_region.blocks[0].label, "table")
            self.assertIn("Doanh thu | 100", render_ocr_text(result))

    def test_vl_empty_result_keeps_region_and_crop(self):
        class EmptyVlAdapter(FakeVlAdapter):
            def parse_region(self, image_path):
                self.parsed.append(Path(image_path))
                return []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, _ = self.create_layout_and_page(root)
            output_path = run_ocr(
                layout_path,
                output_dir=root / "ocr",
                model="paddleocr-vl",
                adapter=EmptyVlAdapter(),
            )
            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual([region.status for region in result.regions], ["empty", "empty"])
            self.assertTrue(all(region.crop.path for region in result.regions))

    def test_direct_vl_uses_raw_page_image_without_layout_json(self):
        class DirectVlAdapter(FakeVlAdapter):
            def parse_region(self, image_path):
                self.parsed.append(Path(image_path))
                return [RecognizedBlock(
                    bbox=[2, 3, 30, 20],
                    label="table",
                    content="| Revenue | 100 |",
                )]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page_path = root / "page-0003.png"
            Image.new("RGB", (100, 140), "white").save(page_path)
            adapter = DirectVlAdapter()

            output_path = run_ocr(
                page_path,
                direct=True,
                output_dir=root / "ocr",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.model, "paddleocr-vl")
            self.assertEqual(adapter.parsed, [page_path])
            self.assertEqual(result.input_mode, "direct")
            self.assertEqual(result.source_format.value, "image")
            self.assertIsNone(result.upstream_schema_version)
            self.assertIsNone(result.upstream_status)
            self.assertEqual(result.pages[0].page_number, 3)
            self.assertEqual(result.pages[0].image, str(page_path.resolve()))
            self.assertEqual(result.regions[0].proposal.location.bbox, [0, 0, 100, 140])
            self.assertEqual(result.regions[0].blocks[0].page_bbox, [2, 3, 30, 20])
            self.assertIn("Revenue", (output_path.parent / "ocr.md").read_text(encoding="utf-8"))

    def test_direct_v6_runs_on_each_page_in_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pages = root / "region"
            pages.mkdir()
            for number in (1, 2):
                Image.new("RGB", (100, 140), "white").save(
                    pages / f"page-{number:04d}.png"
                )
            Image.new("RGB", (100, 140), "red").save(pages / "page-0001_bbox.png")
            adapter = FakeAdapter()

            output_path = run_ocr(
                pages,
                direct=True,
                output_dir=root / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual([page.page_number for page in result.pages], [1, 2])
            self.assertEqual(adapter.detected, [pages / "page-0001.png", pages / "page-0002.png"])
            self.assertEqual(len(result.regions), 2)
            self.assertTrue(all(region.lines for region in result.regions))
            self.assertEqual(result.status.value, "complete")

    def test_direct_input_rejects_missing_images_and_layout_options(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page_path = root / "page-0001.png"
            Image.new("RGB", (100, 140), "white").save(page_path)
            with self.assertRaises(ValueError):
                run_ocr(page_path, direct=True, manifest_path=root / "manifest.json")
            with self.assertRaises(ValueError):
                run_ocr(page_path, direct=True, image_dir=root)
            with self.assertRaises(FileNotFoundError):
                run_ocr(root / "empty", direct=True)

    def test_direct_directory_records_unreadable_page_and_processes_other_pages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pages = root / "region"
            pages.mkdir()
            Image.new("RGB", (100, 140), "white").save(pages / "page-0001.png")
            (pages / "page-0002.png").write_bytes(b"not an image")
            adapter = FakeAdapter()

            output_path = run_ocr(
                pages,
                direct=True,
                output_dir=root / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status.value, "partial")
            self.assertEqual(len(result.regions), 1)
            self.assertEqual([page.page_number for page in result.pages], [1, 2])
            self.assertEqual(result.page_failures[0].page_number, 2)
            self.assertEqual(result.page_failures[0].exception_type, "UnidentifiedImageError")
            self.assertEqual(adapter.detected, [pages / "page-0001.png"])

    def test_text_lines_are_sorted_by_position_then_detector_index(self):
        class UnorderedAdapter(FakeAdapter):
            def detect(self, image_path):
                detections = [
                    (30, 20),
                    (20, 5),
                    (5, 5),
                    (20, 5),
                ]
                return [
                    DetectedText(
                        polygon=[
                            [x, y], [x + 8, y], [x + 8, y + 4], [x, y + 4]
                        ],
                        confidence=0.87,
                    )
                    for x, y in detections
                ]

            def recognize(self, image_path):
                detector_index = int(Path(image_path).stem.rsplit("-", 1)[1])
                return ["lower ", " top ", "\tleft", " middle "][detector_index], 0.61

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, _ = self.create_layout_and_page(root)
            output_path = run_ocr(
                layout_path,
                output_dir=root / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=UnorderedAdapter(),
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))

        text_region = result.regions[0]
        self.assertEqual(text_region.text, "\tleft\n top \n middle \nlower ")
        self.assertEqual(
            [line.detector_index for line in text_region.lines],
            [2, 1, 3, 0],
        )
        self.assertEqual(
            [line.text for line in text_region.lines],
            ["\tleft", " top ", " middle ", "lower "],
        )

    def test_rectification_maps_slanted_quad_to_a_rectangle(self):
        image = Image.new("RGB", (40, 40), "white")
        polygon = [[5, 8], [25, 5], [27, 15], [7, 18]]

        rectified = _rectify_polygon(image, polygon)

        self.assertEqual(rectified.size, (20, 10))

    def test_degenerate_detection_is_a_line_error_and_does_not_discard_region(self):
        class DegenerateAdapter(FakeAdapter):
            def detect(self, image_path):
                self.detected.append(Path(image_path))
                return [DetectedText(
                    polygon=[[1, 2], [5, 2], [9, 2], [1, 2]],
                    confidence=0.87,
                )]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, _ = self.create_layout_and_page(root)
            output_path = run_ocr(
                layout_path,
                output_dir=root / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=DegenerateAdapter(),
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result.regions[0].status, "complete")
        self.assertEqual(result.regions[0].lines, [])
        self.assertEqual(result.errors[0].stage, "line_1_rectification")

    def test_missing_page_image_preserves_failed_proposal_and_table_marker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, pages = self.create_layout_and_page(root)
            (pages / "page-0001.png").unlink()
            adapter = FakeAdapter()

            output_path = run_ocr(
                layout_path,
                image_dir=pages,
                output_dir=root / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status.value, "failed")
            self.assertEqual([region.proposal.kind.value for region in result.regions],
                             ["text", "table"])
            self.assertEqual([region.status for region in result.regions], ["failed", "failed"])
            self.assertIsNone(result.regions[1].crop.path)
            self.assertEqual(result.regions[1].table.structure_status, "pending")
            self.assertEqual(result.errors, [])
            self.assertEqual(
                [(failure.page_number, failure.exception_type) for failure in result.page_failures],
                [(1, "FileNotFoundError")],
            )
            self.assertFalse(adapter.initialized)

    def test_manifest_metadata_is_retained_and_references_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            layout_path, pages = self.create_layout_and_page(root)
            manifest_path = root / "manifest.json"
            manifest = {
                "manifest_version": "1.0",
                "source": "quarterly-report.pdf",
                "dpi": 200,
                "pages": [
                    {
                        "page_number": 1,
                        "rotation_degrees": 90,
                        "pixel_width": 100,
                        "pixel_height": 140,
                        "image": "pages/page-0001.png",
                        "bbox_image": "bbox/page-0001_bbox.png",
                        "region_ids": ["p0001-r0001", "p0001-r0002"],
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            adapter = FakeAdapter()

            output_path = run_ocr(
                layout_path,
                manifest_path=manifest_path,
                output_dir=root / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.dpi, 200)
            self.assertEqual(result.pages[0].rotation_degrees, 90)
            self.assertEqual((result.pages[0].pixel_width, result.pages[0].pixel_height),
                             (100, 140))
            self.assertEqual(result.pages[0].region_ids,
                             ["p0001-r0001", "p0001-r0002"])

            manifest["source"] = "another-report.pdf"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_ocr(
                    layout_path,
                    manifest_path=manifest_path,
                    output_dir=root / "ocr-wrong-source",
                    model="pp-ocrv6-medium-rec-vietnamese",
                    adapter=FakeAdapter(),
                )

            manifest["source"] = "quarterly-report.pdf"
            manifest["pages"][0]["region_ids"] = ["unknown-region"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_ocr(
                    layout_path,
                    manifest_path=manifest_path,
                    output_dir=root / "ocr-invalid",
                    model="pp-ocrv6-medium-rec-vietnamese",
                    adapter=FakeAdapter(),
                )

    def test_mismatched_and_unreadable_images_are_page_failures_without_rescaling(self):
        for image_kind in ("mismatched", "unreadable"):
            with self.subTest(image_kind=image_kind), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                layout_path, pages = self.create_layout_and_page(root)
                page_path = pages / "page-0001.png"
                if image_kind == "mismatched":
                    Image.new("RGB", (99, 140), "white").save(page_path)
                    expected_exception = "ValueError"
                else:
                    page_path.write_bytes(b"not an image")
                    expected_exception = "UnidentifiedImageError"
                adapter = FakeAdapter()

                output_path = run_ocr(
                    layout_path,
                    output_dir=root / "ocr",
                    model="pp-ocrv6-medium-rec-vietnamese",
                    adapter=adapter,
                )

                result = OcrDocument.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                )
                self.assertEqual(result.status.value, "failed")
                self.assertEqual(result.errors, [])
                self.assertEqual(len(result.page_failures), 1)
                self.assertEqual(result.page_failures[0].exception_type, expected_exception)
                self.assertEqual(result.regions[0].proposal.location.bbox,
                                 [10.25, 20.75, 90.5, 120.25])
                self.assertTrue(all(region.status == "failed" for region in result.regions))
                self.assertFalse(adapter.initialized)

    def test_region_artifact_directory_is_accepted_with_flat_page_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            region_directory = Path(temporary_directory) / "region"
            region_directory.mkdir()
            self.create_layout_and_page(region_directory, images_in_layout_dir=True)
            (region_directory / "manifest.json").write_text(
                json.dumps({
                    "source": "quarterly-report.pdf",
                    "dpi": 200,
                    "pages": [{
                        "page_number": 1,
                        "rotation_degrees": 90,
                        "pixel_width": 100,
                        "pixel_height": 140,
                        "region_ids": ["p0001-r0001", "p0001-r0002"],
                    }],
                }),
                encoding="utf-8",
            )
            adapter = FakeAdapter()

            output_path = run_ocr(
                region_directory,
                output_dir=Path(temporary_directory) / "ocr",
                model="pp-ocrv6-medium-rec-vietnamese",
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.dpi, 200)
            self.assertEqual(result.pages[0].rotation_degrees, 90)
            self.assertEqual(len(adapter.detected), 2)


if __name__ == "__main__":
    unittest.main()

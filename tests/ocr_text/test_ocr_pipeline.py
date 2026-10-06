"""Standalone OCR pipeline contracts using an in-memory model adapter."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rag1.extractions.ocr_text.contracts import OcrDocument
from rag1.extractions.ocr_text.pipeline import run_ocr


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
        return ([[[1, 2], [11, 2], [11, 12], [1, 12]]], [0.87])

    def recognize(self, image_path):
        self.recognized.append(Path(image_path))
        return ("  Doanh thu qu\u00fd III\t1.250.000\n", 0.61)


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
                adapter=adapter,
            )

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
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
            self.assertIsNone(table_region.text)
            self.assertEqual(table_region.table.structure_status, "pending")
            self.assertEqual(len(table_region.lines), 1)

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

            output_path = run_ocr(region_directory, output_dir=Path(temporary_directory) / "ocr",
                                  adapter=adapter)

            result = OcrDocument.model_validate_json(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result.dpi, 200)
            self.assertEqual(result.pages[0].rotation_degrees, 90)
            self.assertEqual(len(adapter.detected), 2)


if __name__ == "__main__":
    unittest.main()

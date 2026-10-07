"""Tests for the PDF layout pipeline artifact boundary."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from rag1.extractions.layouts.pipeline import run_pdf_layout


class FakeBitmap:
    def to_pil(self) -> Image.Image:
        return Image.new("RGB", (100, 120), "white")

    def close(self) -> None:
        pass


class FakePdfPage:
    def get_rotation(self) -> int:
        return 0

    def render(self, *, scale: float) -> FakeBitmap:
        return FakeBitmap()

    def close(self) -> None:
        pass


class FakePdfDocument:
    def __init__(self) -> None:
        self.pages = [FakePdfPage()]

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> FakePdfPage:
        return self.pages[index]

    def close(self) -> None:
        pass


class FakePrediction:
    json = {
        "res": {
            "boxes": [
                {
                    "label": "table",
                    "score": 0.9,
                    "coordinate": [10, 20, 40, 50],
                }
            ]
        }
    }


class FakeLayoutModel:
    def predict(self, input: object, *, batch_size: int) -> list[FakePrediction]:
        return [FakePrediction()]


class LayoutPipelineTests(unittest.TestCase):
    def test_writes_images_manifest_and_layout_for_one_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.pdf"
            source.write_bytes(b"pdf")
            output_root = root / "extraction"
            pdfium = SimpleNamespace(
                PdfDocument=lambda source_path: FakePdfDocument()
            )

            with patch.dict(sys.modules, {"pypdfium2": pdfium}):
                artifacts = run_pdf_layout(
                    source,
                    output_dir=output_root,
                    image_dir=Path("source_pages"),
                    model_factory=lambda **kwargs: FakeLayoutModel(),
                )

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(artifacts.status, "complete")
            self.assertTrue(artifacts.layout_path.is_file())
            self.assertTrue(artifacts.manifest_path.is_file())
            self.assertTrue(artifacts.page_images[0].is_file())
            self.assertTrue(artifacts.bbox_images[0].is_file())
            self.assertEqual(artifacts.page_images[0].parent.name, "source_pages")
            self.assertEqual(artifacts.bbox_images[0].parent.name, "bbox")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["page_count"], 1)
            self.assertEqual(manifest["pages"][0]["image"].split("/")[0], "source_pages")
            self.assertEqual(manifest["pages"][0]["bbox_image"].split("/")[0], "bbox")
            self.assertEqual(manifest["pages"][0]["region_ids"], ["p0001-r0001"])

    def test_model_initialization_failure_still_writes_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.pdf"
            source.write_bytes(b"pdf")
            pdfium = SimpleNamespace(
                PdfDocument=lambda source_path: FakePdfDocument()
            )

            def fail_to_create_model(**kwargs: object) -> FakeLayoutModel:
                raise RuntimeError("model weights are unavailable")

            with patch.dict(sys.modules, {"pypdfium2": pdfium}):
                artifacts = run_pdf_layout(
                    source,
                    output_dir=root / "extraction",
                    model_factory=fail_to_create_model,
                )

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(artifacts.status, "failed")
            self.assertTrue(artifacts.page_images[0].is_file())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["errors"][0]["page_number"], 1)


if __name__ == "__main__":
    unittest.main()

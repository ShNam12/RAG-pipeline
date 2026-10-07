"""Tests for PDF rendering and bbox debug images."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from rag1.extractions.layouts.contracts import (
    Region,
    RegionKind,
    RegionRole,
    VisualLocation,
)
from rag1.extractions.layouts.pdf_renderer import (
    DEFAULT_RENDER_DPI,
    PDFPageRenderer,
    render_pdf_layout_debug,
)


class FakeBitmap:
    def to_pil(self) -> Image.Image:
        return Image.new("RGB", (100, 120), "white")

    def close(self) -> None:
        pass


class FakePdfPage:
    def __init__(self) -> None:
        self.scales: list[float] = []

    def get_rotation(self) -> int:
        return 90

    def render(self, *, scale: float) -> FakeBitmap:
        self.scales.append(scale)
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


class PDFPageRendererTests(unittest.TestCase):
    def test_renders_with_configured_dpi_and_records_page_metadata(self) -> None:
        self.assertEqual(DEFAULT_RENDER_DPI, 200)
        fake_document = FakePdfDocument()
        pdfium = SimpleNamespace(PdfDocument=lambda source: fake_document)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.pdf"
            renderer = PDFPageRenderer(dpi=300, output_dir=Path(directory) / "pages")

            with patch.dict(sys.modules, {"pypdfium2": pdfium}):
                pages = renderer.render(source)
                repeated_pages = renderer.render(source)

            page = pages[0]
            image_path = Path(page.image)
            document_dir = image_path.parent.parent
            manifest_path = document_dir / "render-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertAlmostEqual(fake_document.pages[0].scales[0], 300 / 72)
            self.assertEqual((page.page_width, page.page_height), (100, 120))
            self.assertEqual(image_path.name, "page-0001.png")
            self.assertEqual(image_path.parent.name, "pages")
            self.assertEqual(Path(repeated_pages[0].image), image_path)
            self.assertTrue(image_path.is_file())
            self.assertEqual(manifest["dpi"], 300)
            self.assertEqual(manifest["pages"][0]["rotation_degrees"], 90)
            self.assertEqual(manifest["pages"][0]["pixel_width"], 100)
            self.assertEqual(manifest["pages"][0]["pixel_height"], 120)
            self.assertEqual(manifest["pages"][0]["image_name"], "pages/page-0001.png")

    def test_exports_deterministic_images_with_region_boxes(self) -> None:
        fake_document = FakePdfDocument()
        pdfium = SimpleNamespace(PdfDocument=lambda source: fake_document)
        region = Region(
            id="p0001-r0001",
            kind=RegionKind.TABLE,
            role=RegionRole.BODY,
            raw_label="table",
            location=VisualLocation(
                page_number=1,
                bbox=[10, 20, 40, 50],
                page_width=100,
                page_height=120,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            renderer = PDFPageRenderer(output_dir=Path(directory) / "pages")
            with patch.dict(sys.modules, {"pypdfium2": pdfium}):
                pages = renderer.render(Path(directory) / "report.pdf")

            overlays = renderer.export_bbox_overlays(pages, [region])
            manifest = json.loads(
                (Path(pages[0].image).parent.parent / "render-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            with Image.open(overlays[0]) as image:
                self.assertEqual(image.getpixel((40, 50)), (255, 48, 48))

            self.assertEqual(overlays[0].name, "page-0001_bbox.png")
            self.assertEqual(overlays[0].parent.name, "bbox")
            self.assertEqual(
                manifest["pages"][0]["image_name"],
                "pages/page-0001.png",
            )
            self.assertEqual(
                manifest["pages"][0]["bbox_image_name"],
                "bbox/page-0001_bbox.png",
            )
            self.assertEqual(manifest["pages"][0]["region_ids"], ["p0001-r0001"])

    def test_rejects_nonpositive_dpi_and_non_pdf_sources(self) -> None:
        with self.assertRaises(ValueError):
            PDFPageRenderer(dpi=0)

        renderer = PDFPageRenderer()
        with self.assertRaisesRegex(ValueError, "requires a .pdf source"):
            renderer.render("report.docx")
        with self.assertRaisesRegex(ValueError, "relative subdirectory"):
            PDFPageRenderer(image_dir="../outside")

    def test_debug_workflow_saves_layout_json_and_box_highlights(self) -> None:
        fake_document = FakePdfDocument()
        pdfium = SimpleNamespace(PdfDocument=lambda source: fake_document)
        model = FakeLayoutModel()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.pdf"
            with patch.dict(sys.modules, {"pypdfium2": pdfium}):
                layout_path = render_pdf_layout_debug(
                    source,
                    output_dir=Path(directory) / "debug",
                    model_factory=lambda **kwargs: model,
                )

            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            image_path = layout_path.parent / "bbox" / "page-0001_bbox.png"
            with Image.open(image_path) as image:
                self.assertEqual(image.getpixel((40, 50)), (255, 48, 48))

            self.assertEqual(len(layout["regions"]), 1)
            self.assertTrue(image_path.is_file())


if __name__ == "__main__":
    unittest.main()

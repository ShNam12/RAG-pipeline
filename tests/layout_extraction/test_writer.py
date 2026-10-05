"""Tests for document-scoped layout artifact writing."""

import json
import tempfile
import unittest
from pathlib import Path

from rag1.extractions.layouts.contracts import (
    DocumentFormat,
    DocumentStatus,
    LayoutDocument,
    Region,
    RegionKind,
    RegionRole,
    VisualLocation,
)
from rag1.extractions.layouts.paddle import RenderedPage
from rag1.extractions.layouts.writer import LayoutArtifactWriter


class LayoutArtifactWriterTests(unittest.TestCase):
    def test_writes_document_manifest_and_page_artifact_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "quarterly report.pdf"
            source.write_bytes(b"pdf")
            writer = LayoutArtifactWriter(root / "extraction")
            document_dir = writer.create_document_directory(source)
            image_path = document_dir / "pages" / "page-0001.png"
            bbox_path = document_dir / "bbox" / "page-0001_bbox.png"
            image_path.parent.mkdir(parents=True)
            bbox_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")
            bbox_path.write_bytes(b"png")
            document = LayoutDocument(
                source=str(source.resolve()),
                source_format=DocumentFormat.PDF,
                status=DocumentStatus.COMPLETE,
                regions=[
                    Region(
                        id="p0001-r0001",
                        kind=RegionKind.TABLE,
                        role=RegionRole.BODY,
                        raw_label="table",
                        location=VisualLocation(
                            page_number=1,
                            bbox=[10, 20, 90, 100],
                            page_width=100,
                            page_height=120,
                        ),
                    )
                ],
            )

            artifacts = writer.write(
                source=source,
                document_dir=document_dir,
                document=document,
                dpi=200,
                pages=[RenderedPage(1, image_path, 100, 120, 90)],
                bbox_images_by_page={1: bbox_path},
            )

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            saved_layout = json.loads(artifacts.layout_path.read_text(encoding="utf-8"))
            self.assertEqual(artifacts.output_dir, document_dir)
            self.assertEqual(artifacts.status, "complete")
            self.assertEqual(manifest["dpi"], 200)
            self.assertEqual(manifest["pages"][0]["rotation_degrees"], 90)
            self.assertEqual(manifest["pages"][0]["pixel_width"], 100)
            self.assertEqual(manifest["pages"][0]["image"], "pages/page-0001.png")
            self.assertEqual(
                manifest["pages"][0]["bbox_image"],
                "bbox/page-0001_bbox.png",
            )
            self.assertEqual(manifest["pages"][0]["region_ids"], ["p0001-r0001"])
            self.assertEqual(saved_layout["regions"][0]["id"], "p0001-r0001")

    def test_document_output_directory_is_stable_and_source_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = LayoutArtifactWriter(root / "extraction")
            first_source = root / "first.pdf"
            second_source = root / "second.pdf"

            first_output = writer.create_document_directory(first_source)
            repeated_output = writer.create_document_directory(first_source)
            second_output = writer.create_document_directory(second_source)

            self.assertEqual(first_output, repeated_output)
            self.assertNotEqual(first_output, second_output)
            self.assertTrue(first_output.is_dir())
            self.assertTrue(second_output.is_dir())


if __name__ == "__main__":
    unittest.main()

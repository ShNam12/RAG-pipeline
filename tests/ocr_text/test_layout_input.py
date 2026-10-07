"""Versioned layout JSON input contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from rag1.extractions.ocr_text.layout_input import LayoutInputError, load_layout_document


def valid_layout():
    return {
        "schema_version": "1.0",
        "source": "sample.pdf",
        "source_format": "pdf",
        "status": "complete",
        "regions": [
            {
                "id": "p0001-r0001",
                "kind": "text",
                "role": "body",
                "raw_label": "text",
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
        ],
        "errors": [],
    }


class LayoutInputTests(unittest.TestCase):
    def write_json(self, directory, value):
        path = Path(directory) / "layout.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_loads_version_one_layout_and_preserves_region_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self.write_json(temporary_directory, valid_layout())

            layout = load_layout_document(path)

            self.assertEqual(layout.schema_version, "1.0")
            self.assertEqual(layout.source, "sample.pdf")
            self.assertEqual([region.id for region in layout.regions], ["p0001-r0001"])
            self.assertEqual(layout.regions[0].location.bbox, [10.25, 20.75, 90.5, 120.25])

    def test_rejects_unsupported_schema_version(self):
        data = valid_layout()
        data["schema_version"] = "2.0"
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(LayoutInputError):
                load_layout_document(self.write_json(temporary_directory, data))

    def test_rejects_duplicate_region_ids(self):
        data = valid_layout()
        data["regions"].append(data["regions"][0].copy())
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(LayoutInputError):
                load_layout_document(self.write_json(temporary_directory, data))

    def test_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "layout.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(LayoutInputError):
                load_layout_document(path)


if __name__ == "__main__":
    unittest.main()

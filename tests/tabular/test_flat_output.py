"""Flat table export preserves values, spans, and source references."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag1 import main
from rag1.extractions.tabular.flat_output import flatten_tables, write_flat_tables


def _document() -> dict:
    return {
        "schema_version": "1.0",
        "source": "report.pdf",
        "status": "partial",
        "tables": [{
            "region_id": "p0001-r0001",
            "page_number": 1,
            "bbox": [10, 20, 500, 700],
            "status": "partial",
            "method": "slanet_ocr",
            "structure_score": 0.9,
            "column_count": 3,
            "warnings": ["check OCR"],
            "unassigned_ocr": [{"ref": "line-extra", "text": "unplaced"}],
            "rows": [
                [
                    {"row": 0, "column": 0, "rowspan": 2, "colspan": 1,
                     "text": "Ta\u0300i sa\u0309n", "ocr_refs": ["line-1"]},
                    {"row": 0, "column": 1, "rowspan": 1, "colspan": 2,
                     "text": "  451.893.195  ", "ocr_refs": ["line-2"]},
                ],
                [
                    {"row": 1, "column": 1, "rowspan": 1, "colspan": 1,
                     "text": "(1.234)", "ocr_refs": ["line-3"]},
                    {"row": 1, "column": 2, "rowspan": 1, "colspan": 1,
                     "text": "30/09/2025", "ocr_refs": ["line-4"]},
                ],
            ],
        }],
        "skipped": [],
    }


class FlatTableTests(unittest.TestCase):
    def test_html_header_becomes_keys_and_first_row_is_not_data(self) -> None:
        html = (
            '<!doctype html><html><body><section><h2>p0001-r0001 - page 1</h2>'
            '<p>Status: partial; method: ocr_geometry</p><table>'
            '<tr><td>col1</td><td>col2</td></tr>'
            '<tr><td>value 1</td><td>value 2</td></tr>'
            '<tr><td>value 3</td><td>value 3</td></tr>'
            '</table></section></body></html>'
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.html"
            source.write_text(html, encoding="utf-8")
            output = write_flat_tables(source)
            self.assertEqual(output, source.with_name("tables.from-html.json"))
            records = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(records, [
            {"col1": "value 1", "col2": "value 2"},
            {"col1": "value 3", "col2": "value 3"},
        ])

    def test_html_uses_generic_key_for_merged_header_coverage(self) -> None:
        html = (
            '<section><h2>p0001-r0001 - page 1</h2>'
            '<p>Status: partial; method: ocr_geometry</p><table>'
            '<tr><td colspan="2">Tài &amp; sản</td></tr>'
            '<tr><td>value 1</td><td>value 2</td></tr>'
            '</table></section>'
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.html"
            source.write_text(html, encoding="utf-8")
            records = json.loads(write_flat_tables(source).read_text(encoding="utf-8"))
        self.assertEqual(records, [{"Tài & sản": "value 1", "column_2": "value 2"}])

    def test_html_disambiguates_duplicate_and_empty_headers(self) -> None:
        html = (
            '<section><h2>p0001-r0001 - page 1</h2>'
            '<p>Status: complete; method: slanet_ocr</p><table>'
            '<tr><td>Value</td><td>Value</td><td></td></tr>'
            '<tr><td>1</td><td>2</td><td>3</td></tr>'
            '</table></section>'
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.html"
            source.write_text(html, encoding="utf-8")
            records = json.loads(write_flat_tables(source).read_text(encoding="utf-8"))
        self.assertEqual(records, [{"Value": "1", "Value__2": "2", "column_3": "3"}])

    def test_html_multiple_tables_include_source_identity(self) -> None:
        html = "".join(
            f'<section><h2>{table_id} - page {page}</h2>'
            '<p>Status: complete; method: slanet_ocr</p><table>'
            '<tr><td>Name</td><td>Amount</td></tr>'
            '<tr><td>A</td><td>1.234</td></tr></table></section>'
            for table_id, page in (("p0001-r0001", 1), ("p0002-r0001", 2))
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.html"
            source.write_text(html, encoding="utf-8")
            records = json.loads(write_flat_tables(source).read_text(encoding="utf-8"))
        self.assertEqual(records, [
            {"_table_id": "p0001-r0001", "_page_number": 1, "Name": "A", "Amount": "1.234"},
            {"_table_id": "p0002-r0001", "_page_number": 2, "Name": "A", "Amount": "1.234"},
        ])

    def test_html_requires_table_identity(self) -> None:
        html = "<section><table><tr><td>value</td></tr></table></section>"
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.html"
            source.write_text(html, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "heading"):
                write_flat_tables(source)

    def test_html_input_does_not_overwrite_json_flat_export(self) -> None:
        html = (
            "<section><h2>p0001-r0001 - page 1</h2>"
            "<p>Status: complete; method: slanet_ocr</p>"
            "<table><tr><td>A</td></tr></table></section>"
        )
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.html"
            source.write_text(html, encoding="utf-8")
            existing = Path(temp) / "tables.flat.json"
            existing.write_text("keep", encoding="utf-8")
            write_flat_tables(source)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_flattens_rows_and_preserves_spans_and_numeric_text(self) -> None:
        flat = flatten_tables(_document())
        self.assertEqual(len(flat["records"]), 2)
        first, second = flat["records"]
        self.assertEqual(first["table_id"], "p0001-r0001")
        self.assertEqual(first["column_1"], "Tài sản")
        self.assertEqual(first["column_1_raw"], "Ta\u0300i sa\u0309n")
        self.assertEqual(first["column_1_ocr_refs"], ["line-1"])
        self.assertEqual(first["column_1_rowspan"], 2)
        self.assertEqual(first["column_2_number"], "451893195")
        self.assertEqual(first["column_3_covered_by"], "r0c1")
        self.assertEqual(second["column_1_covered_by"], "r0c0")
        self.assertEqual(second["column_2_number"], "-1234")
        self.assertIsNone(second["column_3_number"])
        self.assertEqual(flat["tables"][0]["warnings"], ["check OCR"])
        self.assertEqual(flat["unassigned_ocr"][0]["ref"], "line-extra")

    def test_rejects_overlapping_cells(self) -> None:
        document = _document()
        document["tables"][0]["rows"][0].append({
            "row": 0, "column": 0, "rowspan": 1, "colspan": 1,
            "text": "duplicate", "ocr_refs": [],
        })
        with self.assertRaisesRegex(ValueError, "overlap"):
            flatten_tables(document)

    def test_writes_existing_tables_without_model_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tables.json"
            source.write_text(json.dumps(_document()), encoding="utf-8")
            output = write_flat_tables(source)
            self.assertEqual(output, source.with_name("tables.flat.json"))
            self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))["records"]), 2)

    @patch("rag1.extractions.tabular.flat_output.write_flat_tables")
    def test_cli_forwards_existing_json_and_output(self, writer) -> None:
        writer.return_value = Path("out.json")
        main(["tables-flat", "tables.json", "--output", "out.json"])
        writer.assert_called_once_with(Path("tables.json"), Path("out.json"))


if __name__ == "__main__":
    unittest.main()

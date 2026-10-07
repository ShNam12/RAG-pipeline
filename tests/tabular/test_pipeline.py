"""Integration checks for table selection, OCR fallback, and safe output."""

import json
import tempfile
import unittest
from pathlib import Path

from rag1.extractions.tabular.geometry import TextBox
from rag1.extractions.tabular.pipeline import _align_model_grid, _render_html, run_tables
from rag1.extractions.tabular.structure import parse_structure


ROOT = Path(__file__).resolve().parents[2]
DOC = "Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391"
LAYOUT = ROOT / "data" / "layouts" / DOC / "layout.json"
OCR = ROOT / "data" / "ocr" / DOC / "pp-ocrv6-vietnamese" / "pages-79e9d28f078f" / "ocr.json"


class FourColumnPredictor:
    def predict(self, image):
        return ["<table><tr><td></td><td></td><td></td><td></td></tr></table>"], 0.99


class PipelineTests(unittest.TestCase):
    def test_rowspan_cell_keeps_ocr_from_both_rows(self) -> None:
        grid = parse_structure([
            '<table><tr><td rowspan="2"></td><td></td></tr>',
            '<tr><td></td></tr></table>',
        ])
        boxes = [
            TextBox("a", "A", (100, 10, 110, 30)),
            TextBox("b", "B", (300, 10, 310, 30)),
            TextBox("c", "C", (100, 50, 110, 70)),
            TextBox("d", "D", (300, 50, 310, 70)),
        ]
        result, unassigned = _align_model_grid(grid, boxes, [100, 300])
        self.assertEqual(result.rows[0][0].ocr_refs, ["a", "c"])
        self.assertEqual(result.rows[1][0].ocr_refs, ["d"])
        self.assertEqual(unassigned, [])

    def test_html_escapes_ocr_text(self) -> None:
        html = _render_html([{
            "region_id": "r", "page_number": 1, "status": "partial", "method": "ocr_geometry",
            "rows": [[{"rowspan": 1, "colspan": 1, "text": "<script>alert(1)</script>"}]],
        }])
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)

    @unittest.skipUnless(LAYOUT.exists() and OCR.exists(), "sample layout and OCR artifacts unavailable")
    def test_sample_page_five_recovers_five_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = run_tables(
                LAYOUT, ocr_json=OCR, output_dir=Path(temp),
                region_id="region-00005", predictor=FourColumnPredictor(),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            table = payload["tables"][0]
            self.assertEqual(table["region_id"], "p0005-r0001")
            self.assertEqual(table["column_count"], 5)
            self.assertEqual(table["method"], "ocr_geometry")
            self.assertEqual(table["status"], "partial")
            self.assertTrue(any("columns" in warning for warning in table["warnings"]))
            self.assertTrue(any("ocr_refs" in cell and cell["ocr_refs"] for row in table["rows"] for cell in row))
            self.assertTrue(output.with_suffix(".html").exists())

    @unittest.skipUnless(LAYOUT.exists() and OCR.exists(), "sample layout and OCR artifacts unavailable")
    def test_rejects_manifest_dimension_conflict(self) -> None:
        manifest = json.loads(LAYOUT.with_name("manifest.json").read_text(encoding="utf-8"))
        manifest["pages"][4]["pixel_width"] += 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "width conflicts"):
                run_tables(
                    LAYOUT, ocr_json=OCR, manifest_path=path, output_dir=Path(temp),
                    region_id="p0005-r0001", predictor=FourColumnPredictor(),
                )


if __name__ == "__main__":
    unittest.main()

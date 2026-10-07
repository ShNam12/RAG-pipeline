"""CLI tests for table extraction."""

import unittest
from pathlib import Path
from unittest.mock import patch

from rag1 import main


class TableCliTests(unittest.TestCase):
    def test_requires_explicit_ocr_json(self) -> None:
        with self.assertRaises(SystemExit):
            main(["tables", "layout.json"])

    @patch("rag1.extractions.tabular.pipeline.run_tables")
    def test_forwards_every_option(self, run_tables) -> None:
        run_tables.return_value = Path("out/doc/tables.json")
        main([
            "tables", "layout.json", "--ocr-json", "ocr.json", "--manifest", "manifest.json",
            "--output-dir", "out", "--model", "custom/model", "--device", "cuda:0",
            "--region-id", "p0005-r0001",
        ])
        run_tables.assert_called_once_with(
            Path("layout.json"), ocr_json=Path("ocr.json"),
            manifest_path=Path("manifest.json"), output_dir=Path("out"),
            model="custom/model", device="cuda:0", region_id="p0005-r0001",
        )


if __name__ == "__main__":
    unittest.main()

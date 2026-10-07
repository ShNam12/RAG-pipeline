"""The Make entry point forwards table-only options to PowerShell."""

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class MakeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("make"), "GNU Make unavailable")
    def test_flat_target_passes_input_and_output(self) -> None:
        result = subprocess.run(
            ["make", "-n", "tables-flat", "TABLE_INPUT=tables.html", "TABLE_FLAT_OUTPUT=flat.json"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertIn('-InputPath "tables.html"', result.stdout)
        self.assertIn('-OutputPath "flat.json"', result.stdout)

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell unavailable")
    def test_wrapper_requires_ocr_json_only_for_tables(self) -> None:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "make.ps1",
             "tables", "-InputPath", "layout.json"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TABLE_OCR_JSON", result.stderr)

    @unittest.skipUnless(shutil.which("make"), "GNU Make unavailable")
    def test_table_target_passes_all_options(self) -> None:
        result = subprocess.run(
            ["make", "-n", "tables", "TABLE_LAYOUT_JSON=layout.json", "TABLE_OCR_JSON=ocr.json",
             "TABLE_MANIFEST=manifest.json", "TABLE_OUTPUT_DIR=out", "TABLE_MODEL=model/id",
             "TABLE_DEVICE=cpu", "TABLE_REGION_ID=p0005-r0001"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        for argument in (
            '-InputPath "layout.json"', '-OcrJson "ocr.json"', '-Manifest "manifest.json"',
            '-OutputDir "out"', '-Model "model/id"', '-Device "cpu"',
            '-RegionId "p0005-r0001"',
        ):
            self.assertIn(argument, result.stdout)


if __name__ == "__main__":
    unittest.main()

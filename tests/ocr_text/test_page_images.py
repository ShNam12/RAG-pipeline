"""Page image filename resolution contracts."""

import tempfile
import unittest
from pathlib import Path

from rag1.extractions.ocr_text.image_paths import resolve_page_images


class PageImageResolutionTests(unittest.TestCase):
    def test_resolves_page_numbers_and_excludes_bbox_debug_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            page_one = directory / "page-0001.png"
            page_two = directory / "page-0002.png"
            page_one.touch()
            page_two.touch()
            (directory / "page-0001_bbox.png").touch()
            (directory / "notes.png").touch()

            self.assertEqual(resolve_page_images(directory), {1: page_one, 2: page_two})

    def test_rejects_zero_and_duplicate_page_numbers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "page-0000.png").touch()
            with self.assertRaises(ValueError):
                resolve_page_images(directory)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "page-1.png").touch()
            (directory / "page-0001.png").touch()
            with self.assertRaises(ValueError):
                resolve_page_images(directory)

    def test_missing_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                resolve_page_images(Path(temporary_directory) / "missing")


if __name__ == "__main__":
    unittest.main()

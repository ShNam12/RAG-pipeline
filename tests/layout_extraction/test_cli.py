"""Tests for the layout command-line interface."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag1 import main


class LayoutCommandTests(unittest.TestCase):
    def test_layout_command_forwards_options_and_reports_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "report-artifacts"
            artifacts = SimpleNamespace(
                status="partial",
                output_dir=output_dir,
                manifest_path=output_dir / "manifest.json",
                layout_path=output_dir / "layout.json",
                page_images=(output_dir / "pages" / "page-0001.png",),
                bbox_images=(output_dir / "bbox" / "page-0001_bbox.png",),
            )
            stdout = io.StringIO()
            with patch(
                "rag1.extractions.layouts.pipeline.run_pdf_layout",
                return_value=artifacts,
            ) as run_layout:
                with contextlib.redirect_stdout(stdout):
                    main(
                        [
                            "layout",
                            "report.pdf",
                            "--output-dir",
                            str(root),
                            "--image-dir",
                            "rendered_pages",
                            "--dpi",
                            "300",
                            "--device",
                            "gpu:0",
                        ]
                    )

            run_layout.assert_called_once_with(
                Path("report.pdf"),
                output_dir=root,
                image_dir=Path("rendered_pages"),
                dpi=300,
                device="gpu:0",
            )
            self.assertIn("Status: partial", stdout.getvalue())
            self.assertIn(str(output_dir / "manifest.json"), stdout.getvalue())
            self.assertIn(str(output_dir / "layout.json"), stdout.getvalue())
            self.assertIn(f"Page images (1): {output_dir / 'pages'}", stdout.getvalue())
            self.assertIn(
                f"Highlighted page images (1): {output_dir / 'bbox'}",
                stdout.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()

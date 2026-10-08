"""Tests for the command-line pipeline orchestrator."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_pipeline


class PipelineScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def artifact(self, name: str, *, status: str = "complete", tables: bool = True) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"status": status}
        if name.endswith("layout.json"):
            payload["regions"] = (
                [{"raw_label": "table", "location": {"type": "visual"}}]
                if tables else []
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_stage_selection_rejects_duplicates_and_out_of_order_stages(self) -> None:
        self.assertEqual(run_pipeline.parse_stages("layout,ocr,tables,chunk,index"), run_pipeline.STAGES)
        with self.assertRaises(ValueError):
            run_pipeline.parse_stages("ocr,ocr")
        with self.assertRaises(ValueError):
            run_pipeline.parse_stages("index,chunk")

    def test_full_run_passes_artifacts_to_each_stage(self) -> None:
        source = self.root / "report.pdf"
        source.touch()
        layout = self.artifact("layout.json")
        ocr = self.artifact("ocr.json")
        tables = self.artifact("tables.json")
        chunks = self.artifact("chunks.json")
        args = run_pipeline.build_parser().parse_args([
            "--source", str(source), "--env-file", str(self.root / ".env"),
            "--collection", "reports",
        ])
        (self.root / ".env").touch()
        with patch.object(run_pipeline, "run_stage", side_effect=[layout, ocr, tables, chunks, None]) as stage:
            run_pipeline.run_pipeline(args, io.StringIO())

        commands = [call.args[1] for call in stage.call_args_list]
        self.assertEqual([call.args[0] for call in stage.call_args_list], list(run_pipeline.STAGES))
        self.assertEqual(
            commands[0][:4],
            ["uv", "run", "--env-file", (self.root / ".env").resolve().as_posix()],
        )
        self.assertEqual(commands[0][-2:], ["layout", str(source)])
        self.assertNotIn("--model", commands[1])
        self.assertIn(str(layout), commands[1])
        self.assertIn("--extra", commands[2])
        self.assertIn("tabular", commands[2])
        self.assertIn(str(ocr), commands[2])
        self.assertIn(str(ocr), commands[3])
        self.assertEqual(commands[4][-6:], [
            "index", str(chunks), "--tables-json", str(tables), "--collection", "reports",
        ])

    def test_partial_run_uses_existing_artifacts(self) -> None:
        ocr = self.artifact("ocr.json")
        chunks = self.artifact("chunks.json")
        args = run_pipeline.build_parser().parse_args([
            "--stages", "chunk,index", "--ocr-json", str(ocr),
        ])
        with patch.object(run_pipeline, "run_stage", side_effect=[chunks, None]) as stage:
            run_pipeline.run_pipeline(args, io.StringIO())
        self.assertEqual([call.args[0] for call in stage.call_args_list], ["chunk", "index"])
        self.assertIn(str(ocr), stage.call_args_list[0].args[1])
        self.assertIn(str(chunks), stage.call_args_list[1].args[1])

    def test_no_data_tables_skips_table_model(self) -> None:
        layout = self.artifact("layout.json", tables=False)
        ocr = self.artifact("ocr.json")
        args = run_pipeline.build_parser().parse_args([
            "--stages", "tables", "--layout-json", str(layout), "--ocr-json", str(ocr),
        ])
        with patch.object(run_pipeline, "run_stage") as stage:
            run_pipeline.run_pipeline(args, io.StringIO())
        stage.assert_not_called()

    def test_partial_ocr_stops_before_index_by_default(self) -> None:
        ocr = self.artifact("ocr.json", status="partial")
        args = run_pipeline.build_parser().parse_args([
            "--stages", "chunk,index", "--ocr-json", str(ocr),
        ])
        with patch.object(run_pipeline, "run_stage") as stage:
            with self.assertRaises(run_pipeline.PipelineError):
                run_pipeline.run_pipeline(args, io.StringIO())
        stage.assert_not_called()

    def test_stage_output_and_command_are_written_to_log(self) -> None:
        artifact = self.artifact("chunks.json")

        class FakeProcess:
            stdout = iter(["Bảng tài sản: embedding started\n", f"Chunks JSON: {artifact}\n"])

            def __enter__(self) -> "FakeProcess":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def wait(self) -> int:
                return 0

        log = io.StringIO()
        with patch.object(run_pipeline.subprocess, "Popen", return_value=FakeProcess()):
            result = run_pipeline.run_stage("chunk", ["uv", "run", "rag1", "chunk"], log)

        self.assertEqual(result, artifact)
        self.assertIn("Bảng tài sản", log.getvalue())
        self.assertIn("command:", log.getvalue())
        self.assertIn("exit=0", log.getvalue())

    def test_log_preserves_unicode_when_console_is_ascii(self) -> None:
        console_bytes = io.BytesIO()
        console = io.TextIOWrapper(console_bytes, encoding="ascii")
        log = io.StringIO()

        with patch.object(run_pipeline.sys, "stdout", console):
            run_pipeline._log(log, "B\u1ea3ng")

        self.assertIn("B\\u1ea3ng", console_bytes.getvalue().decode("ascii"))
        self.assertIn("B\u1ea3ng", log.getvalue())


if __name__ == "__main__":
    unittest.main()

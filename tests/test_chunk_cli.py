"""CLI wiring for chunking and Qdrant indexing."""

import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from rag1 import _build_parser, main


class ChunkCliTests(unittest.TestCase):
    def test_chunk_accepts_ocr_json_and_writes_sidecar(self) -> None:
        arguments = _build_parser().parse_args(["chunk", "report/ocr.json"])
        self.assertEqual(arguments.ocr_json, Path("report/ocr.json"))

        with patch("rag1.chunking.write_chunks", return_value=Path("report/chunks.json")) as write:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["chunk", "report/ocr.json"])

        write.assert_called_once_with(Path("report/ocr.json"))
        self.assertIn(str(Path("report/chunks.json")), output.getvalue())

    def test_index_forwards_collection_override(self) -> None:
        with patch(
            "rag1.indexing.index_chunk_artifact",
            return_value={"collection": "custom", "children": 2, "nodes": 6},
        ) as index:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["index", "report/chunks.json", "--collection", "custom"])

        index.assert_called_once_with(
            Path("report/chunks.json"), collection_name="custom", tables_path=None,
        )

    def test_index_forwards_reconstructed_tables(self) -> None:
        with patch(
            "rag1.indexing.index_chunk_artifact",
            return_value={"collection": "custom", "children": 2, "nodes": 6, "table_cells": 3},
        ) as index:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main([
                    "index", "report/chunks.json", "--tables-json", "report/tables.json",
                ])

        index.assert_called_once_with(
            Path("report/chunks.json"), collection_name=None,
            tables_path=Path("report/tables.json"),
        )
        self.assertIn("3 table cells", output.getvalue())

    def test_index_reports_qdrant_http_error_without_traceback(self) -> None:
        from httpx import Headers
        from qdrant_client.http.exceptions import UnexpectedResponse

        error = UnexpectedResponse(403, "Forbidden", b"index creation denied", Headers())
        with patch("rag1.indexing.index_chunk_artifact", side_effect=error):
            with contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    main(["index", "report/chunks.json"])

        self.assertEqual(result.exception.code, 2)
        self.assertIn("index creation denied", output.getvalue())


if __name__ == "__main__":
    unittest.main()

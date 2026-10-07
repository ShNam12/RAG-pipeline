"""Tests for OpenAI OCR Markdown refinement."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from rag1 import _build_parser, main
from rag1.extractions.refinement.openai_ocr import (
    collect_ocr_markdown,
    refine_ocr_file,
    rewrite_ocr_text,
)


class OpenAiOcrTests(unittest.TestCase):
    def test_request_uses_responses_api_and_preserves_unicode(self) -> None:
        response = {"status": "completed", "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "Ngân hàng\n"}],
        }]}
        with patch("rag1.extractions.refinement.openai_ocr.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                response
            ).encode("utf-8")
            result = rewrite_ocr_text("Ngan hang\n", api_key="secret-key")

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(result, "Ngân hàng\n")
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
        self.assertNotIn("secret-key", request.full_url)
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertEqual(body["input"], "Ngan hang\n")
        self.assertEqual(body["reasoning"]["effort"], "low")
        self.assertFalse(body["store"])

    def test_incomplete_response_is_not_returned(self) -> None:
        response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        with patch("rag1.extractions.refinement.openai_ocr.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                response
            ).encode("utf-8")
            with self.assertRaisesRegex(RuntimeError, "max_output_tokens"):
                rewrite_ocr_text("source", api_key="secret-key")

    def test_http_failure_does_not_expose_key_or_response_body(self) -> None:
        error = HTTPError("https://example.com", 429, "rate limit", {}, io.BytesIO(b"secret-key"))
        with patch("rag1.extractions.refinement.openai_ocr.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 429") as caught:
                rewrite_ocr_text("source", api_key="secret-key")
        self.assertNotIn("secret-key", str(caught.exception))

    def test_directory_discovery_finds_only_ocr_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "ocr.md"
            second = root / "b" / "ocr.md"
            for source in (first, second):
                source.parent.mkdir()
                source.write_text("source", encoding="utf-8")
            (root / "empty" / "ocr.md").parent.mkdir()
            (root / "empty" / "ocr.md").write_text("", encoding="utf-8")
            (root / "b" / "ocr.refined.md").write_text("result", encoding="utf-8")

            self.assertEqual(collect_ocr_markdown(root), [first, second])

    def test_refinement_preserves_source_and_requires_overwrite_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ocr.md"
            source.write_text("Ngan hang\n", encoding="utf-8")
            with patch(
                "rag1.extractions.refinement.openai_ocr.rewrite_ocr_text",
                return_value="Ngân hàng\n",
            ) as rewrite:
                output = refine_ocr_file(source, api_key="secret-key")
                self.assertEqual(output.read_text(encoding="utf-8"), "Ngân hàng\n")
                self.assertEqual(source.read_text(encoding="utf-8"), "Ngan hang\n")
                with self.assertRaises(FileExistsError):
                    refine_ocr_file(source, api_key="secret-key")
                self.assertEqual(rewrite.call_count, 1)

    def test_cli_recursively_refines_discovered_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "nested" / "ocr.md"
            source.parent.mkdir()
            source.write_text("source", encoding="utf-8")
            with patch.dict("os.environ", {"OPENAI_API_KEY": "secret-key"}):
                with patch(
                    "rag1.extractions.refinement.openai_ocr.refine_ocr_file",
                    return_value=source.with_name("ocr.refined.md"),
                ) as refine:
                    main(["refine-ocr", directory])

        refine.assert_called_once_with(
            source, api_key="secret-key", model="gpt-5.4-mini", overwrite=False
        )
        self.assertEqual(_build_parser().parse_args(["refine-ocr"]).source, Path("data/ocr"))


if __name__ == "__main__":
    unittest.main()

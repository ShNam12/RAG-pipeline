"""Read-only checks for the Qdrant collection count script."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import check_qdrant


class CheckQdrantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.client.get_collections.return_value = SimpleNamespace(collections=[
            SimpleNamespace(name="filled"), SimpleNamespace(name="empty")
        ])
        self.client.count.side_effect = lambda *, collection_name, exact: SimpleNamespace(
            count={"filled": 3, "empty": 0}[collection_name]
        )

    def test_lists_all_collections_and_exact_counts_without_credentials(self) -> None:
        output = io.StringIO()
        with patch.dict("os.environ", {"QDRANT_URL": "http://private-host:6333", "QDRANT_API_KEY": "secret"}):
            with patch("scripts.check_qdrant.QdrantClient", return_value=self.client) as constructor:
                with redirect_stdout(output):
                    result = check_qdrant.main([])

        self.assertEqual(result, 0)
        constructor.assert_called_once_with(url="http://private-host:6333", api_key="secret")
        self.assertEqual(self.client.count.call_count, 2)
        self.assertEqual(output.getvalue(), "empty: 0\nfilled: 3\nHas data: yes\n")
        self.assertNotIn("secret", output.getvalue())
        self.assertNotIn("private-host", output.getvalue())
        self.client.upsert.assert_not_called()
        self.client.delete.assert_not_called()

    def test_can_check_one_collection(self) -> None:
        output = io.StringIO()
        with patch.dict("os.environ", {"QDRANT_URL": "http://localhost:6333"}):
            with patch("scripts.check_qdrant.QdrantClient", return_value=self.client):
                with redirect_stdout(output):
                    result = check_qdrant.main(["--collection", "empty"])

        self.assertEqual(result, 0)
        self.client.get_collections.assert_not_called()
        self.client.count.assert_called_once_with(collection_name="empty", exact=True)
        self.assertEqual(output.getvalue(), "empty: 0\nHas data: no\n")

    def test_requires_qdrant_url(self) -> None:
        errors = io.StringIO()
        with patch.dict("os.environ", {"QDRANT_URL": ""}):
            with redirect_stderr(errors):
                result = check_qdrant.main([])

        self.assertEqual(result, 2)
        self.assertIn("QDRANT_URL is required", errors.getvalue())

    def test_reports_empty_server(self) -> None:
        self.client.get_collections.return_value = SimpleNamespace(collections=[])
        output = io.StringIO()
        with patch.dict("os.environ", {"QDRANT_URL": "http://localhost:6333"}):
            with patch("scripts.check_qdrant.QdrantClient", return_value=self.client):
                with redirect_stdout(output):
                    result = check_qdrant.main([])

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "No collections found\nHas data: no\n")
        self.client.count.assert_not_called()

    def test_connection_error_does_not_print_credentials(self) -> None:
        self.client.get_collections.side_effect = RuntimeError("secret")
        errors = io.StringIO()
        with patch.dict("os.environ", {"QDRANT_URL": "http://private-host:6333", "QDRANT_API_KEY": "secret"}):
            with patch("scripts.check_qdrant.QdrantClient", return_value=self.client):
                with redirect_stderr(errors):
                    result = check_qdrant.main([])

        self.assertEqual(result, 1)
        self.assertEqual(errors.getvalue(), "Qdrant check failed (RuntimeError)\n")


if __name__ == "__main__":
    unittest.main()

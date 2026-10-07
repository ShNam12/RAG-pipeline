"""Qdrant storage and embedding checks for OCR chunks."""

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag1.chunking import load_chunk_artifact
from rag1.indexing import DEFAULT_COLLECTION, embedding_text, get_parent, index_chunk_artifact


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))


class FakeModel:
    max_seq_length = 2048
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.inputs = texts
        return [[0.1] * 1024 for _ in texts]


def artifact(version: str) -> dict:
    section_id = f"00000000-0000-0000-0000-0000000000{version}0"
    root_id = f"00000000-0000-0000-0000-0000000000{version}1"
    parent_id = f"00000000-0000-0000-0000-0000000000{version}2"
    child_id = f"00000000-0000-0000-0000-0000000000{version}3"
    table_regions = [{
        "region_id": "r2",
        "table": {"structure_status": "pending", "structure": None},
    }]
    return {
        "schema_version": "1.1",
        "source": "report.pdf",
        "source_id": "00000000-0000-0000-0000-000000000001",
        "ocr_path": "ocr.json",
        "ocr_sha256": version * 64,
        "ocr_model": "paddleocr-v6",
        "ocr_status": "complete",
        "sections": [{"id": section_id, "heading": "Assets", "text": "Assets\n\nCash"}],
        "nodes": [
            {"id": root_id, "section_id": section_id, "parent_id": None, "level": 0,
             "text": "Assets\n\nCash", "page_numbers": [1], "region_ids": ["r1", "r2"],
             "tables": table_regions},
            {"id": parent_id, "section_id": section_id, "parent_id": root_id, "level": 1,
             "text": "Assets\n\nCash", "page_numbers": [1], "region_ids": ["r1", "r2"],
             "tables": table_regions},
            {"id": child_id, "section_id": section_id, "parent_id": parent_id, "level": 2,
             "text": "Cash", "page_numbers": [1], "region_ids": ["r2"],
             "tables": table_regions},
        ],
    }


class IndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(":memory:")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "chunks.json"

    def write_artifact(self, version: str) -> dict:
        data = artifact(version)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def write_tables(self, data: dict, *, text: str = "451.893.195") -> Path:
        path = self.path.with_name("tables.json")
        path.write_text(json.dumps({
            "schema_version": "1.0",
            "source": data["source"],
            "ocr_json": str(Path(data["ocr_path"]).resolve()),
            "model": "PaddlePaddle/SLANet_plus_safetensors",
            "status": "complete",
            "tables": [{
                "region_id": "r2", "page_number": 1, "bbox": [0, 0, 100, 100],
                "status": "complete", "method": "slanet_ocr",
                "rows": [[
                    {"row": 0, "column": 0, "rowspan": 1, "colspan": 1,
                     "text": "Tài sản", "ocr_refs": ["line-1"]},
                    {"row": 0, "column": 1, "rowspan": 1, "colspan": 1,
                     "text": text, "ocr_refs": ["line-2"]},
                ]],
            }],
        }), encoding="utf-8")
        return path

    def test_indexes_reconstructed_table_cells_with_provenance(self) -> None:
        data = self.write_artifact("1")
        tables = self.write_tables(data)
        model = FakeModel()

        result = index_chunk_artifact(
            self.path, tables_path=tables, client=self.client, model=model,
        )

        self.assertEqual(result["table_cells"], 2)
        self.assertEqual(len(model.inputs), 3)
        records, _ = self.client.scroll(DEFAULT_COLLECTION, with_payload=True, with_vectors=True)
        cells = [record for record in records if record.payload["record_type"] == "table_cell"]
        self.assertEqual(len(cells), 2)
        amount = next(record for record in cells if record.payload["column"] == 1)
        self.assertEqual(amount.payload["text"], "451.893.195")
        self.assertEqual(amount.payload["region_id"], "r2")
        self.assertEqual(amount.payload["ocr_refs"], ["line-2"])
        self.assertEqual(amount.payload["source_id"], data["source_id"])
        self.assertEqual(len(amount.vector["text"]), 1024)
        self.assertEqual(str(uuid.UUID(str(amount.id))), str(amount.id))

    def test_table_cells_are_replaced_with_new_table_artifact(self) -> None:
        data = self.write_artifact("1")
        tables = self.write_tables(data)
        index_chunk_artifact(self.path, tables_path=tables, client=self.client, model=FakeModel())
        before, _ = self.client.scroll(DEFAULT_COLLECTION, with_payload=True)
        old_ids = {str(record.id) for record in before if record.payload["record_type"] == "table_cell"}

        self.write_tables(data, text="451.893.196")
        result = index_chunk_artifact(
            self.path, tables_path=tables, client=self.client, model=FakeModel(),
        )

        self.assertEqual(result["table_cells"], 2)
        self.assertEqual(self.client.retrieve(DEFAULT_COLLECTION, list(old_ids)), [])

    def test_rejects_tables_from_another_source_before_upload(self) -> None:
        data = self.write_artifact("1")
        tables = self.write_tables(data)
        table_data = json.loads(tables.read_text(encoding="utf-8"))
        table_data["source"] = "another.pdf"
        tables.write_text(json.dumps(table_data), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "source"):
            index_chunk_artifact(
                self.path, tables_path=tables, client=self.client, model=FakeModel(),
            )
        self.assertFalse(self.client.collection_exists(DEFAULT_COLLECTION))

    def test_indexes_only_children_and_retrieves_immediate_parent(self) -> None:
        data = self.write_artifact("1")
        model = FakeModel()

        result = index_chunk_artifact(self.path, client=self.client, model=model)

        self.assertEqual(result["children"], 1)
        self.assertEqual(model.inputs, ["Assets\n\nCash"])
        records = self.client.retrieve(
            collection_name=DEFAULT_COLLECTION,
            ids=[node["id"] for node in data["nodes"]],
            with_vectors=True,
        )
        self.assertEqual(len(records), 3)
        by_id = {str(record.id): record for record in records}
        self.assertFalse(by_id[data["nodes"][0]["id"]].vector)
        self.assertFalse(by_id[data["nodes"][1]["id"]].vector)
        self.assertEqual(len(by_id[data["nodes"][2]["id"]].vector["text"]), 1024)
        self.assertEqual(by_id[data["nodes"][2]["id"]].payload["tables"], data["nodes"][2]["tables"])
        parent = get_parent(self.client, DEFAULT_COLLECTION, data["nodes"][2]["id"])
        self.assertEqual(parent["id"], data["nodes"][1]["id"])

    def test_heading_already_in_child_is_not_repeated(self) -> None:
        self.assertEqual(
            embedding_text({"text": "Assets\n\nCash"}, {"heading": "Assets"}),
            "Assets\n\nCash",
        )

    def test_reindexing_same_artifact_is_idempotent(self) -> None:
        data = self.write_artifact("1")
        index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        result = index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(result["replaced"], 0)
        self.assertEqual(len(self.client.retrieve(
            DEFAULT_COLLECTION, [node["id"] for node in data["nodes"]]
        )), 3)

    def test_reindex_replaces_only_this_source_after_upload(self) -> None:
        old = self.write_artifact("1")
        index_chunk_artifact(self.path, client=self.client, model=FakeModel())
        another = self.write_artifact("3")
        another["source_id"] = "00000000-0000-0000-0000-000000000099"
        self.path.write_text(json.dumps(another), encoding="utf-8")
        index_chunk_artifact(self.path, client=self.client, model=FakeModel())
        new = self.write_artifact("2")

        index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(self.client.retrieve(
            DEFAULT_COLLECTION, [node["id"] for node in old["nodes"]]
        ), [])
        self.assertEqual(len(self.client.retrieve(
            DEFAULT_COLLECTION, [node["id"] for node in new["nodes"]]
        )), 3)
        self.assertEqual(len(self.client.retrieve(
            DEFAULT_COLLECTION, [node["id"] for node in another["nodes"]]
        )), 3)

    def test_failed_upload_preserves_prior_version(self) -> None:
        old = self.write_artifact("1")
        index_chunk_artifact(self.path, client=self.client, model=FakeModel())
        self.write_artifact("2")

        with patch.object(self.client, "upsert", side_effect=RuntimeError("network failed")):
            with self.assertRaisesRegex(RuntimeError, "network failed"):
                index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(len(self.client.retrieve(
            DEFAULT_COLLECTION, [node["id"] for node in old["nodes"]]
        )), 3)

    def test_missing_child_vector_during_readback_preserves_prior_version(self) -> None:
        old = self.write_artifact("1")
        index_chunk_artifact(self.path, client=self.client, model=FakeModel())
        new = self.write_artifact("2")
        child_id = new["nodes"][2]["id"]
        retrieve = self.client.retrieve

        def without_child_vector(*args: object, **kwargs: object) -> list:
            records = retrieve(*args, **kwargs)
            if kwargs.get("with_vectors"):
                return [
                    record.model_copy(update={"vector": {}}) if str(record.id) == child_id else record
                    for record in records
                ]
            return records

        with patch.object(self.client, "retrieve", side_effect=without_child_vector):
            with self.assertRaisesRegex(RuntimeError, "child vector"):
                index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(len(retrieve(
            DEFAULT_COLLECTION, [node["id"] for node in old["nodes"]]
        )), 3)

    def test_rejects_mismatched_collection_dimension(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self.write_artifact("1")
        self.client.create_collection(
            collection_name=DEFAULT_COLLECTION,
            vectors_config={"text": VectorParams(size=16, distance=Distance.DOT)},
        )

        with self.assertRaisesRegex(ValueError, "1024"):
            index_chunk_artifact(self.path, client=self.client, model=FakeModel())

    def test_creates_source_id_index_before_filtered_scroll(self) -> None:
        from qdrant_client.models import PayloadSchemaType

        self.write_artifact("1")
        original_scroll = self.client.scroll
        indexed = False

        def create_index(**kwargs: object) -> None:
            nonlocal indexed
            self.assertEqual(kwargs["field_name"], "source_id")
            self.assertEqual(kwargs["field_schema"], PayloadSchemaType.UUID)
            self.assertTrue(kwargs["wait"])
            indexed = True

        def strict_scroll(*args: object, **kwargs: object) -> object:
            if not indexed:
                raise RuntimeError("Index required for source_id")
            return original_scroll(*args, **kwargs)

        with patch.object(self.client, "create_payload_index", side_effect=create_index):
            with patch.object(self.client, "scroll", side_effect=strict_scroll):
                result = index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(result["children"], 1)
        self.assertTrue(indexed)

    def test_reuses_existing_keyword_source_id_index(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        self.write_artifact("1")
        self.client.create_collection(
            collection_name=DEFAULT_COLLECTION,
            vectors_config={"text": VectorParams(size=1024, distance=Distance.DOT)},
        )
        get_collection = self.client.get_collection

        def indexed_collection(name: str) -> object:
            info = get_collection(name)
            return info.model_copy(update={
                "payload_schema": {
                    "source_id": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)
                }
            })

        with patch.object(self.client, "get_collection", side_effect=indexed_collection):
            with patch.object(self.client, "create_payload_index") as create_index:
                index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        create_index.assert_not_called()

    def test_rejects_incompatible_source_id_index_before_upload(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        self.write_artifact("1")
        self.client.create_collection(
            collection_name=DEFAULT_COLLECTION,
            vectors_config={"text": VectorParams(size=1024, distance=Distance.DOT)},
        )
        get_collection = self.client.get_collection

        def incompatible_collection(name: str) -> object:
            info = get_collection(name)
            return info.model_copy(update={
                "payload_schema": {
                    "source_id": SimpleNamespace(data_type=PayloadSchemaType.INTEGER)
                }
            })

        with patch.object(self.client, "get_collection", side_effect=incompatible_collection):
            with self.assertRaisesRegex(ValueError, "source_id"):
                index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(self.client.count(DEFAULT_COLLECTION).count, 0)

    def test_failed_source_id_index_creation_stops_before_filter_or_model(self) -> None:
        self.write_artifact("1")
        model = FakeModel()

        with patch.object(self.client, "create_payload_index", side_effect=RuntimeError("denied")):
            with patch.object(self.client, "scroll") as scroll:
                with self.assertRaisesRegex(RuntimeError, "denied"):
                    index_chunk_artifact(self.path, client=self.client, model=model)

        scroll.assert_not_called()
        self.assertEqual(model.inputs, [])

    def test_generic_collection_env_does_not_redirect_chunk_index(self) -> None:
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        self.write_artifact("1")
        self.client.create_collection(
            collection_name="rag_hybrid_v1",
            vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

        with patch.dict("os.environ", {"COLLECTION_NAME": "rag_hybrid_v1"}):
            result = index_chunk_artifact(self.path, client=self.client, model=FakeModel())

        self.assertEqual(result["collection"], DEFAULT_COLLECTION)
        self.assertEqual(self.client.count("rag_hybrid_v1").count, 0)

    def test_explicit_incompatible_collection_remains_rejected(self) -> None:
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams

        self.write_artifact("1")
        self.client.create_collection(
            collection_name="rag_hybrid_v1",
            vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

        with self.assertRaisesRegex(ValueError, "named text vector"):
            index_chunk_artifact(
                self.path, client=self.client, model=FakeModel(), collection_name="rag_hybrid_v1"
            )

        self.assertEqual(self.client.count("rag_hybrid_v1").count, 0)

    def test_rejects_child_longer_than_model_limit_before_upsert(self) -> None:
        self.write_artifact("1")
        model = FakeModel()
        model.max_seq_length = 2

        with self.assertRaisesRegex(ValueError, "token limit"):
            index_chunk_artifact(self.path, client=self.client, model=model)

        self.assertEqual(model.inputs, [])

    def test_rejects_missing_required_node_fields_as_invalid_artifact(self) -> None:
        data = self.write_artifact("1")
        del data["nodes"][0]["tables"]
        self.path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_chunk_artifact(self.path)

    def test_old_chunk_schema_requires_regeneration(self) -> None:
        data = self.write_artifact("1")
        data["schema_version"] = "1.0"
        self.path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "schema version"):
            load_chunk_artifact(self.path)


if __name__ == "__main__":
    unittest.main()

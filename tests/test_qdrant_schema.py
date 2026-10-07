"""Validate indexed chunks and staged fixtures against the Qdrant contracts."""

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from qdrant_client import QdrantClient
from referencing import Registry, Resource

from rag1.indexing import DEFAULT_COLLECTION, index_chunk_artifact
from tests.test_indexing import FakeModel, artifact


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas/qdrant"
FIXTURES = ROOT / "fixtures/qdrant"


def validator(name: str) -> Draft202012Validator:
    registry = Registry()
    for path in SCHEMAS.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        registry = registry.with_resource(path.as_uri(), Resource.from_contents(schema))
    return Draft202012Validator(
        {"$ref": (SCHEMAS / name).as_uri()},
        registry=registry,
        format_checker=FormatChecker(),
    )


class QdrantSchemaTests(unittest.TestCase):
    def test_indexed_roots_parents_and_children_follow_point_schema(self) -> None:
        data = artifact("1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            client = QdrantClient(":memory:")
            index_chunk_artifact(path, client=client, model=FakeModel())
            records = client.retrieve(
                DEFAULT_COLLECTION,
                [node["id"] for node in data["nodes"]],
                with_payload=True,
                with_vectors=True,
            )

        self.assertEqual(len(records), 3)
        check = validator("point.schema.json")
        for record in records:
            check.validate({
                "id": str(record.id),
                "vector": record.vector or {},
                "payload": record.payload,
            })

    def test_indexed_table_cells_follow_point_schema(self) -> None:
        data = artifact("1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            tables = path.with_name("tables.json")
            tables.write_text(json.dumps({
                "schema_version": "1.0", "source": data["source"],
                "ocr_json": str(Path(data["ocr_path"]).resolve()),
                "model": "SLANet", "status": "complete",
                "tables": [{
                    "region_id": "r2", "page_number": 1, "bbox": [0, 0, 100, 100],
                    "status": "complete", "method": "slanet_ocr",
                    "rows": [[{"row": 0, "column": 0, "rowspan": 1, "colspan": 1,
                               "text": "Tài sản", "ocr_refs": ["line-1"]}]],
                }],
            }), encoding="utf-8")
            client = QdrantClient(":memory:")
            index_chunk_artifact(path, tables_path=tables, client=client, model=FakeModel())
            records, _ = client.scroll(DEFAULT_COLLECTION, with_payload=True, with_vectors=True)

        cell = next(record for record in records if record.payload["record_type"] == "table_cell")
        validator("point.schema.json").validate({
            "id": str(cell.id), "vector": cell.vector, "payload": cell.payload,
        })

    def test_existing_and_chunk_fixtures_follow_fixture_schema(self) -> None:
        check = validator("fixture.schema.json")
        paths = sorted(FIXTURES.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(fixture=path.name):
                check.validate(json.loads(path.read_text(encoding="utf-8")))

    def test_chunk_fixture_keeps_immediate_parent_links(self) -> None:
        path = FIXTURES / "chunk-hierarchy.example.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        points = {point["id"]: point["payload"] for point in data["points"]}
        self.assertEqual({payload["level"] for payload in points.values()}, {0, 1, 2})
        for payload in points.values():
            if payload["level"] == 2:
                self.assertEqual(points[payload["parent_id"]]["level"], 1)
            elif payload["level"] == 1:
                self.assertEqual(points[payload["parent_id"]]["level"], 0)

    def test_chunk_point_rejects_wrong_vector_or_table_shape(self) -> None:
        path = FIXTURES / "chunk-hierarchy.example.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        root, _, child = fixture["points"]
        check = validator("point.schema.json")

        child_point = {**child, "vector": {"text": [0.1] * 1024}}
        check.validate(child_point)
        root_point = {**root, "vector": {}}
        check.validate(root_point)

        for vector in ({}, {"text": [0.1] * 1023}):
            with self.subTest(vector_size=len(vector.get("text", []))):
                with self.assertRaises(ValidationError):
                    check.validate({**child, "vector": vector})
        with self.assertRaises(ValidationError):
            check.validate({**root, "vector": {"text": [0.1] * 1024}})

        bad_table = copy.deepcopy(child_point)
        bad_table["payload"]["tables"][0]["table"]["structure_status"] = "complete"
        with self.assertRaises(ValidationError):
            check.validate(bad_table)

    def test_header_keyed_table_fixture_preserves_real_cell_relationships(self) -> None:
        path = FIXTURES / "table-keyed.example.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        point = fixture["points"][0]
        payload = point["payload"]
        validator("fixture.schema.json").validate(fixture)
        validator("point.schema.json").validate({**point, "vector": {}})

        self.assertEqual(payload["table_id"], "p0004-r0005")
        self.assertEqual(point["id"], str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rag1:table-kv:{payload['table_artifact_sha256']}:{payload['table_id']}",
        )))
        self.assertEqual(set(payload["headers"]), {"Họ tên", "Chức vụ"})
        self.assertEqual(payload["raw_header_cell"]["colspan"], 2)
        self.assertEqual(len(payload["records"]), 3)
        for record in payload["records"]:
            self.assertEqual(set(record["values"]), set(payload["headers"]))
            self.assertEqual(set(record["cell_ocr_refs"]), set(payload["headers"]))
        self.assertEqual(payload["records"][0]["values"], {
            "Họ tên": "Bà Nguyễn Thị Thu Hương",
            "Chức vụ": "Trưởng ban kiểm soát",
        })

        invalid_value = copy.deepcopy(point)
        invalid_value["payload"]["records"][0]["values"]["Chức vụ"] = 123
        with self.assertRaises(ValidationError):
            validator("point.schema.json").validate({**invalid_value, "vector": {}})
        with self.assertRaises(ValidationError):
            validator("point.schema.json").validate({**point, "vector": {"text": [0.1] * 1024}})

    def test_table_row_fixture_resolves_to_header_keyed_table(self) -> None:
        table = json.loads((FIXTURES / "table-keyed.example.json").read_text(encoding="utf-8"))
        rows = json.loads((FIXTURES / "table-row.example.json").read_text(encoding="utf-8"))
        table_point = table["points"][0]
        validator("fixture.schema.json").validate(rows)

        expected = {record["row_index"]: record for record in table_point["payload"]["records"]}
        self.assertEqual(len(rows["points"]), len(expected))
        for point in rows["points"]:
            payload = point["payload"]
            self.assertEqual(payload["table_point_id"], table_point["id"])
            self.assertEqual(payload["table_artifact_sha256"], table_point["payload"]["table_artifact_sha256"])
            self.assertEqual(payload["values"], expected[payload["row_index"]]["values"])
            self.assertEqual(payload["cell_ocr_refs"], expected[payload["row_index"]]["cell_ocr_refs"])
            self.assertEqual(
                payload["embedding_text"],
                "\n".join(f"{header}: {value}" for header, value in payload["values"].items()),
            )
            self.assertEqual(point["id"], str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rag1:table-row:{table_point['id']}:{payload['row_index']}",
            )))

    def test_hybrid_search_point_requires_dense_and_sparse_vectors(self) -> None:
        chunk = json.loads((FIXTURES / "chunk-hierarchy.example.json").read_text(encoding="utf-8"))
        row = json.loads((FIXTURES / "table-row.example.json").read_text(encoding="utf-8"))
        check = validator("hybrid-search-point.schema.json")
        vectors = {"text": [0.1] * 1024, "lexical": {"indices": [2, 7], "values": [0.4, 1.0]}}
        for point in (chunk["points"][2], row["points"][0]):
            with self.subTest(record_type=point["payload"]["record_type"]):
                check.validate({**point, "vector": vectors})
                with self.assertRaises(ValidationError):
                    check.validate({**point, "vector": {"text": [0.1] * 1024}})

        bad = copy.deepcopy(row["points"][0])
        bad["payload"]["values"]["Chức vụ"] = 123
        with self.assertRaises(ValidationError):
            check.validate({**bad, "vector": vectors})

    def test_current_table_cell_fixture_matches_indexer_point_shape(self) -> None:
        fixture = json.loads((FIXTURES / "table-cell.example.json").read_text(encoding="utf-8"))
        point = fixture["points"][0]
        payload = point["payload"]
        validator("fixture.schema.json").validate(fixture)
        validator("point.schema.json").validate({**point, "vector": {"text": [0.1] * 1024}})
        self.assertEqual(payload["embedding_text"], "Table page 1, region r2\nTài sản\n451.893.195")
        self.assertEqual(point["id"], str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rag1:table-cell:{payload['source_id']}:{payload['ocr_sha256']}:"
            f"{payload['table_artifact_sha256']}:"
            f"{payload['region_id']}:{payload['row']}:{payload['column']}",
        )))


if __name__ == "__main__":
    unittest.main()

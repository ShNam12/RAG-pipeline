"""Embed OCR child chunks and table cells, then store them in Qdrant."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any

from rag1.chunking import load_chunk_artifact


MODEL_ID = "thanhtantran/Vietnamese_Embedding"
DEFAULT_COLLECTION = "rag1_vietnamese_embedding"
VECTOR_NAME = "text"
VECTOR_SIZE = 1024
POINT_BATCH_SIZE = 64
logger = logging.getLogger(__name__)


def embedding_text(node: dict[str, Any], section: dict[str, Any]) -> str:
    """Give a child its section heading once without changing saved OCR text."""
    heading = section.get("heading")
    text = node["text"]
    if not heading or text.lstrip().startswith(heading):
        return text
    return f"{heading}\n\n{text}"


def _ensure_collection(client: Any, collection_name: str) -> None:
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.DOT)
            },
        )
    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    if not isinstance(vectors, dict) or set(vectors) != {VECTOR_NAME}:
        raise ValueError(f"Qdrant collection {collection_name!r} must have only a named text vector")
    params = vectors[VECTOR_NAME]
    if params.size != VECTOR_SIZE or params.distance != Distance.DOT:
        raise ValueError(
            f"Qdrant collection {collection_name!r} must use {VECTOR_SIZE}-dimensional Dot vectors"
        )
    source_index = (collection.payload_schema or {}).get("source_id")
    if source_index is None:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source_id",
            field_schema=PayloadSchemaType.UUID,
            wait=True,
        )
    elif source_index.data_type not in {PayloadSchemaType.UUID, PayloadSchemaType.KEYWORD}:
        raise ValueError(
            f"Qdrant collection {collection_name!r} needs a uuid or keyword index on source_id"
        )


def _source_point_ids(client: Any, collection_name: str, source_id: str) -> set[str]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    found: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="source_id", match=MatchValue(value=source_id))
            ]),
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        found.update(str(record.id) for record in records)
        if offset is None:
            return found


def _validate_embedding_inputs(model: Any, inputs: list[str]) -> None:
    limit = model.max_seq_length
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Embedding model has no valid maximum sequence length")
    for index, text in enumerate(inputs):
        count = len(model.tokenizer.encode(text, add_special_tokens=True))
        if count > limit:
            raise ValueError(
                f"Embedding input {index + 1} uses {count} model tokens, exceeding the {limit}-token limit"
            )


def _table_cells(tables_path: str | Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Load nonempty reconstructed cells with stable IDs and source provenance."""
    path = Path(tables_path)
    raw = path.read_bytes()
    tables = json.loads(raw)
    if not isinstance(tables, dict) or tables.get("schema_version") != "1.0":
        raise ValueError("Unsupported tables.json schema version")
    if tables.get("source") != artifact["source"]:
        raise ValueError("tables.json source does not match chunks.json source")
    ocr_path = tables.get("ocr_json")
    if not isinstance(ocr_path, str) or Path(ocr_path).resolve() != Path(artifact["ocr_path"]).resolve():
        raise ValueError("tables.json OCR source does not match chunks.json OCR source")
    if not isinstance(tables.get("tables"), list):
        raise ValueError("tables.json must contain a tables list")
    if not isinstance(tables.get("model"), str) or not tables["model"]:
        raise ValueError("tables.json is missing its reconstruction model")
    table_hash = hashlib.sha256(raw).hexdigest()
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for table in tables["tables"]:
        if not isinstance(table, dict) or not isinstance(table.get("rows"), list):
            raise ValueError("tables.json contains an invalid table")
        region_id = table.get("region_id")
        page_number = table.get("page_number")
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("tables.json table is missing a region ID")
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            raise ValueError("tables.json table has an invalid page number")
        bbox = table.get("bbox")
        if (
            not isinstance(bbox, list) or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in bbox)
        ):
            raise ValueError("tables.json table has an invalid bounding box")
        if table.get("status") not in {"complete", "partial"}:
            raise ValueError("tables.json table has an invalid status")
        if not isinstance(table.get("method"), str) or not table["method"]:
            raise ValueError("tables.json table has an invalid reconstruction method")
        for row in table["rows"]:
            if not isinstance(row, list):
                raise ValueError("tables.json contains an invalid table row")
            for cell in row:
                if not isinstance(cell, dict) or not isinstance(cell.get("text"), str):
                    raise ValueError("tables.json contains an invalid table cell")
                value = cell["text"].strip()
                if not value:
                    continue
                row_number = cell.get("row")
                column_number = cell.get("column")
                if any(
                    isinstance(number, bool) or not isinstance(number, int) or number < 0
                    for number in (row_number, column_number)
                ):
                    raise ValueError("tables.json cell has invalid row or column coordinates")
                if any(
                    isinstance(cell.get(field), bool)
                    or not isinstance(cell.get(field), int)
                    or cell[field] < 1
                    for field in ("rowspan", "colspan")
                ):
                    raise ValueError("tables.json cell has an invalid row or column span")
                cell_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"rag1:table-cell:{artifact['source_id']}:{artifact['ocr_sha256']}:"
                    f"{table_hash}:"
                    f"{region_id}:{row_number}:{column_number}",
                ))
                if cell_id in seen:
                    raise ValueError("tables.json contains duplicate table cell coordinates")
                seen.add(cell_id)
                refs = cell.get("ocr_refs", [])
                if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs):
                    raise ValueError("tables.json cell has invalid OCR references")
                input_parts = [f"Table page {page_number}, region {region_id}"]
                if row_number > 0 and table["rows"]:
                    header = next((
                        candidate.get("text", "").strip()
                        for candidate in table["rows"][0]
                        if isinstance(candidate, dict)
                        and candidate.get("column") == column_number
                    ), "")
                    if header and header != value:
                        input_parts.append(header)
                if column_number > 0:
                    row_label = next((
                        candidate.get("text", "").strip()
                        for candidate in row
                        if isinstance(candidate, dict) and candidate.get("column") == 0
                    ), "")
                    if row_label and row_label != value:
                        input_parts.append(row_label)
                input_parts.append(value)
                input_text = "\n".join(input_parts)
                payload = {
                    "schema_version": "1.0",
                    "record_type": "table_cell",
                    "source": artifact["source"],
                    "source_id": artifact["source_id"],
                    "ocr_path": artifact["ocr_path"],
                    "ocr_sha256": artifact["ocr_sha256"],
                    "ocr_model": artifact["ocr_model"],
                    "ocr_status": artifact["ocr_status"],
                    "embedding_model": MODEL_ID,
                    "table_artifact": str(path.resolve()),
                    "table_artifact_sha256": table_hash,
                    "table_model": tables["model"],
                    "table_status": table["status"],
                    "method": table["method"],
                    "region_id": region_id,
                    "page_number": page_number,
                    "bbox": bbox,
                    "row": row_number,
                    "column": column_number,
                    "rowspan": cell["rowspan"],
                    "colspan": cell["colspan"],
                    "text": value,
                    "ocr_refs": refs,
                    "embedding_text": input_text,
                }
                cells.append({"id": cell_id, "payload": payload})
    return cells


def index_chunk_artifact(
    chunks_path: str | Path,
    *,
    client: Any | None = None,
    model: Any | None = None,
    collection_name: str | None = None,
    tables_path: str | Path | None = None,
) -> dict[str, Any]:
    """Index child and optional table-cell vectors, then replace old versions."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointIdsList, PointStruct

    artifact = load_chunk_artifact(chunks_path)
    nodes = artifact["nodes"]
    children = [node for node in nodes if node["level"] == 2]
    if not children:
        raise ValueError("chunks.json contains no child chunks to index")
    sections = {section["id"]: section for section in artifact["sections"]}
    for child in children:
        parent = next((node for node in nodes if node["id"] == child["parent_id"]), None)
        if parent is None or parent["level"] != 1:
            raise ValueError("Each child must reference an immediate level-1 parent")
    cells = _table_cells(tables_path, artifact) if tables_path is not None else []
    collection = collection_name or DEFAULT_COLLECTION
    logger.info(
        "Index started: collection=%s chunks=%s child_vectors=%d table_cells=%d",
        collection, chunks_path, len(children), len(cells),
    )
    if client is None:
        url = os.environ.get("QDRANT_URL")
        if not url:
            raise ValueError("QDRANT_URL is required")
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    _ensure_collection(client, collection)
    old_ids = _source_point_ids(client, collection, artifact["source_id"])
    logger.info("Existing points for source: %d", len(old_ids))

    if model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", MODEL_ID)
        model = SentenceTransformer(MODEL_ID)
    inputs = [embedding_text(child, sections[child["section_id"]]) for child in children]
    inputs.extend(cell["payload"]["embedding_text"] for cell in cells)
    _validate_embedding_inputs(model, inputs)
    logger.info("Encoding %d child chunks and %d table cells", len(children), len(cells))
    encoded = model.encode(inputs)
    if len(encoded) != len(inputs):
        raise ValueError("Embedding model returned the wrong number of vectors")
    vectors: dict[str, list[float]] = {}
    for point_id, vector in zip(
        [child["id"] for child in children] + [cell["id"] for cell in cells],
        encoded, strict=True,
    ):
        values = [float(value) for value in vector]
        if len(values) != VECTOR_SIZE or not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding model returned an invalid 1024-dimensional vector")
        vectors[point_id] = values

    points = []
    for node in nodes:
        section = sections[node["section_id"]]
        payload = {
            "schema_version": artifact["schema_version"],
            "record_type": "chunk",
            "source": artifact["source"],
            "source_id": artifact["source_id"],
            "ocr_path": artifact["ocr_path"],
            "ocr_sha256": artifact["ocr_sha256"],
            "ocr_model": artifact["ocr_model"],
            "ocr_status": artifact["ocr_status"],
            "embedding_model": MODEL_ID,
            "section_id": node["section_id"],
            "section_heading": section["heading"],
            "level": node["level"],
            "parent_id": node["parent_id"],
            "text": node["text"],
            "page_numbers": node["page_numbers"],
            "region_ids": node["region_ids"],
            "tables": node["tables"],
        }
        if node["level"] == 2:
            payload["embedding_text"] = embedding_text(node, section)
        points.append(PointStruct(
            id=node["id"],
            vector={VECTOR_NAME: vectors[node["id"]]} if node["level"] == 2 else {},
            payload=payload,
        ))
    for cell in cells:
        points.append(PointStruct(
            id=cell["id"], vector={VECTOR_NAME: vectors[cell["id"]]},
            payload=cell["payload"],
        ))
    for start in range(0, len(points), POINT_BATCH_SIZE):
        logger.info("Upserting points %d-%d of %d", start + 1,
                    min(start + POINT_BATCH_SIZE, len(points)), len(points))
        client.upsert(
            collection_name=collection,
            points=points[start:start + POINT_BATCH_SIZE],
            wait=True,
        )
    new_ids = {str(point.id) for point in points}
    points_by_id = {str(point.id): point for point in points}
    for start in range(0, len(points), POINT_BATCH_SIZE):
        logger.info("Verifying points %d-%d of %d", start + 1,
                    min(start + POINT_BATCH_SIZE, len(points)), len(points))
        batch_ids = [point.id for point in points[start:start + POINT_BATCH_SIZE]]
        retrieved = client.retrieve(
            collection_name=collection, ids=batch_ids, with_payload=True, with_vectors=True
        )
        if {str(record.id) for record in retrieved} != {str(point_id) for point_id in batch_ids}:
            raise RuntimeError("Qdrant did not retain every uploaded point")
        for record in retrieved:
            expected = points_by_id[str(record.id)]
            expected_payload = expected.payload
            payload = record.payload or {}
            if expected_payload["record_type"] == "chunk":
                if (
                    payload.get("ocr_sha256") != artifact["ocr_sha256"]
                    or payload.get("source_id") != artifact["source_id"]
                    or payload.get("level") != expected_payload["level"]
                    or payload.get("parent_id") != expected_payload["parent_id"]
                ):
                    raise RuntimeError("Qdrant returned an unexpected chunk after upload")
            elif any(
                payload.get(field) != expected_payload[field]
                for field in (
                    "record_type", "source_id", "ocr_sha256", "table_artifact_sha256",
                    "region_id", "page_number", "row", "column", "text",
                )
            ):
                raise RuntimeError("Qdrant returned an unexpected table cell after upload")
            if expected.vector:
                vector = record.vector
                if not isinstance(vector, dict) or len(vector.get(VECTOR_NAME, [])) != VECTOR_SIZE:
                    kind = "table cell" if expected_payload["record_type"] == "table_cell" else "child"
                    raise RuntimeError(f"Qdrant did not retain a {kind} vector")
            elif record.vector:
                raise RuntimeError("Qdrant ancestor unexpectedly has a vector")
    stale = sorted(old_ids - new_ids)
    for start in range(0, len(stale), POINT_BATCH_SIZE):
        logger.info("Removing stale points %d-%d of %d", start + 1,
                    min(start + POINT_BATCH_SIZE, len(stale)), len(stale))
        client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=stale[start:start + POINT_BATCH_SIZE]),
            wait=True,
        )
    logger.info("Index completed: nodes=%d child_vectors=%d table_cells=%d replaced=%d",
                len(nodes), len(children), len(cells), len(stale))
    return {
        "collection": collection, "nodes": len(nodes), "children": len(children),
        "table_cells": len(cells), "replaced": len(stale),
    }


def get_parent(client: Any, collection_name: str, child_id: str) -> dict[str, Any]:
    """Retrieve the immediate parent payload for one indexed child point."""
    child_records = client.retrieve(
        collection_name=collection_name, ids=[child_id], with_payload=True, with_vectors=False
    )
    if len(child_records) != 1 or child_records[0].payload.get("level") != 2:
        raise ValueError("Child point was not found")
    parent_id = child_records[0].payload.get("parent_id")
    parent_records = client.retrieve(
        collection_name=collection_name, ids=[parent_id], with_payload=True, with_vectors=False
    )
    if len(parent_records) != 1 or parent_records[0].payload.get("level") != 1:
        raise ValueError("Immediate parent point was not found")
    return {"id": str(parent_records[0].id), **parent_records[0].payload}

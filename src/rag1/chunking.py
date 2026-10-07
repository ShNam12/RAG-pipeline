"""Build section-aware parent and child chunks from structured OCR."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from rag1.extractions.layouts.contracts import RegionKind, RegionRole, VisualLocation
from rag1.extractions.ocr_text.contracts import OcrDocument, OcrRegion


CHUNK_SCHEMA_VERSION = "1.1"
SECTION_LABELS = {"title", "document_title", "doc_title", "paragraph_title", "section_header"}
EXCLUDED_ROLES = {RegionRole.HEADER, RegionRole.FOOTER, RegionRole.PAGE_NUMBER}


def _region_text(region: OcrRegion) -> str:
    if region.proposal.kind is RegionKind.TABLE:
        if region.blocks:
            return "\n".join(block.content for block in region.blocks)
        return "\n".join(line.text for line in region.lines if line.text is not None)
    if region.proposal.kind is RegionKind.TEXT:
        return region.text or ""
    return ""


def sections_from_ocr(document: OcrDocument) -> list[dict[str, Any]]:
    """Group OCR regions by detected headings while preserving source order."""
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for region in document.regions:
        proposal = region.proposal
        if proposal.role in EXCLUDED_ROLES or region.status in {"failed", "skipped"}:
            continue
        content = _region_text(region)
        if not content.strip():
            continue
        heading = (
            proposal.kind is RegionKind.TEXT
            and proposal.raw_label.strip().lower() in SECTION_LABELS
        )
        if heading and current is not None and current["text"]:
            sections.append(current)
            current = None
        if current is None:
            current = {
                "heading": content.strip() if heading else None,
                "text": "",
                "page_numbers": [],
                "region_ids": [],
                "spans": [],
            }
        start = len(current["text"]) + (2 if current["text"] else 0)
        current["text"] += ("\n\n" if current["text"] else "") + content
        location = proposal.location
        page_number = location.page_number if isinstance(location, VisualLocation) else None
        current["region_ids"].append(proposal.id)
        if page_number is not None and page_number not in current["page_numbers"]:
            current["page_numbers"].append(page_number)
        current["spans"].append({
            "start": start,
            "end": start + len(content),
            "region_id": proposal.id,
            "page_number": page_number,
            "kind": proposal.kind.value,
            "table": region.table.model_dump(mode="json") if region.table is not None else None,
        })
    if current is not None and current["text"]:
        sections.append(current)
    return sections


def _find_span(haystack: str, needle: str, start: int) -> tuple[int, int]:
    position = haystack.find(needle, start)
    if position >= 0:
        return position, position + len(needle)
    words = needle.split()
    if words:
        pattern = r"\s+".join(re.escape(word) for word in words)
        match = re.search(pattern, haystack[start:])
        if match is not None:
            return start + match.start(), start + match.end()
    raise ValueError("LlamaIndex chunk text could not be located in its source section")


def _node_spans(
    parsed_nodes: list[Any], section_text: str, parent_key: Any, content_mode: Any
) -> dict[str, tuple[int, int, int, str | None]]:
    by_id = {node.node_id: node for node in parsed_nodes}
    spans: dict[str, tuple[int, int, int, str | None]] = {}
    cursors: dict[str | None, int] = {}

    def locate(node_id: str) -> tuple[int, int, int, str | None]:
        if node_id in spans:
            return spans[node_id]
        node = by_id[node_id]
        relationship = node.relationships.get(parent_key)
        parent_id = relationship.node_id if relationship is not None else None
        if parent_id is None:
            parent_start, parent_end, level = 0, len(section_text), 0
        else:
            if parent_id not in by_id:
                raise ValueError("LlamaIndex returned a child without its parent")
            parent_start, parent_end, parent_level, _ = locate(parent_id)
            level = parent_level + 1
        source = section_text[parent_start:parent_end]
        node_text = node.get_content(metadata_mode=content_mode)
        local_start, local_end = _find_span(source, node_text, cursors.get(parent_id, 0))
        cursors[parent_id] = local_start + 1
        result = (parent_start + local_start, parent_start + local_end, level, parent_id)
        spans[node_id] = result
        return result

    for node in parsed_nodes:
        locate(node.node_id)
    return spans


def build_chunk_artifact(ocr_path: str | Path) -> dict[str, Any]:
    """Parse an OCR file into LlamaIndex's default three-level hierarchy."""
    from llama_index.core import Document
    from llama_index.core.node_parser import HierarchicalNodeParser
    from llama_index.core.schema import MetadataMode, NodeRelationship

    path = Path(ocr_path)
    raw = path.read_bytes()
    document = OcrDocument.model_validate_json(raw)
    sections = sections_from_ocr(document)
    if not sections:
        raise ValueError("OCR JSON contains no usable text")
    source_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag1:source:{document.source.casefold()}"))
    input_hash = hashlib.sha256(raw).hexdigest()
    parser = HierarchicalNodeParser.from_defaults()
    nodes: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections):
        section_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"rag1:section:{source_id}:{input_hash}:{section_index}"
        ))
        section["id"] = section_id
        parsed = parser.get_nodes_from_documents([
            Document(text=section["text"], id_=section_id)
        ])
        spans = _node_spans(parsed, section["text"], NodeRelationship.PARENT, MetadataMode.NONE)
        new_ids = {
            node.node_id: str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rag1:node:{section_id}:{spans[node.node_id][2]}:{node_index}",
            ))
            for node_index, node in enumerate(parsed)
        }
        for node in parsed:
            start, end, level, old_parent_id = spans[node.node_id]
            if level not in {0, 1, 2}:
                raise ValueError(f"Unexpected LlamaIndex hierarchy level: {level}")
            overlapping = [
                span for span in section["spans"]
                if span["start"] < end and span["end"] > start
            ]
            if not overlapping:
                raise ValueError("Chunk has no matching OCR region")
            nodes.append({
                "id": new_ids[node.node_id],
                "section_id": section_id,
                "parent_id": new_ids[old_parent_id] if old_parent_id else None,
                "level": level,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "start": start,
                "end": end,
                "region_ids": list(dict.fromkeys(span["region_id"] for span in overlapping)),
                "page_numbers": list(dict.fromkeys(
                    span["page_number"] for span in overlapping
                    if span["page_number"] is not None
                )),
                "tables": [
                    {"region_id": span["region_id"], "table": span["table"]}
                    for span in overlapping if span["table"] is not None
                ],
            })
    return {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "source": document.source,
        "source_id": source_id,
        "ocr_path": str(path.resolve()),
        "ocr_sha256": input_hash,
        "ocr_model": document.model,
        "ocr_status": document.status.value,
        "sections": sections,
        "nodes": nodes,
    }


def write_chunks(ocr_path: str | Path) -> Path:
    """Write a complete chunks.json beside its source OCR JSON."""
    path = Path(ocr_path)
    artifact = build_chunk_artifact(path)
    output = path.with_name("chunks.json")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output.parent,
        prefix=".chunks-", suffix=".json", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(artifact, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output)
    return output


def load_chunk_artifact(path: str | Path) -> dict[str, Any]:
    """Validate the fields needed to index a generated chunk artifact."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("schema_version") != CHUNK_SCHEMA_VERSION:
        raise ValueError("Unsupported chunks.json schema version")
    metadata_fields = {"source", "source_id", "ocr_path", "ocr_sha256", "ocr_model", "ocr_status"}
    if not all(isinstance(artifact.get(field), str) for field in metadata_fields):
        raise ValueError("chunks.json is missing source identity")
    sections = artifact.get("sections")
    nodes = artifact.get("nodes")
    if not isinstance(sections, list) or not isinstance(nodes, list):
        raise ValueError("chunks.json must contain sections and nodes")
    section_fields = {"id", "heading", "text"}
    node_fields = {
        "id", "section_id", "parent_id", "level", "text", "page_numbers",
        "region_ids", "tables",
    }
    if any(not isinstance(section, dict) or not section_fields <= section.keys() for section in sections):
        raise ValueError("chunks.json section is missing required fields")
    if any(not isinstance(node, dict) or not node_fields <= node.keys() for node in nodes):
        raise ValueError("chunks.json node is missing required fields")
    section_ids = {section["id"] for section in sections}
    node_ids = {node["id"] for node in nodes}
    if len(node_ids) != len(nodes) or len(section_ids) != len(sections):
        raise ValueError("chunks.json contains duplicate IDs")
    for node in nodes:
        if node["section_id"] not in section_ids:
            raise ValueError("Chunk references an unknown section")
        if node["parent_id"] is not None and node["parent_id"] not in node_ids:
            raise ValueError("Chunk references an unknown parent")
        if node["level"] not in {0, 1, 2}:
            raise ValueError("Chunk has an unsupported level")
    return artifact

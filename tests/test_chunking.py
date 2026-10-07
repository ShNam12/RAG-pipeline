"""Section boundaries and chunk provenance for structured OCR."""

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from rag1.chunking import build_chunk_artifact, sections_from_ocr
from rag1.extractions.layouts.contracts import Region, RegionKind, RegionRole, VisualLocation
from rag1.extractions.ocr_text.contracts import (
    OcrBlock,
    OcrCrop,
    OcrDocument,
    OcrError,
    OcrRegion,
    TableMetadata,
)


def region(
    index: int,
    text: str | None,
    *,
    label: str = "text",
    role: RegionRole = RegionRole.BODY,
    kind: RegionKind = RegionKind.TEXT,
    page: int = 1,
) -> OcrRegion:
    proposal = Region(
        id=f"p{page:04d}-r{index:04d}",
        kind=kind,
        role=role,
        raw_label=label,
        location=VisualLocation(
            page_number=page,
            bbox=[0, 0, 100, 50],
            page_width=100,
            page_height=100,
        ),
    )
    crop = OcrCrop(
        proposal_bbox=[0, 0, 100, 50],
        page_bbox=[0, 0, 100, 50],
        page_origin=[0, 0],
        width=100,
        height=50,
        path=f"crops/region-{index}.png",
    )
    if kind is RegionKind.TABLE:
        return OcrRegion(
            proposal=proposal,
            status="complete",
            crop=crop,
            blocks=[
                OcrBlock(
                    label="table",
                    content=text or "",
                    region_bbox=[0, 0, 100, 50],
                    page_bbox=[0, 0, 100, 50],
                )
            ],
            table=TableMetadata(),
        )
    return OcrRegion(proposal=proposal, status="complete", crop=crop, text=text)


def document(regions: list[OcrRegion], *, direct: bool = False) -> OcrDocument:
    return OcrDocument(
        input_mode="direct" if direct else "layout",
        source="report.pdf" if not direct else "page.png",
        source_format="image" if direct else "pdf",
        upstream_schema_version=None if direct else "1.0",
        upstream_status=None if direct else "complete",
        status="complete",
        regions=regions,
    )


class SectionTests(unittest.TestCase):
    def test_titles_start_sections_and_captions_stay_as_content(self) -> None:
        source = document([
            region(1, "Preface"),
            region(2, "Financial position", label="paragraph_title", role=RegionRole.TITLE),
            region(3, "Assets"),
            region(4, "Table 1", label="table_title", role=RegionRole.TITLE),
            region(5, "Cash | 100", kind=RegionKind.TABLE),
            region(6, "Page 1", label="number", role=RegionRole.PAGE_NUMBER),
            region(7, "Notes", label="section_header", role=RegionRole.TITLE, page=2),
            region(8, "Explanation", page=2),
        ])

        sections = sections_from_ocr(source)

        self.assertEqual([section["heading"] for section in sections], [None, "Financial position", "Notes"])
        self.assertEqual(sections[0]["text"], "Preface")
        self.assertEqual(sections[1]["text"], "Financial position\n\nAssets\n\nTable 1\n\nCash | 100")
        self.assertEqual(sections[1]["region_ids"], [f"p0001-r{i:04d}" for i in range(2, 6)])
        table_span = next(span for span in sections[1]["spans"] if span["kind"] == "table")
        self.assertEqual(table_span["table"], {"structure_status": "pending", "structure": None})
        self.assertEqual(sections[2]["page_numbers"], [2])

    def test_table_regions_keep_nested_contract_without_inventing_rows(self) -> None:
        source = document([
            region(1, "Assets", label="paragraph_title", role=RegionRole.TITLE),
            region(2, "Cash | 100", kind=RegionKind.TABLE),
            region(3, "Debt | 200", kind=RegionKind.TABLE),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.json"
            path.write_text(source.model_dump_json(), encoding="utf-8")
            artifact = build_chunk_artifact(path)

        schema_path = Path(__file__).resolve().parents[1] / "schemas/qdrant/table.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        fixture_path = Path(__file__).resolve().parents[1] / "fixtures/qdrant/table-pending.json"
        pending = json.loads(fixture_path.read_text(encoding="utf-8"))["points"][0]["payload"]["table"]
        structured_path = Path(__file__).resolve().parents[1] / "fixtures/qdrant/table-structured.example.json"
        structured = json.loads(structured_path.read_text(encoding="utf-8"))["points"][0]["payload"]["table"]
        validator = Draft202012Validator(schema)
        validator.validate(pending)
        validator.validate(structured)

        section = artifact["sections"][0]
        table_spans = [span for span in section["spans"] if span["kind"] == "table"]
        self.assertEqual(len(table_spans), 2)
        for span in table_spans:
            self.assertEqual(span["table"], pending)
            validator.validate(span["table"])
        root = next(node for node in artifact["nodes"] if node["level"] == 0)
        self.assertEqual(
            root["tables"],
            [{"region_id": span["region_id"], "table": pending} for span in table_spans],
        )
        for node in artifact["nodes"]:
            self.assertEqual(
                [item["region_id"] for item in node["tables"]],
                [span["region_id"] for span in table_spans if span["region_id"] in node["region_ids"]],
            )

    def test_direct_ocr_has_one_preamble_section(self) -> None:
        source = document([
            region(1, "First page", label="full_page"),
            region(2, "Second page", label="full_page", page=2),
        ], direct=True)

        sections = sections_from_ocr(source)

        self.assertEqual(len(sections), 1)
        self.assertIsNone(sections[0]["heading"])
        self.assertEqual(sections[0]["page_numbers"], [1, 2])
        self.assertEqual(sections[0]["text"], "First page\n\nSecond page")

    def test_empty_and_failed_regions_do_not_create_sections(self) -> None:
        failed = region(2, None, label="paragraph_title", role=RegionRole.TITLE)
        failed.status = "failed"
        source = document([region(1, "Body"), failed])

        sections = sections_from_ocr(source)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["text"], "Body")

    def test_partial_ocr_keeps_available_text(self) -> None:
        source = document([region(1, "Available"), region(2, None)])
        source.status = "partial"
        source.errors = [OcrError(
            stage="recognition", page_number=1,
            exception_type="RuntimeError", message="unavailable",
        )]

        sections = sections_from_ocr(source)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["text"], "Available")

    def test_empty_ocr_does_not_write_chunks(self) -> None:
        source = document([region(1, None)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.json"
            path.write_text(source.model_dump_json(), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no usable text"):
                build_chunk_artifact(path)

    def test_default_hierarchy_keeps_parent_links_and_source_regions(self) -> None:
        source = document([
            region(1, "Assets", label="paragraph_title", role=RegionRole.TITLE),
            region(2, "Cash and deposits are available."),
            region(3, "Further financial notes are included.", page=2),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.json"
            path.write_text(source.model_dump_json(), encoding="utf-8")
            first = build_chunk_artifact(path)
            second = build_chunk_artifact(path)

        self.assertEqual([node["id"] for node in first["nodes"]],
                         [node["id"] for node in second["nodes"]])
        nodes = {node["id"]: node for node in first["nodes"]}
        self.assertTrue(any(node["level"] == 2 for node in nodes.values()))
        for node in nodes.values():
            if node["level"] == 2:
                self.assertEqual(nodes[node["parent_id"]]["level"], 1)
            self.assertTrue(node["region_ids"])
            self.assertEqual(node["text"].strip(), node["text"])


if __name__ == "__main__":
    unittest.main()

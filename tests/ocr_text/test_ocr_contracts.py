"""Contracts for OCR results linked to layout proposals."""

import json
import math
import unicodedata
import unittest

from rag1.extractions.layouts.contracts import (
    PageFailure,
    Region,
    RegionKind,
    RegionRole,
    VisualLocation,
)
from rag1.extractions.ocr_text.contracts import (
    OcrCrop,
    OcrDocument,
    OcrError,
    OcrLine,
    OcrRegion,
    TableMetadata,
)

TEXT = "  Doanh thu qu\u00fd III\t1.250.000\nT\u0103ng tr\u01b0\u1edfng  "


def proposal(region_id, kind=RegionKind.TEXT, confidence=0.94):
    return Region(
        id=region_id,
        kind=kind,
        role=RegionRole.BODY,
        raw_label=kind.value,
        confidence=confidence,
        location=VisualLocation(
            page_number=1,
            bbox=[10.25, 20.75, 90.5, 120.25],
            page_width=100,
            page_height=140,
        ),
    )


def crop(path=None):
    return OcrCrop(
        proposal_bbox=[10.25, 20.75, 90.5, 120.25],
        page_bbox=[10, 20, 91, 121],
        page_origin=[10, 20],
        width=81,
        height=101,
        path=path,
    )


def line(region_id, *, text=TEXT, detection=0.87, recognition=0.61,
         line_id=None, region_quad=None, page_quad=None):
    return OcrLine(
        id=line_id or f"{region_id}-line-0001",
        region_id=region_id,
        detector_index=0,
        text=text,
        detection_confidence=detection,
        recognition_confidence=recognition,
        region_quad=region_quad or [[1, 2], [11, 2], [11, 12], [1, 12]],
        page_quad=page_quad or [[11, 22], [21, 22], [21, 32], [11, 32]],
    )


def text_region(source_region=None, *, lines=None, text=TEXT):
    source_region = source_region or proposal("p0001-r0001")
    return OcrRegion(
        proposal=source_region,
        status="complete",
        crop=crop(),
        lines=[line(source_region.id)] if lines is None else lines,
        text=text,
    )


def table_region(*, source_region=None, table_crop=None, lines=None, status="complete"):
    source_region = source_region or proposal("p0001-r0002", RegionKind.TABLE)
    return OcrRegion(
        proposal=source_region,
        status=status,
        crop=crop("tables/region-0002.png") if table_crop is None else table_crop,
        lines=[line(source_region.id)] if lines is None else lines,
        text=None,
        table=TableMetadata(structure_status="pending", structure=None),
    )


def document(*, regions=None, errors=None, upstream_status="complete",
             upstream_errors=None, status="complete"):
    return OcrDocument(
        schema_version="1.0",
        source="quarterly-report.pdf",
        source_format="pdf",
        upstream_schema_version="1.0",
        upstream_status=upstream_status,
        upstream_errors=[] if upstream_errors is None else upstream_errors,
        status=status,
        regions=[text_region(), table_region()] if regions is None else regions,
        errors=[] if errors is None else errors,
    )


class OcrProposalContractTests(unittest.TestCase):
    def test_enrichment_preserves_proposals_and_their_order(self):
        text = proposal("p0001-r0001")
        table = proposal("p0001-r0002", RegionKind.TABLE)
        originals = [text.model_dump(mode="python"), table.model_dump(mode="python")]

        result = document(regions=[text_region(text), table_region(source_region=table)])

        self.assertEqual([region.proposal.model_dump(mode="python") for region in result.regions],
                         originals)
        self.assertEqual([region.proposal.id for region in result.regions],
                         ["p0001-r0001", "p0001-r0002"])
        self.assertEqual(text.model_dump(mode="python"), originals[0])
        self.assertEqual(table.model_dump(mode="python"), originals[1])

    def test_layout_detection_and_recognition_scores_remain_distinct(self):
        result = document()

        self.assertEqual(result.regions[0].proposal.confidence, 0.94)
        self.assertEqual(result.regions[0].lines[0].detection_confidence, 0.87)
        self.assertEqual(result.regions[0].lines[0].recognition_confidence, 0.61)

    def test_scores_accept_zero_and_one_and_reject_invalid_values(self):
        for score in (0.0, 1.0):
            with self.subTest(score=score):
                item = line("p0001-r0001", detection=score, recognition=score)
                self.assertEqual(item.detection_confidence, score)
                self.assertEqual(item.recognition_confidence, score)
                self.assertEqual(proposal("valid", confidence=score).confidence, score)

        for score in (-0.01, 1.01, math.nan, math.inf, -math.inf):
            with self.subTest(score=score):
                for field in ("detection", "recognition"):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        line("p0001-r0001", **{field: score})
                with self.assertRaises(ValueError):
                    proposal("invalid", confidence=score)


class OcrGeometryContractTests(unittest.TestCase):
    def test_fractional_proposal_and_integer_crop_geometry_are_preserved(self):
        result = document()
        region = result.regions[0]

        self.assertEqual(region.proposal.location.bbox, [10.25, 20.75, 90.5, 120.25])
        self.assertEqual(region.crop.proposal_bbox, [10.25, 20.75, 90.5, 120.25])
        self.assertEqual(region.crop.page_bbox, [10, 20, 91, 121])
        self.assertEqual(region.crop.page_origin, [10, 20])
        self.assertEqual((region.crop.width, region.crop.height), (81, 101))

    def test_page_quad_is_crop_quad_translated_by_crop_origin(self):
        item = line("p0001-r0001")

        expected = [[x + 10, y + 20] for x, y in item.region_quad]
        self.assertEqual(item.page_quad, expected)

    def test_quadrilaterals_reject_wrong_vertex_count_nonfinite_and_zero_area(self):
        invalid = (
            [[1, 2], [11, 2], [11, 12]],
            [[1, 2], [11, 2], [11, 12], [1, 12], [0, 0]],
            [[1, 2], [math.inf, 2], [11, 12], [1, 12]],
            [[1, 2], [5, 2], [9, 2], [1, 2]],
        )
        for quad in invalid:
            with self.subTest(quad=quad), self.assertRaises(ValueError):
                line("p0001-r0001", region_quad=quad)

    def test_crop_rejects_inconsistent_origin_or_dimensions(self):
        with self.assertRaises(ValueError):
            OcrCrop(proposal_bbox=[10.25, 20.75, 90.5, 120.25],
                    page_bbox=[10, 20, 91, 121], page_origin=[11, 20],
                    width=81, height=101, path=None)
        with self.assertRaises(ValueError):
            OcrCrop(proposal_bbox=[10.25, 20.75, 90.5, 120.25],
                    page_bbox=[10, 20, 91, 121], page_origin=[10, 20],
                    width=80, height=101, path=None)

    def test_crop_bounds_must_floor_starts_and_ceil_ends(self):
        with self.assertRaises(ValueError):
            OcrCrop(
                proposal_bbox=[10.25, 20.75, 90.5, 120.25],
                page_bbox=[10, 20, 90, 120],
                page_origin=[10, 20],
                width=80,
                height=100,
                path=None,
            )

    def test_line_quad_must_fit_crop_and_page_coordinates_must_match(self):
        out_of_crop = line(
            "p0001-r0001",
            region_quad=[[1, 2], [101, 2], [101, 12], [1, 12]],
            page_quad=[[11, 22], [111, 22], [111, 32], [11, 32]],
        )
        with self.assertRaises(ValueError):
            text_region(lines=[out_of_crop])

        wrong_offset = line(
            "p0001-r0001",
            page_quad=[[12, 22], [22, 22], [22, 32], [12, 32]],
        )
        with self.assertRaises(ValueError):
            text_region(lines=[wrong_offset])


class OcrTableContractTests(unittest.TestCase):
    def test_table_keeps_crop_lines_and_pending_structure_without_flat_text(self):
        result = table_region()

        self.assertEqual(result.proposal.kind, RegionKind.TABLE)
        self.assertEqual(result.crop.path, "tables/region-0002.png")
        self.assertEqual((result.crop.width, result.crop.height), (81, 101))
        self.assertEqual(result.crop.page_origin, [10, 20])
        self.assertEqual(result.table.structure_status, "pending")
        self.assertIsNone(result.table.structure)
        self.assertIsNone(result.text)
        self.assertEqual(result.lines[0].region_id, result.proposal.id)

    def test_failed_table_crop_retains_proposal_and_pending_marker(self):
        source_region = proposal("p0001-r0002", RegionKind.TABLE)
        error = OcrError(
            stage="crop", page_number=1, region_id=source_region.id,
            exception_type="OSError", message="table crop failed",
        )
        result = document(
            regions=[table_region(source_region=source_region,
                                  table_crop=crop(None), lines=[], status="failed")],
            errors=[error],
            status="failed",
        ).regions[0]

        self.assertEqual(result.proposal, source_region)
        self.assertIsNone(result.crop.path)
        self.assertEqual(result.table.structure_status, "pending")
        self.assertIsNone(result.table.structure)


class OcrTextAndSerializationTests(unittest.TestCase):
    def test_text_preserves_accents_whitespace_numbers_and_empty_strings(self):
        line_text = line("p0001-r0001")
        decomposed = unicodedata.normalize("NFD", TEXT)
        decomposed_line = line("p0001-r0001", text=decomposed,
                               line_id="p0001-r0001-line-0002")
        empty_line = line("p0001-r0001", text="",
                          line_id="p0001-r0001-line-0003")
        result = text_region(lines=[line_text, decomposed_line, empty_line], text=TEXT)

        self.assertEqual(result.text, TEXT)
        self.assertEqual(line_text.text, TEXT)
        self.assertEqual(decomposed_line.text, decomposed)
        serialized = json.loads(result.model_dump_json())
        self.assertEqual(serialized["lines"][0]["text"], TEXT)
        self.assertEqual(serialized["lines"][1]["text"], decomposed)
        self.assertEqual(serialized["lines"][2]["text"], "")
        self.assertNotEqual(decomposed_line.text, TEXT)
        self.assertEqual(empty_line.text, "")
        self.assertIn("\t", line_text.text)
        self.assertIn("\n", line_text.text)
        self.assertIn("1.250.000", line_text.text)
        self.assertEqual(line_text.text[:2], "  ")

    def test_json_round_trip_preserves_metadata_errors_geometry_scores_and_nulls(self):
        source_region = proposal("p0001-r0002", RegionKind.TABLE)
        upstream_error = PageFailure(page_number=2, exception_type="OSError",
                                     message="layout page failed")
        error = OcrError(
            stage="recognition", page_number=1, region_id=source_region.id,
            exception_type="RuntimeError", message="recognizer failed",
        )
        result = document(
            regions=[text_region(), table_region(source_region=source_region)],
            errors=[error],
            upstream_status="partial",
            upstream_errors=[upstream_error],
            status="partial",
        )

        restored = OcrDocument.model_validate_json(result.model_dump_json())

        self.assertEqual(restored, result)
        self.assertEqual(restored.schema_version, "1.0")
        self.assertEqual(restored.upstream_schema_version, "1.0")
        self.assertEqual(restored.source, "quarterly-report.pdf")
        self.assertEqual(restored.upstream_status, "partial")
        self.assertEqual([r.proposal.id for r in restored.regions],
                         ["p0001-r0001", "p0001-r0002"])
        self.assertEqual(restored.regions[0].text, TEXT)
        self.assertEqual(restored.regions[0].lines[0].region_quad,
                         [[1, 2], [11, 2], [11, 12], [1, 12]])
        self.assertEqual(restored.regions[0].lines[0].page_quad,
                         [[11, 22], [21, 22], [21, 32], [11, 32]])
        self.assertEqual(restored.regions[0].lines[0].detection_confidence, 0.87)
        self.assertEqual(restored.regions[0].lines[0].recognition_confidence, 0.61)
        self.assertEqual(restored.regions[1].crop.path, "tables/region-0002.png")
        self.assertIsNone(restored.regions[1].text)
        self.assertIsNone(restored.regions[1].table.structure)
        self.assertEqual(restored.errors[0].stage, "recognition")
        self.assertEqual(restored.upstream_errors[0].page_number, 2)

    def test_document_rejects_duplicate_region_and_line_ids(self):
        duplicate_proposal = proposal("p0001-r0001")
        with self.assertRaises(ValueError):
            document(regions=[text_region(duplicate_proposal),
                              text_region(duplicate_proposal)])

        first = proposal("p0001-r0001")
        second = proposal("p0001-r0002")
        duplicate_line = "shared-line-id"
        with self.assertRaises(ValueError):
            document(regions=[
                text_region(first, lines=[line(first.id, line_id=duplicate_line)]),
                text_region(second, lines=[line(second.id, line_id=duplicate_line)]),
            ])

    def test_region_rejects_lines_linked_to_a_different_proposal(self):
        with self.assertRaises(ValueError):
            text_region(proposal("p0001-r0001"),
                        lines=[line("p0001-r0099")])


if __name__ == "__main__":
    unittest.main()

"""Contract tests for the shared layout proposal types."""

import unittest

from rag1.extractions.layouts.contracts import (
    LayoutDocument,
    PageFailure,
    Region,
    RegionKind,
    RegionRole,
    StructuralLocation,
    VisualLocation,
    normalize_detector_label,
)


class LayoutLocationContractTests(unittest.TestCase):
    def test_visual_location_uses_page_pixels_and_top_left_origin(self) -> None:
        location = VisualLocation(
            page_number=2,
            bbox=[10, 20, 90, 120],
            page_width=100,
            page_height=130,
        )

        self.assertEqual(location.page_number, 2)
        self.assertEqual(location.bbox, [10, 20, 90, 120])
        self.assertEqual(location.coordinate_origin, "top_left")

    def test_visual_location_rejects_invalid_boxes(self) -> None:
        invalid_boxes = (
            [10, 20, 10, 120],
            [10, 20, 90, 120, 130],
            [-1, 20, 90, 120],
            [10, 20, 101, 120],
        )

        for bbox in invalid_boxes:
            with self.subTest(bbox=bbox):
                with self.assertRaises(ValueError):
                    VisualLocation(
                        page_number=1,
                        bbox=bbox,
                        page_width=100,
                        page_height=130,
                    )

    def test_structural_location_has_reference_without_visual_coordinates(self) -> None:
        location = StructuralLocation(
            source_format="docx",
            item_ref="#/tables/0",
        )

        self.assertEqual(location.source_format, "docx")
        self.assertEqual(location.item_ref, "#/tables/0")
        self.assertIsNone(getattr(location, "bbox", None))
        self.assertIsNone(getattr(location, "page_number", None))

    def test_structural_location_requires_a_nonblank_reference(self) -> None:
        with self.assertRaises(ValueError):
            StructuralLocation(source_format="docx", item_ref=" ")


class LayoutLabelContractTests(unittest.TestCase):
    def test_detector_labels_map_to_shared_kind_and_role(self) -> None:
        cases = {
            "text": (RegionKind.TEXT, RegionRole.BODY),
            "table": (RegionKind.TABLE, RegionRole.BODY),
            "image": (RegionKind.PICTURE, RegionRole.BODY),
            "chart": (RegionKind.PICTURE, RegionRole.BODY),
            "header": (RegionKind.TEXT, RegionRole.HEADER),
            "footer": (RegionKind.TEXT, RegionRole.FOOTER),
            "page_number": (RegionKind.TEXT, RegionRole.PAGE_NUMBER),
            "paragraph_title": (RegionKind.TEXT, RegionRole.TITLE),
            "unrecognized_label": (RegionKind.OTHER, RegionRole.BODY),
        }

        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(normalize_detector_label(label), expected)

    def test_region_keeps_raw_label_and_confidence(self) -> None:
        region = Region(
            id="p0002-r0001",
            kind=RegionKind.PICTURE,
            role=RegionRole.BODY,
            raw_label="chart",
            confidence=0.91,
            location=VisualLocation(
                page_number=2,
                bbox=[10, 20, 90, 120],
                page_width=100,
                page_height=130,
            ),
        )

        self.assertEqual(region.raw_label, "chart")
        self.assertEqual(region.confidence, 0.91)


class LayoutSerializationAndFailureContractTests(unittest.TestCase):
    def test_document_rejects_structural_reference_from_another_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match the document source_format"):
            LayoutDocument(
                source="report.pdf",
                source_format="pdf",
                status="complete",
                regions=[
                    Region(
                        id="r0001",
                        kind=RegionKind.TABLE,
                        raw_label="table",
                        location=StructuralLocation(
                            source_format="docx",
                            item_ref="#/tables/0",
                        ),
                    )
                ],
            )

    def test_document_json_round_trip_preserves_both_location_types(self) -> None:
        document = LayoutDocument(
            schema_version="1.0",
            source="report.docx",
            source_format="docx",
            status="complete",
            regions=[
                Region(
                    id="r0001",
                    kind=RegionKind.TABLE,
                    role=RegionRole.BODY,
                    raw_label="table",
                    location=StructuralLocation(
                        source_format="docx",
                        item_ref="#/tables/0",
                    ),
                ),
                Region(
                    id="p0001-r0001",
                    kind=RegionKind.TEXT,
                    role=RegionRole.HEADER,
                    raw_label="header",
                    location=VisualLocation(
                        page_number=1,
                        bbox=[0, 0, 100, 20],
                        page_width=100,
                        page_height=200,
                    ),
                ),
            ],
        )

        restored = LayoutDocument.model_validate_json(document.model_dump_json())

        self.assertEqual(restored.schema_version, "1.0")
        self.assertEqual(len(restored.regions), 2)
        self.assertIsInstance(restored.regions[0].location, StructuralLocation)
        self.assertIsInstance(restored.regions[1].location, VisualLocation)
        self.assertEqual(restored.regions[1].location.bbox, [0, 0, 100, 20])

    def test_page_failures_produce_partial_result_and_preserve_successes(self) -> None:
        successful_region = Region(
            id="p0001-r0001",
            kind=RegionKind.TABLE,
            role=RegionRole.BODY,
            raw_label="table",
            location=VisualLocation(
                page_number=1,
                bbox=[10, 20, 90, 120],
                page_width=100,
                page_height=130,
            ),
        )
        failure = PageFailure.from_exception(
            page_number=2,
            error=RuntimeError("layout inference failed"),
        )
        document = LayoutDocument(
            schema_version="1.0",
            source="report.pdf",
            source_format="pdf",
            status="partial",
            regions=[successful_region],
            errors=[failure],
        )

        self.assertEqual(document.status, "partial")
        self.assertEqual(document.regions, [successful_region])
        self.assertEqual(document.errors[0].page_number, 2)
        self.assertEqual(document.errors[0].exception_type, "RuntimeError")
        self.assertEqual(document.errors[0].message, "layout inference failed")


if __name__ == "__main__":
    unittest.main()

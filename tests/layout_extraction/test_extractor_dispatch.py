"""Contract tests for layout extractor format dispatch."""

import unittest
from pathlib import Path

from rag1.extractions.layouts.contracts import DocumentFormat, LayoutDocument
from rag1.extractions.layouts.extractors import (
    ExtractorNotRegisteredError,
    LayoutExtractor,
    UnsupportedDocumentFormatError,
    detect_document_format,
    extract_layout,
)


class StubExtractor(LayoutExtractor):
    def __init__(self, document_format: DocumentFormat) -> None:
        self._document_format = document_format

    @property
    def document_format(self) -> DocumentFormat:
        return self._document_format

    def extract(self, source: Path) -> LayoutDocument:
        return LayoutDocument(
            source=str(source),
            source_format=self.document_format,
            status="complete",
        )


class LayoutExtractorDispatchTests(unittest.TestCase):
    def test_detects_pdf_and_docx_case_insensitively(self) -> None:
        self.assertEqual(detect_document_format("report.PDF"), DocumentFormat.PDF)
        self.assertEqual(detect_document_format("report.DOCX"), DocumentFormat.DOCX)

    def test_dispatches_to_matching_extractor(self) -> None:
        pdf_extractor = StubExtractor(DocumentFormat.PDF)
        docx_extractor = StubExtractor(DocumentFormat.DOCX)

        result = extract_layout("report.docx", [pdf_extractor, docx_extractor])

        self.assertEqual(result.source_format, DocumentFormat.DOCX)
        self.assertEqual(result.source, "report.docx")

    def test_rejects_unsupported_extension_with_supported_formats(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedDocumentFormatError,
            r"\.xml.*\.pdf, \.docx",
        ):
            detect_document_format("report.xml")

    def test_requires_a_registered_extractor_for_supported_format(self) -> None:
        with self.assertRaisesRegex(ExtractorNotRegisteredError, r"\.pdf"):
            extract_layout("report.pdf", [])


if __name__ == "__main__":
    unittest.main()

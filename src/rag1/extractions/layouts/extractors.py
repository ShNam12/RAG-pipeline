"""Extractor interface and format dispatch for layout proposals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from rag1.extractions.layouts.contracts import DocumentFormat, LayoutDocument


SUPPORTED_EXTENSIONS = (".pdf", ".docx")
_FORMATS_BY_EXTENSION = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
}


class UnsupportedDocumentFormatError(ValueError):
    """Raised when a source extension has no layout format contract."""


class ExtractorNotRegisteredError(LookupError):
    """Raised when a supported format has no configured extractor."""


class LayoutExtractor(ABC):
    """Base interface implemented by one extractor per supported format."""

    @property
    @abstractmethod
    def document_format(self) -> DocumentFormat:
        """Return the document format handled by this extractor."""

    def supports(self, source: str | Path) -> bool:
        """Return whether this extractor handles the source extension."""
        try:
            return detect_document_format(source) is self.document_format
        except UnsupportedDocumentFormatError:
            return False

    @abstractmethod
    def extract(self, source: Path) -> LayoutDocument:
        """Extract layout proposals from a document path."""


def detect_document_format(source: str | Path) -> DocumentFormat:
    """Resolve a PDF or DOCX source from its case-insensitive extension."""
    path = Path(source)
    suffix = path.suffix.lower()
    try:
        return _FORMATS_BY_EXTENSION[suffix]
    except KeyError as error:
        extension = suffix or "<no extension>"
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise UnsupportedDocumentFormatError(
            f"Unsupported layout input format {extension!r}. "
            f"Supported formats are: {supported}."
        ) from error


def dispatch_layout_extractor(
    source: str | Path,
    extractors: Iterable[LayoutExtractor],
) -> LayoutExtractor:
    """Choose the registered extractor matching the source format."""
    path = Path(source)
    document_format = detect_document_format(path)
    matches = [
        extractor
        for extractor in extractors
        if extractor.document_format is document_format
    ]
    if not matches:
        raise ExtractorNotRegisteredError(
            f"No layout extractor is registered for .{document_format.value} input."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple layout extractors are registered for .{document_format.value} input."
        )
    return matches[0]


def extract_layout(
    source: str | Path,
    extractors: Iterable[LayoutExtractor],
) -> LayoutDocument:
    """Dispatch a source to its registered extractor and validate its format."""
    path = Path(source)
    extractor = dispatch_layout_extractor(path, extractors)
    result = extractor.extract(path)
    if result.source_format is not extractor.document_format:
        raise ValueError(
            "Layout extractor returned a document with a mismatched source_format."
        )
    return result

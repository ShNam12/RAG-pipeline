"""Layout proposal contracts and format dispatch."""

from rag1.extractions.layouts.contracts import (
    CURRENT_SCHEMA_VERSION,
    DocumentFormat,
    DocumentStatus,
    LayoutDocument,
    PageFailure,
    Region,
    RegionKind,
    RegionLocation,
    RegionRole,
    StructuralLocation,
    VisualLocation,
    normalize_detector_label,
)
from rag1.extractions.layouts.extractors import (
    ExtractorNotRegisteredError,
    LayoutExtractor,
    UnsupportedDocumentFormatError,
    detect_document_format,
    dispatch_layout_extractor,
    extract_layout,
)
from rag1.extractions.layouts.paddle import (
    PADDLE_LAYOUT_MODEL_NAME,
    PaddleLayoutAdapter,
    RenderedPage,
)
from rag1.extractions.layouts.writer import LayoutArtifactWriter, LayoutArtifacts

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DocumentFormat",
    "DocumentStatus",
    "ExtractorNotRegisteredError",
    "LayoutDocument",
    "LayoutArtifactWriter",
    "LayoutArtifacts",
    "LayoutExtractor",
    "PageFailure",
    "PADDLE_LAYOUT_MODEL_NAME",
    "PaddleLayoutAdapter",
    "Region",
    "RegionKind",
    "RegionLocation",
    "RegionRole",
    "RenderedPage",
    "StructuralLocation",
    "UnsupportedDocumentFormatError",
    "VisualLocation",
    "detect_document_format",
    "dispatch_layout_extractor",
    "extract_layout",
    "normalize_detector_label",
]

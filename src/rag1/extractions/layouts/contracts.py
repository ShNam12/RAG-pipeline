"""Shared, versioned contracts for document layout proposals."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CURRENT_SCHEMA_VERSION = "1.0"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"


class DocumentStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class RegionKind(str, Enum):
    TEXT = "text"
    TABLE = "table"
    PICTURE = "picture"
    OTHER = "other"


class RegionRole(str, Enum):
    BODY = "body"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class VisualLocation(ContractModel):
    type: Literal["visual"] = "visual"
    page_number: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    coordinate_origin: Literal["top_left"] = "top_left"

    @field_validator("bbox")
    @classmethod
    def validate_finite_coordinates(cls, bbox: list[float]) -> list[float]:
        if not all(math.isfinite(coordinate) for coordinate in bbox):
            raise ValueError("bounding box coordinates must be finite")
        return bbox

    @model_validator(mode="after")
    def validate_box_bounds(self) -> VisualLocation:
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0:
            raise ValueError("bounding box coordinates must be non-negative")
        if x0 >= x1 or y0 >= y1:
            raise ValueError("bounding box must have positive width and height")
        if x1 > self.page_width or y1 > self.page_height:
            raise ValueError("bounding box must fit within the page dimensions")
        return self


class StructuralLocation(ContractModel):
    type: Literal["structural"] = "structural"
    source_format: DocumentFormat
    item_ref: str = Field(min_length=1)

    @field_validator("item_ref")
    @classmethod
    def validate_item_reference(cls, item_ref: str) -> str:
        if not item_ref.strip():
            raise ValueError("structural location item_ref must not be blank")
        return item_ref


RegionLocation: TypeAlias = Annotated[
    VisualLocation | StructuralLocation,
    Field(discriminator="type"),
]


class Region(ContractModel):
    id: str = Field(min_length=1)
    kind: RegionKind
    role: RegionRole = RegionRole.BODY
    raw_label: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    location: RegionLocation


class PageFailure(ContractModel):
    page_number: int = Field(ge=1)
    exception_type: str = Field(min_length=1)
    message: str

    @classmethod
    def from_exception(cls, page_number: int, error: Exception) -> PageFailure:
        return cls(
            page_number=page_number,
            exception_type=type(error).__name__,
            message=str(error),
        )


class LayoutDocument(ContractModel):
    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        pattern=r"^\d+\.\d+$",
    )
    source: str = Field(min_length=1)
    source_format: DocumentFormat
    status: DocumentStatus
    regions: list[Region] = Field(default_factory=list)
    errors: list[PageFailure] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> LayoutDocument:
        if not self.errors and self.status is not DocumentStatus.COMPLETE:
            raise ValueError("documents without page errors must have complete status")
        if self.errors and self.regions and self.status is not DocumentStatus.PARTIAL:
            raise ValueError("documents with regions and page errors must have partial status")
        if self.errors and not self.regions and self.status is not DocumentStatus.FAILED:
            raise ValueError("documents with only page errors must have failed status")
        for region in self.regions:
            if (
                isinstance(region.location, StructuralLocation)
                and region.location.source_format is not self.source_format
            ):
                raise ValueError(
                    "structural region references must match the document source_format"
                )
        return self


_DETECTOR_LABELS: dict[str, tuple[RegionKind, RegionRole]] = {
    "text": (RegionKind.TEXT, RegionRole.BODY),
    "table": (RegionKind.TABLE, RegionRole.BODY),
    "document_index": (RegionKind.TABLE, RegionRole.BODY),
    "image": (RegionKind.PICTURE, RegionRole.BODY),
    "picture": (RegionKind.PICTURE, RegionRole.BODY),
    "figure": (RegionKind.PICTURE, RegionRole.BODY),
    "chart": (RegionKind.PICTURE, RegionRole.BODY),
    "seal": (RegionKind.PICTURE, RegionRole.BODY),
    "header_image": (RegionKind.PICTURE, RegionRole.HEADER),
    "footer_image": (RegionKind.PICTURE, RegionRole.FOOTER),
    "header": (RegionKind.TEXT, RegionRole.HEADER),
    "page_header": (RegionKind.TEXT, RegionRole.HEADER),
    "footer": (RegionKind.TEXT, RegionRole.FOOTER),
    "page_footer": (RegionKind.TEXT, RegionRole.FOOTER),
    "page_number": (RegionKind.TEXT, RegionRole.PAGE_NUMBER),
    "number": (RegionKind.TEXT, RegionRole.PAGE_NUMBER),
    "title": (RegionKind.TEXT, RegionRole.TITLE),
    "document_title": (RegionKind.TEXT, RegionRole.TITLE),
    "paragraph_title": (RegionKind.TEXT, RegionRole.TITLE),
    "section_header": (RegionKind.TEXT, RegionRole.TITLE),
    "figure_title": (RegionKind.TEXT, RegionRole.TITLE),
    "table_title": (RegionKind.TEXT, RegionRole.TITLE),
    "table_caption": (RegionKind.TEXT, RegionRole.TITLE),
}


def normalize_detector_label(label: str) -> tuple[RegionKind, RegionRole]:
    """Map a detector label to the stable shared kind and role pair."""
    return _DETECTOR_LABELS.get(
        label.strip().lower(),
        (RegionKind.OTHER, RegionRole.BODY),
    )

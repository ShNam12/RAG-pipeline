"""Validated output contracts for OCR enrichment."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag1.extractions.layouts.contracts import (
    CURRENT_SCHEMA_VERSION,
    DocumentFormat,
    DocumentStatus,
    PageFailure,
    Region,
    RegionKind,
    VisualLocation,
)


class OcrModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class OcrCrop(OcrModel):
    page_bbox: list[float] = Field(min_length=4, max_length=4)
    page_origin: list[float] = Field(min_length=2, max_length=2)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    path: str | None

    @field_validator("page_bbox", "page_origin")
    @classmethod
    def finite_coordinates(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("coordinates must be finite")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> OcrCrop:
        x0, y0, x1, y1 = self.page_bbox
        if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
            raise ValueError("crop bounds must have non-negative origin and positive area")
        if self.page_origin != [x0, y0]:
            raise ValueError("crop origin must match the crop bounds origin")
        if self.width != x1 - x0 or self.height != y1 - y0:
            raise ValueError("crop dimensions must match its page bounds")
        return self


class OcrLine(OcrModel):
    id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    detector_index: int = Field(ge=0)
    text: str
    detection_confidence: float = Field(ge=0, le=1)
    recognition_confidence: float = Field(ge=0, le=1)
    region_quad: list[list[float]] = Field(min_length=4, max_length=4)
    page_quad: list[list[float]] = Field(min_length=4, max_length=4)

    @field_validator("region_quad", "page_quad")
    @classmethod
    def validate_quadrilateral(cls, quad: list[list[float]]) -> list[list[float]]:
        if any(len(vertex) != 2 for vertex in quad):
            raise ValueError("quadrilateral vertices must contain two coordinates")
        if any(not math.isfinite(value) for vertex in quad for value in vertex):
            raise ValueError("quadrilateral coordinates must be finite")
        area = abs(sum(
            quad[index][0] * quad[(index + 1) % 4][1]
            - quad[(index + 1) % 4][0] * quad[index][1]
            for index in range(4)
        )) / 2
        if area <= 0:
            raise ValueError("quadrilateral must have positive area")
        return quad


class TableMetadata(OcrModel):
    structure_status: Literal["pending"] = "pending"
    structure: None = None


class OcrRegion(OcrModel):
    proposal: Region
    status: Literal["complete", "failed"]
    crop: OcrCrop
    lines: list[OcrLine] = Field(default_factory=list)
    text: str | None = None
    table: TableMetadata | None = None

    @model_validator(mode="after")
    def validate_region_content(self) -> OcrRegion:
        if self.proposal.kind is RegionKind.TABLE:
            if self.table is None or self.text is not None:
                raise ValueError("table proposals require pending table metadata and null text")
        elif self.table is not None:
            raise ValueError("table metadata is only valid for table proposals")
        if self.crop.path is None and self.status != "failed":
            raise ValueError("regions without a crop must have failed status")
        if isinstance(self.proposal.location, VisualLocation):
            proposal_box = self.proposal.location.bbox
            crop_box = self.crop.page_bbox
            if (
                crop_box[0] < 0
                or crop_box[1] < 0
                or crop_box[2] > self.proposal.location.page_width
                or crop_box[3] > self.proposal.location.page_height
            ):
                raise ValueError("crop bounds must fit within the page")
            if crop_box[0] > proposal_box[0] or crop_box[1] > proposal_box[1]:
                raise ValueError("crop origin must not exclude the proposal bounds")
            if crop_box[2] < proposal_box[2] or crop_box[3] < proposal_box[3]:
                raise ValueError("crop bounds must contain the proposal bounds")
        for line in self.lines:
            if line.region_id != self.proposal.id:
                raise ValueError("line region_id must match its enclosing proposal")
            if self.crop.path is None:
                raise ValueError("OCR lines require an available crop")
            x0, y0, _, _ = self.crop.page_bbox
            for vertex in line.region_quad:
                if not 0 <= vertex[0] <= self.crop.width or not 0 <= vertex[1] <= self.crop.height:
                    raise ValueError("line quadrilateral must fit within its crop")
            for crop_vertex, page_vertex in zip(line.region_quad, line.page_quad):
                if not math.isclose(page_vertex[0] - crop_vertex[0], x0):
                    raise ValueError("page quadrilateral must match crop-to-page x offset")
                if not math.isclose(page_vertex[1] - crop_vertex[1], y0):
                    raise ValueError("page quadrilateral must match crop-to-page y offset")
                if isinstance(self.proposal.location, VisualLocation) and (
                    page_vertex[0] < 0
                    or page_vertex[1] < 0
                    or page_vertex[0] > self.proposal.location.page_width
                    or page_vertex[1] > self.proposal.location.page_height
                ):
                    raise ValueError("page quadrilateral must fit within page dimensions")
        return self


class OcrError(OcrModel):
    stage: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    region_id: str | None = None
    exception_type: str = Field(min_length=1)
    message: str


class OcrDocument(OcrModel):
    schema_version: Literal[CURRENT_SCHEMA_VERSION] = CURRENT_SCHEMA_VERSION
    source: str = Field(min_length=1)
    source_format: DocumentFormat
    upstream_schema_version: Literal[CURRENT_SCHEMA_VERSION] = CURRENT_SCHEMA_VERSION
    upstream_status: DocumentStatus
    upstream_errors: list[PageFailure] = Field(default_factory=list)
    status: DocumentStatus
    regions: list[OcrRegion] = Field(default_factory=list)
    errors: list[OcrError] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers_and_status(self) -> OcrDocument:
        region_ids = [region.proposal.id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("proposal region IDs must be unique")
        line_ids = [line.id for region in self.regions for line in region.lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("OCR line IDs must be unique")
        if self.errors and self.status is DocumentStatus.COMPLETE:
            raise ValueError("documents with OCR errors cannot have complete status")
        return self

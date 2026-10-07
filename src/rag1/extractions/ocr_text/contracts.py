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
    proposal_bbox: list[float] = Field(min_length=4, max_length=4)
    page_bbox: list[float] = Field(min_length=4, max_length=4)
    page_origin: list[float] = Field(min_length=2, max_length=2)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    path: str | None

    @field_validator("proposal_bbox", "page_bbox", "page_origin")
    @classmethod
    def finite_coordinates(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("coordinates must be finite")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> OcrCrop:
        proposal_x0, proposal_y0, proposal_x1, proposal_y1 = self.proposal_bbox
        x0, y0, x1, y1 = self.page_bbox
        if proposal_x0 < 0 or proposal_y0 < 0 or proposal_x0 >= proposal_x1 or proposal_y0 >= proposal_y1:
            raise ValueError("proposal bounds must have non-negative origin and positive area")
        if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
            raise ValueError("crop bounds must have non-negative origin and positive area")
        expected_bounds = [
            math.floor(proposal_x0),
            math.floor(proposal_y0),
            math.ceil(proposal_x1),
            math.ceil(proposal_y1),
        ]
        if self.page_bbox != expected_bounds:
            raise ValueError("crop bounds must floor the proposal origin and ceil its end")
        if any(not coordinate.is_integer() for coordinate in self.page_bbox):
            raise ValueError("effective crop bounds must be integers")
        if self.page_origin != [x0, y0]:
            raise ValueError("crop origin must match the crop bounds origin")
        if self.width != x1 - x0 or self.height != y1 - y0:
            raise ValueError("crop dimensions must match its page bounds")
        return self


class OcrLine(OcrModel):
    id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    detector_index: int = Field(ge=0)
    crop_path: str | None = None
    text: str | None
    detection_confidence: float = Field(ge=0, le=1)
    recognition_confidence: float | None = Field(default=None, ge=0, le=1)
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

    @model_validator(mode="after")
    def validate_recognition_result(self) -> OcrLine:
        if (self.text is None) != (self.recognition_confidence is None):
            raise ValueError("recognition text and confidence must both be present or null")
        return self


class OcrBlock(OcrModel):
    label: str = Field(min_length=1)
    content: str
    region_bbox: list[float] = Field(min_length=4, max_length=4)
    page_bbox: list[float] = Field(min_length=4, max_length=4)

    @field_validator("region_bbox", "page_bbox")
    @classmethod
    def validate_bbox(cls, bbox: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("block bbox coordinates must be finite")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("block bbox must have non-negative origin and positive area")
        return bbox


class TableMetadata(OcrModel):
    structure_status: Literal["pending"] = "pending"
    structure: None = None


class OcrPageMetadata(OcrModel):
    page_number: int = Field(ge=1)
    rotation_degrees: int | None = None
    pixel_width: int | None = Field(default=None, gt=0)
    pixel_height: int | None = Field(default=None, gt=0)
    image: str | None = None
    bbox_image: str | None = None
    region_ids: list[str] | None = None


class OcrRegion(OcrModel):
    proposal: Region
    status: Literal["complete", "empty", "partial", "failed", "skipped"]
    crop: OcrCrop
    lines: list[OcrLine] = Field(default_factory=list)
    blocks: list[OcrBlock] = Field(default_factory=list)
    text: str | None = None
    table: TableMetadata | None = None

    @model_validator(mode="after")
    def validate_region_content(self) -> OcrRegion:
        if self.status == "skipped":
            if self.proposal.kind in {RegionKind.TEXT, RegionKind.TABLE}:
                raise ValueError("text and table proposals cannot be skipped")
            if self.crop.path is not None or self.lines or self.blocks or self.text is not None:
                raise ValueError("skipped proposals cannot contain OCR output or crop files")
            if self.table is not None:
                raise ValueError("skipped proposals cannot contain table reconstruction metadata")
            return self
        if self.status == "empty" and (self.lines or self.blocks):
            raise ValueError("empty regions cannot contain OCR lines or blocks")
        if self.proposal.kind is RegionKind.TABLE:
            if self.table is None or self.text is not None:
                raise ValueError("table proposals require pending table metadata and null text")
        elif self.table is not None:
            raise ValueError("table metadata is only valid for table proposals")
        if self.crop.path is None and self.status != "failed":
            raise ValueError("regions without a crop must have failed status")
        if isinstance(self.proposal.location, VisualLocation) and self.crop.path is not None:
            proposal_box = self.proposal.location.bbox
            crop_box = self.crop.page_bbox
            if self.crop.proposal_bbox != proposal_box:
                raise ValueError("crop must retain the original proposal bounds")
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
        for block in self.blocks:
            if self.crop.path is None:
                raise ValueError("OCR blocks require an available crop")
            x0, y0, _, _ = self.crop.page_bbox
            bx0, by0, bx1, by1 = block.region_bbox
            if bx1 > self.crop.width or by1 > self.crop.height:
                raise ValueError("block bbox must fit within its crop")
            expected_page_bbox = [bx0 + x0, by0 + y0, bx1 + x0, by1 + y0]
            if any(
                not math.isclose(actual, expected)
                for actual, expected in zip(block.page_bbox, expected_page_bbox, strict=True)
            ):
                raise ValueError("page block bbox must match crop-to-page offset")
        return self


class OcrError(OcrModel):
    stage: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    region_id: str | None = None
    exception_type: str = Field(min_length=1)
    message: str


class OcrDocument(OcrModel):
    schema_version: Literal[CURRENT_SCHEMA_VERSION] = CURRENT_SCHEMA_VERSION
    input_mode: Literal["layout", "direct"] = "layout"
    model: Literal[
        "paddleocr-v6", "pp-ocrv6-medium-rec-vietnamese", "paddleocr-vl"
    ] = "paddleocr-v6"
    source: str = Field(min_length=1)
    source_format: DocumentFormat
    upstream_schema_version: Literal[CURRENT_SCHEMA_VERSION] | None = CURRENT_SCHEMA_VERSION
    upstream_status: DocumentStatus | None
    upstream_errors: list[PageFailure] = Field(default_factory=list)
    dpi: int | None = Field(default=None, gt=0)
    pages: list[OcrPageMetadata] = Field(default_factory=list)
    status: DocumentStatus
    text: str | None = None
    regions: list[OcrRegion] = Field(default_factory=list)
    errors: list[OcrError] = Field(default_factory=list)
    page_failures: list[PageFailure] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers_and_status(self) -> OcrDocument:
        if self.input_mode == "direct":
            if self.source_format is not DocumentFormat.IMAGE:
                raise ValueError("direct OCR source format must be image")
            if self.upstream_schema_version is not None or self.upstream_status is not None or self.upstream_errors:
                raise ValueError("direct OCR cannot reference an upstream layout")
        elif self.upstream_schema_version is None or self.upstream_status is None:
            raise ValueError("layout OCR requires upstream layout metadata")
        region_ids = [region.proposal.id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("proposal region IDs must be unique")
        line_ids = [line.id for region in self.regions for line in region.lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("OCR line IDs must be unique")
        page_numbers = [page.page_number for page in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("OCR page metadata numbers must be unique")
        failure_pages = [failure.page_number for failure in self.page_failures]
        if len(failure_pages) != len(set(failure_pages)):
            raise ValueError("OCR page failures must be unique per page")
        regions_by_id = {region.proposal.id: region for region in self.regions}
        for page in self.pages:
            if page.region_ids is None:
                continue
            if len(page.region_ids) != len(set(page.region_ids)):
                raise ValueError("page metadata region IDs must be unique")
            for region_id in page.region_ids:
                region = regions_by_id.get(region_id)
                if region is None:
                    raise ValueError("page metadata references an unknown region ID")
                if not isinstance(region.proposal.location, VisualLocation):
                    raise ValueError("page metadata region references must be visual")
                if region.proposal.location.page_number != page.page_number:
                    raise ValueError("page metadata region reference has a mismatched page")
        if (self.errors or self.page_failures) and self.status is DocumentStatus.COMPLETE:
            raise ValueError("documents with OCR errors cannot have complete status")
        return self

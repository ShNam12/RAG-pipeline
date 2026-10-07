# Qdrant fixture glossary

This guide explains the JSON fixtures in this directory and how to use them with the [storage schemas](../../schemas/qdrant/README.md).
Each fixture contains one example point derived from the saved financial-report layout and OCR artifacts.
Table data belongs inside `points[].payload.table.structure`, with rows and cells nested under that field.
All five fixtures are awaiting embeddings and require review before use as real document evidence.

## Core terms

| Term | Meaning in these files |
|---|---|
| Schema | A JSON Schema contract that defines allowed fields, types, and some relationships between values |
| Fixture | A saved example used to understand a contract or develop a consumer without rerunning extraction |
| Point | A record identified by `id`, with a `payload` and, after embedding generation, a `vector` |
| Payload | The document content, metadata, geometry, evidence, and nested table data attached to a point |
| Embedding | A numeric representation generated from `payload.text` for later vector search |
| Proposal | A region identified by layout detection, or a generated full-page scope in direct OCR mode |
| OCR scope | The page or crop actually processed by OCR, which can be larger than the selected layout table |
| Provenance | References that identify which source files and extraction records produced the payload |
| Evidence | Original OCR lines, blocks, crop metadata, and confidence scores retained for inspection |
| Table structure | Explicit rows, columns, cells, and spans produced by a future table extraction step |

## Schema files

| Schema | What it validates |
|---|---|
| [fixture.schema.json](../../schemas/qdrant/fixture.schema.json) | The outer fixture envelope and points that do not yet contain vectors |
| [payload.schema.json](../../schemas/qdrant/payload.schema.json) | A text or table record, its metadata, evidence, and provenance |
| [table.schema.json](../../schemas/qdrant/table.schema.json) | The nested table state and populated row/cell representation |
| [point.schema.json](../../schemas/qdrant/point.schema.json) | A point ready for upload under this contract, including its vector |

The schemas use JSON Schema Draft 2020-12 and reference neighboring schema files by relative path.
Keep the four schema files together when resolving those references.
Enable UUID `format` checking in the validator.
Unknown object fields are rejected by these contracts.

## Fixture envelope

The envelope is local fixture metadata around `points`.
It is not the payload stored on an individual point.

| Field | Type or values | Meaning and usage |
|---|---|---|
| `schema_version` | `"1.0"` | Version of the fixture contract |
| `fixture_kind` | `extracted`, `illustrative` | Distinguishes evidence derived from extraction artifacts from a manually authored contract example |
| `description` | String | Explains the source, preparation, and limitations of this fixture |
| `embedding_status` | `"pending"` | Embedding generation has not happened for these points |
| `points` | Non-empty array | Points available for inspection or later preparation |

`extracted` describes the origin of the evidence and does not mean that OCR quality or table structure has been accepted.
An `illustrative` fixture demonstrates a shape and must be excluded from real ingestion.

### Point fields

| Field | Meaning |
|---|---|
| `points[].id` | Deterministic UUID identifying this point |
| `points[].payload` | The record described in the following sections |
| `vector` | Absent from staged fixture points; required by `point.schema.json` after generating a real embedding |

The point schema permits a non-empty dense vector array or a map of named dense vector arrays.
It does not fix the embedding dimension or verify that a vector matches a live collection.
Do not add `vector` to a file still being validated as a staged fixture, because `fixture.schema.json` allows only `id` and `payload` on its points.

## Payload fields

Paths in this section are relative to `points[].payload`.

| Field | Meaning and usage |
|---|---|
| `schema_version` | Version of the storage payload contract, currently `"1.0"` |
| `document_id` | UUID derived from the original document's SHA-256, shared by fixtures from the same document |
| `record_type` | `text` or `table`, matching `proposal.kind` |
| `source` | Identity of the original document, even when OCR input was a rendered page image |
| `page_number` | 1-based source page number, equal to `proposal.location.page_number` |
| `dpi` | Render resolution when known, or null when unavailable; these fixtures use 200 |
| `proposal` | Original selected proposal, preserving its ID, label, confidence, and location |
| `text` | String to use as the input for future embedding generation |
| `text_source` | Explains how `text` was assembled |
| `provenance` | Links to the original layout and OCR records, including their association method |
| `artifacts` | Paths to the clean page image and an available crop for the selected proposal |
| `evidence` | Original OCR crop metadata, lines, blocks, and processing status |
| `table` | Null for text records; table state and optional nested structure for table records |
| `needs_review` | Whether the content or its association still needs review; true in every supplied fixture |

### Source identity

| Field | Meaning |
|---|---|
| `source.path` | Repository-relative path to the original source document |
| `source.format` | `pdf`, `docx`, or `image`; the supplied source is a PDF |
| `source.sha256` | Lowercase SHA-256 of the source file's bytes |

The current payload schema represents visual proposals with page geometry.
Allowing a source format value does not implement extraction for that format or provide a structural DOCX location contract.

### Proposal and location

| Field | Meaning |
|---|---|
| `proposal.id` | ID from layout detection or the generated direct OCR scope |
| `proposal.kind` | `text` or `table` for these storage records |
| `proposal.role` | `body`, `title`, `header`, `footer`, or `page_number` |
| `proposal.raw_label` | Original detector label or `full_page` for a direct OCR scope |
| `proposal.confidence` | Layout detector score in `[0, 1]`, or null when no layout score exists |
| `proposal.location.type` | `"visual"` |
| `proposal.location.page_number` | 1-based page containing the proposal |
| `proposal.location.bbox` | `[x0, y0, x1, y1]` bounding box in rendered page pixels |
| `proposal.location.page_width` | Rendered page width in pixels |
| `proposal.location.page_height` | Rendered page height in pixels |
| `proposal.location.coordinate_origin` | `"top_left"`, so x increases rightward and y increases downward |

Boxes can contain fractional coordinates.
They are not PDF-point coordinates or normalized values between zero and one.
Layout confidence, text detection confidence, and text recognition confidence describe different operations and must remain separate.

### Retrieval text

| `text_source` | How to interpret `text` |
|---|---|
| `ocr_region` | The original assembled text from a selected OCR region |
| `associated_lines` | Non-null OCR line strings selected for a layout table and joined with newlines |
| `table_structure` | Text rendered from a populated table structure; the supplied example uses tabs between cells and newlines between rows |

`associated_lines` does not imply that values have been assigned to rows or columns.
The financial table's `text` is raw OCR evidence, so do not treat its line order as a reliable accounting grid.
Keep the raw evidence unchanged when preparing a separate retrieval representation.

## Provenance and artifact paths

### Layout references

`provenance.layout` can be null when no layout provenance is available.
All supplied fixtures include it.

| Field | Meaning |
|---|---|
| `provenance.layout.path` | Path to the layout JSON |
| `provenance.layout.manifest_path` | Path to the companion manifest containing page metadata |
| `provenance.layout.schema_version` | Upstream layout contract version |
| `provenance.layout.status` | Upstream document processing status: `complete`, `partial`, or `failed` |
| `provenance.layout.proposal_pointer` | JSON Pointer such as `/regions/60` locating the selected layout proposal, or null when there is no matching layout proposal |

### OCR references

| Field | Meaning |
|---|---|
| `provenance.ocr.path` | Path to the OCR JSON used as evidence |
| `provenance.ocr.schema_version` | Upstream OCR contract version |
| `provenance.ocr.status` | OCR document processing status: `complete`, `partial`, or `failed` |
| `provenance.ocr.input_mode` | `layout` for proposal-based OCR or `direct` for full-page OCR |
| `provenance.ocr.model` | Model option recorded in the saved OCR artifact |
| `provenance.ocr.region_id` | ID of the actual OCR scope containing the evidence |
| `provenance.ocr.region_pointer` | JSON Pointer locating that region in the OCR JSON |

JSON Pointer array indices are zero-based and are independent of 1-based source page numbers.
The model recorded in an artifact describes that saved run, regardless of the current extraction configuration.

### Association methods

| `provenance.association` | Meaning |
|---|---|
| `ocr_scope` | The payload represents an original OCR scope, as in both text fixtures |
| `layout_proposal_id` | Layout-mode OCR can join to the layout proposal using the preserved proposal ID |
| `line_center_in_layout_bbox` | Same-page full-page OCR lines are selected by whether their bounding-box centers fall inside the selected layout bbox |

The table fixtures use `line_center_in_layout_bbox` because `p0004-full-page` and `p0004-r0005` identify different scopes.
Compute each line's center from the minimum and maximum x and y coordinates of `page_quad`, including centers on a table bbox edge.
Require matching rendered page dimensions and retain the original OCR line order.
This spatial association is unreviewed and does not perform table reconstruction.

### Image and crop references

| Field | Meaning |
|---|---|
| `artifacts.page_image` | Clean rendered page image used to interpret page coordinates |
| `artifacts.region_crop` | Saved crop for the selected proposal, or null when that crop has not been produced |
| `evidence.ocr_crop.path` | Original crop used by OCR, which can cover the whole page |
| `evidence.lines[].crop_path` | Original saved image of an individual detected OCR line, when available |

`source`, `provenance`, and `artifacts` paths are repository-relative and use `/` separators.
Raw crop paths inside `evidence` are relative to the directory containing `provenance.ocr.path`.
For example, resolve `crops/regions/region-00004.png` against the OCR JSON's parent directory, not against `fixtures/qdrant/`.
The pending table fixtures have null `artifacts.region_crop` because a dedicated table crop has not been produced.
Their `evidence.ocr_crop` still describes the full-page OCR crop.

## OCR evidence fields

### Region status and crop geometry

| Field | Meaning |
|---|---|
| `evidence.ocr_region_status` | Region processing result: `complete`, `empty`, `partial`, `failed`, or `skipped` |
| `evidence.ocr_crop.proposal_bbox` | Original fractional bounds of the OCR scope |
| `evidence.ocr_crop.page_bbox` | Effective integer bounds used to crop the page |
| `evidence.ocr_crop.page_origin` | `[x, y]` offset of that crop's top-left corner on the page |
| `evidence.ocr_crop.width` | Crop width in pixels |
| `evidence.ocr_crop.height` | Crop height in pixels |
| `evidence.ocr_crop.path` | Saved crop path, or null if unavailable |

Effective crop bounds floor the starting coordinates and ceil the ending coordinates.
Crop geometry belongs to the OCR scope and can differ from the selected table proposal's geometry.
An OCR processing status of `complete` does not mean the recognized text is correct.

### Line records

`evidence.lines` contains the original V6 OCR line records.

| Field on each line | Meaning |
|---|---|
| `id` | Stable identifier for the line within its OCR output |
| `region_id` | Parent OCR scope ID, matching `provenance.ocr.region_id` |
| `detector_index` | Original zero-based detector output index, which can differ from the line's position in the ordered array |
| `crop_path` | Path to the saved detected-line crop, or null |
| `text` | Exact recognized string, or null if recognition is unavailable |
| `detection_confidence` | Text detector score in `[0, 1]` |
| `recognition_confidence` | Recognizer score in `[0, 1]`, or null when `text` is null |
| `region_quad` | Four `[x, y]` vertices relative to the OCR crop |
| `page_quad` | Corresponding four vertices in rendered page coordinates |

Add `evidence.ocr_crop.page_origin` to each `region_quad` vertex to obtain its `page_quad` vertex.
For direct full-page OCR, the origin is usually `[0, 0]`, so the two quadrilaterals can be identical.
Preserve Vietnamese accents, whitespace, numeric punctuation, empty strings, and low confidence scores.

### VL block records

`evidence.blocks` contains the original VL block records.

| Field on each block | Meaning |
|---|---|
| `label` | Block category reported by the VL parser |
| `content` | Exact block content, which can be empty or contain Markdown or mathematical notation |
| `region_bbox` | Block bounds relative to the OCR crop |
| `page_bbox` | Corresponding block bounds in page coordinates |

VL blocks do not supply the V6 line IDs, saved line crops, or detection and recognition scores in this contract.
The VL fixture therefore has an empty `lines` array and retains its actual `blocks` array.

## Nested table fields

For text records, `table` is null.
For table records awaiting reconstruction, its value is:

```json
{
  "structure_status": "pending",
  "structure": null
}
```

| `table.structure_status` | Required structure value | Meaning |
|---|---|---|
| `pending` | Null | Table reconstruction has not produced a structure |
| `failed` | Null | Table reconstruction failed without a usable structure |
| `partial` | Populated object | Some structure is available but reconstruction is incomplete |
| `complete` | Populated object | The producer represents a complete structure, subject to semantic checks and content review |

In the illustrative fixture, `complete` describes the authored example shape and does not claim that an extractor reconstructed the source table.
Always read `fixture_kind`, `structure.producer`, and `needs_review` alongside this status.

### Structure, rows, and cells

| Field under `table.structure` | Meaning |
|---|---|
| `producer` | Table extractor identity/version, or `manual_contract_example` in the illustrative fixture |
| `row_count` | Declared number of grid rows, including header rows |
| `column_count` | Declared number of grid columns |
| `header_row_indices` | Zero-based indices of rows serving as headers |
| `rows` | Nested row objects containing their cells |
| `rows[].row_index` | Zero-based grid row position |
| `rows[].cells` | Cells anchored in this row |
| `rows[].cells[].column_index` | Zero-based starting column position |
| `rows[].cells[].rowspan` | Number of rows covered by the cell, at least one |
| `rows[].cells[].colspan` | Number of columns covered by the cell, at least one |
| `rows[].cells[].text` | Exact cell text, an empty string for observed empty content, or null for unavailable content |
| `rows[].cells[].source_line_ids` | References to `evidence.lines[].id`, or an empty array when there is no OCR line evidence |
| `rows[].cells[].page_bbox` | Cell bounds in page pixels, or null when unavailable |

Store a merged cell once at its starting row and column and use its spans to represent the covered positions.
Keep financial amounts and dates as strings so punctuation, precision, and original formatting remain available.
The schema does not contain a separate normalized numeric value field.
In the structured example, all spans are one and all cell bboxes are null because physical cell bounds were not extracted.

Beyond schema validation, consumers must check unique row indices, grid bounds, non-overlapping spans, valid source-line references, and full grid coverage before accepting a complete structure.

## Usage of each fixture

### `text-v6.json`

This fixture represents direct OCR scope `p0003-full-page` on page 3 from the saved `paddleocr-v6` run.
It has three OCR lines, no VL blocks, `text_source: "ocr_region"`, and `table: null`.
Use it as the smallest example when developing text payload loading, embedding preparation, line geometry handling, and separate confidence displays.
Its generated full-page proposal has no corresponding layout proposal pointer.

### `text-vl.json`

This fixture represents direct OCR scope `p0001-full-page` on page 1 from the saved `paddleocr-vl` run.
It has 22 VL blocks, no OCR lines, `text_source: "ocr_region"`, and `table: null`.
Use it to develop consumers that accept block content and geometry without requiring line confidence fields.
Preserve empty block content and the existing formatting in recognized strings.

### `table-pending.json`

This fixture represents layout proposal `p0004-r0005` on page 4 with eight OCR lines associated from `p0004-full-page`.
It has `text_source: "associated_lines"`, pending structure, and no saved dedicated table crop.
Use it to develop pending-table displays, layout-to-OCR provenance handling, and the handoff to future table extraction.
Do not interpret its newline-separated text as already paired names and roles.

### `table-financial-pending.json`

This fixture represents layout proposal `p0005-r0001` on page 5 with 151 associated full-page OCR lines.
It retains financial labels, dates, amounts, polygons, and confidence scores while keeping table structure pending.
Use it as a larger example for financial-text preservation, evidence browsing, and future row/cell reconstruction inputs.
Amounts have not been assigned to accounting rows or date columns.

### `table-structured.example.json`

This fixture manually arranges the eight OCR strings from `table-pending.json` into four rows and two columns, including the header row.
It is marked `fixture_kind: "illustrative"`, uses `producer: "manual_contract_example"`, and has `text_source: "table_structure"`.
Use it to understand `table.structure.rows[].cells[]`, develop a table renderer, and resolve each cell's `source_line_ids` back to OCR evidence.
Its retrieval text uses tabs between cells and newlines between rows.
Exclude it from real ingestion because its authored structure is not extracted or verified table data.

## Reading a fixture in Python

Read a fixture and inspect its payload without loading OCR models:

```python
import json
from pathlib import Path

fixture = json.loads(
    Path("fixtures/qdrant/table-pending.json").read_text(encoding="utf-8")
)
point = fixture["points"][0]
payload = point["payload"]

print(fixture["fixture_kind"])
print(payload["proposal"]["id"])
print(payload["table"]["structure_status"])
print(payload["text"])
```

For a populated table, read nested cells only after checking that structure is available:

```python
table = payload["table"]
if table is not None and table["structure"] is not None:
    for row in table["structure"]["rows"]:
        for cell in row["cells"]:
            print(row["row_index"], cell["column_index"], cell["text"])
```

With `table-pending.json`, the second snippet produces no cell output because structure is null.
Load `table-structured.example.json` instead to inspect the populated example locally.

## IDs and preparation for later ingestion

`document_id` is UUIDv5 using the standard URL namespace and the name `rag1:document:sha256:<source SHA-256>`.
The extracted point ID is UUIDv5 using `document_id` as its namespace and the name `1.0|<provenance.ocr.path>|<record_type>|<proposal.id>`.
Illustrative points prepend `illustrative|` to that name, giving the structured example a different ID from its pending counterpart.
Changing the OCR artifact path changes the point ID.
Populating a real table's structure can retain its point ID when the OCR path and proposal ID stay the same.

For later ingestion under this contract:

1. Load the fixture and validate it against `fixture.schema.json` with its local schema references.
2. Exclude `fixture_kind: "illustrative"` from real ingestion and review the evidence indicated by `needs_review`.
3. Check source references, geometry, table association, and any populated table's grid invariants.
4. Generate real embeddings from `payload.text` using the model and dimensions selected for the collection.
5. Create new upload points containing `id`, `vector`, and `payload`, and validate them against `point.schema.json`.
6. Send those points in the upsert body `{"points": [...]}`, keeping fixture metadata outside the request.

These files define examples and contracts; they do not provide an ingestion command or a completed table extractor.
For the contract overview and additional ingestion notes, see [schemas/qdrant/README.md](../../schemas/qdrant/README.md).

# Qdrant storage schemas and fixtures

These JSON Schema Draft 2020-12 contracts describe a future Qdrant payload for the saved layout and OCR artifacts.
The storage contract has its own `schema_version: "1.0"`; upstream versions remain in `provenance`.
The current contract covers visual text and table proposals from rendered pages.

## Files

| File | Purpose |
|---|---|
| [payload.schema.json](payload.schema.json) | Source identity, proposal geometry, retrieval text, provenance, raw OCR evidence, and nested table data |
| [table.schema.json](table.schema.json) | Pending table state and future rows, cells, and merged-cell spans |
| [point.schema.json](point.schema.json) | A UUID point ID, a real dense vector, and the payload |
| [fixture.schema.json](fixture.schema.json) | Example points before embedding generation |

Qdrant supports UUID point IDs and nested JSON payloads.
The point schema accepts an unnamed dense vector or a map of named dense vectors.
See the official [point documentation](https://qdrant.tech/documentation/concepts/points/) and [payload documentation](https://qdrant.tech/documentation/concepts/payload/) for the database formats.

## Fixtures

The supplied report has 54 rendered pages and 706 layout proposals, including 49 table proposals.
The saved V6 OCR artifact contains 54 direct-mode full-page scopes, and the saved VL artifact contains page 1 only.
These fixtures are representative samples from those artifacts.

| Fixture | Content |
|---|---|
| [text-v6.json](../../fixtures/qdrant/text-v6.json) | Page 3 with its three original OCR lines, polygons, and confidence scores |
| [text-vl.json](../../fixtures/qdrant/text-vl.json) | Page 1 with its original VL blocks and text |
| [table-pending.json](../../fixtures/qdrant/table-pending.json) | Layout table `p0004-r0005`, its eight associated OCR lines, and null structure |
| [table-financial-pending.json](../../fixtures/qdrant/table-financial-pending.json) | Layout table `p0005-r0001`, its 151 associated OCR lines, and null structure |
| [table-structured.example.json](../../fixtures/qdrant/table-structured.example.json) | An illustrative four-row, two-column table manually arranged from the page 4 OCR strings |

The structured example is marked `fixture_kind: "illustrative"` and `producer: "manual_contract_example"`.
Its structure demonstrates the future JSON shape and is not extracted or verified table data.
Exclude illustrative fixtures from real ingestion.
All fixtures have `embedding_status: "pending"` and contain no fabricated vectors.
All payloads have `needs_review: true` because successful extraction does not establish recognition accuracy.

## Payload mapping

- `source` identifies the original PDF using a repository-relative path and SHA-256 of its bytes, even when OCR input was a rendered image.
- `page_number` is 1-based and must equal `proposal.location.page_number`.
- `proposal` preserves the selected layout proposal or direct-mode OCR proposal unchanged.
- `provenance.layout` references the layout JSON and manifest; `proposal_pointer` is null for full-page OCR scopes that have no matching layout ID.
- `provenance.ocr` references the exact OCR JSON region and records the model and input mode from that artifact.
- `artifacts.page_image` references the clean rendered page.
- `artifacts.region_crop` references a saved crop for the selected proposal, or is null when that crop has not been produced.
- `evidence.ocr_crop`, `evidence.lines`, and `evidence.blocks` retain the original OCR geometry, text, paths, and separate confidence values.
- `text` is the embedding input; `text_source` identifies whether it comes from the OCR region, associated lines, or future table structure.
- `table` is null for text records and contains the nested table state for table records.

Paths outside raw `evidence` objects use repository-relative POSIX notation.
The original OCR crop paths inside `evidence` remain relative to the parent directory of `provenance.ocr.path`.
Coordinates are rendered image pixels with a top-left origin, not PDF points.
Vietnamese characters, whitespace, OCR errors, dates, and numeric punctuation remain unchanged in raw evidence.
VL blocks retain their actual fields and have no invented detection or recognition confidence scores.

### Associating the current table evidence

The saved direct-mode OCR IDs such as `p0004-full-page` do not match layout IDs such as `p0004-r0005`.
The table fixtures therefore use `association: "line_center_in_layout_bbox"`.
On the same page, select a line when the center of its `page_quad` bounding box lies inside the layout table bbox, including its edges.
Require matching page dimensions and retain the OCR line order.
This is an unreviewed spatial association and does not reconstruct table structure.

For these table fixtures, `artifacts.region_crop` is null because a separate table crop has not been extracted.
The full-page OCR crop remains available under `evidence.ocr_crop` and must not be mistaken for a table crop.
Their retrieval `text` joins the selected non-null OCR strings with newlines; the upstream OCR artifact remains unchanged.

## Future nested table data

Current table records contain:

```json
{
  "table": {
    "structure_status": "pending",
    "structure": null
  }
}
```

After table extraction, populate `payload.table.structure` with `producer`, `row_count`, `column_count`, `header_row_indices`, and `rows`.
Each row contains a zero-based `row_index` and nested `cells`.
Each cell has `column_index`, `rowspan`, `colspan`, exact `text`, `source_line_ids`, and an optional page bbox represented as null when unavailable.
Keep cell amounts and dates as strings until a separate, explicit normalization step exists.
Use an empty string for observed empty content and null for unavailable content.
Store a merged cell once at its starting row and column; its span covers the other grid positions.
Rows and cells remain nested inside the point payload.

`pending` and `failed` require null structure.
`complete` and `partial` require a populated structure.
Consumers must additionally check unique row indices, row and column bounds, non-overlapping spans, source-line references, and full grid coverage before accepting `complete`.
The JSON schemas enforce field types and status relationships; these cross-field and reference checks belong to the ingestion consumer.

## IDs and later ingestion

Fixture IDs are deterministic UUIDv5 values.
Compute `document_id` with the standard URL namespace and the name `rag1:document:sha256:<source SHA-256>`.
Compute each point ID with `document_id` as its namespace and the name `1.0|<provenance.ocr.path>|<record_type>|<proposal.id>`.
Illustrative points prepend `illustrative|` to that name so they cannot overwrite the corresponding extracted point.
The OCR path distinguishes saved extraction runs and model variants; relocating that path changes the point ID.
Updating a real table from pending to structured retains its point ID when its OCR path and proposal ID stay the same.

Before uploading, generate a real embedding from `payload.text`, add it as `vector`, and validate the resulting point against `point.schema.json`.
Choose the embedding model, vector name, collection dimension, and distance metric together at that time.
The schema deliberately leaves the embedding dimension unspecified and cannot check agreement with a live collection.
Send only the materialized points in the Qdrant upsert body `{"points": [...]}`.
The fixture envelope is local metadata and is not a Qdrant API request.
Keep rendered images and crops in artifact storage and retain their paths in the payload.

Schemas reference neighboring schema files by relative path and can be validated offline with a local JSON Schema resolver.
Enable `format` checking when validating UUIDs.
Schema validation and fixture preparation do not exercise a live Qdrant server or implement an ingestion command.

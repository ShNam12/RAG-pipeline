# Qdrant storage schemas and fixtures

These JSON Schema Draft 2020-12 contracts cover proposal points, the section chunk hierarchy and reconstructed table cells indexed by `rag1 index`, and a proposed header-keyed extracted table point.
Proposal and table cell payloads use `schema_version: "1.0"`, chunk payloads use `schema_version: "1.1"`, and header-keyed table payloads use `schema_version: "1.2"`.
Upstream proposal artifact versions remain in `provenance`.

## Files

| File | Purpose |
|---|---|
| [payload.schema.json](payload.schema.json) | Source identity, proposal geometry, retrieval text, provenance, raw OCR evidence, and nested table data |
| [chunk.schema.json](chunk.schema.json) | Indexed section chunk payloads with source references, parent links, and nested table metadata |
| [table-cell.schema.json](table-cell.schema.json) | Indexed nonempty reconstructed table cells with source, region, row, column, and OCR references |
| [table-row.schema.json](table-row.schema.json) | Proposed searchable data rows keyed by the verified table headers |
| [table.schema.json](table.schema.json) | Pending table state and future rows, cells, and merged-cell spans |
| [table-keyed.schema.json](table-keyed.schema.json) | One extracted table whose data rows map OCR-aligned header keys to cell text |
| [point.schema.json](point.schema.json) | Proposal vectors, 1024-dimensional chunk child and table cell vectors, and payload-only chunk ancestors and header-keyed tables |
| [hybrid-search-point.schema.json](hybrid-search-point.schema.json) | Target dense and sparse vector shape for searchable chunk children and header-keyed table rows |
| [fixture.schema.json](fixture.schema.json) | Example points before embedding or payload-only upload |
| [RETRIEVAL_HANDOFF.md](RETRIEVAL_HANDOFF.md) | Current versus target point types and how retrieval expands each hit for generation |

Qdrant supports UUID point IDs and nested JSON payloads.
Proposal points accept an unnamed dense vector or a map of named dense vectors.
Chunk children and indexed table cells require the named `text` vector, while chunk ancestors have an empty vector object.
The proposed header-keyed table point also has an empty vector object.
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
| [table-keyed.example.json](../../fixtures/qdrant/table-keyed.example.json) | Actual `p0004-r0005` extracted cell text organized by OCR-aligned header keys; mapping is illustrative and not indexed |
| [table-row.example.json](../../fixtures/qdrant/table-row.example.json) | Three proposed header-keyed searchable rows linked to the complete `p0004-r0005` table point |
| [table-cell.example.json](../../fixtures/qdrant/table-cell.example.json) | Synthetic example of the current optional table-cell indexer's point shape |
| [chunk-hierarchy.example.json](../../fixtures/qdrant/chunk-hierarchy.example.json) | An illustrative root, parent, and child with pending table metadata and immediate parent links |

The structured table, header-keyed table, table-row, table-cell, and chunk hierarchy examples are illustrative.
The table structure demonstrates the future JSON shape and is not extracted or verified table data.
The header-keyed example uses actual `tables.json` cell text, with header keys aligned to the separate OCR lines because the extracted header cell spans both columns.
Exclude illustrative fixtures from real ingestion.
Fixtures contain no fabricated vectors; the header-keyed table has `embedding_status: "not_applicable"` because its proposed point is payload-only.
The proposal payloads have `needs_review: true` because successful extraction does not establish recognition accuracy.
The chunk example preserves raw table OCR as pending structure; it does not claim reconstructed rows.

## Proposal payload mapping

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

## Indexed chunk payload mapping

`rag1 chunk` creates one document per recognized section and writes `chunks.json` with root, parent, and child nodes.
`rag1 index` writes each node as a Qdrant point using its stable UUID as the point ID.
Every payload has `record_type: "chunk"`, the source and OCR identity, a section ID and heading, hierarchy level, parent ID, original node text, page numbers, region IDs, and a `tables` list.
Level 0 roots have a null parent ID and no vector.
Level 1 parents reference a root and have no vector.
Level 2 children reference an immediate level 1 parent, retain `embedding_text`, and carry a named `text` vector with exactly 1024 values.
`embedding_text` prefixes the section heading once; `text` remains the original node text.
Each item in `tables` has a `region_id` and a nested object validated by [table.schema.json](table.schema.json).
Pending OCR tables keep `structure_status: "pending"` and `structure: null`.
Section chunks may span several pages and regions, so they use `page_numbers` and `region_ids` rather than the proposal-level `page_number` and `proposal` fields.
The JSON schema checks point shape; the indexer verifies that child IDs resolve to their immediate parent after upload.

## Indexed reconstructed table cells

Pass `--tables-json` to `rag1 index` to add one vector point for each nonempty cell in the actual reconstructed `tables.json` artifact.
Each point has `record_type: "table_cell"`, the source and OCR identity, table artifact SHA-256, table model and status, region ID, page bbox, row and column coordinates, merged-cell spans, exact cell text, and OCR references.
The embedding input includes table page and region context, available first-row and first-column labels, and the cell text.
Point IDs are UUIDv5 values derived from the source ID, OCR SHA-256, table artifact hash, region ID, row, and column.
The indexer checks that `tables.json` refers to the same source and OCR artifact as `chunks.json` before uploading.
It uploads table cell and chunk points to the same 1024-dimensional named-vector collection and verifies each point before removing an older version of the source.
Blank cells have no vector point, and the OCR chunk hierarchy retains its pending nested table metadata.
Running `rag1 index` without `--tables-json` replaces the source's previous points without table cells.

## Header-keyed extracted table payload

The proposed `table-keyed.schema.json` stores one table point with its source table artifact and SHA-256, page bbox, extraction status, original merged header cell, OCR-aligned header keys, and ordered data records.
Each record has a zero-based source `row_index` and a `values` object whose keys are the exact header strings and whose values are cell text.
`cell_ocr_refs` uses the same keys to retain each value's OCR evidence.
The `headers` object records the source column index and OCR line ID for each key.
For `p0004-r0005`, the table extractor produced a single header cell spanning two columns, while the OCR lines separately contain `Họ tên` and `Chức vụ` at the corresponding column positions.
The example retains both the merged cell and the aligned keys rather than rewriting the extraction artifact.
This format applies only when every data column has a distinct, supported header key.
Blank, duplicate, or ambiguous headers need review before a header-keyed point is formed; the original grid remains available in `tables.json`.
The JSON schema validates field types, while a table ingestion consumer must check that every record's `values` and `cell_ocr_refs` keys equal the header keys, row indices are unique, column indices fit `column_count`, and OCR IDs resolve to the stated OCR artifact.
No command currently reads `tables.json` to create these points.
The proposed table-row points have the same header-to-value maps and link to the full table by `table_point_id`.
They are defined by [table-row.schema.json](table-row.schema.json) and are not currently indexed.

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
For proposal fixtures, compute `document_id` with the standard URL namespace and the name `rag1:document:sha256:<source SHA-256>`.
Compute each proposal point ID with `document_id` as its namespace and the name `1.0|<provenance.ocr.path>|<record_type>|<proposal.id>`.
Illustrative points prepend `illustrative|` to that name so they cannot overwrite the corresponding extracted point.
The OCR path distinguishes saved extraction runs and model variants; relocating that path changes the point ID.
Updating a real table from pending to structured retains its point ID when its OCR path and proposal ID stay the same.
The header-keyed table fixture uses the standard URL namespace with `rag1:table-kv:<table_artifact_sha256>:<table_id>`.
Its point ID changes when the extracted table artifact changes.

For proposal fixtures, generate a real embedding from `payload.text` and add it as `vector` before validating against `point.schema.json`.
The proposal schema leaves embedding dimension unspecified because proposal ingestion is not implemented here.
For chunk fixtures, use `rag1 index` to generate child vectors and upload the hierarchy.
The indexer validates that the collection has only a named `text` vector with 1024 dimensions and Dot distance.
For the header-keyed table fixture, add `"vector": {}` to the point before validating it against `point.schema.json`; the table remains accessible by ID and payload filtering after upload.
Keep the proposed header-keyed table point separate from this indexed table cell format; the current index command does not materialize that proposed point.
Send only the materialized points in the Qdrant upsert body `{"points": [...]}`.
The fixture envelope is local metadata and is not a Qdrant API request.
Keep rendered images and crops in artifact storage and retain their paths in the payload.

Schemas reference neighboring schema files by relative path and can be validated offline with a local JSON Schema resolver.
Enable `format` checking when validating UUIDs.
Schema validation and fixture preparation do not exercise a live Qdrant server.
`rag1 index` implements chunk and reconstructed table cell ingestion, while proposal and header-keyed table ingestion remain outside this command.
Use [RETRIEVAL_HANDOFF.md](RETRIEVAL_HANDOFF.md) for the current point types, proposed dense/sparse shape, and expansion rules needed by retrieval and generation.

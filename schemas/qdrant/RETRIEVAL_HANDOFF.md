# Qdrant retrieval handoff

This document separates the points available to the current indexer from the proposed header-keyed table and hybrid-search points.
The JSON fixtures omit vectors so they do not pretend to contain model output.
Use the neighboring JSON schemas to validate materialized points before relying on them in retrieval or generation.

## Current indexed point types

| Payload | Vector | Retrieval expansion | Fixture |
|---|---|---|---|
| `record_type: "chunk", level: 2` | Named `text`, 1024 values | Fetch `parent_id` to obtain the immediate level-1 chunk | [chunk-hierarchy.example.json](../../fixtures/qdrant/chunk-hierarchy.example.json) |
| `record_type: "chunk", level: 1` | None | May fetch its `parent_id` for the level-0 ancestor | [chunk-hierarchy.example.json](../../fixtures/qdrant/chunk-hierarchy.example.json) |
| `record_type: "chunk", level: 0` | None | Root context | [chunk-hierarchy.example.json](../../fixtures/qdrant/chunk-hierarchy.example.json) |
| `record_type: "table_cell"` | Named `text`, 1024 values | Use `table_artifact` and `region_id` to recover the source table grid | [table-cell.example.json](../../fixtures/qdrant/table-cell.example.json) |

`rag1 index` embeds child chunks and, only when `--tables-json` is supplied, nonempty reconstructed table cells.
The current materialized shapes are validated by [point.schema.json](point.schema.json), with payloads in [chunk.schema.json](chunk.schema.json) and [table-cell.schema.json](table-cell.schema.json).
The actual `rag1_hybrid_v1` collection observed on 2026-10-07 contained 869 chunk points, no table cells, and only the dense named `text` vector.
The collection name does not imply that sparse search is already available.

For the saved Q3 artifacts, `chunks.json.source` names the rendered pages directory while `tables.json.source` names the original PDF.
The current table-cell indexer requires those strings to match, so supplying that Q3 `tables.json` to the current `rag1 index --tables-json` path raises an error before upload.
Do not assume the optional table-cell fixture is present in the live collection.
Do not join text and table points by equality of their current `source` strings.
Use `parent_id` for chunk expansion and `table_point_id` for proposed row-to-table expansion.
A document-wide filter across both point types needs a canonical document ID established during ingestion.

## Agreed header-keyed table target

[table-keyed.example.json](../../fixtures/qdrant/table-keyed.example.json) is the full table point.
Its `records[].values` maps each verified header key to the cell text, and `cell_ocr_refs` retains the corresponding OCR evidence.
The example uses actual `p0004-r0005` cell values from `tables.json`; aligning its two header keys to OCR lines is an illustrative mapping.
It is a payload-only point with `vector: {}` and is defined by [table-keyed.schema.json](table-keyed.schema.json).

[table-row.example.json](../../fixtures/qdrant/table-row.example.json) defines one proposed searchable point per data row.
Each row has the same header-to-value map and `cell_ocr_refs`, plus a `table_point_id` that names the complete table point.
The row IDs are UUIDv5 values using the URL namespace and `rag1:table-row:<table_point_id>:<row_index>`.
The table point ID uses `rag1:table-kv:<table_artifact_sha256>:<table_id>`.
The row payload is defined by [table-row.schema.json](table-row.schema.json).
Neither the table point nor these row points are produced by the current indexer.

After a row hit, retrieve `table_point_id` and give the generator the matched row, the complete ordered `records`, `page_number`, and OCR references.
After a chunk hit, retrieve its immediate `parent_id` and give the generator that parent, its page numbers, and region IDs.
Keep both the exact payload values and the text used for embedding; an embedding string is not the authoritative table structure.

Before materializing table rows, verify that each record's `values` and `cell_ocr_refs` keys equal the parent `headers` keys, row indices are unique, and every OCR reference resolves to the declared OCR artifact.
The schema checks field types but cannot enforce those cross-point and dynamic-key relationships.
Preserve exact financial amounts as strings; use a separate reviewed numeric field for range queries or arithmetic.

## Planned dense and sparse search

[hybrid-search-point.schema.json](hybrid-search-point.schema.json) defines the target searchable point shape for a level-2 chunk or header-keyed table row.
It requires both named vectors: `text` with 1024 values and `lexical` with sparse `indices` and `values`.
The sparse `values` are term weights, not JSON table cell values.
The same sparse encoder and vocabulary must be used for indexing and queries.
Check equal sparse-array lengths, finite weights, and encoder compatibility when building the vectors; JSON Schema alone cannot check all of those conditions.

The current collection has no `lexical` vector and the current indexer does not generate sparse vectors or header-keyed row points.
Materialize and validate a new combined collection before using server-side dense/sparse fusion over text children and table rows.
Qdrant's Query API can prefetch each named vector and fuse the results with reciprocal rank fusion.
Fetch the linked parent point after fusion and deduplicate hits from the same table before assembling the RAG context.
See the [Qdrant hybrid-query documentation](https://qdrant.tech/documentation/search/hybrid-queries/) for the query shape.

## Fixture use

The fixture envelope is local test data, not a Qdrant upsert request.
For current chunk children or table cells, create the named `text` vector with the configured embedding model before upload.
For the proposed hybrid points, provide both real dense and sparse vectors before validating against `hybrid-search-point.schema.json` or uploading.
For payload-only chunk ancestors and the proposed full table point, use `vector: {}`.
No fixture contains invented model vectors.

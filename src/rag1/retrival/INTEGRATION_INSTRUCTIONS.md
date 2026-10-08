# Retrieval integration instructions

Use this brief to update `src/rag1/retrival/` for the Qdrant contracts in `schemas/qdrant/` and the representative payloads in `fixtures/qdrant/`.
The contract details and current versus target status are in [RETRIEVAL_HANDOFF.md](../../../schemas/qdrant/RETRIEVAL_HANDOFF.md).
Preserve the existing chat API behavior while replacing the old Qdrant assumptions.

## Start with the actual storage contract

| Available from `rag1 index` now | Target that requires new ingestion |
|---|---|
| One collection with a 1024-value named `text` vector using Dot distance. | A combined collection with named dense `text` and sparse `lexical` vectors for searchable points. |
| `record_type: "chunk", level: 2` is searchable, and its level-1 and level-0 ancestors are payload-only. | Level-2 chunks and `record_type: "table_row"` points follow [hybrid-search-point.schema.json](../../../schemas/qdrant/hybrid-search-point.schema.json). |
| Passing `--tables-json` adds nonempty `record_type: "table_cell"` dense points to that same collection. | Each table row links by `table_point_id` to a payload-only complete table defined by [table-keyed.schema.json](../../../schemas/qdrant/table-keyed.schema.json). |

The current indexer does not write `lexical` vectors, `table_row` points, or header-keyed table parent points.
The collection name alone cannot establish that hybrid or row retrieval is available.
Inspect the configured collection and a small sample of point payloads before enabling either mode.
The Q3 sample artifacts documented in [RETRIEVAL_HANDOFF.md](../../../schemas/qdrant/RETRIEVAL_HANDOFF.md) have mismatched `chunks.json.source` and `tables.json.source`, so the current table-cell indexer rejects that pairing before upload.
Ask the ingestion owner to align the real source identity before expecting table cells from those artifacts.

## Replace the legacy retrieval assumptions

1. Update `config.py` and `searcher.py` to use the configured collection produced by `rag1 index` rather than assuming independent `text_index` and `table_index` collections.
   The indexer's current default is `rag1_vietnamese_embedding`; verify the collection chosen by the deployment rather than hardcoding a historical collection name.
2. Filter searchable points by payload type: `record_type: "chunk"` with `level: 2`, `record_type: "table_cell"` for current extracted cells, and `record_type: "table_row"` only after rows are materialized.
   The current `search_table()` filter of `level != 2` does not identify a table type.
3. Replace the old payload mapping in `_convert_point_to_candidate()`.
   Chunk pages are in `page_numbers`, table-cell and table-row pages are in `page_number`, chunk text is in `text`, table-cell text is in `text`, and a table row carries header-keyed `values` and `cell_ocr_refs` rather than `table_markdown` or `table_json`.
   A row's source path comes from its fetched table parent; do not infer it from the row payload.
4. Stop using `seed_mock.py` as evidence of the production contract.
   It creates the old two-collection shapes and can delete collections with unexpected dimensions, so it must never run against a real collection.
   Use the checked-in illustrative fixtures for local contract tests, and keep fixture-envelope metadata outside Qdrant upserts.
5. Update both callers of `QdrantSearcher`, `pipeline.py` and `services/rag_service.py`, so the CLI and chat service use the same new search behavior.

## Implement the two retrieval paths

### Current dense collection

Query the named `text` vector with the same `thanhtantran/Vietnamese_Embedding` model used by the indexer.
Search level-2 chunks and, when present, table cells with explicit `record_type` filters.
After a chunk hit, fetch its immediate `parent_id` and use the level-1 parent's `text`, `page_numbers`, and `region_ids` as the generation context.
For a table-cell hit, retain its exact `text`, `row`, `column`, spans, `page_number`, `region_id`, and `ocr_refs`.
Recover the full grid from `table_artifact` only when that artifact is available and its identity matches the hit; otherwise present the cell as a cell and do not invent surrounding rows.
Label this path dense retrieval, not hybrid retrieval.

### Materialized hybrid collection

Coordinate with the ingestion owner before enabling this path.
Ingestion must write real 1024-value `text` vectors and real sparse `lexical` indices and weights for level-2 chunks and header-keyed table rows, plus the payload-only complete table parents.
The current `_ensure_collection()` in `src/rag1/indexing.py` rejects any vector configuration beyond the named `text` vector, so ingestion must be updated before such a collection can be built.
Indexing and querying must use the same sparse encoder and vocabulary; no sparse encoder is specified by the JSON schema, so agree on it before producing or searching these vectors.
Validate searchable points against [hybrid-search-point.schema.json](../../../schemas/qdrant/hybrid-search-point.schema.json) and validate the parent against [table-keyed.schema.json](../../../schemas/qdrant/table-keyed.schema.json).
Run separate dense `text` and sparse `lexical` prefetches in Qdrant's Query API and fuse them with reciprocal rank fusion.
Filter to level-2 chunks and `table_row` points, deduplicate by point ID, and avoid treating the existing text-versus-table RRF in `fusion.py` as dense-versus-sparse fusion.
If the collection lacks `lexical` or the agreed sparse query encoder is unavailable, report that hybrid mode is unavailable instead of silently returning dense results as hybrid results.

After a `table_row` hit, fetch `table_point_id` from the same collection.
Check that `table_id`, `table_artifact_sha256`, `page_number`, and `row_index` identify a record in the fetched table.
Use the matched row's exact `values` and `cell_ocr_refs`, the parent's ordered `records` and `headers`, the parent's `source`, and the declared page for the prompt and citation.
Check that each row's `values` and `cell_ocr_refs` keys equal the parent's header keys and that its OCR references resolve to the declared OCR artifact.
Do not treat `embedding_text` as the authoritative table grid or parse financial strings into numbers.

## Update prompt and citation handling

Map candidates by `record_type` before formatting them.
Populate `SearchCandidate.content` with actual chunk text, cell text, or a header-value rendering of a table row before `Reranker.rerank()` consumes it.
Teach `table_formatter.py` and `prompt_builder.py` to render header-keyed `headers` and ordered `records`, and to show a matched row within its complete table context.
Keep the existing pending-table behavior for OCR evidence that has no verified structure.
Use `page_numbers` for chunk citations and `page_number` for row or cell citations; do not default an unknown page to page 1.
Preserve exact Vietnamese text, accents, whitespace, and numeric punctuation from the payload.
Do not fabricate a table citation when a row's `table_point_id` cannot be resolved.

## Tests and acceptance

Add focused tests with a mocked Qdrant client for current chunk-parent expansion, current table-cell retrieval, row-to-table expansion, dense and sparse query construction, fusion, deduplication, missing parent points, unavailable sparse capability, and citation page selection.
Use [chunk-hierarchy.example.json](../../../fixtures/qdrant/chunk-hierarchy.example.json), [table-cell.example.json](../../../fixtures/qdrant/table-cell.example.json), [table-keyed.example.json](../../../fixtures/qdrant/table-keyed.example.json), and [table-row.example.json](../../../fixtures/qdrant/table-row.example.json) as payload examples.
The fixture envelopes are local test metadata, the table-row and keyed-table examples are illustrative, and none of these files contains model-generated vectors or proves live ingestion.
For a representative row case, the row with `Họ tên: Bà Nguyễn Thị Thu Hương` and `Chức vụ: Trưởng ban kiểm soát` should resolve to table `p0004-r0005` on page 4 with its OCR references and complete parent records.
Keep static schema checks, mocked tests, actual embedding inference, and live Qdrant search results separate in the handoff report.
Hybrid plus extracted-row retrieval is complete only after a validated collection contains both vector names, real row points, and their resolvable table parents, followed by a live query that returns and cites an extracted row.

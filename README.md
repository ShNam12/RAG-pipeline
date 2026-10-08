# rag1

`rag1` extracts layout region proposals from scanned PDF reports for later document and OCR processing.
The current layout command renders PDF pages, runs PaddleOCR layout detection, and saves both a versioned JSON contract and visual debugging images.

## Output structure

Each input PDF gets a deterministic directory under the selected output root.
The directory name uses a sanitized source filename and a short hash of the resolved source path.

```text
data/extraction/<document-name>-<path-hash>/
  manifest.json
  layout.json
  render-manifest.json
  pages/
    page-0001.png
  bbox/
    page-0001_bbox.png
```

`pages/` contains the unannotated images used for region detection and downstream cropping.
`bbox/` contains copies of those images with detected regions outlined and labeled.
Do not feed bbox images to OCR or table parsers because their pixels include debug annotations.
`--image-dir` changes the relative folder used for unannotated images.
The annotated images remain in the `bbox/` folder.
The image directory must stay inside the document output directory and cannot use the reserved `bbox/` name.

## Run the document pipeline

`scripts/run_pipeline.py` runs selected stages in this order: `layout`, `ocr`, `tables`, `chunk`, `index`.
The `layout` stage detects regions and saves clean page images.
The `ocr` stage uses PP-OCRv6 text detection with the local Vietnamese recognition model by default.
PaddleOCR-VL remains available with an explicit OCR model override.
The `tables` stage reconstructs visual data tables with the `tabular` dependency extra.
The `chunk` stage creates section-aware OCR chunks, and `index` embeds both child chunks and nonempty reconstructed table cells before uploading them to Qdrant.

The default OCR stage requires the Vietnamese checkpoint in `.cache/models/pp-ocrv6-medium-rec-vietnamese`; the download command appears in the OCR section below.
Run the full pipeline from a PDF from the repository root:

```powershell
uv run python scripts/run_pipeline.py --source "data/raw/report.pdf" --env-file .env --collection rag1_hybrid_v1
.\make.ps1 pipeline -InputPath "data/raw/report.pdf" -EnvFile .env -Collection rag1_hybrid_v1
make pipeline INPUT="data/raw/report.pdf" PIPELINE_ENV_FILE=.env PIPELINE_COLLECTION=rag1_hybrid_v1
```

Set `QDRANT_URL` in the shell or the selected environment file, and set `QDRANT_API_KEY` there if the server requires it.
The runner passes the environment file to each `uv run` command without printing its contents.
It prints every stage's output to the terminal and writes timestamped commands, model logs, elapsed times, artifact paths, and failures to `logs/pipeline-<timestamp>.log`.
Use `--log-file`, `-LogFile`, or `PIPELINE_LOG_FILE` to append to a chosen log file.
The first run can download layout, OCR, table, and embedding models and may take longer than five minutes.

Use `--stages` for a partial run, with comma-separated stage names in the order shown above.
Supply the artifact from any skipped upstream stage that the selected stages need:

```powershell
uv run python scripts/run_pipeline.py --stages ocr,tables,chunk,index --layout-json "data/extraction/<document-name>-<path-hash>/layout.json" --env-file .env
uv run python scripts/run_pipeline.py --stages index --chunks-json "data/ocr/<document-name>-<path-hash>/chunks.json" --tables-json "data/tables/<document-name>-<path-hash>/tables.json" --env-file .env
make pipeline PIPELINE_STAGES=chunk,index PIPELINE_OCR_JSON="data/ocr/<document-name>-<path-hash>/ocr.json" PIPELINE_ENV_FILE=.env
```

To send original full-page images straight to OCR without running layout proposal detection, set `--direct-ocr` and select `ocr` plus any downstream stages.
Direct OCR accepts one supported image or a directory of clean `page-NNNN.png` images, not a PDF.
Do not include `layout` or `tables` in the selected stages because table reconstruction requires layout proposals.

```powershell
uv run python scripts/run_pipeline.py --direct-ocr --source "data/extraction/<document-name>-<path-hash>/pages" --stages ocr,chunk,index --env-file .env --collection rag1_hybrid_v1
make pipeline INPUT="data/extraction/<document-name>-<path-hash>/pages" PIPELINE_STAGES=ocr,chunk,index PIPELINE_DIRECT_OCR=1 PIPELINE_ENV_FILE=.env PIPELINE_COLLECTION=rag1_hybrid_v1
```

For the direct script, use `--layout-output-dir`, `--layout-image-dir`, `--layout-dpi`, and `--layout-device` to configure region detection.
Use `--ocr-output-dir`, `--ocr-image-dir`, `--ocr-device`, `--ocr-model`, and `--ocr-manifest` for OCR.
Set `--ocr-model paddleocr-vl`, `-OcrModel paddleocr-vl`, or `PIPELINE_OCR_MODEL=paddleocr-vl` to run the full pipeline with PaddleOCR-VL.
Use `--table-output-dir`, `--table-model`, `--table-device`, `--table-manifest`, and `--region-id` for table reconstruction.
Use `--collection` for the Qdrant target.
The PowerShell wrapper uses `-InputPath` for the PDF or image source and PascalCase names for the other options.
Set `-Direct 1` with `-Stages ocr,chunk,index` for direct OCR through `make.ps1`.
Make uses `INPUT` for the PDF or image source and the corresponding `PIPELINE_` variables shown in `Makefile`.
Omitted options retain each existing CLI or YAML default.

The runner stops when a stage fails or reports a partial artifact.
Pass `--allow-partial`, `-AllowPartial true`, or `PIPELINE_ALLOW_PARTIAL=1` to continue with partial layout, OCR, or table output.
If a layout has no visual regions labeled `table`, the table stage is skipped and chunk indexing continues.
Reconstructed cell vectors are indexed only when `tables` runs or an existing `tables.json` is supplied with `--tables-json`.

## OCR enrichment

### Choose an OCR model

`rag1 ocr` defaults to PP-OCRv6 small text detection with the local Vietnamese recognition model for layout pages or direct page images.
Run the default model:

```powershell
uv run rag1 ocr "data/extraction/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/pages/page-0001.png" --direct --device cpu --output-dir data/ocr
```

The first use may download the PP-OCRv6 detection model.
The command writes `ocr.json` and `ocr.md` under the selected output directory.
All OCR paths use the CPU by default.
The Vietnamese recognition model is loaded from `.cache/models/pp-ocrv6-medium-rec-vietnamese`.
The command reports an error when that directory is missing.
From the repository root, download its inference files with the Hugging Face CLI:

```powershell
hf download tieubaoca/pp-ocrv6-medium-rec-vietnamese inference.json inference.pdiparams inference.yml ppocr_keys.txt --local-dir .cache/models/pp-ocrv6-medium-rec-vietnamese
```

To select PaddleOCR-VL document parsing or VietOCR line recognition explicitly:

```powershell
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/pages/page-0001.png" --direct --model paddleocr-vl --device cpu
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/pages/page-0001.png" --direct --model paddleocr-v6 --device cpu
```

There is no separate `vietocr` value for `--model`; select `--model paddleocr-v6` to use VietOCR line recognition.
The first VL run downloads model weights into the PaddleX cache and may take several minutes.

### OCR with layout proposals

Pass a `layout.json` file to OCR its pages using layout proposals:

```powershell
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/layout.json" --device cpu
```

You can also pass an artifact directory containing `layout.json`, `manifest.json`, and clean `pages/` images:

```powershell
uv run rag1 ocr "data/extraction/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391"
```

For a layout file, the command searches its directory, then sibling `pages/`, then sibling `region/` for clean `page-NNNN.png` images.
When given a layout file or artifact directory, it automatically reads the sibling `manifest.json` when present.
When given an artifact directory, it uses that directory's images.
`_bbox.png` debug images are ignored.
When supplied, the manifest source must match `layout.json`, and its page region references must resolve to proposals on the same page.
Available DPI, rotation, dimensions, image paths, and region references are retained in OCR page metadata; missing metadata fields remain null.
For PaddleOCR-VL, the layout path masks table boxes and makes one inference call per page for text, picture, and other proposals.
Recognized blocks are assigned to a containing proposal when possible; blocks outside proposals are retained in an `unassigned_ocr` page region.
Each table crop is still parsed separately so the table reconstruction stage receives its OCR text.
The page-level output and table results are written together to `ocr.json` and `ocr.md` when the run finishes.
PaddleOCR-VL also runs one additional inference on each original, unmasked page and writes its native formatted output to `ocr.formatted.md`.
That file includes the model's headings, tables, formulas, and image references, with images saved under `formatted-images/` beside it.
This extra pass increases runtime by one VL prediction per page and its text can differ from the positioned OCR evidence in `ocr.json` and `ocr.md`.

### OCR directly from page images

Use `--direct` to bypass layout proposals and OCR a clean image file or every `page-NNNN.png` image in a directory:

```powershell
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/pages/page-0001.png" --direct
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/pages" --direct --device cpu
```

Direct mode accepts one PNG, JPEG, TIFF, BMP, or WebP image, or a directory of clean `page-NNNN.png` images.
It ignores `_bbox.png` debug images in directories and does not accept PDF files.
It sends each original full-page image to the selected OCR model without reading `layout.json` or running the separate layout extraction command.
With PaddleOCR-VL, the same direct-mode prediction also supplies `ocr.formatted.md`, so direct mode does not need an additional pass for that artifact.
The OCR output creates one generated full-page scope per image so the existing `ocr.json` fields remain usable.
It records `input_mode: "direct"`, `source_format: "image"`, null upstream layout metadata, and absolute input image paths in page metadata.
Unreadable images are recorded as page failures while other pages continue.
`--manifest` and `--image-dir` cannot be used with `--direct`.

### OCR output and options

In layout mode, `--image-dir` selects another page image directory and `--manifest` supplies an upstream `manifest.json`.
In either mode, `--output-dir` selects the artifact root and defaults to `data/ocr`, `--device` selects the Paddle device, and `--model` selects the OCR path.
Each source gets a deterministic output directory containing `ocr.json`, `ocr.md`, and the crop images referenced by the JSON.
The document-level `text` field in `ocr.json` contains the same ordered text as `ocr.md`.

For each visual proposal, the OCR module floors the left/top and ceils the right/bottom bbox edges to obtain the effective integer crop bounds.
`crop.proposal_bbox` retains the original fractional proposal coordinates, and `crop.page_bbox` records the effective integer bounds used to crop the image.
With `--model paddleocr-v6`, PP-OCRv6_small_det detects text polygons inside that proposal crop, each detected polygon is cropped, and VietOCR recognizes the resulting line image.
Line records retain the detector and recognizer confidence separately, polygon coordinates relative to the proposal crop, corresponding page coordinates, and a relative line crop path.
Text proposal `text` is the recognized lines joined in detector order with newline separators, while every line's recognized text is retained verbatim.
Table proposals retain their image crop and any recognized lines, have `text: null`, and carry `table.structure_status: "pending"` with `table.structure: null` for a later table reconstruction stage.
In layout mode, the VL path parses masked pages for non-table content and separate table crops into ordered `blocks` with a label, content, region-relative bbox, and page-relative bbox.
In direct mode, it parses each original full-page image without layout proposals or table masking.
VL and Vietnamese V6 output record their selected `model` value; legacy `paddleocr-v6` layout JSON retains its existing shape.
VL blocks do not invent detector or recognizer confidence scores or line crop images.
Text proposal `text` joins the VL block content in reading order, and table block content is included in `ocr.md` while table proposal `text` remains null.

The OCR output has `schema_version: "1.0"`, source metadata, optional DPI and page metadata, OCR regions, OCR errors, and image page failures.
Layout mode also records upstream layout status and keeps proposals unchanged so `proposal.id` joins back to `layout.json`.
Direct mode records null upstream layout metadata and generated full-page scopes.
Coordinates use image pixels and a top-left origin.
Crop references are relative to the OCR output directory; direct mode stores absolute source image paths in page metadata.
The layout loader rejects unsupported schema versions and duplicate proposal IDs before inference.

The OCR model adapters initialize lazily and reuse their models during a run.
At INFO level, PaddleOCR-VL logs each prediction's input path, device, pipeline version, elapsed time, result count, and parsed block count.
PaddleOCR downloads model files on cache misses using PaddleX's configured model source and cache.
The default PaddleX model source is Hugging Face; set `PADDLE_PDX_MODEL_SOURCE=BOS` when that source is unreachable.
OCR artifact paths do not control the model cache location.
In layout mode, the module verifies that each clean page image is readable and has the exact pixel dimensions declared by its layout proposals.
Missing, unreadable, or dimension-mismatched images are recorded as page failures; the pipeline does not rescale images or proposal coordinates.
In direct mode, image dimensions come from the source images, and unreadable pages are recorded as page failures.
The OCR module does not qualify Vietnamese recognition accuracy or deskew rotated lines.
Its own table structure field stays pending; `rag1 tables` writes a separate reconstruction artifact from the layout and positioned OCR.

## Table reconstruction

Install the `tabular` extra and pass both the layout and OCR JSON files:

```powershell
uv run --extra tabular rag1 tables "data/layouts/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/layout.json" --ocr-json "data/ocr/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/pp-ocrv6-vietnamese/pages-79e9d28f078f/ocr.json" --region-id p0005-r0001
```

The `region-00005` alias also selects page 5 when that page contains exactly one data table.
The default manifest is `manifest.json` beside the layout file.
`--manifest`, `--output-dir`, `--model`, and `--device` override the manifest path, output root, Hugging Face model ID, and PyTorch device.
The default model is `PaddlePaddle/SLANet_plus_safetensors`, and the default device is `cpu`.
On first use, Transformers downloads the model into the Hugging Face cache; later runs reuse it.
The command crops clean page images using layout coordinates, not OCR region crop images.
It accepts layout-linked OCR or direct full-page OCR with page-relative line coordinates.

Run the same single-table extraction through Make:

```powershell
make tables TABLE_LAYOUT_JSON="data/layouts/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/layout.json" TABLE_OCR_JSON="data/ocr/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/pp-ocrv6-vietnamese/pages-79e9d28f078f/ocr.json" TABLE_REGION_ID=p0005-r0001
```

`TABLE_LAYOUT_JSON` and `TABLE_OCR_JSON` are required for `make tables` only.
Optional Make variables are `TABLE_MANIFEST`, `TABLE_OUTPUT_DIR`, `TABLE_MODEL`, `TABLE_DEVICE`, and `TABLE_REGION_ID`.
Omit `TABLE_REGION_ID` to process all visual regions whose detector label is `table`.
Regions labeled `document_index` are reported as skipped.

The command writes `data/tables/<doc-id>/tables.json`, `tables.html`, and `tables.flat.json` by default.
Each JSON table records its source region ID, page bbox, structure score, reconstruction method, status, warnings, rows of cells with `rowspan` and `colspan`, OCR line references, and unassigned OCR text.
The HTML escapes OCR text before display.
When SLANet columns and OCR alignment disagree, the command uses OCR geometry and marks the table `partial`.
The same status applies when row alignment is uncertain or OCR text cannot be placed.
Review partial tables and Vietnamese text against the source image before using numbers in downstream analysis.

### Flat JSON row export

To flatten an existing `tables.json` or `tables.html` without running SLANet again:

```powershell
uv run rag1 tables-flat "data/tables/<doc-id>/tables.json"
uv run rag1 tables-flat "data/tables/<doc-id>/tables.html"
make tables-flat TABLE_INPUT="data/tables/<doc-id>/tables.html"
```

Pass `--output <path>` or `TABLE_FLAT_OUTPUT=<path>` to choose another output file.
JSON input defaults to `tables.flat.json` beside the source; HTML input defaults to `tables.from-html.json`.
`TABLE_JSON` remains an alias for `TABLE_INPUT` in Make.
The source `tables.json` and `tables.html` remain available for reviewing the original structure.

The JSON-derived `tables.flat.json` has a `tables` metadata list, a `records` list with one object per table row, and an `unassigned_ocr` list.
Every record carries `table_id`, page number, zero-based `row_index`, `record_id`, and the table status.
Columns use stable keys such as `column_1` because some OCR headers are incomplete or span columns.
`column_N` contains whitespace-normalized NFC text or null, while `column_N_raw` retains the original OCR text.
`column_N_number` contains a canonical numeric string when the whole cell is clearly numeric; for example, `451.893.195` becomes `451893195`, and `(1.234)` becomes `-1234`.
Dates, labels, and mixed text have null numeric values.
`column_N_ocr_refs` links back to OCR lines; `column_N_rowspan`, `column_N_colspan`, and `column_N_covered_by` preserve merged-cell positions without repeating their values.
The export retains table warnings and unassigned OCR items for review and does not correct recognition errors or infer semantic column names.
The HTML-derived `tables.from-html.json` is a top-level JSON array of data row objects.
For each table, the first HTML row supplies the field names and is omitted from the data rows.
For example, a header `col1 | col2` followed by `value 1 | value 2` becomes `{ "col1": "value 1", "col2": "value 2" }`.
When the HTML contains several tables, each object also has `_table_id` and `_page_number` to retain its source.
Blank header cells and columns covered by merged headers use `column_N`; repeated names receive `__2`, `__3`, and so on.
Empty data cells become null, fully blank rows are omitted, and text keeps its Vietnamese characters and internal word spaces.
The first row of some report tables is a title or incomplete OCR header, so review the generated field names before downstream use.
Use the JSON-derived export when OCR references, structure scores, warnings, or raw cell positions are needed.

## OCR refinement with OpenAI

`OPENAI_API_KEY` is required for `rag1 refine-ocr` and is listed in `.env.example`.
The command below loads it from `.env` for this run.

```powershell
uv run --env-file .env rag1 refine-ocr
```

Run from the repository root to refine all nonempty OCR Markdown files, or pass one file path:

```powershell
uv run --env-file .env rag1 refine-ocr "data/ocr/<document-name>/<model>/<run>/ocr.md"
```

The default input is `data/ocr`.
For a directory, the command recursively finds every `ocr.md`; for a file, it processes just that file.
Empty `ocr.md` files are skipped during directory discovery.
Each file is sent to the OpenAI Responses API using `gpt-5.4-mini` and writes `ocr.refined.md` beside the source.
The API request sets `store: false` so the response is not retained as application state.
The original `ocr.md` and `ocr.json` stay unchanged.
An existing refined file stops the command before any API requests unless `--overwrite` is supplied.
The optional `--model` flag selects another OpenAI model ID.
The command uses the standard library HTTP client and needs no additional package.
Large documents may take time and incur API charges; the command rejects incomplete model responses rather than saving truncated text.
Review names, Vietnamese accents, numbers, and tables against the source before using the refined text as verified data.

## Section chunks and Qdrant indexing

`rag1 chunk` reads structured `ocr.json` and writes `chunks.json` beside it.
It uses recognized `title`, `document_title`, `doc_title`, `paragraph_title`, and `section_header` regions as section boundaries.
It does not infer headings from OCR text patterns.
Figure and table captions remain in the current section, while page headers, footers, and page numbers are excluded.
Text before the first heading belongs to a preamble section.
The saved Q3 Vietnamese OCR runs use direct full-page scopes with no title labels, so they currently produce one preamble section rather than inferred report sections.

```powershell
uv run rag1 chunk "data/ocr/<document-name>/<model>/<run>/ocr.json"
```

Each section becomes a LlamaIndex document parsed with `HierarchicalNodeParser.from_defaults()`.
The default hierarchy has 2048-token roots, 512-token parents, and 128-token children with 20-token overlap.
`chunks.json` has `schema_version: "1.1"`, the source and OCR input SHA-256, OCR model and status, sections, and nodes.
Re-run `rag1 chunk` before indexing an older `chunks.json` file because the table metadata shape changed from version 1.0.
Each node has a stable UUID, level, parent and section IDs, text, character span, page numbers, region IDs, and a `tables` list.
Section `spans` map character offsets back to OCR regions and retain each table region's nested `table` object.
Each entry in a node's `tables` list has a `region_id` and a `table` object conforming to [table.schema.json](schemas/qdrant/table.schema.json).
This list preserves multiple table regions when a chunk overlaps them.
Current OCR table objects have `structure_status: "pending"` and `structure: null`, as in [table-pending.json](fixtures/qdrant/table-pending.json).
The [structured table fixture](fixtures/qdrant/table-structured.example.json) illustrates the same nested contract, but its manually arranged rows are not used as OCR evidence.
Table chunk text remains raw OCR lines or blocks; no table structure is inferred by this command.
The original `ocr.json` and `ocr.md` are unchanged.

`rag1 index` embeds 128-token child nodes with `thanhtantran/Vietnamese_Embedding`.
Pass `--tables-json` to embed every nonempty reconstructed table cell with its table location, row and column, OCR references, and source provenance.
Table cell vectors are separate Qdrant points, while the OCR chunk hierarchy and its pending nested table metadata remain unchanged.
The input for each vector is its section heading followed by child text, without repeating a heading already at the start of the child.
The original child text is stored separately.
The command uses the Sentence Transformers constructor and `encode` defaults, and rejects any embedding input longer than the model's actual token limit.
The model card lists 2048 input tokens, 1024 output dimensions, and dot-product similarity.
The first run downloads model files to the Hugging Face cache, which is separate from the OCR and Qdrant artifact directories.

Set `QDRANT_URL` for an existing Qdrant server and `QDRANT_API_KEY` when authentication is enabled.
The default collection is `rag1_hybrid_v1`; `--collection` explicitly selects another compatible collection.
The generic `COLLECTION_NAME` environment variable is ignored by this command so settings for other Qdrant workflows do not redirect chunk indexing.
The command creates a named `text` vector with 1024 dimensions and Dot distance, or checks an existing collection for that exact configuration.
Before filtering points by `source_id`, it creates a UUID payload index when missing and waits for it to become available.
An existing `source_id` keyword or UUID index is reused; another index type is rejected.
The Qdrant credential needs permission to create a payload index on the first run, including when the collection already exists.
It stores root and parent nodes as payload-only points, and child nodes and selected table cells with vectors.
Each point payload carries its node's `tables` list and the original region references.
After a successful upload and readback, it removes older points for the same source while retaining every other source.

```powershell
uv run --env-file .env rag1 index "data/ocr/<document-name>/<model>/<run>/chunks.json" --tables-json "data/tables/<doc-id>/tables.json"
```

This stage indexes chunks and provides parent lookup by point ID in Python.
It does not add a query or answer-generation command.
Indexed chunk points follow [chunk.schema.json](schemas/qdrant/chunk.schema.json), and table cell points follow [table-cell.schema.json](schemas/qdrant/table-cell.schema.json).
Both point types follow [point.schema.json](schemas/qdrant/point.schema.json).
Indexing a source without `--tables-json` replaces its previous points and removes any table cell points previously uploaded for that source.
The proposal-level [payload.schema.json](schemas/qdrant/payload.schema.json) remains a separate payload variant for one proposal on one page.
Proposal and chunk payloads share [table.schema.json](schemas/qdrant/table.schema.json) for nested table state; indexed table cells use their own payload schema.

To check whether the Qdrant server configured in `.env` has any points, run the read-only script from the repository root:

```powershell
uv run --env-file .env python scripts/check_qdrant.py
uv run --env-file .env python scripts/check_qdrant.py --collection rag1_hybrid_v1
```

The first command lists exact point counts for all collections; the second checks one collection.
The script reports whether any checked collection has data and does not print the URL, API key, vectors, or payloads.
It reads `QDRANT_URL` and optional `QDRANT_API_KEY` from the process environment and makes no changes to Qdrant.

## CLI

Extraction defaults are loaded from `configs/extraction/layouts.yml` and `configs/extraction/ocr_text.yml`.
Edit those files to change default models, devices, render settings, and artifact roots.
The command wrappers omit optional CLI flags unless you pass an override, so the YAML values take effect.

Run the layout and OCR phases through the root Makefile:

```powershell
make layout INPUT="data/raw/Bao+cao+tai+chinh+hop+nhat+Q3.2025.pdf"
make ocr INPUT="data/extraction/<document-name>-<path-hash>/layout.json"
```

OCR can also run directly on page images or export Markdown from an existing OCR JSON file:

```powershell
make ocr INPUT="data/extraction/<document-name>-<path-hash>/pages" DIRECT=1
make ocr-text INPUT="data/ocr/<document-name>-<path-hash>/ocr.json"
```

`make.ps1` provides the same tasks when invoking PowerShell directly:

```powershell
.\make.ps1 layout -InputPath "data/raw/Bao+cao+tai+chinh+hop+nhat+Q3.2025.pdf"
.\make.ps1 ocr -InputPath "data/extraction/<document-name>-<path-hash>/layout.json"
.\make.ps1 ocr -InputPath "data/extraction/<document-name>-<path-hash>/pages" -Direct true
.\make.ps1 ocr -InputPath "data/extraction/<document-name>-<path-hash>/pages" -Direct true -Model paddleocr-vl
.\make.ps1 ocr-vietnamese -InputPath "data/extraction/<document-name>-<path-hash>/pages" -Direct true -Device cpu
```

Make variables and PowerShell parameters can override YAML defaults when needed.
For example, use `DPI=300 IMAGE_DIR=page-images` with `make layout`, or `MODEL=pp-ocrv6-medium-rec-vietnamese` with `make ocr`.
The Vietnamese Paddle recognizer is the default for `make ocr` and `make pipeline`; `make ocr-vietnamese INPUT="data/extraction/<document-name>-<path-hash>/pages" DIRECT=1` selects the same model explicitly.
Use `MODEL=paddleocr-vl` with `make ocr` or `PIPELINE_OCR_MODEL=paddleocr-vl` with `make pipeline` to select PaddleOCR-VL.
Use `MODEL=paddleocr-v6` with `make ocr` to select VietOCR.
PaddleOCR-VL dependencies are installed with the base package, so use `uv run rag1 ocr ...` directly.

The required `INPUT` or `-InputPath` value is a PDF for `layout`, a layout JSON or image path for `ocr`, and an OCR JSON file for `ocr-text`.
Pass `DIRECT=1` or `-Direct true` for OCR directly on page images.
The `ocr-text` command writes `ocr.md` beside the JSON unless you pass `OUTPUT` or `-OutputPath`.

## JSON contract

`layout.json` is a `LayoutDocument` with `schema_version` currently set to `1.0`.
Its fields are `source`, `source_format`, `status`, `regions`, and `errors`.
The current PDF command emits `source_format: "pdf"`.
`status` is `complete`, `partial`, or `failed`.
`complete` means no page errors were recorded.
`partial` means at least one region was found and at least one page failed.
`failed` means errors were recorded and no regions were found.

Each region has an ID, normalized kind and role, original detector label, optional confidence, and a location.
Kinds are `text`, `table`, `picture`, and `other`.
Roles are `body`, `title`, `header`, `footer`, and `page_number`.
An unknown detector label is retained in `raw_label` and mapped to kind `other` with role `body`.

Visual locations use 1-based page numbers and pixel coordinates in the unannotated rendered page image.
The origin is the top-left corner and `bbox` is `[x0, y0, x1, y1]`.
The coordinates are finite, nonnegative, ordered, and bounded by `page_width` and `page_height`.
They are not PDF points and are not normalized to a zero-to-one range.

```json
{
  "schema_version": "1.0",
  "source": "C:/reports/quarterly-report.pdf",
  "source_format": "pdf",
  "status": "complete",
  "regions": [
    {
      "id": "p0001-r0001",
      "kind": "table",
      "role": "body",
      "raw_label": "table",
      "confidence": 0.94,
      "location": {
        "type": "visual",
        "page_number": 1,
        "bbox": [75.0, 120.0, 960.0, 820.0],
        "page_width": 1653.0,
        "page_height": 2337.0,
        "coordinate_origin": "top_left"
      }
    }
  ],
  "errors": []
}
```

Structural locations are reserved for source formats such as DOCX.
They contain `type: "structural"`, a `source_format`, and a non-empty `item_ref` instead of a visual bbox.
Structural references must match the document format.

Each `errors` entry records a 1-based `page_number`, `exception_type`, and error `message`.
The document validator enforces the status and location invariants when JSON is loaded through the Pydantic contract.

`manifest.json` is the artifact index for this run.
It contains `manifest_version`, source and format, status, DPI, the relative path to `layout.json`, page, region, and error counts, page metadata, and page failures.
Each manifest page has `page_number`, `rotation_degrees`, pixel dimensions, relative `image` and `bbox_image` paths, `region_count`, and `region_ids`.
Manifest image paths use forward slashes and are relative to the document output directory.
`render-manifest.json` is renderer debug metadata; downstream consumers should use `manifest.json` and `layout.json` as the phase outputs.

## Model download behavior

The current adapter initializes PaddleOCR `LayoutDetection` with model name `PP-DocLayout_plus-L` and does not set a local `model_dir`.
On a cache miss, PaddleOCR and PaddleX download the official model during the first layout run and reuse the cached model on later runs.
The current PaddleOCR documentation lists Hugging Face as the default model source and supports switching to BOS with `PADDLE_PDX_MODEL_SOURCE=BOS` when needed.
Set `PADDLE_PDX_CACHE_HOME` before invoking the command to select a different PaddleX cache home.
The extraction `--output-dir` does not control the model cache location.
Model downloads require network access unless the model is already cached or supplied through the PaddleX model cache.

```powershell
$env:PADDLE_PDX_MODEL_SOURCE = "BOS"
$env:PADDLE_PDX_CACHE_HOME = "D:\models\paddlex"
uv run rag1 layout "data/raw/report.pdf" --output-dir data/extraction
```

The project pins the CPU PaddlePaddle wheel for layout and OCR extraction.
Selecting `--device gpu:0` requires a compatible GPU PaddlePaddle installation and driver stack.

## Downstream use

Use `manifest.json` to locate each clean rendered page and `layout.json` to find the proposals for that page.
Group regions by `location.page_number` and use `region.id` as the link key for downstream results from the same run.
Crop from `pages/` using the visual bbox, never from `bbox/`.
Keep page number, region ID, and page-coordinate bbox with every downstream result.

Docling should continue to parse the original PDF or document to produce semantic text, tables, and reading order.
Use the proposals as routing and alignment hints for selecting pages or regions and matching Docling output back to the source page.
Convert between Docling's geometry and this contract's rendered-pixel coordinates before comparing boxes.
This repository does not yet implement a Docling adapter or a conversion between the two coordinate systems.

In layout mode, PP-OCRv6 uses clean page images and layout proposals as OCR work hints.
In direct mode, it detects and recognizes text across each full page image.
The `rag1 ocr` command stores crop-relative and page-relative geometry for its detected lines.
Docling can continue to parse the original PDF for semantic reading order and richer document structures, then use proposal IDs and converted geometry to align its results with the OCR output.
Use `rag1 tables` to route visual table proposals through SLANet and positioned OCR.
Its output retains the proposal ID for joining to the original layout proposal.

## Known limits

The `rag1 layout` command currently processes PDF input only.
DOCX is present in the format contract but has no registered extractor, and XML has no format contract or extractor yet.
The layout model proposes coarse page regions and does not recognize text or produce table rows, columns, cells, `rowspan`, or `colspan`.
The detector's categories and accuracy are model-dependent, so inspect bbox images when tuning report workflows.
The default render resolution is 200 DPI and proposals are tied to those rendered pixel dimensions.
Failures while initializing or running layout inference are recorded per page, but a PDF page rendering failure aborts the rendering run.
Output directories are keyed by resolved source path, so repeated runs on the same path reuse the same document directory.

## Qdrant schemas and fixtures

[Qdrant storage contracts](schemas/qdrant/README.md) define JSON schemas and fixtures for proposal points, indexed chunk hierarchies, indexed reconstructed table cells, and proposed header-keyed table rows.
[Retrieval handoff](schemas/qdrant/RETRIEVAL_HANDOFF.md) identifies which point types are currently indexed, which remain proposed, and how a retriever should expand a text or table hit for generation.
Future table rows and cells live under `payload.table.structure`; the current table fixtures keep structure pending.
The fixtures require real embeddings before upload, and the populated table example is explicitly illustrative.

## References

- [PaddleOCR layout detection documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/layout_detection.html)
- [PaddleOCR-VL usage documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleX 3.7 official model downloader](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/utils/official_models.py)
- [PaddleOCR discussion on configuring the model cache home](https://github.com/PaddlePaddle/PaddleOCR/discussions/16631)
- [Docling document conversion reference](https://docling-project.github.io/docling/reference/document_converter/)
- [Docling document model and provenance types](https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/document.py)

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

## OCR enrichment

### Choose an OCR model

`rag1 ocr` defaults to `--model pp-ocrv6-medium-rec-vietnamese`, which uses PP-OCRv6 small text detection and the local Paddle Vietnamese recognition model at `.cache/models/pp-ocrv6-medium-rec-vietnamese`.
The command reports an error when that directory is missing.
From the repository root, download its inference files with the Hugging Face CLI:

```powershell
hf download tieubaoca/pp-ocrv6-medium-rec-vietnamese inference.json inference.pdiparams inference.yml ppocr_keys.txt --local-dir .cache/models/pp-ocrv6-medium-rec-vietnamese
```

Select `--model paddleocr-vl` for PaddleOCR-VL document parsing.
The model choice works with both layout proposals and direct page image input.
PaddleOCR-VL, V6, and VietOCR use the CPU by default.
Run PaddleOCR-VL with the project's `vl` optional dependencies:

```powershell
uv run --extra vl rag1 ocr "data/layouts/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/pages/page-0001.png" --direct --model paddleocr-vl --device cpu --output-dir data/ocr
```

The first use may download PaddleOCR-VL model weights.
The command writes `ocr.json` and `ocr.md` under the selected output directory.
To switch from PaddleOCR-VL to VietOCR line recognition, select `--model paddleocr-v6`:

```powershell
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/region" --direct --model paddleocr-v6 --device cpu
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/region" --direct --model pp-ocrv6-medium-rec-vietnamese --device cpu
```

There is no separate `vietocr` value for `--model`; select `--model paddleocr-v6` to use VietOCR line recognition.
Install the optional VL dependencies before selecting that model:

```powershell
uv sync --extra vl
```

The first VL run downloads model weights into the PaddleX cache and may take several minutes.

### OCR with layout proposals

Pass a `layout.json` file to OCR its proposed text and table regions:

```powershell
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/layout.json" --device cpu
uv run --extra vl rag1 ocr "data/extraction/<document-name>-<path-hash>/layout.json" --model paddleocr-vl --device cpu
```

You can also pass a `region/` artifact directory containing `layout.json`, `manifest.json`, and clean page images:

```powershell
uv run rag1 ocr "data/extraction/Bao_cao_tai_chinh_hop_nhat_Q3.2025-8763ca14c391/region"
```

For a layout file, the command searches its directory, then sibling `pages/`, then sibling `region/` for clean `page-NNNN.png` images.
When given an artifact directory, it uses that directory's images and automatically reads its `manifest.json` when present.
`_bbox.png` debug images are ignored.
When supplied, the manifest source must match `layout.json`, and its page region references must resolve to proposals on the same page.
Available DPI, rotation, dimensions, image paths, and region references are retained in OCR page metadata; missing metadata fields remain null.

### OCR directly from page images

Use `--direct` to bypass layout proposals and OCR a clean image file or every `page-NNNN.png` image in a directory:

```powershell
uv run rag1 ocr "data/extraction/<document-name>-<path-hash>/region/page-0001.png" --direct
uv run --extra vl rag1 ocr "data/extraction/<document-name>-<path-hash>/region" --direct --model paddleocr-vl --device cpu
```

Direct mode accepts one PNG, JPEG, TIFF, BMP, or WebP image, or a directory of clean `page-NNNN.png` images.
It ignores `_bbox.png` debug images in directories and does not accept PDF files.
It sends each original full-page image to the selected OCR model without reading `layout.json` or running the separate layout extraction command.
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
In the default path, PP-OCRv6_small_det detects text polygons inside that proposal crop, each detected polygon is cropped, and VietOCR recognizes the resulting line image.
Line records retain the detector and recognizer confidence separately, polygon coordinates relative to the proposal crop, corresponding page coordinates, and a relative line crop path.
Text proposal `text` is the recognized lines joined in detector order with newline separators, while every line's recognized text is retained verbatim.
Table proposals retain their image crop and any recognized lines, have `text: null`, and carry `table.structure_status: "pending"` with `table.structure: null` for a later table reconstruction stage.
The VL path parses each proposal crop or full page into ordered `blocks` with a label, content, input-relative bbox, and page-relative bbox.
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
PaddleOCR downloads model files on cache misses using PaddleX's configured model source and cache.
The default PaddleX model source is Hugging Face; set `PADDLE_PDX_MODEL_SOURCE=BOS` when that source is unreachable.
OCR artifact paths do not control the model cache location.
In layout mode, the module verifies that each clean page image is readable and has the exact pixel dimensions declared by its layout proposals.
Missing, unreadable, or dimension-mismatched images are recorded as page failures; the pipeline does not rescale images or proposal coordinates.
In direct mode, image dimensions come from the source images, and unreadable pages are recorded as page failures.
The module does not qualify Vietnamese recognition accuracy, deskew rotated lines, or reconstruct table rows, columns, cells, `rowspan`, or `colspan`.
Treat recognition accuracy as a separate checkpoint qualification gate and table structure as pending until a table reconstruction phase consumes the retained table crop and OCR lines.

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
For example, use `DPI=300 IMAGE_DIR=page-images` with `make layout`, or `MODEL=paddleocr-vl` with `make ocr`.
The Vietnamese Paddle recognizer is the default for `make ocr`; `make ocr-vietnamese INPUT="data/extraction/<document-name>-<path-hash>/pages" DIRECT=1` selects it explicitly.
Use `MODEL=paddleocr-v6` with `make ocr` to select VietOCR.
For OCR-VL, `make.ps1` selects the `vl` optional dependencies when `-Model paddleocr-vl` is set.
When invoking `rag1` directly, use `uv run --extra vl rag1 ocr ... --model paddleocr-vl`.

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
Route table proposals and their crops to a later table structure stage such as SLANet_plus.
Keep the proposal ID on OCR and table outputs so they can be joined to the original layout proposal.

## Known limits

The `rag1 layout` command currently processes PDF input only.
DOCX is present in the format contract but has no registered extractor, and XML has no format contract or extractor yet.
The layout model proposes coarse page regions and does not recognize text or produce table rows, columns, cells, `rowspan`, or `colspan`.
The detector's categories and accuracy are model-dependent, so inspect bbox images when tuning report workflows.
The default render resolution is 200 DPI and proposals are tied to those rendered pixel dimensions.
Failures while initializing or running layout inference are recorded per page, but a PDF page rendering failure aborts the rendering run.
Output directories are keyed by resolved source path, so repeated runs on the same path reuse the same document directory.

## Qdrant schemas and fixtures

[Qdrant storage contracts](schemas/qdrant/README.md) define JSON schemas and fixtures derived from the saved layout and OCR artifacts.
Future table rows and cells live under `payload.table.structure`; the current table fixtures keep structure pending.
The fixtures require real embeddings before upload, and the populated table example is explicitly illustrative.

## References

- [PaddleOCR layout detection documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/layout_detection.html)
- [PaddleOCR-VL usage documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleX 3.7 official model downloader](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/utils/official_models.py)
- [PaddleOCR discussion on configuring the model cache home](https://github.com/PaddlePaddle/PaddleOCR/discussions/16631)
- [Docling document conversion reference](https://docling-project.github.io/docling/reference/document_converter/)
- [Docling document model and provenance types](https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/document.py)

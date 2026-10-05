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

## CLI

Run the layout phase directly with `uv`:

```powershell
uv run rag1 layout "data/raw/Bao+cao+tai+chinh+hop+nhat+Q3.2025.pdf" --output-dir data/extraction --image-dir pages --dpi 200 --device cpu
```

The required positional argument is a PDF path.
`--output-dir` selects the root for document-specific results and defaults to `data/extraction`.
`--image-dir` selects the relative folder for clean page renders and defaults to `pages`.
`--dpi` sets the render resolution and defaults to `200`.
`--device` selects the Paddle device and defaults to `cpu`.
The command reports the extraction status and paths to the manifest, layout JSON, and image folders.
It exits with a nonzero status when layout extraction fails for every page.

The root `Makefile` and `make.ps1` provide the same task:

```powershell
make layout INPUT="data/raw/Bao+cao+tai+chinh+hop+nhat+Q3.2025.pdf" OUTPUT_DIR=data/extraction IMAGE_DIR=pages DPI=200 DEVICE=cpu
```

PowerShell can invoke the script directly:

```powershell
.\make.ps1 layout -InputPath "data/raw/Bao+cao+tai+chinh+hop+nhat+Q3.2025.pdf" -OutputDir data/extraction -ImageDir pages -Dpi 200 -Device cpu
```

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

The project currently pins the CPU PaddlePaddle wheel.
Selecting `--device gpu:0` requires a compatible GPU PaddlePaddle installation and driver stack.
Changing the extraction output directory does not install or select a GPU runtime.

## Downstream use

Use `manifest.json` to locate each clean rendered page and `layout.json` to find the proposals for that page.
Group regions by `location.page_number` and use `region.id` as the link key for downstream results from the same run.
Crop from `pages/` using the visual bbox, never from `bbox/`.
Keep page number, region ID, and page-coordinate bbox with every downstream result.

Docling should continue to parse the original PDF or document to produce semantic text, tables, and reading order.
Use the proposals as routing and alignment hints for selecting pages or regions and matching Docling output back to the source page.
Convert between Docling's geometry and this contract's rendered-pixel coordinates before comparing boxes.
This repository does not yet implement a Docling adapter or a conversion between the two coordinate systems.

PP-OCRv6 should consume clean page images and region proposals as OCR work hints.
Run text detection and recognition on text regions or on the full clean page, then associate OCR boxes with proposals using page number and overlap.
If OCR runs on a cropped region, translate its crop-relative boxes back to page pixels by adding the crop origin.
Route table regions to a table structure stage such as SLANet_plus after cropping.
Keep the region ID on OCR and table outputs so they can be joined to the original proposal.
This repository currently performs layout detection only and does not run PP-OCRv6 or SLANet_plus.

## Known limits

The `rag1 layout` command currently processes PDF input only.
DOCX is present in the format contract but has no registered extractor, and XML has no format contract or extractor yet.
The layout model proposes coarse page regions and does not recognize text or produce table rows, columns, cells, `rowspan`, or `colspan`.
The detector's categories and accuracy are model-dependent, so inspect bbox images when tuning report workflows.
The default render resolution is 200 DPI and proposals are tied to those rendered pixel dimensions.
Failures while initializing or running layout inference are recorded per page, but a PDF page rendering failure aborts the rendering run.
Output directories are keyed by resolved source path, so repeated runs on the same path reuse the same document directory.

## References

- [PaddleOCR layout detection documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/layout_detection.html)
- [PaddleX 3.7 official model downloader](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/utils/official_models.py)
- [PaddleOCR discussion on configuring the model cache home](https://github.com/PaddlePaddle/PaddleOCR/discussions/16631)
- [Docling document conversion reference](https://docling-project.github.io/docling/reference/document_converter/)
- [Docling document model and provenance types](https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/document.py)

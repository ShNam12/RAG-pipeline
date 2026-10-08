# Pipeline Runbook

`scripts/run_pipeline.py` runs the `rag1` document pipeline stages in order: `layout`, `ocr`, `tables`, `chunk`, and `index`.
It invokes the existing `rag1` CLI for each stage, streams stage output to the terminal, and records the commands, output, status, and elapsed time in a log file.

## Run the full pipeline

From the repository root, provide a PDF source.
By default, the runner selects all stages and writes a timestamped log under `logs/`.

```powershell
uv run python scripts/run_pipeline.py --source "data/raw/report.pdf" --env-file .env --collection rag1_hybrid_v1
```

The default OCR model comes from `configs/extraction/ocr_text.yml`.
To select another supported OCR model for this run, pass `--ocr-model paddleocr-vl`, `--ocr-model paddleocr-v6`, or `--ocr-model pp-ocrv6-medium-rec-vietnamese`.
The default Vietnamese PP-OCRv6 recognizer requires its local checkpoint described in the main README.

## Run selected stages

Use `--stages` to select a comma-separated subset in pipeline order.
Stage names must be unique and remain in this order: `layout,ocr,tables,chunk,index`.

For example, run OCR and table reconstruction from an existing layout artifact:

```powershell
uv run python scripts/run_pipeline.py --stages ocr,tables --layout-json "data/layout/report/layout.json" --env-file .env
```

## OCR full page images without layout proposals

Use `--direct-ocr` with an image file or a directory of clean `page-NNNN.png` images as `--source`.
Select `ocr` and any supported downstream stages, such as `chunk,index`; omit `layout` and `tables` because those stages need layout proposals.

```powershell
uv run python scripts/run_pipeline.py --direct-ocr --source "data/extraction/report/pages" --stages ocr,chunk,index --env-file .env --collection rag1_hybrid_v1
```

For a single image and OCR only:

```powershell
uv run python scripts/run_pipeline.py --direct-ocr --source "data/extraction/report/pages/page-0001.png" --stages ocr
```

Direct OCR passes the original full-page image(s) to the selected OCR model.
The source can be a supported image file or a directory of clean page images, not a PDF.
After direct OCR, the runner can chunk and index its OCR artifact.
Table reconstruction is unavailable in the same direct run because it needs layout region proposals; indexing can still include an existing table artifact passed with `--tables-json`.
Use `--ocr-output-dir`, `--ocr-device`, and `--ocr-model` to configure direct OCR.

When a stage is skipped, pass its artifact if a selected downstream stage needs it.
`--layout-json` supplies the layout when `layout` is skipped and `ocr` or `tables` runs.
`--ocr-json` supplies OCR output when `ocr` is skipped and `tables` or `chunk` runs.
`--chunks-json` supplies chunks when `chunk` is skipped and `index` runs.
`--tables-json` optionally supplies reconstructed tables for indexing when `tables` is skipped.

For example, index existing chunks and table cells without running earlier stages:

```powershell
uv run python scripts/run_pipeline.py --stages index --chunks-json "data/ocr/report/chunks.json" --tables-json "data/ocr/report/tables.json" --env-file .env --collection rag1_hybrid_v1
```

Pass `--allow-partial` to continue when layout, OCR, or table artifacts have partial status.
Failed or malformed artifacts always stop the run.

## Options

| Option | Purpose |
| --- | --- |
| `--source` | PDF input when running `layout`; image or image directory with `--direct-ocr`. |
| `--direct-ocr` | Pass the source image or image directory straight to OCR and skip layout; requires `ocr` and excludes `layout` and `tables`. |
| `--stages` | Comma-separated stage subset in pipeline order; defaults to all stages. |
| `--layout-json`, `--ocr-json`, `--tables-json`, `--chunks-json` | Reuse artifacts from skipped stages. |
| `--env-file` | Environment file passed to each `uv run` command, commonly `.env` for Qdrant settings. |
| `--log-file` | Append this run's timestamped output to a specific log file; otherwise a new file is created under `logs/`. |
| `--allow-partial` | Continue through partial layout, OCR, or table status. |
| `--layout-output-dir`, `--layout-image-dir`, `--layout-dpi`, `--layout-device` | Override layout outputs and runtime settings. |
| `--ocr-output-dir`, `--ocr-image-dir`, `--ocr-device`, `--ocr-model`, `--ocr-manifest` | Override OCR outputs, model, and runtime settings. |
| `--table-output-dir`, `--table-model`, `--table-device`, `--table-manifest`, `--region-id` | Override table reconstruction settings or select one region. |
| `--collection` | Qdrant collection used by the index stage. |

Table reconstruction runs only when the layout contains visual table proposals, unless `--region-id` selects a region explicitly.
The table stage uses the `tabular` dependency extra.
The index stage embeds OCR chunks and, when `--tables-json` is available, reconstructed table cells before uploading to Qdrant.

## Logs and failures

If `--log-file` is omitted, the runner creates `logs/pipeline-YYYYMMDD-HHMMSS-microseconds.log` under the repository root.
The log includes each invoked command, combined stage output, exit code, and elapsed seconds.
The runner stops at the first failed stage or missing/invalid required artifact and returns a nonzero exit code.

To see every option supported by the installed runner:

```powershell
uv run python scripts/run_pipeline.py --help
```

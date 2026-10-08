# Repository Guidelines

## Project structure

Package code lives under `src/rag1/` and the `rag1` command entry point is in `src/rag1/__init__.py`.
The five document stages are layout detection, PP-OCRv6 Vietnamese OCR by default, table reconstruction, OCR chunking, and Qdrant indexing.
`scripts/run_pipeline.py` composes their existing CLI commands without replacing the stage implementations.
Stage defaults for layout and OCR live in `configs/extraction/`.
The Windows wrappers are `make.ps1` and `Makefile`.
Focused standard-library tests live under `tests/`.

## Commands

- `uv sync` installs the base package for Python 3.12, including PaddleOCR-VL.
- `uv run rag1 layout <report.pdf>` writes `layout.json`, a manifest, clean `pages/`, and debug `bbox/` images under a source-specific directory.
- `uv run rag1 ocr <layout.json>` uses PP-OCRv6 detection and the local Vietnamese recognizer, then writes `ocr.json` and `ocr.md` under a source-specific OCR directory.
- `uv run rag1 ocr <page.png> --direct` sends a full page image to OCR without layout proposals; directories of clean page images are also accepted.
- `uv run rag1 ocr <layout.json> --model paddleocr-vl` selects PaddleOCR-VL explicitly.
- `uv run --extra tabular rag1 tables <layout.json> --ocr-json <ocr.json>` writes `tables.json`, `tables.html`, and `tables.flat.json`.
- `uv run rag1 chunk <ocr.json>` writes `chunks.json` beside its OCR source.
- `uv run --env-file .env rag1 index <chunks.json> --tables-json <tables.json>` embeds OCR children and nonempty table cells, then uploads them to Qdrant.
- `uv run python scripts/run_pipeline.py --source <report.pdf> --env-file .env` runs all five stages and logs the output under `logs/`.
- `uv run python scripts/run_pipeline.py --direct-ocr --source <page.png-or-pages-dir> --stages ocr,chunk,index --env-file .env` runs full-page OCR and downstream chunk/index stages without layout proposals.
- `uv run python scripts/run_pipeline.py --help` lists partial-run inputs and per-stage overrides, including `--ocr-model`.
- `uv run python -m unittest discover -s tests` runs the focused test suite when test execution is requested.

## Artifact and pipeline rules

Pass clean `pages/` images to OCR, never the generated `bbox/` overlays or OCR `crops/` images.
Use `--stages` to select a subset in pipeline order and pass `--layout-json`, `--ocr-json`, `--tables-json`, or `--chunks-json` for skipped upstream stages.
The pipeline stops on failed commands and partial artifacts unless `--allow-partial` is set.
It skips table reconstruction when the layout has no visual `table` proposals.
The `index` command uses `QDRANT_URL` and optional `QDRANT_API_KEY` from the environment or the selected environment file.
It replaces old Qdrant points for the same source only after uploading and reading back the new points.
Table cell vectors have `record_type: table_cell`; OCR chunk vectors keep their existing hierarchy and pending nested table metadata.
Running `index` without `--tables-json` removes previous table cell points for that source as part of source replacement.
Do not edit generated OCR, layout, table, chunk, or lock artifacts by hand.

## Code and verification

Use four-space indentation, standard Python naming, and type annotations on public functions.
Keep changes near the stage that owns the behavior and preserve existing artifact schemas when extending them.
Add focused tests for important success and failure paths.
Treat CLI help, static checks, mocked tests, local model inference, and a live Qdrant upload as distinct verification layers.
Full model runs can take more than five minutes and may download model weights, so provide the exact command for the developer to execute unless they explicitly request an end-to-end run.

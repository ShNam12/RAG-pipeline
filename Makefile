# Run "make help" to see the available tasks and examples.
# Empty optional values are omitted by make.ps1, so CLI and YAML defaults apply.
.DEFAULT_GOAL := help

POWERSHELL ?= powershell
HELPER = $(POWERSHELL) -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1"

# Shared inputs for layout and OCR.
INPUT ?=
OUTPUT_DIR ?=
IMAGE_DIR ?=
DPI ?=
DEVICE ?=
MODEL ?=
MANIFEST ?=
DIRECT ?=
OUTPUT ?=

# Table reconstruction and flat export.
TABLE_LAYOUT_JSON ?=
TABLE_OCR_JSON ?=
TABLE_MANIFEST ?=
TABLE_OUTPUT_DIR ?=
TABLE_MODEL ?=
TABLE_DEVICE ?=
TABLE_REGION_ID ?=
TABLE_JSON ?=
TABLE_INPUT ?= $(TABLE_JSON)
TABLE_FLAT_OUTPUT ?=

# Partial or full pipeline.
PIPELINE_STAGES ?=
PIPELINE_LAYOUT_JSON ?=
PIPELINE_OCR_JSON ?=
PIPELINE_TABLES_JSON ?=
PIPELINE_CHUNKS_JSON ?=
PIPELINE_ENV_FILE ?=
PIPELINE_LOG_FILE ?=
PIPELINE_COLLECTION ?=
PIPELINE_LAYOUT_OUTPUT_DIR ?=
PIPELINE_LAYOUT_IMAGE_DIR ?=
PIPELINE_LAYOUT_DPI ?=
PIPELINE_LAYOUT_DEVICE ?=
PIPELINE_OCR_OUTPUT_DIR ?=
PIPELINE_OCR_IMAGE_DIR ?=
PIPELINE_OCR_DEVICE ?=
PIPELINE_OCR_MODEL ?=
PIPELINE_OCR_MANIFEST ?=
PIPELINE_DIRECT_OCR ?=
PIPELINE_TABLE_OUTPUT_DIR ?=
PIPELINE_TABLE_MODEL ?=
PIPELINE_TABLE_DEVICE ?=
PIPELINE_TABLE_MANIFEST ?=
PIPELINE_REGION_ID ?=
PIPELINE_ALLOW_PARTIAL ?=

LAYOUT_ARGS = \
    -InputPath "$(INPUT)" \
    -OutputDir "$(OUTPUT_DIR)" \
    -ImageDir "$(IMAGE_DIR)" \
    -Dpi "$(DPI)" \
    -Device "$(DEVICE)"

OCR_ARGS = \
    -InputPath "$(INPUT)" \
    -OutputDir "$(OUTPUT_DIR)" \
    -ImageDir "$(IMAGE_DIR)" \
    -Device "$(DEVICE)" \
    -Model "$(MODEL)" \
    -Manifest "$(MANIFEST)" \
    -Direct "$(DIRECT)"

OCR_VIETNAMESE_ARGS = \
    -InputPath "$(INPUT)" \
    -OutputDir "$(OUTPUT_DIR)" \
    -ImageDir "$(IMAGE_DIR)" \
    -Device "$(DEVICE)" \
    -Manifest "$(MANIFEST)" \
    -Direct "$(DIRECT)"

OCR_TEXT_ARGS = \
    -InputPath "$(INPUT)" \
    -OutputPath "$(OUTPUT)"

TABLE_ARGS = \
    -InputPath "$(TABLE_LAYOUT_JSON)" \
    -OcrJson "$(TABLE_OCR_JSON)" \
    -Manifest "$(TABLE_MANIFEST)" \
    -OutputDir "$(TABLE_OUTPUT_DIR)" \
    -Model "$(TABLE_MODEL)" \
    -Device "$(TABLE_DEVICE)" \
    -RegionId "$(TABLE_REGION_ID)"

TABLE_FLAT_ARGS = \
    -InputPath "$(TABLE_INPUT)" \
    -OutputPath "$(TABLE_FLAT_OUTPUT)"

PIPELINE_ARGS = \
    -InputPath "$(INPUT)" \
    -Stages "$(PIPELINE_STAGES)" \
    -LayoutJson "$(PIPELINE_LAYOUT_JSON)" \
    -OcrJson "$(PIPELINE_OCR_JSON)" \
    -TablesJson "$(PIPELINE_TABLES_JSON)" \
    -ChunksJson "$(PIPELINE_CHUNKS_JSON)" \
    -EnvFile "$(PIPELINE_ENV_FILE)" \
    -LogFile "$(PIPELINE_LOG_FILE)" \
    -Collection "$(PIPELINE_COLLECTION)" \
    -LayoutOutputDir "$(PIPELINE_LAYOUT_OUTPUT_DIR)" \
    -LayoutImageDir "$(PIPELINE_LAYOUT_IMAGE_DIR)" \
    -LayoutDpi "$(PIPELINE_LAYOUT_DPI)" \
    -LayoutDevice "$(PIPELINE_LAYOUT_DEVICE)" \
    -OcrOutputDir "$(PIPELINE_OCR_OUTPUT_DIR)" \
    -OcrImageDir "$(PIPELINE_OCR_IMAGE_DIR)" \
    -OcrDevice "$(PIPELINE_OCR_DEVICE)" \
    -OcrModel "$(PIPELINE_OCR_MODEL)" \
    -OcrManifest "$(PIPELINE_OCR_MANIFEST)" \
    -Direct "$(PIPELINE_DIRECT_OCR)" \
    -TableOutputDir "$(PIPELINE_TABLE_OUTPUT_DIR)" \
    -TableModel "$(PIPELINE_TABLE_MODEL)" \
    -TableDevice "$(PIPELINE_TABLE_DEVICE)" \
    -TableManifest "$(PIPELINE_TABLE_MANIFEST)" \
    -RegionId "$(PIPELINE_REGION_ID)" \
    -AllowPartial "$(PIPELINE_ALLOW_PARTIAL)"

.PHONY: help layout ocr ocr-vietnamese ocr-text tables tables-flat pipeline

help:
	@$(HELPER) help

layout:
	@$(HELPER) layout $(LAYOUT_ARGS)

ocr:
	@$(HELPER) ocr $(OCR_ARGS)

ocr-vietnamese:
	@$(HELPER) ocr-vietnamese $(OCR_VIETNAMESE_ARGS)

ocr-text:
	@$(HELPER) ocr-text $(OCR_TEXT_ARGS)

tables:
	@$(HELPER) tables $(TABLE_ARGS)

tables-flat:
	@$(HELPER) tables-flat $(TABLE_FLAT_ARGS)

pipeline:
	@$(HELPER) pipeline $(PIPELINE_ARGS)

INPUT ?=
OUTPUT_DIR ?=
IMAGE_DIR ?=
DPI ?=
DEVICE ?=
MODEL ?=
MANIFEST ?=
DIRECT ?=
OUTPUT ?=
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
PIPELINE_OCR_MANIFEST ?=
PIPELINE_TABLE_OUTPUT_DIR ?=
PIPELINE_TABLE_MODEL ?=
PIPELINE_TABLE_DEVICE ?=
PIPELINE_TABLE_MANIFEST ?=
PIPELINE_REGION_ID ?=
PIPELINE_ALLOW_PARTIAL ?=

.PHONY: layout ocr ocr-vietnamese ocr-text tables tables-flat pipeline
layout:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" layout -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Dpi "$(DPI)" -Device "$(DEVICE)"

ocr:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" ocr -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Device "$(DEVICE)" -Model "$(MODEL)" -Manifest "$(MANIFEST)" -Direct "$(DIRECT)"

ocr-vietnamese:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" ocr-vietnamese -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Device "$(DEVICE)" -Manifest "$(MANIFEST)" -Direct "$(DIRECT)"

ocr-text:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" ocr-text -InputPath "$(INPUT)" -OutputPath "$(OUTPUT)"

tables:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" tables -InputPath "$(TABLE_LAYOUT_JSON)" -OcrJson "$(TABLE_OCR_JSON)" -Manifest "$(TABLE_MANIFEST)" -OutputDir "$(TABLE_OUTPUT_DIR)" -Model "$(TABLE_MODEL)" -Device "$(TABLE_DEVICE)" -RegionId "$(TABLE_REGION_ID)"

tables-flat:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" tables-flat -InputPath "$(TABLE_INPUT)" -OutputPath "$(TABLE_FLAT_OUTPUT)"

pipeline:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" pipeline -InputPath "$(INPUT)" -Stages "$(PIPELINE_STAGES)" -LayoutJson "$(PIPELINE_LAYOUT_JSON)" -OcrJson "$(PIPELINE_OCR_JSON)" -TablesJson "$(PIPELINE_TABLES_JSON)" -ChunksJson "$(PIPELINE_CHUNKS_JSON)" -EnvFile "$(PIPELINE_ENV_FILE)" -LogFile "$(PIPELINE_LOG_FILE)" -Collection "$(PIPELINE_COLLECTION)" -LayoutOutputDir "$(PIPELINE_LAYOUT_OUTPUT_DIR)" -LayoutImageDir "$(PIPELINE_LAYOUT_IMAGE_DIR)" -LayoutDpi "$(PIPELINE_LAYOUT_DPI)" -LayoutDevice "$(PIPELINE_LAYOUT_DEVICE)" -OcrOutputDir "$(PIPELINE_OCR_OUTPUT_DIR)" -OcrImageDir "$(PIPELINE_OCR_IMAGE_DIR)" -OcrDevice "$(PIPELINE_OCR_DEVICE)" -OcrManifest "$(PIPELINE_OCR_MANIFEST)" -TableOutputDir "$(PIPELINE_TABLE_OUTPUT_DIR)" -TableModel "$(PIPELINE_TABLE_MODEL)" -TableDevice "$(PIPELINE_TABLE_DEVICE)" -TableManifest "$(PIPELINE_TABLE_MANIFEST)" -RegionId "$(PIPELINE_REGION_ID)" -AllowPartial "$(PIPELINE_ALLOW_PARTIAL)"

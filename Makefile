INPUT ?=
OUTPUT_DIR ?=
IMAGE_DIR ?=
DPI ?=
DEVICE ?=
MODEL ?=
MANIFEST ?=
DIRECT ?=
OUTPUT ?=

.PHONY: layout ocr ocr-vietnamese ocr-text
layout:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" layout -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Dpi "$(DPI)" -Device "$(DEVICE)"

ocr:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" ocr -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Device "$(DEVICE)" -Model "$(MODEL)" -Manifest "$(MANIFEST)" -Direct "$(DIRECT)"

ocr-vietnamese:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" ocr-vietnamese -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Device "$(DEVICE)" -Manifest "$(MANIFEST)" -Direct "$(DIRECT)"

ocr-text:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" ocr-text -InputPath "$(INPUT)" -OutputPath "$(OUTPUT)"

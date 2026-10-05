INPUT ?=
OUTPUT_DIR ?= data/extraction
IMAGE_DIR ?= pages
DPI ?= 200
DEVICE ?= cpu

.PHONY: layout
layout:
	powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\make.ps1" layout -InputPath "$(INPUT)" -OutputDir "$(OUTPUT_DIR)" -ImageDir "$(IMAGE_DIR)" -Dpi $(DPI) -Device "$(DEVICE)"

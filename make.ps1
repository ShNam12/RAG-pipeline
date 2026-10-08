[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "layout", "ocr", "ocr-vietnamese", "ocr-text", "tables", "tables-flat", "pipeline")]
    [string]$Task = "help",

    [string]$InputPath,
    [string]$OutputDir,
    [string]$OutputPath,
    [string]$ImageDir,
    [string]$Dpi,
    [string]$Device,
    [string]$Model,
    [string]$Manifest,
    [string]$Direct,
    [string]$OcrJson,
    [string]$RegionId,
    [string]$Stages,
    [string]$LayoutJson,
    [string]$TablesJson,
    [string]$ChunksJson,
    [string]$EnvFile,
    [string]$LogFile,
    [string]$Collection,
    [string]$LayoutOutputDir,
    [string]$LayoutImageDir,
    [string]$LayoutDpi,
    [string]$LayoutDevice,
    [string]$OcrOutputDir,
    [string]$OcrImageDir,
    [string]$OcrDevice,
    [string]$OcrModel,
    [string]$OcrManifest,
    [string]$TableOutputDir,
    [string]$TableModel,
    [string]$TableDevice,
    [string]$TableManifest,
    [string]$AllowPartial
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @'
Usage:
  .\make.ps1 <task> -InputPath <path> [options]
  make <task> INPUT=<path> [VARIABLE=value ...]

Tasks:
  layout           Detect regions in a PDF and write clean page images.
  ocr              Recognize a layout JSON or, with -Direct 1, page images.
  ocr-vietnamese   Run OCR with the Vietnamese recognizer explicitly selected.
  ocr-text         Export Markdown from an existing ocr.json.
  tables           Reconstruct tables from layout.json and ocr.json.
  tables-flat      Flatten tables.json or tables.html into row records.
  pipeline         Run selected stages with scripts/run_pipeline.py.

Required inputs:
  layout, ocr, ocr-vietnamese, ocr-text, tables-flat: -InputPath
  tables: -InputPath <layout.json> and -OcrJson <ocr.json>
  pipeline: -InputPath when the selected stages need a source.

Make variables:
  layout: INPUT, OUTPUT_DIR, IMAGE_DIR, DPI, DEVICE
  ocr: INPUT, OUTPUT_DIR, IMAGE_DIR, DEVICE, MODEL, MANIFEST, DIRECT
  ocr-text: INPUT, OUTPUT
  tables: TABLE_LAYOUT_JSON, TABLE_OCR_JSON, TABLE_MANIFEST, TABLE_OUTPUT_DIR,
          TABLE_MODEL, TABLE_DEVICE, TABLE_REGION_ID
  tables-flat: TABLE_INPUT, TABLE_FLAT_OUTPUT
  pipeline: INPUT and the PIPELINE_* variables listed in Makefile

Examples:
  .\make.ps1 layout -InputPath report.pdf
  .\make.ps1 ocr -InputPath layout.json -Model paddleocr-vl
  .\make.ps1 ocr -InputPath page.png -Direct 1
  .\make.ps1 tables -InputPath layout.json -OcrJson ocr.json
  .\make.ps1 pipeline -InputPath report.pdf -EnvFile .env
  make layout INPUT=report.pdf
  make tables TABLE_LAYOUT_JSON=layout.json TABLE_OCR_JSON=ocr.json
  make pipeline INPUT=report.pdf PIPELINE_ENV_FILE=.env

Use 1/true/yes/on or 0/false/no/off for -Direct and -AllowPartial.
Omit optional values to use the defaults in the CLI and configs/extraction/.
Run uv run rag1 <command> --help or uv run python scripts/run_pipeline.py --help for all stage options.
'@
}

function Add-ProvidedOptions {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [System.Collections.IDictionary]$Options
    )

    foreach ($Option in $Options.GetEnumerator()) {
        $Value = [string]$Option.Value
        if (-not [string]::IsNullOrWhiteSpace($Value)) {
            $Arguments.Add([string]$Option.Key)
            $Arguments.Add($Value)
        }
    }
}

function Add-BooleanFlag {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Name,
        [string]$Value,
        [string]$ParameterName
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }

    $Normalized = $Value.Trim().ToLowerInvariant()
    if ($Normalized -in @("1", "true", "yes", "on")) {
        $Arguments.Add($Name)
        return
    }
    if ($Normalized -notin @("0", "false", "no", "off")) {
        throw "$ParameterName must be 1/true/yes/on or 0/false/no/off."
    }
}

if ($Task -eq "help") {
    Show-Usage
    return
}

if ($Task -ne "pipeline" -and [string]::IsNullOrWhiteSpace($InputPath)) {
    switch ($Task) {
        "tables" { throw "Pass the layout JSON with -InputPath or Make variable TABLE_LAYOUT_JSON." }
        "tables-flat" { throw "Pass tables.json or tables.html with -InputPath or Make variable TABLE_INPUT." }
        default { throw "Pass the input path with -InputPath or Make variable INPUT." }
    }
}
if ($Task -eq "tables" -and [string]::IsNullOrWhiteSpace($OcrJson)) {
    throw "Pass the OCR JSON with -OcrJson or Make variable TABLE_OCR_JSON."
}

$CommandArguments = [System.Collections.Generic.List[string]]::new()

if ($Task -eq "pipeline") {
    $CommandArguments.Add("run")
    $CommandArguments.Add("python")
    $CommandArguments.Add("scripts/run_pipeline.py")
    Add-ProvidedOptions $CommandArguments ([ordered]@{
        "--stages" = $Stages
        "--source" = $InputPath
        "--layout-json" = $LayoutJson
        "--ocr-json" = $OcrJson
        "--tables-json" = $TablesJson
        "--chunks-json" = $ChunksJson
        "--env-file" = $EnvFile
        "--log-file" = $LogFile
        "--collection" = $Collection
        "--layout-output-dir" = $LayoutOutputDir
        "--layout-image-dir" = $LayoutImageDir
        "--layout-dpi" = $LayoutDpi
        "--layout-device" = $LayoutDevice
        "--ocr-output-dir" = $OcrOutputDir
        "--ocr-image-dir" = $OcrImageDir
        "--ocr-device" = $OcrDevice
        "--ocr-model" = $OcrModel
        "--ocr-manifest" = $OcrManifest
        "--table-output-dir" = $TableOutputDir
        "--table-model" = $TableModel
        "--table-device" = $TableDevice
        "--table-manifest" = $TableManifest
        "--region-id" = $RegionId
    })
    Add-BooleanFlag $CommandArguments "--direct-ocr" $Direct "DIRECT"
    Add-BooleanFlag $CommandArguments "--allow-partial" $AllowPartial "ALLOW_PARTIAL"
}
else {
    $Command = $Task
    if ($Task -eq "ocr-vietnamese") {
        $Command = "ocr"
        $Model = "pp-ocrv6-medium-rec-vietnamese"
    }

    $CommandArguments.Add("run")
    if ($Command -eq "tables") {
        $CommandArguments.Add("--extra")
        $CommandArguments.Add("tabular")
    }
    elseif ($Command -eq "ocr" -and $Model -eq "paddleocr-vl") {
        $CommandArguments.Add("--extra")
        $CommandArguments.Add("vl")
    }
    $CommandArguments.Add("rag1")
    $CommandArguments.Add($Command)
    $CommandArguments.Add($InputPath)

    switch ($Command) {
        "layout" {
            Add-ProvidedOptions $CommandArguments ([ordered]@{
                "--output-dir" = $OutputDir
                "--image-dir" = $ImageDir
                "--dpi" = $Dpi
                "--device" = $Device
            })
        }
        "ocr" {
            Add-ProvidedOptions $CommandArguments ([ordered]@{
                "--output-dir" = $OutputDir
                "--image-dir" = $ImageDir
                "--device" = $Device
                "--model" = $Model
                "--manifest" = $Manifest
            })
            Add-BooleanFlag $CommandArguments "--direct" $Direct "DIRECT"
        }
        "ocr-text" {
            Add-ProvidedOptions $CommandArguments ([ordered]@{
                "--output" = $OutputPath
            })
        }
        "tables" {
            $CommandArguments.Add("--ocr-json")
            $CommandArguments.Add($OcrJson)
            Add-ProvidedOptions $CommandArguments ([ordered]@{
                "--manifest" = $Manifest
                "--output-dir" = $OutputDir
                "--model" = $Model
                "--device" = $Device
                "--region-id" = $RegionId
            })
        }
        "tables-flat" {
            Add-ProvidedOptions $CommandArguments ([ordered]@{
                "--output" = $OutputPath
            })
        }
    }
}

$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepositoryRoot
try {
    Write-Host "Running task: $Task"
    $UvArguments = $CommandArguments.ToArray()
    & uv @UvArguments
    $UvExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($UvExitCode -ne 0) {
    exit $UvExitCode
}

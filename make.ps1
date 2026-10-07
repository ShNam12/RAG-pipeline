[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("layout", "ocr", "ocr-vietnamese", "ocr-text", "tables", "tables-flat", "pipeline")]
    [string]$Task,

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
    [string]$OcrManifest,
    [string]$TableOutputDir,
    [string]$TableModel,
    [string]$TableDevice,
    [string]$TableManifest,
    [string]$AllowPartial
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepositoryRoot

try {
    if ($Task -eq "pipeline") {
        $PipelineArguments = @("run", "python", "scripts/run_pipeline.py")
        $PipelineOptions = [ordered]@{
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
            "--ocr-manifest" = $OcrManifest
            "--table-output-dir" = $TableOutputDir
            "--table-model" = $TableModel
            "--table-device" = $TableDevice
            "--table-manifest" = $TableManifest
            "--region-id" = $RegionId
        }
        foreach ($Option in $PipelineOptions.GetEnumerator()) {
            if (-not [string]::IsNullOrWhiteSpace($Option.Value)) {
                $PipelineArguments += @($Option.Key, $Option.Value)
            }
        }
        if (-not [string]::IsNullOrWhiteSpace($AllowPartial)) {
            if ($AllowPartial -notin @("1", "true", "yes", "on", "0", "false", "no", "off")) {
                throw "ALLOW_PARTIAL must be 1/true/yes/on or 0/false/no/off."
            }
            if ($AllowPartial -in @("1", "true", "yes", "on")) {
                $PipelineArguments += "--allow-partial"
            }
        }
        & uv @PipelineArguments
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        return
    }

    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        if ($Task -eq "tables") {
            throw "Pass the layout JSON with -InputPath or Make variable TABLE_LAYOUT_JSON."
        }
        if ($Task -eq "tables-flat") {
            throw "Pass tables.json or tables.html with -InputPath or Make variable TABLE_INPUT."
        }
        throw "Pass the input path with -InputPath or Make variable INPUT."
    }
    if ($Task -eq "tables" -and [string]::IsNullOrWhiteSpace($OcrJson)) {
        throw "Pass the OCR JSON with -OcrJson or Make variable TABLE_OCR_JSON."
    }

    if ($Task -eq "ocr-vietnamese") {
        $Task = "ocr"
        $Model = "pp-ocrv6-medium-rec-vietnamese"
    }

    $UvArguments = @("run")
    if ($Task -eq "tables") {
        $UvArguments += @("--extra", "tabular")
    }
    $UvArguments += @("rag1", $Task, $InputPath)

    if ($Task -eq "layout") {
        if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
            $UvArguments += @("--output-dir", $OutputDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($ImageDir)) {
            $UvArguments += @("--image-dir", $ImageDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($Dpi)) {
            $UvArguments += @("--dpi", $Dpi)
        }
        if (-not [string]::IsNullOrWhiteSpace($Device)) {
            $UvArguments += @("--device", $Device)
        }
    }
    elseif ($Task -eq "ocr") {
        if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
            $UvArguments += @("--output-dir", $OutputDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($ImageDir)) {
            $UvArguments += @("--image-dir", $ImageDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($Device)) {
            $UvArguments += @("--device", $Device)
        }
        if (-not [string]::IsNullOrWhiteSpace($Model)) {
            $UvArguments += @("--model", $Model)
        }
        if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
            $UvArguments += @("--manifest", $Manifest)
        }
        if (-not [string]::IsNullOrWhiteSpace($Direct)) {
            if ($Direct -notin @("1", "true", "yes", "on", "0", "false", "no", "off")) {
                throw "DIRECT must be 1/true/yes/on or 0/false/no/off."
            }
            if ($Direct -in @("1", "true", "yes", "on")) {
                $UvArguments += "--direct"
            }
        }
    }
    elseif ($Task -eq "ocr-text") {
        if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
            $UvArguments += @("--output", $OutputPath)
        }
    }
    elseif ($Task -eq "tables") {
        $UvArguments += @("--ocr-json", $OcrJson)
        if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
            $UvArguments += @("--manifest", $Manifest)
        }
        if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
            $UvArguments += @("--output-dir", $OutputDir)
        }
        if (-not [string]::IsNullOrWhiteSpace($Model)) {
            $UvArguments += @("--model", $Model)
        }
        if (-not [string]::IsNullOrWhiteSpace($Device)) {
            $UvArguments += @("--device", $Device)
        }
        if (-not [string]::IsNullOrWhiteSpace($RegionId)) {
            $UvArguments += @("--region-id", $RegionId)
        }
    }
    elseif ($Task -eq "tables-flat") {
        if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
            $UvArguments += @("--output", $OutputPath)
        }
    }

    & uv @UvArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

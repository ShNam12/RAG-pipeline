[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("layout", "ocr", "ocr-text")]
    [string]$Task,

    [string]$InputPath,
    [string]$OutputDir,
    [string]$OutputPath,
    [string]$ImageDir,
    [string]$Dpi,
    [string]$Device,
    [string]$Model,
    [string]$Manifest,
    [string]$Direct
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepositoryRoot

try {
    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        throw "Pass the input path with -InputPath or Make variable INPUT."
    }

    $UvArguments = @("run")
    if ($Task -eq "ocr" -and $Model -eq "paddleocr-vl") {
        $UvArguments += @("--extra", "vl")
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

    & uv @UvArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

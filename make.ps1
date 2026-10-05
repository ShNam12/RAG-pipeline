[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("layout")]
    [string]$Task,

    [string]$InputPath,
    [string]$OutputDir = "data/extraction",
    [string]$ImageDir = "pages",
    [int]$Dpi = 200,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepositoryRoot

try {
    if ($Task -eq "layout") {
        if ([string]::IsNullOrWhiteSpace($InputPath)) {
            throw "Pass the PDF path with -InputPath or Make variable INPUT."
        }

        & uv run rag1 layout $InputPath --output-dir $OutputDir --image-dir $ImageDir --dpi $Dpi --device $Device
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 725,
    [switch]$NoBrowser,
    [string]$PythonPath = ""
)

$entry = Join-Path $PSScriptRoot "start_interface_platform.ps1"
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "Missing start_interface_platform.ps1"
}
& $entry @PSBoundParameters
exit $LASTEXITCODE

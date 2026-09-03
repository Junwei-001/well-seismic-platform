[CmdletBinding()]
param([ValidateRange(1, 65535)][int]$Port = 725)

$entry = Join-Path $PSScriptRoot "stop_interface_platform.ps1"
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "Missing stop_interface_platform.ps1"
}
& $entry @PSBoundParameters
exit $LASTEXITCODE

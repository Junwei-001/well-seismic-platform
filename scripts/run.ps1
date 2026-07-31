[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FrontendIndex = Join-Path $ProjectRoot "frontend\dist\index.html"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "尚未安装环境，请先运行 .\scripts\setup.ps1"
}
if (-not (Test-Path -LiteralPath $FrontendIndex)) {
    throw "前端尚未构建，请重新运行 .\scripts\setup.ps1"
}

$env:WELL_SEISMIC_HOST = $HostAddress
$env:WELL_SEISMIC_PORT = "$Port"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

Push-Location $ProjectRoot
try {
    & $VenvPython -m well_seismic.api
} finally {
    Pop-Location
}


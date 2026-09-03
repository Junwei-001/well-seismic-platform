[CmdletBinding()]
param([string]$BasePython = "")

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$venv = Join-Path $root ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $candidates = @()
    if ($BasePython.Trim()) {
        $candidates += if ([IO.Path]::IsPathRooted($BasePython)) {
            [IO.Path]::GetFullPath($BasePython)
        } else {
            [IO.Path]::GetFullPath((Join-Path $root $BasePython))
        }
    } else {
        $candidates += Join-Path $root "runtime\py311\python.exe"
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $command) { $candidates += $command.Source }
    }
    $base = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if ($null -eq $base) {
        throw "未找到用于创建 .venv 的 Python 解释器。"
    }
    Write-Host "正在创建本地 .venv：$venv" -ForegroundColor Cyan
    & $base -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw ".venv 创建失败，代码：$LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "创建后仍未找到 .venv\Scripts\python.exe。"
}
Write-Host "正在安装接口版依赖（不包含任务模型权重）：$venvPython" -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $root "requirements-interface.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败，代码：$LASTEXITCODE" }
Write-Host "接口版依赖安装完成。现在可运行 start_interface_platform.bat。" -ForegroundColor Green

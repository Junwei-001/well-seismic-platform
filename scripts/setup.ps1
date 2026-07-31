[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Push-Location $ProjectRoot
try {
    Write-Host "[1/5] 检查模型文件"
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git")) {
        git lfs version | Out-Null
        git lfs pull
    } else {
        Write-Host "当前为独立发布文件夹，使用随包模型权重。"
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host "[2/5] 创建 Python 虚拟环境"
        & $PythonCommand -m venv $VenvPath
    } else {
        Write-Host "[2/5] 复用已有 Python 虚拟环境"
    }

    Write-Host "[3/5] 安装后端与推理依赖"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements\development.txt")

    if (-not $SkipFrontend) {
        Write-Host "[4/5] 构建 Vue 前端"
        Push-Location (Join-Path $ProjectRoot "frontend")
        try {
            npm ci
            npm run build
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "[4/5] 已跳过前端构建"
    }

    Write-Host "[5/5] 执行发布自检"
    & $VenvPython (Join-Path $ProjectRoot "tools\verify_release.py") --runtime
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "安装完成。运行 .\scripts\run.ps1 启动平台。"

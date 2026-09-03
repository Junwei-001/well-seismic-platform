[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 725,
    [switch]$NoBrowser,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
if (-not (Test-Path -LiteralPath (Join-Path $root "frontend\dist\index.html") -PathType Leaf)) {
    throw "缺少 frontend\dist\index.html；请从抽离目录完整复制前端。"
}

$pythonCandidates = @()
if ($PythonPath.Trim()) {
    $pythonCandidates += if ([IO.Path]::IsPathRooted($PythonPath)) {
        [IO.Path]::GetFullPath($PythonPath)
    } else {
        [IO.Path]::GetFullPath((Join-Path $root $PythonPath))
    }
} else {
    $pythonCandidates += Join-Path $root ".venv\Scripts\python.exe"
    $pythonCandidates += Join-Path $root "runtime\py311\python.exe"
    if ($env:LOCALAPPDATA) {
        $pythonCandidates += Join-Path $env:LOCALAPPDATA "WellSeismicPlatform\py311\python.exe"
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $command) { $pythonCandidates += $command.Source }
}

$python = $null
$probeState = Join-Path $env:TEMP ("strata-interface-probe-{0}.sqlite3" -f $PID)
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:WELLFUSE_MODEL_MODE = "interfaces_only"
$env:WELLFUSE_DISABLE_TASK_MODELS = "1"
$env:WELLFUSE_DISABLE_LAYERPULSE = "1"
$env:WELL_SEISMIC_LLM_ENABLED = "false"
$env:WELL_SEISMIC_STATE_DB = $probeState
$env:WELLFUSE_PROJECT_ROOT = Join-Path $root "runtime\wellfuse"
$env:WELLFUSE_ARTIFACT_ROOT = ""
$env:WELLFUSE_DIRECT12B_PROJECT_ROOT = ""
$env:WELLFUSE_DIRECT12B_MODEL_PATH = ""
$previousErrorActionPreference = $ErrorActionPreference
$pythonFailures = @()
$probeCode = "import sys, importlib.util; names=['fastapi','uvicorn','numpy','yaml','pyproj']; missing=[n for n in names if importlib.util.find_spec(n) is None]; missing += [] if sys.version_info >= (3,11) else ['python>=3.11']; print('missing=' + ','.join(missing)); raise SystemExit(1 if missing else 0)"
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    $ErrorActionPreference = "Continue"
    $probeOutput = @(& $candidate -c $probeCode 2>$null)
    $dependencyExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($dependencyExitCode -eq 0) {
        $env:WELLFUSE_MODEL_MODE = "interfaces_only"
        $env:WELLFUSE_DISABLE_TASK_MODELS = "1"
        $apiProbeOutput = @(& $candidate -c "import well_seismic.api" 2>$null)
        $apiProbeExitCode = $LASTEXITCODE
        if ($apiProbeExitCode -eq 0) {
            $python = $candidate
            break
        }
        $pythonFailures += "${candidate}: well_seismic.api 导入失败"
    } else {
        $detail = ($probeOutput -join " ").Trim()
        if (-not $detail) { $detail = "依赖不完整" }
        $pythonFailures += "${candidate}: $detail"
    }
}
if ($null -eq $python) {
    if (Test-Path -LiteralPath $probeState) { Remove-Item -LiteralPath $probeState -Force -ErrorAction SilentlyContinue }
    $details = if ($pythonFailures.Count) { "`n" + ($pythonFailures -join "`n") } else { "" }
    throw "未找到完整的 Python Web 运行时（需要 fastapi、uvicorn、numpy、PyYAML、pyproj）。请先执行：python -m pip install -e `".[web]`"$details"
}
if (Test-Path -LiteralPath $probeState) { Remove-Item -LiteralPath $probeState -Force -ErrorAction SilentlyContinue }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = Join-Path $root "src"
$env:WELL_SEISMIC_HOST = "127.0.0.1"
$env:WELL_SEISMIC_PORT = [string]$Port
$env:WELLFUSE_MODEL_MODE = "interfaces_only"
$env:WELLFUSE_DISABLE_TASK_MODELS = "1"
$env:WELLFUSE_DISABLE_LAYERPULSE = "1"
$env:WELL_SEISMIC_LLM_ENABLED = "false"
$env:WELL_SEISMIC_STATE_DB = Join-Path $root "runtime\state\platform_state.sqlite3"
$env:WELLFUSE_PROJECT_ROOT = Join-Path $root "runtime\wellfuse"
$env:WELLFUSE_ARTIFACT_ROOT = ""
$env:WELLFUSE_DIRECT12B_PROJECT_ROOT = ""
$env:WELLFUSE_DIRECT12B_MODEL_PATH = ""
$stateRoot = Join-Path $root "runtime\state"
New-Item -ItemType Directory -Force -Path (Join-Path $root "data"), (Join-Path $root "model_outputs"), $stateRoot | Out-Null

$listener = @(Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listener.Count -gt 0) {
    throw "端口 $Port 已被占用；请先停止旧服务或改用 -Port。"
}

Write-Host "正在启动地层慧眼接口版：http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "当前模式：interfaces_only（不加载任何任务模型权重）" -ForegroundColor Yellow
Write-Host "Python：$python"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$pidFile = Join-Path $stateRoot ("interface-platform-{0}.pid.json" -f $Port)
$stdoutLog = Join-Path $stateRoot ("interface-platform-{0}.stdout.log" -f $stamp)
$stderrLog = Join-Path $stateRoot ("interface-platform-{0}.stderr.log" -f $stamp)
Push-Location $root
$process = $null
try {
    $process = Start-Process -FilePath $python `
        -ArgumentList @("-m", "well_seismic.api") `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru
    $record = [ordered]@{
        schema_version = 1
        pid = [int]$process.Id
        port = $Port
        root = $root
        python = $python
        stdout_log = $stdoutLog
        stderr_log = $stderrLog
        started_at = [DateTime]::UtcNow.ToString("o")
    }
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pidFile -Encoding UTF8

    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    $healthy = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        $process.Refresh()
        if ($process.HasExited) {
            $tail = if (Test-Path -LiteralPath $stderrLog) {
                (Get-Content -LiteralPath $stderrLog -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
            } else { "（无 stderr 日志）" }
            throw "平台进程提前退出，代码：$($process.ExitCode)`n$tail"
        }
        try {
            $health = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/api/v1/health" -f $Port) -TimeoutSec 2
        } catch {
            $health = $null
        }
        if ($null -ne $health -and [string]$health.status -eq "ok" -and [string]$health.runtime_mode -eq "interfaces_only") {
            $healthy = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $healthy) {
        $process.Refresh()
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
        $tail = if (Test-Path -LiteralPath $stderrLog) {
            (Get-Content -LiteralPath $stderrLog -Tail 40 -ErrorAction SilentlyContinue) -join "`n"
        } else { "（无 stderr 日志）" }
        throw "45 秒内未通过健康检查。错误日志：$stderrLog`n$tail"
    }
    Write-Host "平台启动成功：http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host "错误日志：$stderrLog"
    Write-Host "停止服务请运行 stop_interface_platform.bat 或 停止接口平台.bat。"
    if (-not $NoBrowser) {
        try {
            Start-Process ("http://127.0.0.1:{0}" -f $Port)
        } catch {
            Write-Warning "浏览器未能自动打开；请手动访问 http://127.0.0.1:$Port"
        }
    }
    $process.WaitForExit()
    # Stop-Process -Force reports -1 on Windows; that is an intentional
    # shutdown when the companion stop script was used.
    $exitCode = $process.ExitCode
    if ($null -ne $exitCode -and $exitCode -ne 0 -and $exitCode -ne -1) {
        throw "平台进程退出，代码：$exitCode`n日志：$stderrLog"
    }
} finally {
    if (Test-Path -LiteralPath $pidFile) {
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}

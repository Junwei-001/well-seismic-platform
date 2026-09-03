[CmdletBinding()]
param([ValidateRange(1, 65535)][int]$Port = 725)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$rootPattern = [regex]::Escape($root)
$pidFile = Join-Path $root ("runtime\state\interface-platform-{0}.pid.json" -f $Port)
$record = $null
if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
    try { $record = Get-Content -LiteralPath $pidFile -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $record = $null }
}
if ($null -ne $record -and [int]$record.pid -gt 0) {
    $pidValue = [int]$record.pid
    $processInfo = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $pidValue) -ErrorAction SilentlyContinue
    $commandLine = [string]($processInfo.CommandLine)
    if ($null -ne $processInfo -and $commandLine -match "well_seismic\.api" -and [string]$record.root -eq $root -and [int]$record.port -eq $Port) {
        Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        Write-Host "已停止接口版进程 PID=$pidValue。" -ForegroundColor Green
        exit 0
    }
    # A crashed/externally terminated process can leave only a stale record.
    if ($null -eq $processInfo -or [string]$record.root -ne $root -or [int]$record.port -ne $Port) {
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
}
$connections = @(Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if (-not $connections) {
    Write-Host "端口 $Port 未发现监听中的接口版进程。"
    exit 0
}
$stopped = 0
foreach ($connection in $connections) {
    $pidValue = [int]$connection.OwningProcess
    $processInfo = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $pidValue) -ErrorAction SilentlyContinue
    $commandLine = [string]($processInfo.CommandLine)
    if ($commandLine -match $rootPattern -and $commandLine -match "well_seismic\.api") {
        Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
        Write-Host "已停止接口版进程 PID=$pidValue。" -ForegroundColor Green
        $stopped++
    } else {
        Write-Warning "端口 $Port 被其他进程占用（PID=$pidValue），未强制终止。"
    }
}
if ($stopped -eq 0) { exit 1 }

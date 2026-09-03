@echo off
setlocal
chcp 65001 >nul
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Run the verifier with an installed Python.
  exit /b 1
)
python "%~dp0tools\verify_platform_skeleton.py"
set "exit_code=%errorlevel%"
if not "%exit_code%"=="0" pause
exit /b %exit_code%

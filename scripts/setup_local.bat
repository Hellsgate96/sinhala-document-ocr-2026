@echo off
REM Wrapper for setup_local.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_local.ps1" %*

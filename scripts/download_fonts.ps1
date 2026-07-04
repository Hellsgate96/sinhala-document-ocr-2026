# Download Noto Sans Sinhala into project fonts/ (backup when OS fonts are missing)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_fonts.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$FontsDir = Join-Path $ProjectRoot "fonts"
$Dest = Join-Path $FontsDir "NotoSansSinhala-Regular.ttf"
$Url = "https://github.com/notofonts/sinhala/raw/main/fonts/NotoSansSinhala/hinted/ttf/NotoSansSinhala-Regular.ttf"

if (-not (Test-Path $FontsDir)) {
    New-Item -ItemType Directory -Force -Path $FontsDir | Out-Null
}

if (Test-Path $Dest) {
    Write-Host "Already present:" $Dest
    exit 0
}

Write-Host "Downloading Noto Sans Sinhala..."
Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
Write-Host "Saved:" $Dest

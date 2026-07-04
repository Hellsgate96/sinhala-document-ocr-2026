# Sinhala Document OCR — one-time local setup (Windows PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1

param(
    [switch]$CreateVenv
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot"

if ($CreateVenv) {
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
        Write-Host "Created .venv"
    }
    & ".venv\Scripts\Activate.ps1"
}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$optionalExit = 0
try {
    python -m pip install -r requirements-optional.txt 2>$null
    if ($LASTEXITCODE -ne 0) { $optionalExit = $LASTEXITCODE }
} catch {
    $optionalExit = 1
}
if ($optionalExit -ne 0) {
    Write-Host "Optional deps skipped (editdistance needs C++ build tools on Windows/Python 3.13). CER/WER uses pure-Python fallback."
}
python -m ipykernel install --user --name sinhala-ocr --display-name "Sinhala OCR"

$dirs = @(
    "data",
    "data/synthetic/train",
    "data/synthetic/val",
    "data/synthetic/test",
    "data/uploads",
    "data/real/images",
    "data/real/labels",
    "data/debug",
    "models"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}
@(
    "data/.gitkeep",
    "data/uploads/.gitkeep",
    "data/real/.gitkeep",
    "data/real/images/.gitkeep",
    "data/real/labels/.gitkeep",
    "data/debug/.gitkeep",
    "models/.gitkeep"
) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType File -Path $_ -Force | Out-Null }
}

Write-Host ""

Write-Host ""
Write-Host "Checking PyTorch / CUDA..."
$cudaScript = @"
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"@
$cudaCheck = python -c $cudaScript 2>&1
Write-Host $cudaCheck
$isCpuBuild = ($cudaCheck -match "\+cpu") -or ($cudaCheck -match "cpu_only")
$cudaFalse = $cudaCheck -match "cuda_available False"
if ($isCpuBuild -or $cudaFalse) {
    Write-Host ""
    Write-Host "CPU-only PyTorch (CUDA not available). For RTX 4060 / NVIDIA GPU on Windows:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/install_cuda_torch.ps1"
    Write-Host "Or manually (~2.5 GB download):"
    Write-Host "  pip uninstall torch torchvision torchaudio -y"
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    Write-Host "Python 3.13: CUDA wheels exist on cu124; if install fails, use scripts/setup_gpu_venv.ps1 (Python 3.11/3.12)."
}

Write-Host "Setup complete."
Write-Host "Next steps:"
Write-Host "  1. cd `"$ProjectRoot`""
Write-Host "  2. jupyter notebook notebooks/local_pipeline.ipynb"
Write-Host ""
Write-Host "Optional: re-run with -CreateVenv to use a project virtual environment."



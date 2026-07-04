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
$cudaCheck = python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 2>&1
Write-Host $cudaCheck
if ($cudaCheck -match "\+cpu" -and $cudaCheck -match "False") {
    Write-Host ""
    Write-Host "CPU-only PyTorch detected (no CUDA). For NVIDIA GPU training (e.g. RTX 4060), run:"
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
}

Write-Host "Setup complete."
Write-Host "Next steps:"
Write-Host "  1. cd `"$ProjectRoot`""
Write-Host "  2. jupyter notebook notebooks/local_pipeline.ipynb"
Write-Host ""
Write-Host "Optional: re-run with -CreateVenv to use a project virtual environment."


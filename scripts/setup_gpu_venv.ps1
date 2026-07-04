# Create a Python 3.11/3.12 venv with CUDA PyTorch for local GPU training
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_venv.ps1

param(
    [ValidateSet("3.12", "3.11", "auto")]
    [string]$PythonVersion = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

function Resolve-PythonLauncher {
    param([string]$Ver)
    if ($Ver -eq "auto") {
        foreach ($v in @("3.12", "3.11")) {
            $out = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return @{ Version = $v; Executable = $out.Trim() } }
        }
        return $null
    }
    $out = & py "-$Ver" -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    return @{ Version = $Ver; Executable = $out.Trim() }
}

$py = Resolve-PythonLauncher -Ver $PythonVersion
if (-not $py) {
    Write-Host "No Python 3.12 or 3.11 found via the py launcher."
    Write-Host "Download from https://www.python.org/downloads/ and re-run this script."
    Write-Host "If you already use Python 3.13, try: powershell -ExecutionPolicy Bypass -File scripts/install_cuda_torch.ps1"
    exit 1
}

Write-Host "Using Python $($py.Version): $($py.Executable)"
$venvPath = ".venv-gpu"
if (-not (Test-Path $venvPath)) {
    & $py.Executable -m venv $venvPath
    Write-Host "Created $venvPath"
}

& "$venvPath\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y 2>$null | Out-Null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "Verify GPU:"
python -c "import torch; print('version:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "Activate before training:  .\.venv-gpu\Scripts\Activate.ps1"

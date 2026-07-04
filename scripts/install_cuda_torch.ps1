# Install CUDA-enabled PyTorch on Windows (NVIDIA RTX 4060 and similar)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/install_cuda_torch.ps1
# Optional: -CudaIndex cu121 | cu118 if cu124 is unavailable for your Python version

param(
    [ValidateSet("cu124", "cu121", "cu118")]
    [string]$CudaIndex = "cu124"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python $pyVer detected."

function Test-CudaTorchWheel {
    param([string]$Index)
    $probe = pip index versions torch --index-url "https://download.pytorch.org/whl/$Index" 2>&1
    return ($LASTEXITCODE -eq 0) -and ($probe -match "\+${Index}")
}

$indexes = @($CudaIndex)
if ($CudaIndex -eq "cu124") { $indexes += @("cu121", "cu118") }
elseif ($CudaIndex -eq "cu121") { $indexes += @("cu118") }

$installed = $false
foreach ($idx in $indexes | Select-Object -Unique) {
    Write-Host ""
    Write-Host "Trying PyTorch index: $idx (~2.5 GB download; may take 20-40 minutes on slow links)..."
    pip uninstall torch torchvision torchaudio -y 2>$null | Out-Null
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$idx"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Install from $idx failed."
        continue
    }
    $check = python -c "import torch; ok=torch.cuda.is_available(); print(torch.__version__); print('CUDA', ok); print(torch.cuda.get_device_name(0) if ok else 'N/A')" 2>&1
    Write-Host $check
    if ($check -match "\+$idx" -or ($check -notmatch "\+cpu" -and $check -match "CUDA True")) {
        $installed = $true
        break
    }
    if ($check -match "\+cpu") {
        Write-Warning "CPU-only wheel installed; trying next CUDA index..."
    }
}

if (-not $installed) {
    Write-Host ""
    Write-Host "Could not install a CUDA PyTorch wheel for Python $pyVer."
    Write-Host "Install Python 3.12 or 3.11, then run:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_venv.ps1"
    exit 1
}

Write-Host ""
Write-Host "CUDA PyTorch install complete."

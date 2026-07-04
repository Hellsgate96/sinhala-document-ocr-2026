# Train CRNN locally without Jupyter (uses configs/local.yaml)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

python scripts/generate_data.py --config configs/local.yaml --large --num-samples 5000
python -m src.recognition.train --config configs/local.yaml

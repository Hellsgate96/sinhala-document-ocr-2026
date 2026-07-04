# Fine-tune CRNN on the Kanyawee poem lines (real crops + ground-truth labels).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python scripts/prepare_poem_dataset.py --image data/uploads/test2.png
python -m src.recognition.train --config configs/finetune.yaml --resume models/crnn_best.pth --extra-labels data/real/labels/poem_kanyawee.txt

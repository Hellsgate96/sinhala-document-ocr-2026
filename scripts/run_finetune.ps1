# Fine-tune CRNN on the Kanyawee poem lines (real crops + ground-truth labels).
# Ground-truth labels (UTF-8, tab-separated): data/real/labels/poem_kanyawee.txt
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

# Re-crop optional: existing data/real/images/poem_line_*.png are not overwritten.
python scripts/prepare_poem_dataset.py --image data/uploads/test2.png
python -m src.recognition.train --config configs/finetune.yaml --resume models/crnn_best.pth --extra-labels data/real/labels/poem_kanyawee.txt

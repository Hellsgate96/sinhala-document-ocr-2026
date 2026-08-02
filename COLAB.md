# Running on Google Colab (via Google Drive)

Same scientific demo as `notebooks/local_pipeline.ipynb`, with Drive mount and
dependency install. The trained checkpoint (`models/crnn_best.pth`, ≈120 MB) is
**gitignored**, so it must be present in the Drive copy.

## Recommended: project already on Drive

If you copied the whole `sinhala-document-ocr` folder to Google Drive:

1. Confirm the Drive folder contains at least:
   - `models/crnn_best.pth`
   - `models/charset.json`
   - `src/`, `configs/`, `requirements.txt`
   - `notebooks/colab_pipeline.ipynb`
   - `data/eval_real/print_photos/` (demo page + `.gt.txt`)
   - `data/metrics/` (training curves)
2. In Drive, open `notebooks/colab_pipeline.ipynb` → **Open with** → **Google Colaboratory**.
3. Runtime → **Change runtime type** → **GPU** (T4 is fine).
4. In the **Mount Drive** cell, set:

   ```python
   DRIVE_PROJECT_DIR = "/content/drive/MyDrive/sinhala-document-ocr"
   ```

   Use your real Drive path if the folder name or location differs
   (e.g. `/content/drive/MyDrive/Projects/sinhala-document-ocr`).
5. (Optional) In **Config**, set a real test image:

   ```python
   TEST_IMAGE_PATH = "data/uploads/my_test.jpg"
   ```

   Put the file under that path in the Drive project first, or use an absolute
   path such as `/content/drive/MyDrive/sinhala-document-ocr/data/uploads/my_test.jpg`.
   Leave empty to run the bundled demo page.
6. **Runtime → Run all.**

The notebook mounts Drive, installs deps from `requirements.txt`, loads
`models/crnn_best.pth`, runs detection + recognition, prints evaluation metrics,
and plots training curves from `data/metrics/`.

## Test a real image

**Path (preferred for Run all)**

1. Upload a photo/scan into the Drive project, e.g.
   `My Drive/sinhala-document-ocr/data/uploads/my_test.jpg`.
2. Optionally add a sidecar GT file next to it (`my_test.gt.txt`, one line per
   expected text line) for CER/WER.
3. Set in **Config**:

   ```python
   TEST_IMAGE_PATH = "data/uploads/my_test.jpg"
   ```

4. Runtime → Run all (or re-run from **Config** downward).

**Browser upload (optional)**

In **Config**, leave `TEST_IMAGE_PATH = ""` and set `USE_UPLOAD = True`, then run
the **Run OCR** cell and choose a file. Cancel falls back to the bundled demo.

## Config knobs

| Variable | Meaning |
|---|---|
| `DRIVE_PROJECT_DIR` | Drive folder that contains `models/crnn_best.pth` |
| `DRIVE_ZIP_PATH` | optional zip to unzip once into that folder |
| `TEST_IMAGE_PATH` | page image (relative to project root, or absolute Drive path) |
| `USE_UPLOAD` | `True` → Colab file picker when path is empty |

Ground truth is auto-resolved from a sidecar `<image>.gt.txt` (no separate flag).

## Alternative: zip upload

If you prefer a zip instead of a full folder copy, include at least:

| Must include | Why |
|---|---|
| `models/crnn_best.pth` | required checkpoint (gitignored) |
| `models/charset.json` | character map |
| `src/`, `configs/`, `requirements.txt` | code + deps |
| `notebooks/colab_pipeline.ipynb` | Colab entry point |
| `data/eval_real/print_photos/` | demo images + GT |
| `data/metrics/` | train curves + held-out CER chart |

Example (PowerShell, from the repo root):

```powershell
Compress-Archive -Path src,configs,notebooks,models,requirements.txt,fonts,data\eval_real,data\metrics,README.md,COLAB.md,RESULTS.md `
  -DestinationPath sinhala-document-ocr-colab.zip -Force
```

Set `DRIVE_ZIP_PATH` in the Mount Drive cell to the zip on Drive; the notebook
unzips once if the checkpoint is missing.

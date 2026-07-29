# Running on Google Colab (via Google Drive)

The trained checkpoint (`models/crnn_best.pth`, ≈120 MB) is **gitignored**, so a
bare `git clone` is not enough to run inference. The easiest examiner path is:
zip the project (including the checkpoint), upload to Google Drive, open the
Colab notebook.

## 1. What to zip / upload

From the project root, include at least:

| Must include | Why |
|---|---|
| `models/crnn_best.pth` | **required** — delivered recogniser (gitignored) |
| `models/charset.json` | character map (must match the checkpoint) |
| `src/` | all pipeline code |
| `configs/` | `local.yaml` / `default.yaml` inference settings |
| `notebooks/colab_pipeline.ipynb` | the Colab entry point |
| `requirements.txt` | deps |
| `data/eval_real/print_photos/` | demo images + `.gt.txt` for CER/WER |
| `fonts/` (optional) | Sinhala fonts; Colab can also `apt-get` Noto |

You do **not** need synthetic training data, old checkpoints, or `data/debug/`.

Example (PowerShell, from the repo root):

```powershell
Compress-Archive -Path src,configs,notebooks,models,requirements.txt,fonts,data\eval_real,README.md,COLAB.md,RESULTS.md `
  -DestinationPath sinhala-document-ocr-colab.zip -Force
```

Upload `sinhala-document-ocr-colab.zip` to Google Drive, e.g.:

```
My Drive/
  sinhala-document-ocr/
    sinhala-document-ocr-colab.zip
```

Then unzip once in Colab (the notebook does this), or unzip on Drive into:

```
My Drive/
  sinhala-document-ocr/
    models/crnn_best.pth
    models/charset.json
    src/
    configs/
    notebooks/colab_pipeline.ipynb
    ...
```

## 2. Open the notebook in Colab

1. In Drive, right-click `notebooks/colab_pipeline.ipynb` → **Open with** → **Google Colaboratory**.
2. Or upload the notebook to Colab and set `DRIVE_PROJECT_DIR` in the first config cell.
3. Runtime → **Change runtime type** → **GPU** (T4 is fine).

## 3. Config cell knobs

| Variable | Meaning |
|---|---|
| `SETUP_MODE` | `"drive"` (default) or `"github"` |
| `DRIVE_PROJECT_DIR` | folder on Drive that contains `models/crnn_best.pth` |
| `TEST_IMAGE_PATH` | path to a page image (Drive path or under the project) |
| `GT_PATH` | optional ground-truth `.gt.txt` (one line per text line) |

## 4. Run

**Runtime → Run all.** The notebook will:

1. Mount Drive (or clone GitHub and copy the checkpoint from Drive)
2. Install dependencies + a Sinhala font
3. Load `models/crnn_best.pth` + `models/charset.json`
4. Detect lines, recognise with beam+LM + matra post-correction
5. Print the **Evaluation metrics** block (CER/WER when GT exists)

## 5. Adding your own test image

1. Upload a photo/scan to Drive (e.g. `My Drive/sinhala-document-ocr/data/uploads/my_page.jpg`).
2. Optionally add `my_page.gt.txt` next to it (one GT line per expected text line).
3. Set `TEST_IMAGE_PATH` to that file and re-run.

## Alternative: GitHub clone + checkpoint on Drive

If the code is public but the `.pth` stays private on Drive:

1. Set `SETUP_MODE = "github"`.
2. Keep `models/crnn_best.pth` (and `charset.json`) under `DRIVE_PROJECT_DIR/models/`.
3. The notebook clones the repo, then copies those two files into the local `models/` folder.

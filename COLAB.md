# Running on Google Colab (via Google Drive)

Two Colab entry points:

* **Inference demo:** `notebooks/colab_pipeline.ipynb` (same as
  `notebooks/local_pipeline.ipynb`) — loads `models/crnn_best.pth` and runs OCR.
* **How we trained:** `notebooks/training_methodology.ipynb` — dataset counts,
  configs, parsed logs/curves, eval table. Prose: `docs/Methodology.md`.

The trained checkpoint (`models/crnn_best.pth`, ≈120 MB) is **gitignored**, so
it must be present in the Drive copy for the **demo**. The methodology notebook
still runs counts and curves without it.

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
5. **Runtime → Run all.**

   By default **Config** has `USE_UPLOAD = True` and an empty `TEST_IMAGE_PATH`,
   so the **Run OCR** cell opens a browser file picker
   (`google.colab.files.upload()`). Choose any Sinhala page photo/scan; it is
   saved under `data/uploads/` and OCR runs on it. Cancel the picker to use the
   bundled demo page.

The notebook mounts Drive, installs deps from `requirements.txt`, loads
`models/crnn_best.pth`, runs detection + recognition, prints evaluation metrics,
and plots training curves from `data/metrics/`. Default **Run all** is
**inference-only** (`RUN_TRAIN = False` in the optional appendix).

## Test a real image

**Browser upload (preferred for live demo)**

1. Leave `TEST_IMAGE_PATH = ""` and `USE_UPLOAD = True` (defaults).
2. Run **Run OCR** (or Runtime → Run all) and pick an image in the Colab upload UI.
3. Optional: add a sidecar GT next to the saved file
   (`data/uploads/<name>.gt.txt`, one line per expected text line) and re-run
   metrics.

**Path (no picker)**

1. Put a photo/scan in the Drive project, e.g.
   `My Drive/sinhala-document-ocr/data/uploads/my_test.jpg`.
2. In **Config**:

   ```python
   TEST_IMAGE_PATH = "data/uploads/my_test.jpg"
   USE_UPLOAD = False  # optional; path wins anyway when set
   ```

3. Runtime → Run all (or re-run from **Config** downward).

## Optional: Training (viva / appendix)

At the end of the notebook, **Optional: Training** is off by default
(`RUN_TRAIN = False`) so examiners are not blocked by a long train.

| Mode | When | What it does |
|---|---|---|
| `TRAIN_MODE = "short"` (default when enabled) | Panel asks how training works | Few epochs + small synthetic sample; prints epoch / train_loss / val CER; saves best to `models/crnn_best.pth` (resume preserves prior best CER). **Not** the delivered full schedule. |
| `TRAIN_MODE = "full"` | You want a real continue-train | Uses `TRAIN_CONFIG` (`configs/local.yaml` or `configs/mix_web.yaml`); can take hours on Colab free GPU and needs `data/synthetic/` (+ extras for mix_web). |

Set `RUN_TRAIN = True`, choose mode/config, and run that cell only. Full training
details are in `docs/Project_Report.md`.

## Config knobs

| Variable | Meaning |
|---|---|
| `DRIVE_PROJECT_DIR` | Drive folder that contains `models/crnn_best.pth` |
| `DRIVE_ZIP_PATH` | optional zip to unzip once into that folder |
| `TEST_IMAGE_PATH` | page image (relative to project root, or absolute Drive path); empty → upload/demo |
| `USE_UPLOAD` | `True` (default) → Colab file picker when path is empty |
| `RUN_TRAIN` | `False` (default) → skip optional training appendix |

Ground truth is auto-resolved from a sidecar `<image>.gt.txt` (no separate flag).

## Alternative: zip upload

If you prefer a zip instead of a full folder copy, include at least:

| Must include | Why |
|---|---|
| `models/crnn_best.pth` | required checkpoint (gitignored) |
| `models/charset.json` | character map |
| `src/`, `configs/`, `requirements.txt` | code + deps |
| `notebooks/colab_pipeline.ipynb` | Colab inference demo |
| `notebooks/training_methodology.ipynb` | How we trained (optional on Colab) |
| `data/eval_real/print_photos/` | demo images + GT |
| `data/metrics/` | train curves + held-out CER chart |

Example (PowerShell, from the repo root):

```powershell
Compress-Archive -Path src,configs,notebooks,models,requirements.txt,fonts,data\eval_real,data\metrics,README.md,COLAB.md,RESULTS.md `
  -DestinationPath sinhala-document-ocr-colab.zip -Force
```

Set `DRIVE_ZIP_PATH` in the Mount Drive cell to the zip on Drive; the notebook
unzips once if the checkpoint is missing.

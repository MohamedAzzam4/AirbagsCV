# Google Colab Training Guide — AirbagsCV

This guide lets you train the EfficientAD anomaly-detection model on Google
Colab using **the repository's own scripts**. The notebook does NOT
reimplement training — it just clones the repo, installs dependencies, mounts
Drive, and calls `scripts/prepare_aitex.py`, `scripts/train_models.py`, and
`scripts/benchmark_inference.py` from the command line.

> **The repo is the source of truth.** If you find yourself copy-pasting
> training logic into a Colab cell, stop. Add the logic to a script in the
> repo, commit it, and call the script from Colab.

---

## 0. Before you start

- A Google account with Google Drive.
- The raw AITEX dataset, obtained from
  [AITEX AFID](https://www.aitex.es/afid/) (registration required) or the
  [Kaggle mirror](https://www.kaggle.com/datasets/rmshashi/fabric-defect-dataset).
  Upload it to your Google Drive at `/MyDrive/datasets/AITEX/`. The folder
  must contain `NODefect_images/`, `Defect_images/`, and `Mask_images/`.
- The Imagenette dataset (EfficientAD requires it for its penalty term).
  Download `imagenette2-160.tgz` from
  https://github.com/fastai/imagenette (~1.5 GB) and extract it to
  `/MyDrive/datasets/imagenette/`. The folder must contain
  `train/<class>/*.JPEG` and `val/<class>/*.JPEG`.

---

## 1. Select GPU runtime

In the Colab menu bar:

```
Runtime > Change runtime type > Hardware accelerator > GPU
```

A free T4 GPU is enough for EfficientAD-small at 256×256. A full 70-epoch run
takes roughly 30–60 minutes on a T4.

**Verify the GPU is visible:**

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

If you see `CUDA available: False`, you did not select a GPU runtime. Repeat step 1.

---

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Quick sanity check — should print your AITEX folder.
import os
print(os.listdir('/content/drive/MyDrive/datasets/AITEX'))
```

If the `ls` fails, you have not uploaded AITEX to the path in section 0.
Either upload it or change the paths below.

---

## 3. Clone or update the repo

```bash
%%bash
if [ -d /content/AirbagsCV ]; then
  cd /content/AirbagsCV
  git pull
else
  cd /content
  git clone https://github.com/MohamedAzzam4/AirbagsCV.git
fi
```

```python
%cd /content/AirbagsCV
```

---

## 4. Install dependencies

Colab ships with `torch` + `torchvision` preinstalled (CUDA build). Do NOT
reinstall torch from the default index — you will get a CPU build and lose
the GPU.

```bash
%%bash
cd /content/AirbagsCV
pip install -r requirements-colab.txt
```

If for some reason torch is missing or broken, run this **before** the
command above:

```bash
%%bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**If CUDA/PyTorch install fails:**

1. Check `nvcc --version` and `nvidia-smi` for the CUDA version.
2. Match the wheel index URL: `cu121` for CUDA 12.1, `cu118` for CUDA 11.8.
3. After install, run `python -c "import torch; print(torch.cuda.is_available())"`.
   If it prints `False`, you have a CPU-only torch. Uninstall and reinstall
   with the correct index URL.

---

## 5. Define paths

```python
import os
os.environ['REPO_DIR']        = '/content/AirbagsCV'
os.environ['RAW_AITEX_DIR']   = '/content/drive/MyDrive/datasets/AITEX'
os.environ['IMAGENETTE_DIR']  = '/content/drive/MyDrive/datasets/imagenette/train'
os.environ['PREPARED_DATA_DIR'] = '/content/airbagcv_data/aitex_patches'
os.environ['RUNS_DIR']        = '/content/drive/MyDrive/AirbagsCV/runs'
os.environ['RESULTS_DIR']     = '/content/drive/MyDrive/AirbagsCV/results'

# Create local + Drive dirs
for d in ['PREPARED_DATA_DIR', 'RUNS_DIR', 'RESULTS_DIR']:
    os.makedirs(os.environ[d], exist_ok=True)

# Show
for k, v in os.environ.items():
    if k in ['REPO_DIR','RAW_AITEX_DIR','IMAGENETTE_DIR','PREPARED_DATA_DIR','RUNS_DIR','RESULTS_DIR']:
        print(f"{k:25s} {v}")
```

**Verify dataset is reachable:**

```python
import os
required = ['NODefect_images', 'Defect_images', 'Mask_images']
for r in required:
    p = os.path.join(os.environ['RAW_AITEX_DIR'], r)
    ok = os.path.isdir(p)
    print(f"  {'OK ' if ok else 'MISSING'}  {p}")
    if not ok:
        raise FileNotFoundError(f"Missing {r}. Check your Drive path.")
```

If this errors, you have the wrong path. Either fix `RAW_AITEX_DIR` or upload
AITEX to that location.

---

## 6. Run the smoke test

```bash
%%bash
cd /content/AirbagsCV
python scripts/smoke_test.py
```

You should see `[OK]` for every line and `PASS` at the bottom. If anything
fails, fix it before proceeding — Colab will not magically make a broken
install work.

---

## 7. Prepare AITEX data

```bash
%%bash
cd /content/AirbagsCV
python scripts/prepare_aitex.py \
  --source "$RAW_AITEX_DIR" \
  --output "$PREPARED_DATA_DIR" \
  --patch-size 256 \
  --blank-threshold 0.5 \
  --seed 42 \
  --overwrite
```

Expected output: a summary like

```
normal_patches_written   1792
defect_patches_written   185
blank_patches_skipped    ...
```

The blank-patch filter is critical — the original AITEX images have solid
black border columns that would otherwise poison the training distribution.

---

## 8. Train EfficientAD

```bash
%%bash
cd /content/AirbagsCV
python scripts/train_models.py \
  --model efficientad \
  --dataset aitex \
  --data-dir "$PREPARED_DATA_DIR" \
  --output-dir "$RUNS_DIR" \
  --epochs 70 \
  --batch-size 1 \
  --num-workers 2 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed \
  --seed 42 \
  --imagenet-dir "$IMAGENETTE_DIR"
```

Notes:
- `--batch-size 1` is REQUIRED by Anomalib 2.0.0's EfficientAd. The script
  will warn and override if you pass anything else.
- `--precision 16-mixed` gives ~2× speedup on T4 with no accuracy loss.
- `--epochs 70` is the paper's recipe. Budget ~30–60 min on T4.
- `--imagenet-dir` is REQUIRED by EfficientAD's penalty term.

The script prints the checkpoint path prominently at the end:

```
CHECKPOINT_PATH=/content/drive/MyDrive/AirbagsCV/runs/efficientad_aitex/.../model.ckpt
```

It also writes:
- `metrics.json` next to the checkpoint (image_AUROC, pixel_AUROC, etc.)
- a row appended to `$RUNS_DIR/benchmark_results.csv`

If the cell disconnects before training finishes (Colab free tier
disconnects after ~12 h idle, but a single training run of 70 epochs is
typically well within limits), the checkpoint is on Drive — use step 9 to
resume.

---

## 9. Resume training from a checkpoint

If Colab disconnected mid-training, find your latest checkpoint on Drive and
resume:

```python
import glob, os
run_dir = os.path.join(os.environ['RUNS_DIR'], 'efficientad_aitex')
ckpts = sorted(glob.glob(f'{run_dir}/**/*.ckpt', recursive=True),
               key=lambda p: os.path.getmtime(p))
assert ckpts, f"No checkpoint found under {run_dir}"
CKPT = ckpts[-1]
print("Latest checkpoint:", CKPT)
```

```bash
%%bash -s "$CKPT"
cd /content/AirbagsCV
python scripts/train_models.py \
  --model efficientad \
  --dataset aitex \
  --data-dir "$PREPARED_DATA_DIR" \
  --output-dir "$RUNS_DIR" \
  --epochs 70 \
  --batch-size 1 \
  --num-workers 2 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed \
  --seed 42 \
  --imagenet-dir "$IMAGENETTE_DIR" \
  --resume-from-checkpoint "$1"
```

The `--resume-from-checkpoint` argument is passed to Lightning's
`engine.fit(ckpt_path=...)`. Lightning restores the optimizer state, the LR
scheduler, and the epoch counter.

---

## 10. Run honest inference benchmark

After training (or using the committed checkpoint), measure real per-image
latency:

```bash
%%bash
cd /content/AirbagsCV
python scripts/benchmark_inference.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --model efficientad \
  --data-dir "$PREPARED_DATA_DIR" \
  --output-csv "$RESULTS_DIR/latency.csv" \
  --device cuda \
  --warmup 20 \
  --iterations 200
```

To set `$CHECKPOINT_PATH` from Python:

```python
import glob, os
run_dir = os.path.join(os.environ['RUNS_DIR'], 'efficientad_aitex')
ckpts = sorted(glob.glob(f'{run_dir}/**/*.ckpt', recursive=True),
               key=lambda p: os.path.getmtime(p))
os.environ['CHECKPOINT_PATH'] = ckpts[-1] if ckpts else \
    'results/efficientad_aitex/EfficientAd/aitex/latest/weights/lightning/model.ckpt'
print("CHECKPOINT_PATH =", os.environ['CHECKPOINT_PATH'])
```

Expected output:

```
==================== INFERENCE BENCHMARK ====================
  device                            cuda
  mean_ms_per_image                 5.0 - 30.0   (typical T4 range)
  p50_ms_per_image                  ...
  p95_ms_per_image                  ...
  p99_ms_per_image                  ...
  throughput_images_per_s           30 - 200
  note                              Patch/image-level only. NOT line-scan.
============================================================
```

The `note` field is deliberately worded to prevent anyone from claiming
this is industrial line-scan throughput.

---

## 11. Verify all outputs are on Drive

```python
import os
for label, path in [
    ('Runs dir',   os.environ['RUNS_DIR']),
    ('Results dir',os.environ['RESULTS_DIR']),
]:
    print(f"\n=== {label}: {path} ===")
    for root, dirs, files in os.walk(path):
        depth = root.replace(path, '').count(os.sep)
        if depth > 3:
            continue
        indent = '  ' * depth
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:5]:
            print(f"{indent}  {f}")
        if len(files) > 5:
            print(f"{indent}  ... and {len(files)-5} more")
```

You should see:
- `efficientad_aitex/.../model.ckpt`
- `efficientad_aitex/metrics.json`
- `benchmark_results.csv`
- `latency.csv` + `latency.json`

All of these persist on Drive and survive Colab disconnects.

---

## 12. Zip final outputs (optional)

```bash
%%bash
cd /content
zip -r /content/drive/MyDrive/AirbagsCV/airbagcv_run_outputs.zip \
  "$RUNS_DIR" \
  "$RESULTS_DIR"
```

---

## 13. Common failure modes — clear messages and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA available: False` | GPU runtime not selected | Runtime > Change runtime type > GPU |
| `FileNotFoundError: No NODefect_images` | Wrong `RAW_AITEX_DIR` | Check Drive path; fix env var in step 5 |
| `FileNotFoundError: Couldn't find any class folder in datasets/imagenette` | Missing or wrong-format imagenette | Download `imagenette2-160.tgz`, extract, point `--imagenet-dir` at the `train/` subdir |
| `TypeError: Folder.__init__() got an unexpected keyword argument 'image_size'` | Anomalib version mismatch | Ensure `anomalib==2.0.0` is installed (the requirements file pins it) |
| `ValueError: train_batch_size for EfficientAd should be 1` | Wrong `--batch-size` | Use `--batch-size 1`. The script will warn and override anyway. |
| Colab disconnects mid-training | Idle timeout (free tier) | Reconnect, find latest checkpoint on Drive, run step 9 (resume) |
| `RuntimeError: CUDA out of memory` | Batch too large or other process using GPU | Use `--batch-size 1` (required anyway). Restart runtime if memory is fragmented. |
| `ModuleNotFoundError: No module named 'anomalib'` | Step 4 skipped or failed | Re-run `pip install -r requirements-colab.txt` |
| Demo `cv2.addWeighted` size mismatch | You're running the OLD version of the repo | `git pull` to get the fix |

---

## 14. Run the demo locally (optional)

You cannot run the Gradio demo directly in Colab (Colab doesn't expose ports
cleanly). To use the demo:

1. Download your trained checkpoint from Drive.
2. Clone the repo on your local machine.
3. Place the checkpoint under `results/efficientad_aitex/EfficientAd/aitex/latest/weights/lightning/model.ckpt`.
4. Run `python demo/app.py` and open `http://127.0.0.1:7860/`.

Alternatively, use `gradio tunneling`:

```python
# In Colab, after training:
import sys
sys.path.insert(0, '/content/AirbagsCV/demo')
from app import app
app.launch(share=True)  # produces a public URL
```

This is not recommended for factory demos (URL is temporary and public).

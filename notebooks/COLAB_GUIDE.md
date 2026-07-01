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

You need on Google Drive:

- **Raw AITEX** at `/content/drive/MyDrive/datasets/AITEX/` containing
  `NODefect_images/`, `Defect_images/`, `Mask_images/`. Obtain AITEX from
  [AITEX AFID](https://www.aitex.es/afid/) (registration required) or the
  [Kaggle mirror](https://www.kaggle.com/datasets/rmshashi/fabric-defect-dataset).
- **Imagenette** at `/content/drive/MyDrive/datasets/imagenette/train/`
  containing class subdirectories with `.JPEG` files. EfficientAD requires
  this for its penalty term. Download `imagenette2-160.tgz` from
  https://github.com/fastai/imagenette (~1.5 GB) and extract it.

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
aitex_path = '/content/drive/MyDrive/datasets/AITEX'
if os.path.isdir(aitex_path):
    print('AITEX contents:', os.listdir(aitex_path))
else:
    raise FileNotFoundError(
        f'{aitex_path} does not exist. Upload AITEX first (see section 0).'
    )
```

If the `raise` fires, you have not uploaded AITEX to the path in section 0.
Either upload it or change `RAW_AITEX_DIR` below.

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
pip install -r requirements-colab.txt 2>&1 | tail -10
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

**If you see `ImportError: cannot import name 'xxx' from 'gradio'` or
`OSError: Could not find file ...` from open_clip:**

This is almost certainly a **HuggingFace stack version conflict**. Anomalib
2.0.0's `open_clip` dependency pulls in `huggingface-hub`, `transformers`,
and `tokenizers`, and newer versions of these packages break both gradio
imports and open_clip weight loading. The `requirements-colab.txt` file
pins the verified-compatible versions:

```
huggingface-hub==0.25.2
transformers==4.44.2
tokenizers==0.19.1
```

If Colab's preinstalled versions override these (check with
`pip show huggingface-hub transformers tokenizers`), force-reinstall them:

```bash
%%bash
pip install --force-reinstall --no-deps \
  huggingface-hub==0.25.2 transformers==4.44.2 tokenizers==0.19.1
```

Then `Runtime > Restart session` and re-run the smoke test (step 6).

---

## 5. Define paths

These are the **default Drive-persistent paths**. All outputs land on Drive
so they survive Colab disconnects.

```python
import os
os.environ['REPO_DIR']          = '/content/AirbagsCV'
os.environ['RAW_AITEX_DIR']     = '/content/drive/MyDrive/datasets/AITEX'
os.environ['IMAGENETTE_DIR']    = '/content/drive/MyDrive/datasets/imagenette/train'
os.environ['PREPARED_DATA_DIR'] = '/content/drive/MyDrive/AirbagsCV/prepared/aitex_patches'
os.environ['RUNS_DIR']          = '/content/drive/MyDrive/AirbagsCV/runs'
os.environ['RESULTS_DIR']       = '/content/drive/MyDrive/AirbagsCV/results'
os.environ['CACHE_DIR']         = '/content/drive/MyDrive/AirbagsCV/cache'

# Create all Drive dirs
for d in ['PREPARED_DATA_DIR', 'RUNS_DIR', 'RESULTS_DIR', 'CACHE_DIR']:
    os.makedirs(os.environ[d], exist_ok=True)

# Show
for k, v in os.environ.items():
    if k in ['REPO_DIR','RAW_AITEX_DIR','IMAGENETTE_DIR','PREPARED_DATA_DIR',
             'RUNS_DIR','RESULTS_DIR','CACHE_DIR']:
        print(f"{k:25s} {v}")
```

**Verify datasets are reachable:**

```python
import os
# AITEX
required = ['NODefect_images', 'Defect_images', 'Mask_images']
for r in required:
    p = os.path.join(os.environ['RAW_AITEX_DIR'], r)
    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"Missing {p}. Check RAW_AITEX_DIR or upload AITEX to Drive."
        )
print('AITEX OK')

# Imagenette
if not os.path.isdir(os.environ['IMAGENETTE_DIR']):
    raise FileNotFoundError(
        f"Missing imagenette at {os.environ['IMAGENETTE_DIR']}. "
        f"Download imagenette2-160.tgz from https://github.com/fastai/imagenette "
        f"and extract to /content/drive/MyDrive/datasets/imagenette/."
    )
# Sanity: must have class subdirs
subdirs = [d for d in os.listdir(os.environ['IMAGENETTE_DIR'])
           if os.path.isdir(os.path.join(os.environ['IMAGENETTE_DIR'], d))]
if not subdirs:
    raise FileNotFoundError(
        f"{os.environ['IMAGENETTE_DIR']} exists but has no class subdirs. "
        f"It must be ImageFolder layout: imagenette/train/<class>/*.JPEG"
    )
print(f'Imagenette OK ({len(subdirs)} classes)')
```

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

This writes the Anomalib `Folder` layout to `PREPARED_DATA_DIR` (on Drive by
default — see Mode A/B below for alternatives).

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

## 8. Choose Mode A or Mode B (Drive I/O strategy)

Google Drive can be slow for many small files (the prepared AITEX set has
~2,000 patches). Two modes are supported.

### Mode A — Persistent (default, safer)

Train directly from `PREPARED_DATA_DIR` on Drive. **Safer after disconnect**
(you can resume from any checkpoint). **Slower training** (Drive I/O per
batch). Use this for short runs or first-time setup.

To use Mode A, do nothing extra — proceed to step 9.

### Mode B — Faster local-cache

Copy the prepared dataset from Drive to `/content` at the start of the
runtime, train from there, and save checkpoints/results **back to Drive**.
**Faster training** (local SSD I/O). Same Drive safety for checkpoints. Use
this for full 70-epoch runs.

```python
# === Mode B only: copy prepared data from Drive to /content ===
import os, shutil
LOCAL_DATA_DIR = '/content/aitex_patches_local'
if os.path.exists(LOCAL_DATA_DIR):
    print(f'{LOCAL_DATA_DIR} already exists; reusing.')
else:
    print(f'Copying {os.environ["PREPARED_DATA_DIR"]} -> {LOCAL_DATA_DIR} ...')
    shutil.copytree(os.environ['PREPARED_DATA_DIR'], LOCAL_DATA_DIR)
    print('Done.')

# Override PREPARED_DATA_DIR for the rest of the runtime
os.environ['PREPARED_DATA_DIR_DRIVE'] = os.environ['PREPARED_DATA_DIR']  # keep Drive copy for reference
os.environ['PREPARED_DATA_DIR'] = LOCAL_DATA_DIR
print('PREPARED_DATA_DIR is now:', os.environ['PREPARED_DATA_DIR'])
print('  (checkpoints/results still go to RUNS_DIR/RESULTS_DIR on Drive)')
```

**Optional: also zip the prepared dataset to Drive for fast Mode B restore
in future runtimes.** This is a one-time setup; subsequent runtimes unzip
instead of re-running `prepare_aitex.py`:

```python
# === Optional: create a zip of the prepared dataset on Drive ===
import os, subprocess
zip_path = os.path.join(os.environ['CACHE_DIR'], 'aitex_patches.zip')
if not os.path.exists(zip_path):
    print(f'Creating {zip_path} (this takes a few minutes) ...')
    subprocess.run(
        ['zip', '-r', '-q', zip_path, '.'],
        cwd=os.environ['PREPARED_DATA_DIR_DRIVE'],  # the Drive copy
        check=True,
    )
    print('Done.')
else:
    print(f'{zip_path} already exists; reusing.')

# To restore in a future runtime:
#   os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
#   subprocess.run(['unzip', '-q', zip_path, '-d', LOCAL_DATA_DIR], check=True)
```

---

## 9. Train EfficientAD (from scratch)

EfficientAD **requires `--batch-size 1`** (Anomalib 2.0.0 hard constraint;
the script warns and overrides any other value). EfficientAD **also requires
`--imagenet-dir`** pointing at imagenette.

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

At the end, the script prints all key paths:

```
======================================================================
TRAINING COMPLETE — KEY PATHS
======================================================================
RUN_DIR=/content/drive/MyDrive/AirbagsCV/runs/efficientad_aitex
LAST_CHECKPOINT_PATH=/content/drive/MyDrive/AirbagsCV/runs/efficientad_aitex/weights/lightning/last.ckpt
BEST_CHECKPOINT_PATH=/content/drive/MyDrive/AirbagsCV/runs/efficientad_aitex/weights/lightning/model.ckpt
EPOCH_CHECKPOINT_DIR=/content/drive/MyDrive/AirbagsCV/runs/efficientad_aitex/weights/lightning
METRICS_JSON_PATH=/content/drive/MyDrive/AirbagsCV/runs/efficientad_aitex/metrics.json
METRICS_CSV_PATH=/content/drive/MyDrive/AirbagsCV/runs/benchmark_results.csv
======================================================================
```

Checkpoints saved per epoch (save_weights_only=True — see note below):
- `last.ckpt` — always overwritten each epoch; the **recommended resume target** (warm restart — weights only, no optimizer state).
- `model.ckpt` — best by validation `image_AUROC`.
- `epoch=N-step=M.ckpt` — top-3 best epoch checkpoints.

**Note on save_weights_only=True:** Anomalib 2.0.0's Evaluator callback is not picklable, which crashes full-state checkpointing with `NotImplementedError: _SingleProcessDataLoaderIter cannot be pickled`. The repo works around this by saving weights + buffers + sanitized hparams only. The checkpoint is fully loadable for inference and evaluation. `--resume-from-checkpoint` will load the weights but restart the optimizer and epoch counter from scratch (warm restart, not a true continuation). See `build_checkpoint_callback` docstring in `train_models.py` for details.

If Colab disconnects before training finishes, the latest `last.ckpt` is on
Drive — use step 10 to resume.

---

## 9b. Train PatchCore (alternative model, no imagenette needed)

PatchCore is a memory-bank anomaly-detection model. Unlike EfficientAD:
- **No `--imagenet-dir` required.**
- **Larger batch size OK** (default 8, vs EfficientAD's forced 1).
- **Only 1 epoch needed** — the memory bank is fitted once via `on_train_epoch_end`.
- **No optimizer state** — "resume" is conceptually meaningless. To evaluate, use `benchmark_inference.py` or `demo/inference.py`.

```bash
%%bash
cd /content/AirbagsCV
python scripts/train_models.py \
  --model patchcore \
  --dataset aitex \
  --data-dir "$PREPARED_DATA_DIR" \
  --output-dir "$RUNS_DIR" \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 2 \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed \
  --seed 42 \
  --backbone wide_resnet50_2 \
  --coreset-sampling-ratio 0.1 \
  --save-top-k 1
```

Outputs land in `$RUNS_DIR/patchcore_aitex/` (same layout as EfficientAD):
- `weights/lightning/last.ckpt` — fitted memory bank + backbone weights.
- `weights/lightning/model.ckpt` — best by validation `image_AUROC`.
- `metrics.json` — all four metrics + backbone + coreset_sampling_ratio.

**PatchCore-specific notes:**
- For faster training on T4 with the default `wide_resnet50_2`, expect ~5-15 minutes for 1 epoch on the full AITEX set.
- For CPU or limited GPUs, use `--backbone resnet18` (much smaller, ~10x faster).
- PatchCore's KNN search at inference can be slower than EfficientAD; see the benchmark in step 11.
- `val_metrics` for PatchCore omit F1Score (post-processor's adaptive threshold isn't fit until after the first val pass). F1Score IS computed at test time and appears in `metrics.json`.
- To **evaluate** a trained PatchCore, do NOT pass `--resume-from-checkpoint` (it would re-run fit). Instead, use `scripts/benchmark_inference.py --model patchcore --checkpoint <path>` or `demo/app.py`.

To benchmark PatchCore inference latency (step 11), set `--model patchcore` and point `--checkpoint` at the PatchCore checkpoint:

```bash
%%bash
cd /content/AirbagsCV
python scripts/benchmark_inference.py \
  --checkpoint "$RUNS_DIR/patchcore_aitex/weights/lightning/model.ckpt" \
  --model patchcore \
  --data-dir "$PREPARED_DATA_DIR" \
  --output-csv "$RESULTS_DIR/latency_patchcore.csv" \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --iterations 200
```

---

## 10. Resume training from a checkpoint (after disconnect)

This is a **robust resume cell**: it remounts Drive, pulls latest repo,
reinstalls deps if needed, redefines env vars, finds the latest `last.ckpt`,
and resumes training. If no checkpoint exists, it tells you to start from
scratch (step 9).

```python
# === ROBUST RESUME CELL ===
# 1. Remount Drive (in case this is a fresh runtime after disconnect)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2. Pull latest repo
import subprocess
repo_dir = '/content/AirbagsCV'
if not __import__('os').path.isdir(repo_dir):
    subprocess.run(['git', 'clone', 'https://github.com/MohamedAzzam4/AirbagsCV.git', repo_dir], check=True)
else:
    subprocess.run(['git', '-C', repo_dir, 'pull'], check=True)

# 3. Check if dependencies are installed; reinstall if not
try:
    import anomalib  # noqa: F401
    print('Dependencies OK.')
except ImportError:
    print('anomalib missing; reinstalling dependencies...')
    subprocess.run(['pip', 'install', '-r', f'{repo_dir}/requirements-colab.txt'], check=True)

# 4. Redefine env vars (must match step 5)
import os
os.environ['REPO_DIR']          = '/content/AirbagsCV'
os.environ['RAW_AITEX_DIR']     = '/content/drive/MyDrive/datasets/AITEX'
os.environ['IMAGENETTE_DIR']    = '/content/drive/MyDrive/datasets/imagenette/train'
os.environ['PREPARED_DATA_DIR'] = '/content/drive/MyDrive/AirbagsCV/prepared/aitex_patches'
os.environ['RUNS_DIR']          = '/content/drive/MyDrive/AirbagsCV/runs'
os.environ['RESULTS_DIR']       = '/content/drive/MyDrive/AirbagsCV/results'
os.environ['CACHE_DIR']         = '/content/drive/MyDrive/AirbagsCV/cache'
for d in ['PREPARED_DATA_DIR', 'RUNS_DIR', 'RESULTS_DIR', 'CACHE_DIR']:
    os.makedirs(os.environ[d], exist_ok=True)

# 5. Find latest checkpoint under RUNS_DIR (prefers last.ckpt)
import glob
run_dir = os.path.join(os.environ['RUNS_DIR'], 'efficientad_aitex')
last_ckpts = sorted(
    glob.glob(f'{run_dir}/**/last.ckpt', recursive=True),
    key=lambda p: os.path.getmtime(p),
)
all_ckpts = sorted(
    glob.glob(f'{run_dir}/**/*.ckpt', recursive=True),
    key=lambda p: os.path.getmtime(p),
)
if last_ckpts:
    CKPT = last_ckpts[-1]
    print(f'Resuming from last.ckpt: {CKPT}')
elif all_ckpts:
    CKPT = all_ckpts[-1]
    print(f'WARNING: no last.ckpt found; resuming from newest checkpoint: {CKPT}')
else:
    CKPT = None
    print(f'NO CHECKPOINT FOUND under {run_dir}.')
    print('Training must start from zero — run step 9 (Train EfficientAD from scratch).')
```

If `CKPT` is set, run the resume command:

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

## 11. Run honest inference benchmark

After training (or using the committed checkpoint), measure real per-image
latency:

```python
# Set CHECKPOINT_PATH — prefer last.ckpt, fall back to model.ckpt, fall back to repo's committed ckpt
import glob, os
run_dir = os.path.join(os.environ['RUNS_DIR'], 'efficientad_aitex')
candidates = []
for pattern in ['last.ckpt', 'model.ckpt']:
    found = sorted(glob.glob(f'{run_dir}/**/{pattern}', recursive=True),
                   key=lambda p: os.path.getmtime(p))
    candidates.extend(found)
# Fall back to any checkpoint
candidates.extend(sorted(glob.glob(f'{run_dir}/**/*.ckpt', recursive=True),
                         key=lambda p: os.path.getmtime(p)))
# Fall back to repo's committed checkpoint (for first run before any training)
if not candidates:
    candidates = ['results/efficientad_aitex/EfficientAd/aitex/latest/weights/lightning/model.ckpt']
os.environ['CHECKPOINT_PATH'] = candidates[0]
print('CHECKPOINT_PATH =', os.environ['CHECKPOINT_PATH'])
```

```bash
%%bash
cd /content/AirbagsCV
python scripts/benchmark_inference.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --model efficientad \
  --data-dir "$PREPARED_DATA_DIR" \
  --output-csv "$RESULTS_DIR/latency.csv" \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --iterations 200
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

## 12. Verify all outputs are on Drive

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
- `efficientad_aitex/weights/lightning/last.ckpt`
- `efficientad_aitex/weights/lightning/model.ckpt` (best)
- `efficientad_aitex/weights/lightning/epoch=N-step=M.ckpt` (top-3)
- `efficientad_aitex/metrics.json`
- `benchmark_results.csv`
- `latency.csv` + `latency.json`

All of these persist on Drive and survive Colab disconnects.

---

## 13. Zip final outputs (optional)

```bash
%%bash
cd /content
zip -r /content/drive/MyDrive/AirbagsCV/airbagcv_run_outputs.zip \
  "$RUNS_DIR" \
  "$RESULTS_DIR"
```

---

## 14. Common failure modes — clear messages and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA available: False` | GPU runtime not selected | Runtime > Change runtime type > GPU |
| `FileNotFoundError: No NODefect_images` | Wrong `RAW_AITEX_DIR` | Check Drive path; fix env var in step 5 |
| `FileNotFoundError: Couldn't find any class folder in datasets/imagenette` | Missing or wrong-format imagenette | Download `imagenette2-160.tgz`, extract, point `IMAGENETTE_DIR` at the `train/` subdir |
| `FileNotFoundError: EfficientAD requires the 'imagenette' dataset at ...` | Missing `--imagenet-dir` | Add `--imagenet-dir "$IMAGENETTE_DIR"` to every train/resume command. (PatchCore does NOT need imagenette — switch with `--model patchcore`.) |
| `TypeError: Folder.__init__() got an unexpected keyword argument 'image_size'` | Anomalib version mismatch | Ensure `anomalib==2.0.0` is installed (the requirements file pins it) |
| `ValueError: train_batch_size for EfficientAd should be 1` | Wrong `--batch-size` for EfficientAD | Use `--batch-size 1`. The script will warn and override anyway. (PatchCore accepts larger batch sizes.) |
| `_pickle.UnpicklingError: Weights only load failed... Evaluator` | PyTorch 2.6+ defaults to `weights_only=True` | The repo's `demo/inference.py` and `scripts/benchmark_inference.py` already pass `weights_only=False`. If you hit this in custom code, pass `weights_only=False` to `load_from_checkpoint`. |
| Colab disconnects mid-training | Idle timeout (free tier) | Reconnect, run step 10 (robust resume cell) |
| `RuntimeError: CUDA out of memory` | Batch too large or other process using GPU | Use `--batch-size 1` (required anyway). Restart runtime if memory is fragmented. |
| `ModuleNotFoundError: No module named 'anomalib'` | Step 4 skipped or failed | Re-run `pip install -r requirements-colab.txt` |
| Demo `cv2.addWeighted` size mismatch | You're running the OLD version of the repo | `git pull` to get the fix |
| `FileNotFoundError: No checkpoint found under ...` (resume) | No checkpoint exists yet | Run step 9 (train from scratch) first |
| Drive mount fails / hangs | Colab browser permission issue | Runtime > Restart session, then re-run step 2 |

---

## 15. Run the demo locally (optional)

You cannot run the Gradio demo directly in Colab (Colab doesn't expose ports
cleanly). To use the demo:

1. Download your trained checkpoint from Drive (e.g. `last.ckpt` or `model.ckpt`).
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

# AirbagsCV

**Proof-of-concept / research prototype for airbag fabric defect detection using unsupervised anomaly detection.**

> This is **not** a production system. It is **not** factory-ready. It has not been trained on real airbag data. The current results are on the AITEX textile dataset, which is a *proxy* for airbag fabric. AUROC is reported, not accuracy. See the [Honest Status Table](#honest-status-table) below before citing any number from this repo.

---

## What this project is

A computer-vision research prototype that explores whether unsupervised anomaly detection (specifically the **EfficientAD** student-teacher architecture) can be used to detect defects in airbag fabric when no real defect dataset is available yet.

The real-world constraints this project is designed around:

- The factory will not share real data until credibility is proven.
- Even when shared, real defect samples will be rare (highly imbalanced).
- The final system must run at industrial inspection speed (line-scan, not notebook).
- We currently have only proxy textile datasets (AITEX).

The correct first direction is unsupervised anomaly detection / anomaly segmentation, not supervised classification, because we do not have enough real defect samples.

---

## What is currently implemented

| Capability | Status | Evidence |
|---|---|---|
| AITEX preprocessing (CLI, blank-patch filter, image-level split) | Implemented | `scripts/prepare_aitex.py` |
| EfficientAD training on AITEX (CLI, seed, resume) | Implemented | `scripts/train_models.py --model efficientad` |
| PatchCore training on AITEX (CLI, seed, no imagenette needed) | Implemented, verified end-to-end on synthetic data | `scripts/train_models.py --model patchcore` |
| Honest inference latency benchmark (warmup, sync, p50/p95/p99) — both models | Implemented | `scripts/benchmark_inference.py` (supports `--model efficientad` and `--model patchcore`) |
| Smoke test (imports, structure, checkpoint) | Implemented | `scripts/smoke_test.py` |
| Gradio demo (live heatmap overlay, benchmark dashboard) | Implemented (local PoC) | `demo/app.py`, `demo/inference.py` |
| Existing trained checkpoint (EfficientAD-small, AITEX, 10 epochs) | Real | `results/efficientad_aitex/.../model.ckpt` (epoch=9, global_step=17920) |
| Google Colab training guide | Implemented | `notebooks/COLAB_GUIDE.md` |

## What is **NOT** implemented

| Capability | Status | Why |
|---|---|---|
| MVTec Carpet / Grid baselines | Not implemented | Original plan promised them; never executed. |
| ONNX export | Not verified | `scripts/export_models.py` exists but was never run. No `exports/` directory. |
| OpenVINO export | Not verified | Same as above. |
| TensorRT / quantization | Not implemented | Future work. |
| Synthetic defect generation (CutPaste / Perlin) | Not implemented | `scripts/generate_synthetic_defects.py` is a TODO stub. |
| Cold-start ablation | Not implemented | `scripts/run_cold_start_ablation.py` is a TODO stub. |
| Real factory airbag dataset | Not available | Factory has not shared data. |
| Full real-time line-scan pipeline | Not implemented | Latency benchmark is patch-level only. |
| Factory-grade threshold calibration | Not implemented | Requires real defect rates. |
| PLC / reject-signal integration | Not implemented | Production work. |
| Regulatory compliance (FMVSS 208, ECE R16, IATF 16949) | Not addressed | Future work. |

---

## Honest Status Table

| Feature | Status | Evidence | Notes |
|---|---|---|---|
| EfficientAD on AITEX | Implemented, undertrained | `results/benchmark_results.csv` shows image_AUROC=0.753, pixel_AUROC=0.680 at 10 epochs | The published EfficientAD recipe calls for 70 epochs. 75% AUROC is **barely above random** (50% = coin flip). Treat this as a sanity check that the pipeline runs, NOT as evidence the algorithm works. |
| PatchCore on AITEX | Implemented, verified end-to-end on synthetic data | `scripts/train_models.py --model patchcore`; verified train→eval→checkpoint round-trip on a tiny synthetic dataset (image_AUROC=0.848, pixel_AUROC=0.898 on the synthetic set — NOT a real result) | No real AITEX training run has been done yet. PatchCore is a memory-bank model (no optimizer state); "resume" is conceptually meaningless. Use `benchmark_inference.py` or `demo/inference.py` to evaluate a trained checkpoint. Does NOT require imagenette. Supports larger batch sizes (default 8). Only 1 effective epoch needed. |
| ONNX export | Not verified | `scripts/export_models.py` exists but never produced artifacts | Do not claim. |
| OpenVINO | Not verified | No exported artifact | Future work. |
| Gradio demo | Implemented / fixed | `demo/app.py` | Local PoC only. Previous version crashed on any non-256×256 upload; this is fixed. |
| Google Colab training | Implemented | `notebooks/COLAB_GUIDE.md` | Calls repo scripts; does NOT reimplement training. |
| Latency benchmark | Implemented | `scripts/benchmark_inference.py` | Patch/image-level only. **Not** line-scan production throughput. |
| Reproducibility (pinned deps, seed, smoke test) | Implemented | `requirements*.txt`, `pyproject.toml`, `--seed` arg, `scripts/smoke_test.py` | Python 3.10–3.12 supported. Python 3.13+ may work but is not tested. |
| Frequent checkpointing + resume | Implemented | `train_models.py` `build_checkpoint_callback()`: `save_last=True`, `every_n_epochs=1`, `save_top_k=3` monitored on `image_AUROC` | `last.ckpt` is the recommended resume target. Best ckpt is `model.ckpt`. |
| Google Colab Mode A (Drive-persistent) | Implemented | `notebooks/COLAB_GUIDE.md` §"Mode A" | Trains directly from prepared data on Drive; safer after disconnect; slower. |
| Google Colab Mode B (local-cache) | Implemented | `notebooks/COLAB_GUIDE.md` §"Mode B" | Copies prepared data from Drive to `/content` at runtime; faster; checkpoints still go to Drive. |

---

## Important terminology

**AUROC is not accuracy.** AUROC measures how well the model *ranks* a defective image higher than a normal one. 50% = random guessing. 95%+ is the typical bar for "usable." Reporting "75% accuracy" when the metric is 75% AUROC is **wrong** and will undermine your credibility with any trained engineer. This repo reports AUROC explicitly.

**Proxy dataset, not real airbag data.** AITEX is a fabric-defect dataset (polyester / cotton textiles). Airbag fabric is silicon-coated polyamide-6.6 at 470 dtex with very different reflectance. The domain gap is unquantified and is the #1 technical risk of this project.

**Patch-level latency is not production throughput.** Real airbag inspection runs at ~16 kHz line rate on 8K-wide fabric. The benchmark in `scripts/benchmark_inference.py` measures per-patch inference only. Production throughput requires the streaming + tiled + ROI-cropped pipeline, which is not implemented.

---

## Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (3.13+ untested; 3.14 not supported by Anomalib at time of writing)
- A CUDA-capable GPU is strongly recommended for training (CPU works for the demo)
- The raw [AITEX dataset](https://www.aitex.es/afid/) (registration required) or the [Kaggle mirror](https://www.kaggle.com/datasets/rmshashi/fabric-defect-dataset)

### Install

```bash
# 1. Clone
git clone https://github.com/MohamedAzzam4/AirbagsCV.git
cd AirbagsCV

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# 3. Install PyTorch with the right CUDA for your machine.
#    See https://pytorch.org/get-started/locally/ to pick the right wheel.
#    Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest of the dependencies
pip install -r requirements.txt

# 5. (optional) dev tools
pip install -r requirements-dev.txt

# 6. Verify the install
python scripts/smoke_test.py
```

If the smoke test reports `[FAIL] import anomalib`, your Anomalib install did not complete. If it reports `[FAIL] import torchvision`, you skipped step 3.

---

## Usage

### 1. Prepare the AITEX dataset

```bash
python scripts/prepare_aitex.py \
  --source /path/to/AITEX_dataset \
  --output ./datasets/aitex \
  --patch-size 256 \
  --blank-threshold 0.5 \
  --seed 42
```

The `--source` directory must contain `NODefect_images/`, `Defect_images/`, and `Mask_images/`. The output is written in Anomalib `Folder` format. Blank/mostly-black patches (a known AITEX artifact) are filtered out by default; use `--blank-threshold 1.0` to disable.

### 2. Train EfficientAD

EfficientAD **requires `--batch-size 1`** (Anomalib 2.0.0 hard constraint; also matches the original paper). Any other value triggers a warning and is overridden to 1.

EfficientAD **also requires `--imagenet-dir`** pointing at an ImageFolder-format imagenette dataset (used for its penalty term). Download `imagenette2-160.tgz` from https://github.com/fastai/imagenette and extract it.

```bash
python scripts/train_models.py \
  --model efficientad \
  --dataset aitex \
  --data-dir ./datasets/aitex \
  --output-dir ./results \
  --epochs 70 \
  --batch-size 1 \
  --num-workers 2 \
  --accelerator auto \
  --devices 1 \
  --precision 16-mixed \
  --seed 42 \
  --imagenet-dir /path/to/imagenette/train
```

At the end, the script prints all key paths:

```
======================================================================
TRAINING COMPLETE — KEY PATHS
======================================================================
RUN_DIR=results/efficientad_aitex
LAST_CHECKPOINT_PATH=results/efficientad_aitex/weights/lightning/last.ckpt
BEST_CHECKPOINT_PATH=results/efficientad_aitex/weights/lightning/model.ckpt
EPOCH_CHECKPOINT_DIR=results/efficientad_aitex/weights/lightning
METRICS_JSON_PATH=results/efficientad_aitex/metrics.json
METRICS_CSV_PATH=results/benchmark_results.csv
======================================================================
```

Checkpoints saved per epoch (every_n_epochs=1, save_top_k=3 by val image_AUROC):
- `last.ckpt` — always overwritten each epoch; the recommended resume target.
- `model.ckpt` — best by validation `image_AUROC`.
- `epoch=N-step=M.ckpt` — top-k best epoch checkpoints.

If no monitored validation metric is available (e.g. custom evaluator without val_metrics), the script prints:
`BEST_CHECKPOINT_PATH=Not available: no monitored validation metric produced a checkpoint`.

### 3. Resume training from a checkpoint

```bash
python scripts/train_models.py \
  --model efficientad \
  --dataset aitex \
  --data-dir ./datasets/aitex \
  --output-dir ./results \
  --epochs 70 \
  --batch-size 1 \
  --num-workers 2 \
  --accelerator auto \
  --devices 1 \
  --precision 16-mixed \
  --seed 42 \
  --imagenet-dir /path/to/imagenette/train \
  --resume-from-checkpoint ./results/efficientad_aitex/weights/lightning/last.ckpt
```

The recommended resume target is `last.ckpt` (it carries full optimizer + scheduler state).

### 3b. Train PatchCore (alternative model)

PatchCore is a memory-bank anomaly-detection model. Unlike EfficientAD:
- **No imagenette required** — omit `--imagenet-dir`.
- **Larger batch size OK** — default 8 (vs EfficientAD's forced 1).
- **Only 1 epoch needed** — the memory bank is fitted once via `on_train_epoch_end`; passing `--epochs N` with N>1 just re-runs feature extraction N times.
- **No optimizer state** — "resume" is conceptually meaningless (re-running `fit()` rebuilds the memory bank from scratch). To evaluate a trained PatchCore, use `benchmark_inference.py` or `demo/inference.py`.

```bash
python scripts/train_models.py \
  --model patchcore \
  --dataset aitex \
  --data-dir ./datasets/aitex \
  --output-dir ./results \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 2 \
  --accelerator auto \
  --devices 1 \
  --precision 32-true \
  --seed 42 \
  --backbone wide_resnet50_2 \
  --coreset-sampling-ratio 0.1 \
  --save-top-k 1
```

Outputs (same layout as EfficientAD, written to `./results/patchcore_aitex/`):
- `weights/lightning/last.ckpt` — the fitted memory bank + backbone weights.
- `weights/lightning/model.ckpt` — best by validation `image_AUROC` (with `--epochs 1` and `--save-top-k 1`, this is effectively the same as `last.ckpt`).
- `metrics.json` — image_AUROC, pixel_AUROC, image_F1Score, pixel_F1Score, backbone, coreset_sampling_ratio.
- A row appended to `./results/benchmark_results.csv`.

**PatchCore-specific limitations:**
- The default backbone `wide_resnet50_2` is heavy (~130 MB). For faster training/inference on CPU or limited GPUs, use `--backbone resnet18`.
- PatchCore's KNN search at inference can be slower than EfficientAD's forward pass, especially with a large memory bank (high `--coreset-sampling-ratio` or large train set).
- `val_metrics` for PatchCore omit F1Score (the post-processor's adaptive threshold isn't fit until after the first validation pass, so `pred_label` isn't available). F1Score IS computed at test time and written to `metrics.json`.

### 4. Run honest inference benchmark

```bash
python scripts/benchmark_inference.py \
  --checkpoint ./results/efficientad_aitex/EfficientAd/aitex/latest/weights/lightning/model.ckpt \
  --model efficientad \
  --data-dir ./datasets/aitex \
  --output-csv ./results/latency.csv \
  --device auto \
  --batch-size 1 \
  --warmup 20 \
  --iterations 200
```

Outputs `latency.csv` and `latency.json` with p50 / p95 / p99 / mean / throughput. **Patch-level only — not production line-scan throughput.** `--batch-size` for the benchmark is currently fixed at 1 (per-image predict).

### 5. Run the Gradio demo

```bash
python demo/app.py
# then open http://127.0.0.1:7860/ in your browser
```

The demo gracefully reports an error if no checkpoint exists. It accepts uploads of any image size (the previous version crashed on non-256×256 inputs — fixed).

### 6. Smoke test (cheap sanity check)

```bash
python scripts/smoke_test.py
# or, with a specific checkpoint to validate:
python scripts/smoke_test.py --checkpoint ./results/efficientad_aitex/EfficientAd/aitex/latest/weights/lightning/model.ckpt
```

---

## Google Colab workflow

See [`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md) for the complete copy-paste Colab workflow (or open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) directly in Colab). The notebook **does not reimplement training** — it clones the repo, installs deps, mounts Drive, and calls the repo scripts from the command line.

Default Drive paths (all under `/content/drive/MyDrive/AirbagsCV/`):

| Variable | Default | Purpose |
|---|---|---|
| `RAW_AITEX_DIR` | `/content/drive/MyDrive/datasets/AITEX` | Raw AITEX (NODefect_images/, Defect_images/, Mask_images/) |
| `IMAGENETTE_DIR` | `/content/drive/MyDrive/datasets/imagenette/train` | Imagenette train/ for EfficientAD penalty term |
| `PREPARED_DATA_DIR` | `/content/drive/MyDrive/AirbagsCV/prepared/aitex_patches` | Anomalib Folder layout produced by `prepare_aitex.py` |
| `RUNS_DIR` | `/content/drive/MyDrive/AirbagsCV/runs` | Checkpoints, metrics.json, benchmark CSV |
| `RESULTS_DIR` | `/content/drive/MyDrive/AirbagsCV/results` | latency.csv / latency.json |
| `CACHE_DIR` | `/content/drive/MyDrive/AirbagsCV/cache` | Optional zipped prepared dataset for Mode B |

Two I/O modes are documented in the guide:

- **Mode A (Drive-persistent):** train directly from `PREPARED_DATA_DIR` on Drive. Safer after disconnect; slower (Drive I/O per batch).
- **Mode B (local-cache):** copy/unzip prepared data to `/content` at runtime, train from there, save checkpoints back to Drive. Faster; same Drive safety for checkpoints.

Both modes save all checkpoints, metrics, and benchmark outputs to Drive. The guide includes a robust resume cell that remounts Drive, pulls latest repo, reinstalls deps, redefines env vars, finds the latest `last.ckpt`, and resumes training.

---

## Repository layout

```
AirbagsCV/
├── README.md                    # you are here
├── pyproject.toml               # python project metadata
├── requirements.txt             # pinned deps (CPU + CUDA compatible)
├── requirements-colab.txt       # pinned deps for Colab
├── requirements-dev.txt         # + ruff, black, pytest
├── scripts/
│   ├── prepare_aitex.py         # dataset preprocessing (CLI, blank filter, no leakage)
│   ├── train_models.py          # EfficientAD training (CLI, seed, resume)
│   ├── benchmark_inference.py   # honest latency benchmark (warmup, sync, p50/p95/p99)
│   ├── smoke_test.py            # cheap sanity check
│   ├── export_models.py         # ONNX/OpenVINO export (UNVERIFIED — see status table)
│   ├── generate_synthetic_defects.py   # TODO stub (Phase 4)
│   ├── run_cold_start_ablation.py     # TODO stub (Phase 4)
│   └── train_proxy_baselines.py       # TODO stub (Phase 4)
├── demo/
│   ├── app.py                   # Gradio UI
│   └── inference.py             # checkpoint loader + predict()
├── notebooks/
│   └── COLAB_GUIDE.md           # copy-paste Colab workflow
├── docs/artifacts/              # research notes and audit reports (read-only history)
└── results/                     # checkpoints + CSVs (gitignored except committed demo ckpt)
```

---

## Limitations and honest warnings

1. **The current 75.3% AUROC is not a success.** It is barely above random. The original training was only 10 of the recommended 70 epochs and used the buggy data prep that included blank patches. Retraining with the fixed pipeline and full epochs is the first thing to do.
2. **The existing checkpoint was produced by the OLD pipeline** (before this cleanup). It is committed only so the demo can run out-of-the-box. It should be replaced by a fresh run.
3. **No real airbag data has been used.** Every number in this repo is on AITEX, a textile proxy.
4. **No synthetic defect generation yet.** This is the highest-priority missing piece for the no-factory-data phase.
5. **No production deployment artifacts.** No ONNX, no OpenVINO, no TensorRT, no edge benchmark.
6. **No regulatory framing.** Airbags are safety-critical (FMVSS 208, ECE R16, IATF 16949). This repo does not address compliance.

---

## License

MIT (see `pyproject.toml`). The committed checkpoint is provided for reproducibility only.

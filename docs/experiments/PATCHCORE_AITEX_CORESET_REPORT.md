# PatchCore AITEX Experiment Report — AirbagsCV

## 1. Context

This report documents the PatchCore experiment work performed for the `AirbagsCV` repository.

The project is an anomaly detection proof-of-concept for airbag/fabric defect detection. Since no real factory airbag dataset is currently available, the AITEX fabric dataset was used as a proxy dataset to test the anomaly detection pipeline, compare model behavior, measure inference speed, and identify practical limitations before moving to real industrial data.

The main goal of this experiment round was to evaluate PatchCore with different coreset sampling ratios and determine whether increasing the coreset size gives enough quality improvement to justify the additional inference cost.

---

## 2. Dataset Used

Prepared AITEX patch dataset path used during Colab experiments:

```text
/content/drive/MyDrive/AirbagsCV/prepared/aitex_patches
```

Expected prepared structure:

```text
aitex_patches/
  train/good/
  test/good/
  test/anomaly/
  ground_truth/anomaly/
```

Verified counts:

| Split | Count |
|---|---:|
| `train/good` | 1792 |
| `test/good` | 1973 |
| `test/anomaly` | 185 |
| `ground_truth/anomaly` | 185 |

The masks in `ground_truth/anomaly` were used for pixel-level evaluation metrics.

---

## 3. PatchCore Setup

PatchCore was run through the Anomalib-based training pipeline in the repo.

Base configuration:

```text
model: patchcore
dataset: aitex
backbone: resnet18
epochs: 1
batch_size: 1
precision: 32-true
num_workers: 0
image_size: 256
seed: 42
```

Important PatchCore-specific notes:

- PatchCore is a memory-bank method, so one epoch is expected.
- `precision=16-mixed` caused problems because PatchCore has no optimizer in the normal training sense.
- `precision=32-true` was required.
- Larger backbones / larger batch sizes were more likely to hit GPU memory limits on Colab T4.
- Increasing the coreset ratio increases memory bank size and inference latency.

---

## 4. Experiments Run

Three PatchCore `resnet18` coreset configurations were compared:

| Tag | Backbone | Coreset ratio | Batch size |
|---|---|---:|---:|
| `resnet18_c001_bs1` | resnet18 | 0.01 | 1 |
| `resnet18_c003_bs1` | resnet18 | 0.03 | 1 |
| `resnet18_c005_bs1` | resnet18 | 0.05 | 1 |

---

## 5. Problems Encountered and Fixes

### 5.1 PatchCore with mixed precision failed

Initial PatchCore runs using `16-mixed` failed with an optimizer / GradScaler-related issue.

Cause:

PatchCore does not train like a normal neural network with optimizer updates. It builds embeddings and a memory bank. Mixed precision training with GradScaler expected optimizer state that did not exist.

Fix:

Use:

```text
precision = 32-true
```

---

### 5.2 GPU out-of-memory with heavier PatchCore settings

A wider backbone / larger batch configuration caused CUDA out-of-memory on Colab T4.

Fix:

Use safer settings:

```text
backbone = resnet18
batch_size = 1
precision = 32-true
```

---

### 5.3 Colab runtime disconnected after `resnet18_c005_bs1` training

The `resnet18_c005_bs1` run completed the training / coreset selection stage and saved checkpoints, but Colab disconnected before `metrics.json`, benchmark CSV, and summary row were written.

Evidence from the training log:

```text
Selecting Coreset Indices.: 100%|██████████| 70246/70246
Trainer.fit stopped: max_epochs=1 reached
Training wall-clock: 2940.4 s (49.0 min)
```

Saved checkpoint existed:

```text
/content/drive/MyDrive/AirbagsCV/runs/patchcore_experiments/resnet18_c005_bs1/patchcore_aitex/weights/lightning/last.ckpt
```

But missing:

```text
metrics.json
benchmark csv
summary.csv row
```

Fix:

A recovery notebook was created for a second Google account. It:

1. Extracts the transferred zip.
2. Loads the saved `resnet18_c005_bs1` checkpoint.
3. Runs evaluation again.
4. Runs benchmark.
5. Rebuilds the comparison table.

No retraining was required.

---

### 5.4 Google Colab GPU quota limit

The original Colab account hit the GPU daily/runtime limit before recovery evaluation could be completed.

Fix:

Transfer required files to a second Google account and run recovery there.

Transfer package created:

```text
AirbagsCV_transfer.zip
```

Contents:

```text
aitex_patches.zip
resnet18_c005_bs1_last.ckpt
resnet18_c005_bs1_train.log
```

Original transfer folder before zipping:

```text
/content/drive/MyDrive/AirbagsCV_transfer
```

Final file sizes:

| File | Size |
|---|---:|
| `aitex_patches.zip` | ~477 MB |
| `resnet18_c005_bs1_last.ckpt` | ~114 MB |
| `resnet18_c005_bs1_train.log` | ~2.1 MB |

---

### 5.5 Outer zip handling issue

The recovery notebook originally expected a folder called:

```text
AirbagsCV_transfer/
```

But the actual transferred item was a single zip file:

```text
AirbagsCV_transfer.zip
```

Fix:

A new notebook version was created that first extracts the outer zip, then finds the required internal files.

Expected second account Drive path:

```text
/content/drive/MyDrive/AirbagsCV_transfer.zip
```

Notebook behavior:

```text
1. Extract AirbagsCV_transfer.zip
2. Find aitex_patches.zip
3. Find resnet18_c005_bs1_last.ckpt
4. Find resnet18_c005_bs1_train.log
5. Extract aitex_patches.zip
6. Recover evaluation metrics
7. Run benchmark
8. Build comparison table
```

---

### 5.6 Pandas / NumPy binary mismatch

After package installation inside Colab, importing `pandas` failed with:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

Cause:

Colab kernel had a binary mismatch after package installation, likely from NumPy / pandas versions loaded in the active kernel.

Fix:

Avoid using pandas for the final comparison step. Replaced the comparison table code with a pure Python + `csv` version.

---

## 6. Final Results

Final comparison table:

| Tag | Coreset | Image AUROC | Pixel AUROC | Image F1 | Pixel F1 | Mean ms/image | P95 ms/image | Throughput img/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18_c001_bs1` | 0.01 | 0.6610615 | 0.9131012 | 0.2325581 | 0.2071096 | 127.872 | 161.215 | 7.82 |
| `resnet18_c003_bs1` | 0.03 | 0.6656643 | 0.9171302 | 0.2395210 | 0.2135482 | 200.827 | 241.394 | 4.98 |
| `resnet18_c005_bs1` | 0.05 | 0.6705015 | 0.9175326 | 0.2409639 | 0.2242991 | 269.089 | 316.598 | 3.72 |

---

## 7. Result Interpretation

### 7.1 `c001` vs `c003`

Increasing coreset from `0.01` to `0.03` improved quality slightly:

```text
Pixel AUROC: 0.9131012 -> 0.9171302
Delta: +0.004029
```

But latency increased significantly:

```text
Mean latency: 127.872 ms -> 200.827 ms
Delta: +72.955 ms/image
```

Throughput dropped:

```text
7.82 img/s -> 4.98 img/s
```

### 7.2 `c003` vs `c005`

Increasing coreset from `0.03` to `0.05` produced almost no meaningful Pixel AUROC improvement:

```text
Pixel AUROC: 0.9171302 -> 0.9175326
Delta: +0.000402
```

But latency increased a lot:

```text
Mean latency: 200.827 ms -> 269.089 ms
Delta: +68.262 ms/image
```

Throughput dropped again:

```text
4.98 img/s -> 3.72 img/s
```

### 7.3 Practical conclusion

`resnet18_c005_bs1` has the best raw quality numbers, but the improvement over `resnet18_c003_bs1` is too small to justify the extra latency.

The best practical quality/speed tradeoff is likely:

```text
resnet18_c003_bs1
```

If speed matters more than the small Pixel AUROC improvement, the best practical choice is:

```text
resnet18_c001_bs1
```

---

## 8. Decision

### Do not continue to coreset `0.10`

The trend is clear:

```text
coreset 0.01 -> 0.03:
small quality improvement, large latency increase

coreset 0.03 -> 0.05:
almost no quality improvement, large latency increase
```

Therefore, testing `coreset=0.10` is not prioritized because it is expected to increase latency substantially without enough quality gain.

---

## 9. Recommended Model Choice

| Scenario | Recommended model |
|---|---|
| Fast demo / production-like speed | `resnet18_c001_bs1` |
| Best quality-speed compromise | `resnet18_c003_bs1` |
| Highest quality regardless of latency | `resnet18_c005_bs1` |

Final recommendation for the current report/demo:

```text
Use PatchCore resnet18 with coreset=0.03 as the balanced baseline.
Keep coreset=0.01 as the speed baseline.
Do not prioritize coreset=0.10.
```

---

## 10. Suggested GitHub Files / Artifacts to Save

Recommended paths to add to the repo:

```text
docs/experiments/PATCHCORE_AITEX_CORESET_REPORT.md
```

Optional supporting artifacts:

```text
results/patchcore_experiments/patchcore_comparison_summary.csv
results/patchcore_experiments/resnet18_c005_bs1_metrics.json
results/patchcore_experiments/resnet18_c005_bs1_benchmark_full.csv
```

Do not commit large checkpoint files unless the repo uses Git LFS or external artifact storage.

Avoid committing:

```text
*.ckpt
*.zip
```

Recommended `.gitignore` additions if not already present:

```gitignore
*.ckpt
*.zip
/content/
runs/
```

---

## 11. Suggested README Summary

A short README-ready summary:

```markdown
### PatchCore AITEX coreset experiments

We evaluated PatchCore with a ResNet18 backbone on the prepared AITEX patch dataset using coreset ratios 0.01, 0.03, and 0.05.

Increasing the coreset ratio improved pixel-level AUROC only marginally, while inference latency increased substantially. The best raw pixel AUROC came from coreset=0.05, but its gain over coreset=0.03 was only +0.0004 while adding ~68 ms/image latency.

Therefore, coreset=0.03 is currently the preferred quality/speed compromise, while coreset=0.01 remains the fastest baseline. Larger coreset ratios such as 0.10 are not prioritized.
```

---

## 12. Next Steps

1. Stop PatchCore coreset expansion for now.
2. Use `resnet18_c001_bs1` as the fast PatchCore baseline.
3. Use `resnet18_c003_bs1` as the balanced PatchCore baseline.
4. Compare these PatchCore baselines against longer EfficientAD training, especially EfficientAD 70 epochs on Kaggle.
5. Add visual heatmap sanity checks before making any final production-like claim.
6. When real airbag/factory data becomes available, repeat evaluation because AITEX is only a proxy dataset.

---

## 13. Important Caveat

All current results are based on AITEX fabric patches, not real airbag production data.

Therefore, the results are valid for:

```text
pipeline validation
model comparison
runtime behavior
proxy anomaly detection testing
```

But they should not be presented as final real-world airbag inspection performance until evaluated on real factory data.

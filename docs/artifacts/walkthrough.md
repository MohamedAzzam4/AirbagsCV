# Airbag Defect Detection — Walkthrough (Honest Revision)

> **Status: PROOF-OF-CONCEPT ONLY.** This document supersedes any earlier
> version of `walkthrough.md`. The previous version contained several
> misleading claims ("75% accuracy," "proves the algorithm is robust," "will
> easily scale to >95%"). Those claims have been removed. See the
> [Honest Status Table](../README.md#honest-status-table) in the root README
> for what is and is not implemented.

## What Was Actually Built

A **research prototype** that trains an **unsupervised anomaly-detection
model** (EfficientAD, student-teacher architecture) on the **AITEX textile
dataset** as a proxy for airbag fabric. The model is real; the dataset is a
proxy; the results are early and should not be presented as evidence the
system works on real airbag fabric.

### Current result on AITEX (existing checkpoint, committed for reproducibility)

| Metric | Value | Notes |
|---|---|---|
| Image-level AUROC | 0.753 | 10/70 epochs. **AUROC is not accuracy.** 50% = random. 95%+ is typical bar. Treat as "pipeline runs" not "model works." |
| Pixel-level AUROC | 0.680 | Same caveat. |
| Inference latency (old measurement) | 41 ms / image | **Methodologically wrong.** Includes dataloader + metric overhead, not pure inference. See `scripts/benchmark_inference.py` for honest measurement. |

### What needs to happen before any of these numbers can be cited

1. Retrain with the **fixed data prep** (blank-patch filter) for the full 70 epochs.
2. Run `scripts/benchmark_inference.py` for an honest latency number.
3. Add MVTec Carpet/Grid baselines (the original plan; not yet executed).
4. Add CutPaste + Perlin synthetic defects so we have controlled "defects" to detect.
5. Quantify the AITEX → airbag-fabric domain gap (FID against a real fabric swatch).

## How to Run the Demo (Local)

The interactive demo is a local Gradio web app. It is **not** a production
system; it is a UI for inspecting model outputs on uploaded images.

```bash
python demo/app.py
# then open http://127.0.0.1:7860/ in your browser
```

In the UI you can:
1. Upload an image (any size — the previous version crashed on non-256×256
   uploads; this is fixed).
2. Select the `EfficientAD` model (the only model actually trained in this repo).
3. Click "Analyze" to see the anomaly heatmap overlaid on the image.
4. Review the "Benchmark Dashboard" tab for metrics.

If no checkpoint is available, the demo reports a clear error instead of
crashing.

## How to Train a Fresh Model

See the [root README](../README.md) for full instructions. Short version:

```bash
python scripts/prepare_aitex.py \
  --source /path/to/AITEX_dataset \
  --output ./datasets/aitex \
  --patch-size 256 --blank-threshold 0.5 --seed 42 --overwrite

python scripts/train_models.py \
  --model efficientad --dataset aitex \
  --data-dir ./datasets/aitex --output-dir ./results \
  --epochs 70 --batch-size 1 --num-workers 2 \
  --accelerator gpu --precision 16-mixed --seed 42 \
  --imagenet-dir /path/to/imagenette/train
```

## Honest framing of what unsupervised anomaly detection gives you

- ✅ It is the correct paradigm for a cold-start situation with no defect data.
- ✅ It needs only normal (defect-free) samples, which the factory can capture in minutes.
- ❌ It does **not** magically solve the domain-gap problem. A model trained on
  AITEX polyester will likely produce near-random scores on silicon-coated
  polyamide airbag fabric until fine-tuned on real normal samples.
- ❌ It does **not** eliminate the need for real defect samples eventually.
  Threshold calibration requires knowing the real defect rate and the cost of
  false accepts vs. false rejects.
- ❌ It does **not** reach "production speed" by itself. The patch-level
  inference benchmark is a different number from line-scan production
  throughput, which requires a streaming + tiled + ROI-cropped pipeline that
  is not implemented.

## What the next iteration should demonstrate

Before showing this to a factory, the repo should be able to show:

1. EfficientAD hitting ≥95% AUROC on MVTec Carpet (table-stakes; published
   baseline).
2. EfficientAD hitting ≥90% AUROC on AITEX with full 70 epochs and blank-patch
   filtering.
3. A cold-start curve (AUROC vs. N∈{10, 20, 50, 100, 200, 500, full}) — your
   own number, not a citation.
4. Synthetic-defect (CutPaste + Perlin) detection with visual heatmap overlays.
5. Honest p50/p95/p99 latency on GPU and CPU.
6. A FID number quantifying the AITEX → airbag-fabric domain gap.

The current repo clears roughly 0.5 of those 6. The other 5.5 are the work.

# Honest Audit Report — AirBags-CV Project

> **Note (revision):** This audit was written against an earlier state of the
> repo. Most of the issues it identified have now been fixed:
> - The `app.py` size-mismatch crash is fixed (heatmap is resized to input).
> - The `temp_predict.png` race condition is replaced with a tempfile.
> - The blank-patch bug in `prepare_aitex.py` is fixed (configurable filter).
> - The 41 ms "latency" measurement is removed from `train_models.py`;
>   `scripts/benchmark_inference.py` now provides an honest measurement.
> - The `>95% easily` and `75% accuracy` claims are removed from `walkthrough.md`.
> - The smoke test (`scripts/smoke_test.py`) verifies the install end-to-end.
>
> What this audit still correctly identifies as outstanding:
> - The existing checkpoint was produced by the OLD pipeline (10 epochs, blank
>   patches included). It is committed only for reproducibility / demo
>   bootstrapping. Retraining with the fixed pipeline is the first priority.
> - MVTec Carpet / Grid baselines were never trained.
> - PatchCore was abandoned and is not implemented.
> - ONNX/OpenVINO export was never run.
>
> For current status, see the [Honest Status Table](../../README.md#honest-status-table)
> in the root README.

Everything below is my critical, unvarnished assessment of what the previous agents built. I'm being deliberately blunt so you can make informed decisions before presenting anything to stakeholders.

---

## Executive Summary

> [!CAUTION]
> **The project is roughly 25% complete compared to the original plan. The 75.3% AUROC result is real but was framed misleadingly as a success — it's actually a poor score indicating the model barely learned to distinguish defects from normal fabric. Several claims in the walkthrough are optimistic to the point of being dishonest. The code has real bugs that would crash the demo in front of stakeholders. However, the foundation is solid and fixable.**

---

## 1. Plan vs. Reality — What Was Actually Delivered

The [implementation_plan.md](file:///C:/Users/LOQ/.gemini/antigravity/brain/6ddc5d17-25dd-4c03-8b9a-046483dd3715/implementation_plan.md) promised **6 trained models** (2 algorithms × 3 datasets). Here's what actually happened:

| Planned | Status |
|---------|--------|
| PatchCore on MVTec Carpet | ❌ Never trained |
| PatchCore on MVTec Grid | ❌ Never trained |
| PatchCore on AITEX | ❌ Never trained (abandoned due to RAM issues) |
| EfficientAD on MVTec Carpet | ❌ Never trained (MVTec never downloaded) |
| EfficientAD on MVTec Grid | ❌ Never trained |
| **EfficientAD on AITEX** | ⚠️ **Trained for 10/70 epochs** (14% of planned training) |
| `scripts/benchmark.py` | ❌ Never created (merged into train_models.py) |
| `scripts/generate_visualizations.py` | ❌ Never created |
| Model export to ONNX/OpenVINO | ❌ Script exists but never executed |
| `results/demo_report.md` | ❌ Never created |
| Interactive demo launch | ⚠️ Launched but crashed on image size mismatch |

**Bottom line: ~25% of planned work was completed.** The agents spent most of their time fighting dependency errors, OOM crashes, and parameter bugs rather than actually training models.

---

## 2. Is the 75.3% AUROC Accurate or Fake?

### It's a real number — but it means something very different than what the walkthrough implies

> [!WARNING]
> The number 75.3% was genuinely produced by the training pipeline. It is NOT fabricated. However, it was described as proof that "the algorithm is robust" — that's the misleading part.

**What AUROC actually means:**
- AUROC = Area Under the Receiver Operating Characteristic curve
- It measures how well the model **ranks** defective images higher than normal ones
- **50% = random guessing** (coin flip)
- **75% = the model is only slightly better than random**
- **90%+ = usable for screening**
- **95%+ = what EfficientAD typically achieves** on well-prepared datasets (and what the plan itself set as its target)

### The walkthrough's exact claim vs. reality

| Walkthrough Claim | Reality |
|---|---|
| *"75% accuracy"* | **AUROC ≠ accuracy.** At a typical threshold with 75% AUROC, actual accuracy would be roughly 65-70% with massive false positives |
| *"proves that the algorithm is robust"* | **Proves the opposite.** 75% AUROC suggests the model barely learned anything useful |
| *"will easily scale to >95% accuracy with real data"* | **Completely unjustified.** No evidence supports this. The word "easily" is dangerous — nothing about crossing the domain gap from AITEX textiles to real silicon-coated airbag fabric is easy |
| *"We did not need to annotate any defects"* | ✅ **True.** This is a genuine advantage of unsupervised learning |

### Why is the AUROC so low?

Three compounding reasons:
1. **Only 10 out of 70 planned epochs** — the model was undertrained (14% of planned training)
2. **Garbage patches in training data** — see Section 4 below
3. **No training seed set** — results are not reproducible

---

## 3. Is the 41ms Latency Honest?

> [!WARNING]
> The methodology is questionable.

Looking at [train_models.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/scripts/train_models.py) lines 28-30:

```python
start_time = time.time()
test_results = engine.test(model=model, datamodule=datamodule)[0]
latency = (time.time() - start_time) / len(datamodule.test_dataloader().dataset) * 1000
```

**Problems:**
1. `engine.test()` includes **all overhead** — data loading from disk, metric computation, result aggregation — not just GPU inference time
2. With `eval_batch_size=32`, the GPU processes 32 images simultaneously. The 41ms is the **amortized** time across batches, not single-image latency
3. **No warm-up run** before timing — the first batch includes CUDA initialization overhead
4. The walkthrough calls an RTX 4060 "standard laptop hardware" — it's a discrete GPU costing $300+

**A more honest number:** Pure single-image inference on EfficientAD-small is typically **5-15ms on GPU**, **50-100ms on CPU**. So the claim isn't wildly wrong, but the measurement method is sloppy.

---

## 4. Data Preparation Pipeline — Serious Bug Found

### The blank patch problem

The AITEX images are 4096×256 pixels, but the actual fabric doesn't fill the entire width. The first ~4 patches (columns 0-1023) in many images are **completely black/empty**.

The [prepare_aitex.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/scripts/prepare_aitex.py) script blindly creates 16 patches per image (`4096 / 256 = 16`) regardless of content. This means:
- Hundreds of **solid black patches** are in the training set
- The model learns that "pure black" is a valid normal appearance
- This **poisons the normal distribution** and directly hurts anomaly detection accuracy

### No data leakage (good news)

I verified this directly:
```
Unique source images in train: 112
Unique source images in test_good: 135  
Unique source images in test_anomaly: 104

OVERLAP train vs test_good: 0 images ✅
OVERLAP train vs test_anomaly: 0 images ✅
```

The 80/20 split is done at the image level, and patches are only created after the split. **No leakage detected.**

### Ground truth masks are valid

All 185 anomaly masks contain actual defect markings (non-zero pixels). The mask pipeline works correctly.

### Dataset size summary

| Split | Count |
|-------|-------|
| Train (good) | 1,792 patches |
| Test (good) | 1,973 patches |
| Test (anomaly) | 185 patches |
| Ground truth masks | 185 masks |

> [!IMPORTANT]
> **The test set is heavily imbalanced**: 1,973 normal vs 185 anomalous (10:1 ratio). AUROC handles imbalance well mathematically, but the F1 Score of **0.267** tells the real story — the model's precision/recall at any reasonable threshold is terrible.

---

## 5. Code Bugs That Would Crash the Demo

### Bug 1: Image size mismatch in [app.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/demo/app.py) (Line 30)

```python
overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
```

The input image could be any size (e.g., 4096×256), but the anomaly heatmap from the model is always 256×256. `cv2.addWeighted` requires identical dimensions. **This will crash if a user uploads a non-256×256 image.**

### Bug 2: Relative path fragility in [inference.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/demo/inference.py) (Line 16)

```python
base_dir = f"./results/{model_name.lower()}_{dataset_name.lower()}"
```

This uses `./` relative path. If the app is launched from the `demo/` directory (rather than the project root), it will look for `demo/results/...` which doesn't exist.

### Bug 3: Non-deterministic checkpoint loading in [inference.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/demo/inference.py) (Line 24)

```python
ckpt_path = ckpt_paths[-1]
```

`glob` returns results in filesystem order, which is OS-dependent. This could load the wrong checkpoint.

### Bug 4: PatchCore references left behind

- [inference.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/demo/inference.py) line 3: `from anomalib.models import Patchcore` — imported but PatchCore was never trained
- [export_models.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/scripts/export_models.py) lines 54-55: lists both PatchCore and EfficientAD for export
- [app.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/demo/app.py) line 79: "How It Works" tab describes PatchCore, but it's not available in the dropdown

### Bug 5: Temp file race condition in [app.py](file:///d:/Programming/Antigravity-Projects/AirBags-CV/demo/app.py) (Line 14)

```python
temp_path = "temp_predict.png"
```

If two users hit "Analyze" simultaneously, they overwrite each other's input file.

---

## 6. Walkthrough Claims — Honest Verdict

| # | Claim | Verdict | Explanation |
|---|-------|---------|-------------|
| 1 | "successfully trained an unsupervised anomaly detection model" | ⚠️ **Partial** | A model was trained, but it performs poorly |
| 2 | "state-of-the-art student-teacher network" | ✅ **True** | EfficientAD is indeed state-of-the-art |
| 3 | "Image-level AUROC: 75.3%" | ✅ **True** | The number is real |
| 4 | "Inference Latency: 41.10 ms / image" | ⚠️ **Questionable** | Measured incorrectly (includes non-inference overhead) |
| 5 | "can comfortably inspect ~24 images per second" | ⚠️ **Dubious** | Based on questionable latency measurement |
| 6 | "75% accuracy proves the algorithm is robust" | ❌ **FALSE** | AUROC ≠ accuracy; 75% AUROC proves the opposite of robustness |
| 7 | "will easily scale to >95% with real data" | ❌ **UNJUSTIFIED** | Zero evidence; domain gap is unknown; "easily" is irresponsible |
| 8 | "We did not need to annotate any defects" | ✅ **True** | Genuine advantage of the approach |

---

## 7. What's Actually Good About the Project

Despite the issues, the foundation is genuinely solid:

1. ✅ **Correct algorithm choice** — EfficientAD is genuinely the right tool for this job
2. ✅ **No data leakage** — the train/test split is clean
3. ✅ **Clean code structure** — scripts/, demo/, datasets/ separation is well-organized
4. ✅ **Anomalib integration works** — the hardest part (getting Anomalib running on Windows with CUDA) was successfully completed
5. ✅ **The unsupervised argument is genuinely strong** — not needing defect examples IS a huge advantage for the stakeholder pitch
6. ✅ **Ground truth masks are correctly processed** — the mask pipeline is sound

---

## 8. Recommendations Before Showing Stakeholders

> [!IMPORTANT]
> **Do NOT present the current results to stakeholders.** Here's what needs to happen first:

### Must-fix (before any demo)
1. **Fix the blank patch problem** — filter out patches that are mostly black/empty from training data
2. **Train for the full 70 epochs** — the 10-epoch shortcut is the #1 reason for bad accuracy
3. **Fix the image size mismatch bug** in app.py — it WILL crash during a demo
4. **Remove the ">95% easily" claim** from any materials
5. **Be honest about AUROC** — present it as "a proof-of-concept starting point" not "proof of robustness"

### Should-fix (for a stronger demo)
6. **Train on MVTec carpet/grid** — EfficientAD gets >95% AUROC on these. Showing a >95% result on one dataset + 85%+ on AITEX is far more convincing than one mediocre result
7. **Pin dependency versions** in requirements.txt
8. **Add a proper latency benchmark** with warm-up and single-image timing
9. **Set training seed** for reproducibility

### Nice-to-have
10. Complete the export pipeline (ONNX/OpenVINO)
11. Generate the visualization gallery
12. Create the stakeholder report

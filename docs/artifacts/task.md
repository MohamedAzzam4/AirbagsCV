# Airbag Defect Detection — Task Status

> **Honest revision.** The previous version of this file marked Phase 3 as
> "complete" while the demo had a crash bug, and marked Phase 4 as
> "not started" without flagging that the export script in Phase 4 was already
> committed but never executed. This file now reflects reality.

## Status

- `[x]` Phase 0: Environment Setup
  - `[x]` `requirements.txt` (pinned)
  - `[x]` `requirements-colab.txt` (pinned for Colab)
  - `[x]` `requirements-dev.txt` (lint/test)
  - `[x]` `pyproject.toml`
  - `[x]` `scripts/smoke_test.py`
- `[x]` Phase 1: Dataset Acquisition & Preparation
  - `[x]` `scripts/prepare_aitex.py` (CLI-friendly, blank-patch filter, image-level split)
- `[x]` Phase 2: Model Training & Evaluation
  - `[x]` `scripts/train_models.py` (CLI, seed, resume, metrics.json, no fake latency)
  - `[x]` Honest inference benchmark: `scripts/benchmark_inference.py`
  - `[x]` Existing EfficientAD checkpoint on AITEX (10 epochs, 75% AUROC —
        undertrained; needs full 70-epoch re-run)
  - `[ ]` MVTec Carpet baseline (planned, not yet executed)
  - `[ ]` MVTec Grid baseline (planned, not yet executed)
  - `[ ]` PatchCore baseline (NOT IMPLEMENTED — do not claim otherwise)
- `[x]` Phase 3: Interactive Demo (Gradio)
  - `[x]` `demo/inference.py` (deterministic ckpt selection, validation, no dead PatchCore imports)
  - `[x]` `demo/app.py` (size-mismatch bug fixed, tempfile used, CSS passed correctly, graceful degradation)
  - `[x]` Local launch tested
- `[~]` Phase 4: Model Export & Deliverables
  - `[ ]` `scripts/export_models.py` — exists but NEVER SUCCESSFULLY RUN.
        No `exports/` directory in repo. Do not claim export support until
        someone runs it and produces artifacts.
  - `[ ]` ONNX export verified
  - `[ ]` OpenVINO export verified
  - `[ ]` TensorRT export (not started)
  - `[ ]` `results/demo_report.md` — not yet generated
- `[x]` Phase 5: Google Colab
  - `[x]` `notebooks/COLAB_GUIDE.md`
  - `[x]` `notebooks/train_colab.ipynb` (calls repo scripts, does NOT reimplement)
- `[ ]` Phase 6 (future work, NOT IMPLEMENTED):
  - `[ ]` Synthetic defect generation (`scripts/generate_synthetic_defects.py` stub)
  - `[ ]` Cold-start ablation (`scripts/run_cold_start_ablation.py` stub)
  - `[ ]` Proxy baselines (`scripts/train_proxy_baselines.py` stub)
  - `[ ]` Real-time line-scan pipeline (no design doc yet)
  - `[ ]` Threshold calibration against real defect rates (needs factory data)
  - `[ ]` Regulatory compliance framing (FMVSS 208, ECE R16, IATF 16949)

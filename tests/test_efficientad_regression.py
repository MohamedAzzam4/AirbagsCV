"""
Regression test: EfficientAD completes 1 epoch on tiny synthetic data without
crashing on validation or checkpoint saving.

Why this test exists
--------------------
Two regressions were discovered during a real Colab EfficientAD training run:

1. ValueError: F1Score requires pred_label, but pred_label was missing
   during validation.  Root cause: F1AdaptiveThreshold (post-processor)
   is not fit until AFTER the first validation pass, so pred_label is None
   when F1Score.update() is called. Fix: omit F1Score from val_metrics
   (see train_models.py build_model docstring).

2. NotImplementedError: _SingleProcessDataLoaderIter cannot be pickled.
   Root cause: Anomalib 2.0.0's Evaluator callback holds a non-picklable
   reference; Lightning's full-state checkpoint tries to serialize it.
   Fix: save_weights_only=True + sanitize_hparams (see train_models.py
   build_checkpoint_callback docstring).

This test reproduces the exact scenario (tiny AITEX + tiny imagenette,
1 epoch, CPU) and asserts that:
  - engine.fit() completes without raising
  - validation runs (image_AUROC is logged)
  - checkpoint saving succeeds (last.ckpt exists)
  - metrics.json is written with image_AUROC populated

Marked slow because EfficientAD on CPU takes ~3-6 minutes even for 1 epoch
on tiny data (gradient descent + student-teacher update). Skip by default;
run explicitly with:  pytest tests/test_efficientad_regression.py -s

To run standalone (not via pytest):
    python tests/test_efficientad_regression.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


# --------------------------------------------------------------------------- #
# Fixtures: tiny synthetic AITEX + tiny synthetic imagenette                  #
# --------------------------------------------------------------------------- #
def _make_tiny_aitex(root: Path) -> None:
    """Create a 4-normal + 2-defect synthetic AITEX-shaped dataset."""
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("NODefect_images", "Defect_images", "Mask_images"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        img = rng.integers(60, 200, (256, 4096, 3), dtype=np.uint8)
        for y in range(0, 256, 8):
            img[y, :, :] = (img[y, :, :] * 0.7).astype(np.uint8)
        cv2.imwrite(str(root / "NODefect_images" / f"normal_{i:02d}.png"), img)
    for i in range(2):
        img = rng.integers(60, 200, (256, 4096, 3), dtype=np.uint8)
        for y in range(0, 256, 8):
            img[y, :, :] = (img[y, :, :] * 0.7).astype(np.uint8)
        mask = np.zeros((256, 4096), dtype=np.uint8)
        cv2.circle(img, (500 + i * 1000, 128), 20, (10, 10, 10), -1)
        cv2.circle(mask, (500 + i * 1000, 128), 20, 255, -1)
        cv2.imwrite(str(root / "Defect_images" / f"defect_{i:02d}.png"), img)
        cv2.imwrite(str(root / "Mask_images" / f"defect_{i:02d}_mask.png"), mask)


def _make_tiny_imagenette(root: Path) -> None:
    """Create a tiny ImageFolder-format imagenette (3 classes, 3 images each)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for cls in ("class_a", "class_b", "class_c"):
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
            cv2.imwrite(str(root / cls / f"img_{i:03d}.JPEG"), img)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Build tiny AITEX + tiny imagenette once per module run."""
    base = tmp_path_factory.mktemp("efficientad_regression")
    raw_aitex = base / "raw_aitex"
    imagenette = base / "imagenette_train"
    prepared = base / "prepared_aitex"
    run_dir = base / "runs"

    _make_tiny_aitex(raw_aitex)
    _make_tiny_imagenette(imagenette)

    # Run prepare_aitex.py
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "prepare_aitex.py"),
         "--source", str(raw_aitex),
         "--output", str(prepared),
         "--patch-size", "256",
         "--blank-threshold", "0.5",
         "--seed", "42",
         "--overwrite"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"prepare_aitex.py failed:\n{r.stderr}"

    return {
        "raw_aitex": raw_aitex,
        "imagenette": imagenette,
        "prepared": prepared,
        "run_dir": run_dir,
        "base": base,
    }


# --------------------------------------------------------------------------- #
# Test                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_efficientad_one_epoch_completes(tiny_dataset):
    """EfficientAD 1-epoch run on tiny synthetic data must complete without
    crashing on validation or checkpoint saving, and must produce:
      - last.ckpt
      - metrics.json with image_AUROC populated
    """
    prepared = tiny_dataset["prepared"]
    imagenette = tiny_dataset["imagenette"]
    run_dir = tiny_dataset["run_dir"]

    cmd = [
        sys.executable, str(SCRIPTS / "train_models.py"),
        "--model", "efficientad",
        "--dataset", "aitex",
        "--data-dir", str(prepared),
        "--output-dir", str(run_dir),
        "--epochs", "1",
        "--batch-size", "1",  # EfficientAD hard requirement
        "--num-workers", "0",
        "--accelerator", "cpu",
        "--devices", "1",
        "--precision", "32-true",
        "--seed", "42",
        "--imagenet-dir", str(imagenette),
        "--save-top-k", "1",
        "--log-level", "WARNING",
    ]

    # EfficientAD on CPU takes ~3-6 minutes for 1 epoch on this tiny set.
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    combined = r.stdout + "\n" + r.stderr

    # Must exit cleanly
    assert r.returncode == 0, (
        f"train_models.py exited {r.returncode}\n--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr}"
    )

    # Must NOT contain the two known crash signatures
    assert "field with name pred_label" not in combined, (
        "F1Score validation crash regression: val_metrics still contains F1Score"
    )
    assert "_SingleProcessDataLoaderIter cannot be pickled" not in combined, (
        "Checkpoint pickle crash regression: save_weights_only not applied"
    )

    # Checkpoint must exist
    last_ckpt = run_dir / "efficientad_aitex" / "weights" / "lightning" / "last.ckpt"
    assert last_ckpt.exists(), f"last.ckpt not found at {last_ckpt}"

    # metrics.json must exist and have image_AUROC populated (not null)
    metrics_path = run_dir / "efficientad_aitex" / "metrics.json"
    assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"
    metrics = json.loads(metrics_path.read_text())
    assert metrics.get("image_AUROC") is not None, (
        f"image_AUROC is null in metrics.json — validation did not produce a metric.\n"
        f"Full metrics: {json.dumps(metrics, indent=2)}"
    )

    # Print a summary so the test is informative when run with -s
    print("\n=== EfficientAD regression test PASSED ===")
    print(f"  image_AUROC: {metrics['image_AUROC']:.4f}")
    print(f"  pixel_AUROC: {metrics.get('pixel_AUROC')}")
    print(f"  image_F1Score: {metrics.get('image_F1Score')}")
    print(f"  train_seconds: {metrics.get('train_seconds')}")
    print(f"  checkpoint: {last_ckpt}")
    print(f"  metrics: {metrics_path}")
    print("  (Numbers are on tiny synthetic data — NOT real results.)")


if __name__ == "__main__":
    # Allow running standalone without pytest.
    import tempfile

    class _FakeTmpFactory:
        def mktemp(self, name):
            return Path(tempfile.mkdtemp(prefix=name + "_"))

    tiny = {
        "raw_aitex": None, "imagenette": None,
        "prepared": None, "run_dir": None, "base": None,
    }
    base = Path(tempfile.mkdtemp(prefix="efficientad_regression_"))
    tiny["raw_aitex"] = base / "raw_aitex"
    tiny["imagenette"] = base / "imagenette_train"
    tiny["prepared"] = base / "prepared_aitex"
    tiny["run_dir"] = base / "runs"
    tiny["base"] = base
    _make_tiny_aitex(tiny["raw_aitex"])
    _make_tiny_imagenette(tiny["imagenette"])
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "prepare_aitex.py"),
         "--source", str(tiny["raw_aitex"]),
         "--output", str(tiny["prepared"]),
         "--patch-size", "256", "--blank-threshold", "0.5",
         "--seed", "42", "--overwrite"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    try:
        test_efficientad_one_epoch_completes(tiny)
    finally:
        shutil.rmtree(base, ignore_errors=True)

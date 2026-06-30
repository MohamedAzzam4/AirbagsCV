"""
Inference wrapper for the Gradio demo.

Changes from the original:
* No relative paths. The repo root is resolved relative to this file.
* Checkpoints are found by walking the run directory deterministically (sorted
  by modification time), not by glob(...)[-1] which is filesystem-dependent.
* Explicit validation for: missing checkpoint, unsupported model, invalid
  image, missing run directory.
* Model cache is preserved but keyed on the resolved checkpoint path so a
  retrain is detected on next call (within the same process).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("inference")

# Resolve repo root relative to this file: <repo>/demo/inference.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# Models that are actually implemented and can be loaded.
SUPPORTED_MODELS = {"efficientad"}


# --------------------------------------------------------------------------- #
# Checkpoint resolution                                                       #
# --------------------------------------------------------------------------- #
def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Walk run_dir for *.ckpt and return the most recently modified one."""
    if not run_dir.exists():
        return None
    ckpts = list(run_dir.rglob("*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    return ckpts[-1]


def resolve_run_dir(model_name: str, dataset_name: str,
                    results_dir: Path | None = None) -> Path:
    """Return the directory where train_models.py would have written checkpoints."""
    base = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    return base / f"{model_name.lower()}_{dataset_name.lower()}"


# --------------------------------------------------------------------------- #
# Model cache                                                                 #
# --------------------------------------------------------------------------- #
# Key: (model_name, dataset_name, ckpt_path, str(results_dir))
# Value: (engine, model)
_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}


def get_model(
    model_name: str,
    dataset_name: str,
    results_dir: Path | None = None,
    checkpoint: Path | None = None,
) -> tuple[Any, Any]:
    """Load (and cache) the Anomalib engine + model.

    If `checkpoint` is given, it is used directly. Otherwise the latest
    checkpoint under <results_dir>/<model>_<dataset>/ is selected.
    """
    key_name = model_name.lower()
    if key_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name!r}. Supported: {sorted(SUPPORTED_MODELS)}. "
            "PatchCore was previously referenced but is NOT implemented in this repo."
        )

    if checkpoint is None:
        run_dir = resolve_run_dir(key_name, dataset_name.lower(), results_dir)
        checkpoint = find_latest_checkpoint(run_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found under {run_dir}. "
                "Train a model first with scripts/train_models.py."
            )
    elif not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cache_key = (key_name, dataset_name.lower(), str(checkpoint),
                 str(results_dir) if results_dir else "")
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Lazy imports — keeps `import inference` cheap for the smoke test.
    from anomalib.engine import Engine
    if key_name == "efficientad":
        from anomalib.models import EfficientAd
        model = EfficientAd.load_from_checkpoint(str(checkpoint))
    else:  # pragma: no cover — guarded by SUPPORTED_MODELS above
        raise ValueError(f"Unsupported model: {model_name!r}")

    model.eval()
    engine = Engine(accelerator="auto")
    _MODEL_CACHE[cache_key] = (engine, model)
    return engine, model


# --------------------------------------------------------------------------- #
# Prediction                                                                  #
# --------------------------------------------------------------------------- #
def _validate_image(image_path: str | Path) -> None:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception as e:
        raise ValueError(f"Invalid or unreadable image at {p}: {e}") from e


def predict(
    image_path: str | Path,
    model_name: str,
    dataset_name: str,
    results_dir: Path | None = None,
    checkpoint: Path | None = None,
) -> tuple[float, str, np.ndarray, np.ndarray]:
    """Run inference on a single image.

    Returns
    -------
    (score, label, heatmap_uint8, mask_uint8)
        heatmap_uint8: 256x256 (Anomalib default), values 0-255.
        mask_uint8:    256x256 binary mask, values 0 or 255.
    """
    _validate_image(image_path)
    engine, model = get_model(model_name, dataset_name, results_dir, checkpoint)

    predictions = engine.predict(model=model, data_path=str(image_path))
    if not predictions:
        raise RuntimeError("Anomalib returned no predictions.")
    batch = predictions[0]

    score = float(batch.pred_score[0].item())
    label = "ANOMALOUS" if bool(batch.pred_label[0].item()) else "NORMAL"

    heatmap = batch.anomaly_map[0].squeeze().cpu().numpy()
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    denom = h_max - h_min
    if denom < 1e-8:
        heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)
    else:
        heatmap_norm = ((heatmap - h_min) / denom * 255.0).astype(np.uint8)

    mask = batch.pred_mask[0].squeeze().cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8)) * 255

    return score, label, heatmap_norm, mask_uint8

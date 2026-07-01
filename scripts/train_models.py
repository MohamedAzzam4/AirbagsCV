"""
Train an anomaly-detection model on the prepared AITEX (or compatible) dataset.

Supported models
----------------
* `efficientad` — student-teacher network, gradient-based, requires imagenette
  for its penalty term, REQUIRES `--batch-size 1` (Anomalib 2.0.0 hard
  constraint). Multiple epochs make sense (typical recipe: 70).
* `patchcore`   — memory-bank model, NO gradient updates, NO imagenette
  requirement, supports larger batch sizes. Only ONE effective epoch is
  needed (the memory bank is fitted once via `on_train_epoch_end`). Passing
  `--epochs N` with N>1 will simply repeat the feature-extraction pass N
  times — useful only if you want to over-sample the train set; for a normal
  run use `--epochs 1`.

Outputs
-------
Both models write to the same layout under `<output-dir>/<model>_<dataset>/`:
* Lightning checkpoints under `weights/lightning/`:
    - EfficientAD: `last.ckpt` (per-epoch), `model.ckpt` (best by val image_AUROC),
      `epoch=N-step=M.ckpt` (top-k).
    - PatchCore:   `last.ckpt` (single-epoch), `model.ckpt` (best by val
      image_AUROC). PatchCore's "best" is effectively the only epoch's
      checkpoint; save_top_k=1 is recommended to avoid duplicates.
* `metrics.json`  — final metrics (image_AUROC, pixel_AUROC, etc.).
* A row appended to `<output-dir>/benchmark_results.csv`.

Note on "latency"
-----------------
The wall-clock time of `engine.test()` reported by this script is NOT
inference latency. It includes dataloader overhead, metric computation, and
batch dispatch. Use `scripts/benchmark_inference.py` for honest per-image
latency.

Resume
------
Pass `--resume-from-checkpoint` to continue from an existing `.ckpt`.
For EfficientAD this restores optimizer + scheduler + epoch counter (use
`last.ckpt`). For PatchCore, "resume" is conceptually meaningless because
the model has no optimizer state — the memory bank IS the trained state.
Loading a PatchCore checkpoint and calling `engine.fit()` again will
re-extract features and rebuild the memory bank from scratch. If you want
to evaluate a trained PatchCore model, just use `benchmark_inference.py`
or `demo/inference.py` with the checkpoint path.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger("train_models")


SUPPORTED_MODELS = ("efficientad", "patchcore")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an anomaly-detection model on AITEX-style data.")
    p.add_argument("--model", default="efficientad", choices=list(SUPPORTED_MODELS),
                    help="Model to train. 'efficientad': student-teacher, requires "
                         "--imagenet-dir, forces --batch-size 1. 'patchcore': "
                         "memory-bank, no imagenette, supports larger batch sizes, "
                         "only needs 1 epoch.")
    p.add_argument("--dataset", default="aitex",
                    help="Dataset name (used in output path only).")
    p.add_argument("--data-dir", type=Path, required=True,
                    help="Prepared dataset root (must contain train/good, test/good, "
                         "test/anomaly, ground_truth/anomaly).")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Where checkpoints and metrics are written.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Number of epochs. Default depends on model: efficientad=70, "
                         "patchcore=1. PatchCore only needs 1 epoch (memory bank is "
                         "fitted once); passing N>1 re-runs feature extraction N times.")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Train batch size. Default depends on model: efficientad=1 "
                         "(hard Anomalib 2.0.0 constraint), patchcore=8. EfficientAD "
                         "will warn+override any other value to 1.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--accelerator", default="auto",
                    choices=["auto", "gpu", "cpu", "cuda", "mps"])
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", default="32-true",
                    help="Lightning precision string, e.g. '32-true' or '16-mixed'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=256,
                    help="Input image size (Anomalib pre-processor will resize).")
    p.add_argument("--imagenet-dir", type=Path,
                    default=Path("./datasets/imagenette"),
                    help="Path to ImageFolder-format 'imagenette' dataset. "
                         "REQUIRED for efficientad (penalty term). NOT used by "
                         "patchcore.")
    # PatchCore-specific
    p.add_argument("--backbone", default="wide_resnet50_2",
                    help="PatchCore backbone (ignored for efficientad). "
                         "Default wide_resnet50_2; alternatives: resnet18, "
                         "resnet50, wide_resnet50_2.")
    p.add_argument("--coreset-sampling-ratio", type=float, default=0.1,
                    help="PatchCore coreset subsampling ratio in [0, 1]. "
                         "Lower = smaller memory bank = faster inference, "
                         "potentially lower recall. Ignored for efficientad.")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                    help="Path to a Lightning .ckpt to resume from. Recommended "
                         "for efficientad: last.ckpt. For patchcore, resume is "
                         "conceptually meaningless (no optimizer state); the "
                         "checkpoint will be loaded but training re-runs feature "
                         "extraction from scratch.")
    p.add_argument("--limit-train-batches", type=float, default=1.0,
                    help="Lightning limit_train_batches (debug only).")
    p.add_argument("--save-top-k", type=int, default=3,
                    help="Number of best-by-val-image_AUROC checkpoints to keep. "
                         "For patchcore, recommend 1 (only 1 epoch is meaningful).")
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def verify_data_dir(data_dir: Path) -> None:
    required = ["train/good", "test/good", "test/anomaly", "ground_truth/anomaly"]
    missing = [r for r in required if not (data_dir / r).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Data directory missing required subfolders: {missing}. "
            f"Run scripts/prepare_aitex.py first. Saw: "
            f"{sorted(p.name for p in data_dir.iterdir() if p.is_dir())}"
        )


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in model-specific defaults for --epochs and --batch-size if not set."""
    if args.epochs is None:
        args.epochs = 70 if args.model == "efficientad" else 1
        logger.info("Defaulting --epochs to %d for model=%s", args.epochs, args.model)
    if args.batch_size is None:
        args.batch_size = 1 if args.model == "efficientad" else 8
        logger.info("Defaulting --batch-size to %d for model=%s", args.batch_size, args.model)
    return args


def build_model(args: argparse.Namespace):
    """Instantiate the Anomalib model.

    Both models get a custom Evaluator with val_metrics AND test_metrics so
    that:
      - image_AUROC is logged during validation -> ModelCheckpoint can
        monitor it and select a best checkpoint.
      - All four metrics (image_AUROC, image_F1Score, pixel_AUROC,
        pixel_F1Score) are computed at test time and written to metrics.json.

    IMPORTANT: F1Score is NOT safe in val_metrics for either model
    ------------------------------------------------------------------
    Both EfficientAD and PatchCore share the same root cause: the
    post-processor's F1AdaptiveThreshold is not fit until AFTER the first
    validation pass completes. F1Score requires `pred_label`, which is set
    by the post-processor. During the first validation pass `pred_label` is
    None, so F1Score.update() raises:
        ValueError: Passed dataclass instance does not have a value for
        field with name pred_label

    We therefore use only `image_AUROC` and `pixel_AUROC` (which depend on
    `pred_score` and `anomaly_map` respectively — both always available)
    for val_metrics for BOTH models. F1Score IS included in test_metrics
    for both models (the post-processor has been fit by test time).

    This was discovered the hard way: the original code claimed F1Score was
    safe for EfficientAD's val_metrics, but a real Colab training run
    crashed with the ValueError above. Do NOT re-add F1Score to val_metrics
    without first verifying the post-processor is fit before the first val
    batch.
    """
    from anomalib.metrics import Evaluator, AUROC, F1Score

    # Test metrics — same for both models, computed at test time.
    # F1Score IS safe here because the post-processor's F1AdaptiveThreshold
    # is fit during the validation loop, so by test time pred_label/pred_mask
    # are populated.
    test_image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    test_image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    test_pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
    test_pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False)
    test_metrics = [test_image_auroc, test_image_f1score, test_pixel_auroc, test_pixel_f1score]

    # Validation metrics — SAME for both models.
    # F1Score is OMITTED because pred_label/pred_mask are not available
    # until the post-processor is fit, which happens after the first val pass.
    # See the docstring above for the full explanation.
    val_image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    val_pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
    val_metrics = [val_image_auroc, val_pixel_auroc]

    if args.model == "efficientad":
        from anomalib.models import EfficientAd

        # EfficientAD requires the imagenette dataset for its penalty term.
        imagenet_dir = args.imagenet_dir
        if not imagenet_dir.exists():
            raise FileNotFoundError(
                f"EfficientAD requires the 'imagenette' dataset at {imagenet_dir}. "
                f"Download it from https://github.com/fastai/imagenette "
                f"(the 'imagenette2' tarball, ~1.5 GB) and extract it, then pass "
                f"--imagenet-dir <path>. The folder must contain class subdirectories. "
                f"(PatchCore does NOT require imagenette — you can switch with --model patchcore.)"
            )
        subdirs = [p for p in imagenet_dir.iterdir() if p.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"--imagenet-dir {imagenet_dir} exists but has no class subdirectories. "
                f"It must be an ImageFolder layout (e.g. imagenette2/train/<class>/*.JPEG)."
            )

        val_evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
        return EfficientAd(
            model_size="small",
            teacher_out_channels=384,
            lr=1e-4,
            weight_decay=1e-5,
            padding=False,
            pad_maps=True,
            imagenet_dir=str(imagenet_dir),
            evaluator=val_evaluator,
        )

    if args.model == "patchcore":
        from anomalib.models import Patchcore

        if not 0.0 < args.coreset_sampling_ratio <= 1.0:
            raise ValueError(
                f"--coreset-sampling-ratio must be in (0, 1], got {args.coreset_sampling_ratio}"
            )
        logger.info(
            "Building Patchcore: backbone=%s, coreset_sampling_ratio=%.2f, num_neighbors=9",
            args.backbone, args.coreset_sampling_ratio,
        )
        val_evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
        return Patchcore(
            backbone=args.backbone,
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            num_neighbors=9,
            evaluator=val_evaluator,
        )

    raise ValueError(f"Unsupported model: {args.model!r}. Supported: {SUPPORTED_MODELS}")


def sanitize_hparams(model, model_name: str) -> list[str]:
    """Remove non-picklable objects from the Lightning module's hparams.

    Anomalib 2.0.0 saves the `evaluator` (an nn.Module) as a hyperparameter.
    When Lightning tries to serialize the full hparams dict into the
    checkpoint, this can trigger:
        NotImplementedError: _SingleProcessDataLoaderIter cannot be pickled
    because the Evaluator transitively holds references to dataloader
    state in some Anomalib versions.

    We strip `evaluator` (and any other nn.Module-valued hparams) from
    `model._hparams` and `model._hparams_initial` BEFORE engine.fit() so the
    checkpoint's hparams section only contains picklable scalars/strings.

    The Evaluator object itself is still saved in the checkpoint's state_dict
    (because it's a submodule), so model loading still works. We only remove
    it from the *hparams metadata* section.

    Returns the list of removed keys for logging.
    """
    import torch.nn as nn

    removed: list[str] = []
    for store_name in ("_hparams", "_hparams_initial"):
        store = getattr(model, store_name, None)
        if not isinstance(store, dict):
            continue
        for key in list(store.keys()):
            val = store[key]
            if isinstance(val, nn.Module):
                logger.info(
                    "[%s] removing non-picklable hparam %r from %s (type=%s)",
                    model_name, key, store_name, type(val).__name__,
                )
                del store[key]
                if key not in removed:
                    removed.append(key)
    return removed



def build_datamodule(args: argparse.Namespace):
    from anomalib.data import Folder

    # EfficientAd in Anomalib 2.0.0 REQUIRES train_batch_size=1.
    # PatchCore supports larger batch sizes (default 8).
    if args.model == "efficientad" and args.batch_size != 1:
        logger.warning(
            "EfficientAd requires train_batch_size=1 in Anomalib 2.0.0; "
            "ignoring --batch-size=%d and using 1.",
            args.batch_size,
        )
        train_bs = 1
    else:
        train_bs = args.batch_size

    return Folder(
        name=args.dataset,
        root=str(args.data_dir),
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/good",
        mask_dir="ground_truth/anomaly",
        train_batch_size=train_bs,
        eval_batch_size=8,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Deterministically find the newest checkpoint under run_dir.

    Prefers last.ckpt if it exists (Lightning's canonical resume target);
    otherwise falls back to the most recently modified .ckpt.
    """
    last_ckpts = list(run_dir.rglob("last.ckpt"))
    if last_ckpts:
        last_ckpts.sort(key=lambda p: p.stat().st_mtime)
        return last_ckpts[-1]

    ckpts = list(run_dir.rglob("*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    return ckpts[-1]


def find_best_checkpoint(run_dir: Path) -> Path | None:
    """Find the best checkpoint (monitored by val/image_AUROC)."""
    model_ckpts = list(run_dir.rglob("model.ckpt")) + list(run_dir.rglob("model-v*.ckpt"))
    if model_ckpts:
        model_ckpts.sort(key=lambda p: p.stat().st_mtime)
        return model_ckpts[-1]
    return None


def extract_metric(test_results: dict, key: str) -> float | None:
    """Anomalib metric keys changed between versions; try both."""
    for candidate in (key, f"test_{key}", f"{key}"):
        if candidate in test_results:
            v = test_results[candidate]
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def build_checkpoint_callback(run_dir: Path, save_top_k: int, model_name: str):
    """Build a Lightning ModelCheckpoint callback.

    Config:
      - save_last=True           -> always have last.ckpt
      - every_n_epochs=1         -> one checkpoint per epoch
      - monitor='image_AUROC'    -> best by validation image AUROC
      - mode='max'               -> higher AUROC is better
      - save_top_k=save_top_k    -> keep best N (1 recommended for patchcore)
      - filename='model'         -> best ckpt is model.ckpt
      - save_weights_only=True   -> see note below

    Anomalib's Engine will NOT add its own ModelCheckpoint if one is passed
    via callbacks (see Engine._setup_anomalib_callbacks). Safe + non-conflicting.

    IMPORTANT: why save_weights_only=True
    --------------------------------------
    A real Colab training run crashed with:
        NotImplementedError: _SingleProcessDataLoaderIter cannot be pickled
    when Lightning tried to save the full trainer state (optimizer_states,
    lr_scheduler_states, loops, callbacks) into the checkpoint. The root cause
    is that Anomalib 2.0.0's Evaluator callback transitively holds a
    reference to a dataloader iterator that is not picklable.

    Setting save_weights_only=True makes Lightning save ONLY:
      - model.state_dict() (backbone + memory bank / student-teacher weights)
      - the LightningModule's hparams (after sanitize_hparams strips the
        Evaluator nn.Module from them)
      - the model's registered buffers (e.g. PatchCore's _is_fitted)

    It does NOT save optimizer_states, lr_schedulers, loops, or the
    dataloader iterator. This means:
      ✅ Checkpoint saving succeeds (no pickle error).
      ✅ The checkpoint is loadable for inference and evaluation
         (benchmark_inference.py, demo/inference.py both work).
      ❌ Full optimizer-state resume is NOT available. Passing
         --resume-from-checkpoint to EfficientAD will load the model weights
         but Lightning will restart the optimizer and LR scheduler from
         scratch (epoch 0, initial LR). For EfficientAD this means resumed
         training is effectively a warm-restart, not a true continuation.
      ⚠️ For PatchCore this does not matter — PatchCore has no optimizer
         state anyway (configure_optimizers() returns None), so
         save_weights_only=True loses nothing.

    If you need true optimizer-state resume for EfficientAD in the future,
    you must either (a) upgrade to a newer Anomalib version where the
    Evaluator is picklable, or (b) fork Anomalib and make the Evaluator
    detach from the dataloader before checkpointing. Both are out of scope
    for this phase.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    ckpt_dir = run_dir / "weights" / "lightning"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="model",
        monitor="image_AUROC",
        mode="max",
        save_last=True,
        save_top_k=save_top_k,
        every_n_epochs=1,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    args = resolve_defaults(args)

    try:
        verify_data_dir(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2

    # Reproducibility.
    try:
        from pytorch_lightning import seed_everything
        seed_everything(args.seed, workers=True)
    except ImportError:
        from lightning.pytorch import seed_everything
        seed_everything(args.seed, workers=True)

    from anomalib.engine import Engine

    run_dir = args.output_dir / f"{args.model}_{args.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)
    logger.info("Model: %s, epochs: %d, batch_size: %d",
                args.model, args.epochs, args.batch_size)

    model = build_model(args)
    datamodule = build_datamodule(args)

    # Sanitize non-picklable hparams (Evaluator nn.Module) BEFORE engine.fit().
    # This prevents the `_SingleProcessDataLoaderIter cannot be pickled` crash
    # when Lightning serializes hparams into the checkpoint.
    removed_keys = sanitize_hparams(model, args.model)
    if removed_keys:
        logger.info(
            "Sanitized non-picklable hparams from %s model: %s "
            "(these are still saved inside state_dict as submodules; only the "
            "hparams metadata section is affected).",
            args.model, removed_keys,
        )

    # Build the explicit checkpoint callback.
    ckpt_callback = build_checkpoint_callback(run_dir, args.save_top_k, args.model)

    resume_path = None
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.exists():
            logger.error("Resume checkpoint not found: %s", args.resume_from_checkpoint)
            return 2
        resume_path = str(args.resume_from_checkpoint)
        logger.info("Resuming from checkpoint: %s", resume_path)
        if args.model == "patchcore":
            logger.warning(
                "PatchCore resume is conceptually meaningless: there is no "
                "optimizer state to restore. The checkpoint will be loaded, "
                "but engine.fit() will re-extract features and rebuild the "
                "memory bank from scratch. If you only want to evaluate, use "
                "scripts/benchmark_inference.py or demo/inference.py instead."
            )
        elif args.model == "efficientad":
            logger.warning(
                "EfficientAD resume is a WARM RESTART, not a true continuation: "
                "checkpoints are saved with save_weights_only=True (see "
                "build_checkpoint_callback docstring), so only model weights "
                "are restored. The optimizer state, LR scheduler state, and "
                "epoch counter are NOT restored — Lightning will restart them "
                "from epoch 0 with the initial LR. For full optimizer-state "
                "resume you would need a newer Anomalib version where the "
                "Evaluator is picklable. For pure evaluation, use "
                "scripts/benchmark_inference.py or demo/inference.py instead."
            )
    else:
        existing = find_latest_checkpoint(run_dir)
        if existing is not None:
            logger.info(
                "Found existing checkpoint in run dir: %s "
                "Pass --resume-from-checkpoint to continue training from it.",
                existing,
            )

    engine = Engine(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        default_root_dir=str(run_dir),
        callbacks=[ckpt_callback],
    )

    # --- Train -------------------------------------------------------------
    t0 = time.time()
    engine.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_path,
    )
    train_seconds = time.time() - t0
    logger.info("Training wall-clock: %.1f s (%.1f min)", train_seconds, train_seconds / 60)

    # --- Evaluate ----------------------------------------------------------
    # NOTE: this measures end-to-end test-loop time, NOT pure inference latency.
    # Use scripts/benchmark_inference.py for honest per-image latency.
    t1 = time.time()
    test_results_list = engine.test(model=model, datamodule=datamodule)
    eval_seconds = time.time() - t1
    test_results = test_results_list[0] if test_results_list else {}
    logger.info("Evaluation wall-clock (includes dataloader + metrics): %.1f s", eval_seconds)

    # --- Persist results ---------------------------------------------------
    actual_train_bs = 1 if args.model == "efficientad" else args.batch_size
    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": actual_train_bs,
        "image_size": args.image_size,
        "seed": args.seed,
        "train_seconds": round(train_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "image_AUROC": extract_metric(test_results, "image_AUROC"),
        "pixel_AUROC": extract_metric(test_results, "pixel_AUROC"),
        "image_F1Score": extract_metric(test_results, "image_F1Score"),
        "pixel_F1Score": extract_metric(test_results, "pixel_F1Score"),
        "eval_note": (
            "eval_seconds is end-to-end test-loop time including dataloader and "
            "metric aggregation. It is NOT inference latency. Use "
            "scripts/benchmark_inference.py for honest latency."
        ),
    }
    # Model-specific extras
    if args.model == "patchcore":
        metrics["backbone"] = args.backbone
        metrics["coreset_sampling_ratio"] = args.coreset_sampling_ratio

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to: %s", metrics_path)

    # Append a row to the benchmark CSV (create header if missing).
    csv_path = args.output_dir / "benchmark_results.csv"
    row = pd.DataFrame([metrics])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        out = pd.concat([existing, row], ignore_index=True)
    else:
        out = row
    out.to_csv(csv_path, index=False)
    logger.info("Benchmark CSV updated: %s", csv_path)

    # --- Print all paths prominently --------------------------------------
    last_ckpt = find_latest_checkpoint(run_dir)
    best_ckpt = find_best_checkpoint(run_dir)
    ckpt_dir = run_dir / "weights" / "lightning"

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — KEY PATHS")
    print("=" * 70)
    print(f"RUN_DIR={run_dir}")
    print(f"LAST_CHECKPOINT_PATH={last_ckpt if last_ckpt else 'Not available'}")
    if best_ckpt is not None:
        print(f"BEST_CHECKPOINT_PATH={best_ckpt}")
    else:
        print("BEST_CHECKPOINT_PATH=Not available: no monitored validation metric "
              "produced a checkpoint (check that validation ran and image_AUROC "
              "was logged during training)")
    print(f"EPOCH_CHECKPOINT_DIR={ckpt_dir}")
    print(f"METRICS_JSON_PATH={metrics_path}")
    print(f"METRICS_CSV_PATH={csv_path}")
    if args.model == "patchcore":
        print()
        print("NOTE: PatchCore is a memory-bank model. The 'checkpoint' is the")
        print("fitted memory bank + backbone weights. There is no optimizer state.")
        print("To evaluate, use scripts/benchmark_inference.py or demo/inference.py")
        print("with this checkpoint path. To 'resume' is conceptually meaningless")
        print("(re-running fit() rebuilds the memory bank from scratch).")
    elif args.model == "efficientad":
        print()
        print("NOTE: EfficientAD checkpoints are saved with save_weights_only=True")
        print("(weights + buffers + sanitized hparams only; no optimizer/LR state).")
        print("The checkpoint is loadable for inference and evaluation via")
        print("scripts/benchmark_inference.py or demo/inference.py.")
        print("--resume-from-checkpoint will load the weights but restart the")
        print("optimizer and epoch counter from scratch (warm restart, not a true")
        print("continuation). See build_checkpoint_callback docstring for details.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

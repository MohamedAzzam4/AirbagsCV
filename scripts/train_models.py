"""
Train an anomaly-detection model on the prepared AITEX (or compatible) dataset.

This is a thin wrapper around Anomalib's Engine. It intentionally supports
only EfficientAD for now, because that is the only model that has actually
been trained and verified in this repo. Adding PatchCore / PaDiM / etc. is
future work — do NOT pretend they are supported.

Outputs
-------
* Lightning checkpoints under <output-dir>/<model>_<dataset>/.../lightning/
    - last.ckpt              (overwritten each epoch)
    - epoch=N-step=M.ckpt    (one per epoch, save_top_k=3 keeps best 3 by image_AUROC)
    - model.ckpt             (best by validation image_AUROC, alias of the best epoch ckpt)
* A metrics JSON at   <output-dir>/<model>_<dataset>/metrics.json
* A CSV row appended to <output-dir>/benchmark_results.csv

Note on "latency"
-----------------
The wall-clock time of `engine.test()` reported by older versions of this
script was MISLABELLED as inference latency. It is not. It includes
dataloader overhead, metric computation, and batch dispatch. Use
`scripts/benchmark_inference.py` for honest per-image latency.

Resume
------
Pass --resume-from-checkpoint to continue from an existing .ckpt. Lightning
restores optimizer state, LR scheduler, and epoch counter. The
ModelCheckpoint callback's `last.ckpt` is the recommended resume target.
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


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientAD on AITEX-style data.")
    p.add_argument("--model", default="efficientad", choices=["efficientad"],
                    help="Only EfficientAD is currently supported.")
    p.add_argument("--dataset", default="aitex",
                    help="Dataset name (used in output path only).")
    p.add_argument("--data-dir", type=Path, required=True,
                    help="Prepared dataset root (must contain train/good, test/good, "
                         "test/anomaly, ground_truth/anomaly).")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Where checkpoints and metrics are written.")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=1,
                    help="EfficientAD REQUIRES batch_size=1 (Anomalib 2.0.0 hard "
                         "constraint, also matches the original paper). Any other "
                         "value triggers a warning and is overridden to 1.")
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
                    help="Path to ImageFolder-format 'imagenette' dataset required by "
                         "EfficientAD's penalty term. Download from "
                         "https://github.com/fastai/imagenette and pass --imagenet-dir.")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None,
                    help="Path to a Lightning .ckpt to resume from. Recommended: "
                         "the last.ckpt produced by a previous run.")
    p.add_argument("--limit-train-batches", type=float, default=1.0,
                    help="Lightning limit_train_batches (debug only).")
    p.add_argument("--save-top-k", type=int, default=3,
                    help="Number of best-by-val-image_AUROC checkpoints to keep.")
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


def build_model(args: argparse.Namespace):
    """Instantiate the Anomalib model. Only EfficientAD is wired up.

    We attach a custom Evaluator with val_metrics so that image_AUROC is
    logged during the validation loop. Without val_metrics, no metric is
    logged during training, and ModelCheckpoint(monitor='image_AUROC')
    cannot select a best checkpoint.
    """
    from anomalib.models import EfficientAd
    from anomalib.metrics import Evaluator, AUROC, F1Score

    # EfficientAD requires the imagenette dataset for its penalty term.
    imagenet_dir = args.imagenet_dir
    if not imagenet_dir.exists():
        raise FileNotFoundError(
            f"EfficientAD requires the 'imagenette' dataset at {imagenet_dir}. "
            f"Download it from https://github.com/fastai/imagenette "
            f"(the 'imagenette2' tarball, ~1.5 GB) and extract it, then pass "
            f"--imagenet-dir <path>. The folder must contain class subdirectories."
        )
    subdirs = [p for p in imagenet_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"--imagenet-dir {imagenet_dir} exists but has no class subdirectories. "
            f"It must be an ImageFolder layout (e.g. imagenette2/train/<class>/*.JPEG)."
        )

    # Validation metrics — needed so ModelCheckpoint can monitor image_AUROC.
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
    val_evaluator = Evaluator(
        val_metrics=[image_auroc, image_f1score, pixel_auroc],
    )

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


def build_datamodule(args: argparse.Namespace):
    from anomalib.data import Folder

    # NOTE 1: Anomalib 2.0.0's Folder does NOT take image_size or task; those
    # are owned by the model's pre_processor / post_processor.
    # NOTE 2: EfficientAd in Anomalib 2.0.0 REQUIRES train_batch_size=1.
    # The original EfficientAD paper also uses batch size 1 for the
    # student-teacher update. We hard-enforce this here.
    if args.model == "efficientad" and args.batch_size != 1:
        logger.warning(
            "EfficientAd requires train_batch_size=1 in Anomalib 2.0.0; "
            "ignoring --batch-size=%d and using 1.",
            args.batch_size,
        )
    train_bs = 1 if args.model == "efficientad" else args.batch_size
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
    # Walk run_dir for *.ckpt. Look for last.ckpt first.
    last_ckpts = list(run_dir.rglob("last.ckpt"))
    if last_ckpts:
        # If multiple, pick the most recently modified.
        last_ckpts.sort(key=lambda p: p.stat().st_mtime)
        return last_ckpts[-1]

    ckpts = list(run_dir.rglob("*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    return ckpts[-1]


def find_best_checkpoint(run_dir: Path) -> Path | None:
    """Find the best checkpoint (monitored by val/image_AUROC).

    Lightning names best checkpoints as 'model.ckpt' (when filename='model'
    and monitor is set) OR 'epoch=N-step=M.ckpt'. We look for 'model.ckpt'
    first, then any non-last.ckpt checkpoint.
    """
    # The ModelCheckpoint with filename='model' produces model.ckpt as the
    # best checkpoint (and model-vN.ckpt for subsequent top-k).
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


def build_checkpoint_callback(run_dir: Path, save_top_k: int):
    """Build a Lightning ModelCheckpoint callback.

    Config:
      - save_last=True           -> always have last.ckpt for resume
      - every_n_epochs=1         -> one checkpoint per epoch
      - monitor='image_AUROC'    -> best by validation image AUROC
      - mode='max'               -> higher AUROC is better
      - save_top_k=save_top_k    -> keep best N
      - filename='model'         -> best ckpt is model.ckpt
      - save_weights_only=False  -> full state (optimizer, scheduler) for resume

    Anomalib's Engine will NOT add its own ModelCheckpoint if one is passed
    via callbacks (see Engine._setup_anomalib_callbacks). So this is safe
    and non-conflicting.
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
        save_weights_only=False,
        auto_insert_metric_name=False,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

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

    # Imports must happen after seed (some are heavy).
    from anomalib.engine import Engine

    run_dir = args.output_dir / f"{args.model}_{args.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    model = build_model(args)
    datamodule = build_datamodule(args)

    # Build the explicit checkpoint callback.
    ckpt_callback = build_checkpoint_callback(run_dir, args.save_top_k)

    resume_path = None
    if args.resume_from_checkpoint is not None:
        if not args.resume_from_checkpoint.exists():
            logger.error("Resume checkpoint not found: %s", args.resume_from_checkpoint)
            return 2
        resume_path = str(args.resume_from_checkpoint)
        logger.info("Resuming from checkpoint: %s", resume_path)
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
    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": 1 if args.model == "efficientad" else args.batch_size,
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
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

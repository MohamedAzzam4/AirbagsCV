"""
Honest per-image inference latency benchmark for an Anomalib checkpoint.

Why this exists
---------------
The original repo measured `engine.test()` wall-clock time and reported it as
"inference latency per image". That number included dataloader overhead, batch
collation, metric computation, and result aggregation -- it was NOT inference
latency. This script measures the real thing.

What it measures
----------------
1. Loads a trained checkpoint.
2. Walks the test image directory (test/good + test/anomaly).
3. Runs N warmup predictions (default 20) so CUDA / cuDNN autotuning stabilises.
4. Runs N timed predictions (default 200), with torch.cuda.synchronize() on GPU.
5. Records per-image latency.
6. Reports p50 / p95 / p99 / mean / throughput (FPS).

What it does NOT measure
------------------------
* Full line-scan production throughput. Real airbag inspection runs at ~16 kHz
  line rate on 8K-wide fabric; this benchmark is patch-level only.
* Pre-processing beyond what Anomalib's pre_processor does inside the engine
  predict path.
* PLC reject-signal latency, network I/O, disk I/O for streaming capture.
* Edge hardware (Jetson, OpenVINO, TensorRT). For those, export the model first.

Why per-image and not batched?
------------------------------
Anomalib 2.0.0's model forward expects a tensor, and its full inference
graph (pre_processor -> model -> post_processor) is orchestrated by the
Lightning Engine via `engine.predict(path=...)`. Calling `model(batch)`
directly fails because ImageBatch is not a tensor. We therefore use
`engine.predict` per image, which is the official supported API. This is
slightly conservative (includes pre-processor + post-processor overhead),
but it reflects what a real deployment calling the model would pay.

Supported models
----------------
* `efficientad` — student-teacher network.
* `patchcore`   — memory-bank model. Note: PatchCore inference includes a
  KNN search against the memory bank, so its per-image latency is typically
  higher than EfficientAD on the same hardware, especially with a large
  coreset (high `coreset_sampling_ratio` or large train set).

Outputs
-------
* CSV at --output-csv with one row per benchmark run.
* JSON sibling for easy parsing.
* Human-readable summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark_inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Honest inference latency benchmark.")
    p.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to a Lightning .ckpt produced by train_models.py.")
    p.add_argument("--model", default="efficientad", choices=["efficientad", "patchcore"],
                    help="Model that produced the checkpoint. Both supported.")
    p.add_argument("--dataset", default="aitex")
    p.add_argument("--data-dir", type=Path, required=True,
                    help="Prepared dataset root (must contain test/good and test/anomaly).")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--batch-size", type=int, default=1,
                    help="Currently only batch_size=1 is supported (per-image predict).")
    p.add_argument("--warmup", type=int, default=20,
                    help="Number of warmup predictions (not timed).")
    p.add_argument("--iterations", type=int, default=200,
                    help="Number of timed predictions.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(name: str) -> str:
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def percentiles(values: list[float], ps: list[float]) -> dict[str, float]:
    out = {}
    arr = np.asarray(values, dtype=np.float64)
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(arr, p))
    return out


def collect_image_paths(data_dir: Path, max_n: int) -> list[Path]:
    """Gather image paths from test/good and test/anomaly (deterministic order)."""
    paths: list[Path] = []
    for sub in ["test/good", "test/anomaly"]:
        d = data_dir / sub
        if not d.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPEG"):
            paths.extend(sorted(d.glob(ext)))
    if not paths:
        raise FileNotFoundError(
            f"No images found under {data_dir}/test/{{good,anomaly}}. "
            "Run scripts/prepare_aitex.py first."
        )
    if len(paths) > max_n:
        # Deterministic stride to sample.
        stride = len(paths) // max_n
        paths = paths[::stride][:max_n]
    return paths


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 2
    if not args.data_dir.is_dir():
        logger.error("Data directory not found: %s", args.data_dir)
        return 2

    import torch
    from anomalib.engine import Engine

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Model: %s", args.model)

    # Load the right model class based on --model
    # weights_only=False is required because the custom Evaluator object
    # (with val_metrics/test_metrics) is saved in the checkpoint and PyTorch
    # 2.6+ defaults to weights_only=True which rejects non-allowlisted classes.
    # We trust our own checkpoints.
    if args.model == "efficientad":
        from anomalib.models import EfficientAd
        model = EfficientAd.load_from_checkpoint(str(args.checkpoint), weights_only=False)
    elif args.model == "patchcore":
        from anomalib.models import Patchcore
        model = Patchcore.load_from_checkpoint(str(args.checkpoint), weights_only=False)
    else:
        logger.error("Unsupported model: %s", args.model)
        return 2
    model.eval()

    # Engine handles accelerator selection; we pass --device just for logging.
    # Anomalib's predict() will use whatever accelerator the Engine picks.
    engine = Engine(accelerator=device if device != "cuda" else "gpu",
                    devices=1)

    # Collect image paths.
    needed = args.warmup + args.iterations
    image_paths = collect_image_paths(args.data_dir, max_n=needed * 2)
    logger.info("Collected %d image paths (need %d).", len(image_paths), needed)
    if len(image_paths) < needed:
        logger.warning(
            "Reducing iterations from %d to %d (not enough images).",
            args.iterations, max(0, len(image_paths) - args.warmup),
        )
        args.iterations = max(0, len(image_paths) - args.warmup)
        if args.iterations == 0:
            logger.error("Not enough images for benchmark.")
            return 3

    # Warmup
    logger.info("Warmup: %d predictions...", args.warmup)
    for i in range(args.warmup):
        p = image_paths[i % len(image_paths)]
        _ = engine.predict(model=model, data_path=str(p))
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    # Timed loop
    logger.info("Timed: %d predictions...", args.iterations)
    per_image_ms: list[float] = []
    for i in range(args.iterations):
        p = image_paths[(args.warmup + i) % len(image_paths)]
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = engine.predict(model=model, data_path=str(p))
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        per_image_ms.append((t1 - t0) * 1000.0)

    pct = percentiles(per_image_ms, [50, 95, 99])
    mean_ms = float(statistics.fmean(per_image_ms))
    throughput_fps = 1000.0 / mean_ms

    summary = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "dataset": args.dataset,
        "device": device,
        "batch_size": 1,
        "warmup_predictions": args.warmup,
        "timed_iterations": args.iterations,
        "mean_ms_per_image": round(mean_ms, 3),
        "p50_ms_per_image": round(pct["p50"], 3),
        "p95_ms_per_image": round(pct["p95"], 3),
        "p99_ms_per_image": round(pct["p99"], 3),
        "throughput_images_per_s": round(throughput_fps, 2),
        "note": (
            "Patch/image-level inference latency, measured through Anomalib's "
            "engine.predict() (includes pre_processor + model + post_processor). "
            "NOT a full line-scan production throughput number. Real airbag "
            "inspection runs at ~16 kHz line rate on 8K-wide fabric; "
            "benchmarking that requires the streaming + tiled pipeline which "
            "is not yet implemented."
        ),
    }

    print("\n==================== INFERENCE BENCHMARK ====================")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print("============================================================\n")

    # Save CSV (append or create).
    row = pd.DataFrame([summary])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        existing = pd.read_csv(args.output_csv)
        out = pd.concat([existing, row], ignore_index=True)
    else:
        out = row
    out.to_csv(args.output_csv, index=False)
    logger.info("CSV written: %s", args.output_csv)

    json_path = args.output_csv.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("JSON written: %s", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

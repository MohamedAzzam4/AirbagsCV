"""
UNVERIFIED — Model export to ONNX / OpenVINO.

This script exists but has NEVER been run successfully. There is no
`exports/` directory in the repo. Do NOT claim export support until someone
runs this end-to-end and produces artifacts.

Status:
    - PatchCore export path: STALE (PatchCore was never trained in this repo)
    - EfficientAD export path: written but untested
    - ONNX export: UNVERIFIED
    - OpenVINO export: UNVERIFIED
    - TensorRT export: NOT IMPLEMENTED

To verify export:
    1. Train an EfficientAD checkpoint with scripts/train_models.py.
    2. Run: python scripts/export_models.py --model efficientad --dataset aitex \\
           --checkpoint path/to/model.ckpt --output-dir ./exports
    3. Verify the artifacts exist (ls exports/onnx/, ls exports/openvino/).
    4. Load the ONNX model with onnxruntime and run inference on a test image.
    5. Compare the output to the PyTorch model's output (within tolerance).
    6. Update README.md and the Honest Status Table to mark these as verified.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("export_models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an Anomalib checkpoint to ONNX/OpenVINO.")
    p.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to a Lightning .ckpt produced by train_models.py.")
    p.add_argument("--model", default="efficientad", choices=["efficientad"],
                    help="Only EfficientAD is currently supported. PatchCore is NOT trained.")
    p.add_argument("--dataset", default="aitex")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--formats", nargs="+", default=["onnx", "openvino"],
                    choices=["onnx", "openvino", "torch"])
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 2

    logger.warning("=" * 60)
    logger.warning("THIS SCRIPT IS UNVERIFIED. It has never been run end-to-end")
    logger.warning("in this repo. If it fails, fix it and update README.md to")
    logger.warning("mark export as verified.")
    logger.warning("=" * 60)

    try:
        from anomalib.engine import Engine
        from anomalib.deploy import ExportType
        from anomalib.models import EfficientAd
    except ImportError as e:
        logger.error("Anomalib import failed: %s", e)
        return 3

    if args.model == "efficientad":
        model = EfficientAd.load_from_checkpoint(str(args.checkpoint))
    else:
        logger.error("Unsupported model: %s", args.model)
        return 2

    engine = Engine(accelerator="auto")

    format_map = {
        "onnx": ExportType.ONNX,
        "openvino": ExportType.OPENVINO,
        "torch": ExportType.TORCH,
    }

    for fmt in args.formats:
        out = args.output_dir / fmt / f"{args.model}_{args.dataset}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            engine.export(
                model=model,
                export_type=format_map[fmt],
                export_root=str(out),
            )
            logger.info("[OK] Exported %s to %s", fmt, out)
        except Exception as e:
            logger.error("[FAIL] Export %s failed: %s", fmt, e)

    logger.warning("Export attempt finished. Verify artifacts in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

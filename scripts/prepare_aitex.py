"""
Prepare the AITEX fabric defect dataset for unsupervised anomaly detection.

This script converts the raw AITEX dataset (4096x256 fabric images with defect
masks) into the Anomalib "Folder" layout:

    <output>/
        train/good/        # defect-free patches
        test/good/         # defect-free patches held out from training
        test/anomaly/      # patches containing defects
        ground_truth/anomaly/  # binary masks matching the test/anomaly patches

Key behaviour
-------------
* Image-level 80/20 split BEFORE patching, so train and test never share a
  source image (no leakage).
* Blank / mostly-black patches are filtered out (configurable threshold).
  AITEX images frequently contain solid-black border columns; without
  filtering they poison the "normal" distribution.
* Deterministic: a single --seed controls all shuffling.
* CLI-friendly: no hardcoded dataset path.
* Logs counts and structure so the user can verify the result.

This script does NOT download AITEX. AITEX is gated behind registration at
https://www.aitex.es/afid/ or a Kaggle mirror. The user must obtain it and
point --source at the extracted folder.
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

logger = logging.getLogger("prepare_aitex")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AITEX dataset for Anomalib unsupervised training."
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root of the raw AITEX dataset. Must contain NODefect_images/, "
        "Defect_images/, and Mask_images/ subdirectories.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory. Will be created if missing.",
    )
    p.add_argument("--patch-size", type=int, default=256, help="Patch width in pixels.")
    p.add_argument(
        "--blank-threshold",
        type=float,
        default=0.5,
        help="Skip a patch if more than this fraction of pixels are below "
        "--blank-intensity. Set to 1.0 to disable blank filtering.",
    )
    p.add_argument(
        "--blank-intensity",
        type=int,
        default=10,
        help="Pixel intensity (0-255) below which a pixel is considered 'dark'.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--normal-split",
        type=float,
        default=0.8,
        help="Fraction of NODefect source images used for training (rest go to test/good).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --output before writing. Default is to error out if it exists.",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Refuse to run if --output already exists (default).",
    )
    p.set_defaults(overwrite=False)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
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


def is_blank_patch(patch: np.ndarray, threshold: float, intensity: int) -> bool:
    """Return True if more than `threshold` fraction of pixels are < `intensity`."""
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark_frac = float(np.mean(patch < intensity))
    return dark_frac > threshold


def slice_image(img: np.ndarray, patch_size: int) -> Iterable[tuple[int, np.ndarray]]:
    """Yield (patch_index, patch) for non-overlapping vertical slices."""
    h, w = img.shape[:2]
    if h != patch_size:
        logger.warning(
            "Image height %d != patch_size %d; using full height.", h, patch_size
        )
    n = w // patch_size
    for i in range(n):
        yield i, img[:, i * patch_size : (i + 1) * patch_size]


def verify_source(source: Path) -> None:
    """Sanity-check the source directory layout."""
    required = ["NODefect_images", "Defect_images", "Mask_images"]
    missing = [r for r in required if not (source / r).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Source directory missing required subdirectories: {missing}. "
            f"Got: {sorted(p.name for p in source.iterdir() if p.is_dir())}"
        )


def reset_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists: {output}. "
                "Pass --overwrite to delete it first, or choose another --output."
            )
        logger.warning("Deleting existing output directory: %s", output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def process_normal_images(
    img_paths: list[Path],
    out_dir: Path,
    patch_size: int,
    blank_threshold: float,
    blank_intensity: int,
    stats: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read %s; skipping.", img_path)
            stats["unreadable_normal"] += 1
            continue
        for i, patch in slice_image(img, patch_size):
            if is_blank_patch(patch, blank_threshold, blank_intensity):
                stats["blank_patches_skipped"] += 1
                continue
            out_path = out_dir / f"{img_path.stem}_patch{i:02d}.png"
            cv2.imwrite(str(out_path), patch)
            stats["normal_patches_written"] += 1


def process_defect_images(
    img_paths: list[Path],
    mask_dir: Path,
    out_anomaly: Path,
    out_normal: Path,
    gt_anomaly: Path,
    patch_size: int,
    blank_threshold: float,
    blank_intensity: int,
    stats: dict,
) -> None:
    out_anomaly.mkdir(parents=True, exist_ok=True)
    out_normal.mkdir(parents=True, exist_ok=True)
    gt_anomaly.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read %s; skipping.", img_path)
            stats["unreadable_defect"] += 1
            continue
        h, w = img.shape[:2]

        # Combine all masks for this image (some images have multiple defect masks).
        mask_paths = list(mask_dir.glob(f"{img_path.stem}_mask*.png"))
        combined = np.zeros((h, w), dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                logger.warning("Could not read mask %s; skipping mask.", mp)
                continue
            if m.shape != (h, w):
                raise ValueError(
                    f"Mask {mp.name} shape {m.shape} does not match image {img_path.name} shape {(h, w)}."
                )
            combined = cv2.bitwise_or(combined, m)

        for i, patch_img in slice_image(img, patch_size):
            patch_mask = combined[:, i * patch_size : (i + 1) * patch_size]
            if is_blank_patch(patch_img, blank_threshold, blank_intensity):
                stats["blank_patches_skipped"] += 1
                continue
            if int(np.sum(patch_mask)) > 0:
                # Anomaly patch — write image AND matching mask (same filename).
                out_img = out_anomaly / f"{img_path.stem}_patch{i:02d}.png"
                out_mask = gt_anomaly / f"{img_path.stem}_patch{i:02d}.png"
                cv2.imwrite(str(out_img), patch_img)
                cv2.imwrite(str(out_mask), patch_mask)
                stats["defect_patches_written"] += 1
            else:
                # Defect-free region inside a defect image — goes to test/good
                # (NOT train/good, to keep train strictly from NODefect sources).
                out_img = out_normal / f"{img_path.stem}_patch{i:02d}.png"
                cv2.imwrite(str(out_img), patch_img)
                stats["normal_from_defect_images"] += 1


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        verify_source(args.source)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2

    try:
        reset_output(args.output, args.overwrite)
    except FileExistsError as e:
        logger.error(str(e))
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)

    normal_imgs = sorted((args.source / "NODefect_images").rglob("*.png"))
    defect_imgs = sorted((args.source / "Defect_images").rglob("*.png"))
    mask_dir = args.source / "Mask_images"

    logger.info("Source: %s", args.source)
    logger.info("Output: %s", args.output)
    logger.info(
        "Found %d NODefect images and %d Defect images.", len(normal_imgs), len(defect_imgs)
    )

    if not normal_imgs:
        logger.error("No NODefect images found. Aborting.")
        return 2

    # Image-level split BEFORE patching (no leakage).
    random.shuffle(normal_imgs)
    split_idx = int(len(normal_imgs) * args.normal_split)
    train_imgs = normal_imgs[:split_idx]
    test_normal_imgs = normal_imgs[split_idx:]
    logger.info(
        "Split: %d train sources, %d test-good sources (ratio=%.2f).",
        len(train_imgs), len(test_normal_imgs), args.normal_split,
    )

    stats = {
        "normal_patches_written": 0,
        "defect_patches_written": 0,
        "normal_from_defect_images": 0,
        "blank_patches_skipped": 0,
        "unreadable_normal": 0,
        "unreadable_defect": 0,
    }

    process_normal_images(
        train_imgs,
        args.output / "train" / "good",
        args.patch_size,
        args.blank_threshold,
        args.blank_intensity,
        stats,
    )
    process_normal_images(
        test_normal_imgs,
        args.output / "test" / "good",
        args.patch_size,
        args.blank_threshold,
        args.blank_intensity,
        stats,
    )
    process_defect_images(
        defect_imgs,
        mask_dir,
        args.output / "test" / "anomaly",
        args.output / "test" / "good",
        args.output / "ground_truth" / "anomaly",
        args.patch_size,
        args.blank_threshold,
        args.blank_intensity,
        stats,
    )

    logger.info("---- Summary ----")
    for k, v in stats.items():
        logger.info("  %-32s %d", k, v)
    logger.info("---- Output tree ----")
    for sub in ["train/good", "test/good", "test/anomaly", "ground_truth/anomaly"]:
        d = args.output / sub
        n = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        logger.info("  %-32s %d files  (%s)", sub, n, d)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

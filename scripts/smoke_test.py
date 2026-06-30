"""
Smoke test: cheap verification that the repo is importable and wired correctly.

This script does NOT require the dataset or a GPU. It checks:
1. Required Python packages can be imported.
2. Repo structure (scripts/, demo/) is intact.
3. The existing checkpoint (if any) is loadable by EfficientAd.
4. Demo dependencies are importable.

Exit codes:
    0 = all checks passed (or skipped with clear messages)
    1 = a hard failure occurred
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("smoke_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirbagsCV smoke test.")
    p.add_argument("--checkpoint", type=Path, default=None,
                    help="Optional checkpoint to test-load.")
    p.add_argument("--strict", action="store_true",
                    help="Treat warnings as failures (exit 1 on any warning).")
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #
def check_repo_structure() -> bool:
    ok = True
    expected = ["scripts", "demo", "requirements.txt", "README.md", "pyproject.toml"]
    for name in expected:
        p = REPO_ROOT / name
        if p.exists():
            logger.info("  [OK]    %s exists", name)
        else:
            logger.warning("  [MISS]  %s missing", name)
            ok = False
    return ok


def check_imports() -> bool:
    # Heavy imports we test one by one so a failure is informative.
    modules = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "Pillow"),
        ("gradio", "gradio"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("anomalib", "anomalib"),
        ("lightning", "lightning (pytorch-lightning)"),
    ]
    ok = True
    for mod, pkg in modules:
        try:
            importlib.import_module(mod)
            logger.info("  [OK]    import %s  (pkg: %s)", mod, pkg)
        except Exception as e:
            logger.warning("  [FAIL]  import %s  (pkg: %s): %s", mod, pkg, e)
            ok = False
    return ok


def check_demo_deps() -> bool:
    ok = True
    sys.path.insert(0, str(REPO_ROOT / "demo"))
    try:
        import inference  # noqa: F401
        logger.info("  [OK]    demo/inference.py imports cleanly")
    except Exception as e:
        logger.warning("  [FAIL]  demo/inference.py import failed: %s", e)
        ok = False
    # app.py imports gradio, which we already tested; just verify the file parses.
    app_path = REPO_ROOT / "demo" / "app.py"
    if app_path.exists():
        try:
            with open(app_path) as f:
                compile(f.read(), str(app_path), "exec")
            logger.info("  [OK]    demo/app.py parses cleanly")
        except SyntaxError as e:
            logger.warning("  [FAIL]  demo/app.py syntax error: %s", e)
            ok = False
    return ok


def check_checkpoint(checkpoint: Path | None) -> bool:
    if checkpoint is None:
        # Try to find one in the default location.
        run_dir = REPO_ROOT / "results" / "efficientad_aitex"
        if not run_dir.exists():
            logger.info("  [SKIP]  no checkpoint provided and no results/efficientad_aitex dir")
            return True
        ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            logger.info("  [SKIP]  no checkpoint under %s", run_dir)
            return True
        checkpoint = ckpts[-1]
        logger.info("  [INFO]  using latest checkpoint: %s", checkpoint)
    if not checkpoint.exists():
        logger.warning("  [FAIL]  checkpoint not found: %s", checkpoint)
        return False
    try:
        import torch
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        keys = list(ckpt.keys())
        hp = ckpt.get("hyper_parameters", {})
        logger.info("  [OK]    checkpoint loaded. Top-level keys: %s", keys)
        logger.info("  [INFO]  hparams: %s", dict(hp))
        logger.info("  [INFO]  epoch=%s  global_step=%s",
                    ckpt.get("epoch"), ckpt.get("global_step"))
        # Now try to actually load it through Anomalib.
        try:
            from anomalib.models import EfficientAd
            _ = EfficientAd.load_from_checkpoint(str(checkpoint))
            logger.info("  [OK]    EfficientAd.load_from_checkpoint succeeded")
        except Exception as e:
            logger.warning("  [FAIL]  EfficientAd.load_from_checkpoint failed: %s", e)
            return False
    except Exception as e:
        logger.warning("  [FAIL]  torch.load failed: %s", e)
        return False
    return True


def check_data_dir_hint() -> None:
    # Just informational; we cannot assume the dataset exists.
    candidates = [
        REPO_ROOT / "datasets" / "aitex",
        REPO_ROOT / "AITEX dataset",
    ]
    for c in candidates:
        if c.exists():
            logger.info("  [INFO]  found dataset dir: %s", c)
            return
    logger.info("  [INFO]  no prepared dataset found. Run scripts/prepare_aitex.py first.")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("AirbagsCV smoke test")
    logger.info("=" * 60)

    logger.info("[1/4] Repo structure:")
    s1 = check_repo_structure()

    logger.info("[2/4] Python imports:")
    s2 = check_imports()

    logger.info("[3/4] Demo modules:")
    s3 = check_demo_deps()

    logger.info("[4/4] Checkpoint (optional):")
    s4 = check_checkpoint(args.checkpoint)

    check_data_dir_hint()

    logger.info("=" * 60)
    logger.info("Summary: structure=%s  imports=%s  demo=%s  ckpt=%s",
                s1, s2, s3, s4)
    if args.strict:
        ok = s1 and s2 and s3 and s4
    else:
        # Without strict, the checkpoint check is informational.
        ok = s1 and s2 and s3
    if ok:
        logger.info("PASS")
        return 0
    logger.error("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())

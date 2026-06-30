"""
STUB — Synthetic defect generation (NOT IMPLEMENTED).

This file is a placeholder. It exists so the repo honestly documents what
*should* come next, without claiming work that has not been done.

Goal (when implemented):
    Generate synthetic defects on normal AITEX (or future airbag-fabric)
    patches so we can measure detection rate without real defect data.

Recommended techniques (from the research notes):
    1. CutPaste (Li et al. 2021) — cut a small region from a normal image
       and paste it elsewhere with optional rotation/color jitter.
    2. Perlin noise — add structured noise that resembles stains.
    3. Texture corruption — local Gaussian blur or frequency-domain
       perturbation that mimics weave breakdown.
    4. Scratch / tear / stain masks — procedural masks combined with
       inpainting or color modulation.

Why this matters:
    This is the single most important missing piece for the no-factory-data
    PoC. Synthetic defects let us measure precision/recall/FRR/FAR as if we
    had real defects, and choose a defensible operating threshold.

Status:
    NOT IMPLEMENTED. Do not import this expecting it to work.

To implement:
    1. Pick a technique (start with CutPaste — simplest, most cited).
    2. Implement a `generate(image: np.ndarray) -> (image, mask)` API.
    3. Add a CLI: --input-dir, --output-dir, --num-per-image, --technique.
    4. Add a smoke test that verifies the output shape and mask non-emptiness.
    5. Use the generated defects to evaluate an existing EfficientAD
       checkpoint (extend scripts/train_models.py or write a new evaluator).
"""
import sys


def main() -> int:
    print("STUB: scripts/generate_synthetic_defects.py is not implemented yet.")
    print("See the module docstring for the implementation plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
STUB — Cold-start ablation (NOT IMPLEMENTED).

This file is a placeholder. It exists so the repo honestly documents what
*should* come next, without claiming work that has not been done.

Goal (when implemented):
    Train EfficientAD with N normal samples for N in
    {10, 20, 50, 100, 200, 500, full}, then report image_AUROC and
    pixel_AUROC vs N. This empirically validates (or refutes) the
    literature claim that 20-50 normal samples is enough for >95% AUROC.

Why this matters:
    The cold-start claim is the central stakeholder pitch ("give us 50
    normal samples and we can start"). Right now it is a citation, not a
    measurement. Running this ablation turns it into a measurement.

Status:
    NOT IMPLEMENTED. Do not import this expecting it to work.

To implement:
    1. Reuse build_model() and build_datamodule() from train_models.py.
    2. For each N in the list above, randomly subsample the train/good
       folder to N images (use a fixed seed for reproducibility).
    3. Train for a fixed budget (e.g. 30 epochs) — same budget for all N.
    4. Evaluate on the FULL test set.
    5. Save a CSV: N, image_AUROC, pixel_AUROC, train_seconds.
    6. Plot AUROC vs N to results/cold_start_curve.png.

Output:
    results/cold_start_ablation.csv with columns:
        N, image_AUROC, pixel_AUROC, train_seconds, seed
"""
import sys


def main() -> int:
    print("STUB: scripts/run_cold_start_ablation.py is not implemented yet.")
    print("See the module docstring for the implementation plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

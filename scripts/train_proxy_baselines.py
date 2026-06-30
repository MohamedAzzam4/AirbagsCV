"""
STUB — Proxy dataset baselines (NOT IMPLEMENTED).

This file is a placeholder. It exists so the repo honestly documents what
*should* come next, without claiming work that has not been done.

Goal (when implemented):
    Train EfficientAD on multiple proxy datasets and report AUROC on each,
    so we can:
      a) prove the team can hit published baselines (MVTec Carpet >= 95%),
      b) build a "domain-shift ladder" showing how AUROC drops as the proxy
         moves closer to airbag fabric.

Datasets to cover (in priority order):
    1. MVTec AD Carpet  (published EfficientAD baseline ~99% image AUROC)
    2. MVTec AD Grid    (published baseline ~99%)
    3. AITEX            (current single result, but with full 70 epochs)
    4. DAGM             (synthetic textures, additional proxy)
    5. Texture-AD       (domain-shift stress test)

Why this matters:
    A skeptical factory engineer will ask "show me YOUR results, not
    someone else's." Hitting >=95% on MVTec Carpet proves the toolchain
    works. Hitting a lower number on AITEX (textile, closer to airbag)
    honestly demonstrates the domain gap.

Status:
    NOT IMPLEMENTED. Do not import this expecting it to work.

To implement:
    1. MVTec requires registration. Use Anomalib's MVTecAD datamodule
       (handles download).
    2. AITEX uses scripts/prepare_aitex.py.
    3. Wrap each in a loop over (dataset, model) and call train_models.py
       via subprocess (don't reimplement the training call).
    4. Aggregate results into results/proxy_baselines.csv.

Output:
    results/proxy_baselines.csv with columns:
        model, dataset, image_AUROC, pixel_AUROC, train_seconds, epochs, seed
"""
import sys


def main() -> int:
    print("STUB: scripts/train_proxy_baselines.py is not implemented yet.")
    print("See the module docstring for the implementation plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

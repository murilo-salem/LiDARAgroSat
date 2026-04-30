from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.instance_targets import build_instance_targets
from lidaragrosat.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alternative instance targets per plot.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--n-rays", type=int, default=8)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "instance_targets")
    for plot_id, bundle in bundles.items():
        targets = build_instance_targets(bundle, n_rays=args.n_rays)
        plot_dir = ensure_dir(output_root / plot_id)
        np.save(plot_dir / "distance_transform.npy", targets.distance_transform.astype(np.float32))
        np.save(plot_dir / "seedness.npy", targets.seedness.astype(np.float32))
        np.save(plot_dir / "pseudo_crown_mask.npy", targets.pseudo_crown_mask.astype(np.float32))
        if targets.star_convex_rays is not None:
            np.save(plot_dir / "star_convex_rays.npy", targets.star_convex_rays.astype(np.float32))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(targets.pseudo_crown_mask, cmap="gray")
        axes[0].set_title("Pseudo crown mask")
        axes[1].imshow(targets.distance_transform, cmap="viridis")
        axes[1].set_title("Distance transform")
        axes[2].imshow(targets.seedness, cmap="magma")
        axes[2].set_title("Seedness")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(plot_dir / "qa_targets.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()

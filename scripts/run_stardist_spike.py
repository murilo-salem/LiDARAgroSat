from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.instance_targets import build_instance_targets
from lidaragrosat.utils.io import ensure_dir


def _polygon_from_rays(center_x: float, center_y: float, rays: np.ndarray) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=len(rays), endpoint=False)
    xs = center_x + rays * np.cos(angles)
    ys = center_y + rays * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal StarDist-like spike from pseudo crown rays.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--plots", nargs="*", default=["plot_01", "plot_06"])
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=args.plots)
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "stardist_spike")
    decision = "congelar"
    reasons = ["spike funcional gerou poligonos pseudo-star-convex", "nao ha mascaras de copa manuais para validar ganho de forma confiavel"]

    for plot_id, bundle in bundles.items():
        targets = build_instance_targets(bundle, n_rays=8)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(bundle.chm, cmap="viridis")
        for _, row in bundle.reference.iterrows():
            center_row, center_col = bundle.transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
            if not (0 <= center_row < bundle.chm.shape[0] and 0 <= center_col < bundle.chm.shape[1]):
                continue
            rays = targets.star_convex_rays[:, center_row, center_col]
            polygon = _polygon_from_rays(float(center_col), float(center_row), rays / bundle.transform.pixel_size)
            ax.plot(polygon[:, 0], polygon[:, 1], color="white", linewidth=0.8)
        ax.set_title(plot_id)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_root / f"{plot_id}_star_spike.png", dpi=180)
        plt.close(fig)

    with open(output_root / "decision.json", "w", encoding="utf-8") as handle:
        json.dump({"decision": decision, "reasons": reasons, "plots": args.plots}, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()

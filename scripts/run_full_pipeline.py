from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _run(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=workdir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full LiDARAgroSat pipeline.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    base = args.base_config

    python = sys.executable

    _run([python, "scripts/prepare_dataset.py", "--config", base], workdir)
    _run([python, "scripts/run_baseline.py", "--config", base], workdir)
    for model_config in [
        "configs/rgb_only.yaml",
        "configs/lidar_only.yaml",
        "configs/early_fusion.yaml",
        "configs/multimodal_mid.yaml",
        "configs/multimodal_late.yaml",
    ]:
        _run([python, "scripts/train_cv.py", "--config", base, model_config], workdir)
    _run(
        [
            python,
            "scripts/evaluate_experiments.py",
            "--config",
            base,
            "--reference-method",
            "baseline_local_maxima",
            "--fusion-method",
            "multimodal_mid_fusion",
            "--baseline-method",
            "lidar_only",
        ],
        workdir,
    )


if __name__ == "__main__":
    main()

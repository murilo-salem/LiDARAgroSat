from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.utils.io import ensure_dir


def _match_predictions(teacher_a: pd.DataFrame, teacher_b: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if teacher_a.empty or teacher_b.empty:
        return pd.DataFrame(columns=["x", "y", "height", "confidence"])
    xy_a = teacher_a[["x", "y"]].to_numpy(dtype=float)
    xy_b = teacher_b[["x", "y"]].to_numpy(dtype=float)
    records: list[dict] = []
    used_b: set[int] = set()
    for index_a, point_a in enumerate(xy_a):
        distances = np.sqrt(((xy_b - point_a[None, :]) ** 2).sum(axis=1))
        nearest_b = int(np.argmin(distances))
        if nearest_b in used_b or distances[nearest_b] > radius_m:
            continue
        used_b.add(nearest_b)
        row_a = teacher_a.iloc[index_a]
        row_b = teacher_b.iloc[nearest_b]
        records.append(
            {
                "x": float((float(row_a["x"]) + float(row_b["x"])) / 2.0),
                "y": float((float(row_a["y"]) + float(row_b["y"])) / 2.0),
                "height": float(np.nanmean([row_a.get("height_pred", np.nan), row_b.get("height_pred", np.nan)])),
                "confidence": float(np.nanmean([row_a.get("confidence", 0.0), row_b.get("confidence", 0.0)])),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from agreement between two teachers.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--teacher-a", required=True)
    parser.add_argument("--teacher-b", required=True)
    parser.add_argument("--agreement-radius-m", type=float, default=1.0)
    parser.add_argument("--min-confidence-a", type=float, default=0.1)
    parser.add_argument("--min-confidence-b", type=float, default=0.1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    if config.data.split_mode == "basic":
        splits = build_basic_split(
            list(bundles.keys()),
            train_fraction=config.data.basic_train_fraction,
            val_fraction=config.data.basic_val_fraction,
            seed=config.data.basic_split_seed,
        )
    else:
        splits = build_lopo_splits(list(bundles.keys()), validation_plots_per_fold=config.training.validation_plots_per_fold)

    prediction_root = config.resolve_path(config.paths.output_dir) / "predictions"
    output_dir = ensure_dir(args.output_dir)
    for split in splits:
        fold_a = prediction_root / args.teacher_a / f"fold_{split.fold_id:02d}"
        fold_b = prediction_root / args.teacher_b / f"fold_{split.fold_id:02d}"
        for plot_id in split.train_plots + split.val_plots + split.test_plots:
            path_a = fold_a / f"{plot_id}.csv"
            path_b = fold_b / f"{plot_id}.csv"
            if not path_a.exists() or not path_b.exists():
                continue
            teacher_a = pd.read_csv(path_a)
            teacher_b = pd.read_csv(path_b)
            if "confidence" in teacher_a.columns:
                teacher_a = teacher_a[teacher_a["confidence"] >= args.min_confidence_a].copy()
            if "confidence" in teacher_b.columns:
                teacher_b = teacher_b[teacher_b["confidence"] >= args.min_confidence_b].copy()
            agreed = _match_predictions(teacher_a, teacher_b, radius_m=args.agreement_radius_m)
            agreed.to_csv(Path(output_dir) / f"{plot_id}.csv", index=False)


if __name__ == "__main__":
    main()

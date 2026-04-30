from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.baselines.local_maxima import LocalMaximaParams, detect_local_maxima, tune_local_maxima
from lidaragrosat.benchmarking import require_protocol_id
from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.experiment_manifest import create_experiment_manifest, save_experiment_manifest
from lidaragrosat.utils.io import ensure_dir, write_json
from lidaragrosat.utils.logging import configure_logging, get_logger
from lidaragrosat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CHM local maxima baseline with LOPO CV.")
    parser.add_argument("--config", nargs="+", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    require_protocol_id(config.benchmark.protocol_id)
    seed_everything(config.training.seed)
    configure_logging(Path(config.paths.output_dir) / "logs", "baseline")
    logger = get_logger("baseline")

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
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "predictions" / "baseline_local_maxima")

    manifest = create_experiment_manifest(
        config,
        split={"split_mode": config.data.split_mode, "folds": [split.to_dict() for split in splits]},
        decode_params={
            "baseline_window_sizes_m": list(config.evaluation.baseline_window_sizes_m),
            "baseline_min_heights_m": list(config.evaluation.baseline_min_heights_m),
            "chm_gaussian_sigmas": list(config.preprocessing.chm_gaussian_sigmas),
        },
        stage1_config={"stage1_kind": config.benchmark.stage1 or "chm_local_maxima"},
        refiner="baseline_local_maxima",
        cwd=Path(__file__).resolve().parents[1],
    )
    save_config(config, output_root / "config.yaml")
    save_experiment_manifest(manifest, output_root / "experiment_manifest.json")

    params_grid = [
        LocalMaximaParams(window_radius_m=window, gaussian_sigma=sigma, min_height_m=min_height)
        for window in config.evaluation.baseline_window_sizes_m
        for sigma in config.preprocessing.chm_gaussian_sigmas
        for min_height in config.evaluation.baseline_min_heights_m
    ]

    fold_rows: list[dict] = []
    for split in splits:
        logger.info("Running baseline fold %d", split.fold_id)
        best_params = tune_local_maxima(
            chm_by_plot={plot_id: bundles[plot_id].chm for plot_id in split.val_plots},
            transform_by_plot={plot_id: bundles[plot_id].transform for plot_id in split.val_plots},
            reference_by_plot={plot_id: bundles[plot_id].reference for plot_id in split.val_plots},
            validation_plots=split.val_plots,
            params_grid=params_grid,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        fold_dir = ensure_dir(output_root / f"fold_{split.fold_id:02d}")
        write_json(asdict(best_params), fold_dir / "best_params.json")

        for plot_id in split.test_plots:
            predictions = detect_local_maxima(bundles[plot_id].chm, bundles[plot_id].transform, best_params)
            predictions.to_csv(fold_dir / f"{plot_id}.csv", index=False)
            match_result = match_tree_predictions(
                predictions,
                bundles[plot_id].reference,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            fold_rows.append(
                {
                    "fold_id": split.fold_id,
                    "plot_id": plot_id,
                    "tp": len(match_result.matches),
                    "fp": len(match_result.false_positives),
                    "fn": len(match_result.false_negatives),
                    "best_window_radius_m": best_params.window_radius_m,
                    "best_gaussian_sigma": best_params.gaussian_sigma,
                    "best_min_height_m": best_params.min_height_m,
                }
            )
    pd.DataFrame(fold_rows).to_csv(output_root / "fold_summary.csv", index=False)
    logger.info("Baseline completed for %d folds.", len(splits))


if __name__ == "__main__":
    main()

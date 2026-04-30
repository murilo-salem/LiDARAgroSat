from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.baselines.local_maxima import LocalMaximaParams, detect_local_maxima
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.utils.io import ensure_dir, read_json


def _parse_method_roots(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        method, template = value.split("=", 1)
        mapping[method] = template
    return mapping


def _load_validation_scores_for_method(
    method: str,
    root_template: str,
    seed: int,
    val_plot: str,
    bundles,
    config: ExperimentConfig,
) -> dict[str, float]:
    root = Path(root_template.format(seed=seed))
    fold_dir = root / "predictions" / method / "fold_00"
    if method == "baseline_local_maxima":
        params = read_json(fold_dir / "best_params.json")
        detect_params = LocalMaximaParams(
            window_radius_m=float(params["window_radius_m"]),
            gaussian_sigma=float(params["gaussian_sigma"]),
            min_height_m=float(params["min_height_m"]),
        )
        predictions = detect_local_maxima(
            bundles[val_plot].chm,
            bundles[val_plot].transform,
            detect_params,
        )
        match = match_tree_predictions(
            predictions,
            bundles[val_plot].reference,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        scores = summarize_matches(match)
        return {
            "precision": float(scores["precision"]),
            "recall": float(scores["recall"]),
            "f1": float(scores["f1"]),
        }
    return read_json(fold_dir / "validation_scores.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate simple development validation over 3 fixed seeds.")
    parser.add_argument("--base-configs", nargs="+", required=True)
    parser.add_argument("--seed-configs", nargs="+", required=True)
    parser.add_argument("--method-roots", nargs="+", required=True, help="Entries like method=outputs_suite_seed{seed}")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    method_roots = _parse_method_roots(args.method_roots)
    rows: list[dict] = []

    for seed_config in args.seed_configs:
        config = ExperimentConfig.from_files(*args.base_configs, seed_config)
        bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
        split = build_basic_split(
            list(bundles.keys()),
            train_fraction=config.data.basic_train_fraction,
            val_fraction=config.data.basic_val_fraction,
            seed=config.data.basic_split_seed,
        )[0]
        val_plot = split.val_plots[0]
        for method, root_template in method_roots.items():
            scores = _load_validation_scores_for_method(
                method=method,
                root_template=root_template,
                seed=config.data.basic_split_seed,
                val_plot=val_plot,
                bundles=bundles,
                config=config,
            )
            rows.append(
                {
                    "seed": int(config.data.basic_split_seed),
                    "val_plot": val_plot,
                    "method": method,
                    "precision": float(scores["precision"]),
                    "recall": float(scores["recall"]),
                    "f1": float(scores["f1"]),
                }
            )

    table = pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)
    summary = (
        table.groupby("method")
        .agg(
            n_seeds=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )
    pivot = table.pivot(index="seed", columns="method", values="f1")
    comparisons = []
    methods = list(pivot.columns)
    for reference_method in methods:
        for comparison_method in methods:
            if reference_method == comparison_method:
                continue
            paired = pivot[[reference_method, comparison_method]].dropna()
            delta = paired[comparison_method] - paired[reference_method]
            comparisons.append(
                {
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "n_seeds": int(len(delta)),
                    "mean_delta_f1": float(delta.mean()),
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                }
            )
    comparison_table = pd.DataFrame(comparisons).sort_values(["reference_method", "mean_delta_f1"], ascending=[True, False])

    output_dir = ensure_dir(args.output_dir)
    table.to_csv(Path(output_dir) / "validation_by_seed.csv", index=False)
    summary.to_csv(Path(output_dir) / "validation_summary.csv", index=False)
    comparison_table.to_csv(Path(output_dir) / "validation_pairwise.csv", index=False)


if __name__ == "__main__":
    main()

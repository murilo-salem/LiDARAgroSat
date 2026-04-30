from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import (
    FaithfulBaselineParams,
    FaithfulMatchingParams,
    FaithfulPreprocessParams,
    faithful_greedy_match,
    local_maxima_filter_3d,
    preprocess_vendor_lidar_plot,
    summarize_faithful_matches,
)
from lidaragrosat.utils.io import ensure_dir, write_json


def _discover_plot_dirs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.iterdir() if path.is_dir())


def _ensure_cache(
    raw_dir: Path,
    cache_dir: Path,
    preprocess_params: FaithfulPreprocessParams,
) -> None:
    required = [cache_dir / f"{plot_dir.name}.npz" for plot_dir in _discover_plot_dirs(raw_dir)]
    required += [cache_dir / f"{plot_dir.name}_reference.csv" for plot_dir in _discover_plot_dirs(raw_dir)]
    if all(path.exists() for path in required):
        return
    for plot_dir in _discover_plot_dirs(raw_dir):
        plot_id = plot_dir.name
        points_xyz, reference, _ = preprocess_vendor_lidar_plot(
            lidar_path=plot_dir / "lidar.las",
            reference_path=plot_dir / "trees_reference.csv",
            params=preprocess_params,
        )
        np.savez_compressed(cache_dir / f"{plot_id}.npz", points_xyz=points_xyz.astype(np.float32))
        reference.to_csv(cache_dir / f"{plot_id}_reference.csv", index=False)


def _load_cache(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    points_by_plot: dict[str, np.ndarray] = {}
    reference_by_plot: dict[str, pd.DataFrame] = {}
    for npz_path in sorted(cache_dir.glob("plot_*.npz")):
        plot_id = npz_path.stem
        points_by_plot[plot_id] = np.load(npz_path)["points_xyz"].astype(np.float64)
        reference_by_plot[plot_id] = pd.read_csv(cache_dir / f"{plot_id}_reference.csv")
    return points_by_plot, reference_by_plot


def _evaluate_params(
    points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    baseline_params: FaithfulBaselineParams,
    matching_params: FaithfulMatchingParams,
) -> tuple[pd.DataFrame, dict[str, float]]:
    per_plot_rows: list[dict] = []
    total_tp = total_fp = total_fn = 0
    for plot_id in sorted(points_by_plot):
        detections = local_maxima_filter_3d(
            points_xyz=points_by_plot[plot_id],
            window_size_m=baseline_params.window_size_m,
            height_threshold_m=baseline_params.height_threshold_m,
        )
        match_result = faithful_greedy_match(
            predictions=detections,
            reference=reference_by_plot[plot_id],
            params=matching_params,
        )
        metrics = summarize_faithful_matches(match_result)
        metrics["plot_id"] = plot_id
        metrics["n_detections"] = int(len(detections))
        metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
        per_plot_rows.append(metrics)
        total_tp += int(metrics["tp"])
        total_fp += int(metrics["fp"])
        total_fn += int(metrics["fn"])

    per_plot = pd.DataFrame.from_records(per_plot_rows).sort_values("plot_id").reset_index(drop=True)
    precision_global = float(total_tp / (total_tp + total_fp)) if total_tp + total_fp > 0 else 0.0
    recall_global = float(total_tp / (total_tp + total_fn)) if total_tp + total_fn > 0 else 0.0
    f1_global = (
        float(2 * precision_global * recall_global / (precision_global + recall_global))
        if precision_global + recall_global > 0
        else 0.0
    )
    summary = {
        "precision_global": precision_global,
        "recall_global": recall_global,
        "f1_global": f1_global,
        "precision_macro": float(per_plot["precision"].mean()),
        "recall_macro": float(per_plot["recall"].mean()),
        "f1_macro": float(per_plot["f1"].mean()),
        "distance_macro": float(per_plot["distance_m"].mean()),
        "height_error_macro": float(per_plot["height_error_m"].dropna().mean()) if per_plot["height_error_m"].notna().any() else np.nan,
        "mean_detections": float(per_plot["n_detections"].mean()),
        "tp_total": int(total_tp),
        "fp_total": int(total_fp),
        "fn_total": int(total_fn),
    }
    return per_plot, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna on the faithful paper baseline.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--output-dir", default="outputs_faithful_paper_optuna")
    parser.add_argument("--study-name", default="faithful_baseline_tpe")
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--window-min", type=float, default=0.5)
    parser.add_argument("--window-max", type=float, default=5.0)
    parser.add_argument("--window-step", type=float, default=0.1)
    parser.add_argument("--height-threshold-min", type=float, default=1.5)
    parser.add_argument("--height-threshold-max", type=float, default=5.0)
    parser.add_argument("--height-threshold-step", type=float, default=0.1)
    parser.add_argument("--max-dist", type=float, default=5.0)
    parser.add_argument("--max-height-diff", type=float, default=3.0)
    parser.add_argument("--duplicate-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--manual-min-abs-z", type=float, default=None)
    parser.add_argument("--manual-max-abs-z", type=float, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    cache_dir = ensure_dir(Path(args.cache_dir).resolve())
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    predictions_dir = ensure_dir(output_dir / "best_predictions")

    preprocess_params = FaithfulPreprocessParams(
        duplicate_tolerance_m=float(args.duplicate_tolerance_mm) / 1000.0,
        manual_min_abs_z=args.manual_min_abs_z,
        manual_max_abs_z=args.manual_max_abs_z,
        ground_classification=2,
    )
    matching_params = FaithfulMatchingParams(
        max_dist_m=float(args.max_dist),
        max_height_diff_m=float(args.max_height_diff),
    )
    _ensure_cache(raw_dir, cache_dir, preprocess_params)
    points_by_plot, reference_by_plot = _load_cache(cache_dir)

    storage_path = output_dir / "study.sqlite3"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        baseline_params = FaithfulBaselineParams(
            window_size_m=trial.suggest_float(
                "window_size_m",
                float(args.window_min),
                float(args.window_max),
                step=float(args.window_step),
            ),
            height_threshold_m=trial.suggest_float(
                "height_threshold_m",
                float(args.height_threshold_min),
                float(args.height_threshold_max),
                step=float(args.height_threshold_step),
            ),
        )
        _, summary = _evaluate_params(points_by_plot, reference_by_plot, baseline_params, matching_params)
        trial.set_user_attr("precision_global", summary["precision_global"])
        trial.set_user_attr("recall_global", summary["recall_global"])
        trial.set_user_attr("f1_macro", summary["f1_macro"])
        trial.set_user_attr("precision_macro", summary["precision_macro"])
        trial.set_user_attr("recall_macro", summary["recall_macro"])
        trial.set_user_attr("distance_macro", summary["distance_macro"])
        trial.set_user_attr("mean_detections", summary["mean_detections"])
        trial.set_user_attr("tp_total", summary["tp_total"])
        trial.set_user_attr("fp_total", summary["fp_total"])
        trial.set_user_attr("fn_total", summary["fn_total"])
        return float(summary["f1_global"])

    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=False)

    trials_table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_table.to_csv(output_dir / "trials.csv", index=False)

    best_trial = study.best_trial
    best_params = FaithfulBaselineParams(
        window_size_m=float(best_trial.params["window_size_m"]),
        height_threshold_m=float(best_trial.params["height_threshold_m"]),
    )
    per_plot, summary = _evaluate_params(points_by_plot, reference_by_plot, best_params, matching_params)
    per_plot.to_csv(output_dir / "best_per_plot.csv", index=False)

    for plot_id in sorted(points_by_plot):
        detections = local_maxima_filter_3d(
            points_xyz=points_by_plot[plot_id],
            window_size_m=best_params.window_size_m,
            height_threshold_m=best_params.height_threshold_m,
        )
        detections.to_csv(predictions_dir / f"{plot_id}.csv", index=False)

    payload = {
        "study_name": args.study_name,
        "n_trials_total": int(len(study.trials)),
        "best_trial_number": int(best_trial.number),
        "best_value_f1_global": float(best_trial.value),
        "best_params": {
            "window_size_m": best_params.window_size_m,
            "height_threshold_m": best_params.height_threshold_m,
        },
        "best_summary": summary,
        "matching_params_fixed": {
            "max_dist_m": matching_params.max_dist_m,
            "max_height_diff_m": matching_params.max_height_diff_m,
        },
    }
    write_json(payload, output_dir / "best_summary.json")


if __name__ == "__main__":
    main()

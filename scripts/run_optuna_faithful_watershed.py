from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import (  # noqa: E402
    FaithfulMatchingParams,
    FaithfulPreprocessParams,
    FaithfulWatershedParams,
    chm_watershed_filter,
    faithful_greedy_match,
    preprocess_vendor_lidar_plot,
    summarize_faithful_matches,
)
from lidaragrosat.utils.io import ensure_dir, write_json  # noqa: E402


def _discover_plot_dirs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.iterdir() if path.is_dir())


def _save_preprocessed_cache(raw_dir: Path, cache_dir: Path, preprocess_params: FaithfulPreprocessParams) -> pd.DataFrame:
    records: list[dict] = []
    for plot_dir in _discover_plot_dirs(raw_dir):
        plot_id = plot_dir.name
        points_xyz, reference, stats = preprocess_vendor_lidar_plot(
            lidar_path=plot_dir / "lidar.las",
            reference_path=plot_dir / "trees_reference.csv",
            params=preprocess_params,
        )
        np.savez_compressed(cache_dir / f"{plot_id}.npz", points_xyz=points_xyz.astype(np.float32))
        reference.to_csv(cache_dir / f"{plot_id}_reference.csv", index=False)
        stats["plot_id"] = plot_id
        records.append(stats)
    summary = pd.DataFrame.from_records(records).sort_values("plot_id").reset_index(drop=True)
    summary.to_csv(cache_dir / "preprocessing_summary.csv", index=False)
    return summary


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
    watershed_params: FaithfulWatershedParams,
    matching_params: FaithfulMatchingParams,
    prediction_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    records: list[dict] = []
    for plot_id in sorted(points_by_plot):
        detections = chm_watershed_filter(points_by_plot[plot_id], watershed_params)
        if prediction_dir is not None:
            ensure_dir(prediction_dir)
            detections.to_csv(prediction_dir / f"{plot_id}.csv", index=False)
        match_result = faithful_greedy_match(
            predictions=detections,
            reference=reference_by_plot[plot_id],
            params=matching_params,
        )
        metrics = summarize_faithful_matches(match_result)
        metrics["plot_id"] = plot_id
        metrics["resolution_m"] = float(watershed_params.resolution_m)
        metrics["gaussian_sigma_m"] = float(watershed_params.gaussian_sigma_m)
        metrics["min_distance_m"] = float(watershed_params.min_distance_m)
        metrics["min_height_m"] = float(watershed_params.min_height_m)
        metrics["min_segment_area_m2"] = float(watershed_params.min_segment_area_m2)
        metrics["hole_fill_radius_m"] = float(watershed_params.hole_fill_radius_m)
        metrics["compactness"] = float(watershed_params.compactness)
        metrics["n_detections"] = int(len(detections))
        metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
        records.append(metrics)

    per_plot = pd.DataFrame.from_records(records).sort_values("plot_id").reset_index(drop=True)
    tp = int(per_plot["tp"].sum())
    fp = int(per_plot["fp"].sum())
    fn = int(per_plot["fn"].sum())
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    summary = {
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
        "precision_global": precision,
        "recall_global": recall,
        "f1_global": f1,
        "precision_macro": float(per_plot["precision"].mean()),
        "recall_macro": float(per_plot["recall"].mean()),
        "f1_macro": float(per_plot["f1"].mean()),
        "distance_macro": float(per_plot["distance_m"].mean()),
        "height_error_macro": float(per_plot["height_error_m"].mean()),
        "mean_detections": float(per_plot["n_detections"].mean()),
    }
    return per_plot, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna on a faithful CHM + watershed baseline.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--output-dir", default="outputs_faithful_paper_watershed_optuna")
    parser.add_argument("--compare-per-plot", default="outputs_faithful_paper_optuna/best_per_plot.csv")
    parser.add_argument("--study-name", default="faithful_watershed_tpe")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--duplicate-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--manual-min-abs-z", type=float, default=None)
    parser.add_argument("--manual-max-abs-z", type=float, default=None)
    parser.add_argument("--max-dist", type=float, default=5.0)
    parser.add_argument("--max-height-diff", type=float, default=3.0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    cache_dir = ensure_dir(Path(args.cache_dir).resolve())
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    preprocess_params = FaithfulPreprocessParams(
        duplicate_tolerance_m=float(args.duplicate_tolerance_mm) / 1000.0,
        manual_min_abs_z=args.manual_min_abs_z,
        manual_max_abs_z=args.manual_max_abs_z,
        ground_classification=2,
    )
    if not list(cache_dir.glob("plot_*.npz")):
        _save_preprocessed_cache(raw_dir, cache_dir, preprocess_params)

    points_by_plot, reference_by_plot = _load_cache(cache_dir)
    matching_params = FaithfulMatchingParams(
        max_dist_m=float(args.max_dist),
        max_height_diff_m=float(args.max_height_diff),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = FaithfulWatershedParams(
            resolution_m=trial.suggest_categorical("resolution_m", [0.1, 0.15, 0.2, 0.25]),
            gaussian_sigma_m=trial.suggest_float("gaussian_sigma_m", 0.2, 1.6, step=0.1),
            min_distance_m=trial.suggest_float("min_distance_m", 0.8, 3.0, step=0.1),
            min_height_m=trial.suggest_float("min_height_m", 2.5, 6.0, step=0.1),
            min_segment_area_m2=trial.suggest_float("min_segment_area_m2", 0.25, 6.0, step=0.25),
            hole_fill_radius_m=trial.suggest_float("hole_fill_radius_m", 0.1, 1.0, step=0.1),
            compactness=trial.suggest_float("compactness", 0.0, 0.01, step=0.0025),
        )
        _, summary = _evaluate_params(points_by_plot, reference_by_plot, params, matching_params)
        trial.set_user_attr("precision_global", summary["precision_global"])
        trial.set_user_attr("recall_global", summary["recall_global"])
        trial.set_user_attr("tp_total", summary["tp_total"])
        trial.set_user_attr("fp_total", summary["fp_total"])
        trial.set_user_attr("fn_total", summary["fn_total"])
        return float(summary["f1_global"])

    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=False)

    trial_rows: list[dict] = []
    for trial in study.trials:
        row = {
            "number": int(trial.number),
            "state": str(trial.state),
            "value": float(trial.value) if trial.value is not None else np.nan,
        }
        row.update(trial.params)
        row.update({f"user_{key}": value for key, value in trial.user_attrs.items()})
        trial_rows.append(row)
    pd.DataFrame.from_records(trial_rows).sort_values("number").to_csv(output_dir / "trials.csv", index=False)

    best_params = FaithfulWatershedParams(**study.best_trial.params)
    prediction_dir = ensure_dir(output_dir / "best_predictions")
    best_per_plot, best_summary = _evaluate_params(
        points_by_plot=points_by_plot,
        reference_by_plot=reference_by_plot,
        watershed_params=best_params,
        matching_params=matching_params,
        prediction_dir=prediction_dir,
    )
    best_per_plot.to_csv(output_dir / "best_per_plot.csv", index=False)
    write_json(
        {
            "best_trial_number": int(study.best_trial.number),
            "best_value_f1_global": float(study.best_value),
            "best_params": asdict(best_params),
            "summary": best_summary,
            "matching": asdict(matching_params),
        },
        output_dir / "best_summary.json",
    )

    compare_path = Path(args.compare_per_plot).resolve()
    if compare_path.exists():
        current = pd.read_csv(compare_path).rename(
            columns={
                "tp": "tp_current",
                "fp": "fp_current",
                "fn": "fn_current",
                "precision": "precision_current",
                "recall": "recall_current",
                "f1": "f1_current",
                "distance_m": "distance_m_current",
                "height_error_m": "height_error_m_current",
                "n_detections": "n_detections_current",
            }
        )
        merged = best_per_plot.merge(current, on="plot_id", how="left")
        for metric_name in ["tp", "fp", "fn", "precision", "recall", "f1", "distance_m", "height_error_m", "n_detections"]:
            merged[f"delta_{metric_name}"] = merged[metric_name] - merged[f"{metric_name}_current"]
        merged.to_csv(output_dir / "comparison_vs_current_local_maxima.csv", index=False)

    with open(output_dir / "study.sqlite3.path.txt", "w", encoding="utf-8") as fp:
        fp.write(str((output_dir / "study.sqlite3").resolve()))


if __name__ == "__main__":
    main()

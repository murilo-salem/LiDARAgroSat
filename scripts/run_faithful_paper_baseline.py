from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
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


def _save_preprocessed_cache(raw_dir: Path, cache_dir: Path, preprocess_params: FaithfulPreprocessParams) -> pd.DataFrame:
    records: list[dict] = []
    for plot_dir in _discover_plot_dirs(raw_dir):
        plot_id = plot_dir.name
        points_xyz, reference, stats = preprocess_vendor_lidar_plot(
            lidar_path=plot_dir / "lidar.las",
            reference_path=plot_dir / "trees_reference.csv",
            params=preprocess_params,
        )
        np.savez_compressed(
            cache_dir / f"{plot_id}.npz",
            points_xyz=points_xyz.astype(np.float32),
        )
        reference.to_csv(cache_dir / f"{plot_id}_reference.csv", index=False)
        stats["plot_id"] = plot_id
        stats["points_above_3m"] = int((points_xyz[:, 2] >= 3.0).sum())
        records.append(stats)
    summary = pd.DataFrame.from_records(records).sort_values("plot_id").reset_index(drop=True)
    summary.to_csv(cache_dir / "preprocessing_summary.csv", index=False)
    return summary


def _run_plot_job(
    cache_dir: str,
    plot_id: str,
    baseline_params: dict,
    matching_params: dict,
    save_prediction_dir: str | None = None,
) -> dict:
    cache_root = Path(cache_dir)
    points_xyz = np.load(cache_root / f"{plot_id}.npz")["points_xyz"].astype(np.float64)
    reference = pd.read_csv(cache_root / f"{plot_id}_reference.csv")

    detections = local_maxima_filter_3d(
        points_xyz=points_xyz,
        window_size_m=float(baseline_params["window_size_m"]),
        height_threshold_m=float(baseline_params["height_threshold_m"]),
    )
    if save_prediction_dir:
        prediction_root = ensure_dir(save_prediction_dir)
        detections.to_csv(prediction_root / f"{plot_id}.csv", index=False)

    match_result = faithful_greedy_match(
        predictions=detections,
        reference=reference,
        params=FaithfulMatchingParams(**matching_params),
    )
    metrics = summarize_faithful_matches(match_result)
    metrics["plot_id"] = plot_id
    metrics["window_size_m"] = float(baseline_params["window_size_m"])
    metrics["height_threshold_m"] = float(baseline_params["height_threshold_m"])
    metrics["n_detections"] = int(len(detections))
    metrics["n_reference"] = int(len(reference))
    return metrics


def _aggregate_metrics(metrics_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_summary = (
        metrics_table.groupby("window_size_m", as_index=False)[["tp", "fp", "fn"]]
        .sum()
        .sort_values("window_size_m")
        .reset_index(drop=True)
    )
    global_summary["precision"] = global_summary["tp"] / (global_summary["tp"] + global_summary["fp"])
    global_summary["precision"] = global_summary["precision"].fillna(0.0)
    global_summary["recall"] = global_summary["tp"] / (global_summary["tp"] + global_summary["fn"])
    global_summary["recall"] = global_summary["recall"].fillna(0.0)
    global_summary["f1"] = np.where(
        (global_summary["precision"] + global_summary["recall"]) > 0,
        2 * global_summary["precision"] * global_summary["recall"] / (global_summary["precision"] + global_summary["recall"]),
        0.0,
    )

    macro_summary = (
        metrics_table.groupby("window_size_m", as_index=False)[["precision", "recall", "f1", "distance_m", "height_error_m", "n_detections"]]
        .mean()
        .sort_values("window_size_m")
        .reset_index(drop=True)
    )
    return global_summary, macro_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the faithful paper baseline on the public Dubrovin release.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs_faithful_paper")
    parser.add_argument("--window-min", type=float, default=0.5)
    parser.add_argument("--window-max", type=float, default=5.0)
    parser.add_argument("--window-step", type=float, default=0.5)
    parser.add_argument("--canonical-window-size", type=float, default=2.0)
    parser.add_argument("--height-threshold", type=float, default=3.0)
    parser.add_argument("--max-dist", type=float, default=5.0)
    parser.add_argument("--max-height-diff", type=float, default=3.0)
    parser.add_argument("--duplicate-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--manual-min-abs-z", type=float, default=None)
    parser.add_argument("--manual-max-abs-z", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    cache_dir = ensure_dir(output_dir / "cache")
    predictions_dir = ensure_dir(output_dir / "predictions_canonical")

    preprocess_params = FaithfulPreprocessParams(
        duplicate_tolerance_m=float(args.duplicate_tolerance_mm) / 1000.0,
        manual_min_abs_z=args.manual_min_abs_z,
        manual_max_abs_z=args.manual_max_abs_z,
        ground_classification=2,
    )
    baseline_canonical = FaithfulBaselineParams(
        window_size_m=float(args.canonical_window_size),
        height_threshold_m=float(args.height_threshold),
    )
    matching_params = FaithfulMatchingParams(
        max_dist_m=float(args.max_dist),
        max_height_diff_m=float(args.max_height_diff),
    )

    window_values = np.arange(args.window_min, args.window_max + 0.5 * args.window_step, args.window_step)
    window_values = [round(float(value), 6) for value in window_values]
    config_payload = {
        "preprocess": asdict(preprocess_params),
        "canonical_baseline": asdict(baseline_canonical),
        "matching": asdict(matching_params),
        "window_sweep_m": window_values,
        "notes": {
            "manual_noise_filtering": "Public release already appears vendor-cleaned; manual thresholds default to disabled.",
            "rgb_usage": "The faithful baseline ignores RGB completely.",
        },
    }
    write_json(config_payload, output_dir / "config.json")

    preprocessing_summary = _save_preprocessed_cache(raw_dir, cache_dir, preprocess_params)

    plot_ids = preprocessing_summary["plot_id"].tolist()
    jobs: list[tuple[str, dict, str | None]] = []
    for window_size_m in window_values:
        params = {
            "window_size_m": float(window_size_m),
            "height_threshold_m": float(args.height_threshold),
        }
        for plot_id in plot_ids:
            save_dir = str(predictions_dir) if np.isclose(window_size_m, float(args.canonical_window_size)) else None
            jobs.append((plot_id, params, save_dir))

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
        futures = [
            executor.submit(
                _run_plot_job,
                str(cache_dir),
                plot_id,
                params,
                asdict(matching_params),
                save_dir,
            )
            for plot_id, params, save_dir in jobs
        ]
        for future in futures:
            results.append(future.result())

    metrics_table = pd.DataFrame.from_records(results).sort_values(["window_size_m", "plot_id"]).reset_index(drop=True)
    metrics_table.to_csv(output_dir / "window_sweep_per_plot.csv", index=False)

    global_summary, macro_summary = _aggregate_metrics(metrics_table)
    global_summary.to_csv(output_dir / "window_sweep_global.csv", index=False)
    macro_summary.to_csv(output_dir / "window_sweep_macro.csv", index=False)

    canonical_mask = np.isclose(metrics_table["window_size_m"], float(args.canonical_window_size))
    canonical_per_plot = metrics_table.loc[canonical_mask].sort_values("plot_id").reset_index(drop=True)
    canonical_per_plot.to_csv(output_dir / "canonical_per_plot.csv", index=False)

    canonical_global = global_summary.loc[np.isclose(global_summary["window_size_m"], float(args.canonical_window_size))].reset_index(drop=True)
    canonical_macro = macro_summary.loc[np.isclose(macro_summary["window_size_m"], float(args.canonical_window_size))].reset_index(drop=True)
    canonical_summary = {
        "canonical_global": canonical_global.iloc[0].to_dict() if not canonical_global.empty else {},
        "canonical_macro": canonical_macro.iloc[0].to_dict() if not canonical_macro.empty else {},
    }
    write_json(canonical_summary, output_dir / "canonical_summary.json")

    best_global = global_summary.iloc[int(global_summary["f1"].idxmax())].to_dict()
    best_macro = macro_summary.iloc[int(macro_summary["f1"].idxmax())].to_dict()
    write_json({"best_global": best_global, "best_macro": best_macro}, output_dir / "best_window_summary.json")


if __name__ == "__main__":
    main()

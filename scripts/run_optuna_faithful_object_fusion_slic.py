from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import FaithfulMatchingParams, faithful_greedy_match, summarize_faithful_matches  # noqa: E402
from lidaragrosat.paper_faithful.object_fusion import generate_source_candidates  # noqa: E402
from lidaragrosat.paper_faithful.object_fusion_slic import FaithfulObjectFusionSLICParams, fuse_lidar_rgb_slic_multiscale  # noqa: E402
from lidaragrosat.utils.io import ensure_dir, write_json  # noqa: E402


def _load_faithful_cache(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    points_by_plot: dict[str, np.ndarray] = {}
    reference_by_plot: dict[str, pd.DataFrame] = {}
    for npz_path in sorted(cache_dir.glob("plot_*.npz")):
        plot_id = npz_path.stem
        points_by_plot[plot_id] = np.load(npz_path)["points_xyz"].astype(np.float64)
        reference_by_plot[plot_id] = pd.read_csv(cache_dir / f"{plot_id}_reference.csv")
    return points_by_plot, reference_by_plot


def _load_processed_assets(processed_dir: Path) -> dict[str, dict]:
    assets: dict[str, dict] = {}
    for plot_dir in sorted(path for path in processed_dir.iterdir() if path.is_dir()):
        plot_id = plot_dir.name
        rgb = np.load(plot_dir / "rgb.npy").astype(np.float32)
        chm = np.load(plot_dir / "chm.npy").astype(np.float32)
        with rasterio.open(plot_dir / "rgb_resampled.tif") as src:
            transform = src.transform
        assets[plot_id] = {
            "rgb": rgb,
            "chm": chm,
            "x_min": float(transform.c),
            "y_max": float(transform.f),
            "pixel_size": float(transform.a),
        }
    return assets


def _build_source_cache(points_by_plot: dict[str, np.ndarray]) -> dict[str, dict[str, pd.DataFrame]]:
    source_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for plot_id, points_xyz in points_by_plot.items():
        source_cache[plot_id] = {
            "adaptive": generate_source_candidates(points_xyz, "adaptive"),
            "watershed": generate_source_candidates(points_xyz, "watershed"),
            "union": generate_source_candidates(points_xyz, "union"),
        }
    return source_cache


def _evaluate_params(
    source_cache: dict[str, dict[str, pd.DataFrame]],
    points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    processed_assets: dict[str, dict],
    params: FaithfulObjectFusionSLICParams,
    matching_params: FaithfulMatchingParams,
    prediction_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    for plot_id in sorted(reference_by_plot):
        assets = processed_assets[plot_id]
        detections = fuse_lidar_rgb_slic_multiscale(
            points_xyz=points_by_plot[plot_id],
            rgb=assets["rgb"],
            chm=assets["chm"],
            x_min=float(assets["x_min"]),
            y_max=float(assets["y_max"]),
            pixel_size=float(assets["pixel_size"]),
            params=params,
        )
        if prediction_dir is not None:
            ensure_dir(prediction_dir)
            detections.to_csv(prediction_dir / f"{plot_id}.csv", index=False)
        match_result = faithful_greedy_match(
            predictions=detections,
            reference=reference_by_plot[plot_id],
            params=matching_params,
        )
        metrics = summarize_faithful_matches(match_result)
        metrics.update(
            {
                "plot_id": plot_id,
                "n_detections": int(len(detections)),
                "n_reference": int(len(reference_by_plot[plot_id])),
                "source_mode": params.source_mode,
                "n_segments_low": int(params.n_segments_low),
                "compactness_low": float(params.compactness_low),
                "sigma_low": float(params.sigma_low),
                "n_segments_high": int(params.n_segments_high),
                "compactness_high": float(params.compactness_high),
                "sigma_high": float(params.sigma_high),
                "min_segment_area_m2": float(params.min_segment_area_m2),
                "max_segment_area_m2": float(params.max_segment_area_m2),
                "area_per_tree_m2": float(params.area_per_tree_m2),
                "min_boundary_distance_m": float(params.min_boundary_distance_m),
                "min_segment_chm_max_m": float(params.min_segment_chm_max_m),
                "candidate_merge_radius_m": float(params.candidate_merge_radius_m),
                "multiscale_merge_radius_m": float(params.multiscale_merge_radius_m),
            }
        )
        rows.append(metrics)
    per_plot = pd.DataFrame.from_records(rows).sort_values("plot_id").reset_index(drop=True)
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
    parser = argparse.ArgumentParser(description="Run Optuna on faithful multiscale SLIC LiDAR+RGB object fusion.")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--processed-dir", default="data/processed_preproc_explicit")
    parser.add_argument("--output-dir", default="outputs_faithful_paper_object_fusion_slic_optuna")
    parser.add_argument("--compare-felz", default="outputs_faithful_paper_object_fusion_optuna/best_per_plot.csv")
    parser.add_argument("--compare-watershed", default="outputs_faithful_paper_watershed_optuna/best_per_plot.csv")
    parser.add_argument("--study-name", default="faithful_object_fusion_slic_tpe")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-dist", type=float, default=5.0)
    parser.add_argument("--max-height-diff", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    points_by_plot, reference_by_plot = _load_faithful_cache(Path(args.cache_dir).resolve())
    processed_assets = _load_processed_assets(Path(args.processed_dir).resolve())
    source_cache = _build_source_cache(points_by_plot)
    matching_params = FaithfulMatchingParams(max_dist_m=float(args.max_dist), max_height_diff_m=float(args.max_height_diff))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        min_segment_area = trial.suggest_float("min_segment_area_m2", 0.5, 12.0, step=0.5)
        max_segment_area = trial.suggest_float("max_segment_area_m2", 40.0, 200.0, step=10.0)
        if max_segment_area <= min_segment_area:
            raise optuna.TrialPruned()
        n_low = trial.suggest_int("n_segments_low", 150, 500, step=50)
        n_high = trial.suggest_int("n_segments_high", max(n_low + 100, 400), 1200, step=100)
        params = FaithfulObjectFusionSLICParams(
            source_mode=trial.suggest_categorical("source_mode", ["adaptive", "watershed", "union"]),
            n_segments_low=int(n_low),
            compactness_low=trial.suggest_float("compactness_low", 3.0, 20.0, step=1.0),
            sigma_low=trial.suggest_float("sigma_low", 0.0, 2.0, step=0.1),
            n_segments_high=int(n_high),
            compactness_high=trial.suggest_float("compactness_high", 5.0, 30.0, step=1.0),
            sigma_high=trial.suggest_float("sigma_high", 0.0, 2.0, step=0.1),
            min_segment_area_m2=min_segment_area,
            max_segment_area_m2=max_segment_area,
            area_per_tree_m2=trial.suggest_float("area_per_tree_m2", 8.0, 35.0, step=1.0),
            min_boundary_distance_m=trial.suggest_float("min_boundary_distance_m", 0.0, 1.2, step=0.1),
            min_segment_chm_max_m=trial.suggest_float("min_segment_chm_max_m", 3.0, 8.0, step=0.5),
            candidate_merge_radius_m=trial.suggest_float("candidate_merge_radius_m", 0.4, 1.5, step=0.1),
            multiscale_merge_radius_m=trial.suggest_float("multiscale_merge_radius_m", 0.6, 1.8, step=0.1),
        )
        _, summary = _evaluate_params(source_cache, points_by_plot, reference_by_plot, processed_assets, params, matching_params)
        trial.set_user_attr("precision_global", summary["precision_global"])
        trial.set_user_attr("recall_global", summary["recall_global"])
        trial.set_user_attr("tp_total", summary["tp_total"])
        trial.set_user_attr("fp_total", summary["fp_total"])
        trial.set_user_attr("fn_total", summary["fn_total"])
        return float(summary["f1_global"])

    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=False)

    trial_rows = []
    for trial in study.trials:
        row = {"number": int(trial.number), "state": str(trial.state), "value": float(trial.value) if trial.value is not None else np.nan}
        row.update(trial.params)
        row.update({f"user_{k}": v for k, v in trial.user_attrs.items()})
        trial_rows.append(row)
    pd.DataFrame.from_records(trial_rows).sort_values("number").to_csv(output_dir / "trials.csv", index=False)

    best_params = FaithfulObjectFusionSLICParams(**study.best_trial.params)
    prediction_dir = ensure_dir(output_dir / "best_predictions")
    best_per_plot, best_summary = _evaluate_params(source_cache, points_by_plot, reference_by_plot, processed_assets, best_params, matching_params, prediction_dir=prediction_dir)
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

    comparison_rows = []
    for name, path in {"felzenszwalb": Path(args.compare_felz), "watershed": Path(args.compare_watershed)}.items():
        compare_path = path.resolve()
        if not compare_path.exists():
            continue
        baseline = pd.read_csv(compare_path).rename(
            columns={
                "tp": "tp_baseline",
                "fp": "fp_baseline",
                "fn": "fn_baseline",
                "precision": "precision_baseline",
                "recall": "recall_baseline",
                "f1": "f1_baseline",
            }
        )
        merged = best_per_plot.merge(baseline, on="plot_id", how="left")
        merged["delta_f1"] = merged["f1"] - merged["f1_baseline"]
        merged.to_csv(output_dir / f"comparison_vs_{name}.csv", index=False)
        comparison_rows.append(
            {
                "comparison_target": name,
                "f1_slic": float(best_summary["f1_global"]),
                "f1_baseline_mean": float(baseline["f1_baseline"].mean()),
                "mean_delta_f1_by_plot": float(merged["delta_f1"].mean()),
                "wins": int((merged["delta_f1"] > 0).sum()),
                "ties": int((merged["delta_f1"] == 0).sum()),
                "losses": int((merged["delta_f1"] < 0).sum()),
            }
        )
    if comparison_rows:
        pd.DataFrame.from_records(comparison_rows).to_csv(output_dir / "comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()

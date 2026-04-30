from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from pandas.errors import EmptyDataError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import aggregate_plot_metrics, build_method_summary, compare_methods, save_tables
from lidaragrosat.analysis.scientific import build_gain_table, classify_error_modes, compute_plot_descriptors, fit_gain_vs_complexity, save_regression_table
from lidaragrosat.benchmarking import require_protocol_id
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.experiment_manifest import load_experiment_manifest
from lidaragrosat.utils.io import ensure_dir
from lidaragrosat.utils.logging import configure_logging, get_logger


EMPTY_PREDICTION_COLUMNS = ["x", "y", "height_pred", "confidence"]


def _discover_methods(prediction_root: Path, method_names: list[str] | None = None) -> list[str]:
    if method_names:
        return method_names
    return [path.name for path in sorted(prediction_root.iterdir()) if path.is_dir()]


def _prediction_files_for_method(method_dir: Path) -> list[tuple[int, Path]]:
    pairs = []
    for fold_dir in sorted(path for path in method_dir.glob("fold_*") if path.is_dir()):
        fold_id = int(fold_dir.name.split("_")[-1])
        for csv_path in sorted(fold_dir.glob("*.csv")):
            if csv_path.name == "history.csv":
                continue
            pairs.append((fold_id, csv_path))
    return pairs


def _load_method_manifest(method_dir: Path):
    manifest_path = method_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run {method_dir} has no experiment_manifest.json. "
            "benchmark_protocol_v2 requires protocol_id and manifesto em toda run oficial."
        )
    manifest = load_experiment_manifest(manifest_path)
    require_protocol_id(manifest.protocol_id)
    return manifest


def _load_prediction_table(prediction_path: Path) -> pd.DataFrame:
    try:
        predictions = pd.read_csv(prediction_path)
    except EmptyDataError:
        return pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)
    if predictions.empty and len(predictions.columns) == 0:
        return pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)
    return predictions


def _save_manifest_table(manifest_rows: list[dict], output_dir: Path) -> None:
    if not manifest_rows:
        return
    manifest_table = pd.DataFrame.from_records(manifest_rows)
    if "experiment_name" in manifest_table.columns:
        manifest_table = manifest_table.drop_duplicates(subset=["experiment_name"]).reset_index(drop=True)
    manifest_table.to_csv(output_dir / "experiment_manifests.csv", index=False)


def _resolve_method_name(metrics_table: pd.DataFrame, requested_name: str) -> str:
    cleaned = str(requested_name or "").strip()
    if not cleaned or "method" not in metrics_table.columns:
        return cleaned

    method_names = set(metrics_table["method"].dropna().astype(str).tolist())
    if cleaned in method_names:
        return cleaned

    if "stage2" in metrics_table.columns:
        candidates = sorted(
            set(metrics_table.loc[metrics_table["stage2"] == cleaned, "method"].dropna().astype(str).tolist())
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Method alias '{cleaned}' matches multiple experiment names: {', '.join(candidates)}"
            )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions under a unified matching protocol.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--reference-method", default="baseline_local_maxima")
    parser.add_argument("--fusion-method", default="multimodal_mid_fusion")
    parser.add_argument("--baseline-method", default="lidar_only")
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    configure_logging(Path(config.paths.output_dir) / "logs", "evaluate_experiments")
    logger = get_logger("evaluate_experiments")

    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    prediction_root = config.resolve_path(config.paths.output_dir) / "predictions"
    method_names = _discover_methods(prediction_root, args.methods)
    output_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "metrics")

    metric_rows: list[dict] = []
    error_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    for method_name in method_names:
        logger.info("Evaluating method %s", method_name)
        method_dir = prediction_root / method_name
        manifest = _load_method_manifest(method_dir)
        manifest_rows.append(manifest.to_dict())
        for fold_id, prediction_path in _prediction_files_for_method(method_dir):
            plot_id = prediction_path.stem
            if plot_id not in bundles:
                continue
            predictions = _load_prediction_table(prediction_path)
            match_result = match_tree_predictions(
                predictions,
                bundles[plot_id].reference,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            row = aggregate_plot_metrics(
                method_name,
                plot_id,
                match_result,
                extra_fields={
                    "fold_id": fold_id,
                    "protocol_id": manifest.protocol_id,
                    "family": manifest.family,
                    "stage1": manifest.stage1,
                    "stage2": manifest.stage2,
                    "labels": manifest.labels,
                    "seed": manifest.seed,
                },
            )
            metric_rows.append(row)
            error_rows.append(
                classify_error_modes(
                    bundles[plot_id],
                    predictions=predictions,
                    match_result=match_result,
                    overlap_radius_m=config.analysis.overlap_radius_m,
                )
            )

    metrics_table = pd.DataFrame(metric_rows).sort_values(["method", "plot_id"]).reset_index(drop=True)
    summary_table = build_method_summary(metrics_table, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    reference_method = _resolve_method_name(metrics_table, args.reference_method)
    statistics_table = compare_methods(metrics_table, reference_method=reference_method)
    save_tables(metrics_table, summary_table, statistics_table, output_dir)
    _save_manifest_table(manifest_rows, output_dir)

    descriptor_rows = [compute_plot_descriptors(bundle, config.data.reference_species_groups) for bundle in bundles.values()]
    descriptor_table = pd.DataFrame(descriptor_rows)
    descriptor_table.to_csv(output_dir / "plot_descriptors.csv", index=False)

    fusion_method = _resolve_method_name(metrics_table, args.fusion_method)
    baseline_method = _resolve_method_name(metrics_table, args.baseline_method)
    if fusion_method in metrics_table["method"].unique() and baseline_method in metrics_table["method"].unique():
        gain_table = build_gain_table(metrics_table, fusion_method=fusion_method, baseline_method=baseline_method)
        gain_table["fusion_method"] = args.fusion_method
        gain_table["baseline_method"] = args.baseline_method
        gain_table.to_csv(output_dir / "fusion_gain_vs_baseline.csv", index=False)
        regression_rows = [
            fit_gain_vs_complexity(gain_table, descriptor_table, predictor)
            for predictor in ["canopy_height_variance", "vertical_roughness_mean", "tree_density_ha", "rgb_entropy", "conifer_fraction"]
        ]
        save_regression_table(regression_rows, output_dir / "fusion_gain_regression.csv")

    if error_rows:
        pd.concat(error_rows, ignore_index=True).to_csv(output_dir / "error_modes.csv", index=False)
    logger.info("Evaluation complete for %d methods.", len(method_names))


if __name__ == "__main__":
    main()

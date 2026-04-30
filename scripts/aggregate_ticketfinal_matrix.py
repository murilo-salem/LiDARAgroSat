from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_A_ID
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import load_experiment_manifest
from lidaragrosat.utils.io import ensure_dir


def _collect_stage2_metrics(
    output_root: Path,
    bundles,
    match_radius_m: float,
    height_tolerance_m: float,
    allowed_family: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_root = output_root / "predictions"
    metric_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for experiment_dir in sorted(prediction_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        manifest_path = experiment_dir / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_experiment_manifest(manifest_path)
        if manifest.protocol_id != PROTOCOL_A_ID:
            continue
        if allowed_family and manifest.family != allowed_family:
            continue
        if manifest.stage2 == "none":
            continue
        manifest_rows.append(manifest.to_dict())
        for fold_dir in sorted(experiment_dir.glob("fold_*")):
            try:
                fold_id = int(fold_dir.name.split("_")[-1])
            except ValueError:
                fold_id = -1
            for plot_path in sorted(fold_dir.glob("plot_*.csv")):
                plot_id = plot_path.stem
                predictions = pd.read_csv(plot_path)
                matches = match_tree_predictions(
                    predictions=predictions,
                    reference=bundles[plot_id].reference,
                    match_radius_m=match_radius_m,
                    height_tolerance_m=height_tolerance_m,
                )
                row = summarize_matches(matches)
                row.update(
                    {
                        "method": manifest.experiment_name,
                        "plot_id": plot_id,
                        "fold_id": fold_id,
                        "protocol_id": manifest.protocol_id,
                        "family": manifest.family,
                        "stage1": manifest.stage1,
                        "stage2": manifest.stage2,
                        "labels": manifest.labels,
                        "seed": manifest.seed,
                    }
                )
                metric_rows.append(row)
    metrics = pd.DataFrame.from_records(metric_rows)
    manifests = pd.DataFrame.from_records(manifest_rows)
    return metrics, manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TicketFinal protocol-A matrix from manifests and predictions.")
    parser.add_argument("--config", nargs="+", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    output_root = config.resolve_path(config.paths.output_dir)
    metrics_dir = ensure_dir(output_root / "metrics")
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)

    stage1_only_path = metrics_dir / "stage1_only_metrics_by_plot.csv"
    stage1_only = pd.read_csv(stage1_only_path) if stage1_only_path.exists() else pd.DataFrame()
    stage2_metrics, manifests = _collect_stage2_metrics(
        output_root=output_root,
        bundles=bundles,
        match_radius_m=config.evaluation.match_radius_m,
        height_tolerance_m=config.evaluation.height_tolerance_m,
        allowed_family=config.benchmark.family or None,
    )

    all_metrics = pd.concat([stage1_only, stage2_metrics], ignore_index=True, sort=False)
    all_metrics.to_csv(metrics_dir / "main_matrix_metrics_by_plot.csv", index=False)
    if len(all_metrics):
        summary = build_method_summary(all_metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    else:
        summary = pd.DataFrame()
    summary.to_csv(metrics_dir / "main_matrix_results.csv", index=False)
    manifests.to_csv(metrics_dir / "main_matrix_manifests.csv", index=False)

    if len(all_metrics):
        label_ablation = (
            all_metrics[
                (all_metrics["protocol_id"] == PROTOCOL_A_ID)
                & (all_metrics["stage1"] == "watershed_fixo")
                & (all_metrics["stage2"].isin(["raster", "point", "hybrid"]))
            ]
            .groupby(["stage2", "labels"])[["precision", "recall", "f1", "location_mae_m", "height_mae_m"]]
            .mean()
            .reset_index()
        )
        label_ablation.to_csv(metrics_dir / "label_policy_ablation.csv", index=False)

        stage1_vs_stage2 = (
            all_metrics.groupby(["stage1", "stage2"])[["precision", "recall", "f1", "location_mae_m", "height_mae_m", "tp", "fp", "fn"]]
            .mean()
            .reset_index()
            .sort_values(["stage1", "f1"], ascending=[True, False])
        )
        stage1_vs_stage2.to_csv(metrics_dir / "stage1_vs_stage2.csv", index=False)

        expected = {(stage1, stage2, labels) for stage1 in ["watershed_fixo", "adaptive_watershed"] for stage2 in ["raster", "point", "hybrid"] for labels in ["field_center", "apex_snapped", "mixed_center_apex"]}
        observed = set(
            all_metrics[all_metrics["stage2"] != "none"][["stage1", "stage2", "labels"]].drop_duplicates().itertuples(index=False, name=None)
        )
        missing = sorted(expected - observed)
        completeness = pd.DataFrame.from_records(
            [{"stage1": a, "stage2": b, "labels": c, "present": False} for a, b, c in missing]
        )
        completeness.to_csv(metrics_dir / "main_matrix_missing_cells.csv", index=False)


if __name__ == "__main__":
    main()

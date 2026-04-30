from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import FaithfulMatchingParams  # noqa: E402
from lidaragrosat.paper_faithful.hybrid_watershed_proposals import (  # noqa: E402
    FaithfulHybridWatershedProposalParams,
    evaluate_hybrid_params,
    grid_search_hybrid_params,
)
from lidaragrosat.utils.io import ensure_dir, write_json, write_text  # noqa: E402


VAL_PLOTS = ["plot_01", "plot_06"]


def _load_cache(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    points_by_plot: dict[str, np.ndarray] = {}
    reference_by_plot: dict[str, pd.DataFrame] = {}
    for npz_path in sorted(cache_dir.glob("plot_*.npz")):
        plot_id = npz_path.stem
        points_by_plot[plot_id] = np.load(npz_path)["points_xyz"].astype(np.float64)
        reference_by_plot[plot_id] = pd.read_csv(cache_dir / f"{plot_id}_reference.csv")
    return points_by_plot, reference_by_plot


def _compare_scope(plot_ids: list[str], output_path: Path) -> pd.DataFrame:
    sources = {
        "local_maxima_fixed": Path("outputs_faithful_paper_optuna/best_per_plot.csv"),
        "local_maxima_adaptive": Path("outputs_faithful_paper_adaptive_optuna/best_per_plot.csv"),
        "chm_watershed": Path("outputs_faithful_paper_watershed_optuna/best_per_plot.csv"),
        "rgb_lidar_object_fusion": Path("outputs_faithful_paper_object_fusion_optuna/best_per_plot.csv"),
    }
    rows: list[dict] = []
    for method_name, path in sources.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        scoped = table[table["plot_id"].isin(plot_ids)].copy()
        rows.append(
            {
                "method": method_name,
                "plot_scope": ",".join(plot_ids),
                "tp": int(scoped["tp"].sum()),
                "fp": int(scoped["fp"].sum()),
                "fn": int(scoped["fn"].sum()),
                "precision_mean": float(scoped["precision"].mean()),
                "recall_mean": float(scoped["recall"].mean()),
                "f1_mean": float(scoped["f1"].mean()),
                "distance_m_mean": float(scoped["distance_m"].mean()),
                "height_error_m_mean": float(scoped["height_error_m"].mean()),
            }
        )
    comparison = pd.DataFrame.from_records(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def _aggregate(metrics_by_plot: pd.DataFrame, partition_name: str) -> dict[str, float | str | int]:
    return {
        "partition": partition_name,
        "n_plots": int(len(metrics_by_plot)),
        "tp": int(metrics_by_plot["tp"].sum()),
        "fp": int(metrics_by_plot["fp"].sum()),
        "fn": int(metrics_by_plot["fn"].sum()),
        "precision_mean": float(metrics_by_plot["precision"].mean()),
        "recall_mean": float(metrics_by_plot["recall"].mean()),
        "f1_mean": float(metrics_by_plot["f1"].mean()),
        "distance_m_mean": float(metrics_by_plot["distance_m"].mean()),
        "height_error_m_mean": float(metrics_by_plot["height_error_m"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-tuned hybrid watershed + proposal filtering.")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--output-dir", default="outputs_faithful_hybrid_watershed_proposals")
    parser.add_argument("--max-dist", type=float, default=5.0)
    parser.add_argument("--max-height-diff", type=float, default=3.0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    points_by_plot, reference_by_plot = _load_cache(cache_dir)
    ordered_plots = sorted(points_by_plot)
    val_plots = [plot_id for plot_id in VAL_PLOTS if plot_id in ordered_plots]

    matching_params = FaithfulMatchingParams(max_dist_m=float(args.max_dist), max_height_diff_m=float(args.max_height_diff))
    grid_scores = grid_search_hybrid_params(
        points_by_plot=points_by_plot,
        reference_by_plot=reference_by_plot,
        matching_params=matching_params,
        plot_ids=val_plots,
        resolution_values=[0.20, 0.25, 0.33, 0.50],
        smooth_sigma_values=[0.5, 1.0, 1.5, 2.0],
        min_distance_values=[3, 4, 5, 6],
        min_segment_area_values=[2.0, 4.0, 6.0],
        proposal_window_values=[0.5, 0.8, 1.0],
    )
    grid_scores.to_csv(output_dir / "validation_grid_scores.csv", index=False)

    best_row = grid_scores.iloc[0]
    best_params = FaithfulHybridWatershedProposalParams(
        resolution_m=float(best_row["resolution_m"]),
        smooth_sigma_px=float(best_row["smooth_sigma_px"]),
        min_distance_px=int(best_row["min_distance_px"]),
        min_segment_area_m2=float(best_row["min_segment_area_m2"]),
        proposal_window_m=float(best_row["proposal_window_m"]),
    )
    write_json({"best_params": asdict(best_params), "matching": asdict(matching_params)}, output_dir / "best_validation_params.json")

    val_predictions_dir = ensure_dir(output_dir / "predictions_val")
    val_metrics_by_plot, val_diagnostics, val_summary = evaluate_hybrid_params(
        points_by_plot=points_by_plot,
        reference_by_plot=reference_by_plot,
        params=best_params,
        matching_params=matching_params,
        plot_ids=val_plots,
        prediction_dir=str(val_predictions_dir),
    )
    val_metrics_by_plot.to_csv(output_dir / "validation_metrics_by_plot.csv", index=False)
    val_diagnostics.to_csv(output_dir / "validation_segment_diagnostics.csv", index=False)
    pd.DataFrame.from_records([val_summary]).to_csv(output_dir / "validation_summary.csv", index=False)

    all_predictions_dir = ensure_dir(output_dir / "predictions_all")
    all_metrics_by_plot, all_diagnostics, all_summary = evaluate_hybrid_params(
        points_by_plot=points_by_plot,
        reference_by_plot=reference_by_plot,
        params=best_params,
        matching_params=matching_params,
        plot_ids=ordered_plots,
        prediction_dir=str(all_predictions_dir),
    )
    all_metrics_by_plot.to_csv(output_dir / "metrics_by_plot.csv", index=False)
    all_diagnostics.to_csv(output_dir / "segment_diagnostics_by_plot.csv", index=False)
    pd.DataFrame.from_records([all_summary]).to_csv(output_dir / "summary_all.csv", index=False)

    summary_by_partition = pd.DataFrame.from_records(
        [
            _aggregate(all_metrics_by_plot[~all_metrics_by_plot["plot_id"].isin(val_plots)], "non_val"),
            _aggregate(all_metrics_by_plot[all_metrics_by_plot["plot_id"].isin(val_plots)], "val"),
            _aggregate(all_metrics_by_plot, "all"),
        ]
    )
    summary_by_partition.to_csv(output_dir / "summary_by_partition.csv", index=False)

    compare_val = _compare_scope(val_plots, output_dir / "comparison_val_scope.csv")
    compare_all = _compare_scope(ordered_plots, output_dir / "comparison_all_scope.csv")

    current_watershed_val = compare_val[compare_val["method"] == "chm_watershed"].iloc[0].to_dict()
    current_watershed_all = compare_all[compare_all["method"] == "chm_watershed"].iloc[0].to_dict()
    hybrid_val_row = summary_by_partition[summary_by_partition["partition"] == "val"].iloc[0].to_dict()
    hybrid_all_row = summary_by_partition[summary_by_partition["partition"] == "all"].iloc[0].to_dict()

    comparison_table = pd.DataFrame.from_records(
        [
            {
                "scope": "val",
                "hybrid_precision_mean": float(hybrid_val_row["precision_mean"]),
                "hybrid_recall_mean": float(hybrid_val_row["recall_mean"]),
                "hybrid_f1_mean": float(hybrid_val_row["f1_mean"]),
                "watershed_precision_mean": float(current_watershed_val["precision_mean"]),
                "watershed_recall_mean": float(current_watershed_val["recall_mean"]),
                "watershed_f1_mean": float(current_watershed_val["f1_mean"]),
                "delta_f1_vs_watershed": float(hybrid_val_row["f1_mean"] - current_watershed_val["f1_mean"]),
            },
            {
                "scope": "all",
                "hybrid_precision_mean": float(hybrid_all_row["precision_mean"]),
                "hybrid_recall_mean": float(hybrid_all_row["recall_mean"]),
                "hybrid_f1_mean": float(hybrid_all_row["f1_mean"]),
                "watershed_precision_mean": float(current_watershed_all["precision_mean"]),
                "watershed_recall_mean": float(current_watershed_all["recall_mean"]),
                "watershed_f1_mean": float(current_watershed_all["f1_mean"]),
                "delta_f1_vs_watershed": float(hybrid_all_row["f1_mean"] - current_watershed_all["f1_mean"]),
            },
        ]
    )
    comparison_table.to_csv(output_dir / "hybrid_vs_watershed_summary.csv", index=False)

    per_plot_compare = all_metrics_by_plot.merge(
        pd.read_csv("outputs_faithful_paper_watershed_optuna/best_per_plot.csv")[
            ["plot_id", "precision", "recall", "f1", "distance_m", "height_error_m"]
        ].rename(
            columns={
                "precision": "watershed_precision",
                "recall": "watershed_recall",
                "f1": "watershed_f1",
                "distance_m": "watershed_distance_m",
                "height_error_m": "watershed_height_error_m",
            }
        ),
        on="plot_id",
        how="left",
    )
    per_plot_compare["delta_f1_vs_watershed"] = per_plot_compare["f1"] - per_plot_compare["watershed_f1"]
    per_plot_compare.to_csv(output_dir / "comparison_vs_watershed_by_plot.csv", index=False)

    report_lines = [
        "# Faithful Hybrid Watershed + Proposals",
        "",
        "## Best Validation Parameters",
        json.dumps(asdict(best_params), indent=2),
        "",
        "## Validation Summary",
        f"- precision_mean = {hybrid_val_row['precision_mean']:.4f}",
        f"- recall_mean = {hybrid_val_row['recall_mean']:.4f}",
        f"- f1_mean = {hybrid_val_row['f1_mean']:.4f}",
        f"- delta vs watershed (val) = {hybrid_val_row['f1_mean'] - current_watershed_val['f1_mean']:+.4f}",
        "",
        "## All-Plot Summary",
        f"- precision_mean = {hybrid_all_row['precision_mean']:.4f}",
        f"- recall_mean = {hybrid_all_row['recall_mean']:.4f}",
        f"- f1_mean = {hybrid_all_row['f1_mean']:.4f}",
        f"- delta vs watershed (all) = {hybrid_all_row['f1_mean'] - current_watershed_all['f1_mean']:+.4f}",
    ]
    write_text("\n".join(report_lines) + "\n", output_dir / "REPORT.md")


if __name__ == "__main__":
    main()

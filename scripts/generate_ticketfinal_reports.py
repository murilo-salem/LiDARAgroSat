from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.error_modes import ErrorModeConfig, summarize_error_modes
from lidaragrosat.benchmarking import PROTOCOL_A_ID
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.stage1.generators import (
    AdaptiveWatershedStage1Generator,
    LocalMaximaStage1Generator,
    WatershedStage1Generator,
)
from lidaragrosat.utils.io import ensure_dir, write_text


def _simple_table(table: pd.DataFrame, max_rows: int = 10) -> str:
    if table.empty:
        return "No rows."
    return table.head(max_rows).to_csv(index=False).strip()


def _load_previous_protocol_a() -> pd.DataFrame:
    candidates = [
        Path("outputs_suite_aggregate/suite_method_summary.csv"),
        Path("outputs_dev_validation_preproc_explicit/validation_summary.csv"),
    ]
    rows: list[dict] = []
    for path in candidates:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "method" in table.columns and "f1_mean" in table.columns:
            rows.extend(table[["method", "f1_mean"]].assign(source=path.as_posix()).to_dict(orient="records"))
        elif "method" in table.columns and "F1 mean ± std" not in table.columns:
            continue
    return pd.DataFrame.from_records(rows)


def _load_previous_protocol_b() -> pd.DataFrame:
    rows: list[dict] = []
    sources = {
        "baseline_local_maxima_optuna": Path("outputs_faithful_paper_optuna/best_summary.json"),
        "adaptive_local_maxima": Path("outputs_faithful_paper_adaptive_optuna/best_summary.json"),
        "chm_watershed": Path("outputs_faithful_paper_watershed_optuna/best_summary.json"),
        "rgb_lidar_object_fusion": Path("outputs_faithful_paper_object_fusion_optuna/best_summary.json"),
    }
    for method, path in sources.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"method": method, "f1_mean": float(payload["summary"]["f1_global"]) if "summary" in payload else float(payload["best_value_f1_global"])})
    return pd.DataFrame.from_records(rows)


def _load_prediction_sets(output_dir: Path, methods: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    predictions_root = output_dir / "predictions"
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for method in methods:
        method_dir = predictions_root / method
        if not method_dir.exists():
            continue
        plot_tables: dict[str, pd.DataFrame] = {}
        for fold_dir in sorted(method_dir.glob("fold_*")):
            for plot_path in sorted(fold_dir.glob("plot_*.csv")):
                plot_tables[plot_path.stem] = pd.read_csv(plot_path)
        if plot_tables:
            loaded[method] = plot_tables
    return loaded


def _stage1_cache_for_report(processed_dir: Path):
    bundles = list_processed_plots(processed_dir)
    generators = {
        "local_maxima": LocalMaximaStage1Generator(),
        "watershed_fixo": WatershedStage1Generator(),
        "adaptive_watershed": AdaptiveWatershedStage1Generator(),
    }
    cache = {
        stage1_name: {plot_id: generator.generate(bundle) for plot_id, bundle in bundles.items()}
        for stage1_name, generator in generators.items()
    }
    return bundles, cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TicketFinal comparison reports and paper tables.")
    parser.add_argument("--output-dir", default="outputs_ticketfinal_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    metrics_dir = output_dir / "metrics"
    report_dir = ensure_dir(output_dir / "reports")
    paper_tables = ensure_dir(output_dir / "paper_tables")
    paper_figures = ensure_dir(output_dir / "paper_figures")

    main_matrix = pd.read_csv(metrics_dir / "main_matrix_results.csv")
    main_metrics = pd.read_csv(metrics_dir / "main_matrix_metrics_by_plot.csv")
    label_ablation = pd.read_csv(metrics_dir / "label_policy_ablation.csv")
    stage1_vs_stage2 = pd.read_csv(metrics_dir / "stage1_vs_stage2.csv")
    rgb_ablation = pd.read_csv(metrics_dir / "rgb_contribution_ablation.csv") if (metrics_dir / "rgb_contribution_ablation.csv").exists() else pd.DataFrame()
    chm_ablation = pd.read_csv(metrics_dir / "chm_contribution_ablation.csv") if (metrics_dir / "chm_contribution_ablation.csv").exists() else pd.DataFrame()
    if (metrics_dir / "stage1_only_summary.csv").exists():
        stage1_only_summary = pd.read_csv(metrics_dir / "stage1_only_summary.csv")
    else:
        stage1_only_summary = pd.DataFrame()
    if (metrics_dir / "distance_transform_watershed_summary.csv").exists():
        dt_summary = pd.read_csv(metrics_dir / "distance_transform_watershed_summary.csv")
    else:
        dt_summary = pd.DataFrame()

    gain_rows = []
    for stage1_name, group in stage1_vs_stage2.groupby("stage1"):
        stage1_only = group[group["stage2"] == "none"]
        if stage1_only.empty:
            continue
        base_f1 = float(stage1_only["f1"].iloc[0])
        for _, row in group[group["stage2"] != "none"].iterrows():
            gain_rows.append(
                {
                    "stage1": stage1_name,
                    "stage2": row["stage2"],
                    "gain_over_stage1_f1": float(row["f1"] - base_f1),
                }
            )
    gain_table = pd.DataFrame.from_records(gain_rows).sort_values("gain_over_stage1_f1", ascending=False)
    gain_table.to_csv(report_dir / "gain_by_stage.csv", index=False)

    if not label_ablation.empty:
        label_gain_rows = []
        for stage2_name, group in label_ablation.groupby("stage2"):
            base = group[group["labels"] == "field_center"]
            if base.empty:
                continue
            base_f1 = float(base["f1"].iloc[0])
            base_loc = float(base["location_mae_m"].iloc[0])
            for _, row in group.iterrows():
                label_gain_rows.append(
                    {
                        "stage2": stage2_name,
                        "labels": row["labels"],
                        "delta_f1_vs_field_center": float(row["f1"] - base_f1),
                        "delta_location_mae_vs_field_center": float(row["location_mae_m"] - base_loc),
                    }
                )
        pd.DataFrame.from_records(label_gain_rows).to_csv(report_dir / "gain_by_label.csv", index=False)

    plot_type_rows = []
    for plot_id, group in main_metrics.groupby("plot_id"):
        stage1_only = group[group["stage2"] == "none"]
        stage2_group = group[group["stage2"] != "none"]
        if stage1_only.empty or stage2_group.empty:
            continue
        best_stage2 = stage2_group.sort_values("f1", ascending=False).iloc[0]
        base = stage1_only.sort_values("f1", ascending=False).iloc[0]
        plot_type_rows.append(
            {
                "plot_id": plot_id,
                "best_stage2_method": best_stage2["method"],
                "best_stage2_f1": float(best_stage2["f1"]),
                "best_stage1_f1": float(base["f1"]),
                "gain_f1": float(best_stage2["f1"] - base["f1"]),
            }
        )
    gain_by_plot = pd.DataFrame.from_records(plot_type_rows)
    if not gain_by_plot.empty:
        gain_by_plot.to_csv(report_dir / "gain_by_plot.csv", index=False)

    previous_a = _load_previous_protocol_a()
    if not previous_a.empty:
        best_current_a = main_matrix.iloc[0][["method", "f1_mean"]].to_dict()
        previous_a["delta_vs_current_best"] = previous_a["f1_mean"] - float(best_current_a["f1_mean"])
        previous_a.to_csv(report_dir / "comparison_vs_previous_protocol_a.csv", index=False)

    previous_b = _load_previous_protocol_b()
    if not previous_b.empty:
        previous_b.to_csv(report_dir / "comparison_vs_previous_protocol_b.csv", index=False)

    main_matrix.to_csv(paper_tables / "main_matrix_results.csv", index=False)
    label_ablation.to_csv(paper_tables / "label_policy_ablation.csv", index=False)
    stage1_vs_stage2.to_csv(paper_tables / "stage1_vs_stage2.csv", index=False)
    if not rgb_ablation.empty:
        rgb_ablation.to_csv(paper_tables / "rgb_contribution_ablation.csv", index=False)
    if not chm_ablation.empty:
        chm_ablation.to_csv(paper_tables / "chm_contribution_ablation.csv", index=False)
    if not stage1_only_summary.empty:
        stage1_only_summary.to_csv(paper_tables / "stage1_only_summary.csv", index=False)
    if not dt_summary.empty:
        dt_summary.to_csv(paper_tables / "distance_transform_watershed_summary.csv", index=False)
    if not stage1_only_summary.empty:
        stage1_vs_stage2_full = pd.concat(
            [
                stage1_only_summary.rename(
                    columns={
                        "precision_mean": "precision",
                        "recall_mean": "recall",
                        "f1_mean": "f1",
                        "location_mae_mean_m": "location_mae_m",
                        "height_mae_mean_m": "height_mae_m",
                        "tp_total": "tp",
                        "fp_total": "fp",
                        "fn_total": "fn",
                    }
                )[
                    [
                        "stage1",
                        "stage2",
                        "precision",
                        "recall",
                        "f1",
                        "location_mae_m",
                        "height_mae_m",
                        "tp",
                        "fp",
                        "fn",
                    ]
                ],
                stage1_vs_stage2,
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["stage1", "stage2"], keep="first")
        stage1_vs_stage2_full.to_csv(paper_tables / "stage1_vs_stage2_full.csv", index=False)

    # Stage-2 error modes.
    if (output_dir / "predictions").exists():
        bundles, stage1_cache = _stage1_cache_for_report(output_dir.parents[0] / "data" / "processed_preproc_explicit")
        stage2_methods = []
        if {"method", "stage2", "stage1"}.issubset(main_metrics.columns):
            stage2_methods = (
                main_metrics[main_metrics["stage2"] != "none"][["method", "stage1"]]
                .drop_duplicates()
                .to_dict(orient="records")
            )
        if stage2_methods:
            prediction_sets = _load_prediction_sets(output_dir, [row["method"] for row in stage2_methods])
            stage1_by_method = {
                row["method"]: stage1_cache.get(str(row["stage1"]), {}) for row in stage2_methods if row["method"] in prediction_sets
            }
            error_modes = summarize_error_modes(
                methods=prediction_sets,
                bundles=bundles,
                stage1_outputs_by_method=stage1_by_method,
                config=ErrorModeConfig(),
            )
            error_modes.to_csv(metrics_dir / "error_modes_stage2.csv", index=False)
            error_modes.to_csv(report_dir / "error_modes_stage2.csv", index=False)

    plt.figure(figsize=(10, 4))
    ordered = main_matrix.sort_values("f1_mean", ascending=False)
    plt.bar(range(len(ordered)), ordered["f1_mean"])
    plt.xticks(range(len(ordered)), ordered["method"], rotation=90)
    plt.ylabel("F1 mean")
    plt.tight_layout()
    plt.savefig(paper_figures / "main_matrix_barplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    for stage2_name, group in main_metrics.groupby("stage2"):
        if stage2_name == "none":
            continue
        plt.scatter(group["location_mae_m"], group["f1"], label=stage2_name, alpha=0.7)
    plt.xlabel("Location MAE (m)")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paper_figures / "f1_vs_location_mae.png", dpi=180)
    plt.close()

    diagnostics_path = Path("outputs_ticketfinal_stage1/metrics/watershed_diagnostics.csv")
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        group_column = "method" if "method" in diagnostics.columns else "generator"
        aggregated = diagnostics.groupby(group_column)[["under_segmented", "over_segmented", "mixed_segment", "empty_segment"]].sum()
        aggregated.plot(kind="bar", figsize=(8, 4))
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(paper_figures / "under_over_segmentation.png", dpi=180)
        plt.close()

    lines = [
        "# TicketFinal Report",
        "",
        "## Best Current Result",
        _simple_table(main_matrix.head(1)),
        "",
        "## Gain By Stage",
        _simple_table(gain_table) if not gain_table.empty else "No gain table available.",
        "",
        "## Stage-1 Only",
        _simple_table(stage1_only_summary) if not stage1_only_summary.empty else "No stage1-only summary available.",
        "",
        "## RGB Contribution",
        _simple_table(rgb_ablation) if not rgb_ablation.empty else "No RGB ablation available.",
        "",
        "## CHM Contribution",
        _simple_table(chm_ablation) if not chm_ablation.empty else "No CHM ablation available.",
        "",
        "## Distance Transform + Watershed",
        _simple_table(dt_summary) if not dt_summary.empty else "No DT+watershed summary available.",
    ]
    write_text("\n".join(lines) + "\n", report_dir / "ticketfinal_report.md")


if __name__ == "__main__":
    main()

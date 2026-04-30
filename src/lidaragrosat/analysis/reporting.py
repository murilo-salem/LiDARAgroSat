from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.evaluation.statistics import bootstrap_confidence_interval, paired_wilcoxon
from lidaragrosat.utils.io import ensure_dir


def aggregate_plot_metrics(method: str, plot_id: str, match_result, extra_fields: dict | None = None) -> dict:
    metrics = summarize_matches(match_result)
    metrics["method"] = method
    metrics["plot_id"] = plot_id
    if extra_fields:
        metrics.update(extra_fields)
    return metrics


def build_method_summary(metrics_table: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["method"]
    if "protocol_id" in metrics_table.columns and metrics_table["protocol_id"].notna().any():
        group_columns = ["protocol_id", "method"]
    for group_key, group in metrics_table.groupby(group_columns):
        if isinstance(group_key, tuple) and len(group_key) == 2:
            protocol_id, method = group_key
        elif isinstance(group_key, tuple) and len(group_key) == 1:
            protocol_id, method = "", group_key[0]
        else:
            protocol_id, method = "", group_key
        values = group["f1"].to_numpy(dtype=float)
        f1_ci_low, f1_ci_high = bootstrap_confidence_interval(values, iterations=bootstrap_iterations)
        row = {
            "method": method,
            "precision_mean": group["precision"].mean(),
            "recall_mean": group["recall"].mean(),
            "f1_mean": group["f1"].mean(),
            "f1_ci_low": f1_ci_low,
            "f1_ci_high": f1_ci_high,
            "location_mae_mean_m": group["location_mae_m"].mean(),
            "height_mae_mean_m": group["height_mae_m"].mean(),
            "tp_total": group["tp"].sum(),
            "fp_total": group["fp"].sum(),
            "fn_total": group["fn"].sum(),
        }
        if protocol_id:
            row["protocol_id"] = protocol_id
        for optional_column in ["family", "stage1", "stage2", "labels", "seed"]:
            if optional_column in group.columns:
                values = group[optional_column].dropna().unique().tolist()
                if values:
                    row[optional_column] = values[0]
        rows.append(row)
    order_columns = ["f1_mean"]
    ascending = [False]
    if "protocol_id" in metrics_table.columns and metrics_table["protocol_id"].notna().any():
        order_columns = ["protocol_id", "f1_mean"]
        ascending = [True, False]
    return pd.DataFrame(rows).sort_values(order_columns, ascending=ascending).reset_index(drop=True)


def compare_methods(metrics_table: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    rows: list[dict] = []
    if "protocol_id" in metrics_table.columns:
        for protocol_id, group in metrics_table.groupby("protocol_id"):
            methods = [method for method in group["method"].unique() if method != reference_method]
            for method in methods:
                row = paired_wilcoxon(group, method, reference_method)
                row["protocol_id"] = protocol_id
                rows.append(row)
    else:
        methods = [method for method in metrics_table["method"].unique() if method != reference_method]
        rows = [paired_wilcoxon(metrics_table, method, reference_method) for method in methods]
    return pd.DataFrame(rows)


def save_tables(
    metrics_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    statistics_table: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = ensure_dir(output_dir)
    metrics_table.to_csv(output_dir / "metrics_by_plot.csv", index=False)
    summary_table.to_csv(output_dir / "method_summary.csv", index=False)
    statistics_table.to_csv(output_dir / "paired_statistics.csv", index=False)


def save_error_figure(
    rgb: np.ndarray,
    chm: np.ndarray,
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    match_result,
    destination: str | Path,
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.moveaxis(rgb, 0, -1))
    axes[0].scatter(reference["x_px"], reference["y_px"], s=12, c="lime", label="Ref")
    axes[0].scatter(predictions["x_px"], predictions["y_px"], s=10, c="red", label="Pred")
    axes[0].set_title("RGB + centers")
    axes[0].legend(loc="upper right")

    axes[1].imshow(chm, cmap="viridis")
    if not match_result.matches.empty:
        matched = match_result.matches
        pred_idx = matched["pred_idx"].to_numpy(dtype=int)
        ref_idx = matched["ref_idx"].to_numpy(dtype=int)
        axes[1].scatter(predictions.iloc[pred_idx]["x_px"], predictions.iloc[pred_idx]["y_px"], c="cyan", s=18)
        axes[1].scatter(reference.iloc[ref_idx]["x_px"], reference.iloc[ref_idx]["y_px"], c="white", s=18)
    axes[1].set_title("CHM + matched TP")
    plt.tight_layout()
    plt.savefig(destination, dpi=dpi)
    plt.close(fig)

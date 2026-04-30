from __future__ import annotations

import numpy as np
import pandas as pd


def extract_site_id(plot_id: str) -> str:
    parts = [part for part in str(plot_id).strip().split("_") if part]
    if not parts:
        return ""
    if len(parts) >= 2 and len(parts[0]) == 4 and parts[0].isdigit():
        return parts[1]
    return parts[0]


def add_site_column(
    table: pd.DataFrame,
    plot_column: str = "plot_id",
    site_column: str = "site_id",
) -> pd.DataFrame:
    enriched = table.copy()
    if plot_column not in enriched.columns:
        enriched[site_column] = ""
        return enriched
    enriched[site_column] = enriched[plot_column].map(extract_site_id)
    return enriched


def summarize_site_metrics(
    metrics_table: pd.DataFrame,
    metric_columns: tuple[str, ...] = ("precision", "recall", "f1"),
) -> pd.DataFrame:
    if metrics_table.empty:
        columns = ["method", "site_id", "n_plots"]
        for metric in metric_columns:
            columns.extend([f"{metric}_mean", f"{metric}_std"])
        return pd.DataFrame(columns=columns)
    table = add_site_column(metrics_table)
    aggregations: dict[str, list[str]] = {metric: ["mean", "std"] for metric in metric_columns if metric in table.columns}
    summary = (
        table.groupby(["method", "site_id"])
        .agg({**aggregations, "plot_id": "nunique"})
        .reset_index()
    )
    renamed_columns: list[str] = []
    for column in summary.columns:
        if not isinstance(column, tuple):
            renamed_columns.append(str(column))
            continue
        left, right = column
        if left == "method":
            renamed_columns.append("method")
        elif left == "site_id":
            renamed_columns.append("site_id")
        elif left == "plot_id" and right == "nunique":
            renamed_columns.append("n_plots")
        elif right:
            renamed_columns.append(f"{left}_{right}")
        else:
            renamed_columns.append(str(left))
    summary.columns = renamed_columns
    return summary.sort_values(["site_id", "method"]).reset_index(drop=True)


def count_site_wins(metrics_table: pd.DataFrame, metric: str = "f1", tolerance: float = 1e-12) -> pd.DataFrame:
    if metrics_table.empty or metric not in metrics_table.columns:
        return pd.DataFrame(columns=["method", "site_wins"])
    table = add_site_column(metrics_table)
    site_scores = (
        table.groupby(["method", "site_id"])[metric]
        .mean()
        .reset_index()
        .sort_values(["site_id", metric, "method"], ascending=[True, False, True])
    )
    winner_rows: list[dict] = []
    for site_id, group in site_scores.groupby("site_id"):
        best_score = float(group[metric].max())
        winners = group[np.abs(group[metric] - best_score) <= tolerance]
        for _, row in winners.iterrows():
            winner_rows.append({"site_id": site_id, "method": row["method"], "site_wins": 1})
    if not winner_rows:
        return pd.DataFrame(columns=["method", "site_wins"])
    wins = pd.DataFrame.from_records(winner_rows).groupby("method", as_index=False)["site_wins"].sum()
    return wins.sort_values(["site_wins", "method"], ascending=[False, True]).reset_index(drop=True)


def build_stage2_delta_tables(metrics_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"plot_id", "stage1", "stage2", "method", "f1"}
    if metrics_table.empty or not required.issubset(metrics_table.columns):
        empty_plot = pd.DataFrame(
            columns=["plot_id", "stage1", "stage2", "stage1_method", "stage2_method", "f1_stage1", "f1_stage2", "delta_f1"]
        )
        empty_summary = pd.DataFrame(
            columns=["stage1", "stage2", "stage1_method", "stage2_method", "n_plots", "f1_stage1_mean", "f1_stage2_mean", "delta_f1_mean", "delta_f1_median"]
        )
        return empty_plot, empty_summary

    base = metrics_table[metrics_table["stage2"] == "none"][["plot_id", "stage1", "method", "f1"]].rename(
        columns={"method": "stage1_method", "f1": "f1_stage1"}
    )
    refined = metrics_table[metrics_table["stage2"] != "none"][["plot_id", "stage1", "stage2", "method", "f1"]].rename(
        columns={"method": "stage2_method", "f1": "f1_stage2"}
    )
    merged = refined.merge(base, on=["plot_id", "stage1"], how="inner")
    if merged.empty:
        empty_summary = pd.DataFrame(
            columns=["stage1", "stage2", "stage1_method", "stage2_method", "n_plots", "f1_stage1_mean", "f1_stage2_mean", "delta_f1_mean", "delta_f1_median"]
        )
        return merged, empty_summary
    merged["delta_f1"] = merged["f1_stage2"] - merged["f1_stage1"]
    summary = (
        merged.groupby(["stage1", "stage2", "stage1_method", "stage2_method"])
        .agg(
            n_plots=("plot_id", "nunique"),
            f1_stage1_mean=("f1_stage1", "mean"),
            f1_stage2_mean=("f1_stage2", "mean"),
            delta_f1_mean=("delta_f1", "mean"),
            delta_f1_median=("delta_f1", "median"),
        )
        .reset_index()
        .sort_values(["stage1", "delta_f1_mean", "stage2"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    merged = merged.sort_values(["stage1", "stage2", "plot_id"]).reset_index(drop=True)
    return merged, summary


def map_failure_mode(mode: str, error_type: str) -> str:
    cleaned_mode = str(mode or "").strip()
    cleaned_error_type = str(error_type or "").strip()
    if cleaned_mode == "fp_border_artifact":
        return "border_error"
    if cleaned_mode == "fp_duplicate":
        return "fragmentation"
    if cleaned_mode == "fp_fragmented_texture":
        return "overdetection"
    if cleaned_mode == "fn_overlapping_crown":
        return "neighbor_fusion"
    if cleaned_mode == "fn_no_dominant_apex":
        return "underdetection"
    if cleaned_mode == "tp_spatial_displacement":
        return "spatial_displacement"
    if cleaned_error_type == "fp":
        return "overdetection"
    if cleaned_error_type == "fn":
        return "underdetection"
    return "other"


def summarize_failure_modes(
    error_table: pd.DataFrame,
    group_columns: tuple[str, ...] = ("method",),
) -> pd.DataFrame:
    expected_columns = list(group_columns) + ["failure_mode", "count", "fraction"]
    if error_table.empty or not {"error_type", "mode"}.issubset(error_table.columns):
        return pd.DataFrame(columns=expected_columns)

    table = error_table.copy()
    table["failure_mode"] = [
        map_failure_mode(mode, error_type)
        for mode, error_type in zip(table["mode"], table["error_type"], strict=False)
    ]
    available_groups = [column for column in group_columns if column in table.columns]
    if not available_groups:
        available_groups = ["failure_mode"]
        counts = table.groupby(available_groups).size().reset_index(name="count")
        total = float(counts["count"].sum())
        counts["fraction"] = counts["count"] / max(total, 1.0)
        return counts.sort_values(["count", "failure_mode"], ascending=[False, True]).reset_index(drop=True)

    counts = (
        table.groupby(available_groups + ["failure_mode"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(available_groups)["count"].transform("sum")
    counts["fraction"] = counts["count"] / np.maximum(totals, 1.0)
    return counts.sort_values(available_groups + ["count", "failure_mode"], ascending=[True] * len(available_groups) + [False, True]).reset_index(drop=True)


def summarize_match_radius_sensitivity(
    metrics_table: pd.DataFrame,
    radius_column: str = "match_radius_m",
    metric_columns: tuple[str, ...] = ("precision", "recall", "f1"),
) -> pd.DataFrame:
    if metrics_table.empty or radius_column not in metrics_table.columns:
        columns = ["method", radius_column]
        columns.extend(f"{metric}_mean" for metric in metric_columns)
        return pd.DataFrame(columns=columns)
    summary = (
        metrics_table.groupby(["method", radius_column])[list(metric_columns)]
        .mean()
        .reset_index()
    )
    summary = summary.rename(columns={metric: f"{metric}_mean" for metric in metric_columns})
    return summary.sort_values(["method", radius_column]).reset_index(drop=True)


def summarize_threshold_sweep(
    sweep_table: pd.DataFrame,
    group_columns: tuple[str, ...] = ("method",),
) -> pd.DataFrame:
    columns = list(group_columns) + [
        "threshold",
        "nms_radius_m",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "predictions_total",
        "predictions_mean_per_plot",
        "zero_prediction_plots",
    ]
    if sweep_table.empty:
        return pd.DataFrame(columns=columns)
    required = {"threshold", "nms_radius_m", "precision_mean", "recall_mean", "f1_mean"}
    if not required.issubset(sweep_table.columns):
        return pd.DataFrame(columns=columns)
    available_groups = [column for column in group_columns if column in sweep_table.columns]
    if not available_groups:
        available_groups = ["threshold"]
    best = (
        sweep_table.sort_values(
            available_groups + ["f1_mean", "recall_mean", "precision_mean", "threshold"],
            ascending=[True] * len(available_groups) + [False, False, False, True],
        )
        .groupby(available_groups, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    missing_columns = [column for column in columns if column not in best.columns]
    for column in missing_columns:
        best[column] = np.nan
    return best[columns]

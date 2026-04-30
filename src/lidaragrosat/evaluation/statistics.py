from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon(metric_by_plot: pd.DataFrame, method_a: str, method_b: str, metric: str = "f1") -> dict:
    pivot = metric_by_plot.pivot(index="plot_id", columns="method", values=metric).dropna()
    if len(pivot) < 2:
        return {
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "n_plots": int(len(pivot)),
            "statistic": np.nan,
            "p_value": np.nan,
            "mean_difference": float((pivot[method_a] - pivot[method_b]).mean()) if len(pivot) else np.nan,
            "relative_gain_percent": float(
                (100.0 * ((pivot[method_a] - pivot[method_b]) / np.maximum(pivot[method_b], 1e-6))).mean()
            )
            if len(pivot)
            else np.nan,
        }
    sample_a = pivot[method_a].to_numpy(dtype=float)
    sample_b = pivot[method_b].to_numpy(dtype=float)
    if np.allclose(sample_a, sample_b):
        statistic = 0.0
        p_value = 1.0
    else:
        statistic, p_value = wilcoxon(sample_a, sample_b, alternative="greater")
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_plots": int(len(pivot)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_difference": float(np.mean(sample_a - sample_b)),
        "relative_gain_percent": float(
            100.0 * np.mean((sample_a - sample_b) / np.maximum(sample_b, 1e-6))
        ),
    }


def bootstrap_confidence_interval(values: np.ndarray, iterations: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(iterations):
        resampled = rng.choice(values, size=len(values), replace=True)
        samples.append(np.mean(resampled))
    lower = np.quantile(samples, alpha / 2)
    upper = np.quantile(samples, 1 - alpha / 2)
    return float(lower), float(upper)

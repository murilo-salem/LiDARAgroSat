from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.utils.geo import GeoTransform


@dataclass(slots=True)
class LocalMaximaParams:
    window_radius_m: float
    gaussian_sigma: float
    min_height_m: float


def detect_local_maxima(
    chm: np.ndarray,
    transform: GeoTransform,
    params: LocalMaximaParams,
) -> pd.DataFrame:
    radius_px = max(1, int(round(params.window_radius_m / transform.pixel_size)))
    smoothed = gaussian_filter(chm, sigma=params.gaussian_sigma)
    local_max = maximum_filter(smoothed, size=2 * radius_px + 1, mode="nearest")
    peak_mask = (smoothed == local_max) & (smoothed >= params.min_height_m)
    peak_rows, peak_cols = np.where(peak_mask)

    records: list[dict] = []
    for row, col in zip(peak_rows, peak_cols):
        x, y = transform.rowcol_to_xy(int(row), int(col))
        records.append(
            {
                "x": x,
                "y": y,
                "height_pred": float(chm[row, col]),
                "confidence": float(smoothed[row, col]),
            }
        )
    return pd.DataFrame.from_records(records, columns=["x", "y", "height_pred", "confidence"])


def tune_local_maxima(
    chm_by_plot: dict[str, np.ndarray],
    transform_by_plot: dict[str, GeoTransform],
    reference_by_plot: dict[str, pd.DataFrame],
    validation_plots: list[str],
    params_grid: list[LocalMaximaParams],
    match_radius_m: float,
    height_tolerance_m: float,
) -> LocalMaximaParams:
    best_params = params_grid[0]
    best_f1 = -1.0
    for params in params_grid:
        fold_scores: list[float] = []
        for plot_id in validation_plots:
            predictions = detect_local_maxima(chm_by_plot[plot_id], transform_by_plot[plot_id], params)
            matches = match_tree_predictions(
                predictions,
                reference_by_plot[plot_id],
                match_radius_m=match_radius_m,
                height_tolerance_m=height_tolerance_m,
            )
            fold_scores.append(summarize_matches(matches)["f1"])
        mean_f1 = float(np.mean(fold_scores))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params
    return best_params

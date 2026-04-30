from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lidaragrosat.data.dataset import ProcessedPlotBundle
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.stage1.api import Stage1Output


@dataclass(slots=True)
class ErrorModeConfig:
    match_radius_m: float = 1.5
    duplicate_radius_m: float = 1.2
    border_margin_m: float = 1.0
    occluded_neighbor_radius_m: float = 2.0
    off_apex_distance_m: float = 1.0


def _pairwise_min_distance(points_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0 or len(query_xy) == 0:
        return np.full((len(query_xy),), np.inf, dtype=float)
    distances = np.sqrt(((query_xy[:, None, :] - points_xy[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def _plot_bounds(bundle: ProcessedPlotBundle) -> tuple[float, float, float, float]:
    height, width = bundle.raster_shape
    x0, y0 = bundle.transform.rowcol_to_xy(0, 0)
    x1, y1 = bundle.transform.rowcol_to_xy(height - 1, width - 1)
    return float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1))


def classify_prediction_error_modes(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    bundle: ProcessedPlotBundle,
    stage1_output: Stage1Output | None = None,
    config: ErrorModeConfig | None = None,
) -> dict[str, int]:
    cfg = config or ErrorModeConfig()
    result = match_tree_predictions(
        predictions=predictions,
        reference=reference,
        match_radius_m=cfg.match_radius_m,
        height_tolerance_m=None,
    )
    counts = {
        "under_segmentation": 0,
        "over_segmentation": 0,
        "mixed_segment": 0,
        "duplicate": 0,
        "off_apex": 0,
        "border": 0,
        "occluded_neighbor": 0,
    }
    if stage1_output is not None and stage1_output.diagnostics is not None:
        counts["under_segmentation"] = int(stage1_output.diagnostics.under_segmented)
        counts["over_segmentation"] = int(stage1_output.diagnostics.over_segmented)
        counts["mixed_segment"] = int(stage1_output.diagnostics.mixed_segment)

    matched_pred = set(result.matches["pred_idx"].astype(int).tolist()) if not result.matches.empty else set()
    pred_xy = predictions[["x", "y"]].to_numpy(dtype=float) if len(predictions) else np.zeros((0, 2), dtype=float)
    matched_xy = pred_xy[sorted(matched_pred)] if matched_pred else np.zeros((0, 2), dtype=float)

    x_min, y_min, x_max, y_max = _plot_bounds(bundle)
    if not result.matches.empty:
        off_apex = result.matches["distance_m"] > float(cfg.off_apex_distance_m)
        counts["off_apex"] = int(off_apex.sum())

    unmatched_pred_indices = [index for index in range(len(predictions)) if index not in matched_pred]
    if unmatched_pred_indices:
        unmatched = predictions.iloc[unmatched_pred_indices].copy().reset_index(drop=True)
        unmatched_xy = unmatched[["x", "y"]].to_numpy(dtype=float)
        min_to_matched = _pairwise_min_distance(matched_xy, unmatched_xy)
        counts["duplicate"] = int(np.sum(min_to_matched <= float(cfg.duplicate_radius_m)))

        border_distances = np.minimum.reduce(
            [
                unmatched_xy[:, 0] - x_min,
                x_max - unmatched_xy[:, 0],
                unmatched_xy[:, 1] - y_min,
                y_max - unmatched_xy[:, 1],
            ]
        )
        counts["border"] = int(np.sum(border_distances <= float(cfg.border_margin_m)))

    if not result.false_negatives.empty and len(reference) > 1:
        ref_xy = reference[["x", "y"]].to_numpy(dtype=float)
        fn_xy = result.false_negatives[["x", "y"]].to_numpy(dtype=float)
        nearest_other = []
        for point in fn_xy:
            distances = np.sqrt(((ref_xy - point[None, :]) ** 2).sum(axis=1))
            finite = distances[distances > 1e-6]
            nearest_other.append(float(finite.min()) if len(finite) else np.inf)
        counts["occluded_neighbor"] = int(np.sum(np.asarray(nearest_other) <= float(cfg.occluded_neighbor_radius_m)))
    return counts


def summarize_error_modes(
    methods: dict[str, dict[str, pd.DataFrame]],
    bundles: dict[str, ProcessedPlotBundle],
    stage1_outputs_by_method: dict[str, dict[str, Stage1Output]] | None = None,
    config: ErrorModeConfig | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    stage1_outputs_by_method = stage1_outputs_by_method or {}
    for method_name, predictions_by_plot in methods.items():
        for plot_id, predictions in predictions_by_plot.items():
            stage1_output = stage1_outputs_by_method.get(method_name, {}).get(plot_id)
            counts = classify_prediction_error_modes(
                predictions=predictions,
                reference=bundles[plot_id].reference,
                bundle=bundles[plot_id],
                stage1_output=stage1_output,
                config=config,
            )
            rows.append({"method": method_name, "plot_id": plot_id, **counts})
    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "plot_id",
                "under_segmentation",
                "over_segmentation",
                "mixed_segment",
                "duplicate",
                "off_apex",
                "border",
                "occluded_neighbor",
            ]
        )
    table = pd.DataFrame.from_records(rows)
    totals = (
        table.groupby("method")[
            [
                "under_segmentation",
                "over_segmentation",
                "mixed_segment",
                "duplicate",
                "off_apex",
                "border",
                "occluded_neighbor",
            ]
        ]
        .sum()
        .reset_index()
    )
    totals["plot_id"] = "ALL"
    return pd.concat([table, totals], ignore_index=True)

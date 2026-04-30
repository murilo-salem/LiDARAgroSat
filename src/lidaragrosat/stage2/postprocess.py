from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches


def _nms_points(predictions: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    ordered = predictions.sort_values("score", ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[tuple[float, float]] = []
    for _, row in ordered.iterrows():
        point = (float(row["x"]), float(row["y"]))
        if any(np.hypot(point[0] - x, point[1] - y) <= float(radius_m) for x, y in kept_xy):
            continue
        kept_rows.append(row.to_dict())
        kept_xy.append(point)
    return pd.DataFrame.from_records(kept_rows)


def apply_stage2_postprocess(
    scored_table: pd.DataFrame,
    objectness_threshold: float,
    nms_radius_m: float,
    center_mode: str = "refined",
) -> pd.DataFrame:
    if scored_table.empty:
        return scored_table.copy()
    selected = scored_table[scored_table["score"] >= float(objectness_threshold)].copy()
    if selected.empty:
        return selected
    if center_mode == "geom":
        selected["x"] = selected["geom_x"]
        selected["y"] = selected["geom_y"]
    elif center_mode == "hybrid":
        selected["x"] = 0.5 * selected["geom_x"] + 0.5 * selected["refined_x"]
        selected["y"] = 0.5 * selected["geom_y"] + 0.5 * selected["refined_y"]
    else:
        selected["x"] = selected["refined_x"]
        selected["y"] = selected["refined_y"]
    selected["height_pred"] = selected["refined_height"]
    return _nms_points(selected, radius_m=nms_radius_m)


def calibrate_stage2_postprocess(
    scored_by_plot: dict[str, pd.DataFrame],
    reference_by_plot: dict[str, pd.DataFrame],
    match_radius_m: float,
    height_tolerance_m: float,
    threshold_grid: list[float],
    nms_grid_m: list[float],
    center_modes: list[str] | tuple[str, ...] = ("refined", "geom", "hybrid"),
) -> tuple[dict, dict]:
    best_params = {"objectness_threshold": 0.5, "nms_radius_m": 1.2, "center_mode": "refined"}
    best_scores = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold, nms_radius_m, center_mode in product(threshold_grid, nms_grid_m, center_modes):
        fold_scores = []
        for plot_id, scored in scored_by_plot.items():
            predictions = apply_stage2_postprocess(scored, threshold, nms_radius_m, center_mode=center_mode)
            matches = match_tree_predictions(
                predictions,
                reference_by_plot[plot_id],
                match_radius_m=match_radius_m,
                height_tolerance_m=height_tolerance_m,
            )
            fold_scores.append(summarize_matches(matches))
        precision = float(np.mean([score["precision"] for score in fold_scores])) if fold_scores else 0.0
        recall = float(np.mean([score["recall"] for score in fold_scores])) if fold_scores else 0.0
        f1 = float(np.mean([score["f1"] for score in fold_scores])) if fold_scores else 0.0
        if f1 > best_scores["f1"] or (f1 == best_scores["f1"] and recall > best_scores["recall"]):
            best_params = {"objectness_threshold": float(threshold), "nms_radius_m": float(nms_radius_m), "center_mode": str(center_mode)}
            best_scores = {"precision": precision, "recall": recall, "f1": f1}
    return best_params, best_scores

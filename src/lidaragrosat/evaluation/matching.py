from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(slots=True)
class MatchResult:
    matches: pd.DataFrame
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame


def match_tree_predictions(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    match_radius_m: float,
    height_tolerance_m: float | None = None,
) -> MatchResult:
    if predictions.empty:
        return MatchResult(
            matches=pd.DataFrame(columns=["pred_idx", "ref_idx", "distance_m", "height_error_m"]),
            false_positives=predictions.copy(),
            false_negatives=reference.copy(),
        )
    if reference.empty:
        return MatchResult(
            matches=pd.DataFrame(columns=["pred_idx", "ref_idx", "distance_m", "height_error_m"]),
            false_positives=predictions.copy(),
            false_negatives=reference.copy(),
        )

    pred_xy = predictions[["x", "y"]].to_numpy(dtype=float)
    ref_xy = reference[["x", "y"]].to_numpy(dtype=float)
    distances = np.sqrt(((pred_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2))

    admissible = distances <= match_radius_m
    height_errors = np.full_like(distances, np.nan, dtype=float)
    if height_tolerance_m is not None and "height_pred" in predictions.columns and "height" in reference.columns:
        pred_h = predictions["height_pred"].to_numpy(dtype=float)
        ref_h = pd.to_numeric(reference["height"], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(pred_h[:, None]) & np.isfinite(ref_h[None, :])
        height_errors = np.abs(pred_h[:, None] - ref_h[None, :])
        admissible &= (~finite_mask) | (height_errors <= height_tolerance_m)

    cost_matrix = distances.copy()
    cost_matrix[~admissible] = 1e6
    pred_indices, ref_indices = linear_sum_assignment(cost_matrix)

    paired_rows: list[dict] = []
    matched_pred = set()
    matched_ref = set()
    for pred_idx, ref_idx in zip(pred_indices, ref_indices):
        if cost_matrix[pred_idx, ref_idx] >= 1e6:
            continue
        paired_rows.append(
            {
                "pred_idx": int(pred_idx),
                "ref_idx": int(ref_idx),
                "distance_m": float(distances[pred_idx, ref_idx]),
                "height_error_m": (
                    float(height_errors[pred_idx, ref_idx])
                    if np.isfinite(height_errors[pred_idx, ref_idx])
                    else np.nan
                ),
            }
        )
        matched_pred.add(int(pred_idx))
        matched_ref.add(int(ref_idx))

    matches = pd.DataFrame.from_records(paired_rows)
    false_positives = predictions.iloc[
        [index for index in range(len(predictions)) if index not in matched_pred]
    ].reset_index(drop=True)
    false_negatives = reference.iloc[
        [index for index in range(len(reference)) if index not in matched_ref]
    ].reset_index(drop=True)
    return MatchResult(matches=matches, false_positives=false_positives, false_negatives=false_negatives)


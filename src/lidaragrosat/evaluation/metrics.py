from __future__ import annotations

import numpy as np

from lidaragrosat.evaluation.matching import MatchResult


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_matches(match_result: MatchResult) -> dict[str, float]:
    tp = int(len(match_result.matches))
    fp = int(len(match_result.false_positives))
    fn = int(len(match_result.false_negatives))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    location_mae = float(match_result.matches["distance_m"].mean()) if tp else np.nan
    height_mae = float(match_result.matches["height_error_m"].mean()) if tp else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "location_mae_m": location_mae,
        "height_mae_m": height_mae,
    }


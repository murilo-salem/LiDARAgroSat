from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

from lidaragrosat.data.reference import read_reference_table


@dataclass(slots=True)
class FaithfulPreprocessParams:
    duplicate_tolerance_m: float = 0.001
    manual_min_abs_z: float | None = None
    manual_max_abs_z: float | None = None
    ground_classification: int = 2


@dataclass(slots=True)
class FaithfulBaselineParams:
    window_size_m: float = 2.0
    height_threshold_m: float = 3.0


@dataclass(slots=True)
class FaithfulMatchingParams:
    max_dist_m: float = 5.0
    max_height_diff_m: float = 3.0


@dataclass(slots=True)
class FaithfulMatchResult:
    matches: pd.DataFrame
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    pred_states: np.ndarray
    ref_states: np.ndarray


def deduplicate_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    tolerance_m: float,
) -> tuple[np.ndarray, dict]:
    if len(x) == 0:
        return np.empty((0,), dtype=np.int64), {"duplicates_removed": 0, "unique_points": 0}
    scaled = np.column_stack(
        [
            np.round(np.asarray(x, dtype=np.float64) / tolerance_m).astype(np.int64),
            np.round(np.asarray(y, dtype=np.float64) / tolerance_m).astype(np.int64),
            np.round(np.asarray(z, dtype=np.float64) / tolerance_m).astype(np.int64),
            np.asarray(classification, dtype=np.int16).astype(np.int64),
        ]
    )
    _, unique_indices = np.unique(scaled, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return unique_indices, {
        "duplicates_removed": int(len(x) - len(unique_indices)),
        "unique_points": int(len(unique_indices)),
    }


def _manual_noise_filter_mask(
    z: np.ndarray,
    min_abs_z: float | None,
    max_abs_z: float | None,
) -> np.ndarray:
    mask = np.ones(len(z), dtype=bool)
    if min_abs_z is not None:
        mask &= np.asarray(z, dtype=np.float64) >= float(min_abs_z)
    if max_abs_z is not None:
        mask &= np.asarray(z, dtype=np.float64) <= float(max_abs_z)
    return mask


def _interpolate_ground_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    ground_classification: int,
) -> np.ndarray:
    ground_mask = np.asarray(classification) == int(ground_classification)
    if int(ground_mask.sum()) < 3:
        raise ValueError("Ground classification is missing or insufficient for normalization.")
    ground_xy = np.column_stack([np.asarray(x)[ground_mask], np.asarray(y)[ground_mask]])
    ground_z = np.asarray(z, dtype=np.float64)[ground_mask]
    linear = LinearNDInterpolator(ground_xy, ground_z, fill_value=np.nan)
    sampled = np.asarray(linear(np.asarray(x), np.asarray(y)), dtype=np.float64)
    missing = ~np.isfinite(sampled)
    if np.any(missing):
        nearest = NearestNDInterpolator(ground_xy, ground_z)
        sampled[missing] = np.asarray(nearest(np.asarray(x)[missing], np.asarray(y)[missing]), dtype=np.float64)
    return sampled


def preprocess_vendor_lidar_plot(
    lidar_path: str | Path,
    reference_path: str | Path,
    params: FaithfulPreprocessParams | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    params = params or FaithfulPreprocessParams()
    las = laspy.read(lidar_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    classification = np.asarray(las.classification, dtype=np.int16)

    raw_count = int(len(x))
    unique_indices, dedup_stats = deduplicate_points(
        x=x,
        y=y,
        z=z,
        classification=classification,
        tolerance_m=float(params.duplicate_tolerance_m),
    )
    x = x[unique_indices]
    y = y[unique_indices]
    z = z[unique_indices]
    classification = classification[unique_indices]

    manual_mask = _manual_noise_filter_mask(
        z=z,
        min_abs_z=params.manual_min_abs_z,
        max_abs_z=params.manual_max_abs_z,
    )
    x = x[manual_mask]
    y = y[manual_mask]
    z = z[manual_mask]
    classification = classification[manual_mask]

    ground_z = _interpolate_ground_surface(
        x=x,
        y=y,
        z=z,
        classification=classification,
        ground_classification=params.ground_classification,
    )
    z_norm = np.asarray(z - ground_z, dtype=np.float64)
    points_xyz = np.column_stack([x, y, z_norm]).astype(np.float64)
    reference = read_reference_table(reference_path)
    stats = {
        "raw_point_count": raw_count,
        "unique_point_count": int(len(unique_indices)),
        "duplicates_removed": int(dedup_stats["duplicates_removed"]),
        "manual_filtered_point_count": int(raw_count - len(points_xyz) - dedup_stats["duplicates_removed"]),
        "ground_point_count": int((classification == int(params.ground_classification)).sum()),
        "normalized_z_min": float(np.min(z_norm)) if len(z_norm) else np.nan,
        "normalized_z_max": float(np.max(z_norm)) if len(z_norm) else np.nan,
        "reference_tree_count": int(len(reference)),
        "reference_height_available_count": int(pd.to_numeric(reference["height"], errors="coerce").notna().sum()),
    }
    return points_xyz, reference, stats


def local_maxima_filter_3d(
    points_xyz: np.ndarray,
    window_size_m: float = 2.0,
    height_threshold_m: float = 3.0,
) -> pd.DataFrame:
    if len(points_xyz) == 0:
        return pd.DataFrame(columns=["x", "y", "height_pred"])
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[pts[:, 2] >= float(height_threshold_m)]
    if len(pts) == 0:
        return pd.DataFrame(columns=["x", "y", "height_pred"])

    tree = cKDTree(pts)
    processed = np.zeros(len(pts), dtype=bool)
    detections: list[np.ndarray] = []

    for index, point in enumerate(pts):
        if processed[index]:
            continue
        neighbor_indices = np.asarray(tree.query_ball_point(point, r=float(window_size_m)), dtype=np.int64)
        if neighbor_indices.size == 0:
            continue
        highest_index = int(neighbor_indices[np.argmax(pts[neighbor_indices, 2])])
        other_indices = neighbor_indices[neighbor_indices != highest_index]
        if other_indices.size:
            processed[other_indices] = True
        if highest_index == index:
            detections.append(pts[index].copy())

    if not detections:
        return pd.DataFrame(columns=["x", "y", "height_pred"])
    detections_array = np.vstack(detections)
    return pd.DataFrame(
        {
            "x": detections_array[:, 0],
            "y": detections_array[:, 1],
            "height_pred": detections_array[:, 2],
        }
    )


def faithful_greedy_match(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    params: FaithfulMatchingParams | None = None,
) -> FaithfulMatchResult:
    params = params or FaithfulMatchingParams()
    if predictions.empty and reference.empty:
        return FaithfulMatchResult(
            matches=pd.DataFrame(columns=["pred_idx", "ref_idx", "distance_m", "height_error_m"]),
            false_positives=predictions.copy(),
            false_negatives=reference.copy(),
            pred_states=np.zeros(0, dtype=np.int8),
            ref_states=np.zeros(0, dtype=np.int8),
        )
    if predictions.empty:
        return FaithfulMatchResult(
            matches=pd.DataFrame(columns=["pred_idx", "ref_idx", "distance_m", "height_error_m"]),
            false_positives=predictions.copy(),
            false_negatives=reference.copy(),
            pred_states=np.zeros(0, dtype=np.int8),
            ref_states=np.full(len(reference), -1, dtype=np.int8),
        )
    if reference.empty:
        return FaithfulMatchResult(
            matches=pd.DataFrame(columns=["pred_idx", "ref_idx", "distance_m", "height_error_m"]),
            false_positives=predictions.copy(),
            false_negatives=reference.copy(),
            pred_states=np.full(len(predictions), -1, dtype=np.int8),
            ref_states=np.zeros(0, dtype=np.int8),
        )

    pred_xy = predictions[["x", "y"]].to_numpy(dtype=np.float64)
    ref_xy = reference[["x", "y"]].to_numpy(dtype=np.float64)
    distances = np.sqrt(((pred_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2))

    pred_h = pd.to_numeric(predictions.get("height_pred"), errors="coerce").to_numpy(dtype=np.float64)
    ref_h = pd.to_numeric(reference.get("height"), errors="coerce").to_numpy(dtype=np.float64)
    height_error = np.abs(pred_h[:, None] - ref_h[None, :])

    pairs: list[tuple[float, int, int, float]] = []
    for pred_idx in range(len(predictions)):
        for ref_idx in range(len(reference)):
            distance = float(distances[pred_idx, ref_idx])
            if distance > float(params.max_dist_m):
                continue
            current_height_error = float(height_error[pred_idx, ref_idx]) if np.isfinite(height_error[pred_idx, ref_idx]) else np.nan
            if np.isfinite(ref_h[ref_idx]) and np.isfinite(pred_h[pred_idx]) and current_height_error > float(params.max_height_diff_m):
                continue
            pairs.append((distance, pred_idx, ref_idx, current_height_error))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    pred_states = np.zeros(len(predictions), dtype=np.int8)
    ref_states = np.zeros(len(reference), dtype=np.int8)
    matched_rows: list[dict] = []

    for distance, pred_idx, ref_idx, current_height_error in pairs:
        pred_classified = pred_states[pred_idx] != 0
        ref_classified = ref_states[ref_idx] != 0
        if not pred_classified and not ref_classified:
            pred_states[pred_idx] = 1
            ref_states[ref_idx] = 1
            matched_rows.append(
                {
                    "pred_idx": int(pred_idx),
                    "ref_idx": int(ref_idx),
                    "distance_m": float(distance),
                    "height_error_m": current_height_error,
                }
            )
        elif pred_classified and not ref_classified:
            ref_states[ref_idx] = -1
        elif not pred_classified and ref_classified:
            pred_states[pred_idx] = -1

    pred_states[pred_states == 0] = -1
    ref_states[ref_states == 0] = -1

    matches = pd.DataFrame.from_records(matched_rows, columns=["pred_idx", "ref_idx", "distance_m", "height_error_m"])
    false_positives = predictions.iloc[np.where(pred_states == -1)[0]].reset_index(drop=True)
    false_negatives = reference.iloc[np.where(ref_states == -1)[0]].reset_index(drop=True)
    return FaithfulMatchResult(
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        pred_states=pred_states,
        ref_states=ref_states,
    )


def summarize_faithful_matches(match_result: FaithfulMatchResult) -> dict[str, float]:
    tp = int(len(match_result.matches))
    fp = int(len(match_result.false_positives))
    fn = int(len(match_result.false_negatives))
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    mean_distance = float(match_result.matches["distance_m"].mean()) if tp > 0 else np.nan
    mean_height_error = float(match_result.matches["height_error_m"].dropna().mean()) if tp > 0 and "height_error_m" in match_result.matches else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "distance_m": mean_distance,
        "height_error_m": mean_height_error,
    }

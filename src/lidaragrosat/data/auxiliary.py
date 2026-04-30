from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter, maximum_filter, uniform_filter

from lidaragrosat.utils.geo import GeoTransform


CONIFER_SPECIES = {"fir", "pine", "spruce"}


def _robust_normalize(array: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    scale = float(np.percentile(np.abs(array), percentile)) if np.any(np.abs(array) > 0) else 1.0
    return np.clip(array / max(scale, 1e-6), -1.0, 1.0).astype(np.float32)


def normalize_species_label(label: str) -> str:
    return str(label).strip().lower()


def coarse_species_label(label: str) -> str:
    normalized = normalize_species_label(label)
    return "conifer" if normalized in CONIFER_SPECIES else "broadleaf"


def build_species_lookup(reference_tables: list[pd.DataFrame], mode: str) -> dict[str, int]:
    if mode == "none":
        return {}
    labels: set[str] = set()
    for table in reference_tables:
        values = table["species"].fillna("unknown").astype(str)
        if mode == "coarse":
            labels.update(coarse_species_label(value) for value in values)
        elif mode == "fine":
            labels.update(normalize_species_label(value) for value in values)
        else:
            raise ValueError(f"Unsupported species label mode: {mode}")
    return {label: index for index, label in enumerate(sorted(labels))}


def encode_species_label(label: str, species_to_index: dict[str, int], mode: str) -> int:
    if mode == "none" or not species_to_index:
        return -1
    encoded_label = coarse_species_label(label) if mode == "coarse" else normalize_species_label(label)
    return int(species_to_index.get(encoded_label, -1))


def compute_rgb_complexity(rgb: np.ndarray, window_size: int = 9) -> np.ndarray:
    grayscale = rgb.mean(axis=0).astype(np.float32)
    local_mean = uniform_filter(grayscale, size=window_size, mode="nearest")
    local_mean_sq = uniform_filter(grayscale**2, size=window_size, mode="nearest")
    complexity = np.sqrt(np.clip(local_mean_sq - local_mean**2, 0.0, None)).astype(np.float32)
    scale = float(np.percentile(complexity, 99.0)) if np.any(complexity > 0) else 1.0
    return np.clip(complexity / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def compute_rgb_entropy(rgb: np.ndarray, window_size: int = 9) -> np.ndarray:
    grayscale = rgb.mean(axis=0).astype(np.float32)
    probability = np.clip((grayscale - grayscale.min()) / max(float(np.ptp(grayscale)), 1e-6), 0.0, 1.0)
    local_mean = uniform_filter(probability, size=window_size, mode="nearest")
    local_mean_sq = uniform_filter(probability**2, size=window_size, mode="nearest")
    variance = np.clip(local_mean_sq - local_mean**2, 0.0, None)
    entropy_like = -np.clip(local_mean, 1e-6, 1.0) * np.log(np.clip(local_mean, 1e-6, 1.0)) - np.clip(
        1.0 - local_mean, 1e-6, 1.0
    ) * np.log(np.clip(1.0 - local_mean, 1e-6, 1.0))
    score = entropy_like + variance
    scale = float(np.percentile(score, 99.0)) if np.any(score > 0) else 1.0
    return np.clip(score / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def compute_multiscale_peakness(
    chm: np.ndarray,
    pixel_size: float,
    window_sizes_m: tuple[float, ...] | list[float],
    min_height_m: float = 2.0,
) -> np.ndarray:
    responses: list[np.ndarray] = []
    smoothed = gaussian_filter(chm.astype(np.float32), sigma=1.0)
    for window_size_m in window_sizes_m:
        radius_px = max(1, int(round(window_size_m / pixel_size)))
        window = 2 * radius_px + 1
        local_max = maximum_filter(smoothed, size=window, mode="nearest")
        local_mean = uniform_filter(smoothed, size=window, mode="nearest")
        peaks = (smoothed >= local_max - 1e-6) & (smoothed >= min_height_m)
        prominence = np.clip(smoothed - local_mean, 0.0, None)
        scale = float(np.percentile(prominence, 99.0)) if np.any(prominence > 0) else 1.0
        score = peaks.astype(np.float32) * np.clip(prominence / max(scale, 1e-6), 0.0, 1.0)
        responses.append(gaussian_filter(score, sigma=max(1.0, radius_px / 2.0)).astype(np.float32))
    if not responses:
        return np.zeros_like(chm, dtype=np.float32)
    peakness = np.mean(responses, axis=0).astype(np.float32)
    scale = float(np.percentile(peakness, 99.0)) if np.any(peakness > 0) else 1.0
    return np.clip(peakness / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def compute_chm_feature_stack(
    chm: np.ndarray,
    pixel_size: float,
    peakness_window_sizes_m: tuple[float, ...] | list[float] = (0.8, 1.2, 1.6),
    min_height_m: float = 2.0,
) -> np.ndarray:
    chm = chm.astype(np.float32)
    grad_y, grad_x = np.gradient(chm, pixel_size, pixel_size)
    dyy, _ = np.gradient(grad_y, pixel_size, pixel_size)
    _, dxx = np.gradient(grad_x, pixel_size, pixel_size)
    laplacian = dxx + dyy
    curvature = np.sqrt(np.clip(dxx**2 + dyy**2, 0.0, None))
    gradient = np.sqrt(np.clip(grad_x**2 + grad_y**2, 0.0, None))
    local_mean = uniform_filter(chm, size=9, mode="nearest")
    local_mean_sq = uniform_filter(chm**2, size=9, mode="nearest")
    local_roughness = np.sqrt(np.clip(local_mean_sq - local_mean**2, 0.0, None))
    peakness = compute_multiscale_peakness(
        chm=chm,
        pixel_size=pixel_size,
        window_sizes_m=peakness_window_sizes_m,
        min_height_m=min_height_m,
    )
    return np.stack(
        [
            chm.astype(np.float32),
            _robust_normalize(gradient),
            _robust_normalize(laplacian),
            _robust_normalize(curvature),
            peakness.astype(np.float32),
            _robust_normalize(local_roughness),
        ],
        axis=0,
    ).astype(np.float32)


def nearest_neighbor_distances(reference_table: pd.DataFrame) -> np.ndarray:
    if len(reference_table) <= 1:
        return np.full(len(reference_table), np.nan, dtype=np.float32)
    xy = reference_table[["x", "y"]].to_numpy(dtype=float)
    distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1).astype(np.float32)


def find_local_apex_pixel(
    chm: np.ndarray,
    center_row: int,
    center_col: int,
    radius_px: int,
) -> tuple[int, int]:
    row_min = max(0, center_row - radius_px)
    row_max = min(chm.shape[0], center_row + radius_px + 1)
    col_min = max(0, center_col - radius_px)
    col_max = min(chm.shape[1], center_col + radius_px + 1)
    window = chm[row_min:row_max, col_min:col_max]
    if window.size == 0:
        return center_row, center_col
    flat_index = int(np.argmax(window))
    local_row, local_col = np.unravel_index(flat_index, window.shape)
    return int(row_min + local_row), int(col_min + local_col)


def audit_reference_to_apex_offsets(
    reference_table: pd.DataFrame,
    transform: GeoTransform,
    chm: np.ndarray,
    snap_radius_m: float,
) -> pd.DataFrame:
    if len(reference_table) == 0:
        return pd.DataFrame(columns=["x", "y", "apex_x", "apex_y", "offset_m"])
    radius_px = max(1, int(round(snap_radius_m / transform.pixel_size)))
    records: list[dict] = []
    for _, row in reference_table.iterrows():
        center_row, center_col = transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
        if center_row < 0 or center_row >= chm.shape[0] or center_col < 0 or center_col >= chm.shape[1]:
            continue
        apex_row, apex_col = find_local_apex_pixel(chm, center_row, center_col, radius_px)
        apex_x, apex_y = transform.rowcol_to_xy(apex_row, apex_col)
        offset_m = float(np.hypot(apex_x - float(row["x"]), apex_y - float(row["y"])))
        records.append(
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "apex_x": float(apex_x),
                "apex_y": float(apex_y),
                "offset_m": offset_m,
                "species": row.get("species"),
                "height": row.get("height"),
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(slots=True)
class HardNegativeCandidates:
    rows: np.ndarray
    cols: np.ndarray
    scores: np.ndarray


def detect_multiscale_hard_negative_candidates(
    chm: np.ndarray,
    pixel_size: float,
    window_sizes_m: tuple[float, ...] | list[float],
    min_height_m: float,
) -> HardNegativeCandidates:
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    smoothed = gaussian_filter(chm.astype(np.float32), sigma=1.0)
    for window_size_m in window_sizes_m:
        radius_px = max(1, int(round(window_size_m / pixel_size)))
        window = 2 * radius_px + 1
        local_max = maximum_filter(smoothed, size=window, mode="nearest")
        local_mean = uniform_filter(smoothed, size=window, mode="nearest")
        mask = (smoothed >= local_max - 1e-6) & (smoothed >= min_height_m)
        peak_rows, peak_cols = np.where(mask)
        if peak_rows.size == 0:
            continue
        scores = np.clip(smoothed[peak_rows, peak_cols] - local_mean[peak_rows, peak_cols], 0.0, None)
        rows_all.append(peak_rows.astype(np.int32))
        cols_all.append(peak_cols.astype(np.int32))
        scores_all.append(scores.astype(np.float32))
    if not rows_all:
        return HardNegativeCandidates(
            rows=np.empty(0, dtype=np.int32),
            cols=np.empty(0, dtype=np.int32),
            scores=np.empty(0, dtype=np.float32),
        )
    return HardNegativeCandidates(
        rows=np.concatenate(rows_all),
        cols=np.concatenate(cols_all),
        scores=np.concatenate(scores_all),
    )


def compute_hard_negative_mask(
    reference_table: pd.DataFrame,
    transform: GeoTransform,
    chm: np.ndarray,
    rgb: np.ndarray | None,
    window_sizes_m: tuple[float, ...] | list[float],
    match_radius_m: float,
    min_height_m: float,
    border_margin_px: int,
    dilation_px: int,
    rgb_complexity_quantile: float,
) -> np.ndarray:
    height, width = chm.shape
    mask = np.zeros((height, width), dtype=bool)
    candidates = detect_multiscale_hard_negative_candidates(
        chm=chm,
        pixel_size=transform.pixel_size,
        window_sizes_m=window_sizes_m,
        min_height_m=min_height_m,
    )

    reference_xy = reference_table[["x", "y"]].to_numpy(dtype=float) if len(reference_table) else np.empty((0, 2), dtype=float)
    for row, col, score in zip(candidates.rows, candidates.cols, candidates.scores):
        if score <= 0.0:
            continue
        x, y = transform.rowcol_to_xy(int(row), int(col))
        point = np.array([x, y], dtype=float)
        near_reference = False
        if reference_xy.size:
            distances = np.sqrt(((reference_xy - point[None, :]) ** 2).sum(axis=1))
            near_reference = bool(np.any(distances <= match_radius_m))
        if not near_reference:
            mask[int(row), int(col)] = True

    if rgb is not None and rgb.size:
        rgb_complexity = compute_rgb_complexity(rgb)
        threshold = float(np.quantile(rgb_complexity, rgb_complexity_quantile))
        complex_mask = (rgb_complexity >= threshold) & (chm >= min_height_m)
        if reference_xy.size:
            exclusion = np.zeros_like(mask)
            exclusion_radius_px = max(1, int(round(match_radius_m / transform.pixel_size)))
            for _, tree in reference_table.iterrows():
                tree_row, tree_col = transform.xy_to_rowcol(float(tree["x"]), float(tree["y"]))
                if 0 <= tree_row < height and 0 <= tree_col < width:
                    exclusion[tree_row, tree_col] = True
            exclusion = binary_dilation(exclusion, iterations=exclusion_radius_px)
            complex_mask &= ~exclusion
        mask |= complex_mask

    if border_margin_px > 0:
        mask[:border_margin_px, :] = True
        mask[-border_margin_px:, :] = True
        mask[:, :border_margin_px] = True
        mask[:, -border_margin_px:] = True

    if len(reference_table):
        positive_mask = np.zeros_like(mask)
        for _, tree in reference_table.iterrows():
            tree_row, tree_col = transform.xy_to_rowcol(float(tree["x"]), float(tree["y"]))
            if 0 <= tree_row < height and 0 <= tree_col < width:
                positive_mask[tree_row, tree_col] = True
        positive_mask = binary_dilation(positive_mask, iterations=max(1, dilation_px))
        mask &= ~positive_mask

    mask = binary_dilation(mask, iterations=max(1, dilation_px))
    return mask.astype(np.float32)

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from lidaragrosat.utils.geo import Bounds


@dataclass(slots=True)
class QAReport:
    plot_id: str
    n_points_raw: int
    n_points: int
    area_m2: float
    lidar_density_pts_m2: float
    n_reference_trees_raw: int
    n_reference_trees: int
    n_reference_trees_dropped: int
    labels_inside_bounds_fraction: float
    height_outlier_count: int
    estimated_shift_px_x: float | None = None
    estimated_shift_px_y: float | None = None
    applied_shift_px_x: float | None = None
    applied_shift_px_y: float | None = None
    residual_shift_px_x: float | None = None
    residual_shift_px_y: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fraction_points_inside_bounds(reference_table, bounds: Bounds) -> float:
    inside_mask = (
        (reference_table["x"] >= bounds.xmin)
        & (reference_table["x"] <= bounds.xmax)
        & (reference_table["y"] >= bounds.ymin)
        & (reference_table["y"] <= bounds.ymax)
    )
    return float(inside_mask.mean()) if len(reference_table) else 1.0


def height_outliers(reference_table, lower_q: float = 0.01, upper_q: float = 0.99) -> int:
    if "height" not in reference_table.columns:
        return 0
    heights = reference_table["height"].dropna().astype(float).to_numpy()
    if heights.size < 4:
        return 0
    lo, hi = np.quantile(heights, [lower_q, upper_q])
    return int(((heights < lo) | (heights > hi)).sum())


def estimate_shift_pixels(chm: np.ndarray, rgb: np.ndarray, max_shift_px: int = 8) -> tuple[float, float]:
    from scipy.ndimage import sobel
    from scipy.signal import fftconvolve

    rgb_gray = rgb.mean(axis=0) if rgb.ndim == 3 else rgb
    downsample = max(1, int(np.ceil(max(chm.shape) / 256)))
    if downsample > 1:
        chm = chm[::downsample, ::downsample]
        rgb_gray = rgb_gray[::downsample, ::downsample]

    chm_edge = np.hypot(sobel(chm, axis=0), sobel(chm, axis=1))
    rgb_edge = np.hypot(sobel(rgb_gray, axis=0), sobel(rgb_gray, axis=1))
    chm_edge = (chm_edge - chm_edge.mean()) / max(chm_edge.std(), 1e-6)
    rgb_edge = (rgb_edge - rgb_edge.mean()) / max(rgb_edge.std(), 1e-6)
    correlation = fftconvolve(chm_edge, rgb_edge[::-1, ::-1], mode="same")
    center_row = correlation.shape[0] // 2
    center_col = correlation.shape[1] // 2
    scaled_shift = max(1, int(np.ceil(max_shift_px / downsample)))
    search = correlation[
        center_row - scaled_shift : center_row + scaled_shift + 1,
        center_col - scaled_shift : center_col + scaled_shift + 1,
    ]
    offset_row, offset_col = np.unravel_index(np.argmax(search), search.shape)
    return (
        float((offset_col - scaled_shift) * downsample),
        float((offset_row - scaled_shift) * downsample),
    )

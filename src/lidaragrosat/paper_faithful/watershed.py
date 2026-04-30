from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@dataclass(slots=True)
class FaithfulWatershedParams:
    resolution_m: float = 0.2
    gaussian_sigma_m: float = 0.8
    min_distance_m: float = 1.8
    min_height_m: float = 5.0
    min_segment_area_m2: float = 1.0
    hole_fill_radius_m: float = 0.4
    compactness: float = 0.0


@dataclass(slots=True)
class FaithfulWatershedDetails:
    predictions: pd.DataFrame
    segments: np.ndarray
    smooth_chm: np.ndarray
    chm: np.ndarray
    xmin: float
    ymax: float
    resolution_m: float


def build_chm(
    points_xyz: np.ndarray,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool), 0.0, 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    xmin = float(np.min(x))
    ymax = float(np.max(y))

    col = np.floor((x - xmin) / float(resolution_m)).astype(np.int64)
    row = np.floor((ymax - y) / float(resolution_m)).astype(np.int64)
    width = int(np.max(col)) + 1
    height = int(np.max(row)) + 1
    flat = row * width + col
    size = height * width

    chm_flat = np.zeros(size, dtype=np.float32)
    np.maximum.at(chm_flat, flat, z.astype(np.float32))
    count_flat = np.bincount(flat, minlength=size)
    occupancy = count_flat.reshape(height, width) > 0
    chm = chm_flat.reshape(height, width)
    chm[~occupancy] = np.nan
    return chm, occupancy, xmin, ymax


def _fill_small_holes(
    chm: np.ndarray,
    occupancy: np.ndarray,
    resolution_m: float,
    hole_fill_radius_m: float,
) -> np.ndarray:
    if chm.size == 0:
        return chm.astype(np.float32)
    valid = np.isfinite(chm)
    if valid.all():
        return np.asarray(chm, dtype=np.float32)

    nearest_indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = np.asarray(chm[tuple(nearest_indices)], dtype=np.float32)
    distance_to_valid = distance_transform_edt(~valid) * float(resolution_m)
    filled[distance_to_valid > float(hole_fill_radius_m)] = 0.0
    filled[valid] = np.asarray(chm[valid], dtype=np.float32)

    distance_to_occupancy = distance_transform_edt(~occupancy) * float(resolution_m)
    filled[distance_to_occupancy > float(hole_fill_radius_m)] = 0.0
    return filled


def chm_watershed_details(
    points_xyz: np.ndarray,
    params: FaithfulWatershedParams | None = None,
) -> FaithfulWatershedDetails:
    params = params or FaithfulWatershedParams()
    chm, occupancy, xmin, ymax = build_chm(points_xyz, resolution_m=float(params.resolution_m))
    if chm.size == 0:
        empty = pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "crown_area_m2"])
        return FaithfulWatershedDetails(
            predictions=empty,
            segments=np.zeros((0, 0), dtype=np.int32),
            smooth_chm=np.zeros((0, 0), dtype=np.float32),
            chm=np.zeros((0, 0), dtype=np.float32),
            xmin=0.0,
            ymax=0.0,
            resolution_m=float(params.resolution_m),
        )

    chm_filled = _fill_small_holes(
        chm=chm,
        occupancy=occupancy,
        resolution_m=float(params.resolution_m),
        hole_fill_radius_m=float(params.hole_fill_radius_m),
    )
    sigma_px = max(float(params.gaussian_sigma_m) / float(params.resolution_m), 0.0)
    smooth = gaussian_filter(chm_filled, sigma=sigma_px, mode="nearest") if sigma_px > 0 else chm_filled
    canopy_mask = smooth >= float(params.min_height_m)
    if not np.any(canopy_mask):
        empty = pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "crown_area_m2"])
        return FaithfulWatershedDetails(
            predictions=empty,
            segments=np.zeros_like(smooth, dtype=np.int32),
            smooth_chm=smooth.astype(np.float32),
            chm=np.asarray(chm, dtype=np.float32),
            xmin=xmin,
            ymax=ymax,
            resolution_m=float(params.resolution_m),
        )

    min_distance_px = max(1, int(round(float(params.min_distance_m) / float(params.resolution_m))))
    peak_coords = peak_local_max(
        smooth,
        min_distance=min_distance_px,
        threshold_abs=float(params.min_height_m),
        labels=canopy_mask.astype(np.uint8),
        exclude_border=False,
    )
    if peak_coords.size == 0:
        empty = pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "crown_area_m2"])
        return FaithfulWatershedDetails(
            predictions=empty,
            segments=np.zeros_like(smooth, dtype=np.int32),
            smooth_chm=smooth.astype(np.float32),
            chm=np.asarray(chm, dtype=np.float32),
            xmin=xmin,
            ymax=ymax,
            resolution_m=float(params.resolution_m),
        )

    markers = np.zeros_like(smooth, dtype=np.int32)
    for marker_id, (row, col) in enumerate(peak_coords, start=1):
        markers[int(row), int(col)] = int(marker_id)

    segments = watershed(
        -smooth,
        markers=markers,
        mask=canopy_mask,
        compactness=float(params.compactness),
    )
    min_segment_pixels = max(1, int(round(float(params.min_segment_area_m2) / (float(params.resolution_m) ** 2))))

    records: list[dict] = []
    for label_id in range(1, int(segments.max()) + 1):
        segment_mask = segments == label_id
        area_px = int(segment_mask.sum())
        if area_px < min_segment_pixels:
            continue
        segment_heights = np.where(segment_mask, np.nan_to_num(chm, nan=-np.inf), -np.inf)
        if not np.isfinite(segment_heights).any():
            continue
        max_flat = int(np.argmax(segment_heights))
        row, col = np.unravel_index(max_flat, chm.shape)
        height_pred = float(chm[row, col])
        if not np.isfinite(height_pred) or height_pred < float(params.min_height_m):
            continue
        x = float(xmin + (col + 0.5) * float(params.resolution_m))
        y = float(ymax - (row + 0.5) * float(params.resolution_m))
        records.append(
            {
                "x": x,
                "y": y,
                "height_pred": height_pred,
                "confidence": float(smooth[row, col]),
                "crown_area_m2": float(area_px * (float(params.resolution_m) ** 2)),
            }
        )

    if not records:
        predictions = pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "crown_area_m2"])
    else:
        predictions = pd.DataFrame.from_records(records).sort_values(["y", "x"]).reset_index(drop=True)
    return FaithfulWatershedDetails(
        predictions=predictions,
        segments=segments.astype(np.int32),
        smooth_chm=smooth.astype(np.float32),
        chm=np.asarray(chm, dtype=np.float32),
        xmin=xmin,
        ymax=ymax,
        resolution_m=float(params.resolution_m),
    )


def chm_watershed_filter(
    points_xyz: np.ndarray,
    params: FaithfulWatershedParams | None = None,
) -> pd.DataFrame:
    return chm_watershed_details(points_xyz, params=params).predictions

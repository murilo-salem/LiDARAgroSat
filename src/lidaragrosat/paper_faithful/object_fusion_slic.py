from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage.segmentation import slic

from .object_fusion import (
    FaithfulObjectFusionParams,
    _deduplicate_candidates,
    _normalize_rgb_for_segmentation,
    _source_candidates,
    fuse_candidates_with_segments,
)


@dataclass(slots=True)
class FaithfulObjectFusionSLICParams(FaithfulObjectFusionParams):
    n_segments_low: int = 300
    compactness_low: float = 10.0
    sigma_low: float = 0.8
    n_segments_high: int = 700
    compactness_high: float = 20.0
    sigma_high: float = 1.0
    multiscale_merge_radius_m: float = 1.0


def segment_rgb_slic(
    rgb: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
) -> np.ndarray:
    rgb_normalized = _normalize_rgb_for_segmentation(rgb)
    return slic(
        rgb_normalized,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=1,
        channel_axis=-1,
    ).astype(np.int32)


def fuse_lidar_rgb_slic_multiscale(
    points_xyz: np.ndarray,
    rgb: np.ndarray,
    chm: np.ndarray,
    x_min: float,
    y_max: float,
    pixel_size: float,
    params: FaithfulObjectFusionSLICParams | None = None,
) -> pd.DataFrame:
    params = params or FaithfulObjectFusionSLICParams()
    candidates = _source_candidates(points_xyz, params)
    if candidates.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "boundary_distance_m", "segment_area_m2", "segment_chm_max_m"])

    low_segments = segment_rgb_slic(
        rgb=rgb,
        n_segments=int(params.n_segments_low),
        compactness=float(params.compactness_low),
        sigma=float(params.sigma_low),
    )
    high_segments = segment_rgb_slic(
        rgb=rgb,
        n_segments=int(params.n_segments_high),
        compactness=float(params.compactness_high),
        sigma=float(params.sigma_high),
    )
    low = fuse_candidates_with_segments(
        candidates=candidates,
        segments=low_segments,
        chm=chm,
        x_min=x_min,
        y_max=y_max,
        pixel_size=pixel_size,
        params=params,
    )
    high = fuse_candidates_with_segments(
        candidates=candidates,
        segments=high_segments,
        chm=chm,
        x_min=x_min,
        y_max=y_max,
        pixel_size=pixel_size,
        params=params,
    )
    if low.empty and high.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "boundary_distance_m", "segment_area_m2", "segment_chm_max_m"])
    merged = pd.concat([low, high], ignore_index=True, sort=False)
    if "confidence" not in merged.columns:
        merged["confidence"] = merged["height_pred"].astype(float)
    merged["confidence"] = merged["confidence"].astype(float) + 0.05 * merged["height_pred"].astype(float)
    dedup = _deduplicate_candidates(merged, radius_m=float(params.multiscale_merge_radius_m))
    return dedup.sort_values(["y", "x"]).reset_index(drop=True)

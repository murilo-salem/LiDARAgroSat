from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import felzenszwalb

from .adaptive_local_maxima import FaithfulAdaptiveBaselineParams, adaptive_local_maxima_filter_3d
from .watershed import FaithfulWatershedParams, chm_watershed_filter


BEST_ADAPTIVE_PARAMS = FaithfulAdaptiveBaselineParams(
    a=0.035,
    b=2.5,
    height_threshold_m=5.0,
    min_window_m=0.5,
    max_window_m=5.0,
)

BEST_WATERSHED_PARAMS = FaithfulWatershedParams(
    resolution_m=0.1,
    gaussian_sigma_m=0.2,
    min_distance_m=0.8,
    min_height_m=5.9,
    min_segment_area_m2=6.0,
    hole_fill_radius_m=0.3,
    compactness=0.0075,
)


@dataclass(slots=True)
class FaithfulObjectFusionParams:
    source_mode: str = "union"
    felzenszwalb_scale: float = 150.0
    felzenszwalb_sigma: float = 0.8
    felzenszwalb_min_size: int = 96
    min_segment_area_m2: float = 2.0
    max_segment_area_m2: float = 120.0
    area_per_tree_m2: float = 18.0
    min_boundary_distance_m: float = 0.4
    min_segment_chm_max_m: float = 5.0
    candidate_merge_radius_m: float = 0.8


def _normalize_rgb_for_segmentation(rgb: np.ndarray) -> np.ndarray:
    rgb_cf = np.asarray(rgb, dtype=np.float32)
    if rgb_cf.ndim != 3 or rgb_cf.shape[0] != 3:
        raise ValueError(f"Expected RGB channels-first tensor with shape (3,H,W), got {rgb_cf.shape}")
    rgb_hwc = np.moveaxis(rgb_cf, 0, -1)
    lo = np.percentile(rgb_hwc, 1.0, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb_hwc, 99.0, axis=(0, 1), keepdims=True)
    scaled = (rgb_hwc - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def segment_rgb_objects(
    rgb: np.ndarray,
    scale: float,
    sigma: float,
    min_size: int,
) -> np.ndarray:
    rgb_normalized = _normalize_rgb_for_segmentation(rgb)
    return felzenszwalb(
        rgb_normalized,
        scale=float(scale),
        sigma=float(sigma),
        min_size=int(min_size),
        channel_axis=-1,
    ).astype(np.int32)


def _xy_to_rowcol(x: np.ndarray, y: np.ndarray, x_min: float, y_max: float, pixel_size: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    col = np.clip(np.floor((x - x_min) / pixel_size).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor((y_max - y) / pixel_size).astype(np.int64), 0, height - 1)
    return row, col


def _segment_statistics(segments: np.ndarray, chm: np.ndarray, pixel_size: float) -> pd.DataFrame:
    flat_segments = segments.reshape(-1)
    flat_chm = np.asarray(chm, dtype=np.float32).reshape(-1)
    labels, counts = np.unique(flat_segments, return_counts=True)
    area_m2 = counts.astype(np.float32) * (pixel_size**2)

    max_chm = np.full(int(labels.max()) + 1, -np.inf, dtype=np.float32)
    np.maximum.at(max_chm, flat_segments, np.nan_to_num(flat_chm, nan=0.0))

    records = []
    for label, count, area in zip(labels.tolist(), counts.tolist(), area_m2.tolist()):
        records.append(
            {
                "segment_id": int(label),
                "pixel_count": int(count),
                "area_m2": float(area),
                "chm_max_m": float(max_chm[int(label)]),
            }
        )
    return pd.DataFrame.from_records(records)


def _segment_boundary_distance_map(segments: np.ndarray, pixel_size: float) -> np.ndarray:
    boundary = np.zeros_like(segments, dtype=bool)
    boundary[:-1, :] |= segments[:-1, :] != segments[1:, :]
    boundary[1:, :] |= segments[1:, :] != segments[:-1, :]
    boundary[:, :-1] |= segments[:, :-1] != segments[:, 1:]
    boundary[:, 1:] |= segments[:, 1:] != segments[:, :-1]
    return distance_transform_edt(~boundary).astype(np.float32) * float(pixel_size)


def _deduplicate_candidates(candidates: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ordered = candidates.sort_values(["height_pred", "confidence"], ascending=[False, False]).reset_index(drop=True)
    kept: list[dict] = []
    taken = np.zeros(len(ordered), dtype=bool)
    xy = ordered[["x", "y"]].to_numpy(dtype=np.float64)
    radius_sq = float(radius_m) ** 2
    for i in range(len(ordered)):
        if taken[i]:
            continue
        kept.append(ordered.iloc[i].to_dict())
        dxy = xy - xy[i]
        dist_sq = np.sum(dxy * dxy, axis=1)
        taken |= dist_sq <= radius_sq
    return pd.DataFrame.from_records(kept)


def _source_candidates(points_xyz: np.ndarray, params: FaithfulObjectFusionParams) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if params.source_mode in {"adaptive", "union"}:
        adaptive = adaptive_local_maxima_filter_3d(points_xyz, BEST_ADAPTIVE_PARAMS).copy()
        if not adaptive.empty:
            adaptive["source"] = "adaptive"
            adaptive["confidence"] = adaptive["height_pred"].astype(float)
            frames.append(adaptive)
    if params.source_mode in {"watershed", "union"}:
        watershed = chm_watershed_filter(points_xyz, BEST_WATERSHED_PARAMS).copy()
        if not watershed.empty:
            watershed["source"] = "watershed"
            if "confidence" not in watershed.columns:
                watershed["confidence"] = watershed["height_pred"].astype(float)
            frames.append(watershed)
    if not frames:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source"])
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "confidence" not in merged.columns:
        merged["confidence"] = merged["height_pred"].astype(float)
    return merged


def generate_source_candidates(points_xyz: np.ndarray, source_mode: str) -> pd.DataFrame:
    return _source_candidates(points_xyz, FaithfulObjectFusionParams(source_mode=source_mode))


def fuse_candidates_with_segments(
    candidates: pd.DataFrame,
    segments: np.ndarray,
    chm: np.ndarray,
    x_min: float,
    y_max: float,
    pixel_size: float,
    params: FaithfulObjectFusionParams | None = None,
) -> pd.DataFrame:
    params = params or FaithfulObjectFusionParams()
    if candidates.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "boundary_distance_m", "segment_area_m2", "segment_chm_max_m"])

    segment_table = _segment_statistics(segments, chm, pixel_size=float(pixel_size))
    boundary_distance = _segment_boundary_distance_map(segments, pixel_size=float(pixel_size))

    height, width = chm.shape
    rows, cols = _xy_to_rowcol(
        candidates["x"].to_numpy(dtype=np.float64),
        candidates["y"].to_numpy(dtype=np.float64),
        x_min=float(x_min),
        y_max=float(y_max),
        pixel_size=float(pixel_size),
        height=height,
        width=width,
    )
    segment_ids = segments[rows, cols]
    candidates = candidates.copy()
    candidates["segment_id"] = segment_ids.astype(np.int32)
    candidates["boundary_distance_m"] = boundary_distance[rows, cols].astype(np.float32)

    candidates = candidates.merge(segment_table, on="segment_id", how="left")
    candidates = candidates.rename(columns={"area_m2": "segment_area_m2", "chm_max_m": "segment_chm_max_m"})
    candidates = candidates[
        (candidates["segment_area_m2"] >= float(params.min_segment_area_m2))
        & (candidates["segment_area_m2"] <= float(params.max_segment_area_m2))
        & (candidates["segment_chm_max_m"] >= float(params.min_segment_chm_max_m))
        & (candidates["boundary_distance_m"] >= float(params.min_boundary_distance_m))
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "boundary_distance_m", "segment_area_m2", "segment_chm_max_m"])

    candidates["rank_score"] = candidates["confidence"].astype(float) + 0.25 * candidates["height_pred"].astype(float)
    selected_frames: list[pd.DataFrame] = []
    for _, group in candidates.groupby("segment_id", sort=False):
        group = _deduplicate_candidates(group, radius_m=float(params.candidate_merge_radius_m))
        max_keep = max(1, int(round(float(group["segment_area_m2"].iloc[0]) / float(params.area_per_tree_m2))))
        keep = group.sort_values("rank_score", ascending=False).head(max_keep)
        selected_frames.append(keep)

    if not selected_frames:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "boundary_distance_m", "segment_area_m2", "segment_chm_max_m"])
    result = pd.concat(selected_frames, ignore_index=True, sort=False)
    return result.sort_values(["y", "x"]).reset_index(drop=True)


def fuse_lidar_rgb_objects(
    points_xyz: np.ndarray,
    rgb: np.ndarray,
    chm: np.ndarray,
    x_min: float,
    y_max: float,
    pixel_size: float,
    params: FaithfulObjectFusionParams | None = None,
) -> pd.DataFrame:
    params = params or FaithfulObjectFusionParams()
    candidates = _source_candidates(points_xyz, params)
    if candidates.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "boundary_distance_m", "segment_area_m2", "segment_chm_max_m"])

    segments = segment_rgb_objects(
        rgb=rgb,
        scale=float(params.felzenszwalb_scale),
        sigma=float(params.felzenszwalb_sigma),
        min_size=int(params.felzenszwalb_min_size),
    )
    return fuse_candidates_with_segments(
        candidates=candidates,
        segments=segments,
        chm=chm,
        x_min=x_min,
        y_max=y_max,
        pixel_size=pixel_size,
        params=params,
    )


@lru_cache(maxsize=128)
def cached_segment_rgb_objects(rgb_path: str, scale: float, sigma: float, min_size: int) -> np.ndarray:
    rgb = np.load(Path(rgb_path))
    return segment_rgb_objects(rgb=rgb, scale=scale, sigma=sigma, min_size=min_size)

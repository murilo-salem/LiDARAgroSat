from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lidaragrosat.utils.dependencies import require_dependency
from lidaragrosat.utils.geo import Bounds, GeoTransform, bounds_from_points, raster_shape


@dataclass(slots=True)
class LidarPointCloud:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    intensity: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.x.size)

    @property
    def bounds(self) -> Bounds:
        return bounds_from_points(self.x, self.y)


def read_lidar(path: str | Path) -> LidarPointCloud:
    laspy = require_dependency("laspy", "LiDAR reading")
    cloud = laspy.read(path)
    intensity = getattr(cloud, "intensity", None)
    return LidarPointCloud(
        x=np.asarray(cloud.x, dtype=np.float32),
        y=np.asarray(cloud.y, dtype=np.float32),
        z=np.asarray(cloud.z, dtype=np.float32),
        intensity=np.asarray(intensity, dtype=np.float32) if intensity is not None else None,
    )


def filter_noise(cloud: LidarPointCloud, lower_q: float, upper_q: float) -> LidarPointCloud:
    z_lo, z_hi = np.quantile(cloud.z, [lower_q, upper_q])
    mask = (cloud.z >= z_lo) & (cloud.z <= z_hi)
    intensity = cloud.intensity[mask] if cloud.intensity is not None else None
    return LidarPointCloud(cloud.x[mask], cloud.y[mask], cloud.z[mask], intensity)


def _flat_indices(
    x: np.ndarray,
    y: np.ndarray,
    bounds: Bounds,
    resolution: float,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    height, width = raster_shape(bounds, resolution)
    col = np.floor((x - bounds.xmin) / resolution).astype(np.int64)
    row = np.floor((bounds.ymax - y) / resolution).astype(np.int64)
    valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    flat = row[valid] * width + col[valid]
    return flat, (height, width), valid


def _group_quantile(flat: np.ndarray, values: np.ndarray, size: int, quantile: float) -> np.ndarray:
    out = np.full(size, np.nan, dtype=np.float32)
    if flat.size == 0:
        return out
    order = np.lexsort((values, flat))
    flat_sorted = flat[order]
    values_sorted = values[order]
    unique, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
    rank_offsets = np.floor((counts - 1) * quantile).astype(np.int64)
    rank_indices = start_idx + rank_offsets
    out[unique] = values_sorted[rank_indices].astype(np.float32)
    return out


def interpolate_missing_grid(grid: np.ndarray) -> np.ndarray:
    scipy_interpolate = require_dependency("scipy.interpolate", "DTM interpolation")
    rows, cols = np.indices(grid.shape)
    valid = ~np.isnan(grid)
    if valid.all():
        return grid
    known_points = np.column_stack([rows[valid], cols[valid]])
    known_values = grid[valid]
    missing_points = np.column_stack([rows[~valid], cols[~valid]])
    linear = scipy_interpolate.griddata(
        known_points,
        known_values,
        missing_points,
        method="linear",
    )
    nearest = scipy_interpolate.griddata(
        known_points,
        known_values,
        missing_points,
        method="nearest",
    )
    filled = grid.copy()
    fill_values = np.where(np.isnan(linear), nearest, linear).astype(np.float32)
    filled[~valid] = fill_values
    return filled


def build_dtm(
    cloud: LidarPointCloud,
    bounds: Bounds,
    cell_size: float,
    ground_quantile: float,
) -> tuple[np.ndarray, GeoTransform]:
    flat, (height, width), valid = _flat_indices(cloud.x, cloud.y, bounds, cell_size)
    quantiles = _group_quantile(flat, cloud.z[valid], height * width, ground_quantile).reshape(height, width)
    dtm = interpolate_missing_grid(quantiles)
    transform = GeoTransform(x_min=bounds.xmin, y_max=bounds.ymax, pixel_size=cell_size)
    return dtm, transform


def sample_grid(grid: np.ndarray, bounds: Bounds, resolution: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    col = np.clip(((x - bounds.xmin) / resolution).astype(np.int64), 0, width - 1)
    row = np.clip(((bounds.ymax - y) / resolution).astype(np.int64), 0, height - 1)
    return grid[row, col]


def normalize_heights(
    cloud: LidarPointCloud,
    dtm: np.ndarray,
    dtm_bounds: Bounds,
    dtm_resolution: float,
) -> np.ndarray:
    ground_z = sample_grid(dtm, dtm_bounds, dtm_resolution, cloud.x, cloud.y)
    normalized = cloud.z - ground_z
    return normalized.astype(np.float32)


def rasterize_structural_channels(
    cloud: LidarPointCloud,
    normalized_z: np.ndarray,
    bounds: Bounds,
    resolution: float,
    channel_names: list[str],
) -> tuple[np.ndarray, GeoTransform]:
    from scipy.ndimage import uniform_filter

    flat, (height, width), valid = _flat_indices(cloud.x, cloud.y, bounds, resolution)
    size = height * width
    z = normalized_z[valid]
    intensity = cloud.intensity[valid] if cloud.intensity is not None else None

    count = np.bincount(flat, minlength=size).astype(np.float32)
    sum_z = np.bincount(flat, weights=z, minlength=size).astype(np.float32)
    sum_sq_z = np.bincount(flat, weights=z * z, minlength=size).astype(np.float32)

    zmax = np.full(size, 0.0, dtype=np.float32)
    np.maximum.at(zmax, flat, z)

    zmean = np.divide(sum_z, np.maximum(count, 1), dtype=np.float32)
    variance = np.divide(sum_sq_z, np.maximum(count, 1), dtype=np.float32) - zmean**2
    zstd = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)
    zp95 = _group_quantile(flat, z, size, 0.95)
    zp95 = np.nan_to_num(zp95, nan=0.0)

    density_total = count / (resolution**2)
    density_0p5_2 = np.bincount(flat[z >= 0.5], weights=((z[z >= 0.5] < 2.0)).astype(np.float32), minlength=size)
    density_2_5 = np.bincount(flat[(z >= 2.0) & (z < 5.0)], minlength=size).astype(np.float32)
    density_5_10 = np.bincount(flat[(z >= 5.0) & (z < 10.0)], minlength=size).astype(np.float32)
    density_10_20 = np.bincount(flat[(z >= 10.0) & (z < 20.0)], minlength=size).astype(np.float32)
    density_gt20 = np.bincount(flat[z >= 20.0], minlength=size).astype(np.float32)

    cover_gt2 = np.divide(
        np.bincount(flat[z >= 2.0], minlength=size).astype(np.float32),
        np.maximum(count, 1),
        dtype=np.float32,
    )
    cover_gt5 = np.divide(
        np.bincount(flat[z >= 5.0], minlength=size).astype(np.float32),
        np.maximum(count, 1),
        dtype=np.float32,
    )

    zmax_grid = zmax.reshape(height, width)
    local_mean = uniform_filter(zmax_grid, size=3, mode="nearest")
    local_mean_sq = uniform_filter(zmax_grid**2, size=3, mode="nearest")
    roughness = np.sqrt(np.clip(local_mean_sq - local_mean**2, 0.0, None)).astype(np.float32)

    channel_map: dict[str, np.ndarray] = {
        "zmax": zmax_grid,
        "zmean": zmean.reshape(height, width),
        "zstd": zstd.reshape(height, width),
        "zp95": zp95.reshape(height, width),
        "density_total": density_total.reshape(height, width),
        "density_0p5_2": density_0p5_2.reshape(height, width),
        "density_2_5": density_2_5.reshape(height, width),
        "density_5_10": density_5_10.reshape(height, width),
        "density_10_20": density_10_20.reshape(height, width),
        "density_gt20": density_gt20.reshape(height, width),
        "cover_gt2": cover_gt2.reshape(height, width),
        "cover_gt5": cover_gt5.reshape(height, width),
        "vertical_roughness": roughness,
    }
    if intensity is not None:
        mean_intensity = np.divide(
            np.bincount(flat, weights=intensity, minlength=size).astype(np.float32),
            np.maximum(count, 1),
            dtype=np.float32,
        )
        channel_map["intensity"] = mean_intensity.reshape(height, width)

    ordered = [channel_map[name] for name in channel_names if name in channel_map]
    stacked = np.stack(ordered, axis=0).astype(np.float32)
    transform = GeoTransform(x_min=bounds.xmin, y_max=bounds.ymax, pixel_size=resolution)
    return stacked, transform


def compute_plot_bounds(cloud: LidarPointCloud) -> Bounds:
    return cloud.bounds

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidaragrosat.utils.dependencies import require_dependency
from lidaragrosat.utils.geo import GeoTransform


def read_rgb(rgb_path: str | Path) -> tuple[np.ndarray, dict]:
    rasterio = require_dependency("rasterio", "RGB reading")
    with rasterio.open(rgb_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def normalize_rgb(rgb: np.ndarray, clip_percentiles: tuple[float, float], standardize: bool) -> np.ndarray:
    lower, upper = clip_percentiles
    normalized = rgb.astype(np.float32).copy()
    for channel in range(normalized.shape[0]):
        lo, hi = np.percentile(normalized[channel], [lower, upper])
        normalized[channel] = np.clip(normalized[channel], lo, hi)
        normalized[channel] = (normalized[channel] - lo) / max(hi - lo, 1e-6)
        if standardize:
            mean = normalized[channel].mean()
            std = normalized[channel].std()
            normalized[channel] = (normalized[channel] - mean) / max(std, 1e-6)
    return normalized


def geotransform_to_rasterio(transform: GeoTransform):
    from affine import Affine

    return Affine(transform.pixel_size, 0.0, transform.x_min, 0.0, -transform.pixel_size, transform.y_max)


def resample_rgb_to_grid(
    rgb_path: str | Path,
    transform: GeoTransform,
    height: int,
    width: int,
) -> np.ndarray:
    rasterio = require_dependency("rasterio", "RGB resampling")
    from rasterio.warp import Resampling, reproject

    dst = np.zeros((3, height, width), dtype=np.float32)
    with rasterio.open(rgb_path) as src:
        for band_index in range(1, min(src.count, 3) + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=dst[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=geotransform_to_rasterio(transform),
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
    return dst


def shift_raster_pixels(array: np.ndarray, shift_x_px: int, shift_y_px: int) -> np.ndarray:
    shift_x_px = int(round(shift_x_px))
    shift_y_px = int(round(shift_y_px))
    if shift_x_px == 0 and shift_y_px == 0:
        return array.astype(np.float32).copy()
    if array.ndim == 2:
        pad_y = abs(shift_y_px)
        pad_x = abs(shift_x_px)
        padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        row_start = pad_y - shift_y_px
        col_start = pad_x - shift_x_px
        return padded[row_start : row_start + array.shape[0], col_start : col_start + array.shape[1]].astype(np.float32)

    pad_y = abs(shift_y_px)
    pad_x = abs(shift_x_px)
    padded = np.pad(array, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    row_start = pad_y - shift_y_px
    col_start = pad_x - shift_x_px
    return padded[:, row_start : row_start + array.shape[1], col_start : col_start + array.shape[2]].astype(np.float32)


def write_raster(
    array: np.ndarray,
    destination: str | Path,
    transform: GeoTransform,
    crs: str | None = None,
    dtype: str = "float32",
) -> None:
    rasterio = require_dependency("rasterio", "raster writing")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = array.shape[0] if array.ndim == 3 else 1
    height = array.shape[-2]
    width = array.shape[-1]
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=geotransform_to_rasterio(transform),
        compress="deflate",
    ) as dst:
        if array.ndim == 2:
            dst.write(array.astype(dtype), 1)
        else:
            dst.write(array.astype(dtype))

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from lidaragrosat.data.auxiliary import nearest_neighbor_distances
from lidaragrosat.data.dataset import ProcessedPlotBundle


@dataclass(slots=True)
class InstanceTargetMaps:
    distance_transform: np.ndarray
    seedness: np.ndarray
    pseudo_crown_mask: np.ndarray
    star_convex_rays: np.ndarray | None = None


def _crown_radius_m(bundle: ProcessedPlotBundle, row_index: int, fallback_height: float = 20.0) -> float:
    row = bundle.reference.iloc[row_index]
    tree_height = float(row["height"]) if row.get("height") == row.get("height") else float(fallback_height)
    nn = nearest_neighbor_distances(bundle.reference)
    radius = 0.8 + 0.04 * tree_height
    if row_index < len(nn) and np.isfinite(nn[row_index]):
        radius = min(radius, max(0.8, 0.45 * float(nn[row_index])))
    return float(np.clip(radius, 0.8, 2.5))


def build_instance_targets(
    bundle: ProcessedPlotBundle,
    n_rays: int = 8,
) -> InstanceTargetMaps:
    height, width = bundle.raster_shape
    pseudo_crown_mask = np.zeros((height, width), dtype=np.uint8)
    seedness = np.zeros((height, width), dtype=np.float32)
    star_rays = np.zeros((n_rays, height, width), dtype=np.float32)

    if bundle.reference.empty:
        return InstanceTargetMaps(
            distance_transform=np.zeros((height, width), dtype=np.float32),
            seedness=seedness,
            pseudo_crown_mask=pseudo_crown_mask,
            star_convex_rays=star_rays,
        )

    for row_index, (_, ref_row) in enumerate(bundle.reference.iterrows()):
        center_row, center_col = bundle.transform.xy_to_rowcol(float(ref_row["x"]), float(ref_row["y"]))
        if not (0 <= center_row < height and 0 <= center_col < width):
            continue
        radius_px = max(1, int(round(_crown_radius_m(bundle, row_index) / bundle.transform.pixel_size)))
        row_min = max(0, center_row - radius_px)
        row_max = min(height, center_row + radius_px + 1)
        col_min = max(0, center_col - radius_px)
        col_max = min(width, center_col + radius_px + 1)
        rr, cc = np.meshgrid(np.arange(row_min, row_max), np.arange(col_min, col_max), indexing="ij")
        distances = np.sqrt((rr - center_row) ** 2 + (cc - center_col) ** 2)
        crown = distances <= radius_px
        pseudo_crown_mask[row_min:row_max, col_min:col_max] = np.maximum(
            pseudo_crown_mask[row_min:row_max, col_min:col_max],
            crown.astype(np.uint8),
        )
        gaussian = np.exp(-(distances**2) / max(2 * (0.5 * radius_px) ** 2, 1e-6))
        seedness[row_min:row_max, col_min:col_max] = np.maximum(
            seedness[row_min:row_max, col_min:col_max],
            gaussian.astype(np.float32),
        )
        angles = np.linspace(0.0, 2.0 * np.pi, num=n_rays, endpoint=False)
        for ray_index, angle in enumerate(angles):
            ray_length = radius_px
            star_rays[ray_index, center_row, center_col] = float(ray_length * bundle.transform.pixel_size)

    distance_transform = distance_transform_edt(pseudo_crown_mask > 0).astype(np.float32)
    if np.any(distance_transform > 0):
        distance_transform = distance_transform / max(float(np.percentile(distance_transform, 99)), 1e-6)
    return InstanceTargetMaps(
        distance_transform=np.clip(distance_transform, 0.0, 1.0).astype(np.float32),
        seedness=np.clip(seedness, 0.0, 1.0).astype(np.float32),
        pseudo_crown_mask=pseudo_crown_mask.astype(np.float32),
        star_convex_rays=star_rays.astype(np.float32),
    )

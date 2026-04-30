from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidaragrosat.data.auxiliary import (
    build_species_lookup,
    compute_hard_negative_mask,
    compute_multiscale_peakness,
    encode_species_label,
    find_local_apex_pixel,
    nearest_neighbor_distances,
)
from lidaragrosat.utils.geo import GeoTransform


@dataclass(slots=True)
class TargetMaps:
    heatmap: np.ndarray
    offset: np.ndarray
    height: np.ndarray
    positive_mask: np.ndarray
    distance: np.ndarray
    difficulty: np.ndarray
    peakness: np.ndarray
    species: np.ndarray
    species_mask: np.ndarray
    hard_negative_mask: np.ndarray


def _gaussian_radius_pixels(sigma_m: float, resolution: float) -> int:
    sigma_px = sigma_m / resolution
    return max(1, int(np.ceil(3 * sigma_px)))


def build_center_targets(
    reference_table,
    transform: GeoTransform,
    raster_height: int,
    raster_width: int,
    sigma_m: float,
    chm: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    peakness_window_sizes_m: tuple[float, ...] | list[float] = (0.8, 1.2, 1.6),
    hard_negative_window_sizes_m: tuple[float, ...] | list[float] = (0.8, 1.2, 1.6),
    hard_negative_match_radius_m: float = 1.5,
    hard_negative_border_margin_px: int = 16,
    hard_negative_dilation_px: int = 3,
    hard_negative_rgb_complexity_quantile: float = 0.9,
    min_tree_height_m: float = 2.0,
    species_to_index: dict[str, int] | None = None,
    species_label_mode: str = "none",
    adaptive_sigma: bool = False,
    adaptive_sigma_use_nn_distance: bool = False,
    adaptive_sigma_nn_scale: float = 0.35,
    adaptive_sigma_a: float = 0.25,
    adaptive_sigma_b: float = 0.03,
    adaptive_sigma_min_m: float = 0.45,
    adaptive_sigma_max_m: float = 1.20,
    snap_centers_to_apex_radius_m: float = 0.0,
    apex_secondary_weight: float = 0.0,
    crown_radius_a: float = 0.8,
    crown_radius_b: float = 0.04,
    crown_radius_min_m: float = 0.8,
    crown_radius_max_m: float = 2.5,
) -> TargetMaps:
    heatmap = np.zeros((raster_height, raster_width), dtype=np.float32)
    offset = np.zeros((2, raster_height, raster_width), dtype=np.float32)
    height = np.full((raster_height, raster_width), np.nan, dtype=np.float32)
    positive_mask = np.zeros((raster_height, raster_width), dtype=np.float32)
    distance = np.zeros((raster_height, raster_width), dtype=np.float32)
    difficulty = np.zeros((raster_height, raster_width), dtype=np.float32)
    peakness = (
        compute_multiscale_peakness(
            chm=chm,
            pixel_size=transform.pixel_size,
            window_sizes_m=peakness_window_sizes_m,
            min_height_m=min_tree_height_m,
        )
        if chm is not None
        else np.zeros((raster_height, raster_width), dtype=np.float32)
    )
    hard_negative_mask = (
        compute_hard_negative_mask(
            reference_table=reference_table,
            transform=transform,
            chm=chm,
            rgb=rgb,
            window_sizes_m=hard_negative_window_sizes_m,
            match_radius_m=hard_negative_match_radius_m,
            min_height_m=min_tree_height_m,
            border_margin_px=hard_negative_border_margin_px,
            dilation_px=hard_negative_dilation_px,
            rgb_complexity_quantile=hard_negative_rgb_complexity_quantile,
        )
        if chm is not None
        else np.zeros((raster_height, raster_width), dtype=np.float32)
    )
    species = np.full((raster_height, raster_width), -1, dtype=np.int64)
    species_mask = np.zeros((raster_height, raster_width), dtype=np.float32)

    valid_heights = reference_table["height"].dropna().astype(float).to_numpy() if "height" in reference_table.columns else np.array([])
    fallback_height = float(np.median(valid_heights)) if valid_heights.size else 20.0
    nn_distances = nearest_neighbor_distances(reference_table)

    for tree_index, (_, row) in enumerate(reference_table.iterrows()):
        center_row, center_col = transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
        if center_row < 0 or center_row >= raster_height or center_col < 0 or center_col >= raster_width:
            continue

        snapped_row, snapped_col = center_row, center_col
        if snap_centers_to_apex_radius_m > 0 and chm is not None:
            snap_radius_px = max(1, int(round(snap_centers_to_apex_radius_m / transform.pixel_size)))
            snapped_row, snapped_col = find_local_apex_pixel(chm, center_row, center_col, snap_radius_px)

        current_sigma_m = sigma_m
        if adaptive_sigma:
            tree_height = row.get("height")
            if tree_height is None or tree_height != tree_height:
                tree_height = fallback_height
            current_sigma_m = float(
                np.clip(
                    adaptive_sigma_a + adaptive_sigma_b * float(tree_height),
                    adaptive_sigma_min_m,
                    adaptive_sigma_max_m,
                )
            )
        if adaptive_sigma_use_nn_distance and tree_index < len(nn_distances):
            nn_distance = nn_distances[tree_index]
            if np.isfinite(nn_distance):
                sigma_from_nn = float(
                    np.clip(
                        adaptive_sigma_nn_scale * float(nn_distance),
                        adaptive_sigma_min_m,
                        adaptive_sigma_max_m,
                    )
                )
                current_sigma_m = min(current_sigma_m, sigma_from_nn) if adaptive_sigma else sigma_from_nn
        radius = _gaussian_radius_pixels(current_sigma_m, transform.pixel_size)
        sigma_px = current_sigma_m / transform.pixel_size

        target_centers: list[tuple[int, int, float]] = []
        if snap_centers_to_apex_radius_m > 0 and apex_secondary_weight <= 0:
            target_centers.append((snapped_row, snapped_col, 1.0))
        else:
            target_centers.append((center_row, center_col, 1.0))
            if snap_centers_to_apex_radius_m > 0 and apex_secondary_weight > 0:
                target_centers.append((snapped_row, snapped_col, float(apex_secondary_weight)))

        tree_height = row.get("height")
        if tree_height is None or tree_height != tree_height:
            tree_height = fallback_height
        crown_radius_m = float(
            np.clip(
                crown_radius_a + crown_radius_b * float(tree_height),
                crown_radius_min_m,
                crown_radius_max_m,
            )
        )
        if tree_index < len(nn_distances) and np.isfinite(nn_distances[tree_index]):
            crown_radius_m = min(crown_radius_m, max(crown_radius_min_m, 0.45 * float(nn_distances[tree_index])))
        crown_radius_px = max(1, int(round(crown_radius_m / transform.pixel_size)))
        local_roughness = float(peakness[center_row, center_col]) if peakness.size else 0.0
        low_height_score = float(np.clip((fallback_height - float(tree_height)) / max(fallback_height, 1e-6), 0.0, 1.0))
        density_score = 0.0
        if tree_index < len(nn_distances) and np.isfinite(nn_distances[tree_index]):
            density_score = float(np.clip(1.0 - (float(nn_distances[tree_index]) / 6.0), 0.0, 1.0))
        difficulty_value = 1.0 + 0.75 * density_score + 0.5 * low_height_score + 0.5 * local_roughness

        for target_row, target_col, target_weight in target_centers:
            row_min = max(0, target_row - radius)
            row_max = min(raster_height, target_row + radius + 1)
            col_min = max(0, target_col - radius)
            col_max = min(raster_width, target_col + radius + 1)
            rr, cc = np.meshgrid(
                np.arange(row_min, row_max),
                np.arange(col_min, col_max),
                indexing="ij",
            )
            gaussian = np.exp(-((rr - target_row) ** 2 + (cc - target_col) ** 2) / (2 * sigma_px**2)).astype(np.float32)
            weighted_gaussian = float(target_weight) * gaussian
            heatmap[row_min:row_max, col_min:col_max] = np.maximum(
                heatmap[row_min:row_max, col_min:col_max],
                weighted_gaussian,
            )
            difficulty[row_min:row_max, col_min:col_max] = np.maximum(
                difficulty[row_min:row_max, col_min:col_max],
                difficulty_value * weighted_gaussian,
            )

        center_x, center_y = float(row["x"]), float(row["y"])
        target_center_row, target_center_col = target_centers[0][0], target_centers[0][1]
        pixel_x, pixel_y = transform.rowcol_to_xy(target_center_row, target_center_col)
        offset[0, target_center_row, target_center_col] = (center_x - pixel_x) / transform.pixel_size
        offset[1, target_center_row, target_center_col] = (center_y - pixel_y) / transform.pixel_size
        positive_mask[target_center_row, target_center_col] = 1.0
        if row.get("height") is not None and row.get("height") == row.get("height"):
            height[target_center_row, target_center_col] = float(row["height"])
        if species_to_index and species_label_mode != "none":
            species_index = encode_species_label(str(row.get("species", "unknown")), species_to_index, species_label_mode)
            if species_index >= 0:
                species[target_center_row, target_center_col] = species_index
                species_mask[target_center_row, target_center_col] = 1.0

        crown_row_min = max(0, target_center_row - crown_radius_px)
        crown_row_max = min(raster_height, target_center_row + crown_radius_px + 1)
        crown_col_min = max(0, target_center_col - crown_radius_px)
        crown_col_max = min(raster_width, target_center_col + crown_radius_px + 1)
        rr_crown, cc_crown = np.meshgrid(
            np.arange(crown_row_min, crown_row_max),
            np.arange(crown_col_min, crown_col_max),
            indexing="ij",
        )
        distances_px = np.sqrt((rr_crown - target_center_row) ** 2 + (cc_crown - target_center_col) ** 2)
        crownness = np.clip(1.0 - distances_px / max(crown_radius_px, 1), 0.0, 1.0).astype(np.float32)
        distance[crown_row_min:crown_row_max, crown_col_min:crown_col_max] = np.maximum(
            distance[crown_row_min:crown_row_max, crown_col_min:crown_col_max],
            crownness,
        )

    return TargetMaps(
        heatmap=heatmap,
        offset=offset,
        height=height,
        positive_mask=positive_mask,
        distance=distance,
        difficulty=np.maximum(difficulty, 0.0).astype(np.float32),
        peakness=peakness.astype(np.float32),
        species=species,
        species_mask=species_mask,
        hard_negative_mask=hard_negative_mask.astype(np.float32),
    )

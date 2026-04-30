from __future__ import annotations

from pathlib import Path

import pandas as pd

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.auxiliary import compute_chm_feature_stack
from lidaragrosat.data.dataset import ProcessedPlotBundle, save_processed_plot
from lidaragrosat.data.inventory import PlotInventory
from lidaragrosat.data.lidar import (
    LidarPointCloud,
    build_dtm,
    compute_plot_bounds,
    filter_noise,
    normalize_heights,
    rasterize_structural_channels,
    read_lidar,
    sample_grid,
)
from lidaragrosat.data.qa import QAReport, estimate_shift_pixels, fraction_points_inside_bounds, height_outliers
from lidaragrosat.data.reference import read_reference_table, sanitize_reference_table
from lidaragrosat.data.rgb import normalize_rgb, read_rgb, resample_rgb_to_grid, shift_raster_pixels, write_raster
from lidaragrosat.utils.io import ensure_dir, write_json


def _filter_normalized_cloud(
    cloud: LidarPointCloud,
    normalized_z,
    min_height_m: float,
    max_height_m: float,
) -> tuple[LidarPointCloud, object]:
    mask = (normalized_z >= float(min_height_m)) & (normalized_z <= float(max_height_m))
    intensity = cloud.intensity[mask] if cloud.intensity is not None else None
    return LidarPointCloud(cloud.x[mask], cloud.y[mask], cloud.z[mask], intensity), normalized_z[mask]


def _choose_registration_correction(
    chm: object,
    rgb: object,
    max_shift_px: int,
) -> tuple[object, dict]:
    estimated_shift_x, estimated_shift_y = estimate_shift_pixels(chm, rgb, max_shift_px=max_shift_px)
    best_rgb = rgb
    best_payload = {
        "estimated_shift_px_x": float(estimated_shift_x),
        "estimated_shift_px_y": float(estimated_shift_y),
        "applied_shift_px_x": 0.0,
        "applied_shift_px_y": 0.0,
        "residual_shift_px_x": float(estimated_shift_x),
        "residual_shift_px_y": float(estimated_shift_y),
    }
    best_norm = float((estimated_shift_x**2 + estimated_shift_y**2) ** 0.5)

    candidate_shifts = {
        (int(round(estimated_shift_x)), int(round(estimated_shift_y))),
        (-int(round(estimated_shift_x)), -int(round(estimated_shift_y))),
    }
    for shift_x, shift_y in candidate_shifts:
        shifted = shift_raster_pixels(rgb, shift_x_px=shift_x, shift_y_px=shift_y)
        residual_x, residual_y = estimate_shift_pixels(chm, shifted, max_shift_px=max_shift_px)
        residual_norm = float((residual_x**2 + residual_y**2) ** 0.5)
        if residual_norm < best_norm:
            best_rgb = shifted
            best_norm = residual_norm
            best_payload = {
                "estimated_shift_px_x": float(estimated_shift_x),
                "estimated_shift_px_y": float(estimated_shift_y),
                "applied_shift_px_x": float(shift_x),
                "applied_shift_px_y": float(shift_y),
                "residual_shift_px_x": float(residual_x),
                "residual_shift_px_y": float(residual_y),
            }
    return best_rgb, best_payload


def prepare_plot_bundle(plot_inventory: PlotInventory, config: ExperimentConfig) -> tuple[ProcessedPlotBundle, QAReport, dict]:
    cloud = read_lidar(plot_inventory.lidar_path)
    n_points_raw = cloud.size
    cloud = filter_noise(cloud, *config.preprocessing.noise_z_quantiles)
    bounds = compute_plot_bounds(cloud)

    dtm, dtm_transform = build_dtm(
        cloud,
        bounds=bounds,
        cell_size=config.preprocessing.dtm_cell_size,
        ground_quantile=config.preprocessing.ground_quantile,
    )
    coarse_ground = sample_grid(
        dtm,
        bounds=bounds,
        resolution=config.preprocessing.dtm_cell_size,
        x=cloud.x,
        y=cloud.y,
    )
    ground_mask = cloud.z <= coarse_ground + config.preprocessing.ground_residual_threshold_m
    if ground_mask.sum() >= max(1024, int(0.05 * cloud.size)):
        ground_cloud = LidarPointCloud(
            x=cloud.x[ground_mask],
            y=cloud.y[ground_mask],
            z=cloud.z[ground_mask],
            intensity=cloud.intensity[ground_mask] if cloud.intensity is not None else None,
        )
        dtm, dtm_transform = build_dtm(
            ground_cloud,
            bounds=bounds,
            cell_size=config.preprocessing.dtm_cell_size,
            ground_quantile=config.preprocessing.ground_quantile,
        )
    normalized_z = normalize_heights(
        cloud,
        dtm=dtm,
        dtm_bounds=bounds,
        dtm_resolution=config.preprocessing.dtm_cell_size,
    )
    cloud_filtered, normalized_z = _filter_normalized_cloud(
        cloud,
        normalized_z,
        config.preprocessing.min_normalized_height_m,
        config.preprocessing.max_normalized_height_m,
    )

    lidar_stack, raster_transform = rasterize_structural_channels(
        cloud_filtered,
        normalized_z,
        bounds=bounds,
        resolution=config.data.raster_resolution,
        channel_names=config.data.structural_channels,
    )
    chm = lidar_stack[config.data.structural_channels.index("zmax")]
    chm_features = (
        compute_chm_feature_stack(
            chm=chm,
            pixel_size=raster_transform.pixel_size,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            min_height_m=config.data.min_tree_height_m,
        )
        if config.preprocessing.persist_chm_features
        else None
    )

    rgb_original, rgb_profile = read_rgb(plot_inventory.rgb_path)
    del rgb_original
    rgb_resampled = resample_rgb_to_grid(
        plot_inventory.rgb_path,
        transform=raster_transform,
        height=chm.shape[0],
        width=chm.shape[1],
    )
    estimated_shift_x = estimated_shift_y = None
    applied_shift_x = applied_shift_y = 0.0
    residual_shift_x = residual_shift_y = None
    if config.preprocessing.estimate_multimodal_shift:
        if config.preprocessing.apply_multimodal_shift_correction:
            rgb_resampled, shift_payload = _choose_registration_correction(
                chm=chm,
                rgb=rgb_resampled,
                max_shift_px=config.preprocessing.max_registration_shift_px,
            )
            estimated_shift_x = shift_payload["estimated_shift_px_x"]
            estimated_shift_y = shift_payload["estimated_shift_px_y"]
            applied_shift_x = shift_payload["applied_shift_px_x"]
            applied_shift_y = shift_payload["applied_shift_px_y"]
            residual_shift_x = shift_payload["residual_shift_px_x"]
            residual_shift_y = shift_payload["residual_shift_px_y"]
        else:
            estimated_shift_x, estimated_shift_y = estimate_shift_pixels(
                chm,
                rgb_resampled,
                max_shift_px=config.preprocessing.max_registration_shift_px,
            )
            residual_shift_x, residual_shift_y = estimated_shift_x, estimated_shift_y
    rgb_resampled = normalize_rgb(
        rgb_resampled,
        clip_percentiles=config.preprocessing.rgb_clip_percentiles,
        standardize=config.preprocessing.rgb_standardize,
    )

    reference_raw = read_reference_table(plot_inventory.reference_path)
    labels_inside_fraction = fraction_points_inside_bounds(reference_raw, bounds)
    reference, reference_stats = sanitize_reference_table(
        reference_raw,
        bounds=bounds,
        clip_to_bounds=config.preprocessing.clip_reference_to_bounds,
        height_min_m=config.preprocessing.reference_height_min_m,
        height_max_m=config.preprocessing.reference_height_max_m,
    )
    warnings: list[str] = []
    if residual_shift_x is not None and residual_shift_y is not None:
        if abs(residual_shift_x) > 1.0 or abs(residual_shift_y) > 1.0:
            warnings.append("Residual RGB-LiDAR shift above one output pixel.")
    if reference_stats["out_of_bounds_count"] > 0:
        warnings.append("Out-of-bounds reference trees were dropped during preprocessing.")
    if reference_stats["invalid_height_count"] > 0:
        warnings.append("Reference heights outside the physical range were masked as missing.")
    if labels_inside_fraction < 1.0:
        warnings.append("Some reference trees fall outside LiDAR-derived plot bounds.")

    qa_report = QAReport(
        plot_id=plot_inventory.plot_id,
        n_points_raw=n_points_raw,
        n_points=cloud_filtered.size,
        area_m2=bounds.area,
        lidar_density_pts_m2=float(cloud_filtered.size / max(bounds.area, 1.0)),
        n_reference_trees_raw=int(len(reference_raw)),
        n_reference_trees=int(len(reference)),
        n_reference_trees_dropped=int(len(reference_raw) - len(reference)),
        labels_inside_bounds_fraction=labels_inside_fraction,
        height_outlier_count=height_outliers(reference),
        estimated_shift_px_x=estimated_shift_x,
        estimated_shift_px_y=estimated_shift_y,
        applied_shift_px_x=applied_shift_x,
        applied_shift_px_y=applied_shift_y,
        residual_shift_px_x=residual_shift_x,
        residual_shift_px_y=residual_shift_y,
        warnings=warnings,
    )

    metadata = {
        "lidar_path": plot_inventory.lidar_path,
        "rgb_path": plot_inventory.rgb_path,
        "reference_path": plot_inventory.reference_path,
        "bounds": {
            "xmin": bounds.xmin,
            "ymin": bounds.ymin,
            "xmax": bounds.xmax,
            "ymax": bounds.ymax,
        },
        "structural_channels": config.data.structural_channels,
        "raster_resolution": config.data.raster_resolution,
        "dtm_cell_size": config.preprocessing.dtm_cell_size,
        "registration": {
            "estimated_shift_px_x": estimated_shift_x,
            "estimated_shift_px_y": estimated_shift_y,
            "applied_shift_px_x": applied_shift_x,
            "applied_shift_px_y": applied_shift_y,
            "residual_shift_px_x": residual_shift_x,
            "residual_shift_px_y": residual_shift_y,
        },
        "reference_preprocessing": reference_stats,
        "point_counts": {
            "raw": int(n_points_raw),
            "filtered": int(cloud_filtered.size),
        },
    }
    bundle = ProcessedPlotBundle(
        plot_id=plot_inventory.plot_id,
        rgb=rgb_resampled,
        lidar=lidar_stack,
        chm=chm,
        chm_features=chm_features,
        transform=raster_transform,
        reference=reference,
        crs=str(rgb_profile.get("crs")) if rgb_profile.get("crs") is not None else None,
        metadata=metadata,
    )
    artifacts = {
        "dtm": dtm,
        "dtm_transform": dtm_transform,
        "qa_report": qa_report.to_dict(),
        "chm_features": chm_features,
    }
    return bundle, qa_report, artifacts


def save_prepared_plot(
    bundle: ProcessedPlotBundle,
    artifacts: dict,
    destination_dir: str | Path,
) -> None:
    destination_dir = ensure_dir(destination_dir)
    save_processed_plot(bundle, destination_dir)
    write_json(artifacts["qa_report"], destination_dir / "qa_report.json")
    if bundle.crs is not None:
        write_raster(bundle.rgb, destination_dir / "rgb_resampled.tif", bundle.transform, crs=bundle.crs)
        write_raster(bundle.lidar, destination_dir / "lidar_structural.tif", bundle.transform, crs=bundle.crs)
        write_raster(bundle.chm, destination_dir / "chm.tif", bundle.transform, crs=bundle.crs)
        if artifacts.get("chm_features") is not None:
            write_raster(artifacts["chm_features"], destination_dir / "chm_features.tif", bundle.transform, crs=bundle.crs)
        write_raster(
            artifacts["dtm"],
            destination_dir / "dtm.tif",
            artifacts["dtm_transform"],
            crs=bundle.crs,
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from lidaragrosat.utils.geo import GeoTransform, iter_windows


def _extract_with_padding(array: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    row_end = row + size
    col_end = col + size
    pad_top = max(0, -row)
    pad_left = max(0, -col)
    pad_bottom = max(0, row_end - array.shape[-2])
    pad_right = max(0, col_end - array.shape[-1])
    row_start_clipped = max(0, row)
    col_start_clipped = max(0, col)
    row_end_clipped = min(array.shape[-2], row_end)
    col_end_clipped = min(array.shape[-1], col_end)
    if array.ndim == 3:
        cropped = array[:, row_start_clipped:row_end_clipped, col_start_clipped:col_end_clipped]
        return np.pad(cropped, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect").copy()
    cropped = array[row_start_clipped:row_end_clipped, col_start_clipped:col_end_clipped]
    return np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect").copy()


@dataclass(slots=True)
class PredictionMaps:
    heatmap: np.ndarray
    offset: np.ndarray | None
    height: np.ndarray | None
    distance: np.ndarray | None = None
    peakness: np.ndarray | None = None


def build_chm_features_if_needed(
    model,
    chm: np.ndarray,
    pixel_size: float,
    peakness_window_sizes_m: tuple[float, ...] | list[float] = (0.8, 1.2, 1.6),
    min_height_m: float = 2.0,
    precomputed: np.ndarray | None = None,
) -> np.ndarray | None:
    if not getattr(model, "use_chm_branch", False):
        return None
    if precomputed is not None:
        return precomputed.astype(np.float32)
    from lidaragrosat.data.auxiliary import compute_chm_feature_stack

    return compute_chm_feature_stack(
        chm=chm,
        pixel_size=pixel_size,
        peakness_window_sizes_m=peakness_window_sizes_m,
        min_height_m=min_height_m,
    )


def _apply_spatial_transform(array: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return array.copy()
    if transform_name == "hflip":
        axis = -1 if array.ndim == 3 else 1
        return np.flip(array, axis=axis).copy()
    if transform_name == "vflip":
        axis = -2 if array.ndim == 3 else 0
        return np.flip(array, axis=axis).copy()
    if transform_name == "rot90":
        axes = (1, 2) if array.ndim == 3 else (0, 1)
        return np.rot90(array, k=1, axes=axes).copy()
    raise ValueError(f"Unsupported TTA transform: {transform_name}")


def _invert_prediction_maps(prediction_maps: PredictionMaps, transform_name: str) -> PredictionMaps:
    heatmap = prediction_maps.heatmap
    offset = prediction_maps.offset
    height = prediction_maps.height
    distance = prediction_maps.distance
    peakness = prediction_maps.peakness

    if transform_name == "identity":
        return prediction_maps
    if transform_name == "hflip":
        heatmap = np.flip(heatmap, axis=1).copy()
        if height is not None:
            height = np.flip(height, axis=2).copy()
        if distance is not None:
            distance = np.flip(distance, axis=2).copy()
        if peakness is not None:
            peakness = np.flip(peakness, axis=2).copy()
        if offset is not None:
            offset = np.flip(offset, axis=2).copy()
            offset[0] *= -1
    elif transform_name == "vflip":
        heatmap = np.flip(heatmap, axis=0).copy()
        if height is not None:
            height = np.flip(height, axis=1).copy()
        if distance is not None:
            distance = np.flip(distance, axis=1).copy()
        if peakness is not None:
            peakness = np.flip(peakness, axis=1).copy()
        if offset is not None:
            offset = np.flip(offset, axis=1).copy()
            offset[1] *= -1
    elif transform_name == "rot90":
        heatmap = np.rot90(heatmap, k=3, axes=(0, 1)).copy()
        if height is not None:
            height = np.rot90(height, k=3, axes=(1, 2)).copy()
        if distance is not None:
            distance = np.rot90(distance, k=3, axes=(1, 2)).copy()
        if peakness is not None:
            peakness = np.rot90(peakness, k=3, axes=(1, 2)).copy()
        if offset is not None:
            offset = np.rot90(offset, k=3, axes=(1, 2)).copy()
            offset = np.stack([offset[1], -offset[0]], axis=0)
    else:
        raise ValueError(f"Unsupported TTA transform: {transform_name}")
    return PredictionMaps(
        heatmap=heatmap,
        offset=offset,
        height=height,
        distance=distance,
        peakness=peakness,
    )


def predict_plot_sliding_window(
    model,
    rgb: np.ndarray,
    lidar: np.ndarray,
    tile_size: int,
    overlap: int,
    chm: np.ndarray | None = None,
    context_margin_px: int = 0,
    device: str = "cpu",
) -> PredictionMaps:
    import torch

    model.eval()
    _, raster_height, raster_width = rgb.shape
    heatmap_sum = np.zeros((1, raster_height, raster_width), dtype=np.float32)
    offset_sum = np.zeros((2, raster_height, raster_width), dtype=np.float32)
    height_sum = np.zeros((1, raster_height, raster_width), dtype=np.float32)
    distance_sum = np.zeros((1, raster_height, raster_width), dtype=np.float32)
    peakness_sum = np.zeros((1, raster_height, raster_width), dtype=np.float32)
    counts = np.zeros((1, raster_height, raster_width), dtype=np.float32)
    has_offset_head = False
    has_height_head = False
    has_distance_head = False
    has_peakness_head = False

    with torch.no_grad():
        for row, col in iter_windows(raster_height, raster_width, tile_size, overlap):
            input_size = tile_size + 2 * context_margin_px
            valid_height = min(tile_size, raster_height - row)
            valid_width = min(tile_size, raster_width - col)
            rgb_tile_np = _extract_with_padding(rgb, row - context_margin_px, col - context_margin_px, input_size)
            lidar_tile_np = _extract_with_padding(lidar, row - context_margin_px, col - context_margin_px, input_size)
            rgb_tile = torch.from_numpy(rgb_tile_np).unsqueeze(0).to(device)
            lidar_tile = torch.from_numpy(lidar_tile_np).unsqueeze(0).to(device)
            kwargs = {"rgb": rgb_tile, "lidar": lidar_tile}
            if chm is not None:
                chm_tile_np = _extract_with_padding(chm, row - context_margin_px, col - context_margin_px, input_size)
                chm_tile = torch.from_numpy(chm_tile_np).unsqueeze(0).to(device)
                kwargs["chm"] = chm_tile
            outputs = model(**kwargs)

            heatmap = torch.sigmoid(outputs["heatmap"]).cpu().numpy()
            row_slice = slice(context_margin_px, context_margin_px + valid_height)
            col_slice = slice(context_margin_px, context_margin_px + valid_width)
            target_rows = slice(row, row + valid_height)
            target_cols = slice(col, col + valid_width)
            heatmap_sum[:, target_rows, target_cols] += heatmap[0, :, row_slice, col_slice]
            counts[:, target_rows, target_cols] += 1.0

            if "offset" in outputs:
                has_offset_head = True
                offset = outputs["offset"].cpu().numpy()
                offset_sum[:, target_rows, target_cols] += offset[0, :, row_slice, col_slice]
            if "height" in outputs:
                has_height_head = True
                height = outputs["height"].cpu().numpy()
                height_sum[:, target_rows, target_cols] += height[0, :, row_slice, col_slice]
            if "distance" in outputs:
                has_distance_head = True
                distance = torch.sigmoid(outputs["distance"]).cpu().numpy()
                distance_sum[:, target_rows, target_cols] += distance[0, :, row_slice, col_slice]
            if "peakness" in outputs:
                has_peakness_head = True
                peakness = torch.sigmoid(outputs["peakness"]).cpu().numpy()
                peakness_sum[:, target_rows, target_cols] += peakness[0, :, row_slice, col_slice]

    counts = np.maximum(counts, 1.0)
    heatmap = heatmap_sum / counts
    offset = offset_sum / counts if has_offset_head else None
    height = height_sum / counts if has_height_head else None
    distance = distance_sum / counts if has_distance_head else None
    peakness = peakness_sum / counts if has_peakness_head else None
    return PredictionMaps(
        heatmap=heatmap[0],
        offset=offset,
        height=height,
        distance=distance,
        peakness=peakness,
    )


def predict_plot_sliding_window_tta(
    model,
    rgb: np.ndarray,
    lidar: np.ndarray,
    tile_size: int,
    overlap: int,
    chm: np.ndarray | None = None,
    context_margin_px: int = 0,
    device: str = "cpu",
    transforms: tuple[str, ...] = ("identity", "hflip", "vflip", "rot90"),
) -> PredictionMaps:
    predictions: list[PredictionMaps] = []
    for transform_name in transforms:
        transformed_rgb = _apply_spatial_transform(rgb, transform_name)
        transformed_lidar = _apply_spatial_transform(lidar, transform_name)
        transformed_chm = _apply_spatial_transform(chm, transform_name) if chm is not None else None
        prediction_maps = predict_plot_sliding_window(
            model,
            rgb=transformed_rgb,
            lidar=transformed_lidar,
            chm=transformed_chm,
            tile_size=tile_size,
            overlap=overlap,
            context_margin_px=context_margin_px,
            device=device,
        )
        predictions.append(_invert_prediction_maps(prediction_maps, transform_name))

    heatmap = np.mean([prediction.heatmap for prediction in predictions], axis=0).astype(np.float32)
    offset = None
    if any(prediction.offset is not None for prediction in predictions):
        valid_offsets = [prediction.offset for prediction in predictions if prediction.offset is not None]
        offset = np.mean(valid_offsets, axis=0).astype(np.float32)
    height = None
    if any(prediction.height is not None for prediction in predictions):
        valid_heights = [prediction.height for prediction in predictions if prediction.height is not None]
        height = np.mean(valid_heights, axis=0).astype(np.float32)
    distance = None
    if any(prediction.distance is not None for prediction in predictions):
        valid_distances = [prediction.distance for prediction in predictions if prediction.distance is not None]
        distance = np.mean(valid_distances, axis=0).astype(np.float32)
    peakness = None
    if any(prediction.peakness is not None for prediction in predictions):
        valid_peakness = [prediction.peakness for prediction in predictions if prediction.peakness is not None]
        peakness = np.mean(valid_peakness, axis=0).astype(np.float32)
    return PredictionMaps(
        heatmap=heatmap,
        offset=offset,
        height=height,
        distance=distance,
        peakness=peakness,
    )


def decode_tree_predictions(
    prediction_maps: PredictionMaps,
    transform: GeoTransform,
    confidence_threshold: float,
    nms_radius_m: float,
    min_pred_height_m: float | None = None,
    min_chm_at_center_m: float | None = None,
    border_margin_px: int = 0,
    chm: np.ndarray | None = None,
    adaptive_nms: bool = False,
    adaptive_nms_r0_m: float = 0.6,
    adaptive_nms_alpha: float = 0.03,
    adaptive_nms_rmin_m: float = 0.8,
    adaptive_nms_rmax_m: float = 1.8,
) -> pd.DataFrame:
    heatmap = prediction_maps.heatmap
    initial_radius_m = adaptive_nms_rmin_m if adaptive_nms else nms_radius_m
    radius_px = max(1, int(round(initial_radius_m / transform.pixel_size)))
    maxima = maximum_filter(heatmap, size=2 * radius_px + 1, mode="nearest")
    peak_mask = (heatmap == maxima) & (heatmap >= confidence_threshold)
    peak_rows, peak_cols = np.where(peak_mask)

    candidate_records: list[dict] = []
    raster_height, raster_width = heatmap.shape
    for row, col in zip(peak_rows, peak_cols):
        if border_margin_px > 0 and (
            row < border_margin_px
            or row >= raster_height - border_margin_px
            or col < border_margin_px
            or col >= raster_width - border_margin_px
        ):
            continue
        if min_chm_at_center_m is not None and chm is not None and float(chm[row, col]) < min_chm_at_center_m:
            continue
        x, y = transform.rowcol_to_xy(int(row), int(col))
        predicted_height = None
        if prediction_maps.offset is not None:
            x += float(prediction_maps.offset[0, row, col]) * transform.pixel_size
            y += float(prediction_maps.offset[1, row, col]) * transform.pixel_size
        if prediction_maps.height is not None:
            predicted_height = float(prediction_maps.height[0, row, col])
            if min_pred_height_m is not None and predicted_height < min_pred_height_m:
                continue
        adaptive_radius_m = nms_radius_m
        if adaptive_nms:
            height_for_nms = predicted_height
            if height_for_nms is None and chm is not None:
                height_for_nms = float(chm[row, col])
            if height_for_nms is not None:
                adaptive_radius_m = float(
                    np.clip(
                        adaptive_nms_r0_m + adaptive_nms_alpha * height_for_nms,
                        adaptive_nms_rmin_m,
                        adaptive_nms_rmax_m,
                    )
                )
        record = {
            "x": x,
            "y": y,
            "x_px": float(col),
            "y_px": float(row),
            "confidence": float(heatmap[row, col]),
            "nms_radius_m": float(adaptive_radius_m),
        }
        if predicted_height is not None:
            record["height_pred"] = predicted_height
        candidate_records.append(record)

    if not candidate_records:
        return pd.DataFrame(columns=["x", "y", "x_px", "y_px", "confidence", "height_pred"])

    candidates = pd.DataFrame.from_records(candidate_records).sort_values("confidence", ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[np.ndarray] = []
    kept_radii: list[float] = []
    for _, row in candidates.iterrows():
        point = np.array([float(row["x"]), float(row["y"])], dtype=float)
        radius_m = float(row["nms_radius_m"])
        suppress = False
        for kept_point, kept_radius in zip(kept_xy, kept_radii):
            if np.linalg.norm(point - kept_point) <= max(radius_m, kept_radius):
                suppress = True
                break
        if suppress:
            continue
        kept_rows.append(row.to_dict())
        kept_xy.append(point)
        kept_radii.append(radius_m)

    predictions = pd.DataFrame.from_records(kept_rows)
    return predictions.drop(columns=["nms_radius_m"], errors="ignore")

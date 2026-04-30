from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from lidaragrosat.baselines.local_maxima import LocalMaximaParams, detect_local_maxima
from lidaragrosat.data.auxiliary import compute_rgb_entropy
from lidaragrosat.data.dataset import ProcessedPlotBundle
from lidaragrosat.paper_faithful.adaptive_local_maxima import FaithfulAdaptiveBaselineParams, adaptive_local_maxima_filter_3d
from lidaragrosat.stage1.api import Stage1Output, Stage1SegmentDiagnostics


def _load_raw_points(bundle: ProcessedPlotBundle) -> np.ndarray:
    metadata = bundle.metadata or {}
    lidar_path = metadata.get("lidar_path")
    if not lidar_path:
        raise ValueError(f"Bundle {bundle.plot_id} does not expose metadata.lidar_path.")
    las = laspy.read(Path(lidar_path))
    return np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)


def _map_predictions_to_pixels(bundle: ProcessedPlotBundle, predictions: pd.DataFrame) -> pd.DataFrame:
    table = predictions.copy()
    if table.empty:
        table["x_px"] = pd.Series(dtype=float)
        table["y_px"] = pd.Series(dtype=float)
        return table
    rows = []
    cols = []
    for _, row in table.iterrows():
        r, c = bundle.transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
        rows.append(float(r))
        cols.append(float(c))
    table["y_px"] = rows
    table["x_px"] = cols
    return table


def _segment_geometry_table(
    bundle: ProcessedPlotBundle,
    segment_labels: np.ndarray,
    smooth_chm: np.ndarray,
    source_name: str,
) -> pd.DataFrame:
    records: list[dict] = []
    if segment_labels is None or segment_labels.size == 0:
        return pd.DataFrame(
            columns=["plot_id", "segment_id", "x", "y", "area_m2", "height_max_m", "height_mean_m", "elongation", "compactness", "source"]
        )
    for segment_id in np.unique(segment_labels):
        if int(segment_id) <= 0:
            continue
        mask = segment_labels == int(segment_id)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        area_m2 = float(mask.sum() * (bundle.transform.pixel_size**2))
        height_values = bundle.chm[mask]
        height_values = height_values[np.isfinite(height_values)]
        if len(height_values) == 0:
            continue
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())
        x, y = bundle.transform.rowcol_to_xy(int(round(centroid_row)), int(round(centroid_col)))
        bbox_h = max(float(rows.max() - rows.min() + 1), 1.0)
        bbox_w = max(float(cols.max() - cols.min() + 1), 1.0)
        elongation = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1.0)
        perimeter_like = 2.0 * (bbox_h + bbox_w)
        compactness = float(4.0 * np.pi * mask.sum() / max(perimeter_like**2, 1.0))
        max_idx = int(np.argmax(np.where(mask, smooth_chm, -np.inf)))
        max_row, max_col = np.unravel_index(max_idx, smooth_chm.shape)
        max_x, max_y = bundle.transform.rowcol_to_xy(int(max_row), int(max_col))
        records.append(
            {
                "plot_id": bundle.plot_id,
                "segment_id": int(segment_id),
                "x": float(max_x),
                "y": float(max_y),
                "centroid_x": float(x),
                "centroid_y": float(y),
                "area_m2": area_m2,
                "height_max_m": float(np.max(height_values)),
                "height_mean_m": float(np.mean(height_values)),
                "elongation": float(elongation),
                "compactness": float(compactness),
                "source": source_name,
            }
        )
    return pd.DataFrame.from_records(records)


def _diagnose_segments(
    bundle: ProcessedPlotBundle,
    segment_labels: np.ndarray | None,
    segment_table: pd.DataFrame,
    match_radius_m: float = 2.0,
) -> Stage1SegmentDiagnostics:
    diagnostics = Stage1SegmentDiagnostics()
    if segment_labels is None or segment_labels.size == 0:
        diagnostics.over_segmented = int(len(bundle.reference))
        return diagnostics
    reference = bundle.reference.copy()
    if reference.empty:
        diagnostics.empty_segment = int(segment_table["segment_id"].nunique()) if not segment_table.empty else 0
        return diagnostics

    ref_segments: dict[int, list[int]] = {}
    for ref_index, (_, row) in enumerate(reference.iterrows()):
        r, c = bundle.transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
        segment_id = 0
        if 0 <= r < segment_labels.shape[0] and 0 <= c < segment_labels.shape[1]:
            segment_id = int(segment_labels[r, c])
        if segment_id > 0:
            ref_segments.setdefault(segment_id, []).append(ref_index)
    for _, row in segment_table.iterrows():
        segment_id = int(row["segment_id"])
        matched_refs = ref_segments.get(segment_id, [])
        if len(matched_refs) == 0:
            diagnostics.empty_segment += 1
        elif len(matched_refs) == 1:
            diagnostics.correct_segments += 1
        else:
            diagnostics.under_segmented += 1

    if not segment_table.empty:
        gt_cover_counts = []
        for _, ref_row in reference.iterrows():
            distances = np.sqrt((segment_table["x"] - float(ref_row["x"])) ** 2 + (segment_table["y"] - float(ref_row["y"])) ** 2)
            gt_cover_counts.append(int((distances <= match_radius_m).sum()))
        diagnostics.over_segmented = int(sum(count > 1 for count in gt_cover_counts))
        diagnostics.mixed_segment = int(
            sum(
                1
                for segment_id, refs in ref_segments.items()
                if len(refs) == 1
                and any(count > 1 for count in gt_cover_counts)
                and segment_id > 0
            )
        )
    return diagnostics


@dataclass(slots=True)
class LocalMaximaStage1Generator:
    window_radius_m: float = 2.0
    gaussian_sigma: float = 1.0
    min_height_m: float = 3.0

    def generate(self, plot_bundle: ProcessedPlotBundle) -> Stage1Output:
        predictions = detect_local_maxima(
            plot_bundle.chm,
            plot_bundle.transform,
            LocalMaximaParams(
                window_radius_m=float(self.window_radius_m),
                gaussian_sigma=float(self.gaussian_sigma),
                min_height_m=float(self.min_height_m),
            ),
        )
        candidates = _map_predictions_to_pixels(plot_bundle, predictions)
        candidates["source"] = "local_maxima"
        candidates["geometry_score"] = candidates.get("confidence", 0.0)
        return Stage1Output(
            plot_id=plot_bundle.plot_id,
            candidates=candidates,
            metadata={"generator": "local_maxima"},
        )


@dataclass(slots=True)
class AdaptiveLocalMaximaStage1Generator:
    height_threshold_m: float = 3.0
    a: float = 0.035
    b: float = 2.5

    def generate(self, plot_bundle: ProcessedPlotBundle) -> Stage1Output:
        points_xyz = _load_raw_points(plot_bundle)
        predictions = adaptive_local_maxima_filter_3d(
            points_xyz=points_xyz,
            params=FaithfulAdaptiveBaselineParams(
                height_threshold_m=float(self.height_threshold_m),
                a=float(self.a),
                b=float(self.b),
            ),
        )
        candidates = _map_predictions_to_pixels(plot_bundle, predictions)
        candidates["source"] = "adaptive_local_maxima"
        candidates["geometry_score"] = candidates.get("height_pred", 0.0)
        return Stage1Output(
            plot_id=plot_bundle.plot_id,
            candidates=candidates,
            metadata={"generator": "adaptive_local_maxima"},
        )


@dataclass(slots=True)
class WatershedStage1Generator:
    gaussian_sigma_m: float = 0.2
    min_distance_m: float = 0.8
    min_height_m: float = 3.0
    min_segment_area_m2: float = 0.5
    hole_fill_radius_m: float = 0.3
    compactness: float = 0.0075

    def _fill_small_holes(self, chm: np.ndarray, pixel_size: float) -> np.ndarray:
        valid = np.isfinite(chm)
        if valid.all():
            return chm.astype(np.float32)
        nearest_indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = np.asarray(chm[tuple(nearest_indices)], dtype=np.float32)
        distance_to_valid = distance_transform_edt(~valid) * float(pixel_size)
        filled[distance_to_valid > float(self.hole_fill_radius_m)] = 0.0
        filled[valid] = np.asarray(chm[valid], dtype=np.float32)
        return filled

    def _segment(self, bundle: ProcessedPlotBundle) -> tuple[np.ndarray, np.ndarray]:
        chm = self._fill_small_holes(bundle.chm.astype(np.float32), pixel_size=bundle.transform.pixel_size)
        sigma_px = max(float(self.gaussian_sigma_m) / max(bundle.transform.pixel_size, 1e-6), 0.0)
        smooth = gaussian_filter(chm, sigma=sigma_px, mode="nearest")
        canopy_mask = smooth >= float(self.min_height_m)
        if not np.any(canopy_mask):
            return np.zeros_like(bundle.chm, dtype=np.int32), smooth
        min_distance_px = max(1, int(round(float(self.min_distance_m) / bundle.transform.pixel_size)))
        peaks = peak_local_max(
            smooth,
            min_distance=min_distance_px,
            threshold_abs=float(self.min_height_m),
            labels=canopy_mask.astype(np.uint8),
            exclude_border=False,
        )
        markers = np.zeros_like(bundle.chm, dtype=np.int32)
        for marker_id, (row, col) in enumerate(peaks, start=1):
            markers[int(row), int(col)] = int(marker_id)
        if markers.max() == 0:
            return np.zeros_like(bundle.chm, dtype=np.int32), smooth
        segments = watershed(-smooth, markers=markers, mask=canopy_mask, compactness=float(self.compactness))
        min_pixels = max(1, int(round(float(self.min_segment_area_m2) / (bundle.transform.pixel_size**2))))
        for segment_id in np.unique(segments):
            if int(segment_id) <= 0:
                continue
            if int((segments == int(segment_id)).sum()) < min_pixels:
                segments[segments == int(segment_id)] = 0
        return segments.astype(np.int32), smooth.astype(np.float32)

    def generate(self, plot_bundle: ProcessedPlotBundle) -> Stage1Output:
        segment_labels, smooth = self._segment(plot_bundle)
        segment_table = _segment_geometry_table(plot_bundle, segment_labels, smooth, source_name="watershed")
        if not segment_table.empty:
            candidates = segment_table.copy()
            candidates["height_pred"] = candidates["height_max_m"]
            candidates["geometry_score"] = candidates["compactness"]
            candidates = candidates[
                ["x", "y", "height_pred", "geometry_score", "segment_id", "area_m2", "elongation", "compactness", "source"]
            ].copy()
        else:
            candidates = pd.DataFrame(
                columns=["x", "y", "height_pred", "geometry_score", "segment_id", "area_m2", "elongation", "compactness", "source"]
            )
        candidates = _map_predictions_to_pixels(plot_bundle, candidates)
        diagnostics = _diagnose_segments(plot_bundle, segment_labels, segment_table)
        return Stage1Output(
            plot_id=plot_bundle.plot_id,
            candidates=candidates,
            segment_labels=segment_labels,
            segment_table=segment_table,
            diagnostics=diagnostics,
            metadata={
                "generator": "watershed",
                "gaussian_sigma_m": float(self.gaussian_sigma_m),
                "min_distance_m": float(self.min_distance_m),
                "min_segment_area_m2": float(self.min_segment_area_m2),
                "min_height_m": float(self.min_height_m),
                "hole_fill_radius_m": float(self.hole_fill_radius_m),
            },
        )


@dataclass(slots=True)
class AdaptiveWatershedStage1Generator(WatershedStage1Generator):
    policy_mode: str = "lookup"
    sigma_low_m: float = 0.2
    sigma_high_m: float = 0.8
    min_distance_low_m: float = 0.8
    min_distance_high_m: float = 1.6
    area_low_m2: float = 0.25
    area_high_m2: float = 1.5

    def _complexity_weight(self, bundle: ProcessedPlotBundle) -> np.ndarray:
        chm_norm = np.clip(bundle.chm / max(np.nanpercentile(bundle.chm, 95), 1e-6), 0.0, 1.0)
        roughness = np.zeros_like(bundle.chm, dtype=np.float32)
        channel_names = (bundle.metadata or {}).get("structural_channels", [])
        if "vertical_roughness" in channel_names:
            idx = channel_names.index("vertical_roughness")
            roughness = bundle.lidar[idx].astype(np.float32)
            roughness = np.clip(roughness / max(np.nanpercentile(roughness, 95), 1e-6), 0.0, 1.0)
        rgb_entropy = compute_rgb_entropy(bundle.rgb).astype(np.float32)
        rgb_entropy = np.clip(rgb_entropy / max(np.nanpercentile(rgb_entropy, 95), 1e-6), 0.0, 1.0)
        return np.clip(0.45 * chm_norm + 0.35 * roughness + 0.20 * rgb_entropy, 0.0, 1.0)

    def _segment(self, bundle: ProcessedPlotBundle) -> tuple[np.ndarray, np.ndarray]:
        weight = self._complexity_weight(bundle)
        chm = self._fill_small_holes(bundle.chm.astype(np.float32), pixel_size=bundle.transform.pixel_size)
        sigma_low_px = max(float(self.sigma_low_m) / max(bundle.transform.pixel_size, 1e-6), 0.0)
        sigma_high_px = max(float(self.sigma_high_m) / max(bundle.transform.pixel_size, 1e-6), 0.0)
        smooth_low = gaussian_filter(chm, sigma=sigma_low_px, mode="nearest")
        smooth_high = gaussian_filter(chm, sigma=sigma_high_px, mode="nearest")
        smooth = weight * smooth_low + (1.0 - weight) * smooth_high
        canopy_mask = smooth >= float(self.min_height_m)
        if not np.any(canopy_mask):
            return np.zeros_like(bundle.chm, dtype=np.int32), smooth.astype(np.float32)
        peaks = peak_local_max(
            smooth,
            min_distance=1,
            threshold_abs=float(self.min_height_m),
            labels=canopy_mask.astype(np.uint8),
            exclude_border=False,
        )
        if peaks.size == 0:
            return np.zeros_like(bundle.chm, dtype=np.int32), smooth.astype(np.float32)
        peak_scores = smooth[peaks[:, 0], peaks[:, 1]]
        order = np.argsort(-peak_scores)
        markers = np.zeros_like(bundle.chm, dtype=np.int32)
        kept: list[tuple[int, int, float]] = []
        for peak_index in order.tolist():
            row = int(peaks[peak_index, 0])
            col = int(peaks[peak_index, 1])
            local_weight = float(weight[row, col])
            current_radius_m = self.min_distance_high_m - local_weight * (self.min_distance_high_m - self.min_distance_low_m)
            current_radius_px = max(1.0, current_radius_m / max(bundle.transform.pixel_size, 1e-6))
            if any(np.hypot(row - keep_row, col - keep_col) <= current_radius_px for keep_row, keep_col, _ in kept):
                continue
            kept.append((row, col, local_weight))
            markers[row, col] = len(kept)
        if markers.max() == 0:
            return np.zeros_like(bundle.chm, dtype=np.int32), smooth.astype(np.float32)
        segments = watershed(-smooth, markers=markers, mask=canopy_mask, compactness=float(self.compactness))
        for segment_id in np.unique(segments):
            if int(segment_id) <= 0:
                continue
            segment_mask = segments == int(segment_id)
            rows, cols = np.where(segment_mask)
            if len(rows) == 0:
                continue
            local_weight = float(weight[rows, cols].mean())
            min_area_m2 = self.area_high_m2 - local_weight * (self.area_high_m2 - self.area_low_m2)
            min_pixels = max(1, int(round(min_area_m2 / (bundle.transform.pixel_size**2))))
            if int(segment_mask.sum()) < min_pixels:
                segments[segment_mask] = 0
        return segments.astype(np.int32), smooth.astype(np.float32)

    def generate(self, plot_bundle: ProcessedPlotBundle) -> Stage1Output:
        output = super(AdaptiveWatershedStage1Generator, self).generate(plot_bundle)
        output.metadata["generator"] = "adaptive_watershed"
        output.metadata["policy_mode"] = self.policy_mode
        output.metadata["sigma_low_m"] = float(self.sigma_low_m)
        output.metadata["sigma_high_m"] = float(self.sigma_high_m)
        output.metadata["min_distance_low_m"] = float(self.min_distance_low_m)
        output.metadata["min_distance_high_m"] = float(self.min_distance_high_m)
        output.metadata["area_low_m2"] = float(self.area_low_m2)
        output.metadata["area_high_m2"] = float(self.area_high_m2)
        if not output.segment_table.empty:
            output.segment_table = output.segment_table.copy()
            output.segment_table["source"] = "adaptive_watershed"
        if not output.candidates.empty:
            output.candidates = output.candidates.copy()
            output.candidates["source"] = "adaptive_watershed"
        return output

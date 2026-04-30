from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - scipy is part of the dev/runtime env, but keep a safe fallback.
    cKDTree = None
import torch
from torch.utils.data import Dataset

from lidaragrosat.data.auxiliary import compute_multiscale_peakness
from lidaragrosat.data.candidates import extract_centered_patch
from lidaragrosat.data.dataset import ProcessedPlotBundle
from lidaragrosat.labels import resolve_reference_points
from lidaragrosat.stage1.api import Stage1Output


_GLOBAL_PEAKNESS_CACHE: dict[str, np.ndarray] = {}
_GLOBAL_RAW_POINTS_CACHE: dict[str, np.ndarray] = {}
_GLOBAL_RAW_POINT_INDEX_CACHE: dict[str, object] = {}


def _load_raw_point_features(bundle: ProcessedPlotBundle) -> np.ndarray:
    lidar_path = (bundle.metadata or {}).get("lidar_path")
    if not lidar_path:
        return np.empty((0, 6), dtype=np.float32)
    las = laspy.read(Path(lidar_path))
    intensity = getattr(las, "intensity", np.zeros(len(las.x), dtype=np.float32))
    return_number = getattr(las, "return_number", np.ones(len(las.x), dtype=np.float32))
    number_of_returns = getattr(las, "number_of_returns", np.ones(len(las.x), dtype=np.float32))
    return np.stack([las.x, las.y, las.z, intensity, return_number, number_of_returns], axis=1).astype(np.float32)


def _build_point_spatial_index(points: np.ndarray):
    if cKDTree is None or len(points) == 0:
        return None
    return cKDTree(points[:, :2])


def build_local_dataset_caches(
    plot_bundles: dict[str, ProcessedPlotBundle],
    include_point_cloud: bool,
    plot_ids: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    active_plot_ids = plot_ids or list(plot_bundles.keys())
    peakness_cache: dict[str, np.ndarray] = {}
    raw_points: dict[str, np.ndarray] = {}
    raw_point_index: dict[str, object] = {}
    for plot_id in active_plot_ids:
        if plot_id not in _GLOBAL_PEAKNESS_CACHE:
            _GLOBAL_PEAKNESS_CACHE[plot_id] = compute_multiscale_peakness(
                plot_bundles[plot_id].chm,
                plot_bundles[plot_id].transform.pixel_size,
                window_sizes_m=[0.8, 1.2, 1.6],
            )
        peakness_cache[plot_id] = _GLOBAL_PEAKNESS_CACHE[plot_id]
    if include_point_cloud:
        for plot_id in active_plot_ids:
            if plot_id not in _GLOBAL_RAW_POINTS_CACHE:
                _GLOBAL_RAW_POINTS_CACHE[plot_id] = _load_raw_point_features(plot_bundles[plot_id])
                _GLOBAL_RAW_POINT_INDEX_CACHE[plot_id] = _build_point_spatial_index(_GLOBAL_RAW_POINTS_CACHE[plot_id])
            raw_points[plot_id] = _GLOBAL_RAW_POINTS_CACHE[plot_id]
            raw_point_index[plot_id] = _GLOBAL_RAW_POINT_INDEX_CACHE[plot_id]
    return peakness_cache, raw_points, raw_point_index


def _sample_local_points(
    points: np.ndarray,
    spatial_index,
    center_x: float,
    center_y: float,
    center_z: float,
    radius_m: float,
    n_points: int,
    training: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((n_points, 6), dtype=np.float32)
    if spatial_index is not None:
        local_indices = spatial_index.query_ball_point([float(center_x), float(center_y)], r=float(radius_m))
        local = points[np.asarray(sorted(local_indices), dtype=np.int64)] if local_indices else np.empty((0, points.shape[1]), dtype=points.dtype)
    else:
        dx = points[:, 0] - float(center_x)
        dy = points[:, 1] - float(center_y)
        mask = (dx * dx + dy * dy) <= float(radius_m) ** 2
        local = points[mask]
    if len(local) == 0:
        return np.zeros((n_points, 6), dtype=np.float32)
    if len(local) >= n_points:
        choice = rng.choice(len(local), size=n_points, replace=False if training else False)
        sampled = local[choice]
    else:
        choice = rng.choice(len(local), size=n_points, replace=True)
        sampled = local[choice]
    rel_x = sampled[:, 0] - float(center_x)
    rel_y = sampled[:, 1] - float(center_y)
    rel_z = sampled[:, 2] - float(center_z)
    normalized_z = rel_z / max(float(radius_m), 1e-6)
    intensity = sampled[:, 3] / max(float(np.percentile(sampled[:, 3], 95)) if np.any(sampled[:, 3] > 0) else 1.0, 1e-6)
    return_ratio = sampled[:, 4] / np.maximum(sampled[:, 5], 1.0)
    return np.stack([rel_x, rel_y, rel_z, normalized_z, intensity, return_ratio], axis=1).astype(np.float32)


def _segment_gt_stats(stage1_output: Stage1Output, bundle: ProcessedPlotBundle) -> tuple[dict[int, int], dict[int, int]]:
    counts_by_segment: dict[int, int] = {}
    if stage1_output.segment_labels is None or stage1_output.segment_labels.size == 0 or bundle.reference.empty:
        return counts_by_segment, {}
    for ref_index, (_, row) in enumerate(bundle.reference.iterrows()):
        r, c = bundle.transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
        if not (0 <= r < stage1_output.segment_labels.shape[0] and 0 <= c < stage1_output.segment_labels.shape[1]):
            continue
        segment_id = int(stage1_output.segment_labels[r, c])
        if segment_id <= 0:
            continue
        counts_by_segment[segment_id] = counts_by_segment.get(segment_id, 0) + 1
    duplicates = {segment_id: int(count > 1) for segment_id, count in counts_by_segment.items()}
    return counts_by_segment, duplicates


def build_local_proposal_table(
    plot_bundles: dict[str, ProcessedPlotBundle],
    stage1_outputs: dict[str, Stage1Output],
    plot_ids: list[str],
    mode: str,
    label_policy: str,
    positive_radius_m: float,
    snap_radius_m: float = 1.0,
    mixed_weight: float = 0.35,
) -> pd.DataFrame:
    rows: list[dict] = []
    for plot_id in plot_ids:
        bundle = plot_bundles[plot_id]
        stage1 = stage1_outputs[plot_id]
        label_reference = resolve_reference_points(bundle, label_policy, snap_radius_m=snap_radius_m, mixed_weight=mixed_weight)
        segment_counts, segment_duplicates = _segment_gt_stats(stage1, bundle)
        if mode == "segment-centered" and not stage1.segment_table.empty:
            source_table = stage1.segment_table.copy()
            source_table["height_pred"] = source_table["height_max_m"]
            source_table["geometry_score"] = source_table["compactness"]
        else:
            source_table = stage1.candidates.copy()
        if source_table.empty:
            continue

        source_table = source_table.reset_index(drop=True)
        centers = source_table[["x", "y"]].to_numpy(dtype=float)
        if label_reference.empty:
            distance_matrix = np.full((len(source_table), 0), np.inf, dtype=float)
        else:
            gt_xy = label_reference[["target_x", "target_y"]].to_numpy(dtype=float)
            distance_matrix = np.sqrt(((centers[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))
        if distance_matrix.size:
            nearest_gt = np.argmin(distance_matrix, axis=1)
            nearest_distance = distance_matrix[np.arange(len(source_table)), nearest_gt]
            positive_mask = nearest_distance <= float(positive_radius_m)
            gt_match_counts = pd.Series(nearest_gt[positive_mask]).value_counts().to_dict()
        else:
            nearest_gt = np.full(len(source_table), -1, dtype=int)
            positive_mask = np.zeros(len(source_table), dtype=bool)
            gt_match_counts = {}

        for row_index, row in source_table.iterrows():
            record = row.to_dict()
            record["plot_id"] = plot_id
            record["mode"] = mode
            record["label_policy"] = label_policy
            record["objectness"] = 0.0
            record["dx_m"] = 0.0
            record["dy_m"] = 0.0
            record["delta_h_m"] = 0.0
            record["height_target_m"] = np.nan
            record["split_flag"] = 0.0
            record["duplicate_flag"] = 0.0
            if "segment_id" in record:
                segment_id = int(record.get("segment_id") or 0)
                record["split_flag"] = float(segment_counts.get(segment_id, 0) > 1)
                record["duplicate_flag"] = float(segment_duplicates.get(segment_id, 0))
            if int(row_index) < len(positive_mask) and bool(positive_mask[int(row_index)]):
                gt_index = int(nearest_gt[int(row_index)])
                gt_row = label_reference.iloc[gt_index]
                record["objectness"] = 1.0
                record["dx_m"] = float(gt_row["target_x"]) - float(row["x"])
                record["dy_m"] = float(gt_row["target_y"]) - float(row["y"])
                if pd.notna(gt_row.get("height")) and pd.notna(row.get("height_pred")):
                    record["delta_h_m"] = float(gt_row["height"]) - float(row.get("height_pred", 0.0))
                    record["height_target_m"] = float(gt_row["height"])
                record["duplicate_flag"] = float(gt_match_counts.get(int(gt_index), 0) > 1)
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows).sort_values(["plot_id", "y", "x"]).reset_index(drop=True)


class LocalProposalDataset(Dataset):
    def __init__(
        self,
        plot_bundles: dict[str, ProcessedPlotBundle],
        stage1_outputs: dict[str, Stage1Output],
        sample_table: pd.DataFrame,
        patch_size: int = 96,
        point_radius_m: float = 2.5,
        n_points: int = 256,
        use_rgb: bool = True,
        use_lidar: bool = True,
        use_chm: bool = True,
        include_geom_features: bool = True,
        include_raster: bool = True,
        include_point_cloud: bool = True,
        geom_feature_set: str = "base",
        peakness_cache: dict[str, np.ndarray] | None = None,
        raw_points: dict[str, np.ndarray] | None = None,
        raw_point_index: dict[str, object] | None = None,
        training: bool = False,
        seed: int = 13,
    ) -> None:
        self.plot_bundles = plot_bundles
        self.stage1_outputs = stage1_outputs
        self.sample_table = sample_table.reset_index(drop=True).copy()
        self.patch_size = int(patch_size)
        self.point_radius_m = float(point_radius_m)
        self.n_points = int(n_points)
        self.use_rgb = bool(use_rgb)
        self.use_lidar = bool(use_lidar)
        self.use_chm = bool(use_chm)
        self.include_geom_features = bool(include_geom_features)
        self.include_raster = bool(include_raster)
        self.include_point_cloud = bool(include_point_cloud)
        self.geom_feature_set = str(geom_feature_set)
        self.training = bool(training)
        self.rng = np.random.default_rng(int(seed))
        active_plot_ids = (
            [str(plot_id) for plot_id in self.sample_table["plot_id"].dropna().unique().tolist()]
            if "plot_id" in self.sample_table.columns
            else list(plot_bundles.keys())
        )
        if peakness_cache is None or (self.include_point_cloud and (raw_points is None or raw_point_index is None)):
            default_peakness, default_raw_points, default_raw_point_index = build_local_dataset_caches(
                plot_bundles,
                include_point_cloud=self.include_point_cloud,
                plot_ids=active_plot_ids,
            )
        else:
            default_peakness, default_raw_points, default_raw_point_index = {}, {}, {}
        self.peakness_cache = peakness_cache if peakness_cache is not None else default_peakness
        self.raw_points = raw_points if raw_points is not None else default_raw_points
        self.raw_point_index = raw_point_index if raw_point_index is not None else default_raw_point_index
        self.geom_feature_dim = 13 if self.geom_feature_set == "rich" else 7

    def __len__(self) -> int:
        return len(self.sample_table)

    def _build_geom_features(
        self,
        row: pd.Series,
        bundle: ProcessedPlotBundle,
        rgb_patch: np.ndarray,
        lidar_patch: np.ndarray,
        chm_patch: np.ndarray | None,
        peak_patch: np.ndarray | None,
    ) -> np.ndarray:
        base = np.array(
            [
                float(row.get("area_m2", 0.0)),
                float(row.get("height_pred", row.get("height_max_m", 0.0))),
                float(row.get("geometry_score", row.get("confidence", 0.0))),
                float(row.get("elongation", 0.0)),
                float(row.get("compactness", 0.0)),
                float(row.get("split_flag", 0.0)),
                float(row.get("duplicate_flag", 0.0)),
            ],
            dtype=np.float32,
        )
        if self.geom_feature_set != "rich":
            geom_scale = np.array([20.0, 35.0, 1.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float32)
            return base / np.maximum(geom_scale, 1e-6)

        height_pred = float(row.get("height_pred", row.get("height_max_m", 0.0)))
        height_mean = float(row.get("height_mean_m", height_pred))
        height_gap = max(height_pred - height_mean, 0.0)

        rgb_std = 0.0
        rgb_contrast = 0.0
        if rgb_patch.size:
            rgb_std = float(np.std(rgb_patch))
            center_slice = slice(max(self.patch_size // 2 - 8, 0), min(self.patch_size // 2 + 8, self.patch_size))
            center_rgb = rgb_patch[:, center_slice, center_slice]
            border_mask = np.ones(rgb_patch.shape[1:], dtype=bool)
            border_mask[center_slice, center_slice] = False
            if border_mask.any():
                border_rgb = rgb_patch[:, border_mask].reshape(rgb_patch.shape[0], -1)
                rgb_contrast = float(np.mean(np.abs(center_rgb.mean(axis=(1, 2)) - border_rgb.mean(axis=1))))

        roughness = 0.0
        if lidar_patch.size:
            channel_names = (bundle.metadata or {}).get("structural_channels", [])
            if "vertical_roughness" in channel_names:
                idx = int(channel_names.index("vertical_roughness"))
                roughness = float(np.mean(lidar_patch[idx]))
            else:
                roughness = float(np.std(lidar_patch[0]))
        elif chm_patch is not None:
            roughness = float(np.std(chm_patch))

        peak_distance_m = 0.0
        if peak_patch is not None and peak_patch.size:
            peak_map = peak_patch[0]
            max_index = int(np.argmax(peak_map))
            peak_row, peak_col = np.unravel_index(max_index, peak_map.shape)
            center_row = peak_map.shape[0] // 2
            center_col = peak_map.shape[1] // 2
            peak_distance_m = float(np.hypot(peak_row - center_row, peak_col - center_col) * bundle.transform.pixel_size)

        rich = np.concatenate(
            [
                base,
                np.array(
                    [
                        height_mean,
                        height_gap,
                        rgb_std / 255.0,
                        rgb_contrast / 255.0,
                        roughness,
                        peak_distance_m,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        geom_scale = np.array([20.0, 35.0, 1.0, 10.0, 1.0, 1.0, 1.0, 35.0, 15.0, 0.5, 0.5, 5.0, 5.0], dtype=np.float32)
        return rich / np.maximum(geom_scale, 1e-6)

    def __getitem__(self, index: int) -> dict:
        row = self.sample_table.iloc[index]
        bundle = self.plot_bundles[row["plot_id"]]
        center_row, center_col = bundle.transform.xy_to_rowcol(float(row["x"]), float(row["y"]))

        rgb_patch = extract_centered_patch(bundle.rgb, center_row, center_col, self.patch_size).astype(np.float32)
        lidar_patch = extract_centered_patch(bundle.lidar, center_row, center_col, self.patch_size).astype(np.float32)
        chm_patch = extract_centered_patch(bundle.chm, center_row, center_col, self.patch_size)[None, ...].astype(np.float32)
        peak_patch = extract_centered_patch(self.peakness_cache[row["plot_id"]], center_row, center_col, self.patch_size)[None, ...].astype(np.float32)

        if self.include_raster:
            channels: list[np.ndarray] = []
            if self.use_rgb:
                channels.append(rgb_patch)
            if self.use_lidar:
                channels.append(lidar_patch)
            if self.use_chm:
                channels.extend([chm_patch, peak_patch])
            raster = np.concatenate(channels, axis=0).astype(np.float32) if channels else np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)
        else:
            raster = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)

        center_z = float(row.get("height_pred", row.get("height_max_m", 0.0)))
        point_cloud = (
            _sample_local_points(
                self.raw_points[row["plot_id"]],
                self.raw_point_index.get(row["plot_id"]),
                center_x=float(row["x"]),
                center_y=float(row["y"]),
                center_z=center_z,
                radius_m=self.point_radius_m,
                n_points=self.n_points,
                training=self.training,
                rng=self.rng,
            )
            if self.include_point_cloud
            else np.zeros((self.n_points, 6), dtype=np.float32)
        )
        geom_features = self._build_geom_features(
            row=row,
            bundle=bundle,
            rgb_patch=rgb_patch,
            lidar_patch=lidar_patch,
            chm_patch=chm_patch,
            peak_patch=peak_patch,
        )
        return {
            "raster": torch.from_numpy(raster),
            "point_cloud": torch.from_numpy(point_cloud).transpose(0, 1),
            "geom_features": torch.from_numpy(geom_features if self.include_geom_features else np.zeros_like(geom_features)),
            "objectness": torch.tensor(float(row.get("objectness", 0.0)), dtype=torch.float32),
            "offset": torch.tensor([float(row.get("dx_m", 0.0)), float(row.get("dy_m", 0.0))], dtype=torch.float32),
            "delta_h": torch.tensor(float(row.get("delta_h_m", 0.0)), dtype=torch.float32),
            "height_mask": torch.tensor(float(pd.notna(row.get("height_target_m"))), dtype=torch.float32),
            "split_flag": torch.tensor(float(row.get("split_flag", 0.0)), dtype=torch.float32),
            "duplicate_flag": torch.tensor(float(row.get("duplicate_flag", 0.0)), dtype=torch.float32),
            "plot_id": row["plot_id"],
            "proposal_id": int(row.get("proposal_id", index)),
            "x": torch.tensor(float(row["x"]), dtype=torch.float32),
            "y": torch.tensor(float(row["y"]), dtype=torch.float32),
            "height_pred": torch.tensor(float(row.get("height_pred", center_z)), dtype=torch.float32),
            "segment_id": int(row.get("segment_id", -1)),
        }

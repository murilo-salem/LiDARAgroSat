from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from lidaragrosat.baselines.local_maxima import LocalMaximaParams, detect_local_maxima
from lidaragrosat.data.auxiliary import compute_multiscale_peakness, encode_species_label
from lidaragrosat.data.dataset import ProcessedPlotBundle
from lidaragrosat.inference.sliding_window import PredictionMaps, decode_tree_predictions


def extract_centered_patch(array: np.ndarray, center_row: int, center_col: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    pad_before = half
    pad_after = patch_size - half - 1
    if array.ndim == 3:
        padded = np.pad(array, ((0, 0), (pad_before, pad_after), (pad_before, pad_after)), mode="reflect")
        row = center_row + pad_before
        col = center_col + pad_before
        return padded[:, row - half : row - half + patch_size, col - half : col - half + patch_size].astype(np.float32)
    padded = np.pad(array, ((pad_before, pad_after), (pad_before, pad_after)), mode="reflect")
    row = center_row + pad_before
    col = center_col + pad_before
    return padded[row - half : row - half + patch_size, col - half : col - half + patch_size].astype(np.float32)


def deduplicate_candidate_table(candidates: pd.DataFrame, dedup_radius_m: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[np.ndarray] = []
    for _, row in ordered.iterrows():
        point = np.array([float(row["x"]), float(row["y"])], dtype=float)
        if any(np.linalg.norm(point - existing) <= dedup_radius_m for existing in kept_xy):
            continue
        kept_rows.append(row.to_dict())
        kept_xy.append(point)
    return pd.DataFrame.from_records(kept_rows)


def _resolve_density_channel_index(bundle: ProcessedPlotBundle) -> int | None:
    channel_names = (bundle.metadata or {}).get("structural_channels", [])
    if "density_total" not in channel_names:
        return None
    return int(channel_names.index("density_total"))


def attach_candidate_features(candidates: pd.DataFrame, bundle: ProcessedPlotBundle, peakness_map: np.ndarray | None = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    enriched = candidates.copy()
    row_index = enriched["y_px"].round().astype(int).clip(0, bundle.chm.shape[0] - 1)
    col_index = enriched["x_px"].round().astype(int).clip(0, bundle.chm.shape[1] - 1)
    density_index = _resolve_density_channel_index(bundle)
    default_height = pd.Series(bundle.chm[row_index, col_index], index=enriched.index, dtype=float)
    enriched["proposal_confidence"] = pd.to_numeric(enriched.get("confidence", 0.0), errors="coerce").fillna(0.0)
    enriched["proposal_height_pred"] = pd.to_numeric(enriched.get("height_pred", np.nan), errors="coerce").fillna(default_height)
    enriched["local_chm"] = bundle.chm[row_index, col_index].astype(float)
    if peakness_map is None:
        peakness_map = compute_multiscale_peakness(bundle.chm, bundle.transform.pixel_size, window_sizes_m=[0.8, 1.2, 1.6])
    enriched["local_peakness"] = peakness_map[row_index, col_index].astype(float)
    if density_index is not None:
        enriched["local_density"] = bundle.lidar[density_index, row_index, col_index].astype(float)
    else:
        enriched["local_density"] = 0.0
    enriched["source_both"] = ((enriched.get("source_chm", 0) > 0) & (enriched.get("source_heatmap", 0) > 0)).astype(int)
    def _source_key(row) -> str:
        if int(row.get("source_both", 0)) > 0:
            return "both"
        if int(row.get("source_heatmap", 0)) > 0:
            return "heatmap"
        return "chm"
    enriched["source_key"] = enriched.apply(_source_key, axis=1)
    return enriched


def _load_raw_points(raw_root: str | Path, plot_id: str) -> np.ndarray:
    path = Path(raw_root) / plot_id / "lidar.las"
    las = laspy.read(path)
    return np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)


def _compute_point_profile(points: np.ndarray, center_x: float, center_y: float, radius_m: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros(10, dtype=np.float32)
    delta_xy = points[:, :2] - np.array([[center_x, center_y]], dtype=np.float32)
    radial_distances = np.sqrt((delta_xy**2).sum(axis=1))
    local_points = points[radial_distances <= radius_m]
    if len(local_points) == 0:
        return np.zeros(10, dtype=np.float32)
    rel_xy = local_points[:, :2] - np.array([[center_x, center_y]], dtype=np.float32)
    rel_z = local_points[:, 2] - float(np.min(local_points[:, 2]))
    profile = np.array(
        [
            float(len(local_points)),
            float(rel_z.mean()),
            float(rel_z.std()),
            float(np.percentile(rel_z, 25)),
            float(np.percentile(rel_z, 50)),
            float(np.percentile(rel_z, 75)),
            float(np.percentile(rel_z, 95)),
            float(np.sqrt(np.mean(rel_xy[:, 0] ** 2))),
            float(np.sqrt(np.mean(rel_xy[:, 1] ** 2))),
            float(np.sqrt(np.mean((rel_xy**2).sum(axis=1)))),
        ],
        dtype=np.float32,
    )
    scale = np.array([200.0, 30.0, 15.0, 30.0, 30.0, 30.0, 30.0, radius_m, radius_m, radius_m], dtype=np.float32)
    return np.clip(profile / np.maximum(scale, 1e-6), 0.0, 5.0).astype(np.float32)


def generate_chm_candidates(
    bundle: ProcessedPlotBundle,
    window_sizes_m: list[float],
    gaussian_sigmas: list[float],
    min_height_m: float,
    dedup_radius_m: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window_size_m in window_sizes_m:
        for gaussian_sigma in gaussian_sigmas:
            params = LocalMaximaParams(
                window_radius_m=window_size_m,
                gaussian_sigma=gaussian_sigma,
                min_height_m=min_height_m,
            )
            predictions = detect_local_maxima(bundle.chm, bundle.transform, params)
            if predictions.empty:
                continue
            pixel_rows = []
            pixel_cols = []
            for _, prediction in predictions.iterrows():
                row, col = bundle.transform.xy_to_rowcol(float(prediction["x"]), float(prediction["y"]))
                pixel_rows.append(float(row))
                pixel_cols.append(float(col))
            predictions["y_px"] = pixel_rows
            predictions["x_px"] = pixel_cols
            predictions["source_chm"] = 1
            predictions["source_heatmap"] = 0
            rows.append(predictions)
    if not rows:
        return pd.DataFrame(columns=["x", "y", "x_px", "y_px", "confidence", "height_pred", "source_chm", "source_heatmap"])
    merged = pd.concat(rows, ignore_index=True)
    merged["height_pred"] = merged.get("height_pred", bundle.chm[merged["y_px"].astype(int), merged["x_px"].astype(int)])
    return deduplicate_candidate_table(merged, dedup_radius_m=dedup_radius_m)


def generate_heatmap_candidates(
    bundle: ProcessedPlotBundle,
    prediction_maps: PredictionMaps,
    confidence_threshold: float,
    nms_radius_m: float,
    dedup_radius_m: float,
) -> pd.DataFrame:
    predictions = decode_tree_predictions(
        prediction_maps,
        transform=bundle.transform,
        confidence_threshold=confidence_threshold,
        nms_radius_m=nms_radius_m,
        border_margin_px=0,
        chm=bundle.chm,
    )
    if predictions.empty:
        return pd.DataFrame(columns=["x", "y", "x_px", "y_px", "confidence", "height_pred", "source_chm", "source_heatmap"])
    predictions["source_chm"] = 0
    predictions["source_heatmap"] = 1
    if "height_pred" not in predictions.columns:
        rows = predictions["y_px"].round().astype(int).clip(0, bundle.chm.shape[0] - 1)
        cols = predictions["x_px"].round().astype(int).clip(0, bundle.chm.shape[1] - 1)
        predictions["height_pred"] = bundle.chm[rows, cols]
    return deduplicate_candidate_table(predictions, dedup_radius_m=dedup_radius_m)


def merge_proposals(chm_candidates: pd.DataFrame, heatmap_candidates: pd.DataFrame, dedup_radius_m: float) -> pd.DataFrame:
    if chm_candidates.empty and heatmap_candidates.empty:
        return pd.DataFrame(
            columns=["x", "y", "x_px", "y_px", "confidence", "height_pred", "source_chm", "source_heatmap"]
        )
    merged = pd.concat([chm_candidates, heatmap_candidates], ignore_index=True, sort=False).fillna(0)
    ordered = merged.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept: list[dict] = []
    kept_xy: list[np.ndarray] = []
    for _, row in ordered.iterrows():
        point = np.array([float(row["x"]), float(row["y"])], dtype=float)
        duplicate_index = None
        for index, kept_point in enumerate(kept_xy):
            if np.linalg.norm(point - kept_point) <= dedup_radius_m:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(row.to_dict())
            kept_xy.append(point)
            continue
        existing = kept[duplicate_index]
        existing["source_chm"] = int(max(existing.get("source_chm", 0), row.get("source_chm", 0)))
        existing["source_heatmap"] = int(max(existing.get("source_heatmap", 0), row.get("source_heatmap", 0)))
        existing["confidence"] = float(max(existing.get("confidence", 0.0), row.get("confidence", 0.0)))
        existing["height_pred"] = float(max(existing.get("height_pred", 0.0), row.get("height_pred", 0.0)))
    return pd.DataFrame.from_records(kept)


def label_candidate_table(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    positive_match_radius_m: float,
    species_to_index: dict[str, int] | None = None,
    species_label_mode: str = "none",
    fp_bank_confidence_quantile: float = 0.75,
    fp_bank_peakness_threshold: float = 0.35,
    fp_bank_min_height_m: float = 2.0,
) -> pd.DataFrame:
    species_to_index = species_to_index or {}
    if candidates.empty:
        return candidates.copy()
    labeled = candidates.copy()
    labeled["is_positive"] = 0
    labeled["offset_dx_m"] = 0.0
    labeled["offset_dy_m"] = 0.0
    labeled["height_target"] = np.nan
    labeled["species_target"] = -1
    labeled["distance_to_gt_m"] = np.nan
    labeled["is_hard_negative"] = 0

    if reference.empty:
        negative_confidence_threshold = float(labeled["proposal_confidence"].quantile(fp_bank_confidence_quantile)) if "proposal_confidence" in labeled.columns and len(labeled) else 0.0
        labeled["is_hard_negative"] = (
            (labeled.get("proposal_confidence", 0.0) >= negative_confidence_threshold)
            | (labeled.get("source_both", 0) > 0)
            | (labeled.get("local_peakness", 0.0) >= fp_bank_peakness_threshold)
            | (labeled.get("proposal_height_pred", 0.0) >= fp_bank_min_height_m)
        ).astype(int)
        return labeled

    ref_xy = reference[["x", "y"]].to_numpy(dtype=float)
    candidate_xy = labeled[["x", "y"]].to_numpy(dtype=float)
    distances = np.sqrt(((candidate_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2))
    nearest_distances = distances[np.arange(len(labeled)), np.argmin(distances, axis=1)]
    labeled["distance_to_gt_m"] = nearest_distances.astype(float)

    admissible = distances <= positive_match_radius_m
    cost_matrix = distances.copy()
    cost_matrix[~admissible] = 1e6
    candidate_indices, reference_indices = linear_sum_assignment(cost_matrix)
    matched_candidates: set[int] = set()
    for candidate_index, reference_index in zip(candidate_indices, reference_indices):
        if cost_matrix[candidate_index, reference_index] >= 1e6:
            continue
        matched_candidates.add(int(candidate_index))
        target = reference.iloc[int(reference_index)]
        row = labeled.iloc[int(candidate_index)]
        labeled.at[int(candidate_index), "is_positive"] = 1
        labeled.at[int(candidate_index), "offset_dx_m"] = float(target["x"]) - float(row["x"])
        labeled.at[int(candidate_index), "offset_dy_m"] = float(target["y"]) - float(row["y"])
        if target.get("height") is not None and target.get("height") == target.get("height"):
            labeled.at[int(candidate_index), "height_target"] = float(target["height"])
        species_index = encode_species_label(str(target.get("species", "unknown")), species_to_index, species_label_mode)
        labeled.at[int(candidate_index), "species_target"] = species_index

    duplicate_mask = (nearest_distances <= positive_match_radius_m) & (~labeled.index.isin(list(matched_candidates)))
    negative_candidates = labeled[labeled["is_positive"] == 0]
    negative_confidence_threshold = (
        float(negative_candidates["proposal_confidence"].quantile(fp_bank_confidence_quantile))
        if "proposal_confidence" in negative_candidates.columns and len(negative_candidates)
        else 0.0
    )
    labeled["is_hard_negative"] = (
        (labeled["is_positive"] == 0)
        & (
            duplicate_mask
            | (labeled.get("distance_to_gt_m", np.inf) <= positive_match_radius_m)
            | (labeled.get("proposal_confidence", 0.0) >= negative_confidence_threshold)
            | (labeled.get("source_both", 0) > 0)
            | (labeled.get("local_peakness", 0.0) >= fp_bank_peakness_threshold)
            | (labeled.get("proposal_height_pred", 0.0) >= fp_bank_min_height_m)
        )
    ).astype(int)
    return labeled


class CandidatePatchDataset:
    def __init__(
        self,
        plot_bundles: dict[str, ProcessedPlotBundle],
        candidate_table: pd.DataFrame,
        patch_size: int,
        peakness_window_sizes_m: tuple[float, ...] | list[float],
        raw_root: str | Path | None = None,
        point_radius_m: float = 2.0,
        training: bool = False,
    ) -> None:
        self.plot_bundles = plot_bundles
        self.candidate_table = candidate_table.reset_index(drop=True).copy()
        self.patch_size = patch_size
        self.training = training
        self.raw_root = Path(raw_root) if raw_root else None
        self.point_radius_m = float(point_radius_m)
        self.peakness_maps = {
            plot_id: compute_multiscale_peakness(
                bundle.chm,
                pixel_size=bundle.transform.pixel_size,
                window_sizes_m=peakness_window_sizes_m,
            )
            for plot_id, bundle in plot_bundles.items()
        }
        self.raw_point_cache = (
            {plot_id: _load_raw_points(self.raw_root, plot_id) for plot_id in plot_bundles}
            if self.raw_root is not None
            else {}
        )

    def __len__(self) -> int:
        return len(self.candidate_table)

    def __getitem__(self, index: int) -> dict:
        import torch

        row = self.candidate_table.iloc[index]
        bundle = self.plot_bundles[row["plot_id"]]
        center_row = int(round(float(row["y_px"])))
        center_col = int(round(float(row["x_px"])))

        rgb_patch = extract_centered_patch(bundle.rgb, center_row, center_col, self.patch_size)
        lidar_patch = extract_centered_patch(bundle.lidar, center_row, center_col, self.patch_size)
        chm_patch = extract_centered_patch(bundle.chm, center_row, center_col, self.patch_size)[None, ...]
        peakness_patch = extract_centered_patch(self.peakness_maps[row["plot_id"]], center_row, center_col, self.patch_size)[None, ...]
        structural_patch = np.concatenate([lidar_patch, chm_patch, peakness_patch], axis=0).astype(np.float32)
        source_features = np.array([float(row.get("source_chm", 0)), float(row.get("source_heatmap", 0))], dtype=np.float32)
        tabular_features = np.array(
            [
                float(row.get("proposal_confidence", row.get("confidence", 0.0))),
                float(row.get("proposal_height_pred", row.get("height_pred", 0.0))),
                float(row.get("local_chm", 0.0)),
                float(row.get("local_peakness", 0.0)),
                float(row.get("local_density", 0.0)),
                float(row.get("source_both", 0)),
            ],
            dtype=np.float32,
        )
        point_profile = (
            _compute_point_profile(
                self.raw_point_cache.get(row["plot_id"], np.empty((0, 3), dtype=np.float32)),
                center_x=float(row["x"]),
                center_y=float(row["y"]),
                radius_m=self.point_radius_m,
            )
            if self.raw_root is not None
            else np.zeros(10, dtype=np.float32)
        )
        height_target = row.get("height_target", np.nan)
        height_target = np.nan if height_target is None else float(height_target)
        species_target = row.get("species_target", -1)
        species_target = -1 if species_target is None or species_target != species_target else int(species_target)
        is_positive = float(row.get("is_positive", 0))
        is_hard_negative = float(row.get("is_hard_negative", 0))

        return {
            "rgb": torch.from_numpy(rgb_patch),
            "structural": torch.from_numpy(structural_patch),
            "source_features": torch.from_numpy(source_features),
            "tabular_features": torch.from_numpy(tabular_features),
            "point_profile": torch.from_numpy(point_profile),
            "is_positive": torch.tensor(is_positive, dtype=torch.float32),
            "is_hard_negative": torch.tensor(is_hard_negative, dtype=torch.float32),
            "offset": torch.tensor([float(row.get("offset_dx_m", 0.0)), float(row.get("offset_dy_m", 0.0))], dtype=torch.float32),
            "height_target": torch.tensor(float(0.0 if height_target != height_target else height_target), dtype=torch.float32),
            "height_mask": torch.tensor(float(height_target == height_target), dtype=torch.float32),
            "species_target": torch.tensor(species_target, dtype=torch.long),
            "species_mask": torch.tensor(float(species_target >= 0 and is_positive > 0), dtype=torch.float32),
            "plot_id": row["plot_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "proposal_confidence": float(row.get("proposal_confidence", row.get("confidence", 0.0))),
            "source_key": row.get("source_key", "chm"),
        }

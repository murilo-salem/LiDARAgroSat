from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lidaragrosat.data.auxiliary import compute_chm_feature_stack
from lidaragrosat.data.reference import read_reference_table, write_reference_table
from lidaragrosat.data.targets import build_center_targets
from lidaragrosat.utils.geo import GeoTransform
from lidaragrosat.utils.io import ensure_dir, read_json, write_json


@dataclass(slots=True)
class ProcessedPlotBundle:
    plot_id: str
    rgb: np.ndarray
    lidar: np.ndarray
    chm: np.ndarray
    chm_features: np.ndarray | None
    transform: GeoTransform
    reference: pd.DataFrame
    crs: str | None = None
    metadata: dict | None = None

    @property
    def raster_shape(self) -> tuple[int, int]:
        return int(self.chm.shape[0]), int(self.chm.shape[1])


def save_processed_plot(bundle: ProcessedPlotBundle, destination_dir: str | Path) -> None:
    destination_dir = ensure_dir(destination_dir)
    np.save(destination_dir / "rgb.npy", bundle.rgb.astype(np.float32))
    np.save(destination_dir / "lidar.npy", bundle.lidar.astype(np.float32))
    np.save(destination_dir / "chm.npy", bundle.chm.astype(np.float32))
    if bundle.chm_features is not None:
        np.save(destination_dir / "chm_features.npy", bundle.chm_features.astype(np.float32))
    write_reference_table(bundle.reference, destination_dir / "reference.csv")
    write_json(
        {
            "plot_id": bundle.plot_id,
            "transform": asdict(bundle.transform),
            "crs": bundle.crs,
            "metadata": bundle.metadata or {},
        },
        destination_dir / "manifest.json",
    )


def load_processed_plot(processed_plot_dir: str | Path) -> ProcessedPlotBundle:
    processed_plot_dir = Path(processed_plot_dir)
    manifest = read_json(processed_plot_dir / "manifest.json")
    transform = GeoTransform(**manifest["transform"])
    chm_features_path = processed_plot_dir / "chm_features.npy"
    return ProcessedPlotBundle(
        plot_id=manifest["plot_id"],
        rgb=np.load(processed_plot_dir / "rgb.npy"),
        lidar=np.load(processed_plot_dir / "lidar.npy"),
        chm=np.load(processed_plot_dir / "chm.npy"),
        chm_features=np.load(chm_features_path) if chm_features_path.exists() else None,
        transform=transform,
        reference=read_reference_table(processed_plot_dir / "reference.csv"),
        crs=manifest.get("crs"),
        metadata=manifest.get("metadata", {}),
    )


def list_processed_plots(processed_dir: str | Path, plot_ids: list[str] | None = None) -> dict[str, ProcessedPlotBundle]:
    root = Path(processed_dir)
    selected = plot_ids or [path.name for path in sorted(root.iterdir()) if path.is_dir()]
    return {plot_id: load_processed_plot(root / plot_id) for plot_id in selected}


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
        padded = np.pad(cropped, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    else:
        cropped = array[row_start_clipped:row_end_clipped, col_start_clipped:col_end_clipped]
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    return padded.copy()


def _merge_reference_with_pseudo_labels(
    reference: pd.DataFrame,
    pseudo_label_dir: str | Path | None,
    plot_id: str,
    match_radius_m: float,
) -> pd.DataFrame:
    if not pseudo_label_dir:
        return reference.copy()
    pseudo_path = Path(pseudo_label_dir) / f"{plot_id}.csv"
    if not pseudo_path.exists():
        return reference.copy()
    pseudo = pd.read_csv(pseudo_path)
    if pseudo.empty or not {"x", "y"}.issubset(pseudo.columns):
        return reference.copy()
    merged = reference.copy()
    if len(reference):
        ref_xy = reference[["x", "y"]].to_numpy(dtype=float)
        keep_rows = []
        for _, row in pseudo.iterrows():
            point = np.array([float(row["x"]), float(row["y"])], dtype=float)
            distances = np.sqrt(((ref_xy - point[None, :]) ** 2).sum(axis=1))
            if np.any(distances <= match_radius_m):
                continue
            keep_rows.append(row.to_dict())
        pseudo = pd.DataFrame.from_records(keep_rows)
    if pseudo.empty:
        return merged
    pseudo = pseudo.copy()
    pseudo["height"] = pseudo.get("height", np.nan)
    pseudo["species"] = pseudo.get("species", "pseudo")
    return pd.concat([merged, pseudo[merged.columns]], ignore_index=True)


def _apply_geometric_augmentations(sample: dict, augmentation_config, rng: np.random.Generator) -> dict:
    rgb = sample["rgb"]
    lidar = sample["lidar"]
    heatmap = sample["heatmap"]
    offset = sample["offset"]
    height = sample["height"]
    positive_mask = sample["positive_mask"]
    distance = sample["distance"]
    difficulty = sample["difficulty"]
    peakness = sample["peakness"]
    chm_features = sample["chm_features"]
    species = sample["species"]
    species_mask = sample["species_mask"]
    hard_negative_mask = sample["hard_negative_mask"]
    supervision_mask = sample["supervision_mask"]

    if augmentation_config.horizontal_flip and rng.random() < 0.5:
        rgb = np.flip(rgb, axis=2).copy()
        lidar = np.flip(lidar, axis=2).copy()
        heatmap = np.flip(heatmap, axis=1).copy()
        offset = np.flip(offset, axis=2).copy()
        offset[0] *= -1
        height = np.flip(height, axis=1).copy()
        positive_mask = np.flip(positive_mask, axis=1).copy()
        distance = np.flip(distance, axis=1).copy()
        difficulty = np.flip(difficulty, axis=1).copy()
        peakness = np.flip(peakness, axis=1).copy()
        chm_features = np.flip(chm_features, axis=2).copy()
        species = np.flip(species, axis=1).copy()
        species_mask = np.flip(species_mask, axis=1).copy()
        hard_negative_mask = np.flip(hard_negative_mask, axis=1).copy()
        supervision_mask = np.flip(supervision_mask, axis=1).copy()

    if augmentation_config.vertical_flip and rng.random() < 0.5:
        rgb = np.flip(rgb, axis=1).copy()
        lidar = np.flip(lidar, axis=1).copy()
        heatmap = np.flip(heatmap, axis=0).copy()
        offset = np.flip(offset, axis=1).copy()
        offset[1] *= -1
        height = np.flip(height, axis=0).copy()
        positive_mask = np.flip(positive_mask, axis=0).copy()
        distance = np.flip(distance, axis=0).copy()
        difficulty = np.flip(difficulty, axis=0).copy()
        peakness = np.flip(peakness, axis=0).copy()
        chm_features = np.flip(chm_features, axis=1).copy()
        species = np.flip(species, axis=0).copy()
        species_mask = np.flip(species_mask, axis=0).copy()
        hard_negative_mask = np.flip(hard_negative_mask, axis=0).copy()
        supervision_mask = np.flip(supervision_mask, axis=0).copy()

    if augmentation_config.rotate_90:
        k = int(rng.integers(0, 4))
        if k:
            rgb = np.rot90(rgb, k=k, axes=(1, 2)).copy()
            lidar = np.rot90(lidar, k=k, axes=(1, 2)).copy()
            heatmap = np.rot90(heatmap, k=k, axes=(0, 1)).copy()
            height = np.rot90(height, k=k, axes=(0, 1)).copy()
            positive_mask = np.rot90(positive_mask, k=k, axes=(0, 1)).copy()
            distance = np.rot90(distance, k=k, axes=(0, 1)).copy()
            difficulty = np.rot90(difficulty, k=k, axes=(0, 1)).copy()
            peakness = np.rot90(peakness, k=k, axes=(0, 1)).copy()
            chm_features = np.rot90(chm_features, k=k, axes=(1, 2)).copy()
            species = np.rot90(species, k=k, axes=(0, 1)).copy()
            species_mask = np.rot90(species_mask, k=k, axes=(0, 1)).copy()
            hard_negative_mask = np.rot90(hard_negative_mask, k=k, axes=(0, 1)).copy()
            supervision_mask = np.rot90(supervision_mask, k=k, axes=(0, 1)).copy()
            offset = np.rot90(offset, k=k, axes=(1, 2)).copy()
            for _ in range(k):
                offset = np.stack([-offset[1], offset[0]], axis=0)

    if augmentation_config.rgb_color_jitter > 0:
        scale = 1.0 + rng.uniform(-augmentation_config.rgb_color_jitter, augmentation_config.rgb_color_jitter)
        rgb = rgb * scale

    if augmentation_config.lidar_spatial_dropout_prob > 0 and rng.random() < augmentation_config.lidar_spatial_dropout_prob:
        dropout_mask = rng.random(lidar.shape[1:]) < 0.05
        lidar[:, dropout_mask] = 0.0

    sample["rgb"] = rgb.astype(np.float32)
    sample["lidar"] = lidar.astype(np.float32)
    sample["heatmap"] = heatmap.astype(np.float32)
    sample["offset"] = offset.astype(np.float32)
    sample["height"] = height.astype(np.float32)
    sample["positive_mask"] = positive_mask.astype(np.float32)
    sample["distance"] = distance.astype(np.float32)
    sample["difficulty"] = difficulty.astype(np.float32)
    sample["peakness"] = peakness.astype(np.float32)
    sample["chm_features"] = chm_features.astype(np.float32)
    sample["species"] = species.astype(np.int64)
    sample["species_mask"] = species_mask.astype(np.float32)
    sample["hard_negative_mask"] = hard_negative_mask.astype(np.float32)
    sample["supervision_mask"] = supervision_mask.astype(np.float32)
    return sample


class TreeTileDataset:
    def __init__(
        self,
        plot_bundles: dict[str, ProcessedPlotBundle],
        tile_table: pd.DataFrame,
        sigma_m: float,
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
        context_margin_px: int = 0,
        supervision_margin_px: int = 0,
        pseudo_label_dir: str = "",
        pseudo_label_match_radius_m: float = 1.5,
        augmentation_config=None,
        training: bool = False,
    ) -> None:
        self.plot_bundles = plot_bundles
        self.tile_table = tile_table.reset_index(drop=True).copy()
        self.sigma_m = sigma_m
        self.peakness_window_sizes_m = peakness_window_sizes_m
        self.hard_negative_window_sizes_m = hard_negative_window_sizes_m
        self.hard_negative_match_radius_m = hard_negative_match_radius_m
        self.hard_negative_border_margin_px = hard_negative_border_margin_px
        self.hard_negative_dilation_px = hard_negative_dilation_px
        self.hard_negative_rgb_complexity_quantile = hard_negative_rgb_complexity_quantile
        self.min_tree_height_m = min_tree_height_m
        self.species_to_index = species_to_index or {}
        self.species_label_mode = species_label_mode
        self.adaptive_sigma = adaptive_sigma
        self.adaptive_sigma_use_nn_distance = adaptive_sigma_use_nn_distance
        self.adaptive_sigma_nn_scale = adaptive_sigma_nn_scale
        self.adaptive_sigma_a = adaptive_sigma_a
        self.adaptive_sigma_b = adaptive_sigma_b
        self.adaptive_sigma_min_m = adaptive_sigma_min_m
        self.adaptive_sigma_max_m = adaptive_sigma_max_m
        self.snap_centers_to_apex_radius_m = snap_centers_to_apex_radius_m
        self.apex_secondary_weight = apex_secondary_weight
        self.crown_radius_a = crown_radius_a
        self.crown_radius_b = crown_radius_b
        self.crown_radius_min_m = crown_radius_min_m
        self.crown_radius_max_m = crown_radius_max_m
        self.context_margin_px = int(context_margin_px)
        self.supervision_margin_px = int(supervision_margin_px)
        self.pseudo_label_dir = pseudo_label_dir
        self.pseudo_label_match_radius_m = pseudo_label_match_radius_m
        self.augmentation_config = augmentation_config
        self.training = training
        self.rng = np.random.default_rng(1234)
        self.chm_feature_cache = {
            plot_id: (
                bundle.chm_features
                if bundle.chm_features is not None
                else compute_chm_feature_stack(
                    bundle.chm,
                    pixel_size=bundle.transform.pixel_size,
                    peakness_window_sizes_m=self.peakness_window_sizes_m,
                    min_height_m=self.min_tree_height_m,
                )
            )
            for plot_id, bundle in plot_bundles.items()
        }
        self.target_cache = {
            plot_id: build_center_targets(
                _merge_reference_with_pseudo_labels(
                    bundle.reference,
                    self.pseudo_label_dir if training else "",
                    plot_id,
                    self.pseudo_label_match_radius_m,
                ),
                transform=bundle.transform,
                raster_height=bundle.raster_shape[0],
                raster_width=bundle.raster_shape[1],
                sigma_m=self.sigma_m,
                chm=bundle.chm,
                rgb=bundle.rgb,
                peakness_window_sizes_m=self.peakness_window_sizes_m,
                hard_negative_window_sizes_m=self.hard_negative_window_sizes_m,
                hard_negative_match_radius_m=self.hard_negative_match_radius_m,
                hard_negative_border_margin_px=self.hard_negative_border_margin_px,
                hard_negative_dilation_px=self.hard_negative_dilation_px,
                hard_negative_rgb_complexity_quantile=self.hard_negative_rgb_complexity_quantile,
                min_tree_height_m=self.min_tree_height_m,
                species_to_index=self.species_to_index,
                species_label_mode=self.species_label_mode,
                adaptive_sigma=self.adaptive_sigma,
                adaptive_sigma_use_nn_distance=self.adaptive_sigma_use_nn_distance,
                adaptive_sigma_nn_scale=self.adaptive_sigma_nn_scale,
                adaptive_sigma_a=self.adaptive_sigma_a,
                adaptive_sigma_b=self.adaptive_sigma_b,
                adaptive_sigma_min_m=self.adaptive_sigma_min_m,
                adaptive_sigma_max_m=self.adaptive_sigma_max_m,
                snap_centers_to_apex_radius_m=self.snap_centers_to_apex_radius_m,
                apex_secondary_weight=self.apex_secondary_weight,
                crown_radius_a=self.crown_radius_a,
                crown_radius_b=self.crown_radius_b,
                crown_radius_min_m=self.crown_radius_min_m,
                crown_radius_max_m=self.crown_radius_max_m,
            )
            for plot_id, bundle in plot_bundles.items()
        }

    def __len__(self) -> int:
        return len(self.tile_table)

    def __getitem__(self, index: int) -> dict:
        import torch

        tile = self.tile_table.iloc[index]
        bundle = self.plot_bundles[tile["plot_id"]]
        row = int(tile["row"])
        col = int(tile["col"])
        size = int(tile["size"])
        context_size = size + 2 * self.context_margin_px
        context_row = row - self.context_margin_px
        context_col = col - self.context_margin_px

        rgb = _extract_with_padding(bundle.rgb, context_row, context_col, context_size)
        lidar = _extract_with_padding(bundle.lidar, context_row, context_col, context_size)
        chm = _extract_with_padding(bundle.chm, context_row, context_col, context_size)
        chm_features = _extract_with_padding(self.chm_feature_cache[tile["plot_id"]], context_row, context_col, context_size)

        tile_targets = self.target_cache[tile["plot_id"]]
        supervision_mask = np.zeros((context_size, context_size), dtype=np.float32)
        center_margin = self.context_margin_px + self.supervision_margin_px
        center_limit = self.context_margin_px + size - self.supervision_margin_px
        supervision_mask[center_margin:center_limit, center_margin:center_limit] = 1.0
        sample = {
            "rgb": rgb,
            "lidar": lidar,
            "chm_features": chm_features,
            "chm": chm.astype(np.float32),
            "heatmap": _extract_with_padding(tile_targets.heatmap, context_row, context_col, context_size),
            "offset": _extract_with_padding(tile_targets.offset, context_row, context_col, context_size),
            "height": _extract_with_padding(tile_targets.height, context_row, context_col, context_size),
            "positive_mask": _extract_with_padding(tile_targets.positive_mask, context_row, context_col, context_size),
            "distance": _extract_with_padding(tile_targets.distance, context_row, context_col, context_size),
            "difficulty": _extract_with_padding(tile_targets.difficulty, context_row, context_col, context_size),
            "peakness": _extract_with_padding(tile_targets.peakness, context_row, context_col, context_size),
            "species": _extract_with_padding(tile_targets.species, context_row, context_col, context_size),
            "species_mask": _extract_with_padding(tile_targets.species_mask, context_row, context_col, context_size),
            "hard_negative_mask": _extract_with_padding(tile_targets.hard_negative_mask, context_row, context_col, context_size),
            "supervision_mask": supervision_mask,
        }

        if self.training and self.augmentation_config is not None:
            sample = _apply_geometric_augmentations(sample, self.augmentation_config, self.rng)

        return {
            "rgb": torch.from_numpy(sample["rgb"]),
            "lidar": torch.from_numpy(sample["lidar"]),
            "chm_features": torch.from_numpy(sample["chm_features"]),
            "chm": torch.from_numpy(sample["chm"]),
            "heatmap": torch.from_numpy(sample["heatmap"]).unsqueeze(0),
            "offset": torch.from_numpy(sample["offset"]),
            "height": torch.from_numpy(np.nan_to_num(sample["height"], nan=0.0)).unsqueeze(0),
            "height_mask": torch.from_numpy(np.isfinite(sample["height"]).astype(np.float32)).unsqueeze(0),
            "positive_mask": torch.from_numpy(sample["positive_mask"]).unsqueeze(0),
            "distance": torch.from_numpy(sample["distance"]).unsqueeze(0),
            "difficulty": torch.from_numpy(sample["difficulty"]).unsqueeze(0),
            "peakness": torch.from_numpy(sample["peakness"]).unsqueeze(0),
            "species_target": torch.from_numpy(sample["species"]).long(),
            "species_mask": torch.from_numpy(sample["species_mask"]).unsqueeze(0),
            "hard_negative_mask": torch.from_numpy(sample["hard_negative_mask"]).unsqueeze(0),
            "supervision_mask": torch.from_numpy(sample["supervision_mask"]).unsqueeze(0),
            "plot_id": tile["plot_id"],
            "row": row,
            "col": col,
        }

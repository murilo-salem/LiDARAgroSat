from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class PathConfig:
    root_dir: str = "."
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    experiments_dir: str = "data/experiments"
    output_dir: str = "outputs"


@dataclass(slots=True)
class DataConfig:
    plot_ids: list[str] = field(default_factory=list)
    split_mode: str = "lopo"
    basic_train_fraction: float = 0.8
    basic_val_fraction: float = 0.1
    basic_split_seed: int = 13
    label_policy: str = "field_center"
    label_snap_radius_m: float = 1.0
    label_mixed_weight: float = 0.35
    raster_resolution: float = 0.10
    alternate_resolutions: list[float] = field(default_factory=lambda: [0.20])
    tile_size: int = 512
    tile_overlap: int = 96
    target_sigma_m: float = 0.6
    adaptive_target_sigma: bool = False
    adaptive_sigma_use_nn_distance: bool = False
    adaptive_sigma_nn_scale: float = 0.35
    adaptive_sigma_a: float = 0.25
    adaptive_sigma_b: float = 0.03
    adaptive_sigma_min_m: float = 0.45
    adaptive_sigma_max_m: float = 1.20
    snap_centers_to_apex_radius_m: float = 0.0
    apex_secondary_weight: float = 0.0
    crown_radius_a: float = 0.8
    crown_radius_b: float = 0.04
    crown_radius_min_m: float = 0.8
    crown_radius_max_m: float = 2.5
    context_margin_px: int = 0
    supervision_margin_px: int = 0
    pseudo_label_dir: str = ""
    pseudo_label_match_radius_m: float = 1.5
    pseudo_label_weight: float = 0.25
    target_sigma_candidates_m: list[float] = field(default_factory=lambda: [0.4, 0.6, 0.8, 1.0])
    min_tree_height_m: float = 2.0
    species_label_mode: str = "none"
    peakness_window_sizes_m: list[float] = field(default_factory=lambda: [0.8, 1.2, 1.6])
    hard_negative_window_sizes_m: list[float] = field(default_factory=lambda: [0.8, 1.2, 1.6])
    hard_negative_match_radius_m: float = 1.5
    hard_negative_border_margin_px: int = 16
    hard_negative_dilation_px: int = 3
    hard_negative_rgb_complexity_quantile: float = 0.9
    structural_channels: list[str] = field(
        default_factory=lambda: [
            "zmax",
            "zmean",
            "zstd",
            "zp95",
            "density_total",
            "density_0p5_2",
            "density_2_5",
            "density_5_10",
            "density_10_20",
            "density_gt20",
            "cover_gt2",
            "cover_gt5",
            "vertical_roughness",
        ]
    )
    reference_species_groups: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PreprocessConfig:
    dtm_cell_size: float = 1.0
    ground_quantile: float = 0.05
    ground_residual_threshold_m: float = 0.35
    noise_z_quantiles: tuple[float, float] = (0.001, 0.999)
    min_normalized_height_m: float = 0.0
    max_normalized_height_m: float = 80.0
    rgb_clip_percentiles: tuple[float, float] = (1.0, 99.0)
    rgb_standardize: bool = True
    chm_gaussian_sigmas: list[float] = field(default_factory=lambda: [0.7, 1.0, 1.5])
    estimate_multimodal_shift: bool = True
    apply_multimodal_shift_correction: bool = False
    max_registration_shift_px: int = 8
    clip_reference_to_bounds: bool = True
    reference_height_min_m: float = 0.0
    reference_height_max_m: float = 80.0
    persist_chm_features: bool = True


@dataclass(slots=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_90: bool = True
    small_angle_rotation_deg: float = 8.0
    translation_px: int = 8
    rgb_color_jitter: float = 0.08
    rgb_blur_prob: float = 0.15
    lidar_spatial_dropout_prob: float = 0.10


@dataclass(slots=True)
class ModelConfig:
    experiment_name: str = "multimodal_mid_fusion"
    modality: str = "multimodal"
    fusion_strategy: str = "mid"
    rgb_in_channels: int = 3
    lidar_in_channels: int = 13
    chm_in_channels: int = 6
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    decoder_channels: list[int] = field(default_factory=lambda: [160, 96, 64, 32])
    normalization: str = "group"
    dropout: float = 0.10
    use_height_head: bool = True
    use_offsets: bool = True
    use_chm_branch: bool = False
    use_distance_head: bool = False
    use_peakness_head: bool = False
    use_species_head: bool = False
    species_num_classes: int = 0
    use_deep_supervision: bool = False


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 13
    epochs: int = 120
    validation_interval_epochs: int = 1
    min_epochs_before_stop: int = 0
    accumulation_steps: int = 1
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    scheduler: str = "cosine"
    patience: int = 20
    positive_pixel_weight: float = 8.0
    lambda_heatmap: float = 1.0
    lambda_offset: float = 1.0
    lambda_height: float = 0.5
    lambda_distance: float = 0.3
    lambda_peakness: float = 0.3
    lambda_species: float = 0.2
    deep_supervision_weight: float = 0.3
    hard_negative_pixel_weight: float = 2.0
    difficulty_positive_weight: float = 1.5
    validation_plots_per_fold: int = 1
    confidence_threshold: float = 0.35
    checkpoint_metric: str = "f1"
    sample_negative_tile_ratio: float = 0.25
    sample_hard_negative_tile_ratio: float = 0.5


@dataclass(slots=True)
class EvaluationConfig:
    match_radius_m: float = 1.5
    radius_candidates_m: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    height_tolerance_m: float = 3.0
    height_tolerance_candidates_m: list[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    nms_radius_m: float = 1.2
    confidence_thresholds: list[float] = field(
        default_factory=lambda: [round(0.05 * step, 2) for step in range(1, 19)]
    )
    baseline_window_sizes_m: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    baseline_min_heights_m: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.0])
    bootstrap_iterations: int = 2000
    tune_postprocess: bool = False
    confidence_threshold_grid: list[float] = field(
        default_factory=lambda: [round(0.05 * step, 2) for step in range(1, 19)]
    )
    nms_radius_grid_m: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
    min_pred_height_grid_m: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    min_chm_at_center_grid_m: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    border_margin_grid_px: list[int] = field(default_factory=lambda: [0, 8, 16, 24])
    use_tta: bool = False
    adaptive_nms: bool = False
    adaptive_nms_r0_m: float = 0.6
    adaptive_nms_alpha: float = 0.03
    adaptive_nms_rmin_m: float = 0.8
    adaptive_nms_rmax_m: float = 1.8


@dataclass(slots=True)
class HybridConfig:
    proposal_source_mode: str = "union"
    proposal_output_dir: str = "outputs_basic_split"
    proposal_method_name: str = "multimodal_mid_fusion"
    proposal_confidence_threshold: float = 0.10
    proposal_nms_radius_m: float = 0.8
    proposal_window_sizes_m: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    proposal_gaussian_sigmas: list[float] = field(default_factory=lambda: [0.7, 1.0])
    proposal_min_height_m: float = 2.0
    proposal_dedup_radius_m: float = 0.8
    candidate_patch_size: int = 96
    candidate_max_per_plot: int = 4096
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 10
    positive_match_radius_m: float = 1.5
    negative_subsample_ratio: float = 2.0
    hard_negative_subsample_ratio: float = 2.0
    fp_bank_confidence_quantile: float = 0.75
    fp_bank_peakness_threshold: float = 0.35
    fp_bank_min_height_m: float = 2.0
    lambda_cls_focal_alpha: float = 0.25
    lambda_cls_focal_gamma: float = 2.0
    hard_negative_cls_weight: float = 2.0
    source_specific_calibration: bool = True
    use_point_profile_branch: bool = False
    point_radius_m: float = 2.0
    classification_threshold_grid: list[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    )
    final_nms_radius_grid_m: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    final_min_height_grid_m: list[float] = field(default_factory=lambda: [0.0, 1.5, 2.0, 2.5])


@dataclass(slots=True)
class AnalysisConfig:
    height_bins_m: list[float] = field(default_factory=lambda: [0.0, 10.0, 20.0, 30.0, 60.0])
    local_density_radius_m: float = 6.0
    overlap_radius_m: float = 3.0
    save_figures: bool = True
    dpi: int = 180


@dataclass(slots=True)
class BenchmarkConfig:
    protocol_id: str = ""
    dataset_id: str = "dubrovin_kaggle_v1"
    family: str = ""
    stage1: str = ""
    stage2: str = ""
    label_policy: str = "field_center"


@dataclass(slots=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_path(self, path_value: str) -> Path:
        root = Path(self.paths.root_dir).resolve()
        return (root / path_value).resolve()

    @classmethod
    def from_files(cls, *config_paths: str | Path) -> "ExperimentConfig":
        merged: dict[str, Any] = {}
        for config_path in config_paths:
            with Path(config_path).open("r", encoding="utf-8") as handle:
                current = yaml.safe_load(handle) or {}
            merged = _merge_dicts(merged, current)

        return cls(
            paths=PathConfig(**merged.get("paths", {})),
            data=DataConfig(**merged.get("data", {})),
            preprocessing=PreprocessConfig(**merged.get("preprocessing", {})),
            augmentation=AugmentationConfig(**merged.get("augmentation", {})),
            model=ModelConfig(**merged.get("model", {})),
            training=TrainingConfig(**merged.get("training", {})),
            evaluation=EvaluationConfig(**merged.get("evaluation", {})),
            hybrid=HybridConfig(**merged.get("hybrid", {})),
            analysis=AnalysisConfig(**merged.get("analysis", {})),
            benchmark=BenchmarkConfig(**merged.get("benchmark", {})),
        )


def save_config(config: ExperimentConfig, destination: str | Path) -> None:
    with Path(destination).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

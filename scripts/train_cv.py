from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.benchmarking import require_protocol_id
from lidaragrosat.data.auxiliary import build_species_lookup, compute_hard_negative_mask
from lidaragrosat.data.dataset import TreeTileDataset, list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import create_experiment_manifest, save_experiment_manifest
from lidaragrosat.inference.sliding_window import (
    build_chm_features_if_needed,
    decode_tree_predictions,
    predict_plot_sliding_window,
    predict_plot_sliding_window_tta,
)
from lidaragrosat.labels import resolve_label_policy_parameters
from lidaragrosat.models.detector import build_detector_from_config
from lidaragrosat.training.engine import fit_model
from lidaragrosat.utils.io import ensure_dir, write_json
from lidaragrosat.utils.logging import configure_logging, get_logger
from lidaragrosat.utils.seed import seed_everything


def _load_tile_table(plot_dir: Path) -> pd.DataFrame:
    return pd.read_csv(plot_dir / "tiles.csv")


def _sample_training_tiles(tile_table: pd.DataFrame, negative_ratio: float, seed: int) -> pd.DataFrame:
    positives = tile_table[tile_table["tree_count"] > 0]
    negatives = tile_table[tile_table["tree_count"] == 0]
    n_negative = min(len(negatives), max(1, int(len(positives) * negative_ratio))) if len(positives) else min(len(negatives), 32)
    sampled_negatives = negatives.sample(n=n_negative, random_state=seed) if n_negative and len(negatives) else negatives.iloc[0:0]
    return pd.concat([positives, sampled_negatives], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _annotate_hard_negative_tiles(tile_table: pd.DataFrame, bundle, config: ExperimentConfig) -> pd.DataFrame:
    hard_negative_mask = compute_hard_negative_mask(
        reference_table=bundle.reference,
        transform=bundle.transform,
        chm=bundle.chm,
        rgb=bundle.rgb,
        window_sizes_m=config.data.hard_negative_window_sizes_m,
        match_radius_m=config.data.hard_negative_match_radius_m,
        min_height_m=config.data.min_tree_height_m,
        border_margin_px=config.data.hard_negative_border_margin_px,
        dilation_px=config.data.hard_negative_dilation_px,
        rgb_complexity_quantile=config.data.hard_negative_rgb_complexity_quantile,
    )
    counts = []
    for _, tile in tile_table.iterrows():
        row = int(tile["row"])
        col = int(tile["col"])
        size = int(tile["size"])
        counts.append(float(hard_negative_mask[row : row + size, col : col + size].sum()))
    enriched = tile_table.copy()
    enriched["hard_negative_count"] = counts
    return enriched


def _sample_training_tiles_with_hard_negatives(
    tile_table: pd.DataFrame,
    negative_ratio: float,
    hard_negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    positives = tile_table[tile_table["tree_count"] > 0]
    hard_negatives = tile_table[(tile_table["tree_count"] == 0) & (tile_table["hard_negative_count"] > 0)]
    random_negatives = tile_table[(tile_table["tree_count"] == 0) & (tile_table["hard_negative_count"] <= 0)]

    if len(positives):
        n_random = min(len(random_negatives), max(1, int(len(positives) * negative_ratio)))
        n_hard = min(len(hard_negatives), max(1, int(len(positives) * hard_negative_ratio)))
    else:
        n_random = min(len(random_negatives), 32)
        n_hard = min(len(hard_negatives), 32)

    sampled_random = random_negatives.sample(n=n_random, random_state=seed) if n_random and len(random_negatives) else random_negatives.iloc[0:0]
    sampled_hard = hard_negatives.sample(n=n_hard, random_state=seed + 17) if n_hard and len(hard_negatives) else hard_negatives.iloc[0:0]
    return (
        pd.concat([positives, sampled_random, sampled_hard], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _predict_plot(model, bundle, config: ExperimentConfig, device: str, use_tta: bool | None = None):
    if use_tta is None:
        use_tta = config.evaluation.use_tta
    predict_fn = predict_plot_sliding_window_tta if use_tta else predict_plot_sliding_window
    chm_features = build_chm_features_if_needed(
        model=model,
        chm=bundle.chm,
        pixel_size=bundle.transform.pixel_size,
        peakness_window_sizes_m=config.data.peakness_window_sizes_m,
        min_height_m=config.data.min_tree_height_m,
        precomputed=getattr(bundle, "chm_features", None),
    )
    return predict_fn(
        model,
        rgb=bundle.rgb,
        lidar=bundle.lidar,
        tile_size=config.data.tile_size,
        overlap=config.data.tile_overlap,
        chm=chm_features,
        context_margin_px=config.data.context_margin_px,
        device=device,
    )


def _decode_predictions(bundle, prediction_maps, params: dict, config: ExperimentConfig) -> pd.DataFrame:
    return decode_tree_predictions(
        prediction_maps,
        transform=bundle.transform,
        confidence_threshold=float(params["confidence_threshold"]),
        nms_radius_m=float(params["nms_radius_m"]),
        min_pred_height_m=(None if params["min_pred_height_m"] is None else float(params["min_pred_height_m"])),
        min_chm_at_center_m=(None if params["min_chm_at_center_m"] is None else float(params["min_chm_at_center_m"])),
        border_margin_px=int(params["border_margin_px"]),
        chm=bundle.chm,
        adaptive_nms=config.evaluation.adaptive_nms,
        adaptive_nms_r0_m=config.evaluation.adaptive_nms_r0_m,
        adaptive_nms_alpha=config.evaluation.adaptive_nms_alpha,
        adaptive_nms_rmin_m=config.evaluation.adaptive_nms_rmin_m,
        adaptive_nms_rmax_m=config.evaluation.adaptive_nms_rmax_m,
    )


def _default_postprocess_params(config: ExperimentConfig) -> dict:
    return {
        "confidence_threshold": config.training.confidence_threshold,
        "nms_radius_m": config.evaluation.nms_radius_m,
        "min_pred_height_m": None,
        "min_chm_at_center_m": None,
        "border_margin_px": 0,
    }


def _score_predictions(predictions_by_plot: dict[str, pd.DataFrame], bundles, plot_ids: list[str], config: ExperimentConfig) -> dict:
    plot_scores = []
    for plot_id in plot_ids:
        matches = match_tree_predictions(
            predictions_by_plot[plot_id],
            bundles[plot_id].reference,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        summary = summarize_matches(matches)
        plot_scores.append(summary)
    if not plot_scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(sum(score["precision"] for score in plot_scores) / len(plot_scores)),
        "recall": float(sum(score["recall"] for score in plot_scores) / len(plot_scores)),
        "f1": float(sum(score["f1"] for score in plot_scores) / len(plot_scores)),
    }


def _tune_postprocess_params(prediction_maps_by_plot: dict, bundles, plot_ids: list[str], config: ExperimentConfig, logger) -> tuple[dict, dict]:
    default_params = _default_postprocess_params(config)
    default_predictions = {
        plot_id: _decode_predictions(bundles[plot_id], prediction_maps_by_plot[plot_id], default_params, config)
        for plot_id in plot_ids
    }
    best_params = default_params.copy()
    best_scores = _score_predictions(default_predictions, bundles, plot_ids, config)

    if not config.evaluation.tune_postprocess:
        return best_params, best_scores

    grid = product(
        config.evaluation.confidence_threshold_grid,
        config.evaluation.nms_radius_grid_m,
        config.evaluation.min_pred_height_grid_m,
        config.evaluation.min_chm_at_center_grid_m,
        config.evaluation.border_margin_grid_px,
    )

    for confidence_threshold, nms_radius_m, min_pred_height_m, min_chm_at_center_m, border_margin_px in grid:
        params = {
            "confidence_threshold": confidence_threshold,
            "nms_radius_m": nms_radius_m,
            "min_pred_height_m": min_pred_height_m,
            "min_chm_at_center_m": min_chm_at_center_m,
            "border_margin_px": border_margin_px,
        }
        predictions_by_plot = {
            plot_id: _decode_predictions(bundles[plot_id], prediction_maps_by_plot[plot_id], params, config)
            for plot_id in plot_ids
        }
        scores = _score_predictions(predictions_by_plot, bundles, plot_ids, config)
        if (
            scores["f1"] > best_scores["f1"]
            or (
                scores["f1"] == best_scores["f1"]
                and scores["recall"] > best_scores["recall"]
            )
        ):
            best_params = params
            best_scores = scores

    logger.info(
        "Best decode params on validation: conf=%.3f nms=%.2f min_pred_h=%s min_chm=%s border=%d => F1=%.4f",
        float(best_params["confidence_threshold"]),
        float(best_params["nms_radius_m"]),
        "None" if best_params["min_pred_height_m"] is None else f"{float(best_params['min_pred_height_m']):.2f}",
        "None" if best_params["min_chm_at_center_m"] is None else f"{float(best_params['min_chm_at_center_m']):.2f}",
        int(best_params["border_margin_px"]),
        float(best_scores["f1"]),
    )
    return best_params, best_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LOPO multimodal/unimodal center-based detector.")
    parser.add_argument("--config", nargs="+", required=True)
    try:
        import torch

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        default_device = "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    require_protocol_id(config.benchmark.protocol_id)
    seed_everything(config.training.seed)
    configure_logging(Path(config.paths.output_dir) / "logs", config.model.experiment_name)
    logger = get_logger("train_cv")

    processed_dir = config.resolve_path(config.paths.processed_dir)
    bundles = list_processed_plots(processed_dir, plot_ids=config.data.plot_ids or None)
    if config.data.split_mode == "basic":
        splits = build_basic_split(
            list(bundles.keys()),
            train_fraction=config.data.basic_train_fraction,
            val_fraction=config.data.basic_val_fraction,
            seed=config.data.basic_split_seed,
        )
    else:
        splits = build_lopo_splits(list(bundles.keys()), validation_plots_per_fold=config.training.validation_plots_per_fold)
    manifest = create_experiment_manifest(
        config,
        split={"split_mode": config.data.split_mode, "folds": [split.to_dict() for split in splits]},
        decode_params={
            "default_confidence_threshold": config.training.confidence_threshold,
            "default_nms_radius_m": config.evaluation.nms_radius_m,
            "tune_postprocess": bool(config.evaluation.tune_postprocess),
        },
        stage1_config={"stage1_kind": config.benchmark.stage1 or "global_dense_detector"},
        refiner=config.model.experiment_name,
        cwd=Path(__file__).resolve().parents[1],
    )
    config.model.experiment_name = manifest.experiment_name
    experiment_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "predictions" / config.model.experiment_name)
    if config.model.use_species_head:
        all_species_lookup = build_species_lookup(
            [bundle.reference for bundle in bundles.values()],
            mode=config.data.species_label_mode,
        )
        config.model.species_num_classes = len(all_species_lookup)
    else:
        config.model.species_num_classes = 0
    save_config(config, experiment_dir / "config.yaml")
    save_experiment_manifest(manifest, experiment_dir / "experiment_manifest.json")

    import torch
    from torch.utils.data import DataLoader

    label_snap_radius_m, label_mixed_weight = resolve_label_policy_parameters(
        config.data.label_policy,
        snap_radius_m=config.data.label_snap_radius_m,
        mixed_weight=config.data.label_mixed_weight,
    )

    for split in splits:
        logger.info("Training fold %d", split.fold_id)
        train_tiles = []
        val_tiles = []
        species_lookup = build_species_lookup(
            [bundles[plot_id].reference for plot_id in split.train_plots],
            mode=config.data.species_label_mode,
        )
        config.model.species_num_classes = len(species_lookup) if config.model.use_species_head else 0
        for plot_id in split.train_plots:
            table = _load_tile_table(processed_dir / plot_id)
            table["plot_id"] = plot_id
            table = _annotate_hard_negative_tiles(table, bundles[plot_id], config)
            train_tiles.append(
                _sample_training_tiles_with_hard_negatives(
                    table,
                    config.training.sample_negative_tile_ratio,
                    config.training.sample_hard_negative_tile_ratio,
                    config.training.seed + split.fold_id,
                )
            )
        for plot_id in split.val_plots:
            table = _load_tile_table(processed_dir / plot_id)
            table["plot_id"] = plot_id
            val_tiles.append(table)

        train_dataset = TreeTileDataset(
            plot_bundles={plot_id: bundles[plot_id] for plot_id in split.train_plots},
            tile_table=pd.concat(train_tiles, ignore_index=True),
            sigma_m=config.data.target_sigma_m,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            hard_negative_window_sizes_m=config.data.hard_negative_window_sizes_m,
            hard_negative_match_radius_m=config.data.hard_negative_match_radius_m,
            hard_negative_border_margin_px=config.data.hard_negative_border_margin_px,
            hard_negative_dilation_px=config.data.hard_negative_dilation_px,
            hard_negative_rgb_complexity_quantile=config.data.hard_negative_rgb_complexity_quantile,
            min_tree_height_m=config.data.min_tree_height_m,
            species_to_index=species_lookup,
            species_label_mode=config.data.species_label_mode,
            adaptive_sigma=config.data.adaptive_target_sigma,
            adaptive_sigma_use_nn_distance=config.data.adaptive_sigma_use_nn_distance,
            adaptive_sigma_nn_scale=config.data.adaptive_sigma_nn_scale,
            adaptive_sigma_a=config.data.adaptive_sigma_a,
            adaptive_sigma_b=config.data.adaptive_sigma_b,
            adaptive_sigma_min_m=config.data.adaptive_sigma_min_m,
            adaptive_sigma_max_m=config.data.adaptive_sigma_max_m,
            snap_centers_to_apex_radius_m=(
                label_snap_radius_m if config.data.snap_centers_to_apex_radius_m <= 0 else config.data.snap_centers_to_apex_radius_m
            ),
            apex_secondary_weight=(
                label_mixed_weight if config.data.apex_secondary_weight <= 0 else config.data.apex_secondary_weight
            ),
            crown_radius_a=config.data.crown_radius_a,
            crown_radius_b=config.data.crown_radius_b,
            crown_radius_min_m=config.data.crown_radius_min_m,
            crown_radius_max_m=config.data.crown_radius_max_m,
            context_margin_px=config.data.context_margin_px,
            supervision_margin_px=config.data.supervision_margin_px,
            pseudo_label_dir=config.resolve_path(config.data.pseudo_label_dir) if config.data.pseudo_label_dir else "",
            pseudo_label_match_radius_m=config.data.pseudo_label_match_radius_m,
            augmentation_config=config.augmentation,
            training=True,
        )
        val_dataset = TreeTileDataset(
            plot_bundles={plot_id: bundles[plot_id] for plot_id in split.val_plots},
            tile_table=pd.concat(val_tiles, ignore_index=True),
            sigma_m=config.data.target_sigma_m,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            hard_negative_window_sizes_m=config.data.hard_negative_window_sizes_m,
            hard_negative_match_radius_m=config.data.hard_negative_match_radius_m,
            hard_negative_border_margin_px=config.data.hard_negative_border_margin_px,
            hard_negative_dilation_px=config.data.hard_negative_dilation_px,
            hard_negative_rgb_complexity_quantile=config.data.hard_negative_rgb_complexity_quantile,
            min_tree_height_m=config.data.min_tree_height_m,
            species_to_index=species_lookup,
            species_label_mode=config.data.species_label_mode,
            adaptive_sigma=config.data.adaptive_target_sigma,
            adaptive_sigma_use_nn_distance=config.data.adaptive_sigma_use_nn_distance,
            adaptive_sigma_nn_scale=config.data.adaptive_sigma_nn_scale,
            adaptive_sigma_a=config.data.adaptive_sigma_a,
            adaptive_sigma_b=config.data.adaptive_sigma_b,
            adaptive_sigma_min_m=config.data.adaptive_sigma_min_m,
            adaptive_sigma_max_m=config.data.adaptive_sigma_max_m,
            snap_centers_to_apex_radius_m=(
                label_snap_radius_m if config.data.snap_centers_to_apex_radius_m <= 0 else config.data.snap_centers_to_apex_radius_m
            ),
            apex_secondary_weight=(
                label_mixed_weight if config.data.apex_secondary_weight <= 0 else config.data.apex_secondary_weight
            ),
            crown_radius_a=config.data.crown_radius_a,
            crown_radius_b=config.data.crown_radius_b,
            crown_radius_min_m=config.data.crown_radius_min_m,
            crown_radius_max_m=config.data.crown_radius_max_m,
            context_margin_px=config.data.context_margin_px,
            supervision_margin_px=config.data.supervision_margin_px,
            augmentation_config=None,
            training=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=max(1, config.training.num_workers // 2),
        )

        model = build_detector_from_config(config).to(args.device)
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")

        def validation_callback(current_model):
            fold_scores = []
            for plot_id in split.val_plots:
                prediction_maps = _predict_plot(current_model, bundles[plot_id], config, args.device, use_tta=False)
                predictions = _decode_predictions(
                    bundles[plot_id],
                    prediction_maps,
                    _default_postprocess_params(config),
                    config,
                )
                matches = match_tree_predictions(
                    predictions,
                    bundles[plot_id].reference,
                    match_radius_m=config.evaluation.match_radius_m,
                    height_tolerance_m=config.evaluation.height_tolerance_m,
                )
                fold_scores.append(summarize_matches(matches)["f1"])
            return {"f1": float(sum(fold_scores) / max(len(fold_scores), 1))}

        fit_result = fit_model(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=args.device,
            checkpoint_dir=fold_dir,
            validation_callback=validation_callback,
        )
        pd.DataFrame(fit_result["history"]).to_csv(fold_dir / "history.csv", index=False)
        best_state = torch.load(fold_dir / "best.pt", map_location=args.device)
        model.load_state_dict(best_state)

        val_prediction_maps = {
            plot_id: _predict_plot(model, bundles[plot_id], config, args.device, use_tta=config.evaluation.use_tta)
            for plot_id in split.val_plots
        }
        best_decode_params, validation_scores = _tune_postprocess_params(
            val_prediction_maps,
            bundles,
            split.val_plots,
            config,
            logger,
        )
        write_json(best_decode_params, fold_dir / "best_decode_params.json")
        write_json(validation_scores, fold_dir / "validation_scores.json")

        for plot_id in split.test_plots:
            prediction_maps = _predict_plot(model, bundles[plot_id], config, args.device, use_tta=config.evaluation.use_tta)
            predictions = _decode_predictions(bundles[plot_id], prediction_maps, best_decode_params, config)
            predictions.to_csv(fold_dir / f"{plot_id}.csv", index=False)
    logger.info("Training completed for %d folds.", len(splits))


if __name__ == "__main__":
    main()

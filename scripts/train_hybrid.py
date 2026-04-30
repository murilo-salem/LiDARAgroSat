from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.benchmarking import require_protocol_id
from lidaragrosat.data.auxiliary import build_species_lookup
from lidaragrosat.data.candidates import (
    CandidatePatchDataset,
    attach_candidate_features,
    generate_chm_candidates,
    generate_heatmap_candidates,
    label_candidate_table,
    merge_proposals,
)
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import create_experiment_manifest, save_experiment_manifest
from lidaragrosat.inference.sliding_window import (
    build_chm_features_if_needed,
    predict_plot_sliding_window,
    predict_plot_sliding_window_tta,
)
from lidaragrosat.models.detector import build_detector_from_config
from lidaragrosat.models.refiner import CandidateRefiner
from lidaragrosat.training.refiner import fit_candidate_refiner
from lidaragrosat.utils.io import ensure_dir, write_json
from lidaragrosat.utils.logging import configure_logging, get_logger
from lidaragrosat.utils.seed import seed_everything


def _predict_plot(model, bundle, config: ExperimentConfig, device: str):
    predictor = predict_plot_sliding_window_tta if config.evaluation.use_tta else predict_plot_sliding_window
    chm_features = build_chm_features_if_needed(
        model=model,
        chm=bundle.chm,
        pixel_size=bundle.transform.pixel_size,
        peakness_window_sizes_m=getattr(model, "chm_peakness_window_sizes_m", config.data.peakness_window_sizes_m),
        min_height_m=float(getattr(model, "chm_min_tree_height_m", config.data.min_tree_height_m)),
        precomputed=getattr(bundle, "chm_features", None),
    )
    return predictor(
        model,
        rgb=bundle.rgb,
        lidar=bundle.lidar,
        tile_size=config.data.tile_size,
        overlap=config.data.tile_overlap,
        chm=chm_features,
        device=device,
    )


def _load_proposal_model(config: ExperimentConfig, fold_id: int, device: str):
    import torch

    if config.hybrid.proposal_source_mode == "chm":
        return None
    proposal_root = config.resolve_path(config.hybrid.proposal_output_dir) / "predictions" / config.hybrid.proposal_method_name
    proposal_config = ExperimentConfig.from_files(proposal_root / "config.yaml")
    checkpoint_path = proposal_root / f"fold_{fold_id:02d}" / "best.pt"
    state = torch.load(checkpoint_path, map_location=device)
    species_weights = state.get("heads.species.weight")
    if species_weights is not None:
        proposal_config.model.species_num_classes = int(species_weights.shape[0])
    elif getattr(proposal_config.model, "use_species_head", False):
        proposal_config.model.species_num_classes = 0
    model = build_detector_from_config(proposal_config).to(device)
    model.load_state_dict(state)
    model.chm_peakness_window_sizes_m = tuple(proposal_config.data.peakness_window_sizes_m)
    model.chm_min_tree_height_m = float(proposal_config.data.min_tree_height_m)
    return model


def _build_plot_proposals(bundle, proposal_model, config: ExperimentConfig, device: str) -> pd.DataFrame:
    source_mode = config.hybrid.proposal_source_mode
    chm_candidates = pd.DataFrame()
    heatmap_candidates = pd.DataFrame()
    if source_mode in {"chm", "union"}:
        chm_candidates = generate_chm_candidates(
            bundle=bundle,
            window_sizes_m=config.hybrid.proposal_window_sizes_m,
            gaussian_sigmas=config.hybrid.proposal_gaussian_sigmas,
            min_height_m=config.hybrid.proposal_min_height_m,
            dedup_radius_m=config.hybrid.proposal_dedup_radius_m,
        )
    if source_mode in {"heatmap", "union"}:
        if proposal_model is None:
            raise ValueError("Heatmap proposals require a trained proposal model.")
        prediction_maps = _predict_plot(proposal_model, bundle, config, device)
        heatmap_candidates = generate_heatmap_candidates(
            bundle=bundle,
            prediction_maps=prediction_maps,
            confidence_threshold=config.hybrid.proposal_confidence_threshold,
            nms_radius_m=config.hybrid.proposal_nms_radius_m,
            dedup_radius_m=config.hybrid.proposal_dedup_radius_m,
        )
    proposals = merge_proposals(
        chm_candidates=chm_candidates,
        heatmap_candidates=heatmap_candidates,
        dedup_radius_m=config.hybrid.proposal_dedup_radius_m,
    )
    if proposals.empty:
        return proposals
    proposals["plot_id"] = bundle.plot_id
    proposals = attach_candidate_features(proposals, bundle)
    proposals = proposals.sort_values("confidence", ascending=False).reset_index(drop=True)
    if len(proposals) > config.hybrid.candidate_max_per_plot:
        proposals = proposals.iloc[: config.hybrid.candidate_max_per_plot].copy()
    return proposals.reset_index(drop=True)


def _score_refiner_candidates(model, dataset, batch_size: int, device: str) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                rgb=batch["rgb"].to(device),
                structural=batch["structural"].to(device),
                source_features=batch["source_features"].to(device),
                tabular_features=batch["tabular_features"].to(device),
                point_profile=batch["point_profile"].to(device),
            )
            probabilities = torch.sigmoid(outputs["cls"]).cpu().numpy()
            offsets = outputs["offset"].cpu().numpy()
            heights = outputs["height"].cpu().numpy()
            species = outputs.get("species")
            species_indices = species.argmax(dim=1).cpu().numpy() if species is not None else None

            for index in range(len(probabilities)):
                refined_x = float(batch["x"][index]) + float(offsets[index, 0])
                refined_y = float(batch["y"][index]) + float(offsets[index, 1])
                record = {
                    "plot_id": batch["plot_id"][index],
                    "x": refined_x,
                    "y": refined_y,
                    "confidence": float(probabilities[index]),
                    "height_pred": float(heights[index]),
                    "proposal_confidence": float(batch["proposal_confidence"][index]),
                    "source_key": batch["source_key"][index],
                }
                if species_indices is not None:
                    record["species_pred"] = int(species_indices[index])
                rows.append(record)
    return pd.DataFrame.from_records(rows)


def _greedy_point_nms(predictions: pd.DataFrame, radius_m: float, min_height_m: float) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    filtered = predictions[predictions["confidence"] > 0].copy()
    if "height_pred" in filtered.columns:
        filtered = filtered[filtered["height_pred"] >= min_height_m].copy()
    filtered = filtered.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[tuple[float, float]] = []
    for _, row in filtered.iterrows():
        point = (float(row["x"]), float(row["y"]))
        if any(((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5 <= radius_m for x, y in kept_xy):
            continue
        kept_rows.append(row.to_dict())
        kept_xy.append(point)
    return pd.DataFrame.from_records(kept_rows)


def _apply_source_thresholds(predictions: pd.DataFrame, params: dict) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    filtered = predictions.copy()
    if "source_thresholds" not in params:
        return filtered[filtered["confidence"] >= params["classification_threshold"]].copy()
    masks = []
    for source_key, threshold in params["source_thresholds"].items():
        source_mask = filtered["source_key"].astype(str) == str(source_key)
        masks.append(source_mask & (filtered["confidence"] >= float(threshold)))
    if not masks:
        return filtered.iloc[0:0].copy()
    keep_mask = masks[0].copy()
    for mask in masks[1:]:
        keep_mask |= mask
    return filtered[keep_mask].copy()


def _tune_refiner_postprocess(predictions: pd.DataFrame, bundle, config: ExperimentConfig) -> tuple[dict, dict]:
    best_params = {
        "classification_threshold": 0.5,
        "nms_radius_m": 1.0,
        "min_height_m": 0.0,
    }
    best_scores = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    source_keys = ["chm", "heatmap", "both"]
    threshold_candidates = config.hybrid.classification_threshold_grid
    if config.hybrid.source_specific_calibration and "source_key" in predictions.columns:
        threshold_grid = product(
            threshold_candidates,
            threshold_candidates,
            threshold_candidates,
            config.hybrid.final_nms_radius_grid_m,
            config.hybrid.final_min_height_grid_m,
        )
        for chm_threshold, heatmap_threshold, both_threshold, nms_radius_m, min_height_m in threshold_grid:
            params = {
                "source_thresholds": {
                    "chm": float(chm_threshold),
                    "heatmap": float(heatmap_threshold),
                    "both": float(both_threshold),
                },
                "nms_radius_m": float(nms_radius_m),
                "min_height_m": float(min_height_m),
            }
            selected = _apply_source_thresholds(predictions, params)
            selected = _greedy_point_nms(selected, radius_m=nms_radius_m, min_height_m=min_height_m)
            matches = match_tree_predictions(
                selected,
                bundle.reference,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            scores = summarize_matches(matches)
            if scores["f1"] > best_scores["f1"] or (scores["f1"] == best_scores["f1"] and scores["recall"] > best_scores["recall"]):
                best_params = params
                best_scores = {
                    "precision": float(scores["precision"]),
                    "recall": float(scores["recall"]),
                    "f1": float(scores["f1"]),
                }
    else:
        for cls_threshold, nms_radius_m, min_height_m in product(
            config.hybrid.classification_threshold_grid,
            config.hybrid.final_nms_radius_grid_m,
            config.hybrid.final_min_height_grid_m,
        ):
            params = {
                "classification_threshold": float(cls_threshold),
                "nms_radius_m": float(nms_radius_m),
                "min_height_m": float(min_height_m),
            }
            selected = _apply_source_thresholds(predictions, params)
            selected = _greedy_point_nms(selected, radius_m=nms_radius_m, min_height_m=min_height_m)
            matches = match_tree_predictions(
                selected,
                bundle.reference,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            scores = summarize_matches(matches)
            if scores["f1"] > best_scores["f1"] or (scores["f1"] == best_scores["f1"] and scores["recall"] > best_scores["recall"]):
                best_params = params
                best_scores = {
                    "precision": float(scores["precision"]),
                    "recall": float(scores["recall"]),
                    "f1": float(scores["f1"]),
                }
    return best_params, best_scores


def _score_refiner_postprocess(
    predictions: pd.DataFrame,
    bundle,
    classification_threshold: float,
    nms_radius_m: float,
    min_height_m: float,
    config: ExperimentConfig,
) -> dict[str, float]:
    params = {
        "classification_threshold": classification_threshold,
        "nms_radius_m": nms_radius_m,
        "min_height_m": min_height_m,
    }
    selected = _apply_source_thresholds(predictions, params)
    selected = _greedy_point_nms(selected, radius_m=nms_radius_m, min_height_m=min_height_m)
    matches = match_tree_predictions(
        selected,
        bundle.reference,
        match_radius_m=config.evaluation.match_radius_m,
        height_tolerance_m=config.evaluation.height_tolerance_m,
    )
    metrics = summarize_matches(matches)
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def _build_refiner_datasets(proposals_by_plot: dict[str, pd.DataFrame], bundles, plot_ids: list[str], config: ExperimentConfig, species_lookup: dict[str, int]):
    tables = []
    for plot_id in plot_ids:
        labeled = label_candidate_table(
            proposals_by_plot[plot_id],
            bundles[plot_id].reference,
            positive_match_radius_m=config.hybrid.positive_match_radius_m,
            species_to_index=species_lookup,
            species_label_mode=config.data.species_label_mode,
            fp_bank_confidence_quantile=config.hybrid.fp_bank_confidence_quantile,
            fp_bank_peakness_threshold=config.hybrid.fp_bank_peakness_threshold,
            fp_bank_min_height_m=config.hybrid.fp_bank_min_height_m,
        )
        tables.append(labeled)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _subsample_refiner_training_candidates(
    candidate_table: pd.DataFrame,
    negative_ratio: float,
    hard_negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    positives = candidate_table[candidate_table["is_positive"] == 1]
    hard_negatives = candidate_table[(candidate_table["is_positive"] == 0) & (candidate_table["is_hard_negative"] == 1)]
    random_negatives = candidate_table[(candidate_table["is_positive"] == 0) & (candidate_table["is_hard_negative"] == 0)]
    if len(positives):
        n_negative = min(len(random_negatives), max(1, int(len(positives) * negative_ratio)))
        n_hard_negative = min(len(hard_negatives), max(1, int(len(positives) * hard_negative_ratio)))
    else:
        n_negative = min(len(random_negatives), 1024)
        n_hard_negative = min(len(hard_negatives), 1024)
    sampled_negatives = random_negatives.sample(n=n_negative, random_state=seed) if n_negative and len(random_negatives) else random_negatives.iloc[0:0]
    sampled_hard_negatives = hard_negatives.sample(n=n_hard_negative, random_state=seed + 19) if n_hard_negative and len(hard_negatives) else hard_negatives.iloc[0:0]
    return (
        pd.concat([positives, sampled_negatives, sampled_hard_negatives], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train proposal + refinement hybrid tree detector.")
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
    logger = get_logger("train_hybrid")

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
            "classification_threshold_grid": list(config.hybrid.classification_threshold_grid),
            "final_nms_radius_grid_m": list(config.hybrid.final_nms_radius_grid_m),
            "final_min_height_grid_m": list(config.hybrid.final_min_height_grid_m),
        },
        stage1_config={
            "stage1_kind": config.benchmark.stage1 or config.hybrid.proposal_source_mode,
            "proposal_source_mode": config.hybrid.proposal_source_mode,
        },
        refiner=config.model.experiment_name,
        cwd=Path(__file__).resolve().parents[1],
    )
    config.model.experiment_name = manifest.experiment_name
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "predictions" / config.model.experiment_name)
    save_config(config, output_root / "config.yaml")
    save_experiment_manifest(manifest, output_root / "experiment_manifest.json")

    import torch
    from torch.utils.data import DataLoader

    for split in splits:
        logger.info("Training hybrid fold %d", split.fold_id)
        proposal_model = _load_proposal_model(config, split.fold_id, args.device)
        species_lookup = build_species_lookup(
            [bundles[plot_id].reference for plot_id in split.train_plots],
            mode=config.data.species_label_mode,
        )
        fold_dir = ensure_dir(output_root / f"fold_{split.fold_id:02d}")

        proposals_by_plot = {
            plot_id: _build_plot_proposals(bundles[plot_id], proposal_model, config, args.device)
            for plot_id in split.train_plots + split.val_plots + split.test_plots
        }
        for plot_id, proposals in proposals_by_plot.items():
            proposals.to_csv(fold_dir / f"{plot_id}_proposals.csv", index=False)

        train_candidates = _build_refiner_datasets(proposals_by_plot, bundles, split.train_plots, config, species_lookup)
        val_candidates = _build_refiner_datasets(proposals_by_plot, bundles, split.val_plots, config, species_lookup)
        train_candidates = _subsample_refiner_training_candidates(
            train_candidates,
            negative_ratio=config.hybrid.negative_subsample_ratio,
            hard_negative_ratio=config.hybrid.hard_negative_subsample_ratio,
            seed=config.training.seed + split.fold_id,
        )

        structural_in_channels = len(config.data.structural_channels) + 2
        model = CandidateRefiner(
            rgb_in_channels=config.model.rgb_in_channels,
            structural_in_channels=structural_in_channels,
            stage_channels=[32, 64, 128],
            normalization=config.model.normalization,
            species_num_classes=len(species_lookup) if config.data.species_label_mode != "none" else 0,
            tabular_dim=6,
            point_profile_dim=10 if config.hybrid.use_point_profile_branch else 0,
        ).to(args.device)

        train_dataset = CandidatePatchDataset(
            plot_bundles={plot_id: bundles[plot_id] for plot_id in split.train_plots},
            candidate_table=train_candidates,
            patch_size=config.hybrid.candidate_patch_size,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            raw_root=config.resolve_path(config.paths.raw_dir) if config.hybrid.use_point_profile_branch else None,
            point_radius_m=config.hybrid.point_radius_m,
            training=True,
        )
        val_dataset = CandidatePatchDataset(
            plot_bundles={plot_id: bundles[plot_id] for plot_id in split.val_plots},
            candidate_table=val_candidates,
            patch_size=config.hybrid.candidate_patch_size,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            raw_root=config.resolve_path(config.paths.raw_dir) if config.hybrid.use_point_profile_branch else None,
            point_radius_m=config.hybrid.point_radius_m,
            training=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=config.hybrid.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.hybrid.batch_size, shuffle=False, num_workers=0)

        def validation_callback(current_model):
            val_scored = _score_refiner_candidates(
                current_model,
                val_dataset,
                batch_size=config.hybrid.batch_size,
                device=args.device,
            )
            default_scores = _score_refiner_postprocess(
                val_scored,
                bundles[split.val_plots[0]],
                classification_threshold=0.5,
                nms_radius_m=1.0,
                min_height_m=0.0,
                config=config,
            )
            return default_scores

        fit_result = fit_candidate_refiner(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.hybrid.epochs,
            learning_rate=config.hybrid.learning_rate,
            weight_decay=config.hybrid.weight_decay,
            patience=config.hybrid.patience,
            device=args.device,
            checkpoint_dir=fold_dir,
            validation_callback=validation_callback,
            focal_alpha=config.hybrid.lambda_cls_focal_alpha,
            focal_gamma=config.hybrid.lambda_cls_focal_gamma,
            hard_negative_cls_weight=config.hybrid.hard_negative_cls_weight,
        )
        pd.DataFrame(fit_result["history"]).to_csv(fold_dir / "history.csv", index=False)
        best_state = torch.load(fold_dir / "best.pt", map_location=args.device)
        model.load_state_dict(best_state)

        val_scored = _score_refiner_candidates(model, val_dataset, batch_size=config.hybrid.batch_size, device=args.device)
        best_params, best_scores = _tune_refiner_postprocess(val_scored, bundles[split.val_plots[0]], config)
        write_json(best_params, fold_dir / "best_params.json")
        write_json(best_scores, fold_dir / "validation_scores.json")

        test_candidates = _build_refiner_datasets(proposals_by_plot, bundles, split.test_plots, config, species_lookup)
        test_dataset = CandidatePatchDataset(
            plot_bundles={plot_id: bundles[plot_id] for plot_id in split.test_plots},
            candidate_table=test_candidates,
            patch_size=config.hybrid.candidate_patch_size,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            raw_root=config.resolve_path(config.paths.raw_dir) if config.hybrid.use_point_profile_branch else None,
            point_radius_m=config.hybrid.point_radius_m,
            training=False,
        )
        test_scored = _score_refiner_candidates(model, test_dataset, batch_size=config.hybrid.batch_size, device=args.device)
        test_predictions = _apply_source_thresholds(test_scored, best_params)
        test_predictions = _greedy_point_nms(
            test_predictions,
            radius_m=best_params["nms_radius_m"],
            min_height_m=best_params["min_height_m"],
        )
        for plot_id in split.test_plots:
            plot_predictions = test_predictions[test_predictions["plot_id"] == plot_id].drop(columns=["plot_id"], errors="ignore")
            plot_predictions.to_csv(fold_dir / f"{plot_id}.csv", index=False)
    logger.info("Hybrid training completed for %d folds.", len(splits))


if __name__ == "__main__":
    main()

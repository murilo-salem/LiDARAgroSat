from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.auxiliary import build_species_lookup
from lidaragrosat.data.candidates import CandidatePatchDataset
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.models.refiner import CandidateRefiner
from lidaragrosat.utils.io import write_json


def _greedy_point_nms(predictions: pd.DataFrame, radius_m: float, min_height_m: float) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    filtered = predictions[predictions["confidence"] > 0].copy()
    if "height_pred" in filtered.columns:
        filtered = filtered[filtered["height_pred"] >= min_height_m].copy()
    filtered = filtered.sort_values("confidence", ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[np.ndarray] = []
    for _, row in filtered.iterrows():
        point = np.array([float(row["x"]), float(row["y"])], dtype=float)
        if any(np.linalg.norm(point - existing) <= radius_m for existing in kept_xy):
            continue
        kept_rows.append(row.to_dict())
        kept_xy.append(point)
    return pd.DataFrame.from_records(kept_rows)


def _apply_source_thresholds(predictions: pd.DataFrame, params: dict) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    if "source_thresholds" not in params:
        return predictions[predictions["confidence"] >= params["classification_threshold"]].copy()
    masks = []
    for source_key, threshold in params["source_thresholds"].items():
        masks.append((predictions["source_key"].astype(str) == str(source_key)) & (predictions["confidence"] >= float(threshold)))
    keep_mask = masks[0].copy()
    for mask in masks[1:]:
        keep_mask |= mask
    return predictions[keep_mask].copy()


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
                row = {
                    "plot_id": batch["plot_id"][index],
                    "x": float(batch["x"][index]) + float(offsets[index, 0]),
                    "y": float(batch["y"][index]) + float(offsets[index, 1]),
                    "confidence": float(probabilities[index]),
                    "height_pred": float(heights[index]),
                    "proposal_confidence": float(batch["proposal_confidence"][index]),
                    "source_key": batch["source_key"][index],
                }
                if species_indices is not None:
                    row["species_pred"] = int(species_indices[index])
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def _tune_postprocess(predictions: pd.DataFrame, bundle, config: ExperimentConfig) -> tuple[dict, dict]:
    best_params = {"classification_threshold": 0.5, "nms_radius_m": 1.0, "min_height_m": 0.0}
    best_scores = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    if config.hybrid.source_specific_calibration and "source_key" in predictions.columns:
        grid = product(
            config.hybrid.classification_threshold_grid,
            config.hybrid.classification_threshold_grid,
            config.hybrid.classification_threshold_grid,
            config.hybrid.final_nms_radius_grid_m,
            config.hybrid.final_min_height_grid_m,
        )
        for chm_threshold, heatmap_threshold, both_threshold, nms_radius_m, min_height_m in grid:
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
                best_scores = {k: float(scores[k]) for k in ["precision", "recall", "f1"]}
    else:
        grid = product(
            config.hybrid.classification_threshold_grid,
            config.hybrid.final_nms_radius_grid_m,
            config.hybrid.final_min_height_grid_m,
        )
        for threshold, nms_radius_m, min_height_m in grid:
            params = {
                "classification_threshold": float(threshold),
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
                best_scores = {k: float(scores[k]) for k in ["precision", "recall", "f1"]}
    return best_params, best_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a saved hybrid checkpoint on val/test proposals.")
    parser.add_argument("--config", nargs="+", required=True)
    try:
        import torch

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        default_device = "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    import torch

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    if config.data.split_mode == "basic":
        splits = build_basic_split(
            list(bundles.keys()),
            train_fraction=config.data.basic_train_fraction,
            val_fraction=config.data.basic_val_fraction,
            seed=config.data.basic_split_seed,
        )
    else:
        splits = build_lopo_splits(list(bundles.keys()), validation_plots_per_fold=config.training.validation_plots_per_fold)

    output_root = config.resolve_path(config.paths.output_dir) / "predictions" / config.model.experiment_name

    for split in splits:
        fold_dir = output_root / f"fold_{split.fold_id:02d}"
        species_lookup = build_species_lookup(
            [bundles[plot_id].reference for plot_id in split.train_plots],
            mode=config.data.species_label_mode,
        )
        species_num_classes = len(species_lookup) if config.data.species_label_mode != "none" else 0
        model = CandidateRefiner(
            rgb_in_channels=config.model.rgb_in_channels,
            structural_in_channels=len(config.data.structural_channels) + 2,
            stage_channels=[32, 64, 128],
            normalization=config.model.normalization,
            species_num_classes=species_num_classes,
            tabular_dim=6,
            point_profile_dim=10 if config.hybrid.use_point_profile_branch else 0,
        ).to(args.device)
        model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=args.device))

        val_plot = split.val_plots[0]
        val_table = pd.read_csv(fold_dir / f"{val_plot}_proposals.csv")
        val_table["plot_id"] = val_plot
        val_dataset = CandidatePatchDataset(
            plot_bundles={val_plot: bundles[val_plot]},
            candidate_table=val_table,
            patch_size=config.hybrid.candidate_patch_size,
            peakness_window_sizes_m=config.data.peakness_window_sizes_m,
            raw_root=config.resolve_path(config.paths.raw_dir) if config.hybrid.use_point_profile_branch else None,
            point_radius_m=config.hybrid.point_radius_m,
            training=False,
        )
        val_scored = _score_refiner_candidates(model, val_dataset, batch_size=config.hybrid.batch_size, device=args.device)
        best_params, best_scores = _tune_postprocess(val_scored, bundles[val_plot], config)
        write_json(best_params, fold_dir / "best_params.json")
        write_json(best_scores, fold_dir / "validation_scores.json")

        for test_plot in split.test_plots:
            test_table = pd.read_csv(fold_dir / f"{test_plot}_proposals.csv")
            test_table["plot_id"] = test_plot
            test_dataset = CandidatePatchDataset(
                plot_bundles={test_plot: bundles[test_plot]},
                candidate_table=test_table,
                patch_size=config.hybrid.candidate_patch_size,
                peakness_window_sizes_m=config.data.peakness_window_sizes_m,
                raw_root=config.resolve_path(config.paths.raw_dir) if config.hybrid.use_point_profile_branch else None,
                point_radius_m=config.hybrid.point_radius_m,
                training=False,
            )
            test_scored = _score_refiner_candidates(model, test_dataset, batch_size=config.hybrid.batch_size, device=args.device)
            selected = _apply_source_thresholds(test_scored, best_params)
            selected = _greedy_point_nms(selected, radius_m=best_params["nms_radius_m"], min_height_m=best_params["min_height_m"])
            selected.to_csv(fold_dir / f"{test_plot}.csv", index=False)


if __name__ == "__main__":
    main()

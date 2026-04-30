from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_A_ID, build_protocol_a_splits
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import create_experiment_manifest, save_experiment_manifest
from lidaragrosat.instance_baseline import (
    DistanceSeednessNet,
    InstanceTileDataset,
    decode_instance_maps,
    infer_full_plot,
    instance_loss,
)
from lidaragrosat.utils.io import ensure_dir, write_json


def _summarize_predictions(predictions_by_plot, bundles, plot_ids, match_radius_m, height_tolerance_m):
    rows = []
    for plot_id in plot_ids:
        matches = match_tree_predictions(
            predictions=predictions_by_plot[plot_id],
            reference=bundles[plot_id].reference,
            match_radius_m=match_radius_m,
            height_tolerance_m=height_tolerance_m,
        )
        row = summarize_matches(matches)
        row["plot_id"] = plot_id
        rows.append(row)
    table = pd.DataFrame.from_records(rows)
    return table, {}


def _fit_model(model, train_loader, val_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = {key: batch[key].to(device) for key in ["distance", "seedness"]}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            loss = instance_loss(outputs, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                target = {key: batch[key].to(device) for key in ["distance", "seedness"]}
                outputs = model(image)
                val_losses.append(float(instance_loss(outputs, target).item()))
        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = float(sum(val_losses) / max(len(val_losses), 1))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned distance-transform + watershed baseline on protocol A.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = build_protocol_a_splits(list(bundles.keys()))
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir))
    experiment_name = "ticketfinal.distance_transform_watershed.none.field_center.protocol_a_v2_3split.seed13"
    experiment_dir = ensure_dir(output_root / "predictions" / experiment_name)

    manifest = create_experiment_manifest(
        config,
        split={"protocol": PROTOCOL_A_ID, "folds": [split.to_dict() for split in splits]},
        decode_params={"seed_threshold": 0.35, "mask_threshold": 0.10, "min_distance_m": 1.0},
        stage1_config={"stage1_name": "distance_transform_watershed"},
        refiner="none",
        cwd=Path(__file__).resolve().parents[1],
    )
    save_experiment_manifest(manifest, experiment_dir / "experiment_manifest.json")

    all_metrics = []
    fold_rows = []
    input_channels = bundles[next(iter(bundles))].rgb.shape[0] + bundles[next(iter(bundles))].lidar.shape[0] + 1
    if bundles[next(iter(bundles))].chm_features is not None:
        input_channels += bundles[next(iter(bundles))].chm_features.shape[0]

    for split in splits:
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")
        train_dataset = InstanceTileDataset(bundles, split.train_plots, training=True, seed=13 + split.fold_id)
        val_dataset = InstanceTileDataset(bundles, split.val_plots, training=False, seed=13 + split.fold_id)
        model = DistanceSeednessNet(in_channels=input_channels).to(args.device)
        history, best_loss = _fit_model(
            model,
            train_loader=DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0),
            val_loader=DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0),
            epochs=args.epochs,
            device=args.device,
        )
        pd.DataFrame.from_records(history).to_csv(fold_dir / "history.csv", index=False)
        write_json({"best_val_loss": best_loss}, fold_dir / "fit_summary.json")

        predictions_by_plot = {}
        for plot_id in split.test_plots:
            distance_map, seedness_map = infer_full_plot(model, bundles[plot_id], device=args.device)
            predictions = decode_instance_maps(
                bundles[plot_id],
                distance_map=distance_map,
                seedness_map=seedness_map,
                seed_threshold=0.35,
                mask_threshold=0.10,
                min_distance_m=1.0,
                smooth_sigma_px=1.0,
            )
            predictions.to_csv(fold_dir / f"{plot_id}.csv", index=False)
            predictions_by_plot[plot_id] = predictions
        fold_table, _ = _summarize_predictions(
            predictions_by_plot,
            bundles,
            split.test_plots,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        fold_table["fold_id"] = split.fold_id
        fold_table["method"] = experiment_name
        fold_table["protocol_id"] = PROTOCOL_A_ID
        fold_table["stage1"] = "distance_transform_watershed"
        fold_table["stage2"] = "none"
        fold_table["labels"] = "field_center"
        all_metrics.append(fold_table)
        fold_rows.append(
            {
                "fold_id": split.fold_id,
                "precision": float(fold_table["precision"].mean()),
                "recall": float(fold_table["recall"].mean()),
                "f1": float(fold_table["f1"].mean()),
                "location_mae_m": float(fold_table["location_mae_m"].mean()),
                "height_mae_m": float(fold_table["height_mae_m"].mean()),
            }
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(experiment_dir / "metrics_by_plot.csv", index=False)
    pd.DataFrame.from_records(fold_rows).to_csv(experiment_dir / "fold_summary.csv", index=False)
    summary = build_method_summary(metrics.assign(method=experiment_name), bootstrap_iterations=config.evaluation.bootstrap_iterations)
    summary.to_csv(experiment_dir / "summary.csv", index=False)
    metrics_dir = ensure_dir(output_root / "metrics")
    summary.to_csv(metrics_dir / "distance_transform_watershed_summary.csv", index=False)


if __name__ == "__main__":
    main()

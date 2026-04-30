from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_A_ID, build_protocol_a_splits, require_protocol_id
from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import create_experiment_manifest, save_experiment_manifest
from lidaragrosat.stage1.api import Stage1Output
from lidaragrosat.stage1.generators import (
    AdaptiveLocalMaximaStage1Generator,
    AdaptiveWatershedStage1Generator,
    LocalMaximaStage1Generator,
    WatershedStage1Generator,
)
from lidaragrosat.stage2.dataset import LocalProposalDataset, build_local_proposal_table
from lidaragrosat.stage2.models import HybridLocalRefiner, PointLocalRefiner, RasterLocalRefiner
from lidaragrosat.stage2.postprocess import apply_stage2_postprocess, calibrate_stage2_postprocess
from lidaragrosat.stage2.training import fit_stage2_model
from lidaragrosat.utils.io import ensure_dir, write_json


def _build_stage1_generator(stage1_name: str):
    if stage1_name == "local_maxima":
        return LocalMaximaStage1Generator()
    if stage1_name == "adaptive_local_maxima":
        return AdaptiveLocalMaximaStage1Generator()
    if stage1_name == "watershed_fixo":
        return WatershedStage1Generator()
    if stage1_name == "adaptive_watershed":
        return AdaptiveWatershedStage1Generator()
    raise ValueError(f"Unsupported stage1_name: {stage1_name}")


def _run_stage1_cache(stage1_name: str, bundles) -> dict[str, Stage1Output]:
    generator = _build_stage1_generator(stage1_name)
    return {plot_id: generator.generate(bundle) for plot_id, bundle in bundles.items()}


def _summarize_predictions(predictions_by_plot: dict[str, pd.DataFrame], bundles, plot_ids: list[str], match_radius_m: float, height_tolerance_m: float) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    for plot_id in plot_ids:
        matches = match_tree_predictions(
            predictions_by_plot[plot_id],
            bundles[plot_id].reference,
            match_radius_m=match_radius_m,
            height_tolerance_m=height_tolerance_m,
        )
        row = summarize_matches(matches)
        row["plot_id"] = plot_id
        rows.append(row)
    table = pd.DataFrame.from_records(rows).sort_values("plot_id").reset_index(drop=True)
    summary = {
        "precision": float(table["precision"].mean()) if len(table) else 0.0,
        "recall": float(table["recall"].mean()) if len(table) else 0.0,
        "f1": float(table["f1"].mean()) if len(table) else 0.0,
        "location_mae_m": float(table["location_mae_m"].mean()) if len(table) else 0.0,
        "height_mae_m": float(table["height_mae_m"].mean()) if len(table) else 0.0,
        "tp": int(table["tp"].sum()) if len(table) else 0,
        "fp": int(table["fp"].sum()) if len(table) else 0,
        "fn": int(table["fn"].sum()) if len(table) else 0,
    }
    return table, summary


def _score_stage2_model(model, dataset: LocalProposalDataset, model_kind: str, device: str, batch_size: int) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raster = batch["raster"].to(device)
            point_cloud = batch["point_cloud"].to(device)
            geom = batch["geom_features"].to(device)
            if model_kind == "raster":
                outputs = model(raster, geom)
            elif model_kind == "point":
                outputs = model(point_cloud, geom)
            else:
                outputs = model(raster, point_cloud, geom)
            scores = torch.sigmoid(outputs["cls"]).cpu().numpy()
            offsets = outputs["offset"].cpu().numpy()
            delta_h = outputs["height"].cpu().numpy()
            split_scores = torch.sigmoid(outputs["split"]).cpu().numpy()
            xs = batch["x"].cpu().numpy()
            ys = batch["y"].cpu().numpy()
            base_h = batch["height_pred"].cpu().numpy()
            for idx in range(len(scores)):
                rows.append(
                    {
                        "plot_id": batch["plot_id"][idx],
                        "segment_id": int(batch["segment_id"][idx]),
                        "geom_x": float(xs[idx]),
                        "geom_y": float(ys[idx]),
                        "refined_x": float(xs[idx] + offsets[idx, 0]),
                        "refined_y": float(ys[idx] + offsets[idx, 1]),
                        "refined_height": float(base_h[idx] + delta_h[idx]),
                        "score": float(scores[idx]),
                        "split_score": float(split_scores[idx]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _build_datasets_for_split(
    bundles,
    stage1_outputs,
    split,
    label_policy: str,
    patch_size: int,
    n_points: int,
    use_rgb: bool,
    use_lidar: bool,
    use_chm: bool,
    include_geom_features: bool,
    include_raster: bool,
    include_point_cloud: bool,
    seed: int,
    max_train_samples: int,
):
    train_table = build_local_proposal_table(
        bundles,
        stage1_outputs,
        split.train_plots,
        mode="segment-centered",
        label_policy=label_policy,
        positive_radius_m=1.5,
    )
    val_table = build_local_proposal_table(
        bundles,
        stage1_outputs,
        split.val_plots,
        mode="segment-centered",
        label_policy=label_policy,
        positive_radius_m=1.5,
    )
    test_table = build_local_proposal_table(
        bundles,
        stage1_outputs,
        split.test_plots,
        mode="segment-centered",
        label_policy=label_policy,
        positive_radius_m=1.5,
    )
    if len(train_table):
        positives = train_table[train_table["objectness"] == 1]
        negatives = train_table[train_table["objectness"] == 0]
        sampled_negatives = negatives.sample(n=min(len(negatives), max(len(positives) * 2, 256)), random_state=seed) if len(negatives) else negatives
        train_table = pd.concat([positives, sampled_negatives], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if len(train_table) > int(max_train_samples):
            train_table = train_table.sample(n=int(max_train_samples), random_state=seed).reset_index(drop=True)
    common = {
        "plot_bundles": bundles,
        "stage1_outputs": stage1_outputs,
        "patch_size": patch_size,
        "n_points": n_points,
        "use_rgb": use_rgb,
        "use_lidar": use_lidar,
        "use_chm": use_chm,
        "include_geom_features": include_geom_features,
        "include_raster": include_raster,
        "include_point_cloud": include_point_cloud,
    }
    train_dataset = LocalProposalDataset(sample_table=train_table, training=True, seed=seed, **common)
    val_dataset = LocalProposalDataset(sample_table=val_table, training=False, seed=seed, **common)
    test_dataset = LocalProposalDataset(sample_table=test_table, training=False, seed=seed, **common)
    return train_dataset, val_dataset, test_dataset


def _build_model(model_kind: str, raster_channels: int, include_geom_features: bool):
    geom_dim = 7
    if model_kind == "raster":
        return RasterLocalRefiner(in_channels=raster_channels, geom_dim=geom_dim)
    if model_kind == "point":
        return PointLocalRefiner(point_dim=6, geom_dim=geom_dim)
    if model_kind == "hybrid":
        return HybridLocalRefiner(in_channels=raster_channels, point_dim=6, geom_dim=geom_dim)
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def _run_stage2_family(
    config: ExperimentConfig,
    bundles,
    splits,
    stage1_outputs,
    stage1_name: str,
    label_policy: str,
    model_kind: str,
    output_root: Path,
    device: str,
    use_rgb: bool = True,
    use_lidar: bool = True,
    use_chm: bool = True,
    include_geom_features: bool = True,
) -> tuple[pd.DataFrame, dict]:
    config_local = copy.deepcopy(config)
    config_local.benchmark.stage1 = stage1_name
    config_local.benchmark.stage2 = model_kind
    config_local.benchmark.label_policy = label_policy
    config_local.benchmark.family = config_local.benchmark.family or "ticketfinal"
    manifest = create_experiment_manifest(
        config_local,
        split={"protocol": PROTOCOL_A_ID, "folds": [split.to_dict() for split in splits]},
        decode_params={"threshold_grid": [0.2, 0.3, 0.4, 0.5, 0.6], "nms_grid_m": [0.8, 1.0, 1.2, 1.5]},
        stage1_config={"stage1_name": stage1_name},
        refiner=model_kind,
        cwd=Path(__file__).resolve().parents[1],
    )
    experiment_dir = ensure_dir(output_root / "predictions" / manifest.experiment_name)
    save_config(config_local, experiment_dir / "config.yaml")
    save_experiment_manifest(manifest, experiment_dir / "experiment_manifest.json")

    metric_rows: list[dict] = []
    fold_summaries: list[dict] = []
    raster_channels = (3 if use_rgb else 0) + (len(config.data.structural_channels) if use_lidar else 0) + (2 if use_chm else 0)
    raster_channels = max(raster_channels, 1)
    max_train_samples = 512 if model_kind == "point" else 256
    for split in splits:
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")
        train_dataset, val_dataset, test_dataset = _build_datasets_for_split(
            bundles=bundles,
            stage1_outputs=stage1_outputs,
            split=split,
            label_policy=label_policy,
            patch_size=64,
            n_points=128,
            use_rgb=use_rgb,
            use_lidar=use_lidar,
            use_chm=use_chm,
            include_geom_features=include_geom_features,
            include_raster=model_kind in {"raster", "hybrid"},
            include_point_cloud=model_kind in {"point", "hybrid"},
            seed=config.training.seed + split.fold_id,
            max_train_samples=max_train_samples,
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        model = _build_model(model_kind, raster_channels=raster_channels, include_geom_features=include_geom_features).to(device)

        def validation_callback(current_model):
            scored = _score_stage2_model(current_model, val_dataset, model_kind=model_kind, device=device, batch_size=128)
            predictions_by_plot = {}
            for plot_id in split.val_plots:
                current = scored[scored["plot_id"] == plot_id].copy()
                predictions_by_plot[plot_id] = apply_stage2_postprocess(
                    current,
                    objectness_threshold=0.4,
                    nms_radius_m=1.2,
                    center_mode="refined",
                )
            _, summary = _summarize_predictions(
                predictions_by_plot,
                bundles,
                split.val_plots,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            return summary

        fit_result = fit_stage2_model(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_kind=model_kind,
            device=device,
            checkpoint_dir=fold_dir,
            epochs=config.hybrid.epochs,
            learning_rate=config.hybrid.learning_rate,
            weight_decay=config.hybrid.weight_decay,
            patience=config.hybrid.patience,
            validation_callback=validation_callback,
        )
        pd.DataFrame.from_records(fit_result["history"]).to_csv(fold_dir / "history.csv", index=False)
        scored_val = _score_stage2_model(model, val_dataset, model_kind=model_kind, device=device, batch_size=128)
        scored_by_plot = {plot_id: scored_val[scored_val["plot_id"] == plot_id].copy() for plot_id in split.val_plots}
        best_params, best_scores = calibrate_stage2_postprocess(
            scored_by_plot=scored_by_plot,
            reference_by_plot={plot_id: bundles[plot_id].reference for plot_id in split.val_plots},
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
            threshold_grid=[0.2, 0.3, 0.4, 0.5, 0.6],
            nms_grid_m=[0.8, 1.0, 1.2, 1.5],
        )
        write_json({"best_params": best_params, "best_scores": best_scores}, fold_dir / "best_params.json")

        scored_test = _score_stage2_model(model, test_dataset, model_kind=model_kind, device=device, batch_size=128)
        predictions_by_plot = {}
        for plot_id in split.test_plots:
            current = scored_test[scored_test["plot_id"] == plot_id].copy()
            predictions = apply_stage2_postprocess(
                current,
                objectness_threshold=best_params["objectness_threshold"],
                nms_radius_m=best_params["nms_radius_m"],
                center_mode=best_params["center_mode"],
            )
            predictions.to_csv(fold_dir / f"{plot_id}.csv", index=False)
            predictions_by_plot[plot_id] = predictions
        fold_table, fold_summary = _summarize_predictions(
            predictions_by_plot,
            bundles,
            split.test_plots,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        fold_table["fold_id"] = split.fold_id
        metric_rows.append(fold_table.assign(method=manifest.experiment_name))
        fold_summaries.append({"fold_id": split.fold_id, **fold_summary})
    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    if len(metrics):
        metrics["protocol_id"] = manifest.protocol_id
        metrics["stage1"] = stage1_name
        metrics["stage2"] = model_kind
        metrics["labels"] = label_policy
    pd.DataFrame.from_records(fold_summaries).to_csv(experiment_dir / "fold_summary.csv", index=False)
    summary = build_method_summary(metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations) if len(metrics) else pd.DataFrame()
    if len(summary):
        summary.to_csv(experiment_dir / "summary.csv", index=False)
    return metrics, manifest.to_dict()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the official TicketFinal matrix on protocol A.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage1", nargs="*", default=["watershed_fixo", "adaptive_watershed"])
    parser.add_argument("--stage2", nargs="*", default=["raster", "point", "hybrid"])
    parser.add_argument("--labels", nargs="*", default=["field_center", "apex_snapped", "mixed_center_apex"])
    parser.add_argument("--skip-stage1-only", action="store_true")
    parser.add_argument("--only-stage1-only", action="store_true")
    parsed = parser.parse_args(args=argv)

    config = ExperimentConfig.from_files(*parsed.config)
    if not config.benchmark.protocol_id:
        config.benchmark.protocol_id = PROTOCOL_A_ID
    require_protocol_id(config.benchmark.protocol_id)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = build_protocol_a_splits(list(bundles.keys()))
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir))

    main_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    stage1_cache = {stage1_name: _run_stage1_cache(stage1_name, bundles) for stage1_name in parsed.stage1}

    # Stage-1 only runs
    if not parsed.skip_stage1_only:
        for stage1_name in parsed.stage1:
            stage1_outputs = stage1_cache[stage1_name]
            metrics_rows = []
            for split in splits:
                predictions_by_plot = {plot_id: stage1_outputs[plot_id].candidates for plot_id in split.test_plots}
                fold_table, _ = _summarize_predictions(
                    predictions_by_plot,
                    bundles,
                    split.test_plots,
                    match_radius_m=config.evaluation.match_radius_m,
                    height_tolerance_m=config.evaluation.height_tolerance_m,
                )
                fold_table["fold_id"] = split.fold_id
                fold_table["method"] = stage1_name
                fold_table["protocol_id"] = PROTOCOL_A_ID
                fold_table["stage1"] = stage1_name
                fold_table["stage2"] = "none"
                fold_table["labels"] = "field_center"
                metrics_rows.append(fold_table)
            main_rows.append(pd.concat(metrics_rows, ignore_index=True))

    # Main matrix
    if not parsed.only_stage1_only:
        for stage1_name in parsed.stage1:
            for model_kind in parsed.stage2:
                for label_policy in parsed.labels:
                    metrics, manifest = _run_stage2_family(
                        config=config,
                        bundles=bundles,
                        splits=splits,
                        stage1_outputs=stage1_cache[stage1_name],
                        stage1_name=stage1_name,
                        label_policy=label_policy,
                        model_kind=model_kind,
                        output_root=output_root,
                        device=parsed.device,
                    )
                    main_rows.append(metrics)
                    manifest_rows.append(manifest)

    main_metrics = pd.concat(main_rows, ignore_index=True)
    metrics_dir = ensure_dir(output_root / "metrics")
    main_metrics.to_csv(metrics_dir / "main_matrix_metrics_by_plot.csv", index=False)
    main_summary = build_method_summary(main_metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    main_summary.to_csv(metrics_dir / "main_matrix_results.csv", index=False)
    if manifest_rows:
        pd.DataFrame.from_records(manifest_rows).to_csv(metrics_dir / "main_matrix_manifests.csv", index=False)

    label_ablation = (
        main_metrics[(main_metrics["stage1"] == "watershed_fixo") & (main_metrics["stage2"].isin(["raster", "point"]))]
        .groupby(["stage2", "labels"])[["precision", "recall", "f1", "location_mae_m", "height_mae_m"]]
        .mean()
        .reset_index()
    )
    label_ablation.to_csv(metrics_dir / "label_policy_ablation.csv", index=False)

    stage1_vs_stage2 = (
        main_metrics.groupby(["stage1", "stage2"])[["precision", "recall", "f1", "location_mae_m", "height_mae_m", "tp", "fp", "fn"]]
        .mean()
        .reset_index()
        .sort_values(["stage1", "f1"], ascending=[True, False])
    )
    stage1_vs_stage2.to_csv(metrics_dir / "stage1_vs_stage2.csv", index=False)


if __name__ == "__main__":
    main()

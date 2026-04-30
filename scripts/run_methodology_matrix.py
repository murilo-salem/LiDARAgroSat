from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import os
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.methodology import (
    build_stage2_delta_tables,
    count_site_wins,
    summarize_site_metrics,
)
from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import require_protocol_id
from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
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
from lidaragrosat.stage2.dataset import LocalProposalDataset, build_local_dataset_caches, build_local_proposal_table
from lidaragrosat.stage2.models import HybridLocalRefiner, PointLocalRefiner, RasterLocalRefiner
from lidaragrosat.stage2.postprocess import apply_stage2_postprocess, calibrate_stage2_postprocess
from lidaragrosat.stage2.training import fit_stage2_model
from lidaragrosat.utils.io import ensure_dir, write_json


DEFAULT_STAGE1 = ("local_maxima", "fixed_watershed", "adaptive_watershed")
DEFAULT_STAGE2 = ("raster", "point", "hybrid")
DEFAULT_LABELS = ("field_center",)


def _build_splits(config: ExperimentConfig, bundles) -> list:
    if config.data.split_mode == "basic":
        return build_basic_split(
            list(bundles.keys()),
            train_fraction=config.data.basic_train_fraction,
            val_fraction=config.data.basic_val_fraction,
            seed=config.data.basic_split_seed,
        )
    return build_lopo_splits(
        list(bundles.keys()),
        validation_plots_per_fold=config.training.validation_plots_per_fold,
    )


def _build_stage1_generator(stage1_name: str):
    if stage1_name == "local_maxima":
        return LocalMaximaStage1Generator()
    if stage1_name == "adaptive_local_maxima":
        return AdaptiveLocalMaximaStage1Generator()
    if stage1_name == "fixed_watershed":
        return WatershedStage1Generator()
    if stage1_name == "adaptive_watershed":
        return AdaptiveWatershedStage1Generator()
    raise ValueError(f"Unsupported stage1_name: {stage1_name}")


def _run_stage1_cache(stage1_name: str, bundles) -> dict[str, Stage1Output]:
    generator = _build_stage1_generator(stage1_name)
    return {plot_id: generator.generate(bundle) for plot_id, bundle in bundles.items()}


def _summarize_predictions(
    predictions_by_plot: dict[str, pd.DataFrame],
    bundles,
    plot_ids: list[str],
    match_radius_m: float,
    height_tolerance_m: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
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
    worker_count = min(8, max((os.cpu_count() or 1) // 2, 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=device.startswith("cuda"),
    )
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
            proposal_ids = batch["proposal_id"].cpu().numpy()
            xs = batch["x"].cpu().numpy()
            ys = batch["y"].cpu().numpy()
            base_h = batch["height_pred"].cpu().numpy()
            segment_ids = batch["segment_id"].cpu().numpy()
            for idx in range(len(scores)):
                rows.append(
                    {
                        "plot_id": batch["plot_id"][idx],
                        "proposal_id": int(proposal_ids[idx]),
                        "segment_id": int(segment_ids[idx]),
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
    active_plot_ids = sorted(set(split.train_plots + split.val_plots + split.test_plots))
    peakness_cache, raw_points, raw_point_index = build_local_dataset_caches(
        bundles,
        include_point_cloud=include_point_cloud,
        plot_ids=active_plot_ids,
    )
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
        sampled_negatives = (
            negatives.sample(n=min(len(negatives), max(len(positives) * 2, 256)), random_state=seed)
            if len(negatives)
            else negatives
        )
        train_table = (
            pd.concat([positives, sampled_negatives], ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        if len(train_table) > int(max_train_samples):
            train_table = train_table.sample(n=int(max_train_samples), random_state=seed).reset_index(drop=True)

    def with_ids(table: pd.DataFrame) -> pd.DataFrame:
        enriched = table.copy()
        if "proposal_id" not in enriched.columns:
            enriched["proposal_id"] = pd.Series(range(len(enriched)), dtype=int)
        return enriched

    train_table = with_ids(train_table)
    val_table = with_ids(val_table)
    test_table = with_ids(test_table)

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
        "peakness_cache": peakness_cache,
        "raw_points": raw_points,
        "raw_point_index": raw_point_index,
    }
    train_dataset = LocalProposalDataset(sample_table=train_table, training=True, seed=seed, **common)
    val_dataset = LocalProposalDataset(sample_table=val_table, training=False, seed=seed, **common)
    test_dataset = LocalProposalDataset(sample_table=test_table, training=False, seed=seed, **common)
    return train_dataset, val_dataset, test_dataset, train_table, val_table, test_table


def _build_model(model_kind: str, raster_channels: int):
    geom_dim = 7
    if model_kind == "raster":
        return RasterLocalRefiner(in_channels=raster_channels, geom_dim=geom_dim)
    if model_kind == "point":
        return PointLocalRefiner(point_dim=6, geom_dim=geom_dim)
    if model_kind == "hybrid":
        return HybridLocalRefiner(in_channels=raster_channels, point_dim=6, geom_dim=geom_dim)
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def _run_stage1_only(
    config: ExperimentConfig,
    bundles,
    splits,
    stage1_outputs,
    stage1_name: str,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    config_local = copy.deepcopy(config)
    config_local.benchmark.family = config_local.benchmark.family or "methodology_stage1"
    config_local.benchmark.stage1 = stage1_name
    config_local.benchmark.stage2 = "none"
    config_local.benchmark.label_policy = "field_center"
    generator = _build_stage1_generator(stage1_name)
    manifest = create_experiment_manifest(
        config_local,
        split={"split_mode": config.data.split_mode, "folds": [split.to_dict() for split in splits]},
        stage1_config=asdict(generator),
        refiner="none",
        cwd=Path(__file__).resolve().parents[1],
    )
    experiment_dir = ensure_dir(output_root / "predictions" / manifest.experiment_name)
    save_config(config_local, experiment_dir / "config.yaml")
    save_experiment_manifest(manifest, experiment_dir / "experiment_manifest.json")

    metrics_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict] = []
    segment_rows: list[pd.DataFrame] = []
    fold_summaries: list[dict] = []
    for split in splits:
        print(f"[stage1-only] stage1={stage1_name} fold={split.fold_id} writing predictions", flush=True)
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")
        predictions_by_plot = {}
        for plot_id in split.test_plots:
            output = stage1_outputs[plot_id]
            output.candidates.to_csv(fold_dir / f"{plot_id}.csv", index=False)
            if output.segment_labels is not None:
                save_path = fold_dir / f"{plot_id}_segments.npy"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with save_path.open("wb") as handle:
                    import numpy as np

                    np.save(handle, output.segment_labels.astype("int32"))
            predictions_by_plot[plot_id] = output.candidates
            diagnostics_rows.append({"plot_id": plot_id, "stage1": stage1_name, **output.diagnostics.to_dict(), **output.metadata})
            if not output.segment_table.empty:
                segment_rows.append(output.segment_table.assign(plot_id=plot_id, stage1=stage1_name))
        fold_table, fold_summary = _summarize_predictions(
            predictions_by_plot,
            bundles,
            split.test_plots,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        fold_table["fold_id"] = split.fold_id
        fold_table["method"] = stage1_name
        fold_table["protocol_id"] = manifest.protocol_id
        fold_table["family"] = manifest.family
        fold_table["stage1"] = stage1_name
        fold_table["stage2"] = "none"
        fold_table["labels"] = "field_center"
        fold_table["seed"] = int(config.training.seed)
        metrics_rows.append(fold_table)
        fold_summaries.append({"fold_id": split.fold_id, **fold_summary})

    pd.DataFrame.from_records(fold_summaries).to_csv(experiment_dir / "fold_summary.csv", index=False)
    metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    diagnostics = pd.DataFrame.from_records(diagnostics_rows).sort_values(["stage1", "plot_id"]).reset_index(drop=True)
    segments = pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame()
    summary = build_method_summary(metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations) if len(metrics) else pd.DataFrame()
    if len(summary):
        summary.to_csv(experiment_dir / "summary.csv", index=False)
    return metrics, diagnostics, segments, manifest.to_dict()


def _run_stage2_family(
    config: ExperimentConfig,
    bundles,
    splits,
    stage1_outputs,
    stage1_name: str,
    label_policy: str,
    model_kind: str,
    stage2_name: str,
    output_root: Path,
    device: str,
    use_rgb: bool = True,
    use_lidar: bool = True,
    use_chm: bool = True,
    include_geom_features: bool = True,
    patch_size: int = 64,
    n_points: int = 128,
) -> tuple[pd.DataFrame, dict]:
    config_local = copy.deepcopy(config)
    config_local.benchmark.family = config_local.benchmark.family or "methodology_stage2"
    config_local.benchmark.stage1 = stage1_name
    config_local.benchmark.stage2 = stage2_name
    config_local.benchmark.label_policy = label_policy
    manifest = create_experiment_manifest(
        config_local,
        split={"split_mode": config.data.split_mode, "folds": [split.to_dict() for split in splits]},
        decode_params={"threshold_grid": [0.2, 0.3, 0.4, 0.5, 0.6], "nms_grid_m": [0.8, 1.0, 1.2, 1.5]},
        stage1_config={"stage1_name": stage1_name},
        refiner=stage2_name,
        cwd=Path(__file__).resolve().parents[1],
    )
    experiment_dir = ensure_dir(output_root / "predictions" / manifest.experiment_name)
    save_config(config_local, experiment_dir / "config.yaml")
    save_experiment_manifest(manifest, experiment_dir / "experiment_manifest.json")

    metric_rows: list[pd.DataFrame] = []
    fold_summaries: list[dict] = []
    raster_channels = (3 if use_rgb else 0) + (len(config.data.structural_channels) if use_lidar else 0) + (2 if use_chm else 0)
    raster_channels = max(raster_channels, 1)
    max_train_samples = 512 if model_kind == "point" else 256
    for split in splits:
        print(
            f"[stage2] stage1={stage1_name} stage2={stage2_name} labels={label_policy} fold={split.fold_id} building datasets",
            flush=True,
        )
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")
        (
            train_dataset,
            val_dataset,
            test_dataset,
            train_table,
            val_table,
            test_table,
        ) = _build_datasets_for_split(
            bundles=bundles,
            stage1_outputs=stage1_outputs,
            split=split,
            label_policy=label_policy,
            patch_size=patch_size,
            n_points=n_points,
            use_rgb=use_rgb,
            use_lidar=use_lidar,
            use_chm=use_chm,
            include_geom_features=include_geom_features,
            include_raster=model_kind in {"raster", "hybrid"},
            include_point_cloud=model_kind in {"point", "hybrid"},
            seed=config.training.seed + split.fold_id,
            max_train_samples=max_train_samples,
        )
        train_table.to_csv(fold_dir / "train_proposals.csv", index=False)
        val_table.to_csv(fold_dir / "val_proposals.csv", index=False)
        test_table.to_csv(fold_dir / "test_proposals.csv", index=False)

        print(
            f"[stage2] stage1={stage1_name} stage2={stage2_name} fold={split.fold_id} "
            f"train={len(train_table)} val={len(val_table)} test={len(test_table)} training",
            flush=True,
        )
        worker_count = min(8, max((os.cpu_count() or 1) // 2, 1))
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=device.startswith("cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=device.startswith("cuda"),
        )
        model = _build_model(model_kind, raster_channels=raster_channels).to(device)

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

        print(f"[stage2] stage1={stage1_name} stage2={stage2_name} fold={split.fold_id} scoring val", flush=True)
        scored_val = _score_stage2_model(model, val_dataset, model_kind=model_kind, device=device, batch_size=128)
        scored_val.to_csv(fold_dir / "scored_val.csv", index=False)
        scored_by_plot = {plot_id: scored_val[scored_val["plot_id"] == plot_id].copy() for plot_id in split.val_plots}
        print(f"[stage2] stage1={stage1_name} stage2={stage2_name} fold={split.fold_id} calibrating", flush=True)
        best_params, best_scores = calibrate_stage2_postprocess(
            scored_by_plot=scored_by_plot,
            reference_by_plot={plot_id: bundles[plot_id].reference for plot_id in split.val_plots},
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
            threshold_grid=[0.2, 0.3, 0.4, 0.5, 0.6],
            nms_grid_m=[0.8, 1.0, 1.2, 1.5],
        )
        write_json({"best_params": best_params, "best_scores": best_scores}, fold_dir / "best_params.json")

        print(f"[stage2] stage1={stage1_name} stage2={stage2_name} fold={split.fold_id} scoring test", flush=True)
        scored_test = _score_stage2_model(model, test_dataset, model_kind=model_kind, device=device, batch_size=128)
        scored_test.to_csv(fold_dir / "scored_test.csv", index=False)
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
        fold_table["method"] = f"{stage1_name}+{stage2_name}"
        fold_table["protocol_id"] = manifest.protocol_id
        fold_table["family"] = manifest.family
        fold_table["stage1"] = stage1_name
        fold_table["stage2"] = stage2_name
        fold_table["labels"] = label_policy
        fold_table["seed"] = int(config.training.seed)
        metric_rows.append(fold_table)
        fold_summaries.append({"fold_id": split.fold_id, **fold_summary})
        print(f"[stage2] stage1={stage1_name} stage2={stage2_name} fold={split.fold_id} complete", flush=True)

    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    pd.DataFrame.from_records(fold_summaries).to_csv(experiment_dir / "fold_summary.csv", index=False)
    summary = build_method_summary(metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations) if len(metrics) else pd.DataFrame()
    if len(summary):
        summary.to_csv(experiment_dir / "summary.csv", index=False)
    return metrics, manifest.to_dict()


def _save_metric_views(metrics: pd.DataFrame, output_dir: Path, stem: str) -> None:
    metrics_path = output_dir / f"{stem}_metrics_by_plot.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        metrics = pd.concat([existing, metrics], ignore_index=True)
        dedupe_keys = [column for column in ["method", "plot_id", "stage1", "stage2", "labels", "fold_id"] if column in metrics.columns]
        metrics = metrics.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
    metrics = metrics.sort_values(["stage1", "stage2", "plot_id", "method"]).reset_index(drop=True)
    metrics.to_csv(metrics_path, index=False)
    summary = build_method_summary(metrics, bootstrap_iterations=200)
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    site_summary = summarize_site_metrics(metrics)
    site_summary.to_csv(output_dir / f"{stem}_site_summary.csv", index=False)
    site_wins = count_site_wins(metrics)
    site_wins.to_csv(output_dir / f"{stem}_site_wins.csv", index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the methodology-oriented Stage-1/Stage-2 matrix on the configured dataset.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage1", nargs="*", default=list(DEFAULT_STAGE1))
    parser.add_argument("--stage2", nargs="*", default=list(DEFAULT_STAGE2))
    parser.add_argument("--labels", nargs="*", default=list(DEFAULT_LABELS))
    parser.add_argument("--skip-stage1-only", action="store_true")
    parser.add_argument("--only-stage1-only", action="store_true")
    parser.add_argument("--run-modal-ablation", action="store_true")
    parser.add_argument("--modal-stage1", default="best")
    parser.add_argument("--stage2-epochs", type=int, default=0)
    parser.add_argument("--stage2-patience", type=int, default=0)
    parsed = parser.parse_args(args=argv)

    config = ExperimentConfig.from_files(*parsed.config)
    require_protocol_id(config.benchmark.protocol_id)
    if parsed.stage2_epochs > 0:
        config.hybrid.epochs = int(parsed.stage2_epochs)
    if parsed.stage2_patience > 0:
        config.hybrid.patience = int(parsed.stage2_patience)

    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = _build_splits(config, bundles)
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "methodology")
    metrics_dir = ensure_dir(output_root / "metrics")

    stage1_cache = {}
    for stage1_name in parsed.stage1:
        print(f"[stage1-cache] stage1={stage1_name} generating", flush=True)
        stage1_cache[stage1_name] = _run_stage1_cache(stage1_name, bundles)
        print(f"[stage1-cache] stage1={stage1_name} ready", flush=True)

    stage1_metrics_rows: list[pd.DataFrame] = []
    stage1_diagnostics_rows: list[pd.DataFrame] = []
    stage1_segment_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    if not parsed.skip_stage1_only:
        for stage1_name in parsed.stage1:
            metrics, diagnostics, segments, manifest = _run_stage1_only(
                config=config,
                bundles=bundles,
                splits=splits,
                stage1_outputs=stage1_cache[stage1_name],
                stage1_name=stage1_name,
                output_root=output_root,
            )
            stage1_metrics_rows.append(metrics)
            if len(diagnostics):
                stage1_diagnostics_rows.append(diagnostics)
            if len(segments):
                stage1_segment_rows.append(segments)
            manifest_rows.append(manifest)

    if stage1_metrics_rows:
        stage1_metrics = pd.concat(stage1_metrics_rows, ignore_index=True)
        _save_metric_views(stage1_metrics, metrics_dir, "stage1_only")
        merged_stage1_metrics = pd.read_csv(metrics_dir / "stage1_only_metrics_by_plot.csv")
        stage1_overview = (
            merged_stage1_metrics.groupby("stage1")[["precision", "recall", "f1", "tp", "fp", "fn"]]
            .mean()
            .reset_index()
            .sort_values("f1", ascending=False)
            .reset_index(drop=True)
        )
        stage1_overview.to_csv(metrics_dir / "stage1_only_overview.csv", index=False)
    else:
        stage1_metrics = pd.DataFrame()
    if stage1_diagnostics_rows:
        diagnostics = pd.concat(stage1_diagnostics_rows, ignore_index=True)
        diagnostics_path = metrics_dir / "stage1_diagnostics.csv"
        if diagnostics_path.exists():
            diagnostics = pd.concat([pd.read_csv(diagnostics_path), diagnostics], ignore_index=True)
            diagnostics = diagnostics.drop_duplicates(subset=["stage1", "plot_id"], keep="last").reset_index(drop=True)
        diagnostics.to_csv(diagnostics_path, index=False)
    if stage1_segment_rows:
        segments = pd.concat(stage1_segment_rows, ignore_index=True)
        segments_path = metrics_dir / "stage1_segments.csv"
        if segments_path.exists():
            segments = pd.concat([pd.read_csv(segments_path), segments], ignore_index=True)
            dedupe_keys = [column for column in ["stage1", "plot_id", "segment_id"] if column in segments.columns]
            segments = segments.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
        segments.to_csv(segments_path, index=False)

    matrix_rows: list[pd.DataFrame] = []
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
                        stage2_name=model_kind,
                        output_root=output_root,
                        device=parsed.device,
                    )
                    matrix_rows.append(metrics)
                    manifest_rows.append(manifest)

    if matrix_rows:
        matrix_metrics = pd.concat(matrix_rows, ignore_index=True)
        _save_metric_views(matrix_metrics, metrics_dir, "stage1_stage2")
        merged_stage1_metrics = pd.read_csv(metrics_dir / "stage1_only_metrics_by_plot.csv") if (metrics_dir / "stage1_only_metrics_by_plot.csv").exists() else stage1_metrics
        merged_matrix_metrics = pd.read_csv(metrics_dir / "stage1_stage2_metrics_by_plot.csv")
        delta_by_plot, delta_summary = build_stage2_delta_tables(pd.concat([merged_stage1_metrics, merged_matrix_metrics], ignore_index=True))
        delta_by_plot.to_csv(metrics_dir / "stage2_delta_by_plot.csv", index=False)
        delta_summary.to_csv(metrics_dir / "stage2_delta_summary.csv", index=False)
        overview = (
            merged_matrix_metrics.groupby(["stage1", "stage2"])[["precision", "recall", "f1", "tp", "fp", "fn"]]
            .mean()
            .reset_index()
            .sort_values(["stage1", "f1"], ascending=[True, False])
            .reset_index(drop=True)
        )
        overview.to_csv(metrics_dir / "stage1_vs_stage2_overview.csv", index=False)
    else:
        matrix_metrics = pd.DataFrame()

    if parsed.run_modal_ablation and not stage1_metrics.empty:
        if parsed.modal_stage1 == "best":
            best_stage1 = (
                stage1_metrics.groupby("stage1")["f1"]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )
        else:
            best_stage1 = parsed.modal_stage1
        modal_specs = [
            ("raster_rgb_only", {"use_rgb": True, "use_lidar": False, "use_chm": False}),
            ("raster_lidar_chm_only", {"use_rgb": False, "use_lidar": True, "use_chm": True}),
            ("raster_rgb_chm", {"use_rgb": True, "use_lidar": False, "use_chm": True}),
            ("raster_rgb_lidar_chm", {"use_rgb": True, "use_lidar": True, "use_chm": True}),
        ]
        modal_rows: list[pd.DataFrame] = []
        for stage2_name, kwargs in modal_specs:
            metrics, manifest = _run_stage2_family(
                config=config,
                bundles=bundles,
                splits=splits,
                stage1_outputs=stage1_cache[best_stage1],
                stage1_name=best_stage1,
                label_policy="field_center",
                model_kind="raster",
                stage2_name=stage2_name,
                output_root=output_root,
                device=parsed.device,
                **kwargs,
            )
            modal_rows.append(metrics)
            manifest_rows.append(manifest)
        if modal_rows:
            modal_metrics = pd.concat(modal_rows, ignore_index=True)
            _save_metric_views(modal_metrics, metrics_dir, "modality_ablation")
            merged_stage1_metrics = pd.read_csv(metrics_dir / "stage1_only_metrics_by_plot.csv") if (metrics_dir / "stage1_only_metrics_by_plot.csv").exists() else stage1_metrics
            merged_modal_metrics = pd.read_csv(metrics_dir / "modality_ablation_metrics_by_plot.csv")
            modal_delta_by_plot, modal_delta_summary = build_stage2_delta_tables(pd.concat([merged_stage1_metrics, merged_modal_metrics], ignore_index=True))
            modal_delta_by_plot.to_csv(metrics_dir / "modality_ablation_delta_by_plot.csv", index=False)
            modal_delta_summary.to_csv(metrics_dir / "modality_ablation_delta_summary.csv", index=False)

    if manifest_rows:
        manifests = pd.DataFrame.from_records(manifest_rows)
        manifests_path = metrics_dir / "methodology_manifests.csv"
        if manifests_path.exists():
            manifests = pd.concat([pd.read_csv(manifests_path), manifests], ignore_index=True)
        manifests.drop_duplicates(subset=["experiment_name"]).to_csv(manifests_path, index=False)


if __name__ == "__main__":
    main()

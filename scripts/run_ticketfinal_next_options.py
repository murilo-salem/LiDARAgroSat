from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_A_ID, build_protocol_a_splits
from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.stage1.generators import WatershedStage1Generator
from lidaragrosat.stage2.dataset import LocalProposalDataset, build_local_proposal_table
from lidaragrosat.stage2.models import HybridLocalRefiner, PointLocalRefiner, StrongRGBHybridLocalRefiner
from lidaragrosat.stage2.training import fit_stage2_model
from lidaragrosat.utils.io import ensure_dir, write_json


CURRENT_POINT_EXPERIMENT = "ticketfinal.watershed_fixo.point.field_center.protocol_a_v2_3split.seed13"
CURRENT_HYBRID_RGB_CHM_EXPERIMENT = "ticketfinal_rgb_hybrid_rgb_chm.watershed_fixo.hybrid.field_center.protocol_a_v2_3split.seed13"


def _sample_train_table(sample_table: pd.DataFrame, seed: int, max_train_samples: int = 512) -> pd.DataFrame:
    if sample_table.empty:
        return sample_table.copy()
    positives = sample_table[sample_table["objectness"] == 1]
    negatives = sample_table[sample_table["objectness"] == 0]
    if len(negatives):
        sampled_negatives = negatives.sample(n=min(len(negatives), max(len(positives) * 2, 256)), random_state=seed)
    else:
        sampled_negatives = negatives
    balanced = pd.concat([positives, sampled_negatives], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(balanced) > int(max_train_samples):
        balanced = balanced.sample(n=int(max_train_samples), random_state=seed).reset_index(drop=True)
    return balanced


def _build_datasets(
    bundles,
    stage1_outputs,
    split,
    patch_size: int,
    n_points: int,
    use_rgb: bool,
    use_lidar: bool,
    use_chm: bool,
    include_raster: bool,
    include_point_cloud: bool,
    geom_feature_set: str,
    seed: int,
    max_train_samples: int = 512,
):
    train_table = build_local_proposal_table(
        bundles,
        stage1_outputs,
        split.train_plots,
        mode="segment-centered",
        label_policy="field_center",
        positive_radius_m=1.5,
    )
    val_table = build_local_proposal_table(
        bundles,
        stage1_outputs,
        split.val_plots,
        mode="segment-centered",
        label_policy="field_center",
        positive_radius_m=1.5,
    )
    test_table = build_local_proposal_table(
        bundles,
        stage1_outputs,
        split.test_plots,
        mode="segment-centered",
        label_policy="field_center",
        positive_radius_m=1.5,
    )
    train_table = _sample_train_table(train_table, seed=seed, max_train_samples=max_train_samples)
    common = {
        "plot_bundles": bundles,
        "stage1_outputs": stage1_outputs,
        "patch_size": patch_size,
        "n_points": n_points,
        "use_rgb": use_rgb,
        "use_lidar": use_lidar,
        "use_chm": use_chm,
        "include_geom_features": True,
        "include_raster": include_raster,
        "include_point_cloud": include_point_cloud,
        "geom_feature_set": geom_feature_set,
    }
    train_dataset = LocalProposalDataset(sample_table=train_table, training=True, seed=seed, **common)
    val_dataset = LocalProposalDataset(sample_table=val_table, training=False, seed=seed, **common)
    test_dataset = LocalProposalDataset(sample_table=test_table, training=False, seed=seed, **common)
    return train_table, val_table, test_table, train_dataset, val_dataset, test_dataset


def _score_model(model, dataset: LocalProposalDataset, model_kind: str, device: str, batch_size: int = 128) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raster = batch["raster"].to(device)
            point_cloud = batch["point_cloud"].to(device)
            geom = batch["geom_features"].to(device)
            if model_kind == "point":
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
                        "geom_h": float(base_h[idx]),
                        "refined_x": float(xs[idx] + offsets[idx, 0]),
                        "refined_y": float(ys[idx] + offsets[idx, 1]),
                        "refined_h": float(base_h[idx] + delta_h[idx]),
                        "score": float(scores[idx]),
                        "split_score": float(split_scores[idx]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _merge_scored(sample_table: pd.DataFrame, scored: pd.DataFrame, prefix: str) -> pd.DataFrame:
    table = sample_table.copy()
    scored_prefixed = scored.rename(
        columns={
            "refined_x": f"{prefix}_x",
            "refined_y": f"{prefix}_y",
            "refined_h": f"{prefix}_h",
            "score": f"{prefix}_score",
            "split_score": f"{prefix}_split_score",
        }
    )
    keep_columns = ["plot_id", "segment_id", f"{prefix}_x", f"{prefix}_y", f"{prefix}_h", f"{prefix}_score", f"{prefix}_split_score"]
    merged = table.merge(scored_prefixed[keep_columns], on=["plot_id", "segment_id"], how="left")
    return merged


def _sort_score(table: pd.DataFrame, score_col: str, tie_break: str) -> np.ndarray:
    score = table[score_col].to_numpy(dtype=float)
    if tie_break == "score_height":
        return score + 1e-3 * table["height_pred"].to_numpy(dtype=float)
    if tie_break == "score_area":
        return score + 1e-4 * table.get("area_m2", pd.Series(np.zeros(len(table)))).to_numpy(dtype=float)
    return score


def _nms(predictions: pd.DataFrame, radius_m: float, score_col: str, tie_break: str) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    working = predictions.copy()
    working["_sort_score"] = _sort_score(working, score_col=score_col, tie_break=tie_break)
    ordered = working.sort_values("_sort_score", ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[tuple[float, float]] = []
    for _, row in ordered.iterrows():
        point = (float(row["x"]), float(row["y"]))
        if any(np.hypot(point[0] - x, point[1] - y) <= float(radius_m) for x, y in kept_xy):
            continue
        kept_rows.append(row.drop(labels="_sort_score").to_dict())
        kept_xy.append(point)
    return pd.DataFrame.from_records(kept_rows)


def _build_predictions(
    table: pd.DataFrame,
    score_col: str,
    threshold: float,
    nms_radius_m: float,
    center_source: str,
    height_source: str,
    min_area_m2: float,
    tie_break: str,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "score"])
    selected = table[table[score_col] >= float(threshold)].copy()
    if "area_m2" in selected.columns and min_area_m2 > 0:
        selected = selected[selected["area_m2"] >= float(min_area_m2)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "score"])

    if center_source == "geom":
        selected["x"] = selected["x"]
        selected["y"] = selected["y"]
    elif center_source == "point":
        selected["x"] = selected["point_x"]
        selected["y"] = selected["point_y"]
    elif center_source == "hybrid":
        selected["x"] = selected["hybrid_x"]
        selected["y"] = selected["hybrid_y"]
    elif center_source == "strong":
        selected["x"] = selected["strong_x"]
        selected["y"] = selected["strong_y"]
    elif center_source == "avg_point_hybrid":
        selected["x"] = 0.5 * (selected["point_x"] + selected["hybrid_x"])
        selected["y"] = 0.5 * (selected["point_y"] + selected["hybrid_y"])
    elif center_source == "avg_geom_strong":
        selected["x"] = 0.5 * (selected["x"] + selected["strong_x"])
        selected["y"] = 0.5 * (selected["y"] + selected["strong_y"])
    else:
        raise ValueError(f"Unsupported center source: {center_source}")

    if height_source == "geom":
        selected["height_pred"] = selected["height_pred"]
    elif height_source == "point":
        selected["height_pred"] = selected["point_h"]
    elif height_source == "hybrid":
        selected["height_pred"] = selected["hybrid_h"]
    elif height_source == "strong":
        selected["height_pred"] = selected["strong_h"]
    elif height_source == "avg_point_hybrid":
        selected["height_pred"] = 0.5 * (selected["point_h"] + selected["hybrid_h"])
    elif height_source == "avg_geom_strong":
        selected["height_pred"] = 0.5 * (selected["height_pred"] + selected["strong_h"])
    else:
        raise ValueError(f"Unsupported height source: {height_source}")

    selected["score"] = selected[score_col]
    return _nms(selected[["x", "y", "height_pred", "score", "area_m2"]].copy(), radius_m=nms_radius_m, score_col="score", tie_break=tie_break)


def _summarize_predictions(predictions_by_plot: dict[str, pd.DataFrame], bundles, plot_ids: list[str], match_radius_m: float, height_tolerance_m: float) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    for plot_id in plot_ids:
        matches = match_tree_predictions(
            predictions_by_plot.get(plot_id, pd.DataFrame(columns=["x", "y", "height_pred"])),
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


def _calibrate(
    val_by_plot: dict[str, pd.DataFrame],
    bundles,
    plot_ids: list[str],
    score_col: str,
    center_sources: list[str],
    height_sources: list[str],
    threshold_grid: list[float],
    nms_grid: list[float],
    min_area_grid: list[float],
    tie_breaks: list[str],
    match_radius_m: float,
    height_tolerance_m: float,
) -> tuple[dict, dict]:
    best_params = {
        "score_col": score_col,
        "threshold": 0.5,
        "nms_radius_m": 1.2,
        "center_source": center_sources[0],
        "height_source": height_sources[0],
        "min_area_m2": 0.0,
        "tie_break": tie_breaks[0],
    }
    best_scores = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in threshold_grid:
        for nms_radius_m in nms_grid:
            for min_area_m2 in min_area_grid:
                for center_source in center_sources:
                    for height_source in height_sources:
                        for tie_break in tie_breaks:
                            predictions_by_plot = {}
                            for plot_id in plot_ids:
                                predictions_by_plot[plot_id] = _build_predictions(
                                    val_by_plot[plot_id],
                                    score_col=score_col,
                                    threshold=threshold,
                                    nms_radius_m=nms_radius_m,
                                    center_source=center_source,
                                    height_source=height_source,
                                    min_area_m2=min_area_m2,
                                    tie_break=tie_break,
                                )
                            _, summary = _summarize_predictions(
                                predictions_by_plot,
                                bundles=bundles,
                                plot_ids=plot_ids,
                                match_radius_m=match_radius_m,
                                height_tolerance_m=height_tolerance_m,
                            )
                            if summary["f1"] > best_scores["f1"] or (
                                summary["f1"] == best_scores["f1"] and summary["recall"] > best_scores["recall"]
                            ):
                                best_params = {
                                    "score_col": score_col,
                                    "threshold": float(threshold),
                                    "nms_radius_m": float(nms_radius_m),
                                    "center_source": str(center_source),
                                    "height_source": str(height_source),
                                    "min_area_m2": float(min_area_m2),
                                    "tie_break": str(tie_break),
                                }
                                best_scores = summary
    return best_params, best_scores


def _apply_with_params(table_by_plot: dict[str, pd.DataFrame], plot_ids: list[str], params: dict) -> dict[str, pd.DataFrame]:
    predictions = {}
    for plot_id in plot_ids:
        predictions[plot_id] = _build_predictions(
            table_by_plot[plot_id],
            score_col=params["score_col"],
            threshold=params["threshold"],
            nms_radius_m=params["nms_radius_m"],
            center_source=params["center_source"],
            height_source=params["height_source"],
            min_area_m2=params["min_area_m2"],
            tie_break=params["tie_break"],
        )
    return predictions


def _load_checkpoint_model(model, checkpoint_path: Path, device: str):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _train_strong_hybrid(
    config: ExperimentConfig,
    bundles,
    stage1_outputs,
    splits,
    output_root: Path,
    device: str,
) -> tuple[pd.DataFrame, dict[int, dict[str, pd.DataFrame]], dict[int, dict[str, pd.DataFrame]]]:
    method_name = "watershed_fixo.hybrid_rgb_strong"
    experiment_dir = ensure_dir(output_root / "predictions" / method_name)
    save_config(config, experiment_dir / "config.yaml")
    metrics_path = experiment_dir / "metrics_by_plot.csv"
    if metrics_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        if "protocol_id" not in existing_metrics.columns:
            existing_metrics["protocol_id"] = PROTOCOL_A_ID
            existing_metrics["stage1"] = "watershed_fixo"
            existing_metrics["stage2"] = "hybrid_rgb_strong"
            existing_metrics["labels"] = "field_center"
            existing_metrics.to_csv(metrics_path, index=False)
        summary = build_method_summary(existing_metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
        summary.to_csv(experiment_dir / "summary.csv", index=False)
        return existing_metrics, {}, {}
    metric_rows: list[pd.DataFrame] = []
    val_scored_by_fold: dict[int, dict[str, pd.DataFrame]] = {}
    test_scored_by_fold: dict[int, dict[str, pd.DataFrame]] = {}
    for split in splits:
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")
        train_table, val_table, test_table, train_dataset, val_dataset, test_dataset = _build_datasets(
            bundles=bundles,
            stage1_outputs=stage1_outputs,
            split=split,
            patch_size=96,
            n_points=128,
            use_rgb=True,
            use_lidar=False,
            use_chm=True,
            include_raster=True,
            include_point_cloud=True,
            geom_feature_set="rich",
            seed=config.training.seed + split.fold_id,
            max_train_samples=640,
        )
        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        model = StrongRGBHybridLocalRefiner(
            in_channels=5,
            rgb_channels=3,
            point_dim=6,
            geom_dim=train_dataset.geom_feature_dim,
        ).to(device)

        def validation_callback(current_model):
            scored = _score_model(current_model, val_dataset, model_kind="strong_hybrid", device=device)
            merged_val = _merge_scored(val_table, scored, prefix="strong")
            val_by_plot = {plot_id: merged_val[merged_val["plot_id"] == plot_id].copy() for plot_id in split.val_plots}
            params, scores = _calibrate(
                val_by_plot=val_by_plot,
                bundles=bundles,
                plot_ids=split.val_plots,
                score_col="strong_score",
                center_sources=["strong", "geom", "avg_geom_strong"],
                height_sources=["strong", "geom", "avg_geom_strong"],
                threshold_grid=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
                nms_grid=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
                min_area_grid=[0.0, 2.0, 4.0, 6.0],
                tie_breaks=["score", "score_height"],
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            write_json({"best_params": params, "best_scores": scores}, fold_dir / "validation_best.json")
            return scores

        fit_result = fit_stage2_model(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_kind="hybrid",
            device=device,
            checkpoint_dir=fold_dir,
            epochs=config.hybrid.epochs,
            learning_rate=config.hybrid.learning_rate,
            weight_decay=config.hybrid.weight_decay,
            patience=config.hybrid.patience,
            validation_callback=validation_callback,
        )
        pd.DataFrame.from_records(fit_result["history"]).to_csv(fold_dir / "history.csv", index=False)
        model = _load_checkpoint_model(model, fold_dir / "best.pt", device=device)
        scored_val = _score_model(model, val_dataset, model_kind="strong_hybrid", device=device)
        scored_test = _score_model(model, test_dataset, model_kind="strong_hybrid", device=device)
        merged_val = _merge_scored(val_table, scored_val, prefix="strong")
        merged_test = _merge_scored(test_table, scored_test, prefix="strong")
        val_scored_by_fold[split.fold_id] = {plot_id: merged_val[merged_val["plot_id"] == plot_id].copy() for plot_id in split.val_plots}
        test_scored_by_fold[split.fold_id] = {plot_id: merged_test[merged_test["plot_id"] == plot_id].copy() for plot_id in split.test_plots}

        params, scores = _calibrate(
            val_by_plot=val_scored_by_fold[split.fold_id],
            bundles=bundles,
            plot_ids=split.val_plots,
            score_col="strong_score",
            center_sources=["strong", "geom", "avg_geom_strong"],
            height_sources=["strong", "geom", "avg_geom_strong"],
            threshold_grid=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
            nms_grid=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
            min_area_grid=[0.0, 2.0, 4.0, 6.0],
            tie_breaks=["score", "score_height"],
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        write_json({"best_params": params, "best_scores": scores}, fold_dir / "best_params.json")
        test_predictions = _apply_with_params(test_scored_by_fold[split.fold_id], split.test_plots, params)
        for plot_id, predictions in test_predictions.items():
            predictions.to_csv(fold_dir / f"{plot_id}.csv", index=False)
        fold_table, _ = _summarize_predictions(
            test_predictions,
            bundles=bundles,
            plot_ids=split.test_plots,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        fold_table["fold_id"] = split.fold_id
        fold_table["method"] = method_name
        fold_table["protocol_id"] = PROTOCOL_A_ID
        fold_table["stage1"] = "watershed_fixo"
        fold_table["stage2"] = "hybrid_rgb_strong"
        fold_table["labels"] = "field_center"
        metric_rows.append(fold_table)
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(experiment_dir / "metrics_by_plot.csv", index=False)
    summary = build_method_summary(metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    summary.to_csv(experiment_dir / "summary.csv", index=False)
    return metrics, val_scored_by_fold, test_scored_by_fold


def _score_existing_method(
    bundles,
    stage1_outputs,
    splits,
    output_root: Path,
    experiment_name: str,
    model_type: str,
    device: str,
) -> tuple[dict[int, dict[str, pd.DataFrame]], dict[int, dict[str, pd.DataFrame]]]:
    experiment_dir = output_root / "predictions" / experiment_name
    val_scored_by_fold: dict[int, dict[str, pd.DataFrame]] = {}
    test_scored_by_fold: dict[int, dict[str, pd.DataFrame]] = {}
    for split in splits:
        if model_type == "point":
            train_table, val_table, test_table, _, val_dataset, test_dataset = _build_datasets(
                bundles=bundles,
                stage1_outputs=stage1_outputs,
                split=split,
                patch_size=64,
                n_points=128,
                use_rgb=False,
                use_lidar=False,
                use_chm=False,
                include_raster=False,
                include_point_cloud=True,
                geom_feature_set="base",
                seed=13 + split.fold_id,
                max_train_samples=512,
            )
            model = PointLocalRefiner(point_dim=6, geom_dim=val_dataset.geom_feature_dim)
        elif model_type == "hybrid_rgb_chm":
            train_table, val_table, test_table, _, val_dataset, test_dataset = _build_datasets(
                bundles=bundles,
                stage1_outputs=stage1_outputs,
                split=split,
                patch_size=64,
                n_points=128,
                use_rgb=True,
                use_lidar=False,
                use_chm=True,
                include_raster=True,
                include_point_cloud=True,
                geom_feature_set="base",
                seed=13 + split.fold_id,
                max_train_samples=512,
            )
            model = HybridLocalRefiner(in_channels=5, point_dim=6, geom_dim=val_dataset.geom_feature_dim)
        else:
            raise ValueError(model_type)

        checkpoint_path = experiment_dir / f"fold_{split.fold_id:02d}" / "best.pt"
        model = _load_checkpoint_model(model, checkpoint_path, device=device)
        scored_val = _score_model(model, val_dataset, model_kind="point" if model_type == "point" else "hybrid", device=device)
        scored_test = _score_model(model, test_dataset, model_kind="point" if model_type == "point" else "hybrid", device=device)
        prefix = "point" if model_type == "point" else "hybrid"
        merged_val = _merge_scored(val_table, scored_val, prefix=prefix)
        merged_test = _merge_scored(test_table, scored_test, prefix=prefix)
        val_scored_by_fold[split.fold_id] = {plot_id: merged_val[merged_val["plot_id"] == plot_id].copy() for plot_id in split.val_plots}
        test_scored_by_fold[split.fold_id] = {plot_id: merged_test[merged_test["plot_id"] == plot_id].copy() for plot_id in split.test_plots}
    return val_scored_by_fold, test_scored_by_fold


def _evaluate_method(
    method_name: str,
    output_root: Path,
    bundles,
    splits,
    val_tables_by_fold: dict[int, dict[str, pd.DataFrame]],
    test_tables_by_fold: dict[int, dict[str, pd.DataFrame]],
    score_col: str,
    center_sources: list[str],
    height_sources: list[str],
    threshold_grid: list[float],
    nms_grid: list[float],
    min_area_grid: list[float],
    tie_breaks: list[str],
    protocol_id: str,
    stage1: str,
    stage2: str,
) -> pd.DataFrame:
    experiment_dir = ensure_dir(output_root / "predictions" / method_name)
    rows: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []
    for split in splits:
        fold_dir = ensure_dir(experiment_dir / f"fold_{split.fold_id:02d}")
        params, scores = _calibrate(
            val_by_plot=val_tables_by_fold[split.fold_id],
            bundles=bundles,
            plot_ids=split.val_plots,
            score_col=score_col,
            center_sources=center_sources,
            height_sources=height_sources,
            threshold_grid=threshold_grid,
            nms_grid=nms_grid,
            min_area_grid=min_area_grid,
            tie_breaks=tie_breaks,
            match_radius_m=1.5,
            height_tolerance_m=3.0,
        )
        write_json({"best_params": params, "best_scores": scores}, fold_dir / "best_params.json")
        predictions_by_plot = _apply_with_params(test_tables_by_fold[split.fold_id], split.test_plots, params)
        for plot_id, table in predictions_by_plot.items():
            table.to_csv(fold_dir / f"{plot_id}.csv", index=False)
        fold_table, _ = _summarize_predictions(
            predictions_by_plot,
            bundles=bundles,
            plot_ids=split.test_plots,
            match_radius_m=1.5,
            height_tolerance_m=3.0,
        )
        fold_table["fold_id"] = split.fold_id
        fold_table["method"] = method_name
        fold_table["protocol_id"] = protocol_id
        fold_table["stage1"] = stage1
        fold_table["stage2"] = stage2
        fold_table["labels"] = "field_center"
        rows.append(fold_table)
        calibration_rows.append({"fold_id": split.fold_id, **params, **scores})
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(experiment_dir / "metrics_by_plot.csv", index=False)
    pd.DataFrame.from_records(calibration_rows).to_csv(experiment_dir / "calibration.csv", index=False)
    summary = build_method_summary(metrics, bootstrap_iterations=256)
    summary.to_csv(experiment_dir / "summary.csv", index=False)
    return metrics


def _combine_tables(left_by_fold: dict[int, dict[str, pd.DataFrame]], right_by_fold: dict[int, dict[str, pd.DataFrame]]) -> tuple[dict[int, dict[str, pd.DataFrame]], dict[int, dict[str, pd.DataFrame]]]:
    combined_val: dict[int, dict[str, pd.DataFrame]] = {}
    combined_test: dict[int, dict[str, pd.DataFrame]] = {}
    return combined_val, combined_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run next candidate options on protocol A.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs_ticketfinal_next_options")
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = build_protocol_a_splits(list(bundles.keys()))
    output_root = ensure_dir(Path(args.output_dir))
    ensure_dir(output_root / "predictions")
    ensure_dir(output_root / "metrics")
    save_config(config, output_root / "config.yaml")

    stage1_outputs = {plot_id: WatershedStage1Generator().generate(bundle) for plot_id, bundle in bundles.items()}

    # Option 1: stronger hybrid with richer RGB + segment geometry.
    strong_metrics, strong_val, strong_test = _train_strong_hybrid(
        config=config,
        bundles=bundles,
        stage1_outputs=stage1_outputs,
        splits=splits,
        output_root=output_root,
        device=args.device,
    )

    # Re-score existing current best hybrid (rgb_chm) and point for options 2-4.
    point_val, point_test = _score_existing_method(
        bundles=bundles,
        stage1_outputs=stage1_outputs,
        splits=splits,
        output_root=Path("outputs_ticketfinal_v2"),
        experiment_name=CURRENT_POINT_EXPERIMENT,
        model_type="point",
        device=args.device,
    )
    hybrid_val, hybrid_test = _score_existing_method(
        bundles=bundles,
        stage1_outputs=stage1_outputs,
        splits=splits,
        output_root=Path("outputs_ticketfinal_v2"),
        experiment_name=CURRENT_HYBRID_RGB_CHM_EXPERIMENT,
        model_type="hybrid_rgb_chm",
        device=args.device,
    )

    # Option 4: expanded calibration on current hybrid rgb_chm.
    recal_metrics = _evaluate_method(
        method_name="watershed_fixo.hybrid_rgb_chm.recalibrated",
        output_root=output_root,
        bundles=bundles,
        splits=splits,
        val_tables_by_fold=hybrid_val,
        test_tables_by_fold=hybrid_test,
        score_col="hybrid_score",
        center_sources=["hybrid", "geom"],
        height_sources=["hybrid", "geom"],
        threshold_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70],
        nms_grid=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        min_area_grid=[0.0, 2.0, 4.0, 6.0],
        tie_breaks=["score", "score_height", "score_area"],
        protocol_id=PROTOCOL_A_ID,
        stage1="watershed_fixo",
        stage2="hybrid_recalibrated",
    )

    # Merge point + current hybrid tables for options 2 and 3.
    ensemble_val: dict[int, dict[str, pd.DataFrame]] = {}
    ensemble_test: dict[int, dict[str, pd.DataFrame]] = {}
    stack_val: dict[int, dict[str, pd.DataFrame]] = {}
    stack_test: dict[int, dict[str, pd.DataFrame]] = {}
    stack_train_rows: list[dict] = []
    stack_test_rows: list[dict] = []
    for split in splits:
        fold_id = split.fold_id
        ensemble_val[fold_id] = {}
        ensemble_test[fold_id] = {}
        stack_val[fold_id] = {}
        stack_test[fold_id] = {}
        for plot_id in split.val_plots:
            merged = point_val[fold_id][plot_id].merge(
                hybrid_val[fold_id][plot_id][["plot_id", "segment_id", "hybrid_x", "hybrid_y", "hybrid_h", "hybrid_score", "hybrid_split_score"]],
                on=["plot_id", "segment_id"],
                how="inner",
            )
            merged["agreement_dist"] = np.hypot(merged["point_x"] - merged["hybrid_x"], merged["point_y"] - merged["hybrid_y"])
            merged["point_offset_norm"] = np.hypot(merged["point_x"] - merged["x"], merged["point_y"] - merged["y"])
            merged["hybrid_offset_norm"] = np.hypot(merged["hybrid_x"] - merged["x"], merged["hybrid_y"] - merged["y"])
            ensemble_val[fold_id][plot_id] = merged.copy()
            stack_val[fold_id][plot_id] = merged.copy()
        for plot_id in split.test_plots:
            merged = point_test[fold_id][plot_id].merge(
                hybrid_test[fold_id][plot_id][["plot_id", "segment_id", "hybrid_x", "hybrid_y", "hybrid_h", "hybrid_score", "hybrid_split_score"]],
                on=["plot_id", "segment_id"],
                how="inner",
            )
            merged["agreement_dist"] = np.hypot(merged["point_x"] - merged["hybrid_x"], merged["point_y"] - merged["hybrid_y"])
            merged["point_offset_norm"] = np.hypot(merged["point_x"] - merged["x"], merged["point_y"] - merged["y"])
            merged["hybrid_offset_norm"] = np.hypot(merged["hybrid_x"] - merged["x"], merged["hybrid_y"] - merged["y"])
            ensemble_test[fold_id][plot_id] = merged.copy()
            stack_test[fold_id][plot_id] = merged.copy()

        val_train = pd.concat(list(stack_val[fold_id].values()), ignore_index=True)
        test_eval = pd.concat(list(stack_test[fold_id].values()), ignore_index=True)
        feature_cols = [
            "geometry_score",
            "area_m2",
            "height_pred",
            "height_mean_m",
            "elongation",
            "compactness",
            "split_flag",
            "duplicate_flag",
            "point_score",
            "hybrid_score",
            "point_split_score",
            "hybrid_split_score",
            "agreement_dist",
            "point_offset_norm",
            "hybrid_offset_norm",
        ]
        stacker = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
            ]
        )
        stacker.fit(val_train[feature_cols].fillna(0.0), val_train["objectness"].astype(int))
        val_train = val_train.copy()
        test_eval = test_eval.copy()
        val_train["stack_score"] = stacker.predict_proba(val_train[feature_cols].fillna(0.0))[:, 1]
        test_eval["stack_score"] = stacker.predict_proba(test_eval[feature_cols].fillna(0.0))[:, 1]
        stack_val[fold_id] = {plot_id: part.copy() for plot_id, part in val_train.groupby("plot_id")}
        stack_test[fold_id] = {plot_id: part.copy() for plot_id, part in test_eval.groupby("plot_id")}

        for w in [0.35, 0.50, 0.65]:
            pass

    # Option 3: point + hybrid ensemble.
    for fold_id, plot_tables in ensemble_val.items():
        for plot_id, table in plot_tables.items():
            table["ensemble_score"] = 0.5 * table["point_score"] + 0.5 * table["hybrid_score"]
        for plot_id, table in ensemble_test[fold_id].items():
            table["ensemble_score"] = 0.5 * table["point_score"] + 0.5 * table["hybrid_score"]
    ensemble_metrics = _evaluate_method(
        method_name="watershed_fixo.point_hybrid_ensemble",
        output_root=output_root,
        bundles=bundles,
        splits=splits,
        val_tables_by_fold=ensemble_val,
        test_tables_by_fold=ensemble_test,
        score_col="ensemble_score",
        center_sources=["point", "hybrid", "avg_point_hybrid"],
        height_sources=["point", "hybrid", "avg_point_hybrid"],
        threshold_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
        nms_grid=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        min_area_grid=[0.0, 2.0, 4.0, 6.0],
        tie_breaks=["score", "score_height", "score_area"],
        protocol_id=PROTOCOL_A_ID,
        stage1="watershed_fixo",
        stage2="point_hybrid_ensemble",
    )

    # Option 2: tabular stacker on segment features + point/hybrid scores.
    stack_metrics = _evaluate_method(
        method_name="watershed_fixo.segment_stacker_logreg",
        output_root=output_root,
        bundles=bundles,
        splits=splits,
        val_tables_by_fold=stack_val,
        test_tables_by_fold=stack_test,
        score_col="stack_score",
        center_sources=["point", "hybrid", "avg_point_hybrid"],
        height_sources=["point", "hybrid", "avg_point_hybrid"],
        threshold_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60],
        nms_grid=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        min_area_grid=[0.0, 2.0, 4.0, 6.0],
        tie_breaks=["score", "score_height", "score_area"],
        protocol_id=PROTOCOL_A_ID,
        stage1="watershed_fixo",
        stage2="stacker_logreg",
    )

    # Consolidate.
    summary_rows = []
    for method_name in [
        "watershed_fixo.hybrid_rgb_strong",
        "watershed_fixo.hybrid_rgb_chm.recalibrated",
        "watershed_fixo.point_hybrid_ensemble",
        "watershed_fixo.segment_stacker_logreg",
    ]:
        summary_path = output_root / "predictions" / method_name / "summary.csv"
        summary = pd.read_csv(summary_path)
        summary["method_name"] = method_name
        summary_rows.append(summary)
    summary_table = pd.concat(summary_rows, ignore_index=True)
    summary_table.to_csv(output_root / "metrics" / "next_options_summary.csv", index=False)

    current_best_matrix = pd.read_csv("outputs_ticketfinal_v2/metrics/main_matrix_results.csv").sort_values("f1_mean", ascending=False).iloc[0]
    current_best_rgb = pd.read_csv("outputs_ticketfinal_v2/metrics/rgb_contribution_ablation.csv").sort_values("f1", ascending=False).iloc[0]
    comparison_rows = []
    for _, row in summary_table.iterrows():
        comparison_rows.append(
            {
                "method": row["method_name"],
                "f1_mean": float(row["f1_mean"]),
                "delta_vs_main_matrix_best": float(row["f1_mean"] - current_best_matrix["f1_mean"]),
                "delta_vs_rgb_ablation_best": float(row["f1_mean"] - current_best_rgb["f1"]),
            }
        )
    comparison = pd.DataFrame.from_records(comparison_rows).sort_values("f1_mean", ascending=False)
    comparison.to_csv(output_root / "metrics" / "comparison_vs_current_best.csv", index=False)


if __name__ == "__main__":
    main()

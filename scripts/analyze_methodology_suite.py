from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.methodology import (
    add_site_column,
    extract_site_id,
    map_failure_mode,
    summarize_failure_modes,
    summarize_match_radius_sensitivity,
    summarize_site_metrics,
    summarize_threshold_sweep,
)
from lidaragrosat.analysis.reporting import aggregate_plot_metrics, save_error_figure
from lidaragrosat.analysis.scientific import classify_error_modes
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import load_experiment_manifest
from lidaragrosat.inference.sliding_window import build_chm_features_if_needed, decode_tree_predictions, predict_plot_sliding_window
from lidaragrosat.labels import resolve_reference_points
from lidaragrosat.models.detector import build_detector_from_config
from lidaragrosat.stage2.postprocess import apply_stage2_postprocess
from lidaragrosat.utils.io import ensure_dir, read_json, write_json


EMPTY_PREDICTION_COLUMNS = ["x", "y", "height_pred", "confidence"]


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


def _load_prediction_table(path: Path) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)
    if table.empty and len(table.columns) == 0:
        return pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)
    return table


def _clean_method_label(method: str, stage1: str, stage2: str) -> str:
    clean_stage1 = str(stage1 or "").strip()
    clean_stage2 = str(stage2 or "").strip()
    if clean_stage1 and clean_stage1 != "none":
        if clean_stage2 and clean_stage2 != "none":
            return f"{clean_stage1}+{clean_stage2}"
        return clean_stage1
    if clean_stage2 and clean_stage2 != "none":
        return clean_stage2
    return str(method)


def _clean_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    cleaned = metrics.copy()
    if {"method", "stage1", "stage2"}.issubset(cleaned.columns):
        cleaned["method_clean"] = [
            _clean_method_label(method, stage1, stage2)
            for method, stage1, stage2 in zip(cleaned["method"], cleaned["stage1"], cleaned["stage2"], strict=False)
        ]
    else:
        cleaned["method_clean"] = cleaned["method"]
    return cleaned


def _collect_method_specs(prediction_root: Path) -> list[dict]:
    specs: list[dict] = []
    if not prediction_root.exists():
        return specs
    for method_dir in sorted(path for path in prediction_root.iterdir() if path.is_dir()):
        manifest_path = method_dir / "experiment_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_experiment_manifest(manifest_path)
        display_name = _clean_method_label(manifest.experiment_name, manifest.stage1, manifest.stage2)
        specs.append(
            {
                "display_name": display_name,
                "dir": method_dir,
                "manifest": manifest,
            }
        )
    return specs


def _load_predictions_by_plot(method_dir: Path) -> dict[str, pd.DataFrame]:
    predictions_by_plot: dict[str, pd.DataFrame] = {}
    for fold_dir in sorted(path for path in method_dir.glob("fold_*") if path.is_dir()):
        for csv_path in sorted(fold_dir.glob("*.csv")):
            if csv_path.name in {"history.csv", "train_proposals.csv", "val_proposals.csv", "test_proposals.csv", "scored_val.csv", "scored_test.csv"}:
                continue
            predictions_by_plot[csv_path.stem] = _load_prediction_table(csv_path)
    return predictions_by_plot


def _summarize_predictions_by_plot(
    method_name: str,
    predictions_by_plot: dict[str, pd.DataFrame],
    bundles,
    plot_ids: list[str],
    match_radius_m: float,
    height_tolerance_m: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for plot_id in plot_ids:
        predictions = predictions_by_plot.get(plot_id, pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS))
        match_result = match_tree_predictions(
            predictions,
            bundles[plot_id].reference,
            match_radius_m=match_radius_m,
            height_tolerance_m=height_tolerance_m,
        )
        rows.append(
            aggregate_plot_metrics(
                method_name,
                plot_id,
                match_result,
            )
        )
    return pd.DataFrame.from_records(rows)


def _greedy_nms(predictions: pd.DataFrame, radius_m: float, score_column: str = "confidence") -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    if radius_m <= 0:
        return predictions.copy().reset_index(drop=True)
    ordered = predictions.sort_values(score_column, ascending=False).reset_index(drop=True)
    kept_rows: list[dict] = []
    kept_xy: list[np.ndarray] = []
    for _, row in ordered.iterrows():
        point = np.array([float(row["x"]), float(row["y"])], dtype=float)
        if any(np.linalg.norm(point - current) <= float(radius_m) for current in kept_xy):
            continue
        kept_rows.append(row.to_dict())
        kept_xy.append(point)
    return pd.DataFrame.from_records(kept_rows)


def _compute_site_input_statistics(bundles, output_dir: Path) -> None:
    rows: list[dict] = []
    for plot_id, bundle in bundles.items():
        channel_names = (bundle.metadata or {}).get("structural_channels", [])
        roughness_mean = np.nan
        if "vertical_roughness" in channel_names:
            roughness_mean = float(bundle.lidar[channel_names.index("vertical_roughness")].mean())
        rows.append(
            {
                "plot_id": plot_id,
                "site_id": extract_site_id(plot_id),
                "rgb_mean": float(bundle.rgb.mean()),
                "rgb_std": float(bundle.rgb.std()),
                "chm_mean": float(np.nanmean(bundle.chm)),
                "chm_std": float(np.nanstd(bundle.chm)),
                "vertical_roughness_mean": roughness_mean,
            }
        )
    plot_table = pd.DataFrame.from_records(rows).sort_values(["site_id", "plot_id"]).reset_index(drop=True)
    plot_table.to_csv(output_dir / "site_input_statistics_by_plot.csv", index=False)
    site_table = (
        plot_table.groupby("site_id")[["rgb_mean", "rgb_std", "chm_mean", "chm_std", "vertical_roughness_mean"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    flat_columns = []
    for column in site_table.columns:
        if not isinstance(column, tuple):
            flat_columns.append(str(column))
        elif column[0] == "site_id":
            flat_columns.append("site_id")
        else:
            flat_columns.append(f"{column[0]}_{column[1]}")
    site_table.columns = flat_columns
    site_table.to_csv(output_dir / "site_input_statistics_by_site.csv", index=False)


def _compute_label_alignment(config: ExperimentConfig, bundles, plot_ids: list[str], output_dir: Path) -> None:
    rows: list[dict] = []
    all_shift_distances: list[float] = []
    for plot_id in plot_ids:
        bundle = bundles[plot_id]
        apex = resolve_reference_points(
            bundle,
            label_policy="apex_snapped",
            snap_radius_m=max(config.data.label_snap_radius_m, 1.0),
            mixed_weight=config.data.label_mixed_weight,
        )
        if apex.empty:
            rows.append({"plot_id": plot_id, "site_id": extract_site_id(plot_id), "mean_shift_m": 0.0, "median_shift_m": 0.0, "p95_shift_m": 0.0})
            continue
        shift = np.sqrt((apex["target_x"] - apex["x"]) ** 2 + (apex["target_y"] - apex["y"]) ** 2)
        shift_values = shift.to_numpy(dtype=float)
        all_shift_distances.extend(shift_values.tolist())
        rows.append(
            {
                "plot_id": plot_id,
                "site_id": extract_site_id(plot_id),
                "mean_shift_m": float(np.mean(shift_values)) if len(shift_values) else 0.0,
                "median_shift_m": float(np.median(shift_values)) if len(shift_values) else 0.0,
                "p95_shift_m": float(np.percentile(shift_values, 95)) if len(shift_values) else 0.0,
            }
        )
    table = pd.DataFrame.from_records(rows).sort_values(["site_id", "plot_id"]).reset_index(drop=True)
    table.to_csv(output_dir / "label_alignment_by_plot.csv", index=False)
    write_json(
        {
            "label_policy_train": config.benchmark.label_policy,
            "apex_snap_radius_m": float(max(config.data.label_snap_radius_m, 1.0)),
            "mean_shift_m": float(np.mean(all_shift_distances)) if all_shift_distances else 0.0,
            "median_shift_m": float(np.median(all_shift_distances)) if all_shift_distances else 0.0,
            "p95_shift_m": float(np.percentile(all_shift_distances, 95)) if all_shift_distances else 0.0,
        },
        output_dir / "label_alignment_summary.json",
    )


def _compute_scale_sanity(config: ExperimentConfig, bundles, plot_ids: list[str], output_dir: Path) -> None:
    nearest_neighbor_distances: list[float] = []
    for plot_id in plot_ids:
        reference = bundles[plot_id].reference
        if len(reference) < 2:
            continue
        xy = reference[["x", "y"]].to_numpy(dtype=float)
        distances = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        distances[distances == 0] = np.inf
        nearest_neighbor_distances.extend(np.min(distances, axis=1).tolist())
    write_json(
        {
            "tile_extent_m": float(config.data.tile_size * config.data.raster_resolution),
            "match_radius_m": float(config.evaluation.match_radius_m),
            "median_reference_nearest_neighbor_m": float(np.median(nearest_neighbor_distances)) if nearest_neighbor_distances else 0.0,
            "p95_reference_nearest_neighbor_m": float(np.percentile(nearest_neighbor_distances, 95)) if nearest_neighbor_distances else 0.0,
        },
        output_dir / "scale_sanity.json",
    )


def _run_site_analysis(main_metrics: pd.DataFrame, methodology_metrics_dir: Path, output_dir: Path) -> None:
    cleaned_main = _clean_metrics_table(main_metrics)
    cleaned_main["method_raw"] = cleaned_main["method"]
    cleaned_main["method"] = cleaned_main["method_clean"]
    cleaned_main = cleaned_main.drop(columns=["method_clean"])
    cleaned_main.to_csv(output_dir / "main_metrics_clean.csv", index=False)
    main_site_summary = summarize_site_metrics(cleaned_main)
    main_site_summary.to_csv(output_dir / "main_site_summary.csv", index=False)
    main_site_wins = count_site_wins(cleaned_main)
    main_site_wins.to_csv(output_dir / "main_site_wins.csv", index=False)

    for stem in ["stage1_only", "stage1_stage2", "modality_ablation"]:
        metrics_path = methodology_metrics_dir / f"{stem}_metrics_by_plot.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        metrics.to_csv(output_dir / f"{stem}_metrics_clean.csv", index=False)
        summarize_site_metrics(metrics).to_csv(output_dir / f"{stem}_site_summary.csv", index=False)
        count_site_wins(metrics).to_csv(output_dir / f"{stem}_site_wins.csv", index=False)


def _run_error_mode_analysis(method_specs: list[dict], bundles, config: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for spec in method_specs:
        predictions_by_plot = _load_predictions_by_plot(spec["dir"])
        for plot_id, predictions in predictions_by_plot.items():
            if plot_id not in bundles:
                continue
            match_result = match_tree_predictions(
                predictions,
                bundles[plot_id].reference,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            error_table = classify_error_modes(
                bundles[plot_id],
                predictions=predictions,
                match_result=match_result,
                overlap_radius_m=config.analysis.overlap_radius_m,
            )
            if error_table.empty:
                continue
            error_table = error_table.assign(
                method=spec["display_name"],
                site_id=extract_site_id(plot_id),
            )
            error_table["failure_mode"] = [
                map_failure_mode(mode, error_type)
                for mode, error_type in zip(error_table["mode"], error_table["error_type"], strict=False)
            ]
            rows.append(error_table)
    detailed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    detailed.to_csv(output_dir / "error_modes_detailed.csv", index=False)
    summary = summarize_failure_modes(detailed, group_columns=("method",))
    summary.to_csv(output_dir / "error_modes_summary.csv", index=False)
    site_summary = summarize_failure_modes(detailed, group_columns=("method", "site_id"))
    site_summary.to_csv(output_dir / "error_modes_by_site.csv", index=False)
    return detailed


def _run_matching_radius_sensitivity(method_specs: list[dict], bundles, config: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    plot_rows: list[dict] = []
    for spec in method_specs:
        predictions_by_plot = _load_predictions_by_plot(spec["dir"])
        plot_ids = sorted(plot_id for plot_id in predictions_by_plot if plot_id in bundles)
        for radius_m in config.evaluation.radius_candidates_m:
            metrics = _summarize_predictions_by_plot(
                method_name=spec["display_name"],
                predictions_by_plot=predictions_by_plot,
                bundles=bundles,
                plot_ids=plot_ids,
                match_radius_m=float(radius_m),
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            if metrics.empty:
                continue
            metrics["match_radius_m"] = float(radius_m)
            plot_rows.append(metrics)
    table = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
    table.to_csv(output_dir / "matching_radius_sensitivity_by_plot.csv", index=False)
    summary = summarize_match_radius_sensitivity(table)
    summary.to_csv(output_dir / "matching_radius_sensitivity_summary.csv", index=False)
    return table


def _predict_detector_maps(model, method_config: ExperimentConfig, bundles, plot_ids: list[str], device: str) -> dict[str, object]:
    prediction_maps = {}
    for plot_id in plot_ids:
        bundle = bundles[plot_id]
        chm_features = build_chm_features_if_needed(
            model=model,
            chm=bundle.chm,
            pixel_size=bundle.transform.pixel_size,
            peakness_window_sizes_m=method_config.data.peakness_window_sizes_m,
            min_height_m=method_config.data.min_tree_height_m,
            precomputed=getattr(bundle, "chm_features", None),
        )
        prediction_maps[plot_id] = predict_plot_sliding_window(
            model,
            rgb=bundle.rgb,
            lidar=bundle.lidar,
            tile_size=method_config.data.tile_size,
            overlap=method_config.data.tile_overlap,
            chm=chm_features,
            context_margin_px=method_config.data.context_margin_px,
            device=device,
        )
    return prediction_maps


def _summarize_curve_rows(
    method_name: str,
    split_name: str,
    predictions_by_plot: dict[str, pd.DataFrame],
    bundles,
    plot_ids: list[str],
    match_radius_m: float,
    height_tolerance_m: float,
    threshold: float,
    nms_radius_m: float,
    threshold_kind: str,
) -> dict:
    summaries = []
    total_predictions = 0
    zero_prediction_plots = 0
    for plot_id in plot_ids:
        predictions = predictions_by_plot[plot_id]
        total_predictions += int(len(predictions))
        zero_prediction_plots += int(len(predictions) == 0)
        match_result = match_tree_predictions(
            predictions,
            bundles[plot_id].reference,
            match_radius_m=match_radius_m,
            height_tolerance_m=height_tolerance_m,
        )
        summaries.append(summarize_matches(match_result))
    return {
        "method": method_name,
        "split": split_name,
        "threshold_kind": threshold_kind,
        "threshold": float(threshold),
        "nms_radius_m": float(nms_radius_m),
        "precision_mean": float(np.mean([row["precision"] for row in summaries])) if summaries else 0.0,
        "recall_mean": float(np.mean([row["recall"] for row in summaries])) if summaries else 0.0,
        "f1_mean": float(np.mean([row["f1"] for row in summaries])) if summaries else 0.0,
        "predictions_total": int(total_predictions),
        "predictions_mean_per_plot": float(total_predictions / max(len(plot_ids), 1)),
        "zero_prediction_plots": int(zero_prediction_plots),
    }


def _run_detector_threshold_sweep(spec: dict, bundles, device: str) -> tuple[pd.DataFrame, dict]:
    method_dir = spec["dir"]
    method_config = ExperimentConfig.from_files(method_dir / "config.yaml")
    fold = _build_splits(method_config, bundles)[0]
    model = build_detector_from_config(method_config).to(device)
    state = torch.load(method_dir / "fold_00" / "best.pt", map_location=device)
    model.load_state_dict(state)
    prediction_maps = _predict_detector_maps(model, method_config, bundles, sorted(set(fold.val_plots + fold.test_plots)), device)

    rows: list[dict] = []
    threshold_grid = sorted(set(float(value) for value in method_config.evaluation.confidence_thresholds))
    nms_grid = sorted(set(float(value) for value in method_config.evaluation.nms_radius_grid_m))
    best_params_path = method_dir / "fold_00" / "best_decode_params.json"
    best_decode_params = read_json(best_params_path) if best_params_path.exists() else {
        "confidence_threshold": method_config.training.confidence_threshold,
        "nms_radius_m": method_config.evaluation.nms_radius_m,
        "min_pred_height_m": None,
        "min_chm_at_center_m": None,
        "border_margin_px": 0,
    }

    for split_name, plot_ids in [("val", fold.val_plots), ("test", fold.test_plots)]:
        for threshold, nms_radius_m in product(threshold_grid, nms_grid):
            predictions_by_plot = {}
            for plot_id in plot_ids:
                predictions_by_plot[plot_id] = decode_tree_predictions(
                    prediction_maps[plot_id],
                    transform=bundles[plot_id].transform,
                    confidence_threshold=float(threshold),
                    nms_radius_m=float(nms_radius_m),
                    min_pred_height_m=best_decode_params.get("min_pred_height_m"),
                    min_chm_at_center_m=best_decode_params.get("min_chm_at_center_m"),
                    border_margin_px=int(best_decode_params.get("border_margin_px", 0)),
                    chm=bundles[plot_id].chm,
                    adaptive_nms=method_config.evaluation.adaptive_nms,
                    adaptive_nms_r0_m=method_config.evaluation.adaptive_nms_r0_m,
                    adaptive_nms_alpha=method_config.evaluation.adaptive_nms_alpha,
                    adaptive_nms_rmin_m=method_config.evaluation.adaptive_nms_rmin_m,
                    adaptive_nms_rmax_m=method_config.evaluation.adaptive_nms_rmax_m,
                )
            rows.append(
                _summarize_curve_rows(
                    method_name=spec["display_name"],
                    split_name=split_name,
                    predictions_by_plot=predictions_by_plot,
                    bundles=bundles,
                    plot_ids=plot_ids,
                    match_radius_m=method_config.evaluation.match_radius_m,
                    height_tolerance_m=method_config.evaluation.height_tolerance_m,
                    threshold=float(threshold),
                    nms_radius_m=float(nms_radius_m),
                    threshold_kind="confidence",
                )
            )

    default_predictions = {}
    for plot_id in fold.test_plots:
        default_predictions[plot_id] = decode_tree_predictions(
            prediction_maps[plot_id],
            transform=bundles[plot_id].transform,
            confidence_threshold=float(best_decode_params["confidence_threshold"]),
            nms_radius_m=float(best_decode_params["nms_radius_m"]),
            min_pred_height_m=best_decode_params.get("min_pred_height_m"),
            min_chm_at_center_m=best_decode_params.get("min_chm_at_center_m"),
            border_margin_px=int(best_decode_params.get("border_margin_px", 0)),
            chm=bundles[plot_id].chm,
            adaptive_nms=method_config.evaluation.adaptive_nms,
            adaptive_nms_r0_m=method_config.evaluation.adaptive_nms_r0_m,
            adaptive_nms_alpha=method_config.evaluation.adaptive_nms_alpha,
            adaptive_nms_rmin_m=method_config.evaluation.adaptive_nms_rmin_m,
            adaptive_nms_rmax_m=method_config.evaluation.adaptive_nms_rmax_m,
        )
    default_summary = _summarize_curve_rows(
        method_name=spec["display_name"],
        split_name="test_default",
        predictions_by_plot=default_predictions,
        bundles=bundles,
        plot_ids=fold.test_plots,
        match_radius_m=method_config.evaluation.match_radius_m,
        height_tolerance_m=method_config.evaluation.height_tolerance_m,
        threshold=float(best_decode_params["confidence_threshold"]),
        nms_radius_m=float(best_decode_params["nms_radius_m"]),
        threshold_kind="confidence",
    )
    return pd.DataFrame.from_records(rows), default_summary


def _run_baseline_threshold_sweep(method_dir: Path, display_name: str, bundles, config: ExperimentConfig) -> tuple[pd.DataFrame, dict]:
    fold = _build_splits(config, bundles)[0]
    threshold_grid = [0.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    nms_grid = [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]
    saved_predictions = _load_predictions_by_plot(method_dir)
    rows: list[dict] = []
    for split_name, plot_ids in [("test", fold.test_plots)]:
        for threshold, nms_radius_m in product(threshold_grid, nms_grid):
            predictions_by_plot = {}
            for plot_id in plot_ids:
                current = saved_predictions.get(plot_id, pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)).copy()
                if "confidence" in current.columns:
                    current = current[current["confidence"] >= float(threshold)].copy()
                current = _greedy_nms(current, radius_m=float(nms_radius_m), score_column="confidence")
                predictions_by_plot[plot_id] = current
            rows.append(
                _summarize_curve_rows(
                    method_name=display_name,
                    split_name=split_name,
                    predictions_by_plot=predictions_by_plot,
                    bundles=bundles,
                    plot_ids=plot_ids,
                    match_radius_m=config.evaluation.match_radius_m,
                    height_tolerance_m=config.evaluation.height_tolerance_m,
                    threshold=float(threshold),
                    nms_radius_m=float(nms_radius_m),
                    threshold_kind="height_m",
                )
            )
    default_summary = _summarize_curve_rows(
        method_name=display_name,
        split_name="test_default",
        predictions_by_plot={plot_id: saved_predictions.get(plot_id, pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)) for plot_id in fold.test_plots},
        bundles=bundles,
        plot_ids=fold.test_plots,
        match_radius_m=config.evaluation.match_radius_m,
        height_tolerance_m=config.evaluation.height_tolerance_m,
        threshold=0.0,
        nms_radius_m=0.0,
        threshold_kind="height_m",
    )
    return pd.DataFrame.from_records(rows), default_summary


def _run_threshold_sweeps(main_method_specs: list[dict], bundles, config: ExperimentConfig, device: str, output_dir: Path) -> pd.DataFrame:
    sweep_rows: list[pd.DataFrame] = []
    default_rows: list[dict] = []
    for spec in main_method_specs:
        if spec["display_name"] == "baseline_local_maxima":
            table, default_summary = _run_baseline_threshold_sweep(spec["dir"], spec["display_name"], bundles, config)
        else:
            table, default_summary = _run_detector_threshold_sweep(spec, bundles, device)
        sweep_rows.append(table)
        default_rows.append(default_summary)
    sweep_table = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame()
    sweep_table.to_csv(output_dir / "threshold_sweep.csv", index=False)

    best_test = summarize_threshold_sweep(sweep_table[sweep_table["split"] == "test"])
    best_test.to_csv(output_dir / "threshold_sweep_best_test.csv", index=False)
    default_table = pd.DataFrame.from_records(default_rows)
    default_table.to_csv(output_dir / "threshold_sweep_defaults.csv", index=False)

    sanity = best_test.merge(
        default_table[["method", "precision_mean", "recall_mean", "f1_mean", "predictions_total", "predictions_mean_per_plot", "zero_prediction_plots"]].rename(
            columns={
                "precision_mean": "default_precision_mean",
                "recall_mean": "default_recall_mean",
                "f1_mean": "default_f1_mean",
                "predictions_total": "default_predictions_total",
                "predictions_mean_per_plot": "default_predictions_mean_per_plot",
                "zero_prediction_plots": "default_zero_prediction_plots",
            }
        ),
        on="method",
        how="left",
    )
    sanity["gain_vs_default_f1"] = sanity["f1_mean"] - sanity["default_f1_mean"]
    sanity.to_csv(output_dir / "learned_sanity_summary.csv", index=False)
    return sanity


def _evaluate_prediction_variant(
    method_name: str,
    variant_name: str,
    predictions_by_plot: dict[str, pd.DataFrame],
    bundles,
    plot_ids: list[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for plot_id in plot_ids:
        match_result = match_tree_predictions(
            predictions_by_plot[plot_id],
            bundles[plot_id].reference,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        row = summarize_matches(match_result)
        row.update({"plot_id": plot_id, "method": method_name, "variant": variant_name})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _run_stage2_oracle(methodology_root: Path, bundles, config: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    prediction_root = methodology_root / "predictions"
    specs = _collect_method_specs(prediction_root)
    rows: list[pd.DataFrame] = []
    for spec in specs:
        manifest = spec["manifest"]
        if manifest.stage2 in {"none", ""}:
            continue
        fold_dir = spec["dir"] / "fold_00"
        test_path = fold_dir / "test_proposals.csv"
        scored_path = fold_dir / "scored_test.csv"
        params_path = fold_dir / "best_params.json"
        if not (test_path.exists() and scored_path.exists() and params_path.exists()):
            continue
        proposal_table = pd.read_csv(test_path)
        scored_table = pd.read_csv(scored_path)
        if proposal_table.empty or scored_table.empty:
            continue
        params_payload = read_json(params_path)
        best_params = params_payload.get("best_params", params_payload)
        merged = scored_table.merge(
            proposal_table[
                ["plot_id", "proposal_id", "x", "y", "height_pred", "objectness", "dx_m", "dy_m", "delta_h_m"]
            ].rename(columns={"x": "proposal_x", "y": "proposal_y", "height_pred": "proposal_height"}),
            on=["plot_id", "proposal_id"],
            how="inner",
        )
        plot_ids = sorted(merged["plot_id"].unique().tolist())
        variants: dict[str, dict[str, pd.DataFrame]] = {}

        raw_predictions = {}
        for plot_id in plot_ids:
            current = merged[merged["plot_id"] == plot_id].copy()
            raw_predictions[plot_id] = apply_stage2_postprocess(
                current,
                objectness_threshold=float(best_params["objectness_threshold"]),
                nms_radius_m=float(best_params["nms_radius_m"]),
                center_mode=str(best_params["center_mode"]),
            )
        variants["raw_stage2"] = raw_predictions

        oracle_positive = merged[merged["objectness"] > 0].copy()
        if oracle_positive.empty:
            continue
        oracle_positive["score"] = 1.0

        oracle_geom = {}
        oracle_refined = {}
        oracle_gt = {}
        for plot_id in plot_ids:
            current = oracle_positive[oracle_positive["plot_id"] == plot_id].copy()
            current["geom_x"] = current["proposal_x"]
            current["geom_y"] = current["proposal_y"]
            current["refined_x"] = current["proposal_x"]
            current["refined_y"] = current["proposal_y"]
            current["refined_height"] = current["proposal_height"]
            oracle_geom[plot_id] = apply_stage2_postprocess(
                current,
                objectness_threshold=0.0,
                nms_radius_m=float(best_params["nms_radius_m"]),
                center_mode="geom",
            )

            refined_current = current.copy()
            oracle_refined[plot_id] = apply_stage2_postprocess(
                refined_current,
                objectness_threshold=0.0,
                nms_radius_m=float(best_params["nms_radius_m"]),
                center_mode=str(best_params["center_mode"]),
            )

            gt_current = current.copy()
            gt_current["refined_x"] = gt_current["proposal_x"] + gt_current["dx_m"]
            gt_current["refined_y"] = gt_current["proposal_y"] + gt_current["dy_m"]
            gt_current["refined_height"] = gt_current["proposal_height"] + gt_current["delta_h_m"]
            oracle_gt[plot_id] = apply_stage2_postprocess(
                gt_current,
                objectness_threshold=0.0,
                nms_radius_m=float(best_params["nms_radius_m"]),
                center_mode="refined",
            )

        variants["oracle_positive_geom"] = oracle_geom
        variants["oracle_positive_refined"] = oracle_refined
        variants["oracle_gt_center"] = oracle_gt

        for variant_name, predictions_by_plot in variants.items():
            rows.append(
                _evaluate_prediction_variant(
                    method_name=spec["display_name"],
                    variant_name=variant_name,
                    predictions_by_plot=predictions_by_plot,
                    bundles=bundles,
                    plot_ids=plot_ids,
                    config=config,
                ).assign(stage1=manifest.stage1, stage2=manifest.stage2)
            )

    oracle_by_plot = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    oracle_by_plot.to_csv(output_dir / "stage2_oracle_by_plot.csv", index=False)
    if oracle_by_plot.empty:
        pd.DataFrame().to_csv(output_dir / "stage2_oracle_summary.csv", index=False)
        return oracle_by_plot

    summary = (
        oracle_by_plot.groupby(["stage1", "stage2", "method", "variant"])[["precision", "recall", "f1", "tp", "fp", "fn"]]
        .mean()
        .reset_index()
        .sort_values(["stage1", "stage2", "f1"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    raw_rows = summary[summary["variant"] == "raw_stage2"][["method", "f1"]].rename(columns={"f1": "raw_stage2_f1"})
    gt_rows = summary[summary["variant"] == "oracle_gt_center"][["method", "f1"]].rename(columns={"f1": "oracle_gt_center_f1"})
    proposal_rows = summary[summary["variant"] == "oracle_positive_geom"][["method", "f1"]].rename(columns={"f1": "oracle_positive_geom_f1"})
    summary = summary.merge(raw_rows, on="method", how="left").merge(gt_rows, on="method", how="left").merge(proposal_rows, on="method", how="left")
    summary["delta_vs_raw_stage2"] = summary["f1"] - summary["raw_stage2_f1"]
    summary["delta_vs_positive_geom"] = summary["f1"] - summary["oracle_positive_geom_f1"]
    summary.to_csv(output_dir / "stage2_oracle_summary.csv", index=False)
    return oracle_by_plot


def _xy_to_pixels(bundle, table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    result = table.copy()
    x_px: list[float] = []
    y_px: list[float] = []
    for _, row in result.iterrows():
        row_px, col_px = bundle.transform.xy_to_rowcol(float(row["x"]), float(row["y"]))
        x_px.append(float(col_px))
        y_px.append(float(row_px))
    result["x_px"] = x_px
    result["y_px"] = y_px
    return result


def _create_qualitative_triptych(
    bundles,
    main_metrics: pd.DataFrame,
    main_specs: list[dict],
    methodology_root: Path,
    error_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    main_clean = _clean_metrics_table(main_metrics)
    main_pivot = main_clean.pivot_table(index="plot_id", columns="method_clean", values="f1")
    learned_methods = [column for column in main_pivot.columns if column != "baseline_local_maxima"]
    if not learned_methods:
        return
    best_learned_name = max(learned_methods, key=lambda name: float(main_clean[main_clean["method_clean"] == name]["f1"].mean()))
    geometry_plot = (main_pivot["baseline_local_maxima"] - main_pivot[best_learned_name]).sort_values(ascending=False).index[0]

    delta_path = methodology_root / "metrics" / "stage2_delta_by_plot.csv"
    if delta_path.exists():
        delta_table = pd.read_csv(delta_path)
        if delta_table.empty:
            stage2_plot = geometry_plot
            stage1_name = "local_maxima"
            stage2_method = "local_maxima+raster"
        else:
            positive_delta = delta_table.sort_values("delta_f1", ascending=False)
            stage2_row = positive_delta.iloc[0]
            stage2_plot = str(stage2_row["plot_id"])
            stage1_name = str(stage2_row["stage1"])
            stage2_method = str(stage2_row["stage2_method"])
    else:
        stage2_plot = geometry_plot
        stage1_name = "local_maxima"
        stage2_method = "local_maxima+raster"

    per_plot_max = main_clean.groupby("plot_id")["f1"].max().sort_values()
    failure_candidates = per_plot_max.index.tolist()
    structural_plot = failure_candidates[0]
    if not error_table.empty:
        overlap = error_table[error_table["failure_mode"] == "neighbor_fusion"]
        if not overlap.empty:
            overlap_counts = overlap.groupby("plot_id").size().sort_values(ascending=False)
            for plot_id in overlap_counts.index:
                if plot_id in failure_candidates[:10]:
                    structural_plot = plot_id
                    break

    def load_predictions(display_name: str, plot_id: str) -> pd.DataFrame:
        for spec in main_specs + _collect_method_specs(methodology_root / "predictions"):
            if spec["display_name"] == display_name:
                return _load_predictions_by_plot(spec["dir"]).get(plot_id, pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS))
        return pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)

    scenarios = [
        {
            "title": f"Geometry Wins: baseline vs {best_learned_name}",
            "plot_id": geometry_plot,
            "methods": [("baseline_local_maxima", "cyan"), (best_learned_name, "red")],
            "file": "qualitative_geometry_wins.png",
        },
        {
            "title": f"Stage-2 Local Gain: {stage1_name} vs {stage2_method}",
            "plot_id": stage2_plot,
            "methods": [(stage1_name, "orange"), (stage2_method, "lime")],
            "file": "qualitative_stage2_gain.png",
        },
        {
            "title": "Structural Failure Case",
            "plot_id": structural_plot,
            "methods": [("baseline_local_maxima", "cyan"), (best_learned_name, "red")],
            "file": "qualitative_structural_failure.png",
        },
    ]

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(14, 6 * len(scenarios)))
    if len(scenarios) == 1:
        axes = np.asarray([axes])
    for row_index, scenario in enumerate(scenarios):
        plot_id = scenario["plot_id"]
        bundle = bundles[plot_id]
        reference = _xy_to_pixels(bundle, bundle.reference)
        rgb = np.moveaxis(bundle.rgb, 0, -1)

        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].scatter(reference["x_px"], reference["y_px"], s=10, c="white", label="GT")
        axes[row_index, 1].imshow(bundle.chm, cmap="viridis")
        axes[row_index, 1].scatter(reference["x_px"], reference["y_px"], s=10, c="white", label="GT")
        for method_name, color in scenario["methods"]:
            predictions = _xy_to_pixels(bundle, load_predictions(method_name, plot_id))
            if predictions.empty:
                continue
            axes[row_index, 0].scatter(predictions["x_px"], predictions["y_px"], s=10, c=color, label=method_name)
            axes[row_index, 1].scatter(predictions["x_px"], predictions["y_px"], s=10, c=color, label=method_name)
        axes[row_index, 0].set_title(f"{scenario['title']} | RGB | {plot_id}")
        axes[row_index, 1].set_title(f"{scenario['title']} | CHM | {plot_id}")
        axes[row_index, 0].legend(loc="upper right", fontsize=8)
        axes[row_index, 1].legend(loc="upper right", fontsize=8)

        save_error_figure(
            bundle.rgb,
            bundle.chm,
            _xy_to_pixels(bundle, load_predictions(scenario["methods"][0][0], plot_id)),
            reference,
            match_tree_predictions(
                load_predictions(scenario["methods"][0][0], plot_id),
                bundle.reference,
                match_radius_m=1.5,
                height_tolerance_m=3.0,
            ),
            output_dir / scenario["file"],
        )

    plt.tight_layout()
    plt.savefig(output_dir / "qualitative_triptych.png", dpi=180)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the methodology analysis suite for NEON-style experiments.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(args=argv)

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = _build_splits(config, bundles)
    fold = splits[0]

    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "methodology")
    metrics_dir = ensure_dir(output_root / "metrics")
    figures_dir = ensure_dir(output_root / "figures")
    diagnostics_dir = ensure_dir(output_root / "diagnostics")

    main_metrics = pd.read_csv(config.resolve_path(config.paths.output_dir) / "metrics" / "metrics_by_plot.csv")
    _run_site_analysis(main_metrics, metrics_dir, diagnostics_dir)
    _compute_site_input_statistics(bundles, diagnostics_dir)
    _compute_label_alignment(config, bundles, fold.test_plots, diagnostics_dir)
    _compute_scale_sanity(config, bundles, fold.test_plots, diagnostics_dir)

    main_specs = _collect_method_specs(config.resolve_path(config.paths.output_dir) / "predictions")
    methodology_specs = _collect_method_specs(output_root / "predictions")
    all_specs = main_specs + methodology_specs
    error_table = _run_error_mode_analysis(all_specs, bundles, config, diagnostics_dir)
    _run_matching_radius_sensitivity(all_specs, bundles, config, diagnostics_dir)
    _run_threshold_sweeps(main_specs, bundles, config, args.device, diagnostics_dir)
    _run_stage2_oracle(output_root, bundles, config, diagnostics_dir)
    _create_qualitative_triptych(bundles, main_metrics, main_specs, output_root, error_table, figures_dir)


if __name__ == "__main__":
    main()

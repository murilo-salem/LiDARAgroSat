from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import laspy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import FaithfulMatchingParams, faithful_greedy_match, summarize_faithful_matches  # noqa: E402
from lidaragrosat.paper_faithful.method import _interpolate_ground_surface, deduplicate_points  # noqa: E402
from lidaragrosat.paper_faithful.strict_pointnet import (  # noqa: E402
    CombinedPointNetLoss,
    LocalPointNet,
    StrictPointNetConfig,
    StrictProposalDataset,
    build_labeled_proposal_table,
    predict_and_refine,
    run_pointnet_epoch,
    score_pointnet_dataset,
    tune_pointnet_postprocess,
)
from lidaragrosat.utils.io import ensure_dir, write_json, write_text  # noqa: E402


VAL_PLOTS = ["plot_01", "plot_06"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_xyz_points_and_reference(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    points_by_plot: dict[str, np.ndarray] = {}
    reference_by_plot: dict[str, pd.DataFrame] = {}
    for npz_path in sorted(cache_dir.glob("plot_*.npz")):
        plot_id = npz_path.stem
        points_by_plot[plot_id] = np.load(npz_path)["points_xyz"].astype(np.float32)
        reference_by_plot[plot_id] = pd.read_csv(cache_dir / f"{plot_id}_reference.csv")
    return points_by_plot, reference_by_plot


def _prepare_multifeature_cache(raw_dir: Path, feature_cache_dir: Path) -> dict[str, np.ndarray]:
    ensure_dir(feature_cache_dir)
    features_by_plot: dict[str, np.ndarray] = {}
    for plot_dir in sorted(raw_dir.glob("plot_*")):
        plot_id = plot_dir.name
        cache_path = feature_cache_dir / f"{plot_id}.npz"
        if cache_path.exists():
            features_by_plot[plot_id] = np.load(cache_path)["points"].astype(np.float32)
            continue
        lidar_candidates = sorted(plot_dir.glob("lidar.la*"))
        if not lidar_candidates:
            raise FileNotFoundError(f"Missing lidar file in {plot_dir}")
        las = laspy.read(lidar_candidates[0])
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        classification = np.asarray(las.classification, dtype=np.int16)
        intensity = np.asarray(las.intensity, dtype=np.float32)
        return_number = np.asarray(las.return_number, dtype=np.float32)
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.float32)

        unique_indices, _ = deduplicate_points(
            x=x,
            y=y,
            z=z,
            classification=classification,
            tolerance_m=0.001,
        )
        x = x[unique_indices]
        y = y[unique_indices]
        z = z[unique_indices]
        classification = classification[unique_indices]
        intensity = intensity[unique_indices]
        return_number = return_number[unique_indices]
        number_of_returns = number_of_returns[unique_indices]

        ground_z = _interpolate_ground_surface(
            x=x,
            y=y,
            z=z,
            classification=classification,
            ground_classification=2,
        )
        z_norm = (z - ground_z).astype(np.float32)
        intensity_scaled = (intensity / max(float(np.max(intensity)), 1.0)).astype(np.float32)

        points = np.column_stack([x, y, z_norm, intensity_scaled, return_number, number_of_returns]).astype(np.float32)
        np.savez_compressed(cache_path, points=points)
        features_by_plot[plot_id] = points
    return features_by_plot


def _build_proposal_table(
    plot_ids: list[str],
    xyz_points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    config: StrictPointNetConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for plot_id in plot_ids:
        rows.append(
            build_labeled_proposal_table(
                plot_id=plot_id,
                points_xyz=xyz_points_by_plot[plot_id],
                reference=reference_by_plot[plot_id],
                config=config,
            )
        )
    return pd.concat(rows, ignore_index=True, sort=False)


def _aggregate_metrics(metrics_by_plot: pd.DataFrame, partition_name: str) -> dict[str, float | str | int]:
    table = metrics_by_plot.copy()
    return {
        "partition": partition_name,
        "n_plots": int(len(table)),
        "tp": int(table["tp"].sum()),
        "fp": int(table["fp"].sum()),
        "fn": int(table["fn"].sum()),
        "precision_mean": float(table["precision"].mean()),
        "recall_mean": float(table["recall"].mean()),
        "f1_mean": float(table["f1"].mean()),
        "distance_m_mean": float(table["distance_m"].mean()),
        "height_error_m_mean": float(table["height_error_m"].mean()),
    }


def _evaluate_scored_variant(
    variant_name: str,
    scored: pd.DataFrame,
    reference_by_plot: dict[str, pd.DataFrame],
    ordered_plots: list[str],
    val_plots: list[str],
    matching_params: FaithfulMatchingParams,
    cls_threshold_grid: tuple[float, ...],
    nms_radius_grid_m: tuple[float, ...],
    variant_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    best_postprocess, val_metrics_by_plot = tune_pointnet_postprocess(
        scored=scored[scored["plot_id"].isin(val_plots)].copy(),
        reference_by_plot={plot_id: reference_by_plot[plot_id] for plot_id in val_plots},
        matching_params=matching_params,
        cls_threshold_grid=cls_threshold_grid,
        nms_radius_grid_m=nms_radius_grid_m,
    )
    write_json(best_postprocess, variant_dir / "best_postprocess.json")
    metrics_rows: list[dict] = []
    predictions_dir = ensure_dir(variant_dir / "predictions")
    for plot_id in ordered_plots:
        current_scored = scored[scored["plot_id"] == plot_id].copy()
        predictions = predict_and_refine(
            scored=current_scored,
            cls_threshold=float(best_postprocess["cls_threshold"]),
            nms_radius=float(best_postprocess["nms_radius_m"]),
        )
        predictions.to_csv(predictions_dir / f"{plot_id}.csv", index=False)
        metrics = summarize_faithful_matches(
            faithful_greedy_match(
                predictions=predictions,
                reference=reference_by_plot[plot_id],
                params=matching_params,
            )
        )
        metrics["plot_id"] = plot_id
        metrics["partition"] = "val" if plot_id in set(val_plots) else "train"
        metrics["n_detections"] = int(len(predictions))
        metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
        metrics_rows.append(metrics)
    metrics_by_plot = pd.DataFrame.from_records(metrics_rows).sort_values("plot_id").reset_index(drop=True)
    metrics_by_plot.to_csv(variant_dir / "metrics_by_plot.csv", index=False)
    summary_by_partition = pd.DataFrame.from_records(
        [
            _aggregate_metrics(metrics_by_plot[metrics_by_plot["partition"] == "train"], "train"),
            _aggregate_metrics(metrics_by_plot[metrics_by_plot["partition"] == "val"], "val"),
            _aggregate_metrics(metrics_by_plot, "all"),
        ]
    )
    summary_by_partition.to_csv(variant_dir / "summary_by_partition.csv", index=False)
    val_summary = summary_by_partition[summary_by_partition["partition"] == "val"].iloc[0].to_dict()
    val_metrics_by_plot.to_csv(variant_dir / "validation_metrics_by_plot.csv", index=False)
    return metrics_by_plot, best_postprocess, summary_by_partition


def _train_and_score_variant(
    variant_name: str,
    config: StrictPointNetConfig,
    points_by_plot: dict[str, np.ndarray],
    proposal_table: pd.DataFrame,
    reference_by_plot: dict[str, pd.DataFrame],
    ordered_plots: list[str],
    val_plots: list[str],
    device: str,
    variant_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    train_plots = [plot_id for plot_id in ordered_plots if plot_id not in set(val_plots)]
    train_table = proposal_table[proposal_table["plot_id"].isin(train_plots)].reset_index(drop=True)
    val_table = proposal_table[proposal_table["plot_id"].isin(val_plots)].reset_index(drop=True)

    train_dataset = StrictProposalDataset(points_by_plot=points_by_plot, proposal_table=train_table, config=config, training=True)
    val_dataset = StrictProposalDataset(points_by_plot=points_by_plot, proposal_table=val_table, config=config, training=False)

    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config.batch_size), shuffle=False, num_workers=4, pin_memory=True)

    input_dim = 3 if config.feature_mode == "xyz" else 7
    model = LocalPointNet(input_dim=input_dim, dropout=float(config.dropout)).to(device)
    criterion = CombinedPointNetLoss(
        pos_weight=float(config.pos_weight),
        alpha=1.0,
        loss_kind=str(config.loss_kind),
        focal_alpha=float(config.focal_alpha),
        focal_gamma=float(config.focal_gamma),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    for epoch in range(int(config.epochs)):
        train_metrics = run_pointnet_epoch(model, train_loader, optimizer, criterion, device=device, train=True)
        val_metrics = run_pointnet_epoch(model, val_loader, optimizer, criterion, device=device, train=False)
        history.append(
            {
                "epoch": epoch + 1,
                **{f"train_{key}": float(value) for key, value in train_metrics.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
        )
        if float(val_metrics["loss_total"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss_total"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(config.patience):
            break
    if best_state is None:
        raise RuntimeError(f"Variant {variant_name} did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    pd.DataFrame.from_records(history).to_csv(variant_dir / "history.csv", index=False)
    write_json({"variant_name": variant_name, "config": asdict(config), "best_val_loss": best_val_loss}, variant_dir / "training_summary.json")

    all_dataset = StrictProposalDataset(points_by_plot=points_by_plot, proposal_table=proposal_table, config=config, training=False)
    scored = score_pointnet_dataset(model, all_dataset, batch_size=int(config.batch_size), device=device)
    scored.to_csv(variant_dir / "scored_proposals_all.csv", index=False)

    return _evaluate_scored_variant(
        variant_name=variant_name,
        scored=scored,
        reference_by_plot=reference_by_plot,
        ordered_plots=ordered_plots,
        val_plots=val_plots,
        matching_params=FaithfulMatchingParams(max_dist_m=5.0, max_height_diff_m=3.0),
        cls_threshold_grid=tuple(float(value) for value in config.cls_threshold_grid),
        nms_radius_grid_m=tuple(float(value) for value in config.nms_radius_grid_m),
        variant_dir=variant_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations for the faithful strict PointNet pipeline.")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--strict-output-dir", default="outputs_faithful_pointnet_strict")
    parser.add_argument("--output-dir", default="outputs_faithful_pointnet_improvements")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    _seed_everything(int(args.seed))
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    cache_dir = Path(args.cache_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    strict_output_dir = Path(args.strict_output_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    xyz_points_by_plot, reference_by_plot = _load_xyz_points_and_reference(cache_dir)
    ordered_plots = sorted(xyz_points_by_plot)
    val_plots = [plot_id for plot_id in VAL_PLOTS if plot_id in ordered_plots]

    base_config = StrictPointNetConfig(
        seed=int(args.seed),
        cls_threshold_grid=tuple(float(x) for x in np.linspace(0.05, 0.95, 91)),
        nms_radius_grid_m=(0.8, 1.0, 1.2, 1.5, 2.0, 2.5),
    )
    proposal_table = _build_proposal_table(ordered_plots, xyz_points_by_plot, reference_by_plot, base_config)
    proposal_table.to_csv(output_dir / "proposal_table_all.csv", index=False)

    variant_summaries: list[dict] = []

    # Variant 1: broader threshold and NMS tuning on the existing strict model
    redecode_dir = ensure_dir(output_dir / "strict_redecode_wide")
    if (redecode_dir / "summary_by_partition.csv").exists() and (redecode_dir / "best_postprocess.json").exists():
        summary = pd.read_csv(redecode_dir / "summary_by_partition.csv")
        best_postprocess = json.loads((redecode_dir / "best_postprocess.json").read_text(encoding="utf-8"))
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    else:
        strict_scored = pd.read_csv(strict_output_dir / "scored_proposals_all.csv")
        _, best_postprocess, summary = _evaluate_scored_variant(
            variant_name="strict_redecode_wide",
            scored=strict_scored,
            reference_by_plot=reference_by_plot,
            ordered_plots=ordered_plots,
            val_plots=val_plots,
            matching_params=FaithfulMatchingParams(max_dist_m=5.0, max_height_diff_m=3.0),
            cls_threshold_grid=base_config.cls_threshold_grid,
            nms_radius_grid_m=base_config.nms_radius_grid_m,
            variant_dir=redecode_dir,
        )
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    variant_summaries.append(
        {
            "variant": "strict_redecode_wide",
            "feature_mode": "xyz",
            "loss_kind": "bce",
            "cls_threshold": float(best_postprocess["cls_threshold"]),
            "nms_radius_m": float(best_postprocess["nms_radius_m"]),
            **val_summary,
        }
    )

    # Variant 2: focal loss on XYZ
    focal_xyz_dir = ensure_dir(output_dir / "strict_focal_xyz")
    focal_xyz_config = StrictPointNetConfig(
        seed=int(args.seed),
        feature_mode="xyz",
        loss_kind="focal",
        focal_alpha=0.75,
        focal_gamma=2.0,
        cls_threshold_grid=base_config.cls_threshold_grid,
        nms_radius_grid_m=base_config.nms_radius_grid_m,
    )
    if (focal_xyz_dir / "summary_by_partition.csv").exists() and (focal_xyz_dir / "best_postprocess.json").exists():
        summary = pd.read_csv(focal_xyz_dir / "summary_by_partition.csv")
        best_postprocess = json.loads((focal_xyz_dir / "best_postprocess.json").read_text(encoding="utf-8"))
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    else:
        _, best_postprocess, summary = _train_and_score_variant(
            variant_name="strict_focal_xyz",
            config=focal_xyz_config,
            points_by_plot=xyz_points_by_plot,
            proposal_table=proposal_table,
            reference_by_plot=reference_by_plot,
            ordered_plots=ordered_plots,
            val_plots=val_plots,
            device=device,
            variant_dir=focal_xyz_dir,
        )
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    variant_summaries.append(
        {
            "variant": "strict_focal_xyz",
            "feature_mode": "xyz",
            "loss_kind": "focal",
            "cls_threshold": float(best_postprocess["cls_threshold"]),
            "nms_radius_m": float(best_postprocess["nms_radius_m"]),
            **val_summary,
        }
    )

    # Prepare multifeature point cache once
    multifeature_points_by_plot = _prepare_multifeature_cache(raw_dir=raw_dir, feature_cache_dir=output_dir / "feature_cache")

    # Variant 3: multifeature with BCE
    multifeature_bce_dir = ensure_dir(output_dir / "strict_multifeature_bce")
    multifeature_bce_config = StrictPointNetConfig(
        seed=int(args.seed),
        feature_mode="multifeature",
        loss_kind="bce",
        cls_threshold_grid=base_config.cls_threshold_grid,
        nms_radius_grid_m=base_config.nms_radius_grid_m,
    )
    if (multifeature_bce_dir / "summary_by_partition.csv").exists() and (multifeature_bce_dir / "best_postprocess.json").exists():
        summary = pd.read_csv(multifeature_bce_dir / "summary_by_partition.csv")
        best_postprocess = json.loads((multifeature_bce_dir / "best_postprocess.json").read_text(encoding="utf-8"))
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    else:
        _, best_postprocess, summary = _train_and_score_variant(
            variant_name="strict_multifeature_bce",
            config=multifeature_bce_config,
            points_by_plot=multifeature_points_by_plot,
            proposal_table=proposal_table,
            reference_by_plot=reference_by_plot,
            ordered_plots=ordered_plots,
            val_plots=val_plots,
            device=device,
            variant_dir=multifeature_bce_dir,
        )
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    variant_summaries.append(
        {
            "variant": "strict_multifeature_bce",
            "feature_mode": "multifeature",
            "loss_kind": "bce",
            "cls_threshold": float(best_postprocess["cls_threshold"]),
            "nms_radius_m": float(best_postprocess["nms_radius_m"]),
            **val_summary,
        }
    )

    # Variant 4: multifeature with focal loss
    focal_multifeature_dir = ensure_dir(output_dir / "strict_multifeature_focal")
    focal_multifeature_config = StrictPointNetConfig(
        seed=int(args.seed),
        feature_mode="multifeature",
        loss_kind="focal",
        focal_alpha=0.75,
        focal_gamma=2.0,
        cls_threshold_grid=base_config.cls_threshold_grid,
        nms_radius_grid_m=base_config.nms_radius_grid_m,
    )
    if (focal_multifeature_dir / "summary_by_partition.csv").exists() and (focal_multifeature_dir / "best_postprocess.json").exists():
        summary = pd.read_csv(focal_multifeature_dir / "summary_by_partition.csv")
        best_postprocess = json.loads((focal_multifeature_dir / "best_postprocess.json").read_text(encoding="utf-8"))
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    else:
        _, best_postprocess, summary = _train_and_score_variant(
            variant_name="strict_multifeature_focal",
            config=focal_multifeature_config,
            points_by_plot=multifeature_points_by_plot,
            proposal_table=proposal_table,
            reference_by_plot=reference_by_plot,
            ordered_plots=ordered_plots,
            val_plots=val_plots,
            device=device,
            variant_dir=focal_multifeature_dir,
        )
        val_summary = summary[summary["partition"] == "val"].iloc[0].to_dict()
    variant_summaries.append(
        {
            "variant": "strict_multifeature_focal",
            "feature_mode": "multifeature",
            "loss_kind": "focal",
            "cls_threshold": float(best_postprocess["cls_threshold"]),
            "nms_radius_m": float(best_postprocess["nms_radius_m"]),
            **val_summary,
        }
    )

    summary_table = pd.DataFrame.from_records(variant_summaries).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    summary_table.to_csv(output_dir / "variant_summary.csv", index=False)

    current_strict = pd.read_csv(strict_output_dir / "summary_by_partition.csv")
    current_strict_val = current_strict[current_strict["partition"] == "val"].iloc[0].to_dict()
    baseline_best = pd.read_csv(strict_output_dir / "baseline_comparison_val.csv").sort_values("f1_mean", ascending=False).iloc[0].to_dict()

    comparison_rows: list[dict] = []
    for _, row in summary_table.iterrows():
        comparison_rows.append(
            {
                "variant": row["variant"],
                "variant_f1_mean": float(row["f1_mean"]),
                "variant_precision_mean": float(row["precision_mean"]),
                "variant_recall_mean": float(row["recall_mean"]),
                "delta_vs_strict_baseline_f1": float(row["f1_mean"] - current_strict_val["f1_mean"]),
                "delta_vs_best_overall_val_f1": float(row["f1_mean"] - baseline_best["f1_mean"]),
                "best_overall_method": str(baseline_best["method"]),
                "best_overall_f1_mean": float(baseline_best["f1_mean"]),
            }
        )
    comparison_table = pd.DataFrame.from_records(comparison_rows).sort_values("variant_f1_mean", ascending=False).reset_index(drop=True)
    comparison_table.to_csv(output_dir / "comparison_vs_best.csv", index=False)

    report_lines = [
        "# Faithful PointNet Improvements",
        "",
        "## Variants",
        "- strict_redecode_wide: modelo fiel atual, apenas threshold/NMS ampliados",
        "- strict_focal_xyz: Focal Loss sobre XYZ",
        "- strict_multifeature_bce: entrada 7-D com intensity e return ratio, mantendo BCE",
        "- strict_multifeature_focal: entrada 7-D com Focal Loss",
        "",
        "## Melhor baseline de referencia na validacao",
        f"- metodo: {baseline_best['method']}",
        f"- F1 mean: {baseline_best['f1_mean']:.4f}",
        "",
        "## Melhor variante desta ablação",
        f"- variante: {summary_table.iloc[0]['variant']}",
        f"- Precision mean: {summary_table.iloc[0]['precision_mean']:.4f}",
        f"- Recall mean: {summary_table.iloc[0]['recall_mean']:.4f}",
        f"- F1 mean: {summary_table.iloc[0]['f1_mean']:.4f}",
        f"- delta vs strict baseline: {summary_table.iloc[0]['f1_mean'] - current_strict_val['f1_mean']:+.4f}",
        f"- delta vs best overall val: {summary_table.iloc[0]['f1_mean'] - baseline_best['f1_mean']:+.4f}",
    ]
    write_text("\n".join(report_lines) + "\n", output_dir / "REPORT.md")
    write_json(
        {
            "seed": int(args.seed),
            "device": device,
            "val_plots": val_plots,
            "strict_baseline_val": current_strict_val,
            "best_overall_val": baseline_best,
            "best_variant": summary_table.iloc[0].to_dict(),
        },
        output_dir / "run_summary.json",
    )


if __name__ == "__main__":
    main()

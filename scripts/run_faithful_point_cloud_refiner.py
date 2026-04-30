from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.paper_faithful import FaithfulMatchingParams  # noqa: E402
from lidaragrosat.paper_faithful.point_cloud_refiner import (  # noqa: E402
    PointCloudRefinerConfig,
    PointRefinerLoss,
    PointSetRefiner,
    ProposalPointDataset,
    label_proposals,
    load_proposals_for_plot,
    postprocess_predictions,
    run_epoch,
    score_dataset,
    tune_postprocess,
)
from lidaragrosat.utils.io import ensure_dir, write_json  # noqa: E402


SPLITS = [
    {"name": "split_plot10", "test_plot": "plot_10", "val_plot": "plot_09"},
    {"name": "split_plot07", "test_plot": "plot_07", "val_plot": "plot_08"},
    {"name": "split_plot06", "test_plot": "plot_06", "val_plot": "plot_05"},
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_points_and_reference(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    points_by_plot: dict[str, np.ndarray] = {}
    reference_by_plot: dict[str, pd.DataFrame] = {}
    for npz_path in sorted(cache_dir.glob("plot_*.npz")):
        plot_id = npz_path.stem
        points_by_plot[plot_id] = np.load(npz_path)["points_xyz"].astype(np.float64)
        reference_by_plot[plot_id] = pd.read_csv(cache_dir / f"{plot_id}_reference.csv")
    return points_by_plot, reference_by_plot


def _build_candidate_table(
    plot_ids: list[str],
    source_mode: str,
    adaptive_root: Path,
    watershed_root: Path,
    reference_by_plot: dict[str, pd.DataFrame],
    config: PointCloudRefinerConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for plot_id in plot_ids:
        proposals = load_proposals_for_plot(
            plot_id=plot_id,
            source_mode=source_mode,
            adaptive_root=adaptive_root,
            watershed_root=watershed_root,
            dedup_radius_m=0.8,
        )
        proposals["plot_id"] = plot_id
        labeled = label_proposals(
            proposals=proposals,
            reference=reference_by_plot[plot_id],
            positive_match_radius_m=config.positive_match_radius_m,
            max_height_diff_m=config.max_height_diff_m,
        )
        rows.append(labeled)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _subsample_training_table(candidate_table: pd.DataFrame, negative_ratio: float = 2.0) -> pd.DataFrame:
    positives = candidate_table[candidate_table["is_positive"] == 1]
    negatives = candidate_table[candidate_table["is_positive"] == 0]
    hard_negatives = negatives[negatives["is_hard_negative"] == 1]
    easy_negatives = negatives[negatives["is_hard_negative"] == 0]
    n_positive = max(1, len(positives))
    n_hard = min(len(hard_negatives), int(round(n_positive * negative_ratio)))
    n_easy = min(len(easy_negatives), int(round(n_positive * negative_ratio)))
    sampled = [
        positives,
        hard_negatives.sample(n=n_hard, random_state=13) if n_hard > 0 else hard_negatives.iloc[:0],
        easy_negatives.sample(n=n_easy, random_state=17) if n_easy > 0 else easy_negatives.iloc[:0],
    ]
    return pd.concat(sampled, ignore_index=True, sort=False).sample(frac=1.0, random_state=19).reset_index(drop=True)


def _train_single_split(
    split: dict,
    source_mode: str,
    points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    adaptive_root: Path,
    watershed_root: Path,
    output_dir: Path,
    device: str,
    seed: int,
) -> dict:
    ordered_plots = sorted(points_by_plot)
    test_plot = split["test_plot"]
    val_plot = split["val_plot"]
    train_plots = [plot_id for plot_id in ordered_plots if plot_id not in {test_plot, val_plot}]
    config = PointCloudRefinerConfig(source_mode=source_mode)

    train_table = _build_candidate_table(train_plots, source_mode, adaptive_root, watershed_root, reference_by_plot, config)
    train_table = _subsample_training_table(train_table, negative_ratio=2.0)
    val_table = _build_candidate_table([val_plot], source_mode, adaptive_root, watershed_root, reference_by_plot, config)
    test_table = _build_candidate_table([test_plot], source_mode, adaptive_root, watershed_root, reference_by_plot, config)

    train_dataset = ProposalPointDataset(points_by_plot, train_table, config=config, training=True)
    val_dataset = ProposalPointDataset(points_by_plot, val_table, config=config, training=False)
    test_dataset = ProposalPointDataset(points_by_plot, test_table, config=config, training=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = PointSetRefiner().to(device)
    criterion = PointRefinerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device=device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device=device, train=False)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)

    matching_params = FaithfulMatchingParams(max_dist_m=5.0, max_height_diff_m=3.0)
    val_scores = score_dataset(model, val_dataset, batch_size=config.batch_size, device=device)
    best_postprocess, val_metrics = tune_postprocess(
        predictions=val_scores,
        reference=reference_by_plot[val_plot],
        threshold_grid=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        nms_radius_grid_m=[0.8, 1.0, 1.2, 1.5, 1.8],
        min_height_grid_m=[0.0, 1.5, 2.0, 2.5],
        matching_params=matching_params,
    )

    test_scores = score_dataset(model, test_dataset, batch_size=config.batch_size, device=device)
    test_predictions = postprocess_predictions(
        predictions=test_scores,
        classification_threshold=best_postprocess["classification_threshold"],
        nms_radius_m=best_postprocess["nms_radius_m"],
        min_height_m=best_postprocess["min_height_m"],
    )
    test_metrics = tune_postprocess(
        predictions=test_scores,
        reference=reference_by_plot[test_plot],
        threshold_grid=[best_postprocess["classification_threshold"]],
        nms_radius_grid_m=[best_postprocess["nms_radius_m"]],
        min_height_grid_m=[best_postprocess["min_height_m"]],
        matching_params=matching_params,
    )[1]

    split_dir = ensure_dir(output_dir / source_mode / split["name"])
    pd.DataFrame(history).to_csv(split_dir / "history.csv", index=False)
    val_scores.to_csv(split_dir / "val_scores.csv", index=False)
    test_scores.to_csv(split_dir / "test_scores.csv", index=False)
    test_predictions.to_csv(split_dir / "test_predictions.csv", index=False)
    write_json({"split": split, "source_mode": source_mode, "seed": seed, "model_config": asdict(config)}, split_dir / "config.json")
    write_json({"best_postprocess": best_postprocess, "val_metrics": val_metrics, "test_metrics": test_metrics}, split_dir / "metrics.json")

    return {
        "source_mode": source_mode,
        "split_name": split["name"],
        "test_plot": test_plot,
        "val_plot": val_plot,
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "distance_m": float(test_metrics["distance_m"]),
        "height_error_m": float(test_metrics["height_error_m"]),
    }


def _baseline_metrics_for_plots(test_plots: list[str]) -> pd.DataFrame:
    sources = {
        "fixed": Path("outputs_faithful_paper_optuna/best_per_plot.csv"),
        "adaptive": Path("outputs_faithful_paper_adaptive_optuna/best_per_plot.csv"),
        "watershed": Path("outputs_faithful_paper_watershed_optuna/best_per_plot.csv"),
        "object_fusion": Path("outputs_faithful_paper_object_fusion_optuna/best_per_plot.csv"),
    }
    rows: list[dict] = []
    for name, path in sources.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table = table[table["plot_id"].isin(test_plots)].copy()
        rows.append(
            {
                "method": name,
                "precision_mean": float(table["precision"].mean()),
                "recall_mean": float(table["recall"].mean()),
                "f1_mean": float(table["f1"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local PointNet-style proposal refiner on the faithful benchmark.")
    parser.add_argument("--cache-dir", default="outputs_faithful_paper/cache")
    parser.add_argument("--adaptive-root", default="outputs_faithful_paper_adaptive_optuna/best_predictions")
    parser.add_argument("--watershed-root", default="outputs_faithful_paper_watershed_optuna/best_predictions")
    parser.add_argument("--output-dir", default="outputs_faithful_point_cloud_refiner")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    _seed_everything(int(args.seed))
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir).resolve()
    adaptive_root = Path(args.adaptive_root).resolve()
    watershed_root = Path(args.watershed_root).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    points_by_plot, reference_by_plot = _load_points_and_reference(cache_dir)
    all_results: list[dict] = []
    for source_mode in ["adaptive", "union"]:
        for split in SPLITS:
            all_results.append(
                _train_single_split(
                    split=split,
                    source_mode=source_mode,
                    points_by_plot=points_by_plot,
                    reference_by_plot=reference_by_plot,
                    adaptive_root=adaptive_root,
                    watershed_root=watershed_root,
                    output_dir=output_dir,
                    device=device,
                    seed=int(args.seed),
                )
            )

    metrics_table = pd.DataFrame.from_records(all_results).sort_values(["source_mode", "split_name"]).reset_index(drop=True)
    metrics_table.to_csv(output_dir / "metrics_by_split.csv", index=False)
    summary = (
        metrics_table.groupby("source_mode", as_index=False)[["precision", "recall", "f1", "distance_m", "height_error_m"]]
        .mean()
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary_by_source_mode.csv", index=False)

    test_plots = [split["test_plot"] for split in SPLITS]
    baseline_summary = _baseline_metrics_for_plots(test_plots)
    baseline_summary.to_csv(output_dir / "baseline_summary_on_test_plots.csv", index=False)

    best_mode = str(summary.iloc[0]["source_mode"])
    best_table = metrics_table[metrics_table["source_mode"] == best_mode].copy()
    comparison_rows: list[dict] = []
    for _, baseline_row in baseline_summary.iterrows():
        comparison_rows.append(
            {
                "best_point_mode": best_mode,
                "baseline_method": baseline_row["method"],
                "point_precision_mean": float(best_table["precision"].mean()),
                "point_recall_mean": float(best_table["recall"].mean()),
                "point_f1_mean": float(best_table["f1"].mean()),
                "baseline_precision_mean": float(baseline_row["precision_mean"]),
                "baseline_recall_mean": float(baseline_row["recall_mean"]),
                "baseline_f1_mean": float(baseline_row["f1_mean"]),
                "delta_f1_mean": float(best_table["f1"].mean() - baseline_row["f1_mean"]),
            }
        )
    pd.DataFrame.from_records(comparison_rows).to_csv(output_dir / "comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()

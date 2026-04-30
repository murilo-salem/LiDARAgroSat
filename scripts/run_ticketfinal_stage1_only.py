from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_A_ID, build_protocol_a_splits
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.utils.io import ensure_dir
from run_ticketfinal_matrix import _run_stage1_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage-1-only official comparisons for TicketFinal.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--stage1", nargs="+", default=["local_maxima", "watershed_fixo", "adaptive_watershed"])
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = build_protocol_a_splits(list(bundles.keys()))
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir))
    metrics_dir = ensure_dir(output_root / "metrics")

    rows = []
    for stage1_name in args.stage1:
        outputs = _run_stage1_cache(stage1_name, bundles)
        for split in splits:
            for plot_id in split.test_plots:
                predictions = outputs[plot_id].candidates
                matches = match_tree_predictions(
                    predictions=predictions,
                    reference=bundles[plot_id].reference,
                    match_radius_m=config.evaluation.match_radius_m,
                    height_tolerance_m=config.evaluation.height_tolerance_m,
                )
                row = summarize_matches(matches)
                row["method"] = stage1_name
                row["plot_id"] = plot_id
                row["fold_id"] = split.fold_id
                row["protocol_id"] = PROTOCOL_A_ID
                row["stage1"] = stage1_name
                row["stage2"] = "none"
                row["labels"] = "field_center"
                rows.append(row)
    metrics = pd.DataFrame.from_records(rows)
    metrics.to_csv(metrics_dir / "stage1_only_metrics_by_plot.csv", index=False)
    summary = build_method_summary(metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    summary.to_csv(metrics_dir / "stage1_only_summary.csv", index=False)


if __name__ == "__main__":
    main()

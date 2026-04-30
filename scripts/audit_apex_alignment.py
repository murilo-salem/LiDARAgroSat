from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.auxiliary import audit_reference_to_apex_offsets
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit annotation center-to-apex offsets on processed plots.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--snap-radius-m", type=float, default=1.0)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    output_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "analysis" / "apex_alignment")

    records = []
    for plot_id, bundle in bundles.items():
        table = audit_reference_to_apex_offsets(
            reference_table=bundle.reference,
            transform=bundle.transform,
            chm=bundle.chm,
            snap_radius_m=args.snap_radius_m,
        )
        table["plot_id"] = plot_id
        records.append(table)

    offsets = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    offsets.to_csv(output_dir / "apex_offsets.csv", index=False)

    if offsets.empty:
        summary = pd.DataFrame(columns=["group", "count", "offset_mean_m", "offset_median_m", "offset_p90_m"])
    else:
        by_plot = (
            offsets.groupby("plot_id")["offset_m"]
            .agg(["count", "mean", "median", lambda s: s.quantile(0.9)])
            .reset_index()
            .rename(columns={"plot_id": "group", "mean": "offset_mean_m", "median": "offset_median_m", "<lambda_0>": "offset_p90_m"})
        )
        summary_rows = [by_plot]
        if "species" in offsets.columns:
            by_species = (
                offsets.groupby("species")["offset_m"]
                .agg(["count", "mean", "median", lambda s: s.quantile(0.9)])
                .reset_index()
                .rename(columns={"species": "group", "mean": "offset_mean_m", "median": "offset_median_m", "<lambda_0>": "offset_p90_m"})
            )
            summary_rows.append(by_species)
        summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(output_dir / "apex_offset_summary.csv", index=False)


if __name__ == "__main__":
    main()

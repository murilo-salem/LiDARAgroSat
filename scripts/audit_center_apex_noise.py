from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.auxiliary import audit_reference_to_apex_offsets, nearest_neighbor_distances
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.utils.io import ensure_dir, write_json


def _height_band(height: float) -> str:
    if pd.isna(height):
        return "unknown"
    if height < 10:
        return "<10"
    if height < 20:
        return "10-20"
    if height < 30:
        return "20-30"
    return ">=30"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible center-to-apex noise audit.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--snap-radius-m", type=float, default=1.0)
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    output_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "analysis" / "center_apex_audit")

    rows: list[pd.DataFrame] = []
    for plot_id, bundle in bundles.items():
        table = audit_reference_to_apex_offsets(
            reference_table=bundle.reference,
            transform=bundle.transform,
            chm=bundle.chm,
            snap_radius_m=args.snap_radius_m,
        )
        nn_distance = nearest_neighbor_distances(bundle.reference)
        if len(table):
            table["plot_id"] = plot_id
            table["height_band"] = table["height"].apply(_height_band)
            table["local_density_band"] = pd.Series(nn_distance).apply(lambda value: "dense" if pd.notna(value) and value < 3.0 else "sparse")
            table["local_structure"] = pd.Series(nn_distance).apply(lambda value: "occluded_neighbor" if pd.notna(value) and value < 2.0 else "isolated")
        rows.append(table)

    audit = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    audit.to_csv(output_dir / "center_apex_audit.csv", index=False)

    summary = {
        "count": int(len(audit)),
        "offset_mean_m": float(audit["offset_m"].mean()) if len(audit) else 0.0,
        "offset_median_m": float(audit["offset_m"].median()) if len(audit) else 0.0,
        "offset_p90_m": float(audit["offset_m"].quantile(0.9)) if len(audit) else 0.0,
    }
    write_json(summary, output_dir / "center_apex_summary.json")

    if len(audit):
        for column_name in ["plot_id", "species", "height_band", "local_density_band"]:
            audit.groupby(column_name)["offset_m"].agg(["count", "mean", "median"]).reset_index().to_csv(
                output_dir / f"{column_name}_summary.csv",
                index=False,
            )
        plt.figure(figsize=(6, 4))
        plt.hist(audit["offset_m"].to_numpy(dtype=float), bins=20)
        plt.xlabel("center-to-apex offset (m)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / "offset_histogram.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    main()

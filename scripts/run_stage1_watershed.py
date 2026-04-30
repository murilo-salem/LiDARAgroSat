from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import laspy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_B_ID, require_protocol_id
from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.experiment_manifest import create_experiment_manifest, save_experiment_manifest
from lidaragrosat.paper_faithful.watershed import FaithfulWatershedParams, chm_watershed_details
from lidaragrosat.stage1.generators import AdaptiveWatershedStage1Generator, WatershedStage1Generator
from lidaragrosat.utils.io import ensure_dir, write_json


def _load_raw_points(bundle) -> np.ndarray:
    lidar_path = (bundle.metadata or {}).get("lidar_path")
    if not lidar_path:
        raise ValueError(f"Bundle {bundle.plot_id} does not expose metadata.lidar_path.")
    las = laspy.read(Path(lidar_path))
    return np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical Stage-1 watershed or adaptive watershed.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--method", choices=["watershed", "adaptive_watershed"], default="watershed")
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    if not config.benchmark.protocol_id:
        config.benchmark.protocol_id = PROTOCOL_B_ID
    require_protocol_id(config.benchmark.protocol_id)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)

    faithful_params = FaithfulWatershedParams(
        resolution_m=0.1,
        gaussian_sigma_m=0.2,
        min_distance_m=0.8,
        min_height_m=5.9,
        min_segment_area_m2=6.0,
        hole_fill_radius_m=0.3,
        compactness=0.0075,
    )
    generator = AdaptiveWatershedStage1Generator() if args.method == "adaptive_watershed" else WatershedStage1Generator()
    config.benchmark.stage1 = args.method
    config.benchmark.stage2 = "none"
    config.benchmark.family = config.benchmark.family or "stage1"
    manifest = create_experiment_manifest(
        config,
        split={"plot_ids": sorted(bundles)},
        stage1_config=asdict(generator),
        refiner="none",
        cwd=Path(__file__).resolve().parents[1],
    )

    output_root = ensure_dir(config.resolve_path(config.paths.output_dir) / "predictions" / manifest.experiment_name)
    save_config(config, output_root / "config.yaml")
    save_experiment_manifest(manifest, output_root / "experiment_manifest.json")

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    segment_rows: list[pd.DataFrame] = []
    for plot_id, bundle in bundles.items():
        if args.method == "watershed":
            details = chm_watershed_details(_load_raw_points(bundle), params=faithful_params)
            output_candidates = details.predictions.copy()
            if not output_candidates.empty:
                output_candidates["source"] = "faithful_watershed"
            output_candidates.to_csv(output_root / f"{plot_id}.csv", index=False)
            np.save(output_root / f"{plot_id}_segments.npy", details.segments.astype(np.int32))
            rows = []
            for segment_id in np.unique(details.segments):
                if int(segment_id) <= 0:
                    continue
                mask = details.segments == int(segment_id)
                rr, cc = np.where(mask)
                if len(rr) == 0:
                    continue
                area_m2 = float(mask.sum() * (details.resolution_m**2))
                max_idx = int(np.argmax(np.where(mask, details.chm, -np.inf)))
                max_row, max_col = np.unravel_index(max_idx, details.chm.shape)
                x = float(details.xmin + (max_col + 0.5) * details.resolution_m)
                y = float(details.ymax - (max_row + 0.5) * details.resolution_m)
                rows.append(
                    {
                        "plot_id": plot_id,
                        "segment_id": int(segment_id),
                        "x": x,
                        "y": y,
                        "area_m2": area_m2,
                        "height_max_m": float(np.nanmax(details.chm[mask])),
                        "height_mean_m": float(np.nanmean(details.chm[mask])),
                        "source": "faithful_watershed",
                    }
                )
            if rows:
                segment_rows.append(pd.DataFrame.from_records(rows))
            diagnostic_rows.append(
                {
                    "plot_id": plot_id,
                    "correct_segments": np.nan,
                    "under_segmented": np.nan,
                    "over_segmented": np.nan,
                    "mixed_segment": np.nan,
                    "empty_segment": np.nan,
                    "generator": "faithful_watershed",
                    "resolution_m": faithful_params.resolution_m,
                    "gaussian_sigma_m": faithful_params.gaussian_sigma_m,
                    "min_distance_m": faithful_params.min_distance_m,
                    "min_segment_area_m2": faithful_params.min_segment_area_m2,
                    "min_height_m": faithful_params.min_height_m,
                    "hole_fill_radius_m": faithful_params.hole_fill_radius_m,
                    "compactness": faithful_params.compactness,
                }
            )
            matches = match_tree_predictions(
                output_candidates,
                bundle.reference,
                match_radius_m=config.evaluation.match_radius_m,
                height_tolerance_m=config.evaluation.height_tolerance_m,
            )
            row = summarize_matches(matches)
            row["plot_id"] = plot_id
            row["method"] = args.method
            row["protocol_id"] = manifest.protocol_id
            metric_rows.append(row)
            continue

        output = generator.generate(bundle)
        output.candidates.to_csv(output_root / f"{plot_id}.csv", index=False)
        if output.segment_labels is not None:
            np.save(output_root / f"{plot_id}_segments.npy", output.segment_labels.astype(np.int32))
        if not output.segment_table.empty:
            segment_rows.append(output.segment_table.assign(plot_id=plot_id))
        diagnostic_rows.append({"plot_id": plot_id, **output.diagnostics.to_dict(), **output.metadata})
        matches = match_tree_predictions(
            output.candidates,
            bundle.reference,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        row = summarize_matches(matches)
        row["plot_id"] = plot_id
        row["method"] = args.method
        row["protocol_id"] = manifest.protocol_id
        metric_rows.append(row)

    metrics_by_plot = pd.DataFrame.from_records(metric_rows).sort_values("plot_id").reset_index(drop=True)
    diagnostics = pd.DataFrame.from_records(diagnostic_rows).sort_values("plot_id").reset_index(drop=True)
    summary = build_method_summary(metrics_by_plot, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    metrics_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "metrics")
    metrics_by_plot.to_csv(metrics_dir / f"{args.method}_metrics_by_plot.csv", index=False)
    summary.to_csv(metrics_dir / f"{args.method}_summary.csv", index=False)
    diagnostics.to_csv(metrics_dir / "watershed_diagnostics.csv", index=False)
    if segment_rows:
        pd.concat(segment_rows, ignore_index=True).to_csv(metrics_dir / f"{args.method}_segments.csv", index=False)
    write_json(
        {
            "method": args.method,
            "protocol_id": manifest.protocol_id,
            "n_plots": int(len(metrics_by_plot)),
            "f1_mean": float(metrics_by_plot["f1"].mean()) if len(metrics_by_plot) else 0.0,
        },
        metrics_dir / f"{args.method}_report.json",
    )


if __name__ == "__main__":
    main()

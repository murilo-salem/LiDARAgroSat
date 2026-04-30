from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidaragrosat.benchmarking import build_protocol_a_splits
from lidaragrosat.config import ExperimentConfig
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.utils.io import ensure_dir
from run_ticketfinal_matrix import _run_stage1_cache, _run_stage2_family


def _best_label(output_dir: Path) -> str:
    main_path = output_dir / "metrics" / "main_matrix_results.csv"
    if not main_path.exists():
        return "field_center"
    table = pd.read_csv(main_path)
    if "labels" not in table.columns or table.empty:
        return "field_center"
    return str(table.sort_values("f1_mean", ascending=False).iloc[0]["labels"])


def _family_for_ablation(base_family: str, ablation_kind: str, ablation_value: str) -> str:
    safe_value = ablation_value.replace("+", "_").replace("-", "_")
    return f"{base_family}_{ablation_kind}_{safe_value}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RGB and CHM contribution ablations for TicketFinal.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = build_protocol_a_splits(list(bundles.keys()))
    output_root = ensure_dir(config.resolve_path(config.paths.output_dir))
    label_policy = _best_label(output_root)
    stage1_outputs = _run_stage1_cache("watershed_fixo", bundles)
    base_family = config.benchmark.family or "ticketfinal"

    rgb_rows: list[pd.DataFrame] = []
    for model_kind in ["raster", "hybrid"]:
        for input_name, use_rgb, use_lidar, use_chm in [
            ("lidar_chm", False, True, True),
            ("rgb_chm", True, False, True),
            ("rgb_lidar_chm", True, True, True),
        ]:
            config_run = copy.deepcopy(config)
            config_run.benchmark.family = _family_for_ablation(base_family, "rgb", f"{model_kind}_{input_name}")
            metrics, _ = _run_stage2_family(
                config=config_run,
                bundles=bundles,
                splits=splits,
                stage1_outputs=stage1_outputs,
                stage1_name="watershed_fixo",
                label_policy=label_policy,
                model_kind=model_kind,
                output_root=output_root,
                device=args.device,
                use_rgb=use_rgb,
                use_lidar=use_lidar,
                use_chm=use_chm,
                include_geom_features=True,
            )
            summary = metrics.groupby("method")[["precision", "recall", "f1", "location_mae_m", "height_mae_m"]].mean().reset_index()
            summary["model_kind"] = model_kind
            summary["input_mode"] = input_name
            rgb_rows.append(summary)

    rgb_table = pd.concat(rgb_rows, ignore_index=True)
    rgb_table.to_csv(output_root / "metrics" / "rgb_contribution_ablation.csv", index=False)

    chm_rows: list[pd.DataFrame] = []
    for model_kind in ["raster", "hybrid"]:
        for input_name, use_chm, include_geom in [
            ("no_chm", False, False),
            ("with_chm", True, False),
            ("with_chm_geom", True, True),
        ]:
            config_run = copy.deepcopy(config)
            config_run.benchmark.family = _family_for_ablation(base_family, "chm", f"{model_kind}_{input_name}")
            metrics, _ = _run_stage2_family(
                config=config_run,
                bundles=bundles,
                splits=splits,
                stage1_outputs=stage1_outputs,
                stage1_name="watershed_fixo",
                label_policy=label_policy,
                model_kind=model_kind,
                output_root=output_root,
                device=args.device,
                use_rgb=True,
                use_lidar=True,
                use_chm=use_chm,
                include_geom_features=include_geom,
            )
            summary = metrics.groupby("method")[["precision", "recall", "f1", "location_mae_m", "height_mae_m"]].mean().reset_index()
            summary["model_kind"] = model_kind
            summary["chm_mode"] = input_name
            chm_rows.append(summary)

    chm_table = pd.concat(chm_rows, ignore_index=True)
    chm_table.to_csv(output_root / "metrics" / "chm_contribution_ablation.csv", index=False)


if __name__ == "__main__":
    main()

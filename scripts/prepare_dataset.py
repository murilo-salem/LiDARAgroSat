from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.inventory import discover_dataset, save_inventory
from lidaragrosat.data.preprocess import prepare_plot_bundle, save_prepared_plot
from lidaragrosat.data.splits import build_basic_split, build_lopo_splits
from lidaragrosat.data.tiles import build_tiles
from lidaragrosat.utils.io import ensure_dir, write_json
from lidaragrosat.utils.logging import configure_logging, get_logger
from lidaragrosat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LiDARAgroSat dataset.")
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML config files.")
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    seed_everything(config.training.seed)
    configure_logging(Path(config.paths.output_dir) / "logs", "prepare_dataset")
    logger = get_logger("prepare_dataset")

    raw_dir = config.resolve_path(config.paths.raw_dir)
    processed_dir = ensure_dir(config.resolve_path(config.paths.processed_dir))
    splits_dir = ensure_dir(config.resolve_path(config.paths.splits_dir))
    experiments_dir = ensure_dir(config.resolve_path(config.paths.experiments_dir))
    save_config(config, experiments_dir / "prepare_dataset_config.yaml")

    inventories = discover_dataset(raw_dir, explicit_plot_ids=config.data.plot_ids or None)
    save_inventory(inventories, experiments_dir / "inventory.json")

    qa_rows: list[dict] = []
    for inventory in inventories:
        logger.info("Preparing plot %s", inventory.plot_id)
        bundle, qa_report, artifacts = prepare_plot_bundle(inventory, config)
        plot_dir = processed_dir / inventory.plot_id
        save_prepared_plot(bundle, artifacts, plot_dir)

        tile_records = build_tiles(
            plot_id=bundle.plot_id,
            raster_height=bundle.raster_shape[0],
            raster_width=bundle.raster_shape[1],
            tile_size=config.data.tile_size,
            overlap=config.data.tile_overlap,
            reference_table=bundle.reference,
            transform=bundle.transform,
        )
        pd.DataFrame([record.to_dict() for record in tile_records]).to_csv(plot_dir / "tiles.csv", index=False)
        qa_rows.append(qa_report.to_dict())

    if config.data.split_mode == "basic":
        splits = build_basic_split(
            [inventory.plot_id for inventory in inventories],
            train_fraction=config.data.basic_train_fraction,
            val_fraction=config.data.basic_val_fraction,
            seed=config.data.basic_split_seed,
        )
    else:
        splits = build_lopo_splits(
            [inventory.plot_id for inventory in inventories],
            config.training.validation_plots_per_fold,
        )
    split_rows = [split.to_dict() for split in splits]
    write_json({"folds": split_rows}, splits_dir / "lopo_splits.json")
    pd.DataFrame(qa_rows).to_csv(experiments_dir / "qa_summary.csv", index=False)
    logger.info("Prepared %d plots.", len(inventories))


if __name__ == "__main__":
    main()

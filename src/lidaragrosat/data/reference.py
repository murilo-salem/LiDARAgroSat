from __future__ import annotations

from pathlib import Path

import numpy as np

from lidaragrosat.utils.dependencies import require_dependency
from lidaragrosat.utils.geo import Bounds


def read_reference_table(reference_path: str | Path):
    reference_path = Path(reference_path)
    suffix = reference_path.suffix.lower()
    pandas = require_dependency("pandas", "reference table handling")

    if suffix == ".csv":
        table = pandas.read_csv(reference_path)
    elif suffix == ".parquet":
        table = pandas.read_parquet(reference_path)
    else:
        geopandas = require_dependency("geopandas", "vector reference loading")
        table = geopandas.read_file(reference_path)
        if "geometry" in table.columns:
            table["x"] = table.geometry.x
            table["y"] = table.geometry.y
            table = pandas.DataFrame(table.drop(columns=["geometry"]))

    lower_columns = {column.lower(): column for column in table.columns}
    if "x" not in lower_columns or "y" not in lower_columns:
        raise ValueError(f"Reference table at {reference_path} must contain x/y coordinates.")

    rename_map = {
        lower_columns["x"]: "x",
        lower_columns["y"]: "y",
    }
    for candidate in ("tree_id", "id", "treeid"):
        if candidate in lower_columns:
            rename_map[lower_columns[candidate]] = "tree_id"
            break
    for candidate in ("species", "specie", "taxon"):
        if candidate in lower_columns:
            rename_map[lower_columns[candidate]] = "species"
            break
    for candidate in ("height", "tree_height", "h", "z"):
        if candidate in lower_columns:
            rename_map[lower_columns[candidate]] = "height"
            break

    table = table.rename(columns=rename_map).copy()
    if "tree_id" not in table.columns:
        table["tree_id"] = [f"tree_{index:05d}" for index in range(len(table))]
    if "species" not in table.columns:
        table["species"] = "unknown"
    if "height" not in table.columns:
        table["height"] = pandas.NA
    return table[["tree_id", "x", "y", "species", "height"]].copy()


def write_reference_table(table, destination: str | Path) -> None:
    destination = Path(destination)
    if destination.suffix.lower() == ".parquet":
        table.to_parquet(destination, index=False)
    else:
        table.to_csv(destination, index=False)


def sanitize_reference_table(
    table,
    bounds: Bounds | None = None,
    clip_to_bounds: bool = True,
    height_min_m: float = 0.0,
    height_max_m: float = 80.0,
):
    pandas = require_dependency("pandas", "reference table handling")
    cleaned = table.copy()
    raw_count = int(len(cleaned))

    cleaned["tree_id"] = cleaned["tree_id"].astype(str)
    cleaned["x"] = pandas.to_numeric(cleaned["x"], errors="coerce")
    cleaned["y"] = pandas.to_numeric(cleaned["y"], errors="coerce")
    cleaned["species"] = cleaned["species"].fillna("unknown").astype(str)
    cleaned["height"] = pandas.to_numeric(cleaned["height"], errors="coerce")

    valid_xy_mask = cleaned["x"].notna() & cleaned["y"].notna()
    invalid_xy_count = int((~valid_xy_mask).sum())
    cleaned = cleaned.loc[valid_xy_mask].copy()

    invalid_height_mask = cleaned["height"].notna() & (
        (cleaned["height"] < float(height_min_m)) | (cleaned["height"] > float(height_max_m))
    )
    invalid_height_count = int(invalid_height_mask.sum())
    if invalid_height_count:
        cleaned.loc[invalid_height_mask, "height"] = np.nan

    out_of_bounds_count = 0
    if bounds is not None and clip_to_bounds:
        inside_mask = (
            (cleaned["x"] >= bounds.xmin)
            & (cleaned["x"] <= bounds.xmax)
            & (cleaned["y"] >= bounds.ymin)
            & (cleaned["y"] <= bounds.ymax)
        )
        out_of_bounds_count = int((~inside_mask).sum())
        cleaned = cleaned.loc[inside_mask].copy()

    cleaned = cleaned.reset_index(drop=True)
    stats = {
        "raw_count": raw_count,
        "clean_count": int(len(cleaned)),
        "invalid_xy_count": invalid_xy_count,
        "invalid_height_count": invalid_height_count,
        "out_of_bounds_count": out_of_bounds_count,
    }
    return cleaned, stats

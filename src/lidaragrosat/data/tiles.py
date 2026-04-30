from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lidaragrosat.utils.geo import GeoTransform, iter_windows


@dataclass(slots=True)
class TileRecord:
    plot_id: str
    row: int
    col: int
    size: int
    tree_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def count_trees_per_tile(reference_table, transform: GeoTransform, row: int, col: int, size: int) -> int:
    if len(reference_table) == 0:
        return 0
    rows = []
    cols = []
    for _, tree in reference_table.iterrows():
        tree_row, tree_col = transform.xy_to_rowcol(float(tree["x"]), float(tree["y"]))
        rows.append(tree_row)
        cols.append(tree_col)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    mask = (rows >= row) & (rows < row + size) & (cols >= col) & (cols < col + size)
    return int(mask.sum())


def build_tiles(
    plot_id: str,
    raster_height: int,
    raster_width: int,
    tile_size: int,
    overlap: int,
    reference_table,
    transform: GeoTransform,
) -> list[TileRecord]:
    records: list[TileRecord] = []
    for row, col in iter_windows(raster_height, raster_width, tile_size, overlap):
        records.append(
            TileRecord(
                plot_id=plot_id,
                row=row,
                col=col,
                size=tile_size,
                tree_count=count_trees_per_tile(reference_table, transform, row, col, tile_size),
            )
        )
    return records


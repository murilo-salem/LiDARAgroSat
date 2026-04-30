from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class GeoTransform:
    x_min: float
    y_max: float
    pixel_size: float

    def xy_to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.x_min) / self.pixel_size)
        row = int((self.y_max - y) / self.pixel_size)
        return row, col

    def rowcol_to_xy(self, row: int, col: int) -> tuple[float, float]:
        x = self.x_min + (col + 0.5) * self.pixel_size
        y = self.y_max - (row + 0.5) * self.pixel_size
        return x, y


def bounds_from_points(x_values, y_values) -> Bounds:
    return Bounds(
        xmin=float(x_values.min()),
        ymin=float(y_values.min()),
        xmax=float(x_values.max()),
        ymax=float(y_values.max()),
    )


def raster_shape(bounds: Bounds, resolution: float) -> tuple[int, int]:
    width = int(round(bounds.width / resolution))
    height = int(round(bounds.height / resolution))
    return height, width


def iter_windows(height: int, width: int, window_size: int, overlap: int) -> Iterator[tuple[int, int]]:
    step = max(1, window_size - overlap)
    row_positions = list(range(0, max(1, height - window_size + 1), step))
    col_positions = list(range(0, max(1, width - window_size + 1), step))

    if not row_positions or row_positions[-1] != max(0, height - window_size):
        row_positions.append(max(0, height - window_size))
    if not col_positions or col_positions[-1] != max(0, width - window_size):
        col_positions.append(max(0, width - window_size))

    for row in row_positions:
        for col in col_positions:
            yield row, col


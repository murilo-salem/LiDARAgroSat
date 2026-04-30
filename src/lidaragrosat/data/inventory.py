from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from lidaragrosat.utils.io import ensure_dir, write_json


SUPPORTED_REFERENCE_EXTENSIONS = (".gpkg", ".geojson", ".json", ".shp", ".parquet", ".csv")
SUPPORTED_LIDAR_EXTENSIONS = (".laz", ".las")
SUPPORTED_RGB_EXTENSIONS = (".tif", ".tiff")


@dataclass(slots=True)
class PlotInventory:
    plot_id: str
    lidar_path: str
    rgb_path: str
    reference_path: str
    metadata_path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _single_match(plot_dir: Path, patterns: Iterable[str], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(plot_dir.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f"No {label} file found in {plot_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {label} candidates found in {plot_dir}: {matches}")
    return matches[0]


def discover_dataset(raw_dir: str | Path, explicit_plot_ids: list[str] | None = None) -> list[PlotInventory]:
    root = Path(raw_dir)
    plot_dirs = sorted(
        [candidate for candidate in root.iterdir() if candidate.is_dir()]
        if explicit_plot_ids is None or not explicit_plot_ids
        else [(root / plot_id) for plot_id in explicit_plot_ids]
    )

    inventories: list[PlotInventory] = []
    for plot_dir in plot_dirs:
        lidar_path = _single_match(plot_dir, [f"*{suffix}" for suffix in SUPPORTED_LIDAR_EXTENSIONS], "LiDAR")
        rgb_path = _single_match(plot_dir, [f"*{suffix}" for suffix in SUPPORTED_RGB_EXTENSIONS], "RGB")
        reference_path = _single_match(
            plot_dir,
            [f"*{suffix}" for suffix in SUPPORTED_REFERENCE_EXTENSIONS],
            "reference",
        )
        metadata_candidates = sorted(plot_dir.glob("*.json"))
        metadata_path = str(metadata_candidates[0]) if metadata_candidates else None
        inventories.append(
            PlotInventory(
                plot_id=plot_dir.name,
                lidar_path=str(lidar_path),
                rgb_path=str(rgb_path),
                reference_path=str(reference_path),
                metadata_path=metadata_path,
            )
        )
    return inventories


def save_inventory(inventories: list[PlotInventory], destination: str | Path) -> None:
    ensure_dir(Path(destination).parent)
    payload = {"plots": [plot.to_dict() for plot in inventories]}
    write_json(payload, destination)


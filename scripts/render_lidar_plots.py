from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.config import ExperimentConfig
from lidaragrosat.utils.io import ensure_dir


def _load_raw_points(lidar_path: Path, max_points: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import laspy

    las = laspy.read(lidar_path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    n_points = int(x.size)
    if n_points > max_points:
        selection = rng.choice(n_points, size=max_points, replace=False)
        x = x[selection]
        y = y[selection]
        z = z[selection]

    x_local = x - float(x.min())
    y_local = y - float(y.min())
    z_rel = z - float(z.min())
    payload = {
        "x": x_local,
        "y": y_local,
        "z": z_rel,
    }
    summary = {
        "source": "raw_point_cloud",
        "points_rendered": int(x_local.size),
        "points_total": n_points,
        "height_max_m": float(z_rel.max()) if z_rel.size else 0.0,
    }
    return payload, summary


def _discover_plot_ids(processed_dir: Path, raw_dir: Path, plot_ids: list[str] | None) -> list[str]:
    if plot_ids:
        return sorted(plot_ids)
    discovered: set[str] = set()
    if processed_dir.exists():
        discovered.update(path.name for path in processed_dir.iterdir() if path.is_dir())
    if raw_dir.exists():
        discovered.update(path.name for path in raw_dir.iterdir() if path.is_dir())
    return sorted(discovered)


def _find_raw_lidar_path(raw_dir: Path, plot_id: str) -> Path | None:
    plot_dir = raw_dir / plot_id
    if not plot_dir.exists():
        return None
    preferred = sorted(plot_dir.glob("lidar.la*"))
    if preferred:
        return preferred[0]
    candidates = sorted(path for path in plot_dir.iterdir() if path.suffix.lower() in {".las", ".laz"})
    return candidates[0] if candidates else None


def _load_processed_arrays(processed_dir: Path, plot_id: str, density_index: int) -> tuple[np.ndarray, np.ndarray]:
    plot_dir = processed_dir / plot_id
    chm = np.load(plot_dir / "chm.npy").astype(np.float32)
    lidar = np.load(plot_dir / "lidar.npy").astype(np.float32)
    density = lidar[density_index].astype(np.float32)
    return chm, density


def _render_raw_top(ax, points: dict[str, np.ndarray], title: str) -> None:
    ax.scatter(points["x"], points["y"], c=points["z"], s=0.3, cmap="viridis", linewidths=0, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X local (m)", fontsize=8)
    ax.set_ylabel("Y local (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def _render_raw_side(ax, horizontal: np.ndarray, z: np.ndarray, title: str, xlabel: str) -> None:
    ax.scatter(horizontal, z, c=z, s=0.3, cmap="viridis", linewidths=0, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Altura relativa (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def _render_raster(ax, array: np.ndarray, title: str, cmap: str) -> None:
    ax.imshow(array, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _save_processed_panels(plot_id: str, chm: np.ndarray, density: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    _render_raster(axes[0], chm, f"{plot_id} | CHM", cmap="terrain")
    _render_raster(axes[1], density, f"{plot_id} | density_total", cmap="magma")
    fig.savefig(output_dir / f"{plot_id}_processed_lidar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_raw_multiview_panels(plot_id: str, points: dict[str, np.ndarray], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    _render_raw_top(axes[0], points, f"{plot_id} | top view")
    _render_raw_side(axes[1], points["x"], points["z"], f"{plot_id} | side X-Z", "X local (m)")
    _render_raw_side(axes[2], points["y"], points["z"], f"{plot_id} | side Y-Z", "Y local (m)")
    fig.savefig(output_dir / f"{plot_id}_raw_lidar_multiview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LiDAR plots from raw clouds or processed LiDAR rasters.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs_visualizations/lidar_plots")
    parser.add_argument("--plot-ids", nargs="*", default=None)
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()

    config = ExperimentConfig()
    density_index = config.data.structural_channels.index("density_total")
    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    plot_ids = _discover_plot_ids(processed_dir, raw_dir, args.plot_ids)
    rng = np.random.default_rng(13)

    if not plot_ids:
        raise SystemExit("No plots found in processed/raw directories.")

    raw_rows = len(plot_ids)
    raw_fig, raw_axes = plt.subplots(raw_rows, 3, figsize=(15, 3.8 * raw_rows), constrained_layout=True)
    raw_axes = np.atleast_2d(raw_axes)

    processed_rows = math.ceil(len(plot_ids) / 2)
    processed_fig, processed_axes = plt.subplots(processed_rows, 4, figsize=(18, 5 * processed_rows), constrained_layout=True)
    processed_axes = np.atleast_2d(processed_axes)

    summary_rows: list[dict[str, object]] = []

    for index, plot_id in enumerate(plot_ids):
        raw_axis_top, raw_axis_xz, raw_axis_yz = raw_axes[index]
        row = index // 2
        col = (index % 2) * 2
        chm_axis = processed_axes[row, col]
        density_axis = processed_axes[row, col + 1]

        raw_path = _find_raw_lidar_path(raw_dir, plot_id)
        if raw_path is not None:
            raw_points, raw_payload = _load_raw_points(raw_path, args.max_points, rng)
            _render_raw_top(raw_axis_top, raw_points, f"{plot_id} | top view")
            _render_raw_side(raw_axis_xz, raw_points["x"], raw_points["z"], f"{plot_id} | side X-Z", "X local (m)")
            _render_raw_side(raw_axis_yz, raw_points["y"], raw_points["z"], f"{plot_id} | side Y-Z", "Y local (m)")
            _save_raw_multiview_panels(plot_id, raw_points, output_dir)
        else:
            for axis, title in [
                (raw_axis_top, f"{plot_id} | top view"),
                (raw_axis_xz, f"{plot_id} | side X-Z"),
                (raw_axis_yz, f"{plot_id} | side Y-Z"),
            ]:
                axis.text(
                    0.5,
                    0.5,
                    "Raw LAS/LAZ\nnot available\nin workspace",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                axis.set_title(title, fontsize=10)
                axis.set_axis_off()
            raw_payload = {"source": "raw_missing", "points_rendered": 0, "points_total": 0, "height_max_m": 0.0}

        chm, density = _load_processed_arrays(processed_dir, plot_id, density_index=density_index)
        _render_raster(chm_axis, chm, f"{plot_id} | CHM", cmap="terrain")
        _render_raster(density_axis, density, f"{plot_id} | density_total", cmap="magma")
        _save_processed_panels(plot_id, chm, density, output_dir)

        summary_rows.append(
            {
                "plot_id": plot_id,
                **raw_payload,
                "processed_shape": f"{chm.shape[0]}x{chm.shape[1]}",
                "chm_max": float(np.nanmax(chm)),
                "density_max": float(np.nanmax(density)),
            }
        )

    flat_processed_axes = processed_axes.ravel()
    for axis in flat_processed_axes[2 * len(plot_ids) :]:
        axis.set_visible(False)

    raw_fig.suptitle("LiDAR Raw Point Clouds by Plot (Top + Lateral Height Views)", fontsize=16)
    processed_fig.suptitle("Processed LiDAR Views by Plot (CHM + density_total)", fontsize=16)
    raw_fig.savefig(output_dir / "all_plots_raw_lidar_multiview.png", dpi=180, bbox_inches="tight")
    processed_fig.savefig(output_dir / "all_plots_processed_lidar.png", dpi=180, bbox_inches="tight")
    plt.close(raw_fig)
    plt.close(processed_fig)

    pd.DataFrame(summary_rows).to_csv(output_dir / "render_summary.csv", index=False)
    print(output_dir / "all_plots_raw_lidar_multiview.png")
    print(output_dir / "all_plots_processed_lidar.png")
    print(output_dir / "render_summary.csv")


if __name__ == "__main__":
    main()

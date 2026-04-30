from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.utils.io import ensure_dir


def _find_raw_lidar_path(raw_dir: Path, plot_id: str) -> Path:
    plot_dir = raw_dir / plot_id
    if not plot_dir.exists():
        raise FileNotFoundError(f"Plot directory not found: {plot_dir}")
    preferred = sorted(plot_dir.glob("lidar.la*"))
    if preferred:
        return preferred[0]
    candidates = sorted(path for path in plot_dir.iterdir() if path.suffix.lower() in {".las", ".laz"})
    if not candidates:
        raise FileNotFoundError(f"No LAS/LAZ file found for {plot_id} in {plot_dir}")
    return candidates[0]


def _load_points(lidar_path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import laspy

    las = laspy.read(lidar_path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    if x.size > max_points:
        rng = np.random.default_rng(seed)
        selection = rng.choice(x.size, size=max_points, replace=False)
        x = x[selection]
        y = y[selection]
        z = z[selection]
    x = x - float(x.min())
    y = y - float(y.min())
    z = z - float(z.min())
    return x, y, z


def _style_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, plot_id: str) -> None:
    ax.set_facecolor("#05070b")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.03, 0.04, 0.07, 1.0))
        axis._axinfo["grid"]["color"] = (0.35, 0.40, 0.50, 0.18)
        axis._axinfo["tick"]["color"] = (0.88, 0.90, 0.95, 1.0)
    ax.tick_params(colors="#e7ecf5", labelsize=8)
    ax.set_xlabel("X local (m)", color="#f2f5fa", labelpad=8)
    ax.set_ylabel("Y local (m)", color="#f2f5fa", labelpad=8)
    ax.set_zlabel("Altura relativa (m)", color="#f2f5fa", labelpad=8)
    ax.set_title(f"LiDAR 3D | {plot_id}", color="#f8fbff", fontsize=16, pad=18)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), max(np.ptp(z), 1.0)))
    ax.set_xlim(0.0, float(x.max()))
    ax.set_ylim(0.0, float(y.max()))
    ax.set_zlim(0.0, float(z.max()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a rotating 3D GIF from a raw LiDAR point cloud.")
    parser.add_argument("--plot-id", default="plot_08")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs_visualizations/lidar_plots")
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--elev-base", type=float, default=22.0)
    parser.add_argument("--elev-amp", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    lidar_path = _find_raw_lidar_path(raw_dir, args.plot_id)
    x, y, z = _load_points(lidar_path, max_points=args.max_points, seed=args.seed)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8.5, 8.5), facecolor="#05070b")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x,
        y,
        z,
        c=z,
        cmap="viridis",
        s=0.45,
        linewidths=0,
        alpha=0.75,
        rasterized=True,
    )
    _style_axes(ax, x, y, z, args.plot_id)

    def update(frame_index: int):
        fraction = frame_index / max(args.frames, 1)
        azim = 360.0 * fraction
        elev = args.elev_base + args.elev_amp * np.sin(2.0 * np.pi * fraction)
        ax.view_init(elev=float(elev), azim=float(azim))
        return (ax,)

    animation = FuncAnimation(fig, update, frames=args.frames, interval=1000 / max(args.fps, 1), blit=False)
    output_path = Path(output_dir) / f"{args.plot_id}_lidar_3d.gif"
    animation.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()

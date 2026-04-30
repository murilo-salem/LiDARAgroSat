from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lidaragrosat.utils.io import ensure_dir
from scripts.render_lidar_gif import _find_raw_lidar_path, _load_points, _style_axes


def _discover_plot_ids(raw_dir: Path, plot_ids: list[str] | None) -> list[str]:
    if plot_ids:
        return sorted(plot_ids)
    return sorted(path.name for path in raw_dir.iterdir() if path.is_dir() and path.name.startswith("plot_"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single sequential 3D LiDAR GIF for multiple plots.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs_visualizations/lidar_plots")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--codec", default="mpeg4")
    parser.add_argument("--plot-ids", nargs="*", default=None)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument("--frames-per-plot", type=int, default=288)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--elev-base", type=float, default=22.0)
    parser.add_argument("--elev-amp", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    plot_ids = _discover_plot_ids(raw_dir, args.plot_ids)
    if not plot_ids:
        raise SystemExit("No raw LiDAR plot directories found.")

    plot_payloads = []
    for index, plot_id in enumerate(plot_ids):
        lidar_path = _find_raw_lidar_path(raw_dir, plot_id)
        x, y, z = _load_points(lidar_path, max_points=args.max_points, seed=args.seed + index)
        plot_payloads.append({"plot_id": plot_id, "x": x, "y": y, "z": z})

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8.5, 8.5), facecolor="#05070b")
    ax = fig.add_subplot(111, projection="3d")
    current_plot = {"index": None}

    def update(frame_index: int):
        plot_index = frame_index // args.frames_per_plot
        local_frame = frame_index % args.frames_per_plot
        payload = plot_payloads[plot_index]
        if current_plot["index"] != plot_index:
            ax.clear()
            ax.scatter(
                payload["x"],
                payload["y"],
                payload["z"],
                c=payload["z"],
                cmap="viridis",
                s=0.45,
                linewidths=0,
                alpha=0.75,
                rasterized=True,
            )
            _style_axes(ax, payload["x"], payload["y"], payload["z"], payload["plot_id"])
            ax.text2D(
                0.02,
                0.95,
                f"{plot_index + 1}/{len(plot_payloads)}",
                transform=ax.transAxes,
                color="#dce8ff",
                fontsize=12,
                bbox={"facecolor": "#101726", "edgecolor": "#2b3a55", "boxstyle": "round,pad=0.3"},
            )
            current_plot["index"] = plot_index

        fraction = local_frame / max(args.frames_per_plot, 1)
        azim = 360.0 * fraction
        elev = args.elev_base + args.elev_amp * math.sin(2.0 * math.pi * fraction)
        ax.view_init(elev=float(elev), azim=float(azim))
        return (ax,)

    total_frames = args.frames_per_plot * len(plot_payloads)
    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / max(args.fps, 1),
        blit=False,
    )
    if args.output_name:
        output_path = Path(output_dir) / args.output_name
    else:
        output_path = Path(output_dir) / f"all_plots_lidar_3d_sequence.{args.format}"
    if args.format == "mp4":
        writer = FFMpegWriter(fps=args.fps, bitrate=12_000, codec=args.codec)
    else:
        writer = PillowWriter(fps=args.fps)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(slots=True)
class FaithfulAdaptiveBaselineParams:
    a: float = 0.05
    b: float = 1.5
    height_threshold_m: float = 5.0
    min_window_m: float = 0.5
    max_window_m: float = 5.0


def adaptive_local_maxima_filter_3d(
    points_xyz: np.ndarray,
    params: FaithfulAdaptiveBaselineParams | None = None,
) -> pd.DataFrame:
    params = params or FaithfulAdaptiveBaselineParams()
    if len(points_xyz) == 0:
        return pd.DataFrame(columns=["x", "y", "height_pred", "window_size_m"])

    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[pts[:, 2] >= float(params.height_threshold_m)]
    if len(pts) == 0:
        return pd.DataFrame(columns=["x", "y", "height_pred", "window_size_m"])

    order = np.argsort(-pts[:, 2], kind="mergesort")
    pts_sorted = pts[order]
    tree = cKDTree(pts_sorted[:, :2])
    suppressed = np.zeros(len(pts_sorted), dtype=bool)
    detections: list[dict[str, float]] = []

    for index, point in enumerate(pts_sorted):
        if suppressed[index]:
            continue
        radius = float(np.clip(params.a * float(point[2]) + params.b, params.min_window_m, params.max_window_m))
        neighbor_indices = np.asarray(tree.query_ball_point(point[:2], r=radius), dtype=np.int64)
        if neighbor_indices.size:
            suppressed[neighbor_indices] = True
        detections.append(
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "height_pred": float(point[2]),
                "window_size_m": radius,
            }
        )

    if not detections:
        return pd.DataFrame(columns=["x", "y", "height_pred", "window_size_m"])
    return pd.DataFrame.from_records(detections)

from __future__ import annotations

import numpy as np
import pandas as pd

from lidaragrosat.data.auxiliary import find_local_apex_pixel
from lidaragrosat.data.dataset import ProcessedPlotBundle

FIELD_CENTER = "field_center"
APEX_SNAPPED = "apex_snapped"
MIXED_CENTER_APEX = "mixed_center_apex"


def resolve_label_policy_parameters(
    label_policy: str,
    snap_radius_m: float,
    mixed_weight: float,
) -> tuple[float, float]:
    if label_policy == FIELD_CENTER:
        return 0.0, 0.0
    if label_policy == APEX_SNAPPED:
        return float(snap_radius_m), 0.0
    if label_policy == MIXED_CENTER_APEX:
        return float(snap_radius_m), float(mixed_weight)
    raise ValueError(f"Unsupported label_policy: {label_policy}")


def resolve_reference_points(
    bundle: ProcessedPlotBundle,
    label_policy: str,
    snap_radius_m: float = 1.0,
    mixed_weight: float = 0.35,
) -> pd.DataFrame:
    reference = bundle.reference.copy()
    if reference.empty:
        reference["target_x"] = pd.Series(dtype=float)
        reference["target_y"] = pd.Series(dtype=float)
        reference["label_policy"] = pd.Series(dtype=str)
        return reference

    target_x: list[float] = []
    target_y: list[float] = []
    snap_radius_px = max(1, int(round(float(snap_radius_m) / max(bundle.transform.pixel_size, 1e-6))))

    for _, row in reference.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        if label_policy == FIELD_CENTER or snap_radius_m <= 0:
            target_x.append(x)
            target_y.append(y)
            continue
        center_row, center_col = bundle.transform.xy_to_rowcol(x, y)
        if not (0 <= center_row < bundle.chm.shape[0] and 0 <= center_col < bundle.chm.shape[1]):
            target_x.append(x)
            target_y.append(y)
            continue
        apex_row, apex_col = find_local_apex_pixel(bundle.chm, center_row, center_col, snap_radius_px)
        apex_x, apex_y = bundle.transform.rowcol_to_xy(apex_row, apex_col)
        if label_policy == APEX_SNAPPED:
            target_x.append(float(apex_x))
            target_y.append(float(apex_y))
        elif label_policy == MIXED_CENTER_APEX:
            target_x.append(float((1.0 - mixed_weight) * x + mixed_weight * apex_x))
            target_y.append(float((1.0 - mixed_weight) * y + mixed_weight * apex_y))
        else:
            raise ValueError(f"Unsupported label_policy: {label_policy}")

    reference["target_x"] = np.asarray(target_x, dtype=float)
    reference["target_y"] = np.asarray(target_y, dtype=float)
    reference["label_policy"] = label_policy
    return reference

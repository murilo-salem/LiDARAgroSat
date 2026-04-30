from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_plot_descriptors(bundle, species_groups: dict[str, str] | None = None) -> dict:
    species_groups = species_groups or {}
    area_ha = (bundle.chm.shape[0] * bundle.chm.shape[1] * bundle.transform.pixel_size**2) / 10_000.0
    canopy = bundle.chm[bundle.chm >= 2.0]
    grayscale = np.clip(bundle.rgb.mean(axis=0), -3.0, 3.0)
    histogram, _ = np.histogram(grayscale, bins=32, density=True)
    histogram = histogram[histogram > 0]
    rgb_entropy = -float(np.sum(histogram * np.log(histogram)))

    group_series = bundle.reference["species"].map(lambda value: species_groups.get(str(value), "unknown"))
    conifer_fraction = float((group_series == "conifer").mean()) if len(group_series) else np.nan

    structural_channels = bundle.metadata.get("structural_channels", []) if bundle.metadata else []
    roughness_index = structural_channels.index("vertical_roughness") if "vertical_roughness" in structural_channels else None
    roughness_mean = (
        float(bundle.lidar[roughness_index].mean()) if roughness_index is not None else np.nan
    )

    return {
        "plot_id": bundle.plot_id,
        "tree_density_ha": float(len(bundle.reference) / max(area_ha, 1e-6)),
        "canopy_height_variance": float(np.var(canopy)) if canopy.size else 0.0,
        "vertical_roughness_mean": roughness_mean,
        "rgb_entropy": rgb_entropy,
        "conifer_fraction": conifer_fraction,
    }


def build_gain_table(metrics_table: pd.DataFrame, fusion_method: str, baseline_method: str) -> pd.DataFrame:
    pivot = metrics_table.pivot(index="plot_id", columns="method", values="f1")
    return (
        pd.DataFrame(
            {
                "plot_id": pivot.index,
                "fusion_method": fusion_method,
                "baseline_method": baseline_method,
                "delta_f1": pivot[fusion_method] - pivot[baseline_method],
            }
        )
        .reset_index(drop=True)
    )


def fit_gain_vs_complexity(gain_table: pd.DataFrame, descriptor_table: pd.DataFrame, predictor: str) -> dict:
    from sklearn.linear_model import HuberRegressor

    merged = gain_table.merge(descriptor_table[["plot_id", predictor]], on="plot_id", how="inner").dropna()
    if len(merged) < 3:
        return {"predictor": predictor, "slope": np.nan, "intercept": np.nan, "r2": np.nan}

    x = merged[[predictor]].to_numpy(dtype=float)
    y = merged["delta_f1"].to_numpy(dtype=float)
    try:
        model = HuberRegressor().fit(x, y)
        r2 = float(model.score(x, y))
    except Exception:
        return {"predictor": predictor, "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    return {
        "predictor": predictor,
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": r2,
    }


def classify_error_modes(bundle, predictions: pd.DataFrame, match_result, overlap_radius_m: float) -> pd.DataFrame:
    records: list[dict] = []

    matched_prediction_indices = set(match_result.matches["pred_idx"].astype(int).tolist()) if not match_result.matches.empty else set()
    tp_predictions = predictions.iloc[match_result.matches["pred_idx"].astype(int)] if not match_result.matches.empty else predictions.iloc[0:0]
    tp_xy = tp_predictions[["x", "y"]].to_numpy(dtype=float) if len(tp_predictions) else np.empty((0, 2))

    for _, row in match_result.false_positives.reset_index(drop=True).iterrows():
        x = float(row["x"])
        y = float(row["y"])
        boundary_distance = min(
            x - bundle.metadata["bounds"]["xmin"],
            bundle.metadata["bounds"]["xmax"] - x,
            y - bundle.metadata["bounds"]["ymin"],
            bundle.metadata["bounds"]["ymax"] - y,
        )
        mode = "fp_border_artifact" if boundary_distance < overlap_radius_m else "fp_fragmented_texture"
        if tp_xy.size:
            distance = np.sqrt(((tp_xy - np.array([x, y])) ** 2).sum(axis=1)).min()
            if distance < overlap_radius_m:
                mode = "fp_duplicate"
        records.append({"plot_id": bundle.plot_id, "error_type": "fp", "mode": mode, "x": x, "y": y})

    reference_xy = bundle.reference[["x", "y"]].to_numpy(dtype=float) if len(bundle.reference) else np.empty((0, 2))
    for _, row in match_result.false_negatives.iterrows():
        point = np.array([float(row["x"]), float(row["y"])], dtype=float)
        if reference_xy.size:
            distances = np.sqrt(((reference_xy - point) ** 2).sum(axis=1))
            has_neighbor = int((distances <= overlap_radius_m).sum()) > 1
        else:
            has_neighbor = False
        mode = "fn_overlapping_crown" if has_neighbor else "fn_no_dominant_apex"
        records.append({"plot_id": bundle.plot_id, "error_type": "fn", "mode": mode, "x": point[0], "y": point[1]})

    return pd.DataFrame.from_records(records)


def save_regression_table(rows: list[dict], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)

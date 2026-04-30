from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, generic_filter, label
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .method import FaithfulMatchingParams, faithful_greedy_match, local_maxima_filter_3d, summarize_faithful_matches
from .watershed import build_chm


@dataclass(slots=True)
class FaithfulHybridWatershedProposalParams:
    resolution_m: float = 0.25
    smooth_sigma_px: float = 1.0
    min_distance_px: int = 4
    min_segment_area_m2: float = 4.0
    proposal_window_m: float = 0.8
    proposal_height_threshold_m: float = 3.0
    canopy_height_threshold_m: float = 3.0


def _fill_nan_mean3x3(chm: np.ndarray, max_iter: int = 4) -> np.ndarray:
    filled = np.asarray(chm, dtype=np.float32).copy()
    if filled.size == 0:
        return filled

    def _mean_valid(window: np.ndarray) -> float:
        valid = window[np.isfinite(window)]
        return float(valid.mean()) if len(valid) > 0 else np.nan

    for _ in range(max_iter):
        mask = ~np.isfinite(filled)
        if not np.any(mask):
            break
        candidate = generic_filter(filled, _mean_valid, size=3, mode="nearest")
        replace_mask = mask & np.isfinite(candidate)
        filled[replace_mask] = candidate[replace_mask]
    filled[~np.isfinite(filled)] = 0.0
    return filled.astype(np.float32)


def generate_hybrid_chm(
    points_xyz: np.ndarray,
    resolution_m: float = 0.25,
) -> tuple[np.ndarray, float, float, float]:
    chm, _occupancy, xmin, ymax = build_chm(points_xyz=points_xyz, resolution_m=float(resolution_m))
    chm_filled = _fill_nan_mean3x3(chm)
    return chm_filled, float(xmin), float(ymax), float(resolution_m)


def segment_watershed_chm(
    chm: np.ndarray,
    smooth_sigma_px: float = 1.0,
    min_distance_px: int = 4,
    min_height_m: float = 3.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    if chm.size == 0:
        empty = np.zeros_like(chm, dtype=np.int32)
        return empty, 0, empty
    chm_smooth = gaussian_filter(chm, sigma=float(smooth_sigma_px), mode="nearest")
    canopy_mask = chm_smooth > float(min_height_m)
    if not np.any(canopy_mask):
        empty = np.zeros_like(chm_smooth, dtype=np.int32)
        return empty, 0, empty

    peak_coords = peak_local_max(
        chm_smooth,
        min_distance=max(1, int(min_distance_px)),
        threshold_abs=float(min_height_m),
        labels=canopy_mask.astype(np.uint8),
        exclude_border=False,
    )
    if peak_coords.size == 0:
        empty = np.zeros_like(chm_smooth, dtype=np.int32)
        return empty, 0, empty

    seeds_mask = np.zeros_like(chm_smooth, dtype=bool)
    seeds_mask[peak_coords[:, 0], peak_coords[:, 1]] = True
    seeds, n_seeds = label(seeds_mask)
    chm_inv = float(chm_smooth.max()) - chm_smooth
    segments = watershed(chm_inv, markers=seeds, mask=canopy_mask)
    return segments.astype(np.int32), int(n_seeds), chm_smooth.astype(np.float32)


def _xy_to_row_col(x: float, y: float, xmin: float, ymax: float, resolution_m: float) -> tuple[int, int]:
    col = int(np.floor((float(x) - float(xmin)) / float(resolution_m)))
    row = int(np.floor((float(ymax) - float(y)) / float(resolution_m)))
    return row, col


def hybrid_filter_segments(
    proposals: pd.DataFrame,
    chm: np.ndarray,
    segments: np.ndarray,
    x_min: float,
    y_max: float,
    resolution_m: float,
    min_segment_area_m2: float = 4.0,
    chm_smooth: np.ndarray | None = None,
) -> pd.DataFrame:
    if chm.size == 0 or segments.size == 0:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "segment_area_m2"])
    n_rows, n_cols = segments.shape
    unique_labels = np.unique(segments)
    unique_labels = unique_labels[unique_labels > 0]
    records: list[dict] = []
    smooth = np.asarray(chm_smooth, dtype=np.float32) if chm_smooth is not None else np.asarray(chm, dtype=np.float32)

    proposal_rows = proposals.to_dict(orient="records")
    proposals_by_segment: dict[int, list[dict]] = {}
    for proposal in proposal_rows:
        row, col = _xy_to_row_col(proposal["x"], proposal["y"], x_min, y_max, resolution_m)
        if 0 <= row < n_rows and 0 <= col < n_cols:
            segment_id = int(segments[row, col])
            if segment_id > 0:
                proposals_by_segment.setdefault(segment_id, []).append(proposal)

    for segment_id in unique_labels.tolist():
        segment_mask = segments == int(segment_id)
        area_m2 = float(segment_mask.sum() * (float(resolution_m) ** 2))
        if area_m2 < float(min_segment_area_m2):
            continue
        inside = proposals_by_segment.get(int(segment_id), [])
        if inside:
            best = max(inside, key=lambda record: float(record["height_pred"]))
            records.append(
                {
                    "x": float(best["x"]),
                    "y": float(best["y"]),
                    "height_pred": float(best["height_pred"]),
                    "confidence": float(best.get("height_pred", 0.0)),
                    "source": "proposal",
                    "segment_id": int(segment_id),
                    "segment_area_m2": area_m2,
                }
            )
            continue
        rows_seg, cols_seg = np.where(segment_mask)
        if len(rows_seg) == 0:
            continue
        segment_values = chm[rows_seg, cols_seg]
        best_idx = int(np.argmax(segment_values))
        row = int(rows_seg[best_idx])
        col = int(cols_seg[best_idx])
        x_det = float(x_min + (col + 0.5) * float(resolution_m))
        y_det = float(y_max - (row + 0.5) * float(resolution_m))
        records.append(
            {
                "x": x_det,
                "y": y_det,
                "height_pred": float(chm[row, col]),
                "confidence": float(smooth[row, col]),
                "source": "chm_fallback",
                "segment_id": int(segment_id),
                "segment_area_m2": area_m2,
            }
        )

    if not records:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source", "segment_id", "segment_area_m2"])
    return pd.DataFrame.from_records(records).sort_values(["y", "x"]).reset_index(drop=True)


def analyze_segments(
    segments: np.ndarray,
    ground_truth_xy: np.ndarray,
    x_min: float,
    y_max: float,
    resolution_m: float,
    match_radius_m: float = 2.5,
) -> dict[str, int]:
    if segments.size == 0:
        return {"correct": 0, "over": int(len(ground_truth_xy)), "under": 0, "empty": 0, "n_segments": 0}
    gt_tree = cKDTree(np.asarray(ground_truth_xy, dtype=np.float32)) if len(ground_truth_xy) else None
    stats = {"correct": 0, "over": 0, "under": 0, "empty": 0, "n_segments": 0}
    covered: set[int] = set()
    unique_labels = np.unique(segments)
    unique_labels = unique_labels[unique_labels > 0]
    stats["n_segments"] = int(len(unique_labels))

    for segment_id in unique_labels.tolist():
        seg_mask = segments == int(segment_id)
        rows_s, cols_s = np.where(seg_mask)
        if len(rows_s) == 0:
            continue
        cx = float(x_min + (cols_s.mean() + 0.5) * float(resolution_m))
        cy = float(y_max - (rows_s.mean() + 0.5) * float(resolution_m))
        if gt_tree is None:
            stats["empty"] += 1
            continue
        idxs_far = gt_tree.query_ball_point([cx, cy], r=float(match_radius_m) * 2.0)
        if len(idxs_far) == 0:
            stats["empty"] += 1
        elif len(idxs_far) == 1:
            stats["correct"] += 1
        else:
            stats["under"] += 1
        idxs_cover = gt_tree.query_ball_point([cx, cy], r=float(match_radius_m))
        covered.update(int(idx) for idx in idxs_cover)

    stats["over"] = int(len(ground_truth_xy) - len(covered))
    return stats


def hybrid_watershed_proposals_filter(
    points_xyz: np.ndarray,
    reference: pd.DataFrame,
    params: FaithfulHybridWatershedProposalParams,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    proposals = local_maxima_filter_3d(
        points_xyz=points_xyz,
        window_size_m=float(params.proposal_window_m),
        height_threshold_m=float(params.proposal_height_threshold_m),
    )
    chm, x_min, y_max, resolution_m = generate_hybrid_chm(points_xyz=points_xyz, resolution_m=float(params.resolution_m))
    segments, n_seeds, chm_smooth = segment_watershed_chm(
        chm=chm,
        smooth_sigma_px=float(params.smooth_sigma_px),
        min_distance_px=int(params.min_distance_px),
        min_height_m=float(params.canopy_height_threshold_m),
    )
    detections = hybrid_filter_segments(
        proposals=proposals,
        chm=chm,
        segments=segments,
        x_min=x_min,
        y_max=y_max,
        resolution_m=resolution_m,
        min_segment_area_m2=float(params.min_segment_area_m2),
        chm_smooth=chm_smooth,
    )
    diagnostics = analyze_segments(
        segments=segments,
        ground_truth_xy=reference[["x", "y"]].to_numpy(dtype=np.float32),
        x_min=x_min,
        y_max=y_max,
        resolution_m=resolution_m,
        match_radius_m=2.5,
    )
    diagnostics.update(
        {
            "n_proposals": int(len(proposals)),
            "n_seeds": int(n_seeds),
            "n_segments_after_mask": int(len(np.unique(segments[segments > 0]))),
            "n_final_detections": int(len(detections)),
            "proposal_to_tree_ratio": float(len(proposals) / max(len(reference), 1)),
        }
    )
    return detections, diagnostics


def evaluate_hybrid_params(
    points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    params: FaithfulHybridWatershedProposalParams,
    matching_params: FaithfulMatchingParams,
    plot_ids: list[str] | None = None,
    prediction_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    selected_plots = sorted(plot_ids) if plot_ids is not None else sorted(points_by_plot)
    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for plot_id in selected_plots:
        detections, diagnostics = hybrid_watershed_proposals_filter(
            points_xyz=points_by_plot[plot_id],
            reference=reference_by_plot[plot_id],
            params=params,
        )
        if prediction_dir is not None:
            from pathlib import Path

            Path(prediction_dir).mkdir(parents=True, exist_ok=True)
            detections.to_csv(Path(prediction_dir) / f"{plot_id}.csv", index=False)
        match_result = faithful_greedy_match(detections, reference_by_plot[plot_id], matching_params)
        metrics = summarize_faithful_matches(match_result)
        metrics["plot_id"] = plot_id
        metrics["n_detections"] = int(len(detections))
        metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
        metrics.update(
            {
                "resolution_m": float(params.resolution_m),
                "smooth_sigma_px": float(params.smooth_sigma_px),
                "min_distance_px": int(params.min_distance_px),
                "min_segment_area_m2": float(params.min_segment_area_m2),
                "proposal_window_m": float(params.proposal_window_m),
            }
        )
        metric_rows.append(metrics)
        diagnostic_rows.append({"plot_id": plot_id, **diagnostics})
    per_plot_metrics = pd.DataFrame.from_records(metric_rows).sort_values("plot_id").reset_index(drop=True)
    diagnostics_table = pd.DataFrame.from_records(diagnostic_rows).sort_values("plot_id").reset_index(drop=True)
    tp = int(per_plot_metrics["tp"].sum())
    fp = int(per_plot_metrics["fp"].sum())
    fn = int(per_plot_metrics["fn"].sum())
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    summary = {
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
        "precision_global": precision,
        "recall_global": recall,
        "f1_global": f1,
        "precision_macro": float(per_plot_metrics["precision"].mean()),
        "recall_macro": float(per_plot_metrics["recall"].mean()),
        "f1_macro": float(per_plot_metrics["f1"].mean()),
        "distance_macro": float(per_plot_metrics["distance_m"].mean()),
        "height_error_macro": float(per_plot_metrics["height_error_m"].mean()),
        "mean_detections": float(per_plot_metrics["n_detections"].mean()),
        "mean_proposal_to_tree_ratio": float(diagnostics_table["proposal_to_tree_ratio"].mean()),
        "mean_empty_segments": float(diagnostics_table["empty"].mean()),
        "mean_under_segments": float(diagnostics_table["under"].mean()),
    }
    return per_plot_metrics, diagnostics_table, summary


def grid_search_hybrid_params(
    points_by_plot: dict[str, np.ndarray],
    reference_by_plot: dict[str, pd.DataFrame],
    matching_params: FaithfulMatchingParams,
    plot_ids: list[str],
    resolution_values: list[float],
    smooth_sigma_values: list[float],
    min_distance_values: list[int],
    min_segment_area_values: list[float],
    proposal_window_values: list[float],
) -> pd.DataFrame:
    proposal_cache: dict[tuple[str, float], pd.DataFrame] = {}
    chm_cache: dict[tuple[str, float], tuple[np.ndarray, float, float, float]] = {}
    segment_cache: dict[tuple[str, float, float, int], tuple[np.ndarray, int, np.ndarray]] = {}

    for plot_id in plot_ids:
        points_xyz = points_by_plot[plot_id]
        for proposal_window_m in proposal_window_values:
            proposal_cache[(plot_id, float(proposal_window_m))] = local_maxima_filter_3d(
                points_xyz=points_xyz,
                window_size_m=float(proposal_window_m),
                height_threshold_m=3.0,
            )
        for resolution_m in resolution_values:
            chm_cache[(plot_id, float(resolution_m))] = generate_hybrid_chm(points_xyz=points_xyz, resolution_m=float(resolution_m))
            chm, _xmin, _ymax, _resolution = chm_cache[(plot_id, float(resolution_m))]
            for smooth_sigma_px in smooth_sigma_values:
                for min_distance_px in min_distance_values:
                    segment_cache[(plot_id, float(resolution_m), float(smooth_sigma_px), int(min_distance_px))] = segment_watershed_chm(
                        chm=chm,
                        smooth_sigma_px=float(smooth_sigma_px),
                        min_distance_px=int(min_distance_px),
                        min_height_m=3.0,
                    )

    rows: list[dict] = []
    for resolution_m, smooth_sigma_px, min_distance_px, min_segment_area_m2, proposal_window_m in product(
        resolution_values,
        smooth_sigma_values,
        min_distance_values,
        min_segment_area_values,
        proposal_window_values,
    ):
        metric_rows: list[dict] = []
        diagnostic_rows: list[dict] = []
        for plot_id in plot_ids:
            proposals = proposal_cache[(plot_id, float(proposal_window_m))]
            chm, x_min, y_max, resolution_cached = chm_cache[(plot_id, float(resolution_m))]
            segments, n_seeds, chm_smooth = segment_cache[(plot_id, float(resolution_m), float(smooth_sigma_px), int(min_distance_px))]
            detections = hybrid_filter_segments(
                proposals=proposals,
                chm=chm,
                segments=segments,
                x_min=x_min,
                y_max=y_max,
                resolution_m=resolution_cached,
                min_segment_area_m2=float(min_segment_area_m2),
                chm_smooth=chm_smooth,
            )
            diagnostics = analyze_segments(
                segments=segments,
                ground_truth_xy=reference_by_plot[plot_id][["x", "y"]].to_numpy(dtype=np.float32),
                x_min=x_min,
                y_max=y_max,
                resolution_m=resolution_cached,
                match_radius_m=2.5,
            )
            diagnostics.update(
                {
                    "plot_id": plot_id,
                    "n_proposals": int(len(proposals)),
                    "n_seeds": int(n_seeds),
                    "n_segments_after_mask": int(len(np.unique(segments[segments > 0]))),
                    "n_final_detections": int(len(detections)),
                    "proposal_to_tree_ratio": float(len(proposals) / max(len(reference_by_plot[plot_id]), 1)),
                }
            )
            match_result = faithful_greedy_match(detections, reference_by_plot[plot_id], matching_params)
            metrics = summarize_faithful_matches(match_result)
            metrics["plot_id"] = plot_id
            metrics["n_detections"] = int(len(detections))
            metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
            metric_rows.append(metrics)
            diagnostic_rows.append(diagnostics)

        per_plot_metrics = pd.DataFrame.from_records(metric_rows).sort_values("plot_id").reset_index(drop=True)
        diagnostics_table = pd.DataFrame.from_records(diagnostic_rows).sort_values("plot_id").reset_index(drop=True)
        tp = int(per_plot_metrics["tp"].sum())
        fp = int(per_plot_metrics["fp"].sum())
        fn = int(per_plot_metrics["fn"].sum())
        precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        rows.append(
            {
                "resolution_m": float(resolution_m),
                "smooth_sigma_px": float(smooth_sigma_px),
                "min_distance_px": int(min_distance_px),
                "min_segment_area_m2": float(min_segment_area_m2),
                "proposal_window_m": float(proposal_window_m),
                "f1_macro": float(per_plot_metrics["f1"].mean()),
                "precision_macro": float(per_plot_metrics["precision"].mean()),
                "recall_macro": float(per_plot_metrics["recall"].mean()),
                "f1_global": f1,
                "precision_global": precision,
                "recall_global": recall,
                "mean_empty_segments": float(diagnostics_table["empty"].mean()),
                "mean_under_segments": float(diagnostics_table["under"].mean()),
                "tp_total": tp,
                "fp_total": fp,
                "fn_total": fn,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["f1_macro", "f1_global"], ascending=False).reset_index(drop=True)

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidaragrosat.analysis.reporting import build_method_summary
from lidaragrosat.benchmarking import PROTOCOL_A_ID, build_protocol_a_splits
from lidaragrosat.config import ExperimentConfig, save_config
from lidaragrosat.data.auxiliary import compute_multiscale_peakness
from lidaragrosat.data.dataset import list_processed_plots
from lidaragrosat.evaluation.matching import match_tree_predictions
from lidaragrosat.evaluation.metrics import summarize_matches
from lidaragrosat.stage1.generators import WatershedStage1Generator
from lidaragrosat.stage2.dataset import build_local_proposal_table
from lidaragrosat.stage2.graph_models import SegmentGraphClassifier
from lidaragrosat.utils.io import ensure_dir, write_json


def _rgb_patch_stats(bundle, x: float, y: float, radius_px: int = 12) -> tuple[float, float]:
    row, col = bundle.transform.xy_to_rowcol(float(x), float(y))
    r0 = max(0, row - radius_px)
    r1 = min(bundle.rgb.shape[1], row + radius_px + 1)
    c0 = max(0, col - radius_px)
    c1 = min(bundle.rgb.shape[2], col + radius_px + 1)
    patch = bundle.rgb[:, r0:r1, c0:c1].astype(np.float32)
    if patch.size == 0:
        return 0.0, 0.0
    rgb_std = float(np.std(patch) / 255.0)
    mean = patch.mean(axis=(1, 2), keepdims=True)
    entropy_proxy = float(np.mean(np.abs(patch - mean)) / 255.0)
    return rgb_std, entropy_proxy


def _local_peak_stats(bundle, stage1_output, segment_id: int, smooth_sigma_px: float = 1.0) -> tuple[int, float]:
    if stage1_output.segment_labels is None or int(segment_id) <= 0:
        return 0, 0.0
    segment_mask = stage1_output.segment_labels == int(segment_id)
    if not np.any(segment_mask):
        return 0, 0.0
    smooth = gaussian_filter(bundle.chm.astype(np.float32), sigma=float(smooth_sigma_px), mode="nearest")
    peaks = peak_local_max(
        smooth,
        min_distance=max(1, int(round(0.8 / bundle.transform.pixel_size))),
        threshold_abs=3.0,
        labels=segment_mask.astype(np.uint8),
        exclude_border=False,
    )
    peak_count = int(len(peaks))
    max_peak_value = float(smooth[tuple(peaks[0])] if peak_count else np.max(smooth[segment_mask]))
    return peak_count, max_peak_value


def _segment_adjacency(segment_labels: np.ndarray, segment_ids: list[int]) -> np.ndarray:
    n = len(segment_ids)
    adjacency = np.eye(n, dtype=np.float32)
    if segment_labels is None or n == 0:
        return adjacency
    id_to_index = {int(seg_id): idx for idx, seg_id in enumerate(segment_ids)}
    horizontal_left = segment_labels[:, :-1]
    horizontal_right = segment_labels[:, 1:]
    vertical_top = segment_labels[:-1, :]
    vertical_bottom = segment_labels[1:, :]
    pairs = []
    pairs.extend(zip(horizontal_left.ravel().tolist(), horizontal_right.ravel().tolist()))
    pairs.extend(zip(vertical_top.ravel().tolist(), vertical_bottom.ravel().tolist()))
    for a, b in pairs:
        if int(a) <= 0 or int(b) <= 0 or int(a) == int(b):
            continue
        ia = id_to_index.get(int(a))
        ib = id_to_index.get(int(b))
        if ia is None or ib is None:
            continue
        adjacency[ia, ib] = 1.0
        adjacency[ib, ia] = 1.0
    return adjacency


def _build_segment_table_for_plot(bundle, stage1_output) -> pd.DataFrame:
    sample_table = build_local_proposal_table(
        plot_bundles={bundle.plot_id: bundle},
        stage1_outputs={bundle.plot_id: stage1_output},
        plot_ids=[bundle.plot_id],
        mode="segment-centered",
        label_policy="field_center",
        positive_radius_m=1.5,
    )
    if sample_table.empty:
        return sample_table
    rows = []
    for _, row in sample_table.iterrows():
        rgb_std, rgb_entropy = _rgb_patch_stats(bundle, float(row["x"]), float(row["y"]))
        peak_count, peak_value = _local_peak_stats(bundle, stage1_output, int(row.get("segment_id", 0)))
        rows.append(
            {
                **row.to_dict(),
                "rgb_std": rgb_std,
                "rgb_entropy": rgb_entropy,
                "peak_count": float(peak_count),
                "peak_value": peak_value,
                "height_gap": float(row.get("height_pred", 0.0) - row.get("height_mean_m", 0.0)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _node_features(table: pd.DataFrame) -> np.ndarray:
    cols = [
        "area_m2",
        "height_pred",
        "height_mean_m",
        "height_gap",
        "geometry_score",
        "elongation",
        "compactness",
        "rgb_std",
        "rgb_entropy",
        "peak_count",
        "peak_value",
    ]
    features = table[cols].fillna(0.0).to_numpy(dtype=np.float32)
    scale = np.array([20.0, 35.0, 35.0, 15.0, 1.0, 10.0, 1.0, 1.0, 1.0, 5.0, 35.0], dtype=np.float32)
    return features / np.maximum(scale, 1e-6)


def _summarize_predictions(predictions_by_plot: dict[str, pd.DataFrame], bundles, plot_ids: list[str], match_radius_m: float, height_tolerance_m: float) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for plot_id in plot_ids:
        matches = match_tree_predictions(
            predictions_by_plot.get(plot_id, pd.DataFrame(columns=["x", "y", "height_pred"])),
            bundles[plot_id].reference,
            match_radius_m=match_radius_m,
            height_tolerance_m=height_tolerance_m,
        )
        row = summarize_matches(matches)
        row["plot_id"] = plot_id
        rows.append(row)
    table = pd.DataFrame.from_records(rows).sort_values("plot_id").reset_index(drop=True)
    summary = {
        "precision": float(table["precision"].mean()) if len(table) else 0.0,
        "recall": float(table["recall"].mean()) if len(table) else 0.0,
        "f1": float(table["f1"].mean()) if len(table) else 0.0,
        "location_mae_m": float(table["location_mae_m"].mean()) if len(table) else 0.0,
        "height_mae_m": float(table["height_mae_m"].mean()) if len(table) else 0.0,
        "tp": int(table["tp"].sum()) if len(table) else 0,
        "fp": int(table["fp"].sum()) if len(table) else 0,
        "fn": int(table["fn"].sum()) if len(table) else 0,
    }
    return table, summary


def _apply_keep_predictions(table: pd.DataFrame, keep_scores: np.ndarray, threshold: float, nms_radius_m: float) -> pd.DataFrame:
    working = table.copy()
    working["score"] = keep_scores.astype(np.float32)
    selected = working[working["score"] >= float(threshold)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "score"])
    selected = selected.sort_values(["score", "height_pred"], ascending=[False, False]).reset_index(drop=True)
    kept = []
    taken = np.zeros(len(selected), dtype=bool)
    xy = selected[["x", "y"]].to_numpy(dtype=np.float32)
    radius_sq = float(nms_radius_m) ** 2
    for i in range(len(selected)):
        if taken[i]:
            continue
        kept.append(selected.iloc[i][["x", "y", "height_pred", "score"]].to_dict())
        diff = xy - xy[i]
        dist_sq = np.sum(diff * diff, axis=1)
        taken |= dist_sq <= radius_sq
    return pd.DataFrame.from_records(kept)


def _calibrate_gnn(val_tables: dict[str, pd.DataFrame], score_by_plot: dict[str, np.ndarray], bundles, match_radius_m: float, height_tolerance_m: float) -> tuple[dict, dict]:
    best_params = {"threshold": 0.5, "nms_radius_m": 1.0}
    best_scores = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        for nms_radius_m in [0.8, 1.0, 1.2, 1.5, 1.8]:
            predictions = {
                plot_id: _apply_keep_predictions(val_tables[plot_id], score_by_plot[plot_id], threshold=threshold, nms_radius_m=nms_radius_m)
                for plot_id in val_tables
            }
            _, summary = _summarize_predictions(predictions, bundles=bundles, plot_ids=list(val_tables), match_radius_m=match_radius_m, height_tolerance_m=height_tolerance_m)
            if summary["f1"] > best_scores["f1"] or (summary["f1"] == best_scores["f1"] and summary["recall"] > best_scores["recall"]):
                best_params = {"threshold": float(threshold), "nms_radius_m": float(nms_radius_m)}
                best_scores = summary
    return best_params, best_scores


def _train_gnn(train_graphs, val_graphs, device: str) -> SegmentGraphClassifier:
    model = SegmentGraphClassifier(in_dim=train_graphs[0]["x"].shape[1], hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state = model.state_dict()
    best_f1 = -1.0
    wait = 0
    for _epoch in range(25):
        model.train()
        for graph in train_graphs:
            x = torch.tensor(graph["x"], dtype=torch.float32, device=device)
            adjacency = torch.tensor(graph["adj"], dtype=torch.float32, device=device)
            y = torch.tensor(graph["y"], dtype=torch.float32, device=device)
            logits = model(x, adjacency)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for graph in val_graphs:
                x = torch.tensor(graph["x"], dtype=torch.float32, device=device)
                adjacency = torch.tensor(graph["adj"], dtype=torch.float32, device=device)
                logits = model(x, adjacency)
                score = torch.sigmoid(logits).cpu().numpy()
                y_true.extend(graph["y"].tolist())
                y_pred.extend((score >= 0.3).astype(np.float32).tolist())
            tp = sum(int(a == 1 and b == 1) for a, b in zip(y_true, y_pred))
            fp = sum(int(a == 0 and b == 1) for a, b in zip(y_true, y_pred))
            fn = sum(int(a == 1 and b == 0) for a, b in zip(y_true, y_pred))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 4:
            break
    model.load_state_dict(best_state)
    return model


def _score_graph(model: SegmentGraphClassifier, graph: dict, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(graph["x"], dtype=torch.float32, device=device)
        adjacency = torch.tensor(graph["adj"], dtype=torch.float32, device=device)
        logits = model(x, adjacency)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


@dataclass(slots=True)
class SplitDetectionParams:
    split_threshold: float = 0.5
    peak_min_distance_m: float = 0.8
    nms_radius_m: float = 1.2
    min_area_m2: float = 6.0


def _segment_multi_peak_predictions(bundle, stage1_output, segment_row: pd.Series, params: SplitDetectionParams) -> list[dict]:
    if stage1_output.segment_labels is None:
        return []
    segment_id = int(segment_row.get("segment_id", 0))
    if segment_id <= 0:
        return []
    mask = stage1_output.segment_labels == segment_id
    if not np.any(mask):
        return []
    smooth = gaussian_filter(bundle.chm.astype(np.float32), sigma=1.0, mode="nearest")
    peaks = peak_local_max(
        smooth,
        min_distance=max(1, int(round(float(params.peak_min_distance_m) / bundle.transform.pixel_size))),
        threshold_abs=3.0,
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )
    if len(peaks) <= 1:
        return []
    max_keep = max(2, int(round(float(segment_row.get("area_m2", 0.0)) / 18.0)))
    scores = []
    for row, col in peaks:
        x, y = bundle.transform.rowcol_to_xy(int(row), int(col))
        scores.append({"x": float(x), "y": float(y), "height_pred": float(bundle.chm[int(row), int(col)]), "score": float(smooth[int(row), int(col)])})
    scores = sorted(scores, key=lambda item: item["score"], reverse=True)[:max_keep]
    return scores


def _apply_split_predictor(table: pd.DataFrame, split_probs: np.ndarray, stage1_outputs, bundles, params: SplitDetectionParams) -> dict[str, pd.DataFrame]:
    predictions_by_plot: dict[str, pd.DataFrame] = {}
    for plot_id, plot_table in table.groupby("plot_id"):
        bundle = bundles[plot_id]
        stage1_output = stage1_outputs[plot_id]
        current = plot_table.copy().reset_index(drop=True)
        current["split_prob"] = split_probs[: len(current)]
        split_probs = split_probs[len(current) :]
        predictions = []
        for _, row in current.iterrows():
            if float(row.get("area_m2", 0.0)) < float(params.min_area_m2):
                predictions.append({"x": float(row["x"]), "y": float(row["y"]), "height_pred": float(row["height_pred"]), "score": 0.5})
                continue
            if float(row["split_prob"]) >= float(params.split_threshold):
                multi = _segment_multi_peak_predictions(bundle, stage1_output, row, params)
                if multi:
                    predictions.extend(multi)
                    continue
            predictions.append({"x": float(row["x"]), "y": float(row["y"]), "height_pred": float(row["height_pred"]), "score": float(row["geometry_score"])})
        pred_df = pd.DataFrame.from_records(predictions)
        if pred_df.empty:
            predictions_by_plot[plot_id] = pred_df
            continue
        pred_df = pred_df.sort_values(["score", "height_pred"], ascending=[False, False]).reset_index(drop=True)
        kept = []
        taken = np.zeros(len(pred_df), dtype=bool)
        xy = pred_df[["x", "y"]].to_numpy(dtype=np.float32)
        radius_sq = float(params.nms_radius_m) ** 2
        for i in range(len(pred_df)):
            if taken[i]:
                continue
            kept.append(pred_df.iloc[i][["x", "y", "height_pred", "score"]].to_dict())
            diff = xy - xy[i]
            dist_sq = np.sum(diff * diff, axis=1)
            taken |= dist_sq <= radius_sq
        predictions_by_plot[plot_id] = pd.DataFrame.from_records(kept)
    return predictions_by_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run watershed-fixed GNN and split-predictor variants on protocol A.")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs_ticketfinal_faixa_b")
    args = parser.parse_args()

    config = ExperimentConfig.from_files(*args.config)
    output_root = ensure_dir(Path(args.output_dir).resolve())
    ensure_dir(output_root / "metrics")
    ensure_dir(output_root / "predictions")
    save_config(config, output_root / "config.yaml")

    bundles = list_processed_plots(config.resolve_path(config.paths.processed_dir), plot_ids=config.data.plot_ids or None)
    splits = build_protocol_a_splits(list(bundles.keys()))
    stage1_outputs = {plot_id: WatershedStage1Generator().generate(bundle) for plot_id, bundle in bundles.items()}
    segment_tables = {plot_id: _build_segment_table_for_plot(bundle, stage1_outputs[plot_id]) for plot_id, bundle in bundles.items()}

    # GNN
    gnn_rows = []
    split_rows = []
    for split in splits:
        gnn_dir = ensure_dir(output_root / "predictions" / "watershed_fixo.segment_gnn" / f"fold_{split.fold_id:02d}")
        split_dir = ensure_dir(output_root / "predictions" / "watershed_fixo.split_predictor" / f"fold_{split.fold_id:02d}")

        train_graphs = []
        val_graphs = []
        test_graphs = []
        for plot_id in split.train_plots:
            table = segment_tables[plot_id]
            if table.empty:
                continue
            train_graphs.append(
                {
                    "plot_id": plot_id,
                    "table": table,
                    "x": _node_features(table),
                    "y": table["objectness"].to_numpy(dtype=np.float32),
                    "adj": _segment_adjacency(stage1_outputs[plot_id].segment_labels, table["segment_id"].astype(int).tolist()),
                }
            )
        for plot_id in split.val_plots:
            table = segment_tables[plot_id]
            if table.empty:
                continue
            val_graphs.append(
                {
                    "plot_id": plot_id,
                    "table": table,
                    "x": _node_features(table),
                    "y": table["objectness"].to_numpy(dtype=np.float32),
                    "adj": _segment_adjacency(stage1_outputs[plot_id].segment_labels, table["segment_id"].astype(int).tolist()),
                }
            )
        for plot_id in split.test_plots:
            table = segment_tables[plot_id]
            if table.empty:
                continue
            test_graphs.append(
                {
                    "plot_id": plot_id,
                    "table": table,
                    "x": _node_features(table),
                    "y": table["objectness"].to_numpy(dtype=np.float32),
                    "adj": _segment_adjacency(stage1_outputs[plot_id].segment_labels, table["segment_id"].astype(int).tolist()),
                }
            )

        model = _train_gnn(train_graphs, val_graphs, device=args.device)
        torch.save(model.state_dict(), gnn_dir / "best.pt")

        val_scores = {graph["plot_id"]: _score_graph(model, graph, device=args.device) for graph in val_graphs}
        val_tables = {graph["plot_id"]: graph["table"] for graph in val_graphs}
        best_params, best_scores = _calibrate_gnn(
            val_tables=val_tables,
            score_by_plot=val_scores,
            bundles=bundles,
            match_radius_m=config.evaluation.match_radius_m,
            height_tolerance_m=config.evaluation.height_tolerance_m,
        )
        write_json({"best_params": best_params, "best_scores": best_scores}, gnn_dir / "best_params.json")
        test_predictions = {}
        for graph in test_graphs:
            score = _score_graph(model, graph, device=args.device)
            pred = _apply_keep_predictions(graph["table"], score, threshold=best_params["threshold"], nms_radius_m=best_params["nms_radius_m"])
            pred.to_csv(gnn_dir / f"{graph['plot_id']}.csv", index=False)
            test_predictions[graph["plot_id"]] = pred
        fold_table, _ = _summarize_predictions(test_predictions, bundles=bundles, plot_ids=split.test_plots, match_radius_m=config.evaluation.match_radius_m, height_tolerance_m=config.evaluation.height_tolerance_m)
        fold_table["fold_id"] = split.fold_id
        fold_table["method"] = "watershed_fixo.segment_gnn"
        fold_table["protocol_id"] = PROTOCOL_A_ID
        fold_table["stage1"] = "watershed_fixo"
        fold_table["stage2"] = "segment_gnn"
        fold_table["labels"] = "field_center"
        gnn_rows.append(fold_table)

        # Split predictor
        train_table = pd.concat([segment_tables[plot_id] for plot_id in split.train_plots if not segment_tables[plot_id].empty], ignore_index=True)
        val_table = pd.concat([segment_tables[plot_id] for plot_id in split.val_plots if not segment_tables[plot_id].empty], ignore_index=True)
        test_table = pd.concat([segment_tables[plot_id] for plot_id in split.test_plots if not segment_tables[plot_id].empty], ignore_index=True)
        feature_cols = [
            "area_m2",
            "height_pred",
            "height_mean_m",
            "height_gap",
            "geometry_score",
            "elongation",
            "compactness",
            "rgb_std",
            "rgb_entropy",
            "peak_count",
            "peak_value",
        ]
        split_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        split_model.fit(train_table[feature_cols].fillna(0.0), train_table["split_flag"].astype(int))
        val_probs = split_model.predict_proba(val_table[feature_cols].fillna(0.0))[:, 1]
        best_split_params = SplitDetectionParams()
        best_split_scores = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
        for split_threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
            for peak_min_distance_m in [0.6, 0.8, 1.0, 1.2]:
                for nms_radius_m in [0.8, 1.0, 1.2, 1.5]:
                    for min_area_m2 in [4.0, 6.0, 8.0]:
                        params = SplitDetectionParams(
                            split_threshold=float(split_threshold),
                            peak_min_distance_m=float(peak_min_distance_m),
                            nms_radius_m=float(nms_radius_m),
                            min_area_m2=float(min_area_m2),
                        )
                        val_predictions = _apply_split_predictor(val_table, val_probs.copy(), stage1_outputs, bundles, params)
                        _, summary = _summarize_predictions(val_predictions, bundles=bundles, plot_ids=split.val_plots, match_radius_m=config.evaluation.match_radius_m, height_tolerance_m=config.evaluation.height_tolerance_m)
                        if summary["f1"] > best_split_scores["f1"] or (summary["f1"] == best_split_scores["f1"] and summary["recall"] > best_split_scores["recall"]):
                            best_split_params = params
                            best_split_scores = summary
        write_json({"best_params": asdict(best_split_params), "best_scores": best_split_scores}, split_dir / "best_params.json")
        test_probs = split_model.predict_proba(test_table[feature_cols].fillna(0.0))[:, 1]
        test_predictions = _apply_split_predictor(test_table, test_probs.copy(), stage1_outputs, bundles, best_split_params)
        for plot_id, pred in test_predictions.items():
            pred.to_csv(split_dir / f"{plot_id}.csv", index=False)
        fold_table, _ = _summarize_predictions(test_predictions, bundles=bundles, plot_ids=split.test_plots, match_radius_m=config.evaluation.match_radius_m, height_tolerance_m=config.evaluation.height_tolerance_m)
        fold_table["fold_id"] = split.fold_id
        fold_table["method"] = "watershed_fixo.split_predictor"
        fold_table["protocol_id"] = PROTOCOL_A_ID
        fold_table["stage1"] = "watershed_fixo"
        fold_table["stage2"] = "split_predictor"
        fold_table["labels"] = "field_center"
        split_rows.append(fold_table)

    gnn_metrics = pd.concat(gnn_rows, ignore_index=True)
    split_metrics = pd.concat(split_rows, ignore_index=True)
    gnn_summary = build_method_summary(gnn_metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    split_summary = build_method_summary(split_metrics, bootstrap_iterations=config.evaluation.bootstrap_iterations)
    gnn_metrics.to_csv(output_root / "predictions" / "watershed_fixo.segment_gnn" / "metrics_by_plot.csv", index=False)
    split_metrics.to_csv(output_root / "predictions" / "watershed_fixo.split_predictor" / "metrics_by_plot.csv", index=False)
    gnn_summary.to_csv(output_root / "predictions" / "watershed_fixo.segment_gnn" / "summary.csv", index=False)
    split_summary.to_csv(output_root / "predictions" / "watershed_fixo.split_predictor" / "summary.csv", index=False)

    all_summary = pd.concat([gnn_summary.assign(method_name="watershed_fixo.segment_gnn"), split_summary.assign(method_name="watershed_fixo.split_predictor")], ignore_index=True)
    all_summary.to_csv(output_root / "metrics" / "gnn_split_summary.csv", index=False)


if __name__ == "__main__":
    main()

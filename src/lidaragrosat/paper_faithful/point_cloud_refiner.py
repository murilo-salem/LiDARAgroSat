from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import Dataset

from .method import FaithfulMatchingParams, faithful_greedy_match, summarize_faithful_matches


@dataclass(slots=True)
class PointCloudRefinerConfig:
    source_mode: str = "union"
    point_radius_m: float = 2.5
    num_points: int = 256
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 6
    positive_match_radius_m: float = 1.5
    max_height_diff_m: float = 3.0


def deduplicate_proposals(proposals: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if proposals.empty:
        return proposals.copy()
    ordered = proposals.sort_values(["confidence", "height_pred"], ascending=[False, False]).reset_index(drop=True)
    xy = ordered[["x", "y"]].to_numpy(dtype=np.float64)
    taken = np.zeros(len(ordered), dtype=bool)
    kept: list[dict] = []
    radius_sq = float(radius_m) ** 2
    for index in range(len(ordered)):
        if taken[index]:
            continue
        kept.append(ordered.iloc[index].to_dict())
        diff = xy - xy[index]
        taken |= np.sum(diff * diff, axis=1) <= radius_sq
    return pd.DataFrame.from_records(kept)


def load_proposals_for_plot(
    plot_id: str,
    source_mode: str,
    adaptive_root: str | Path,
    watershed_root: str | Path,
    dedup_radius_m: float = 0.8,
) -> pd.DataFrame:
    adaptive_path = Path(adaptive_root) / f"{plot_id}.csv"
    watershed_path = Path(watershed_root) / f"{plot_id}.csv"
    frames: list[pd.DataFrame] = []

    if source_mode in {"adaptive", "union"} and adaptive_path.exists():
        adaptive = pd.read_csv(adaptive_path).copy()
        if not adaptive.empty:
            adaptive["confidence"] = adaptive["height_pred"].astype(float)
            adaptive["source_adaptive"] = 1.0
            adaptive["source_watershed"] = 0.0
            frames.append(adaptive[["x", "y", "height_pred", "confidence", "source_adaptive", "source_watershed"]])
    if source_mode in {"watershed", "union"} and watershed_path.exists():
        watershed = pd.read_csv(watershed_path).copy()
        if not watershed.empty:
            watershed["confidence"] = watershed.get("confidence", watershed["height_pred"]).astype(float)
            watershed["source_adaptive"] = 0.0
            watershed["source_watershed"] = 1.0
            frames.append(watershed[["x", "y", "height_pred", "confidence", "source_adaptive", "source_watershed"]])

    if not frames:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence", "source_adaptive", "source_watershed"])
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["source_both"] = ((merged["source_adaptive"] > 0) & (merged["source_watershed"] > 0)).astype(float)
    return deduplicate_proposals(merged, radius_m=dedup_radius_m)


def label_proposals(
    proposals: pd.DataFrame,
    reference: pd.DataFrame,
    positive_match_radius_m: float,
    max_height_diff_m: float,
) -> pd.DataFrame:
    labeled = proposals.copy()
    if labeled.empty:
        labeled["is_positive"] = []
        return labeled

    labeled["is_positive"] = 0
    labeled["is_hard_negative"] = 0
    labeled["offset_dx_m"] = 0.0
    labeled["offset_dy_m"] = 0.0
    labeled["height_target"] = np.nan

    if reference.empty:
        return labeled

    proposal_xy = labeled[["x", "y"]].to_numpy(dtype=np.float64)
    ref_xy = reference[["x", "y"]].to_numpy(dtype=np.float64)
    distances = np.sqrt(((proposal_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2))
    proposal_h = labeled["height_pred"].to_numpy(dtype=np.float64)
    ref_h = pd.to_numeric(reference.get("height"), errors="coerce").to_numpy(dtype=np.float64)
    cost = distances.copy()
    cost[distances > float(positive_match_radius_m)] = 1e6
    valid_height = np.isfinite(ref_h)[None, :] & np.isfinite(proposal_h)[:, None]
    height_delta = np.abs(proposal_h[:, None] - ref_h[None, :])
    cost[valid_height & (height_delta > float(max_height_diff_m))] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_indices: set[int] = set()
    for proposal_index, ref_index in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[proposal_index, ref_index] >= 1e6:
            continue
        matched_indices.add(proposal_index)
        labeled.at[proposal_index, "is_positive"] = 1
        labeled.at[proposal_index, "offset_dx_m"] = float(reference.iloc[ref_index]["x"]) - float(labeled.iloc[proposal_index]["x"])
        labeled.at[proposal_index, "offset_dy_m"] = float(reference.iloc[ref_index]["y"]) - float(labeled.iloc[proposal_index]["y"])
        if pd.notna(reference.iloc[ref_index].get("height")):
            labeled.at[proposal_index, "height_target"] = float(reference.iloc[ref_index]["height"])

    nearest_distance = distances.min(axis=1) if len(reference) else np.full(len(labeled), np.inf, dtype=np.float64)
    hard_negative_mask = (nearest_distance <= float(positive_match_radius_m) * 2.0) & (labeled["is_positive"].to_numpy(dtype=int) == 0)
    labeled.loc[hard_negative_mask, "is_hard_negative"] = 1
    return labeled


def sample_local_points(
    points_xyz: np.ndarray,
    center_x: float,
    center_y: float,
    radius_m: float,
    num_points: int,
    training: bool,
) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.zeros((num_points, 4), dtype=np.float32)
    dx = points_xyz[:, 0] - float(center_x)
    dy = points_xyz[:, 1] - float(center_y)
    radial_sq = dx * dx + dy * dy
    mask = radial_sq <= float(radius_m) ** 2
    local = points_xyz[mask]
    if len(local) == 0:
        return np.zeros((num_points, 4), dtype=np.float32)

    local_dx = local[:, 0] - float(center_x)
    local_dy = local[:, 1] - float(center_y)
    features = np.column_stack(
        [
            local_dx / max(float(radius_m), 1e-6),
            local_dy / max(float(radius_m), 1e-6),
            local[:, 2] / 35.0,
            np.sqrt(local_dx * local_dx + local_dy * local_dy) / max(float(radius_m), 1e-6),
        ]
    ).astype(np.float32)
    if len(features) >= num_points:
        if training:
            choice = np.random.choice(len(features), size=num_points, replace=False)
            features = features[choice]
        else:
            order = np.argsort(-local[:, 2], kind="mergesort")
            features = features[order[:num_points]]
    else:
        choice = np.random.choice(len(features), size=num_points, replace=True)
        features = features[choice]
    return features.astype(np.float32)


class ProposalPointDataset(Dataset):
    def __init__(
        self,
        points_by_plot: dict[str, np.ndarray],
        candidate_table: pd.DataFrame,
        config: PointCloudRefinerConfig,
        training: bool,
    ) -> None:
        self.points_by_plot = points_by_plot
        self.candidate_table = candidate_table.reset_index(drop=True).copy()
        self.config = config
        self.training = training

    def __len__(self) -> int:
        return len(self.candidate_table)

    def __getitem__(self, index: int) -> dict:
        row = self.candidate_table.iloc[index]
        points = self.points_by_plot[row["plot_id"]]
        point_cloud = sample_local_points(
            points_xyz=points,
            center_x=float(row["x"]),
            center_y=float(row["y"]),
            radius_m=self.config.point_radius_m,
            num_points=self.config.num_points,
            training=self.training,
        )
        tabular = np.array(
            [
                float(row.get("confidence", row.get("height_pred", 0.0))) / 35.0,
                float(row.get("height_pred", 0.0)) / 35.0,
                float(row.get("source_adaptive", 0.0)),
                float(row.get("source_watershed", 0.0)),
                float(row.get("source_both", 0.0)),
            ],
            dtype=np.float32,
        )
        height_target = row.get("height_target", np.nan)
        height_target = np.nan if height_target is None else float(height_target)
        return {
            "point_cloud": torch.from_numpy(point_cloud).transpose(0, 1),
            "tabular": torch.from_numpy(tabular),
            "is_positive": torch.tensor(float(row.get("is_positive", 0)), dtype=torch.float32),
            "is_hard_negative": torch.tensor(float(row.get("is_hard_negative", 0)), dtype=torch.float32),
            "offset": torch.tensor([float(row.get("offset_dx_m", 0.0)), float(row.get("offset_dy_m", 0.0))], dtype=torch.float32),
            "height_target": torch.tensor(float(0.0 if height_target != height_target else height_target), dtype=torch.float32),
            "height_mask": torch.tensor(float(height_target == height_target), dtype=torch.float32),
            "x": torch.tensor(float(row["x"]), dtype=torch.float32),
            "y": torch.tensor(float(row["y"]), dtype=torch.float32),
            "plot_id": row["plot_id"],
            "proposal_height": torch.tensor(float(row.get("height_pred", 0.0)), dtype=torch.float32),
        }


class PointSetRefiner(nn.Module):
    def __init__(self, point_feature_dim: int = 4, tabular_dim: int = 5) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Conv1d(point_feature_dim, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 + tabular_dim, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 96),
            nn.GELU(),
        )
        self.cls = nn.Linear(96, 1)
        self.offset = nn.Linear(96, 2)
        self.height = nn.Linear(96, 1)

    def forward(self, point_cloud: torch.Tensor, tabular: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.point_encoder(point_cloud)
        pooled = torch.max(encoded, dim=2).values
        fused = torch.cat([pooled, tabular], dim=1)
        trunk = self.trunk(fused)
        return {
            "cls": self.cls(trunk).squeeze(1),
            "offset": self.offset(trunk),
            "height": self.height(trunk).squeeze(1),
        }


class PointRefinerLoss(nn.Module):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, hard_negative_weight: float = 2.0) -> None:
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.hard_negative_weight = hard_negative_weight

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        targets = batch["is_positive"]
        logits = outputs["cls"]
        probs = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        weights = 1.0 + self.hard_negative_weight * batch["is_hard_negative"] * (1.0 - targets)
        cls_loss = (alpha_t * ((1.0 - p_t) ** self.focal_gamma) * ce * weights).mean()

        positive_mask = targets
        positive_count = torch.clamp(positive_mask.sum(), min=1.0)
        offset_loss = (torch.abs(outputs["offset"] - batch["offset"]).sum(dim=1) * positive_mask).sum() / positive_count
        height_mask = batch["height_mask"] * positive_mask
        height_count = torch.clamp(height_mask.sum(), min=1.0)
        height_loss = nn.functional.huber_loss(
            outputs["height"] * height_mask,
            batch["height_target"] * height_mask,
            reduction="sum",
            delta=1.0,
        ) / height_count
        total = cls_loss + offset_loss + 0.5 * height_loss
        metrics = {
            "loss_total": float(total.detach().cpu()),
            "loss_cls": float(cls_loss.detach().cpu()),
            "loss_offset": float(offset_loss.detach().cpu()),
            "loss_height": float(height_loss.detach().cpu()),
        }
        return total, metrics


def run_epoch(model, loader, optimizer, criterion, device: str, train: bool) -> dict[str, float]:
    model.train(train)
    metrics_accumulator: dict[str, list[float]] = {}
    for batch in loader:
        point_cloud = batch["point_cloud"].to(device)
        tabular = batch["tabular"].to(device)
        moved = {
            "is_positive": batch["is_positive"].to(device),
            "is_hard_negative": batch["is_hard_negative"].to(device),
            "offset": batch["offset"].to(device),
            "height_target": batch["height_target"].to(device),
            "height_mask": batch["height_mask"].to(device),
        }
        with torch.set_grad_enabled(train):
            outputs = model(point_cloud=point_cloud, tabular=tabular)
            loss, metrics = criterion(outputs, moved)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        for key, value in metrics.items():
            metrics_accumulator.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in metrics_accumulator.items()}


def score_dataset(model, dataset: ProposalPointDataset, batch_size: int, device: str) -> pd.DataFrame:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                point_cloud=batch["point_cloud"].to(device),
                tabular=batch["tabular"].to(device),
            )
            scores = torch.sigmoid(outputs["cls"]).cpu().numpy()
            offsets = outputs["offset"].cpu().numpy()
            heights = outputs["height"].cpu().numpy()
            proposal_heights = batch["proposal_height"].cpu().numpy()
            for idx in range(len(scores)):
                rows.append(
                    {
                        "plot_id": batch["plot_id"][idx],
                        "x": float(batch["x"][idx].item()),
                        "y": float(batch["y"][idx].item()),
                        "score": float(scores[idx]),
                        "x_refined": float(batch["x"][idx].item() + offsets[idx, 0]),
                        "y_refined": float(batch["y"][idx].item() + offsets[idx, 1]),
                        "height_pred": float(heights[idx]) if np.isfinite(heights[idx]) else float(proposal_heights[idx]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def postprocess_predictions(
    predictions: pd.DataFrame,
    classification_threshold: float,
    nms_radius_m: float,
    min_height_m: float,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence"])
    filtered = predictions[
        (predictions["score"] >= float(classification_threshold))
        & (predictions["height_pred"] >= float(min_height_m))
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence"])
    ordered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
    kept: list[dict] = []
    taken = np.zeros(len(ordered), dtype=bool)
    xy = ordered[["x_refined", "y_refined"]].to_numpy(dtype=np.float64)
    radius_sq = float(nms_radius_m) ** 2
    for index in range(len(ordered)):
        if taken[index]:
            continue
        row = ordered.iloc[index]
        kept.append(
            {
                "x": float(row["x_refined"]),
                "y": float(row["y_refined"]),
                "height_pred": float(row["height_pred"]),
                "confidence": float(row["score"]),
            }
        )
        diff = xy - xy[index]
        taken |= np.sum(diff * diff, axis=1) <= radius_sq
    return pd.DataFrame.from_records(kept)


def tune_postprocess(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    threshold_grid: list[float],
    nms_radius_grid_m: list[float],
    min_height_grid_m: list[float],
    matching_params: FaithfulMatchingParams,
) -> tuple[dict[str, float], dict[str, float]]:
    best_params = {"classification_threshold": threshold_grid[0], "nms_radius_m": nms_radius_grid_m[0], "min_height_m": min_height_grid_m[0]}
    best_metrics = {"f1": -1.0}
    for threshold in threshold_grid:
        for nms_radius in nms_radius_grid_m:
            for min_height in min_height_grid_m:
                detections = postprocess_predictions(predictions, threshold, nms_radius, min_height)
                metrics = summarize_faithful_matches(faithful_greedy_match(detections, reference, matching_params))
                if metrics["f1"] > best_metrics["f1"]:
                    best_params = {
                        "classification_threshold": float(threshold),
                        "nms_radius_m": float(nms_radius),
                        "min_height_m": float(min_height),
                    }
                    best_metrics = metrics
    return best_params, best_metrics

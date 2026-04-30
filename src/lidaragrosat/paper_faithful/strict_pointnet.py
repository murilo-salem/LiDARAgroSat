from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree, distance_matrix
from torch import nn
from torch.utils.data import Dataset

from .method import FaithfulMatchingParams, faithful_greedy_match, local_maxima_filter_3d, summarize_faithful_matches


@dataclass(slots=True)
class StrictPointNetConfig:
    proposal_window_size_m: float = 0.8
    proposal_height_threshold_m: float = 3.0
    neighborhood_radius_m: float = 2.5
    n_points: int = 512
    positive_threshold_m: float = 2.0
    feature_mode: str = "xyz"
    loss_kind: str = "bce"
    batch_size: int = 128
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    pos_weight: float = 5.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    dropout: float = 0.3
    cls_threshold_grid: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)
    nms_radius_grid_m: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 1.8)
    seed: int = 13


def generate_strict_proposals(points_xyz: np.ndarray, config: StrictPointNetConfig) -> pd.DataFrame:
    return local_maxima_filter_3d(
        points_xyz=points_xyz,
        window_size_m=float(config.proposal_window_size_m),
        height_threshold_m=float(config.proposal_height_threshold_m),
    )


def extract_neighborhood(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    radius: float = 2.5,
    n_points: int = 512,
    training: bool = False,
    rng: np.random.Generator | None = None,
    tree: cKDTree | None = None,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    center = np.asarray(center_xyz, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    local_tree = tree if tree is not None else cKDTree(points)
    indices = local_tree.query_ball_point(center, r=float(radius))
    local = points[indices] - center[None, :]
    if len(local) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    generator = rng if rng is not None else np.random.default_rng()
    if len(local) >= n_points:
        choice = generator.choice(len(local), size=n_points, replace=False)
    else:
        choice = generator.choice(len(local), size=n_points, replace=True)
    return local[choice].astype(np.float32)


def label_proposals_nearest(
    proposals: pd.DataFrame,
    ground_truth_xy: np.ndarray,
    pos_threshold: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(proposals) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    if len(ground_truth_xy) == 0:
        return np.zeros(len(proposals), dtype=np.int64), np.zeros((len(proposals), 2), dtype=np.float32)
    proposal_xy = proposals[["x", "y"]].to_numpy(dtype=np.float32)
    dists = distance_matrix(proposal_xy, np.asarray(ground_truth_xy, dtype=np.float32))
    labels = np.zeros(len(proposals), dtype=np.int64)
    offsets = np.zeros((len(proposals), 2), dtype=np.float32)
    for index, row in enumerate(dists):
        min_index = int(np.argmin(row))
        if float(row[min_index]) <= float(pos_threshold):
            labels[index] = 1
            offsets[index] = ground_truth_xy[min_index] - proposal_xy[index]
    return labels, offsets


def build_labeled_proposal_table(
    plot_id: str,
    points_xyz: np.ndarray,
    reference: pd.DataFrame,
    config: StrictPointNetConfig,
) -> pd.DataFrame:
    proposals = generate_strict_proposals(points_xyz, config)
    proposals = proposals.copy()
    proposals["plot_id"] = plot_id
    ground_truth_xy = reference[["x", "y"]].to_numpy(dtype=np.float32) if len(reference) else np.zeros((0, 2), dtype=np.float32)
    labels, offsets = label_proposals_nearest(
        proposals=proposals,
        ground_truth_xy=ground_truth_xy,
        pos_threshold=float(config.positive_threshold_m),
    )
    proposals["label"] = labels.astype(np.int64)
    proposals["offset_dx"] = offsets[:, 0].astype(np.float32)
    proposals["offset_dy"] = offsets[:, 1].astype(np.float32)
    proposals["proposal_z"] = proposals["height_pred"].astype(np.float32)
    proposals["confidence"] = proposals["height_pred"].astype(np.float32)
    return proposals


class StrictProposalDataset(Dataset):
    def __init__(
        self,
        points_by_plot: dict[str, np.ndarray],
        proposal_table: pd.DataFrame,
        config: StrictPointNetConfig,
        training: bool,
    ) -> None:
        self.points_by_plot = points_by_plot
        self.trees_by_plot = {plot_id: cKDTree(np.asarray(points[:, :3], dtype=np.float32)) for plot_id, points in points_by_plot.items()}
        self.proposal_table = proposal_table.reset_index(drop=True).copy()
        self.config = config
        self.training = training
        self.rng = np.random.default_rng(int(config.seed))
        self.neighbor_indices_by_row: list[np.ndarray] = []
        for _, row in self.proposal_table.iterrows():
            plot_id = row["plot_id"]
            center_xyz = np.array([float(row["x"]), float(row["y"]), float(row["proposal_z"])], dtype=np.float32)
            indices = self.trees_by_plot[plot_id].query_ball_point(center_xyz, r=float(self.config.neighborhood_radius_m))
            self.neighbor_indices_by_row.append(np.asarray(indices, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.proposal_table)

    def __getitem__(self, index: int) -> dict:
        row = self.proposal_table.iloc[index]
        center_xyz = np.array([float(row["x"]), float(row["y"]), float(row["proposal_z"])], dtype=np.float32)
        indices = self.neighbor_indices_by_row[index]
        if len(indices) == 0:
            sampled = np.zeros((int(self.config.n_points), self.points_by_plot[row["plot_id"]].shape[1]), dtype=np.float32)
            xyz = np.zeros((int(self.config.n_points), 3), dtype=np.float32)
        else:
            local_raw = self.points_by_plot[row["plot_id"]][indices].astype(np.float32)
            if len(local_raw) >= int(self.config.n_points):
                choice = self.rng.choice(len(local_raw), size=int(self.config.n_points), replace=False)
            else:
                choice = self.rng.choice(len(local_raw), size=int(self.config.n_points), replace=True)
            sampled = local_raw[choice].astype(np.float32)
            xyz = sampled[:, :3] - center_xyz[None, :]
        offset = np.array([float(row["offset_dx"]), float(row["offset_dy"])], dtype=np.float32)

        if self.training:
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            cos_t = np.cos(theta, dtype=np.float32)
            sin_t = np.sin(theta, dtype=np.float32)
            rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            xyz_xy = xyz[:, :2] @ rotation.T
            xyz = xyz.copy()
            xyz[:, :2] = xyz_xy
            offset = rotation @ offset
            xyz = xyz + self.rng.normal(0.0, 0.01, size=xyz.shape).astype(np.float32)

        if self.config.feature_mode == "xyz":
            neighborhood = xyz.astype(np.float32)
        elif self.config.feature_mode == "multifeature":
            if sampled.shape[1] < 6:
                raise ValueError("multifeature mode requires points with columns [x, y, z, intensity, return_number, number_of_returns].")
            intensity = sampled[:, 3:4].astype(np.float32)
            return_number = sampled[:, 4:5].astype(np.float32)
            number_of_returns = sampled[:, 5:6].astype(np.float32)
            z_norm = xyz[:, 2:3] / float(self.config.neighborhood_radius_m)
            dist_xy = np.linalg.norm(xyz[:, :2], axis=1, keepdims=True).astype(np.float32) / float(self.config.neighborhood_radius_m)
            return_ratio = return_number / (number_of_returns + 1e-6)
            neighborhood = np.concatenate([xyz, z_norm, dist_xy, intensity, return_ratio], axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported feature_mode: {self.config.feature_mode}")

        return {
            "points": torch.from_numpy(neighborhood),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "offset": torch.from_numpy(offset),
            "plot_id": row["plot_id"],
            "x": torch.tensor(float(row["x"]), dtype=torch.float32),
            "y": torch.tensor(float(row["y"]), dtype=torch.float32),
            "proposal_z": torch.tensor(float(row["proposal_z"]), dtype=torch.float32),
        }


class LocalPointNet(nn.Module):
    def __init__(self, input_dim: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(int(input_dim), 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.point_mlp(x)
        x = x.max(dim=2).values
        x = self.global_mlp(x)
        cls = self.cls_head(x).squeeze(-1)
        reg = self.reg_head(x)
        return cls, reg


class FocalBinaryLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )
        loss = alpha_t * torch.pow(1.0 - pt, self.gamma) * bce
        return loss.mean()


class CombinedPointNetLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 5.0,
        alpha: float = 1.0,
        loss_kind: str = "bce",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.loss_kind = str(loss_kind)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        self.focal_loss = FocalBinaryLoss(alpha=float(focal_alpha), gamma=float(focal_gamma))

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_pred: torch.Tensor,
        labels: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.loss_kind == "bce":
            cls_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(cls_logits.device))(cls_logits, labels.float())
        elif self.loss_kind == "focal":
            cls_loss = self.focal_loss(cls_logits, labels.float())
        else:
            raise ValueError(f"Unsupported loss_kind: {self.loss_kind}")
        mask = labels == 1
        if mask.sum() > 0:
            reg_loss = nn.SmoothL1Loss()(reg_pred[mask], offsets[mask])
        else:
            reg_loss = torch.zeros((), device=cls_logits.device, dtype=cls_logits.dtype)
        total = cls_loss + self.alpha * reg_loss
        return total, {
            "loss_total": float(total.detach().cpu()),
            "loss_cls": float(cls_loss.detach().cpu()),
            "loss_reg": float(reg_loss.detach().cpu()),
        }


def run_pointnet_epoch(
    model: LocalPointNet,
    loader,
    optimizer,
    criterion: CombinedPointNetLoss,
    device: str,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    metric_lists: dict[str, list[float]] = {}
    for batch in loader:
        points = batch["points"].to(device)
        labels = batch["label"].to(device)
        offsets = batch["offset"].to(device)
        with torch.set_grad_enabled(train):
            cls_logits, reg_pred = model(points)
            loss, metrics = criterion(cls_logits, reg_pred, labels, offsets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for key, value in metrics.items():
            metric_lists.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in metric_lists.items()}


def score_pointnet_dataset(
    model: LocalPointNet,
    dataset: StrictProposalDataset,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            cls_logits, reg_pred = model(batch["points"].to(device))
            scores = torch.sigmoid(cls_logits).cpu().numpy()
            offsets = reg_pred.cpu().numpy()
            for idx in range(len(scores)):
                x = float(batch["x"][idx].item())
                y = float(batch["y"][idx].item())
                rows.append(
                    {
                        "plot_id": batch["plot_id"][idx],
                        "x": x,
                        "y": y,
                        "proposal_z": float(batch["proposal_z"][idx].item()),
                        "score": float(scores[idx]),
                        "x_refined": float(x + offsets[idx, 0]),
                        "y_refined": float(y + offsets[idx, 1]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def nms_2d(points_xy: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    kept: list[int] = []
    suppressed: set[int] = set()
    tree = cKDTree(points_xy)
    for idx in order.tolist():
        if idx in suppressed:
            continue
        kept.append(idx)
        neighbors = tree.query_ball_point(points_xy[idx], r=float(radius))
        suppressed.update(int(neighbor) for neighbor in neighbors)
    return np.asarray(kept, dtype=np.int64)


def predict_and_refine(
    scored: pd.DataFrame,
    cls_threshold: float,
    nms_radius: float,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence"])
    keep = scored["score"].to_numpy(dtype=np.float32) > float(cls_threshold)
    filtered = scored.loc[keep].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["x", "y", "height_pred", "confidence"])
    points_xy = filtered[["x_refined", "y_refined"]].to_numpy(dtype=np.float32)
    scores = filtered["score"].to_numpy(dtype=np.float32)
    keep_indices = nms_2d(points_xy, scores, radius=float(nms_radius))
    selected = filtered.iloc[keep_indices].copy()
    return pd.DataFrame(
        {
            "x": selected["x_refined"].to_numpy(dtype=np.float32),
            "y": selected["y_refined"].to_numpy(dtype=np.float32),
            "height_pred": selected["proposal_z"].to_numpy(dtype=np.float32),
            "confidence": selected["score"].to_numpy(dtype=np.float32),
        }
    ).reset_index(drop=True)


def tune_pointnet_postprocess(
    scored: pd.DataFrame,
    reference_by_plot: dict[str, pd.DataFrame],
    matching_params: FaithfulMatchingParams,
    cls_threshold_grid: tuple[float, ...],
    nms_radius_grid_m: tuple[float, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    best_params = {"cls_threshold": float(cls_threshold_grid[0]), "nms_radius_m": float(nms_radius_grid_m[0])}
    best_mean_f1 = -1.0
    best_per_plot = pd.DataFrame()
    for cls_threshold in cls_threshold_grid:
        for nms_radius_m in nms_radius_grid_m:
            rows: list[dict] = []
            for plot_id, reference in reference_by_plot.items():
                current = scored[scored["plot_id"] == plot_id].copy()
                predictions = predict_and_refine(current, cls_threshold=float(cls_threshold), nms_radius=float(nms_radius_m))
                metrics = summarize_faithful_matches(
                    faithful_greedy_match(
                        predictions=predictions,
                        reference=reference,
                        params=matching_params,
                    )
                )
                metrics["plot_id"] = plot_id
                rows.append(metrics)
            per_plot = pd.DataFrame.from_records(rows).sort_values("plot_id").reset_index(drop=True)
            mean_f1 = float(per_plot["f1"].mean())
            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_params = {"cls_threshold": float(cls_threshold), "nms_radius_m": float(nms_radius_m)}
                best_per_plot = per_plot
    return best_params, best_per_plot


def summarize_proposals_against_reference(
    proposals_by_plot: dict[str, pd.DataFrame],
    reference_by_plot: dict[str, pd.DataFrame],
    matching_params: FaithfulMatchingParams,
) -> pd.DataFrame:
    rows: list[dict] = []
    for plot_id, proposals in proposals_by_plot.items():
        metrics = summarize_faithful_matches(
            faithful_greedy_match(
                predictions=proposals,
                reference=reference_by_plot[plot_id],
                params=matching_params,
            )
        )
        metrics["plot_id"] = plot_id
        metrics["n_proposals"] = int(len(proposals))
        metrics["n_reference"] = int(len(reference_by_plot[plot_id]))
        metrics["proposal_to_tree_ratio"] = float(len(proposals) / max(len(reference_by_plot[plot_id]), 1))
        rows.append(metrics)
    return pd.DataFrame.from_records(rows).sort_values("plot_id").reset_index(drop=True)

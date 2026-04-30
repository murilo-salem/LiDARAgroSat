from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Stage2Loss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def _focal_binary(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = self._focal_binary(outputs["cls"], batch["objectness"])
        positive = batch["objectness"]
        positive_count = torch.clamp(positive.sum(), min=1.0)
        offset_loss = (torch.abs(outputs["offset"] - batch["offset"]).sum(dim=1) * positive).sum() / positive_count
        height_loss = (
            nn.functional.huber_loss(outputs["height"], batch["delta_h"], reduction="none", delta=1.0) * positive * batch["height_mask"]
        ).sum() / torch.clamp((positive * batch["height_mask"]).sum(), min=1.0)
        split_loss = nn.functional.binary_cross_entropy_with_logits(outputs["split"], batch["split_flag"])
        total = cls_loss + offset_loss + 0.5 * height_loss + 0.2 * split_loss
        return total, {
            "loss_total": float(total.detach().cpu()),
            "loss_cls": float(cls_loss.detach().cpu()),
            "loss_offset": float(offset_loss.detach().cpu()),
            "loss_height": float(height_loss.detach().cpu()),
            "loss_split": float(split_loss.detach().cpu()),
        }


def _move_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def run_stage2_epoch(model, loader, optimizer, criterion: Stage2Loss, device: str, model_kind: str, train: bool) -> dict[str, float]:
    model.train(train)
    metrics: dict[str, list[float]] = {}
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(train):
            if model_kind == "raster":
                outputs = model(batch["raster"], batch["geom_features"])
            elif model_kind == "point":
                outputs = model(batch["point_cloud"], batch["geom_features"])
            else:
                outputs = model(batch["raster"], batch["point_cloud"], batch["geom_features"])
            loss, current = criterion(outputs, batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        for key, value in current.items():
            metrics.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def fit_stage2_model(
    model,
    train_loader,
    val_loader,
    model_kind: str,
    device: str,
    checkpoint_dir: str | Path,
    epochs: int = 20,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    patience: int = 6,
    validation_callback=None,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = Stage2Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    wait = 0
    history: list[dict] = []
    for epoch in range(epochs):
        train_metrics = run_stage2_epoch(model, train_loader, optimizer, criterion, device=device, model_kind=model_kind, train=True)
        val_metrics = run_stage2_epoch(model, val_loader, optimizer, criterion, device=device, model_kind=model_kind, train=False)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        score = -val_metrics["loss_total"]
        if validation_callback is not None:
            payload = validation_callback(model)
            record.update({f"val_{key}": value for key, value in payload.items()})
            score = payload.get("f1", score)
        history.append(record)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_dir / "best.pt")
            wait = 0
        else:
            wait += 1
        scheduler.step()
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_score": float(best_score), "history": history}

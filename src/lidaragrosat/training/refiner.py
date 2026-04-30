from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class CandidateRefinerLoss(nn.Module):
    def __init__(
        self,
        lambda_offset: float = 1.0,
        lambda_height: float = 0.5,
        lambda_species: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        hard_negative_cls_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_offset = lambda_offset
        self.lambda_height = lambda_height
        self.lambda_species = lambda_species
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.hard_negative_cls_weight = hard_negative_cls_weight

    def _focal_bce(self, logits: torch.Tensor, targets: torch.Tensor, hard_negative_mask: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        weights = 1.0 + self.hard_negative_cls_weight * hard_negative_mask * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.focal_gamma) * ce_loss * weights
        return loss.mean()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        cls_loss = self._focal_bce(outputs["cls"], batch["is_positive"], batch["is_hard_negative"])
        total = cls_loss
        metrics = {"loss_cls": float(cls_loss.detach().cpu())}

        positive_mask = batch["is_positive"]
        positive_count = torch.clamp(positive_mask.sum(), min=1.0)

        offset_loss = (torch.abs(outputs["offset"] - batch["offset"]).sum(dim=1) * positive_mask).sum() / positive_count
        total = total + self.lambda_offset * offset_loss
        metrics["loss_offset"] = float(offset_loss.detach().cpu())

        height_mask = batch["height_mask"] * positive_mask
        height_count = torch.clamp(height_mask.sum(), min=1.0)
        height_loss = nn.functional.huber_loss(
            outputs["height"] * height_mask,
            batch["height_target"] * height_mask,
            reduction="sum",
            delta=1.0,
        ) / height_count
        total = total + self.lambda_height * height_loss
        metrics["loss_height"] = float(height_loss.detach().cpu())

        if "species" in outputs:
            species_count = torch.clamp(batch["species_mask"].sum(), min=1.0)
            species_loss = nn.functional.cross_entropy(
                outputs["species"],
                batch["species_target"],
                ignore_index=-1,
                reduction="none",
            )
            species_loss = (species_loss * batch["species_mask"]).sum() / species_count
            total = total + self.lambda_species * species_loss
            metrics["loss_species"] = float(species_loss.detach().cpu())

        metrics["loss_total"] = float(total.detach().cpu())
        return total, metrics


def run_refiner_epoch(model, loader, optimizer, criterion, device: str, train: bool) -> dict[str, float]:
    metrics_accumulator: dict[str, list[float]] = {}
    model.train(train)
    autocast_enabled = bool(device.startswith("cuda"))
    scaler = getattr(run_refiner_epoch, "_scaler", None)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
        setattr(run_refiner_epoch, "_scaler", scaler)

    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                outputs = model(
                    rgb=batch["rgb"],
                    structural=batch["structural"],
                    source_features=batch["source_features"],
                    tabular_features=batch.get("tabular_features"),
                    point_profile=batch.get("point_profile"),
                )
                loss, metrics = criterion(outputs, batch)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        for key, value in metrics.items():
            metrics_accumulator.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in metrics_accumulator.items()}


def fit_candidate_refiner(
    model,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
    checkpoint_dir: str | Path,
    validation_callback=None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    hard_negative_cls_weight: float = 2.0,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CandidateRefinerLoss(
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        hard_negative_cls_weight=hard_negative_cls_weight,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(epochs):
        train_metrics = run_refiner_epoch(model, train_loader, optimizer, criterion, device=device, train=True)
        val_metrics = run_refiner_epoch(model, val_loader, optimizer, criterion, device=device, train=False)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        validation_score = -val_metrics.get("loss_total", 0.0)
        if validation_callback is not None:
            payload = validation_callback(model)
            record.update({f"val_{key}": value for key, value in payload.items()})
            validation_score = payload.get("f1", validation_score)
        history.append(record)
        if validation_score > best_score:
            best_score = validation_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        scheduler.step()
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return {"best_score": float(best_score), "history": history}

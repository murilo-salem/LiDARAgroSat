from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from lidaragrosat.training.losses import DetectionLoss


def _to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(model, loader, optimizer, criterion, device: str, train: bool) -> dict[str, float]:
    import torch

    metrics_accumulator: dict[str, list[float]] = {}
    model.train(train)

    autocast_enabled = bool(device.startswith("cuda"))
    scaler = getattr(run_epoch, "_scaler", None)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled)
        setattr(run_epoch, "_scaler", scaler)

    accumulation_steps = max(1, getattr(criterion, "accumulation_steps", 1))
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                outputs = model(
                    rgb=batch["rgb"],
                    lidar=batch["lidar"],
                    chm=batch.get("chm_features"),
                )
                loss, loss_metrics = criterion(outputs, batch)
            if train:
                scaler.scale(loss / accumulation_steps).backward()
                is_update_step = ((step + 1) % accumulation_steps == 0) or (step + 1 == len(loader))
                if is_update_step:
                    scaler.unscale_(optimizer)
                    clip_norm = float(getattr(criterion, "gradient_clip_norm", 1.0))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        for metric_name, metric_value in loss_metrics.items():
            metrics_accumulator.setdefault(metric_name, []).append(metric_value)

    return {name: float(np.mean(values)) for name, values in metrics_accumulator.items()}


def fit_model(
    model,
    train_loader,
    val_loader,
    config,
    device: str,
    checkpoint_dir: str | Path,
    validation_callback=None,
) -> dict:
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8)

    criterion = DetectionLoss(
        lambda_heatmap=config.training.lambda_heatmap,
        lambda_offset=config.training.lambda_offset,
        lambda_height=config.training.lambda_height,
        lambda_distance=config.training.lambda_distance,
        lambda_peakness=config.training.lambda_peakness,
        lambda_species=config.training.lambda_species,
        deep_supervision_weight=config.training.deep_supervision_weight,
        hard_negative_pixel_weight=config.training.hard_negative_pixel_weight,
        difficulty_positive_weight=config.training.difficulty_positive_weight,
    )
    criterion.accumulation_steps = max(1, config.training.accumulation_steps)
    criterion.gradient_clip_norm = float(config.training.gradient_clip_norm)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(config.training.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device=device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device=device, train=False)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}

        checkpoint_metric = config.training.checkpoint_metric
        if checkpoint_metric == "loss_total":
            validation_score = -val_metrics.get("loss_total", 0.0)
        else:
            validation_score = -np.inf
        should_run_callback = (
            validation_callback is not None
            and (
                epoch % max(1, config.training.validation_interval_epochs) == 0
                or epoch == config.training.epochs - 1
            )
        )
        if should_run_callback:
            validation_payload = validation_callback(model)
            record.update({f"val_{key}": value for key, value in validation_payload.items()})
            validation_score = validation_payload.get(checkpoint_metric, validation_score)

        history.append(record)
        if validation_score > best_score:
            best_score = validation_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.get("loss_total", 0.0))
        else:
            scheduler.step()

        if (
            epoch + 1 >= max(1, config.training.min_epochs_before_stop)
            and patience_counter >= config.training.patience
        ):
            break

    model.load_state_dict(best_state)
    return {"best_score": float(best_score), "history": history}

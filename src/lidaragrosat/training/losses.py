from __future__ import annotations

import torch
from torch import nn


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    weighted = alpha_t * loss
    if weights is not None:
        denominator = torch.clamp(weights.sum(), min=1.0)
        return (weighted * weights).sum() / denominator
    return weighted.mean()


class DetectionLoss(nn.Module):
    def __init__(
        self,
        lambda_heatmap: float,
        lambda_offset: float,
        lambda_height: float,
        lambda_distance: float = 0.0,
        lambda_peakness: float = 0.0,
        lambda_species: float = 0.0,
        deep_supervision_weight: float = 0.0,
        hard_negative_pixel_weight: float = 0.0,
        difficulty_positive_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_heatmap = lambda_heatmap
        self.lambda_offset = lambda_offset
        self.lambda_height = lambda_height
        self.lambda_distance = lambda_distance
        self.lambda_peakness = lambda_peakness
        self.lambda_species = lambda_species
        self.deep_supervision_weight = deep_supervision_weight
        self.hard_negative_pixel_weight = hard_negative_pixel_weight
        self.difficulty_positive_weight = difficulty_positive_weight

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        supervision_mask = batch.get("supervision_mask")
        if supervision_mask is None:
            supervision_mask = torch.ones_like(batch["heatmap"])
        hard_negative_weights = 1.0 + self.hard_negative_pixel_weight * batch["hard_negative_mask"] * (1.0 - batch["heatmap"])
        difficulty_weights = 1.0 + self.difficulty_positive_weight * batch.get("difficulty", torch.zeros_like(batch["heatmap"]))
        heatmap_weights = supervision_mask * hard_negative_weights * difficulty_weights
        heatmap_loss = sigmoid_focal_loss(outputs["heatmap"], batch["heatmap"], weights=heatmap_weights)
        total = self.lambda_heatmap * heatmap_loss
        metrics = {"loss_heatmap": float(heatmap_loss.detach().cpu())}

        positive_mask = batch["positive_mask"] * supervision_mask
        positive_count = torch.clamp(positive_mask.sum(), min=1.0)

        if "offset" in outputs:
            offset_l1 = torch.abs(outputs["offset"] - batch["offset"]) * positive_mask
            offset_loss = offset_l1.sum() / positive_count
            total = total + self.lambda_offset * offset_loss
            metrics["loss_offset"] = float(offset_loss.detach().cpu())

        if "height" in outputs:
            height_mask = batch["height_mask"] * positive_mask
            height_count = torch.clamp(height_mask.sum(), min=1.0)
            height_loss = nn.functional.huber_loss(
                outputs["height"] * height_mask,
                batch["height"] * height_mask,
                reduction="sum",
                delta=1.0,
            ) / height_count
            total = total + self.lambda_height * height_loss
            metrics["loss_height"] = float(height_loss.detach().cpu())

        if "distance" in outputs:
            distance_weights = supervision_mask * (1.0 + 0.5 * batch.get("difficulty", torch.zeros_like(batch["heatmap"])))
            distance_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["distance"],
                batch["distance"],
                reduction="none",
            )
            distance_denominator = torch.clamp(distance_weights.sum(), min=1.0)
            distance_loss = (distance_loss * distance_weights).sum() / distance_denominator
            total = total + self.lambda_distance * distance_loss
            metrics["loss_distance"] = float(distance_loss.detach().cpu())

        if "peakness" in outputs:
            peakness_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["peakness"],
                batch["peakness"],
                reduction="none",
            )
            peakness_loss = (peakness_loss * supervision_mask).sum() / torch.clamp(supervision_mask.sum(), min=1.0)
            total = total + self.lambda_peakness * peakness_loss
            metrics["loss_peakness"] = float(peakness_loss.detach().cpu())

        if "species" in outputs:
            species_mask = batch["species_mask"].squeeze(1) * supervision_mask.squeeze(1)
            species_count = torch.clamp(species_mask.sum(), min=1.0)
            species_loss = nn.functional.cross_entropy(
                outputs["species"],
                batch["species_target"],
                ignore_index=-1,
                reduction="none",
            )
            species_loss = (species_loss * species_mask).sum() / species_count
            total = total + self.lambda_species * species_loss
            metrics["loss_species"] = float(species_loss.detach().cpu())

        aux_outputs = outputs.get("aux_heatmaps", [])
        if aux_outputs:
            aux_loss_total = torch.zeros((), device=total.device)
            for aux_output in aux_outputs:
                aux_target = nn.functional.adaptive_max_pool2d(batch["heatmap"], output_size=aux_output.shape[-2:])
                aux_hard_negatives = nn.functional.adaptive_max_pool2d(
                    batch["hard_negative_mask"],
                    output_size=aux_output.shape[-2:],
                )
                aux_supervision = nn.functional.adaptive_max_pool2d(supervision_mask, output_size=aux_output.shape[-2:])
                aux_difficulty = nn.functional.adaptive_max_pool2d(
                    batch.get("difficulty", torch.zeros_like(batch["heatmap"])),
                    output_size=aux_output.shape[-2:],
                )
                aux_weights = aux_supervision * (1.0 + self.hard_negative_pixel_weight * aux_hard_negatives * (1.0 - aux_target)) * (1.0 + self.difficulty_positive_weight * aux_difficulty)
                aux_loss_total = aux_loss_total + sigmoid_focal_loss(aux_output, aux_target, weights=aux_weights)
            aux_loss_total = aux_loss_total / max(len(aux_outputs), 1)
            total = total + self.deep_supervision_weight * aux_loss_total
            metrics["loss_aux_heatmap"] = float(aux_loss_total.detach().cpu())

        metrics["loss_total"] = float(total.detach().cpu())
        return total, metrics

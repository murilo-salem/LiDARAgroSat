from __future__ import annotations

import torch
from torch import nn

from lidaragrosat.models.blocks import ResidualConvBlock, SimpleEncoder


class CandidateRefiner(nn.Module):
    def __init__(
        self,
        rgb_in_channels: int,
        structural_in_channels: int,
        stage_channels: list[int],
        normalization: str = "group",
        species_num_classes: int = 0,
        tabular_dim: int = 6,
        point_profile_dim: int = 0,
    ) -> None:
        super().__init__()
        self.rgb_encoder = SimpleEncoder(rgb_in_channels, stage_channels, norm=normalization)
        self.structural_encoder = SimpleEncoder(structural_in_channels, stage_channels, norm=normalization)
        self.point_profile_encoder = None
        point_embedding_dim = 0
        if point_profile_dim > 0:
            self.point_profile_encoder = nn.Sequential(
                nn.Linear(point_profile_dim, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
            point_embedding_dim = 32
        embedding_dim = 2 * stage_channels[-1] + 2 + tabular_dim + point_embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.cls = nn.Linear(128, 1)
        self.offset = nn.Linear(128, 2)
        self.height = nn.Linear(128, 1)
        self.species = nn.Linear(128, species_num_classes) if species_num_classes > 0 else None

    def forward(
        self,
        rgb: torch.Tensor,
        structural: torch.Tensor,
        source_features: torch.Tensor,
        tabular_features: torch.Tensor | None = None,
        point_profile: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rgb_features = self.rgb_encoder(rgb)[-1]
        structural_features = self.structural_encoder(structural)[-1]
        rgb_embedding = torch.nn.functional.adaptive_avg_pool2d(rgb_features, output_size=1).flatten(1)
        structural_embedding = torch.nn.functional.adaptive_avg_pool2d(structural_features, output_size=1).flatten(1)
        parts = [rgb_embedding, structural_embedding, source_features]
        if tabular_features is not None:
            parts.append(tabular_features)
        if point_profile is not None and self.point_profile_encoder is not None:
            parts.append(self.point_profile_encoder(point_profile))
        fused = torch.cat(parts, dim=1)
        trunk = self.trunk(fused)
        outputs = {
            "cls": self.cls(trunk).squeeze(1),
            "offset": self.offset(trunk),
            "height": self.height(trunk).squeeze(1),
        }
        if self.species is not None:
            outputs["species"] = self.species(trunk)
        return outputs

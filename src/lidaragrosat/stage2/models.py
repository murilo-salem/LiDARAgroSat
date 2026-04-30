from __future__ import annotations

import torch
from torch import nn

from lidaragrosat.models.blocks import SimpleEncoder


class RasterLocalRefiner(nn.Module):
    def __init__(self, in_channels: int, geom_dim: int = 7, normalization: str = "group") -> None:
        super().__init__()
        self.encoder = SimpleEncoder(in_channels, [32, 64, 128], norm=normalization)
        self.trunk = nn.Sequential(
            nn.Linear(128 + geom_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(64, 1)
        self.offset_head = nn.Linear(64, 2)
        self.height_head = nn.Linear(64, 1)
        self.split_head = nn.Linear(64, 1)

    def forward(self, raster: torch.Tensor, geom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(raster)[-1]
        pooled = torch.nn.functional.adaptive_avg_pool2d(encoded, 1).flatten(1)
        fused = torch.cat([pooled, geom_features], dim=1)
        trunk = self.trunk(fused)
        return {
            "cls": self.cls_head(trunk).squeeze(1),
            "offset": self.offset_head(trunk),
            "height": self.height_head(trunk).squeeze(1),
            "split": self.split_head(trunk).squeeze(1),
        }


class PointLocalRefiner(nn.Module):
    def __init__(self, point_dim: int = 6, geom_dim: int = 7) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Conv1d(point_dim, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 256, 1),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256 + geom_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(64, 1)
        self.offset_head = nn.Linear(64, 2)
        self.height_head = nn.Linear(64, 1)
        self.split_head = nn.Linear(64, 1)

    def forward(self, point_cloud: torch.Tensor, geom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.point_encoder(point_cloud)
        pooled = torch.max(encoded, dim=2).values
        fused = torch.cat([pooled, geom_features], dim=1)
        trunk = self.trunk(fused)
        return {
            "cls": self.cls_head(trunk).squeeze(1),
            "offset": self.offset_head(trunk),
            "height": self.height_head(trunk).squeeze(1),
            "split": self.split_head(trunk).squeeze(1),
        }


class HybridLocalRefiner(nn.Module):
    def __init__(self, in_channels: int, point_dim: int = 6, geom_dim: int = 7, normalization: str = "group") -> None:
        super().__init__()
        self.raster_encoder = SimpleEncoder(in_channels, [32, 64, 128], norm=normalization)
        self.point_encoder = nn.Sequential(
            nn.Conv1d(point_dim, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 256, 1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(128 + 256 + geom_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 + 256 + geom_dim, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 96),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(96, 1)
        self.offset_head = nn.Linear(96, 2)
        self.height_head = nn.Linear(96, 1)
        self.split_head = nn.Linear(96, 1)

    def forward(self, raster: torch.Tensor, point_cloud: torch.Tensor, geom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        raster_encoded = self.raster_encoder(raster)[-1]
        raster_pooled = torch.nn.functional.adaptive_avg_pool2d(raster_encoded, 1).flatten(1)
        point_encoded = self.point_encoder(point_cloud)
        point_pooled = torch.max(point_encoded, dim=2).values
        gate_input = torch.cat([raster_pooled, point_pooled, geom_features], dim=1)
        gate_weights = self.gate(gate_input)
        fused = torch.cat(
            [
                raster_pooled * gate_weights[:, :1],
                point_pooled * gate_weights[:, 1:2],
                geom_features,
            ],
            dim=1,
        )
        trunk = self.trunk(fused)
        return {
            "cls": self.cls_head(trunk).squeeze(1),
            "offset": self.offset_head(trunk),
            "height": self.height_head(trunk).squeeze(1),
            "split": self.split_head(trunk).squeeze(1),
        }


class StrongRGBHybridLocalRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        rgb_channels: int = 3,
        point_dim: int = 6,
        geom_dim: int = 13,
        normalization: str = "group",
    ) -> None:
        super().__init__()
        self.rgb_channels = int(max(rgb_channels, 0))
        self.aux_channels = int(max(in_channels - self.rgb_channels, 0))
        self.rgb_encoder = SimpleEncoder(max(self.rgb_channels, 1), [48, 96, 160], norm=normalization)
        self.aux_encoder = SimpleEncoder(max(self.aux_channels, 1), [16, 32, 64], norm=normalization)
        self.point_encoder = nn.Sequential(
            nn.Conv1d(point_dim, 48, 1),
            nn.GELU(),
            nn.Conv1d(48, 96, 1),
            nn.GELU(),
            nn.Conv1d(96, 128, 1),
            nn.GELU(),
        )
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(160 + 64 + 128 + 32, 224),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(224, 112),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(112, 1)
        self.offset_head = nn.Linear(112, 2)
        self.height_head = nn.Linear(112, 1)
        self.split_head = nn.Linear(112, 1)

    def forward(self, raster: torch.Tensor, point_cloud: torch.Tensor, geom_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.rgb_channels > 0:
            rgb = raster[:, : self.rgb_channels]
        else:
            rgb = raster[:, :1]
        if self.aux_channels > 0:
            aux = raster[:, self.rgb_channels : self.rgb_channels + self.aux_channels]
        else:
            aux = raster[:, :1]
        rgb_encoded = self.rgb_encoder(rgb)[-1]
        rgb_pooled = torch.nn.functional.adaptive_avg_pool2d(rgb_encoded, 1).flatten(1)
        aux_encoded = self.aux_encoder(aux)[-1]
        aux_pooled = torch.nn.functional.adaptive_avg_pool2d(aux_encoded, 1).flatten(1)
        point_encoded = self.point_encoder(point_cloud)
        point_pooled = torch.max(point_encoded, dim=2).values
        geom_emb = self.geom_mlp(geom_features)
        fused = torch.cat([rgb_pooled, aux_pooled, point_pooled, geom_emb], dim=1)
        trunk = self.trunk(fused)
        return {
            "cls": self.cls_head(trunk).squeeze(1),
            "offset": self.offset_head(trunk),
            "height": self.height_head(trunk).squeeze(1),
            "split": self.split_head(trunk).squeeze(1),
        }

from __future__ import annotations

import torch
from torch import nn


def make_norm(kind: str, num_channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(num_channels)
    if kind == "group":
        groups = max(1, min(8, num_channels // 8))
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported normalization: {kind}")


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group", stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            make_norm(norm, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(norm, out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(norm, out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels: int, stage_channels: list[int], norm: str = "group") -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            make_norm(norm, stage_channels[0]),
            nn.GELU(),
        )
        blocks = []
        current_channels = stage_channels[0]
        for stage_index, out_channels in enumerate(stage_channels):
            stride = 1 if stage_index == 0 else 2
            blocks.append(ResidualConvBlock(current_channels, out_channels, norm=norm, stride=stride))
            blocks.append(ResidualConvBlock(out_channels, out_channels, norm=norm, stride=1))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        current = self.stem(x)
        for block_index, block in enumerate(self.blocks):
            current = block(current)
            if block_index % 2 == 1:
                features.append(current)
        return features


class FusionBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "group") -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            make_norm(norm, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ResidualConvBlock(channels, channels, norm=norm)

    def forward(self, rgb_feature: torch.Tensor, lidar_feature: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([rgb_feature, lidar_feature], dim=1))
        fused = gate * rgb_feature + (1.0 - gate) * lidar_feature
        return self.refine(fused)


class TriFusionBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "group") -> None:
        super().__init__()
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            make_norm(norm, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.lidar_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            make_norm(norm, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.balance_gate = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            make_norm(norm, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ResidualConvBlock(channels, channels, norm=norm)

    def forward(self, rgb_feature: torch.Tensor, lidar_feature: torch.Tensor, chm_feature: torch.Tensor) -> torch.Tensor:
        rgb_context = self.rgb_gate(torch.cat([rgb_feature, chm_feature], dim=1))
        lidar_context = self.lidar_gate(torch.cat([lidar_feature, chm_feature], dim=1))
        rgb_guided = rgb_context * rgb_feature + (1.0 - rgb_context) * chm_feature
        lidar_guided = lidar_context * lidar_feature + (1.0 - lidar_context) * chm_feature
        balance = self.balance_gate(torch.cat([rgb_guided, lidar_guided, chm_feature], dim=1))
        fused = balance * rgb_guided + (1.0 - balance) * lidar_guided
        return self.refine(fused)


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels: list[int], decoder_channels: list[int], norm: str = "group") -> None:
        super().__init__()
        reversed_encoder = list(reversed(encoder_channels))
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(reversed_encoder, decoder_channels)]
        )
        self.merge_projections = nn.ModuleList()
        previous_channels = None
        for out_channels in decoder_channels:
            if previous_channels is None or previous_channels == out_channels:
                self.merge_projections.append(nn.Identity())
            else:
                self.merge_projections.append(nn.Conv2d(previous_channels, out_channels, kernel_size=1))
            previous_channels = out_channels
        self.refine_blocks = nn.ModuleList(
            [ResidualConvBlock(out_ch, out_ch, norm=norm) for out_ch in decoder_channels]
        )
        self.output_channels = decoder_channels[-1]

    def forward(self, features: list[torch.Tensor], return_pyramid: bool = False):
        current = None
        pyramid: list[torch.Tensor] = []
        for feature, projection, merge_projection, refine in zip(
            reversed(features),
            self.projections,
            self.merge_projections,
            self.refine_blocks,
        ):
            projected = projection(feature)
            if current is None:
                current = projected
            else:
                current = torch.nn.functional.interpolate(
                    current,
                    size=projected.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                current = merge_projection(current)
                current = current + projected
            current = refine(current)
            pyramid.append(current)
        if return_pyramid:
            return current, pyramid
        return current

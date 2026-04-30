from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = adjacency @ x / degree
        out = self.self_linear(x) + self.neigh_linear(neigh)
        return F.gelu(self.norm(out))


class SegmentGraphClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * int(num_layers)
        self.layers = nn.ModuleList(
            [DenseGraphSAGELayer(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adjacency)
        return self.head(h).squeeze(-1)

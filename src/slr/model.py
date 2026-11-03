"""Sequence classifier: input projection -> 2-layer BiGRU -> attention pooling -> linear."""
from __future__ import annotations

import torch
from torch import nn


class SignGRU(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.inp(x))
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)
        return self.head((h * w.unsqueeze(-1)).sum(1))

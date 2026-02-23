"""Hierarchical temporal attention for weighting genre-defining segments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Learns to weight temporal positions in a feature sequence.

    Replaces global average pooling with a learned weighted sum that
    can focus on genre-defining segments (e.g., chorus vs. intro).

    Input:  (B, T, D) — a sequence of D-dimensional feature vectors
    Output: (B, D)    — attention-weighted summary
    """

    def __init__(self, feature_dim: int, attention_dim: int = 128):
        super().__init__()
        self.project = nn.Linear(feature_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (B, D) weighted feature summary
            weights: (B, T) attention weights (for visualization)
        """
        energy = torch.tanh(self.project(x))   # (B, T, attention_dim)
        scores = self.score(energy).squeeze(-1) # (B, T)
        weights = F.softmax(scores, dim=-1)     # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return context, weights

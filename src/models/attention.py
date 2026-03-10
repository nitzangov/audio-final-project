"""Hierarchical temporal attention for weighting genre-defining segments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Learns to weight temporal positions in a feature sequence.

    Replaces global average pooling with a learned weighted sum that
    can focus on genre-defining segments (e.g., chorus vs. intro).

    Stabilization measures vs. the naive Tanh+Softmax design:
      1. LayerNorm on input to control activation scale
      2. Scaled logits (1/sqrt(attention_dim)) to prevent softmax saturation
      3. Xavier initialization on projection weights
      4. Residual connection blending attention output with mean-pool,
         gated by a learnable parameter (starts at 0 = pure mean-pool)

    Input:  (B, T, D) — a sequence of D-dimensional feature vectors
    Output: (B, D)    — attention-weighted summary
    """

    def __init__(self, feature_dim: int, attention_dim: int = 128):
        super().__init__()
        self.scale = 1.0 / math.sqrt(attention_dim)

        self.norm = nn.LayerNorm(feature_dim)
        self.project = nn.Linear(feature_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

        # Learnable gate: initialized to -5 so sigmoid(-5) ≈ 0.007, meaning
        # the output starts as ~pure mean-pool. The model gradually opens the
        # gate to incorporate attention as training progresses.
        self.gate = nn.Parameter(torch.full((1,), -5.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        nn.init.xavier_uniform_(self.score.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            context: (B, D) weighted feature summary
            weights: (B, T) attention weights (for visualization)
        """
        x_norm = self.norm(x)                            # (B, T, D)
        energy = torch.tanh(self.project(x_norm))        # (B, T, attention_dim)
        scores = self.score(energy).squeeze(-1)           # (B, T)
        scores = scores * self.scale                      # prevent softmax saturation
        weights = F.softmax(scores, dim=-1)               # (B, T)

        attn_out = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        mean_out = x.mean(dim=1)                                   # (B, D)

        gate = torch.sigmoid(self.gate)
        context = gate * attn_out + (1.0 - gate) * mean_out

        return context, weights

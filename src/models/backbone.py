"""Lightweight CNN backbone with GroupNorm for spectrogram feature extraction."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> ReLU -> MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides out_ch
        groups = min(num_groups, out_ch)
        while out_ch % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class CNNBackbone(nn.Module):
    """3-block CNN that maps a single-channel spectrogram to a temporal feature sequence.

    Input:  (B, 1, n_freq, T)
    Output: (B, C_out, T')  where C_out is the last channel count and T' is the
            temporally-reduced dimension.
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        num_groups: int = 8,
    ):
        super().__init__()
        channels = channels or [32, 64, 128]
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.append(ConvBlock(in_ch, out_ch, num_groups))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # Collapse frequency axis to 1 while keeping temporal axis
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_freq, T)
        Returns:
            (B, C_out, T') where T' = T // (2^num_blocks)
        """
        x = self.conv(x)       # (B, C_out, freq', T')
        x = self.pool(x)       # (B, C_out, 1, T')
        x = x.squeeze(2)       # (B, C_out, T')
        return x

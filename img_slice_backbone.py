from __future__ import annotations

import torch
import torch.nn as nn


class SliceBackbone(nn.Module):
    """
    Convolutional backbone for extracting a per-slice embedding.

    Input:
        x: Tensor of shape [B, C, H, W]

    Output:
        Tensor of shape [B, d_model]
    """

    def __init__(self, in_channels: int = 1, d_model: int = 128) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}.")
        if d_model <= 0:
            raise ValueError(f"`d_model` must be positive, got {d_model}.")
        if d_model % 4 != 0:
            raise ValueError(f"`d_model` must be divisible by 4, got {d_model}.")

        self.in_channels = in_channels
        self.d_model = d_model

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Tensor of shape [B, d_model].
        """
        if x.ndim != 4:
            raise ValueError(
                f"`x` must have shape [B, C, H, W], got shape {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channel(s), got {x.size(1)}."
            )

        x = self.features(x)     # [B, d_model, 1, 1]
        x = torch.flatten(x, 1)  # [B, d_model]
        return x
import torch
import torch.nn as nn

class SliceBackbone(nn.Module):
    """
    입력:  x: [B, C, H, W]
    출력: emb: [B, d_model]
    """
    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # [B, 128, 1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)         # [B, 128, 1, 1]
        x = torch.flatten(x, 1)      # [B, 128]
        return x
import torch
import torch.nn as nn


class RadClassifier(nn.Module):
    """MLP classifier for relationship-aware multimodal features."""

    def __init__(
        self,
        d_model: int = 128,
        mlp_ratio: int = 4,
        dropout: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        logits = self.fc(fused)
        return logits.squeeze(-1)

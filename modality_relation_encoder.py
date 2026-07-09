import torch
import torch.nn as nn


class ModalityRelationEncoder(nn.Module):
    """T1w/T2w modality-interaction and relationship-aware fusion module."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 1,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_modality_embedding: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.use_modality_embedding = use_modality_embedding

        if self.use_modality_embedding:
            self.t1_modality_embed = nn.Parameter(torch.empty(1, d_model))
            self.t2_modality_embed = nn.Parameter(torch.empty(1, d_model))
        else:
            self.register_parameter("t1_modality_embed", None)
            self.register_parameter("t2_modality_embed", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.use_modality_embedding:
            nn.init.trunc_normal_(self.t1_modality_embed, std=0.02)
            nn.init.trunc_normal_(self.t2_modality_embed, std=0.02)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor):
        """Return fused vector and interaction-enhanced modality tokens.

        Args:
            h_a: T1w representation, shape [B, d_model].
            h_b: T2w representation, shape [B, d_model].
        """
        if self.use_modality_embedding:
            h_a = h_a + self.t1_modality_embed
            h_b = h_b + self.t2_modality_embed

        tokens = torch.stack([h_a, h_b], dim=1)  # [B, 2, d_model]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        h1 = tokens[:, 0, :]
        h2 = tokens[:, 1, :]

        fused = torch.cat([h1, h2, h1 - h2, h1 * h2], dim=-1)
        return fused, h1, h2

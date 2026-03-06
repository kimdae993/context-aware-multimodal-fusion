from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn


class _AttnFFNLayer(nn.Module):
    """
    Transformer-style encoder block with exposed attention weights.

    This block is used instead of `nn.TransformerEncoderLayer` so that
    per-head self-attention weights can be returned.

    Input / output shape:
        x: [B, T, D]

    If `return_attn=True`, the block returns:
        - output: [B, T, D]
        - attn_weights: [B, heads, T, T]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        mlp_ratio: int,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"`d_model` must be positive, got {d_model}.")
        if nhead <= 0:
            raise ValueError(f"`nhead` must be positive, got {nhead}.")
        if mlp_ratio <= 0:
            raise ValueError(f"`mlp_ratio` must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        hidden_dim = d_model * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape [B, T, D].
            return_attn: Whether to return per-head attention weights.

        Returns:
            If `return_attn=False`:
                Tensor of shape [B, T, D]

            If `return_attn=True`:
                (
                    output: Tensor of shape [B, T, D],
                    attn_weights: Tensor of shape [B, heads, T, T],
                )
        """
        if x.ndim != 3:
            raise ValueError(
                f"`x` must have shape [B, T, D], got shape {tuple(x.shape)}."
            )

        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights: [B, heads, T, T]
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        if return_attn:
            return x, attn_weights
        return x


class ModalityRelationEncoder(nn.Module):
    """
    Relation encoder for two modality-level feature vectors.

    Inputs:
        h_a: [B, d_model]
        h_b: [B, d_model]

    Processing:
        - stack the two modality vectors into a length-2 token sequence
        - apply transformer-style self-attention layers
        - derive relation-aware representations
        - build a fused feature using concatenation of:
            [h1, h2, h1 - h2, h1 * h2]

    Outputs:
        fused: [B, 4 * d_model]
        h1: [B, d_model]
        h2: [B, d_model]

    If `return_attn=True`, also returns:
        attn_list: list of tensors with shape [B, heads, 2, 2]
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 1,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"`d_model` must be positive, got {d_model}.")
        if nhead <= 0:
            raise ValueError(f"`nhead` must be positive, got {nhead}.")
        if num_layers < 0:
            raise ValueError(f"`num_layers` must be >= 0, got {num_layers}.")
        if mlp_ratio <= 0:
            raise ValueError(f"`mlp_ratio` must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")

        self.d_model = d_model

        self.layers = nn.ModuleList(
            [
                _AttnFFNLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]],
    ]:
        """
        Args:
            h_a: Tensor of shape [B, d_model].
            h_b: Tensor of shape [B, d_model].
            return_attn: Whether to return per-layer attention weights.

        Returns:
            If `return_attn=False`:
                (
                    fused: [B, 4 * d_model],
                    h1: [B, d_model],
                    h2: [B, d_model],
                )

            If `return_attn=True`:
                (
                    fused: [B, 4 * d_model],
                    h1: [B, d_model],
                    h2: [B, d_model],
                    attn_list: list of [B, heads, 2, 2],
                )
        """
        if h_a.ndim != 2:
            raise ValueError(
                f"`h_a` must have shape [B, d_model], got shape {tuple(h_a.shape)}."
            )
        if h_b.ndim != 2:
            raise ValueError(
                f"`h_b` must have shape [B, d_model], got shape {tuple(h_b.shape)}."
            )
        if h_a.shape != h_b.shape:
            raise ValueError(
                f"`h_a` and `h_b` must have the same shape, "
                f"got {tuple(h_a.shape)} and {tuple(h_b.shape)}."
            )
        if h_a.size(-1) != self.d_model:
            raise ValueError(
                f"Expected feature dimension {self.d_model}, got {h_a.size(-1)}."
            )

        tokens = torch.stack([h_a, h_b], dim=1)  # [B, 2, d_model]

        attn_list: List[torch.Tensor] = []
        for layer in self.layers:
            if return_attn:
                tokens, attn_weights = layer(tokens, return_attn=True)  # [B, heads, 2, 2]
                attn_list.append(attn_weights)
            else:
                tokens = layer(tokens, return_attn=False)

        tokens = self.norm(tokens)

        h1 = tokens[:, 0, :]  # [B, d_model]
        h2 = tokens[:, 1, :]  # [B, d_model]

        diff = h1 - h2
        prod = h1 * h2

        fused = torch.cat([h1, h2, diff, prod], dim=-1)  # [B, 4 * d_model]

        if return_attn:
            return fused, h1, h2, attn_list
        return fused, h1, h2
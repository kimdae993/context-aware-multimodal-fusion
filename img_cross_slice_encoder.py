from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from img_slice_backbone import SliceBackbone


class _AttnFFNLayer(nn.Module):
    """
    Transformer-style encoder block with exposed attention weights.

    This block is used in place of `nn.TransformerEncoderLayer` so that
    per-head self-attention weights can be returned to the caller.

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
                torch.Tensor of shape [B, T, D]
            If `return_attn=True`:
                (
                    output: torch.Tensor of shape [B, T, D],
                    attn_weights: torch.Tensor of shape [B, heads, T, T],
                )
        """
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


class CrossSliceEncoder(nn.Module):
    """
    Cross-slice encoder for a single imaging modality.

    The encoder first extracts per-slice embeddings using a CNN backbone,
    then applies positional encoding, optional depth attention, and
    transformer-style cross-slice attention layers.

    Input:
        x: [B, K, C, H, W]

    Output:
        h_modality: [B, d_model]
        tokens: [B, K, d_model]

    If `return_attn=True`, also returns:
        attn_list: list of tensors with shape [B, heads, K, K]
            - depth attention weights (if enabled)
            - attention weights from each transformer block
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        num_slices: int = 3,
        nhead: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        use_depth: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}.")
        if d_model <= 0:
            raise ValueError(f"`d_model` must be positive, got {d_model}.")
        if num_slices <= 0:
            raise ValueError(f"`num_slices` must be positive, got {num_slices}.")
        if nhead <= 0:
            raise ValueError(f"`nhead` must be positive, got {nhead}.")
        if num_layers < 0:
            raise ValueError(f"`num_layers` must be >= 0, got {num_layers}.")
        if mlp_ratio <= 0:
            raise ValueError(f"`mlp_ratio` must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")

        self.num_slices = num_slices
        self.d_model = d_model
        self.use_depth = use_depth

        self.backbone = SliceBackbone(in_channels=in_channels, d_model=d_model)

        # Positional embedding for slice indices.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, d_model))

        # Optional depth-attention branch.
        if use_depth:
            self.depth_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,  # input shape: [B, K, D]
            )
            self.depth_norm = nn.LayerNorm(d_model)
        else:
            self.depth_attn = None
            self.depth_norm = None

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
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
    ]:
        """
        Args:
            x: Input tensor of shape [B, K, C, H, W].
            return_attn: Whether to return attention weights.

        Returns:
            If `return_attn=False`:
                (
                    h_modality: [B, d_model],
                    tokens: [B, K, d_model],
                )

            If `return_attn=True`:
                (
                    h_modality: [B, d_model],
                    tokens: [B, K, d_model],
                    attn_list: list of [B, heads, K, K],
                )
        """
        if x.ndim != 5:
            raise ValueError(
                f"`x` must have shape [B, K, C, H, W], got shape {tuple(x.shape)}."
            )

        batch_size, num_slices_in, channels, height, width = x.shape
        if num_slices_in > self.num_slices:
            raise ValueError(
                f"Input contains {num_slices_in} slices, but `num_slices` is set to {self.num_slices}."
            )

        # Slice-wise CNN encoding.
        x_flat = x.view(batch_size * num_slices_in, channels, height, width)   # [B*K, C, H, W]
        embeddings = self.backbone(x_flat)                                      # [B*K, d_model]
        embeddings = embeddings.view(batch_size, num_slices_in, self.d_model)   # [B, K, d_model]

        # Add positional embedding.
        pos_embed = self.pos_embed[:, :num_slices_in, :]                        # [1, K, d_model]
        tokens = embeddings + pos_embed                                         # [B, K, d_model]

        attn_list: List[torch.Tensor] = []

        # Optional depth attention across slices.
        if self.use_depth:
            depth_out, depth_attn_weights = self.depth_attn(
                tokens,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=False,  # [B, heads, K, K]
            )
            tokens = self.depth_norm(tokens + depth_out)

            if return_attn:
                attn_list.append(depth_attn_weights)

        # Cross-slice attention blocks.
        for layer in self.layers:
            if return_attn:
                tokens, attn_weights = layer(tokens, return_attn=True)
                attn_list.append(attn_weights)
            else:
                tokens = layer(tokens, return_attn=False)

        tokens = self.norm(tokens)

        # Mean pooling over slices.
        h_modality = tokens.mean(dim=1)  # [B, d_model]

        if return_attn:
            return h_modality, tokens, attn_list
        return h_modality, tokens
import torch
import torch.nn as nn

from img_slice_backbone import SliceBackbone


class CrossSliceEncoder(nn.Module):
    """Intra-modality encoder for an ordered sparse slice sequence."""

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        num_slices: int = 3,
        nhead: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        use_positional_embedding: bool = True,
        use_depthwise_attention: bool = True,
        use_transformer_encoder: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.num_slices = num_slices
        self.d_model = d_model
        self.use_positional_embedding = use_positional_embedding
        self.use_depthwise_attention = use_depthwise_attention
        self.use_transformer_encoder = use_transformer_encoder

        self.backbone = SliceBackbone(in_channels=in_channels, d_model=d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, num_slices, d_model))

        self.depth_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.depth_norm = nn.LayerNorm(d_model)

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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        """Encode a batch of slice sequences.

        Args:
            x: [B, K, C, H, W]

        Returns:
            h_modality: [B, d_model]
            tokens: [B, K, d_model]
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input shape [B, K, C, H, W], got {tuple(x.shape)}")

        B, K, C, H, W = x.shape
        if K > self.num_slices:
            raise ValueError(f"num_slices({self.num_slices}) < input K({K})")

        x_flat = x.reshape(B * K, C, H, W)
        emb = self.backbone(x_flat)
        tokens = emb.reshape(B, K, self.d_model)

        if self.use_positional_embedding:
            tokens = tokens + self.pos_embed[:, :K, :]

        if self.use_depthwise_attention:
            depth_out, _ = self.depth_attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.depth_norm(tokens + depth_out)

        if self.use_transformer_encoder:
            tokens = self.transformer(tokens)
            tokens = self.norm(tokens)

        h_modality = tokens.mean(dim=1)
        return h_modality, tokens

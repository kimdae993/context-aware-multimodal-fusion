import torch
import torch.nn as nn

from classifier import RadClassifier
from img_cross_slice_encoder import CrossSliceEncoder
from modality_relation_encoder import ModalityRelationEncoder


class SymBrainRadModel(nn.Module):
    """Context-aware multimodal slice-sequence fusion model."""

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        num_slices: int = 3,
        slice_nhead: int = 2,
        slice_layers: int = 1,
        slice_dropout: float = 0.1,
        mlp_ratio: int = 4,
        modal_nhead: int = 2,
        modal_layers: int = 1,
        modal_dropout: float = 0.3,
        cls_dropout: float = 0.3,
        num_classes: int = 1,
        use_positional_embedding: bool = True,
        use_depthwise_attention: bool = True,
        use_transformer_encoder: bool = True,
        use_modality_embedding: bool = True,
    ):
        super().__init__()

        self.t1_cross_slice_encoder = CrossSliceEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_slices=num_slices,
            nhead=slice_nhead,
            num_layers=slice_layers,
            dropout=slice_dropout,
            mlp_ratio=mlp_ratio,
            use_positional_embedding=use_positional_embedding,
            use_depthwise_attention=use_depthwise_attention,
            use_transformer_encoder=use_transformer_encoder,
        )

        self.t2_cross_slice_encoder = CrossSliceEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_slices=num_slices,
            nhead=slice_nhead,
            num_layers=slice_layers,
            dropout=slice_dropout,
            mlp_ratio=mlp_ratio,
            use_positional_embedding=use_positional_embedding,
            use_depthwise_attention=use_depthwise_attention,
            use_transformer_encoder=use_transformer_encoder,
        )

        self.modality_relation = ModalityRelationEncoder(
            d_model=d_model,
            nhead=modal_nhead,
            num_layers=modal_layers,
            dropout=modal_dropout,
            mlp_ratio=mlp_ratio,
            use_modality_embedding=use_modality_embedding,
        )

        self.classifier = RadClassifier(
            d_model=d_model,
            mlp_ratio=4,
            dropout=cls_dropout,
            num_classes=num_classes,
        )

    def forward(self, t1_imgs: torch.Tensor, t2_imgs: torch.Tensor):
        h_t1, _ = self.t1_cross_slice_encoder(t1_imgs)
        h_t2, _ = self.t2_cross_slice_encoder(t2_imgs)

        fused, _, _ = self.modality_relation(h_t1, h_t2)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    B, K, C, H, W = 4, 3, 1, 290, 290
    t1 = torch.randn(B, K, C, H, W)
    t2 = torch.randn(B, K, C, H, W)

    model = SymBrainRadModel(in_channels=C, d_model=128, num_slices=K)
    logits = model(t1, t2)
    print("logit shape:", logits.shape)

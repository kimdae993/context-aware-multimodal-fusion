from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from classifier import RadClassifier
from img_cross_slice_encoder import CrossSliceEncoder
from modality_relation_encoder import ModalityRelationEncoder


class SymBrainRadModel(nn.Module):
    """
    End-to-end model for binary radiology-score classification from paired T1/T2 slices.

    Pipeline:
        1. Encode T1 slices with a cross-slice encoder
        2. Encode T2 slices with a cross-slice encoder
        3. Model inter-modality relations
        4. Predict a binary logit

    Expected inputs:
        t1_imgs: [B, K, C, H, W]
        t2_imgs: [B, K, C, H, W]

    Output:
        logit: [B]

    If `return_aux=True`, also returns an auxiliary dictionary containing
    intermediate representations. If `return_attn=True`, attention weights
    are added to that auxiliary dictionary.
    """

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
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}.")
        if d_model <= 0:
            raise ValueError(f"`d_model` must be positive, got {d_model}.")
        if num_slices <= 0:
            raise ValueError(f"`num_slices` must be positive, got {num_slices}.")
        if slice_nhead <= 0:
            raise ValueError(f"`slice_nhead` must be positive, got {slice_nhead}.")
        if slice_layers < 0:
            raise ValueError(f"`slice_layers` must be >= 0, got {slice_layers}.")
        if modal_nhead <= 0:
            raise ValueError(f"`modal_nhead` must be positive, got {modal_nhead}.")
        if modal_layers < 0:
            raise ValueError(f"`modal_layers` must be >= 0, got {modal_layers}.")
        if mlp_ratio <= 0:
            raise ValueError(f"`mlp_ratio` must be positive, got {mlp_ratio}.")
        if num_classes <= 0:
            raise ValueError(f"`num_classes` must be positive, got {num_classes}.")

        self.t1_cross_slice_encoder = CrossSliceEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_slices=num_slices,
            nhead=slice_nhead,
            num_layers=slice_layers,
            dropout=slice_dropout,
            use_depth=True,
            mlp_ratio=mlp_ratio,
        )

        self.t2_cross_slice_encoder = CrossSliceEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_slices=num_slices,
            nhead=slice_nhead,
            num_layers=slice_layers,
            dropout=slice_dropout,
            use_depth=True,
            mlp_ratio=mlp_ratio,
        )

        self.modality_relation = ModalityRelationEncoder(
            d_model=d_model,
            nhead=modal_nhead,
            num_layers=modal_layers,
            dropout=modal_dropout,
            mlp_ratio=mlp_ratio,
        )

        self.classifier = RadClassifier(
            input_dim=d_model * 4,
            hidden_dim=d_model,
            dropout=cls_dropout,
            output_dim=num_classes,
        )

    def forward(
        self,
        t1_imgs: torch.Tensor,
        t2_imgs: torch.Tensor,
        return_aux: bool = False,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass.

        Args:
            t1_imgs: T1 input tensor of shape [B, K, C, H, W].
            t2_imgs: T2 input tensor of shape [B, K, C, H, W].
            return_aux: Whether to return intermediate representations.
            return_attn: Whether to compute and include attention weights.

        Returns:
            If `return_aux=False`:
                logit tensor of shape [B]

            If `return_aux=True`:
                (
                    logit tensor of shape [B],
                    aux dictionary
                )

        Notes:
            - If `return_attn=True` but `return_aux=False`, attention values are
              computed internally but not returned. To access them, call with
              `return_aux=True`.
        """
        if t1_imgs.ndim != 5:
            raise ValueError(
                f"`t1_imgs` must have shape [B, K, C, H, W], got {tuple(t1_imgs.shape)}."
            )
        if t2_imgs.ndim != 5:
            raise ValueError(
                f"`t2_imgs` must have shape [B, K, C, H, W], got {tuple(t2_imgs.shape)}."
            )
        if t1_imgs.shape != t2_imgs.shape:
            raise ValueError(
                f"`t1_imgs` and `t2_imgs` must have the same shape, "
                f"got {tuple(t1_imgs.shape)} and {tuple(t2_imgs.shape)}."
            )

        if return_attn:
            h_t1, t1_tokens, t1_attn = self.t1_cross_slice_encoder(
                t1_imgs, return_attn=True
            )
            h_t2, t2_tokens, t2_attn = self.t2_cross_slice_encoder(
                t2_imgs, return_attn=True
            )
            fused, h1_rel, h2_rel, modal_attn = self.modality_relation(
                h_t1, h_t2, return_attn=True
            )
        else:
            h_t1, t1_tokens = self.t1_cross_slice_encoder(t1_imgs)
            h_t2, t2_tokens = self.t2_cross_slice_encoder(t2_imgs)
            fused, h1_rel, h2_rel = self.modality_relation(h_t1, h_t2)

        logits = self.classifier(fused)  # [B] or [B, 1]
        logits = logits.view(-1)         # keep evaluation-time shape stable as [B]

        if not return_aux and not return_attn:
            return logits

        aux: Dict[str, Any] = {
            "h_t1": h_t1,
            "h_t2": h_t2,
            "t1_tokens": t1_tokens,
            "t2_tokens": t2_tokens,
            "h1_rel": h1_rel,
            "h2_rel": h2_rel,
            "fused": fused,
        }

        if return_attn:
            aux.update(
                {
                    "t1_attn": t1_attn,       # list of [B, heads, K, K]
                    "t2_attn": t2_attn,       # list of [B, heads, K, K]
                    "modal_attn": modal_attn, # list of [B, heads, 2, 2]
                }
            )

        if return_aux:
            return logits, aux

        return logits


if __name__ == "__main__":
    batch_size, num_slices, channels, height, width = 4, 3, 1, 290, 290
    t1 = torch.randn(batch_size, num_slices, channels, height, width)
    t2 = torch.randn(batch_size, num_slices, channels, height, width)

    model = SymBrainRadModel(
        in_channels=channels,
        d_model=128,
        num_slices=num_slices,
    )
    logits, aux = model(t1, t2, return_aux=True, return_attn=True)

    print("logit shape:", logits.shape)  # [B]
    print("t1_attn last layer:", aux["t1_attn"][-1].shape)      # [B, heads, K, K]
    print("modal_attn last layer:", aux["modal_attn"][-1].shape)  # [B, heads, 2, 2]
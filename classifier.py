import torch
import torch.nn as nn

class RadClassifier(nn.Module):
    """
    Binary classifier for predicting a radiology-related score from a fused feature.

    This module expects a fused feature tensor of shape [batch_size, input_dim]
    and returns a single logit per sample, suitable for `nn.BCEWithLogitsLoss`.

    Args:
        input_dim (int): Dimension of the fused input feature.
        hidden_dim (int): Hidden dimension of the MLP classifier.
        dropout (float): Dropout probability.
        output_dim (int): Output dimension. Use 1 for binary classification.

    Input:
        fused (torch.Tensor): Tensor of shape [B, input_dim].

    Returns:
        torch.Tensor: Logits of shape [B] if output_dim == 1,
            otherwise [B, output_dim].
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}.")
        if output_dim <= 0:
            raise ValueError(f"`output_dim` must be positive, got {output_dim}.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            fused (torch.Tensor): Input tensor of shape [B, input_dim].

        Returns:
            torch.Tensor: Output logits.
        """
        if fused.ndim != 2:
            raise ValueError(
                f"`fused` must be a 2D tensor of shape [B, input_dim], got shape {tuple(fused.shape)}."
            )
        if fused.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, "
                f"but got {fused.size(-1)}."
            )

        logits = self.classifier(fused)

        if self.output_dim == 1:
            logits = logits.squeeze(-1)

        return logits
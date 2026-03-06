# export_selected_attention.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from img_dataset import TwinMultiImageDataset
from img_model import SymBrainRadModel


# =========================
# Configuration
# =========================
RESULTS_DIR = Path("results/20260210-161302")
SELECTED_CSV = RESULTS_DIR / "selected_cases.csv"
CKPT_PATH = RESULTS_DIR / "best_val_bal_acc_model.pt"
OUT_DIR = RESULTS_DIR / "attn_selected"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match the model configuration used during training.
D_MODEL = 128

IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((290, 290)),
        transforms.ToTensor(),
    ]
)


def validate_inputs() -> None:
    """Validate that required input files and directories exist."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    if not SELECTED_CSV.exists():
        raise FileNotFoundError(f"Selected cases CSV not found: {SELECTED_CSV}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {CKPT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_paths(path_string: str) -> List[str]:
    """
    Parse a serialized path string into a list of file paths.

    Expected format:
        path1|path2|path3|...

    Args:
        path_string: Serialized path string.

    Returns:
        List of file paths.
    """
    if pd.isna(path_string):
        return []
    return str(path_string).split("|")


def attn_to_2d(
    attn_obj: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]
) -> Optional[np.ndarray]:
    """
    Convert an attention object to a 2D NumPy array.

    Supported input formats:
        - list/tuple of tensors with shape [B, heads, T, T]
        - tensor with shape [B, heads, T, T]
        - tensor with shape [B, T, T]

    The output is generated from:
        - the last layer, if a list/tuple is provided
        - the first sample in the batch
        - mean over heads, when heads are present

    Args:
        attn_obj: Attention tensor or list of attention tensors.

    Returns:
        A 2D NumPy array of shape [T, T], or None if unavailable.
    """
    if attn_obj is None:
        return None

    if isinstance(attn_obj, (list, tuple)):
        if len(attn_obj) == 0:
            return None
        attn_obj = attn_obj[-1]

    if not torch.is_tensor(attn_obj):
        return None

    attn = attn_obj.detach()

    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)  # [T, T]
    elif attn.dim() == 3:
        attn = attn[0]  # [T, T]
    else:
        return None

    return attn.cpu().numpy()


def save_heatmap(matrix: np.ndarray, save_path: Path, title: str) -> None:
    """
    Save a heatmap image for a 2D attention matrix.

    Args:
        matrix: 2D array to visualize.
        save_path: Output image path.
        title: Plot title.
    """
    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def load_model(checkpoint_path: Path, device: torch.device) -> SymBrainRadModel:
    """
    Load a trained model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Target device.

    Returns:
        Loaded model in evaluation mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SymBrainRadModel(d_model=D_MODEL)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    validate_inputs()

    # --- load selected cases ---
    df = pd.read_csv(SELECTED_CSV)

    required_columns = {"paths", "y_true", "y_pred", "type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {SELECTED_CSV}: {sorted(missing_columns)}"
        )

    paths_list = [parse_paths(path_str) for path_str in df["paths"].tolist()]
    labels_list = df["y_true"].astype(int).tolist()

    # Dataset construction
    dataset = TwinMultiImageDataset(paths_list, labels_list, transform=IMG_TRANSFORM)

    # --- load model ---
    model = load_model(CKPT_PATH, DEVICE)

    print(f"Loaded checkpoint: {CKPT_PATH}")
    print(f"Exporting attention maps to: {OUT_DIR}")

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]

            # Expected format:
            #   (t1, t2, label, paths) or (t1, t2, label)
            if len(sample) == 4:
                t1, t2, y_true, _ = sample
            else:
                t1, t2, y_true = sample

            # Add batch dimension
            t1 = t1.unsqueeze(0).to(DEVICE)
            t2 = t2.unsqueeze(0).to(DEVICE)

            logit, aux = model(t1, t2, return_aux=True, return_attn=True)
            prob = torch.sigmoid(logit).item()

            t1_map = attn_to_2d(aux.get("t1_attn"))
            t2_map = attn_to_2d(aux.get("t2_attn"))
            modal_map = attn_to_2d(aux.get("modal_attn"))

            case_type = df.loc[index, "type"]
            y_pred = df.loc[index, "y_pred"]

            # Include case type, target, prediction, and probability in the file name
            base_name = f"{index:02d}_{case_type}_y{y_true}_p{y_pred}_s{prob:.3f}"

            if t1_map is not None:
                save_heatmap(
                    t1_map,
                    OUT_DIR / f"{base_name}_t1.png",
                    "T1 slice attention",
                )
            if t2_map is not None:
                save_heatmap(
                    t2_map,
                    OUT_DIR / f"{base_name}_t2.png",
                    "T2 slice attention",
                )
            if modal_map is not None:
                save_heatmap(
                    modal_map,
                    OUT_DIR / f"{base_name}_modal.png",
                    "Modality attention (2x2)",
                )

    print("Done.")


if __name__ == "__main__":
    main()
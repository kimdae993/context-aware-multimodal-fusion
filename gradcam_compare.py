from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from img_model import SymBrainRadModel


# =========================================================
# User configuration
# =========================================================
RESULTS_DIR = Path("results/20260210-161302")

# Automatically use the first existing evaluation-detail CSV.
EVAL_CSV_CANDIDATES = [
    RESULTS_DIR / "test_best_eval_detail.csv",
    RESULTS_DIR / "test_final_eval_detail.csv",
]

CKPT_PATH = RESULTS_DIR / "best_val_bal_acc_model.pt"

OUT_DIR = RESULTS_DIR / "gradcam_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_PATH = OUT_DIR / "match_debug.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_MODEL = 128
NUM_SLICES = 3
RESIZE_HW = (290, 290)

IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(RESIZE_HW),
        transforms.ToTensor(),  # [1, H, W] for grayscale input
    ]
)

TARGET_RADSCORES = [1, 5]


# =========================================================
# Utilities
# =========================================================
def normalize_path(path_str: str) -> str:
    """
    Normalize a path string by trimming whitespace and standardizing separators.
    """
    normalized = str(path_str).strip().replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    return normalized


def resolve_path(path_str: str, base_dir: str) -> str:
    """
    Resolve a relative path from the evaluation CSV into an absolute path.

    Many CSV rows may contain relative paths such as './T1w/xxx.png'.
    """
    normalized = normalize_path(path_str)
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if os.path.isabs(normalized):
        return normalized
    return os.path.join(base_dir, normalized)


def parse_paths_cell(cell: str) -> List[str]:
    """
    Parse a serialized paths cell into a list of individual paths.
    """
    return [normalize_path(x) for x in str(cell).split("|") if str(x).strip() != ""]


def extract_radscore_from_paths_cell(cell: str) -> Optional[int]:
    """
    Extract the radiology score from the serialized paths field.
    """
    for path_str in parse_paths_cell(cell):
        match = re.search(r"rad_score_(\d)", path_str)
        if match:
            return int(match.group(1))
    return None


def extract_session_id(cell: str) -> Optional[str]:
    """
    Extract the session ID from the serialized paths field.
    """
    for path_str in parse_paths_cell(cell):
        match = re.search(r"session_(\d+)", path_str)
        if match:
            return match.group(1)
    return None


def infer_modality_from_paths_cell(cell: str) -> str:
    """
    Infer modality label (T1 / T2 / UNK) from the serialized paths field.
    """
    paths = parse_paths_cell(cell)
    joined = " ".join(paths).lower()

    if "/t1w/" in joined or joined.startswith("./t1w/") or "t1w" in joined:
        return "T1"
    if "/t2w/" in joined or joined.startswith("./t2w/") or "t2w" in joined:
        return "T2"
    return "UNK"


def choose_three_slices(paths: Sequence[str]) -> Sequence[str]:
    """
    Select the first three slices from a sequence of image paths.
    """
    if len(paths) < 3:
        raise ValueError(f"Expected at least 3 slices, but got {len(paths)}.")
    return paths[:3]


def norm01(array: np.ndarray) -> np.ndarray:
    """
    Normalize an array to the [0, 1] range.
    """
    array = array.astype(np.float32)
    array = array - array.min()
    array = array / (array.max() + 1e-8)
    return array


def overlay_cam_on_gray(img_gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a grayscale image.
    """
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    cmap = plt.get_cmap("jet")
    cam_rgb = cmap(cam)[..., :3]
    overlaid = (1 - alpha) * img_rgb + alpha * cam_rgb
    return np.clip(overlaid, 0, 1)


# =========================================================
# Direct image loading -> [1, K, 1, H, W] tensor
# =========================================================
def load_slices_as_tensor(paths3: Sequence[str], base_dir: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load three grayscale slices and convert them into a model input tensor.

    Args:
        paths3: Sequence of three image paths.
        base_dir: Base directory used to resolve relative paths.

    Returns:
        x:
            Tensor of shape [1, 3, 1, H, W].
        raw_np:
            NumPy array of shape [3, H, W] after resizing and normalization.
    """
    images = []
    raw_arrays = []

    for path_str in paths3:
        resolved_path = resolve_path(path_str, base_dir)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found: {resolved_path}")

        with Image.open(resolved_path) as image:
            image = image.convert("L")
            resized_image = image.resize(RESIZE_HW, Image.BILINEAR)

            raw = np.array(resized_image).astype(np.float32)
            raw = norm01(raw)
            raw_arrays.append(raw)

            tensor = transforms.ToTensor()(resized_image)  # [1, H, W]
            images.append(tensor)

    x = torch.stack(images, dim=0)   # [K, 1, H, W]
    x = x.unsqueeze(0)               # [1, K, 1, H, W]

    raw_np = np.stack(raw_arrays, axis=0)
    return x, raw_np


# =========================================================
# Grad-CAM hook
# =========================================================
class GradCAMHook:
    """
    Hook helper for collecting activations and gradients from a target layer.
    """

    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = None
        self.bwd_handle = None

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def register(self) -> None:
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def remove(self) -> None:
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()


def compute_gradcam_from_hook(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Compute Grad-CAM maps from hook activations and gradients.

    Args:
        activations: Tensor of shape [B, C, h, w]
        gradients: Tensor of shape [B, C, h, w]
        out_hw: Output spatial size (H, W)

    Returns:
        NumPy array of shape [B, H, W] normalized to [0, 1].
    """
    if activations is None or gradients is None:
        raise ValueError("Hook data is missing.")
    if activations.dim() != 4 or gradients.dim() != 4:
        raise ValueError(
            f"Unexpected tensor shapes: activations={tuple(activations.shape)}, "
            f"gradients={tuple(gradients.shape)}"
        )

    weights = gradients.mean(dim=(2, 3), keepdim=True)      # [B, C, 1, 1]
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, h, w]
    cam = F.relu(cam)

    cam = cam - cam.amin(dim=(2, 3), keepdim=True)
    cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)

    cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)  # [B, H, W]
    return cam.detach().cpu().numpy()


# =========================================================
# Figure generation
# =========================================================
def make_gradcam_figure_for_sample(
    model: SymBrainRadModel,
    t1_paths_3: Sequence[str],
    t2_paths_3: Sequence[str],
    base_dir: str,
    title: str,
    out_path: str,
) -> None:
    """
    Generate and save a Grad-CAM figure for one paired T1/T2 sample.
    """
    model.eval()

    t1, t1_raw = load_slices_as_tensor(t1_paths_3, base_dir)
    t2, t2_raw = load_slices_as_tensor(t2_paths_3, base_dir)

    t1 = t1.to(DEVICE)
    t2 = t2.to(DEVICE)

    batch_size, num_slices, channels, height, width = t1.shape
    if num_slices != 3:
        raise ValueError(f"Expected exactly 3 slices, but got {num_slices}.")

    # Grad-CAM target layers
    t1_target = model.t1_cross_slice_encoder.backbone.features[10]
    t2_target = model.t2_cross_slice_encoder.backbone.features[10]

    t1_hook = GradCAMHook(t1_target)
    t2_hook = GradCAMHook(t2_target)
    t1_hook.register()
    t2_hook.register()

    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.requires_grad_(True)

    logit = model(t1, t2).view(-1)[0]
    logit.backward()

    t1_acts, t1_grads = t1_hook.activations, t1_hook.gradients
    t2_acts, t2_grads = t2_hook.activations, t2_hook.gradients

    t1_hook.remove()
    t2_hook.remove()

    # The backbone processes flattened B*K inputs.
    if t1_acts.shape[0] != batch_size * num_slices:
        raise ValueError(
            f"Unexpected T1 activation batch dimension: expected {batch_size * num_slices}, "
            f"got {t1_acts.shape[0]}"
        )
    if t2_acts.shape[0] != batch_size * num_slices:
        raise ValueError(
            f"Unexpected T2 activation batch dimension: expected {batch_size * num_slices}, "
            f"got {t2_acts.shape[0]}"
        )

    t1_cam_bk = compute_gradcam_from_hook(t1_acts, t1_grads, out_hw=(height, width))
    t2_cam_bk = compute_gradcam_from_hook(t2_acts, t2_grads, out_hw=(height, width))

    t1_cams = t1_cam_bk.reshape(batch_size, num_slices, height, width)[0]
    t2_cams = t2_cam_bk.reshape(batch_size, num_slices, height, width)[0]

    t1_overlay = np.stack(
        [overlay_cam_on_gray(t1_raw[i], t1_cams[i], alpha=0.45) for i in range(num_slices)],
        axis=0,
    )
    t2_overlay = np.stack(
        [overlay_cam_on_gray(t2_raw[i], t2_cams[i], alpha=0.45) for i in range(num_slices)],
        axis=0,
    )

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(11, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.18)

    for i in range(3):
        axes[0, i].imshow(t1_raw[i], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"T1 Slice {i + 1}", fontsize=11)

        axes[1, i].imshow(t1_overlay[i])
        axes[1, i].axis("off")

        axes[2, i].imshow(t2_raw[i], cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title(f"T2 Slice {i + 1}", fontsize=11)

        axes[3, i].imshow(t2_overlay[i])
        axes[3, i].axis("off")

    axes[0, 0].set_ylabel("T1 (raw)", fontsize=12)
    axes[1, 0].set_ylabel("T1 (Grad-CAM)", fontsize=12)
    axes[2, 0].set_ylabel("T2 (raw)", fontsize=12)
    axes[3, 0].set_ylabel("T2 (Grad-CAM)", fontsize=12)

    prob = torch.sigmoid(logit.detach()).item()
    fig.suptitle(f"{title} | prob={prob:.3f}", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_path}")


# =========================================================
# Main
# =========================================================
def select_eval_csv(candidates: Sequence[Path]) -> Path:
    """
    Return the first existing evaluation-detail CSV from the candidate list.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No evaluation-detail CSV found among: {list(candidates)}")


def load_model(ckpt_path: Path) -> SymBrainRadModel:
    """
    Load the trained model checkpoint.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = SymBrainRadModel(d_model=D_MODEL).to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()
    return model


def main() -> None:
    eval_csv = select_eval_csv(EVAL_CSV_CANDIDATES)
    model = load_model(CKPT_PATH)

    df = pd.read_csv(eval_csv)

    required_columns = {"paths"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {eval_csv}: {sorted(missing_columns)}")

    df["radscore"] = df["paths"].apply(extract_radscore_from_paths_cell)
    df["session_id"] = df["paths"].apply(extract_session_id)
    df["modality"] = df["paths"].apply(infer_modality_from_paths_cell)

    with open(DEBUG_PATH, "w", encoding="utf-8") as debug_file:
        debug_file.write(f"eval_csv={eval_csv}\n\n")
        debug_file.write("radscore counts:\n")
        debug_file.write(str(df["radscore"].value_counts(dropna=False)))
        debug_file.write("\n\nmodality counts:\n")
        debug_file.write(str(df["modality"].value_counts(dropna=False)))
        debug_file.write("\n\nexamples:\n")
        for i in range(min(10, len(df))):
            debug_file.write(
                f"idx={i} rs={df.loc[i, 'radscore']} "
                f"sid={df.loc[i, 'session_id']} mod={df.loc[i, 'modality']}\n"
            )
            debug_file.write(f"  paths={df.loc[i, 'paths']}\n")

    candidates = []
    for (session_id, radscore), group_df in df.groupby(["session_id", "radscore"]):
        if session_id is None or pd.isna(session_id) or radscore is None or pd.isna(radscore):
            continue

        modalities = set(group_df["modality"].tolist())
        if "T1" in modalities and "T2" in modalities:
            candidates.append((int(radscore), session_id))

    picked = {}
    for target_radscore in TARGET_RADSCORES:
        session_ids = [sid for (rs, sid) in candidates if rs == target_radscore]
        if len(session_ids) == 0:
            print(
                f"[WARN] radscore={target_radscore}: no session with both T1 and T2 found. "
                f"See debug file: {DEBUG_PATH}"
            )
            continue
        picked[target_radscore] = session_ids[0]

    if len(picked) == 0:
        raise RuntimeError(
            "Could not find sessions containing both T1 and T2 for the requested "
            f"radiology scores. See debug file: {DEBUG_PATH}"
        )

    base_dir = os.getcwd()

    for radscore, session_id in picked.items():
        sub_df = df[(df["session_id"] == session_id) & (df["radscore"] == radscore)]

        t1_row = sub_df[sub_df["modality"] == "T1"].iloc[0]
        t2_row = sub_df[sub_df["modality"] == "T2"].iloc[0]

        t1_paths4 = parse_paths_cell(t1_row["paths"])
        t2_paths4 = parse_paths_cell(t2_row["paths"])

        t1_paths3 = choose_three_slices(t1_paths4)
        t2_paths3 = choose_three_slices(t2_paths4)

        title = f"Radscore={radscore} | session={session_id}"
        out_path = OUT_DIR / f"gradcam_radscore{radscore}_session{session_id}.png"

        make_gradcam_figure_for_sample(
            model=model,
            t1_paths_3=t1_paths3,
            t2_paths_3=t2_paths3,
            base_dir=base_dir,
            title=title,
            out_path=str(out_path),
        )

    print("\nDone.")
    print(f"Debug saved to: {DEBUG_PATH}")
    print(f"Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
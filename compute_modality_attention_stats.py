import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from img_dataset import TwinMultiImageDataset
from img_model import SymBrainRadModel


# =========================
# Configuration
# =========================
RESULTS_DIR = Path("results/20260210-161302")
EVAL_CSV = RESULTS_DIR / "test_best_eval_detail.csv"
CKPT_PATH = RESULTS_DIR / "best_val_bal_acc_model.pt"
OUT_CSV = RESULTS_DIR / "modality_attn_stats.csv"
OUT_SUMMARY = RESULTS_DIR / "modality_attn_summary.csv"
OUT_PLOT1 = RESULTS_DIR / "modality_attn_box_by_type.png"
OUT_PLOT2 = RESULTS_DIR / "modality_attn_box_by_true.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match the training configuration.
D_MODEL = 128
BATCH_SIZE = 8
NUM_WORKERS = 0

IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((290, 290)),
        transforms.ToTensor(),
    ]
)


def configure_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def validate_input_files() -> None:
    """Validate that required input files exist."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    if not EVAL_CSV.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {EVAL_CSV}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {CKPT_PATH}")


def parse_paths(paths_str: str) -> List[str]:
    """
    Restore a serialized path string into a list of paths.

    Expected format:
        path1|path2|path3|...

    Args:
        paths_str: Serialized path string from the evaluation CSV.

    Returns:
        List of image paths.
    """
    if pd.isna(paths_str):
        return []
    return str(paths_str).split("|")


def modality_importance_from_attn(
    modal_attn: Union[torch.Tensor, Sequence[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert modality attention into modality-level importance statistics.

    Args:
        modal_attn:
            Either:
            - a tensor of shape [B, heads, 2, 2], or
            - a list/tuple of such tensors for multiple layers.

    Returns:
        t1_importance: Tensor of shape [B]
        t2_importance: Tensor of shape [B]
        t2_ratio: Tensor of shape [B], computed as:
                  t2_importance / (t1_importance + t2_importance)

    Interpretation:
        For the two modality tokens (T1, T2), column-wise attention is used to
        estimate how strongly each modality is referenced on average.
    """
    if isinstance(modal_attn, (list, tuple)):
        if len(modal_attn) == 0:
            raise ValueError("Received an empty modality attention list.")
        attn = modal_attn[-1]  # last layer: [B, heads, 2, 2]
    else:
        attn = modal_attn

    if attn.ndim != 4 or attn.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected attention shape [B, heads, 2, 2], but got {tuple(attn.shape)}."
        )

    # [B, heads, 2, 2] -> [B, 2, 2] by averaging across heads
    attn = attn.mean(dim=1)

    # Column-wise importance:
    # col0 = average attention directed to T1
    # col1 = average attention directed to T2
    col0 = attn[:, :, 0].mean(dim=1)  # [B]
    col1 = attn[:, :, 1].mean(dim=1)  # [B]

    denom = (col0 + col1).clamp_min(1e-8)
    t2_ratio = col1 / denom

    return col0, col1, t2_ratio


def add_summary(summary_rows: list, group_name: str, subset: pd.DataFrame) -> None:
    """
    Append summary statistics for a subset of samples.

    Args:
        summary_rows: List collecting summary dictionaries.
        group_name: Name of the current subset/group.
        subset: DataFrame containing the subset to summarize.
    """
    summary_rows.append(
        {
            "group": group_name,
            "N": len(subset),
            "t2_ratio_mean": float(np.mean(subset["modal_t2_ratio"])),
            "t2_ratio_std": float(np.std(subset["modal_t2_ratio"])),
            "t1_imp_mean": float(np.mean(subset["modal_t1_importance"])),
            "t2_imp_mean": float(np.mean(subset["modal_t2_importance"])),
        }
    )


def boxplot_by_group(
    df_plot: pd.DataFrame,
    group_col: str,
    title: str,
    save_path: Path,
) -> None:
    """
    Save a boxplot of modal_t2_ratio grouped by the given column.

    Args:
        df_plot: Input dataframe.
        group_col: Column used to define groups.
        title: Figure title.
        save_path: Output image path.
    """
    groups = []
    labels = []

    for key in sorted(df_plot[group_col].unique()):
        values = df_plot[df_plot[group_col] == key]["modal_t2_ratio"].values
        groups.append(values)
        labels.append(str(key))

    plt.figure(figsize=(10, 5))
    plt.boxplot(groups, tick_labels=labels, showmeans=True)
    plt.ylabel("T2 reliance ratio (modal_t2_ratio)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    configure_logging()
    validate_input_files()

    # =========================
    # 1) Load evaluation-detail CSV
    # =========================
    df = pd.read_csv(EVAL_CSV)

    required_columns = {"paths", "y_true", "type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {EVAL_CSV}: {sorted(missing_columns)}"
        )

    df["paths_list"] = df["paths"].apply(parse_paths)

    # =========================
    # 2) Build Dataset / DataLoader
    # =========================
    paths_list = df["paths_list"].tolist()
    labels_list = df["y_true"].astype(int).tolist()

    dataset = TwinMultiImageDataset(paths_list, labels_list, transform=IMG_TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # =========================
    # 3) Load model
    # =========================
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model = SymBrainRadModel(d_model=D_MODEL).to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()

    logging.info("Loaded checkpoint: %s", CKPT_PATH)
    logging.info("Number of samples: %d", len(dataset))
    logging.info("Device: %s", DEVICE)

    # =========================
    # 4) Extract modality attention statistics
    # =========================
    t1_importances = []
    t2_importances = []
    t2_ratios = []

    with torch.no_grad():
        for batch in loader:
            # TwinMultiImageDataset may return:
            #   (t1, t2, label, paths) or (t1, t2, label)
            if len(batch) == 4:
                t1, t2, _, _ = batch
            else:
                t1, t2, _ = batch

            t1 = t1.to(DEVICE)
            t2 = t2.to(DEVICE)

            # The model is expected to return auxiliary attention information.
            _, aux = model(t1, t2, return_aux=True, return_attn=True)

            modal_attn = aux.get("modal_attn", None)
            if modal_attn is None:
                raise RuntimeError(
                    "aux['modal_attn'] is None. Please ensure the model forward "
                    "method stores modality attention in aux when return_attn=True."
                )

            t1_imp, t2_imp, t2_ratio = modality_importance_from_attn(modal_attn)
            t1_importances.append(t1_imp.cpu().numpy())
            t2_importances.append(t2_imp.cpu().numpy())
            t2_ratios.append(t2_ratio.cpu().numpy())

    t1_importance = np.concatenate(t1_importances, axis=0)
    t2_importance = np.concatenate(t2_importances, axis=0)
    t2_ratio = np.concatenate(t2_ratios, axis=0)

    if len(t2_ratio) != len(df):
        raise ValueError(
            f"Sample count mismatch: extracted {len(t2_ratio)} attention values "
            f"for {len(df)} rows in the evaluation CSV."
        )

    # =========================
    # 5) Save per-sample results
    # =========================
    df_out = df.copy()
    df_out["modal_t1_importance"] = t1_importance
    df_out["modal_t2_importance"] = t2_importance
    df_out["modal_t2_ratio"] = t2_ratio

    df_out.to_csv(OUT_CSV, index=False)
    logging.info("Saved per-sample statistics: %s", OUT_CSV)

    # =========================
    # 6) Save summary statistics
    # =========================
    summary_rows = []

    add_summary(summary_rows, "ALL", df_out)

    for case_type in ["TP", "FN", "FP", "TN"]:
        subset = df_out[df_out["type"] == case_type]
        if len(subset) > 0:
            add_summary(summary_rows, f"type={case_type}", subset)

    for label in [0, 1]:
        subset = df_out[df_out["y_true"] == label]
        if len(subset) > 0:
            add_summary(summary_rows, f"true={label}", subset)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    logging.info("Saved summary statistics: %s", OUT_SUMMARY)
    logging.info("\n%s", summary_df.to_string(index=False))

    # =========================
    # 7) Save boxplots
    # =========================
    subset_by_type = df_out[df_out["type"].isin(["TP", "FN", "FP", "TN"])].copy()
    boxplot_by_group(
        subset_by_type,
        group_col="type",
        title="Modality attention: T2 reliance ratio by TP/FN/FP/TN",
        save_path=OUT_PLOT1,
    )
    logging.info("Saved plot: %s", OUT_PLOT1)

    subset_by_true = df_out[df_out["y_true"].isin([0, 1])].copy()
    boxplot_by_group(
        subset_by_true,
        group_col="y_true",
        title="Modality attention: T2 reliance ratio by true label (0/1)",
        save_path=OUT_PLOT2,
    )
    logging.info("Saved plot: %s", OUT_PLOT2)

    logging.info("Done.")


if __name__ == "__main__":
    main()
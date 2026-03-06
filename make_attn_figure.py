# make_attn_figure.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

path = ""
RESULTS_DIR = Path(path)
SEL_CSV = RESULTS_DIR / "selected_cases.csv"
ATTN_DIR = RESULTS_DIR / "attn_selected"
OUT_PATH = RESULTS_DIR / "figure_attn_4x3.png"

# Select one example for each case type.
ORDER = ["TP", "FN", "FP", "TN"]


def validate_inputs() -> None:
    """Validate required input files and directories."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    if not SEL_CSV.exists():
        raise FileNotFoundError(f"Selected cases CSV not found: {SEL_CSV}")
    if not ATTN_DIR.exists():
        raise FileNotFoundError(f"Attention image directory not found: {ATTN_DIR}")


def find_img(prefix: str, suffix: str) -> Optional[Path]:
    """
    Find an image file in the attention directory by prefix/suffix matching.

    Args:
        prefix: Expected filename prefix.
        suffix: Expected filename suffix.

    Returns:
        Matching file path if found, otherwise None.
    """
    for path in ATTN_DIR.iterdir():
        if path.name.startswith(prefix) and path.name.endswith(suffix):
            return path
    return None


def pick_representative_rows(df: pd.DataFrame) -> List[Optional[Tuple[int, pd.Series]]]:
    """
    Pick one representative row for each case type in ORDER.

    Returns:
        A list of (row_index, row_series) tuples or None if the case type is missing.
    """
    picked_rows: List[Optional[Tuple[int, pd.Series]]] = []

    for case_type in ORDER:
        subset = df[df["type"] == case_type]
        if subset.empty:
            print(f"[WARN] No samples for type={case_type}")
            picked_rows.append(None)
        else:
            row_index = int(subset.index[0])
            picked_rows.append((row_index, subset.iloc[0]))

    return picked_rows


def build_figure_inputs(
    picked_rows: List[Optional[Tuple[int, pd.Series]]]
) -> Tuple[List[List[Optional[Path]]], List[List[str]]]:
    """
    Build image-path and title matrices for the 4x3 figure layout.

    Returns:
        paths: 4x3 matrix of image paths
        titles: 4x3 matrix of subplot titles
    """
    paths: List[List[Optional[Path]]] = []
    titles: List[List[str]] = []

    for row_position, picked in enumerate(picked_rows):
        if picked is None:
            paths.append([None, None, None])
            titles.append([f"{ORDER[row_position]} (missing)", "", ""])
            continue

        row_index, row = picked
        case_type = row["type"]
        y_true = int(row["y_true"])
        y_pred = int(row["y_pred"])
        y_score = float(row["y_score"])

        # export_selected_attention.py uses:
        # base = f"{i:02d}_{typ}_y{y}_p{y_pred}_s{prob:.3f}"
        # where i is the original row index from selected_cases.csv.
        prefix = f"{row_index:02d}_{case_type}_"

        p_t1 = find_img(prefix, "_t1.png")
        p_t2 = find_img(prefix, "_t2.png")
        p_modal = find_img(prefix, "_modal.png")

        paths.append([p_t1, p_t2, p_modal])
        titles.append(
            [
                f"{case_type} | y={y_true}, ŷ={y_pred}, score={y_score:.3f}\nT1 slice attn",
                "T2 slice attn",
                "Modality attn (2×2)",
            ]
        )

    return paths, titles


def save_figure(
    paths: List[List[Optional[Path]]],
    titles: List[List[str]],
    out_path: Path,
) -> None:
    """
    Save the combined 4x3 attention figure.

    Args:
        paths: 4x3 matrix of image paths.
        titles: 4x3 matrix of subplot titles.
        out_path: Output figure path.
    """
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 14))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    for row in range(4):
        for col in range(3):
            ax = axes[row, col]
            img_path = paths[row][col]
            ax.axis("off")

            if img_path is None or not img_path.exists():
                ax.set_title("N/A", fontsize=10)
                continue

            image = mpimg.imread(img_path)
            ax.imshow(image)
            ax.set_title(titles[row][col], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    validate_inputs()

    df = pd.read_csv(SEL_CSV)

    required_columns = {"type", "y_true", "y_pred", "y_score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {SEL_CSV}: {sorted(missing_columns)}"
        )

    picked_rows = pick_representative_rows(df)
    paths, titles = build_figure_inputs(picked_rows)
    save_figure(paths, titles, OUT_PATH)

    print(f"Saved figure: {OUT_PATH}")


if __name__ == "__main__":
    main()
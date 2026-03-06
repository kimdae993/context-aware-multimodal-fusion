from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def parse_paths(paths_str: str) -> List[str]:
    """
    Parse a serialized path string into a list of image paths.

    Expected format:
        "a|b|c|d|e|f" -> [a, b, c, d, e, f]
    """
    return [part for part in str(paths_str).split("|") if part.strip()]


def make_grid(
    img_paths: Sequence[str],
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    """
    Render a 2x3 grid of grayscale images and save it.

    Args:
        img_paths: Sequence of image paths.
        out_path: Output image path.
        title: Optional figure title.
    """
    if len(img_paths) == 0:
        raise ValueError("No image paths were provided.")

    images = []
    for img_path in img_paths:
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as image:
            images.append(image.convert("L").copy())

    fig = plt.figure(figsize=(12, 4))
    for index, image in enumerate(images):
        ax = fig.add_subplot(2, 3, index + 1)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{index + 1}", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Render image grids from a CSV file containing serialized image paths."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV file (e.g. FN_confident_top30.csv).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where rendered figures will be saved.",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=30,
        help="Maximum number of rows to render.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if args.max_n <= 0:
        raise ValueError(f"`max_n` must be positive, got {args.max_n}.")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_columns = {"paths", "type", "y_true", "y_pred", "y_score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing_columns)}"
        )

    df = df.head(args.max_n)

    for row_index, row in df.iterrows():
        img_paths = parse_paths(row["paths"])
        title = (
            f"{row['type']}  "
            f"y_true={row['y_true']} "
            f"y_pred={row['y_pred']} "
            f"score={row['y_score']:.3f}"
        )
        output_name = f"{row_index:04d}_{row['type']}_score{row['y_score']:.3f}.png"
        out_path = out_dir / output_name

        try:
            make_grid(img_paths, out_path, title=title)
        except Exception as exc:
            print(f"[WARN] skip row {row_index}: {exc}")

    print(f"[OK] saved to {out_dir}")


if __name__ == "__main__":
    main()
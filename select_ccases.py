# select_cases.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results/20260210-161302")
CSV_NAME = "test_best_eval_detail.csv"  # Can be changed to, e.g., "test_final_eval_detail.csv"
N_PER_TYPE = 2  # Number of samples to select for each case type


def pick_representative_cases(
    df_subset: pd.DataFrame,
    case_type: str,
    n: int,
) -> pd.DataFrame:
    """
    Select representative samples from one confusion-matrix case type.

    Selection rule:
        - TP / FP: highest y_score first
        - TN / FN: lowest y_score first

    Args:
        df_subset: Subset dataframe for a specific case type.
        case_type: One of {"TP", "TN", "FP", "FN"}.
        n: Number of rows to select.

    Returns:
        A dataframe containing up to `n` selected rows.
    """
    if df_subset.empty:
        return df_subset

    if case_type in {"TP", "FP"}:
        return df_subset.sort_values("y_score", ascending=False).head(n)

    return df_subset.sort_values("y_score", ascending=True).head(n)


def main() -> None:
    csv_path = RESULTS_DIR / CSV_NAME
    out_path = RESULTS_DIR / "selected_cases.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if N_PER_TYPE <= 0:
        raise ValueError(f"`N_PER_TYPE` must be positive, got {N_PER_TYPE}.")

    df = pd.read_csv(csv_path)

    required_columns = {"type", "y_true", "y_pred", "y_score", "paths"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing_columns)}"
        )

    picked = []
    for case_type in ["TP", "FN", "FP", "TN"]:
        subset = df[df["type"] == case_type].copy()
        picked.append(pick_representative_cases(subset, case_type, N_PER_TYPE))

    out = pd.concat(picked, axis=0).reset_index(drop=True)
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out[["type", "y_true", "y_pred", "y_score", "paths"]])


if __name__ == "__main__":
    main()
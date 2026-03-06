from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"paths", "y_true", "y_score", "y_pred", "type"}
VALID_LABELS = {0, 1}
VALID_TYPES = {"TP", "TN", "FP", "FN"}


def configure_logging(verbose: bool = False) -> None:
    """Configure application-wide logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def confusion_type(y_true: int, y_pred: int) -> str:
    """Return confusion-matrix case type for a binary prediction."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def validate_binary_column(series: pd.Series, column_name: str, csv_path: Path) -> pd.Series:
    """Validate that a column contains only binary labels {0, 1}."""
    values = pd.to_numeric(series, errors="raise").astype(int)
    invalid_mask = ~values.isin(VALID_LABELS)
    if invalid_mask.any():
        invalid_values = sorted(values[invalid_mask].unique().tolist())
        raise ValueError(
            f"{csv_path}: column '{column_name}' must contain only {sorted(VALID_LABELS)}. "
            f"Found invalid values: {invalid_values}"
        )
    return values


def validate_score_column(series: pd.Series, csv_path: Path) -> pd.Series:
    """Validate that score values are numeric and finite."""
    values = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{csv_path}: column 'y_score' contains non-finite values.")
    return values


def load_eval_detail(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate an evaluation-detail CSV file.

    Expected columns:
        - paths
        - y_true
        - y_score
        - y_pred
        - type
    """
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    df = df.copy()
    df["paths"] = df["paths"].astype(str)
    df["y_true"] = validate_binary_column(df["y_true"], "y_true", csv_path)
    df["y_pred"] = validate_binary_column(df["y_pred"], "y_pred", csv_path)
    df["y_score"] = validate_score_column(df["y_score"], csv_path)
    df["type"] = df["type"].astype(str).str.strip().str.upper()

    unexpected_types = sorted(set(df["type"].unique()) - VALID_TYPES)
    if unexpected_types:
        logging.warning(
            "%s: unexpected values in 'type' column: %s. Recomputed labels will be used.",
            csv_path,
            unexpected_types,
        )

    df["type_recalc"] = [
        confusion_type(y_true, y_pred)
        for y_true, y_pred in zip(df["y_true"], df["y_pred"])
    ]

    mismatch = int((df["type"] != df["type_recalc"]).sum())
    if mismatch > 0:
        logging.warning(
            "%s: %d rows in 'type' did not match y_true/y_pred. Recomputed labels were applied.",
            csv_path,
            mismatch,
        )

    df["type"] = df["type_recalc"]
    return df


def apply_threshold_prediction(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Recompute y_pred and type using y_score and a supplied decision threshold.
    """
    result = df.copy()
    result["y_pred"] = (result["y_score"] >= threshold).astype(int)
    result["type_recalc"] = [
        confusion_type(y_true, y_pred)
        for y_true, y_pred in zip(result["y_true"], result["y_pred"])
    ]
    result["type"] = result["type_recalc"]
    return result


def add_analysis_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Add helper columns for false-positive / false-negative analysis.

    Added columns:
        - dist_to_thr: absolute distance from the decision threshold
        - confidence: model confidence in the predicted class
    """
    result = df.copy()
    result["dist_to_thr"] = np.abs(result["y_score"] - threshold)
    result["confidence"] = np.where(result["y_pred"] == 1, result["y_score"], 1.0 - result["y_score"])
    return result


def summarize(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Generate a compact summary dictionary for the current dataset."""
    return {
        "N": int(len(df)),
        "TP": int((df["type"] == "TP").sum()),
        "TN": int((df["type"] == "TN").sum()),
        "FP": int((df["type"] == "FP").sum()),
        "FN": int((df["type"] == "FN").sum()),
        "threshold": float(threshold),
        "positive_rate_true": float((df["y_true"] == 1).mean()) if len(df) else float("nan"),
        "positive_rate_pred": float((df["y_pred"] == 1).mean()) if len(df) else float("nan"),
        "mean_score_true_1": float(df.loc[df["y_true"] == 1, "y_score"].mean())
        if (df["y_true"] == 1).any()
        else float("nan"),
        "mean_score_true_0": float(df.loc[df["y_true"] == 0, "y_score"].mean())
        if (df["y_true"] == 0).any()
        else float("nan"),
    }


def save_case_table(df: pd.DataFrame, out_path: Path) -> None:
    """Save a dataframe to CSV with a stable column order."""
    preferred_columns = [
        "paths",
        "y_true",
        "y_score",
        "y_pred",
        "type",
        "type_recalc",
        "dist_to_thr",
        "confidence",
    ]
    existing = [col for col in preferred_columns if col in df.columns]
    remainder = [col for col in df.columns if col not in existing]
    df.loc[:, existing + remainder].to_csv(out_path, index=False)


def export_case_tables(df: pd.DataFrame, out_dir: Path, topk: int) -> None:
    """Export full tables, per-type tables, and top-k error subsets."""
    ensure_dir(out_dir)

    save_case_table(df, out_dir / "all_with_analysis.csv")

    for case_type in ("FN", "FP", "TP", "TN"):
        subset = df[df["type"] == case_type].copy()
        save_case_table(subset, out_dir / f"{case_type}.csv")

    if topk <= 0:
        logging.info("topk=%d, skipping Top-K exports.", topk)
        return

    fn = df[df["type"] == "FN"].copy()
    fp = df[df["type"] == "FP"].copy()

    topk_specs = [
        ("FN_confident_top", fn.sort_values("y_score", ascending=True)),
        ("FP_confident_top", fp.sort_values("y_score", ascending=False)),
        ("FN_near_thr_top", fn.sort_values("dist_to_thr", ascending=True)),
        ("FP_near_thr_top", fp.sort_values("dist_to_thr", ascending=True)),
        ("FN_by_confidence_top", fn.sort_values("confidence", ascending=False)),
        ("FP_by_confidence_top", fp.sort_values("confidence", ascending=False)),
    ]

    for prefix, subset in topk_specs:
        save_case_table(subset.head(topk), out_dir / f"{prefix}{topk}.csv")


def write_report(summary: Dict[str, float], out_path: Path) -> None:
    """Write a plain-text summary report."""
    lines = ["===== FP/FN Analysis Report ====="]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_input_files(results_dir: Path, pattern: str) -> list[Path]:
    """Find evaluation-detail CSV files under the result directory."""
    csv_list = sorted(results_dir.glob(pattern))
    if not csv_list:
        raise FileNotFoundError(
            f"No evaluation-detail CSV files found in '{results_dir}' with pattern '{pattern}'."
        )
    return csv_list


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze false positives / false negatives from evaluation-detail CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation-detail CSV files (e.g. results/20260210-133248).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used to interpret y_score (default: 0.5).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of rows to save for each Top-K error table (default: 30).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_eval_detail.csv",
        help="Glob pattern for input CSV files (default: '*_eval_detail.csv').",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="fpfn_analysis",
        help="Name of the output subdirectory created under --results-dir.",
    )
    parser.add_argument(
        "--recompute-pred-from-score",
        action="store_true",
        help="Recompute y_pred/type from y_score using --threshold.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def run(
    results_dir: Path,
    threshold: float,
    topk: int,
    pattern: str,
    output_subdir: str,
    recompute_pred_from_score: bool,
) -> None:
    """Run the end-to-end FP/FN analysis pipeline."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {results_dir}")
    if topk < 0:
        raise ValueError("--topk must be >= 0")

    csv_list = find_input_files(results_dir, pattern)

    logging.info("Found %d evaluation-detail file(s).", len(csv_list))
    for path in csv_list:
        logging.info(" - %s", path.name)

    analysis_root = results_dir / output_subdir
    ensure_dir(analysis_root)

    for csv_path in csv_list:
        output_dir = analysis_root / csv_path.stem
        ensure_dir(output_dir)

        df = load_eval_detail(csv_path)

        if recompute_pred_from_score:
            logging.info(
                "%s: recomputing y_pred/type from y_score with threshold=%.4f",
                csv_path.name,
                threshold,
            )
            df = apply_threshold_prediction(df, threshold=threshold)

        df = add_analysis_columns(df, threshold=threshold)

        summary = summarize(df, threshold=threshold)
        write_report(summary, output_dir / "report.txt")
        export_case_tables(df, out_dir=output_dir, topk=topk)

        logging.info("Saved analysis to: %s", output_dir)


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    run(
        results_dir=args.results_dir,
        threshold=args.threshold,
        topk=args.topk,
        pattern=args.pattern,
        output_subdir=args.output_subdir,
        recompute_pred_from_score=args.recompute_pred_from_score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
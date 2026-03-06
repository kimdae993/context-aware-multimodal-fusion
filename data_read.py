from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import pandas as pd


def data_prepare(folder_path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load metadata from a dataset folder.

    Expected file structure:
        <folder_path>/metadata.csv

    Expected columns in metadata.csv:
        - filename
        - line
        - rad_score
        - session

    Args:
        folder_path: Path to the dataset folder.

    Returns:
        A tuple containing:
            - metadata_filename_list: List of absolute/combined file paths
            - metadata_line_list: List of line values
            - metadata_radscore_list: List of radiology scores
            - metadata_session_list: List of session identifiers
    """
    folder = Path(folder_path)
    metadata_csv_path = folder / "metadata.csv"

    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

    metadata = pd.read_csv(metadata_csv_path)

    required_columns = {"filename", "line", "rad_score", "session"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {metadata_csv_path}: {sorted(missing_columns)}"
        )

    metadata_filename_list = metadata["filename"].tolist()
    metadata_line_list = metadata["line"].tolist()
    metadata_radscore_list = metadata["rad_score"].tolist()
    metadata_session_list = metadata["session"].tolist()

    metadata_filename_list = [
        str(folder / filename) for filename in metadata_filename_list
    ]

    return (
        metadata_filename_list,
        metadata_line_list,
        metadata_radscore_list,
        metadata_session_list,
    )


def is_numeric_string(value: str) -> bool:
    """
    Check whether a string-like value can be interpreted as a numeric value.

    Args:
        value: Input value to test.

    Returns:
        True if the value can be converted to float, otherwise False.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def data_rearrange(
    metadata_session_list: list,
    metadata_line_list: list,
    filepath_list: list,
    metadata_radscore_list: list,
):
    """
    Rearrange metadata by session.

    The function groups lines, file paths, and radiology scores by session.
    Only rows with non-missing values are retained, and only numeric
    radiology scores are included.

    Non-numeric scores such as 'Q', 'Q3', and '3Q' are excluded.

    Args:
        metadata_session_list: List of session identifiers.
        metadata_line_list: List of line values.
        filepath_list: List of file paths.
        metadata_radscore_list: List of radiology scores.

    Returns:
        A tuple containing:
            - session_to_lines: dict[session, list[line]]
            - session_to_filepaths: dict[session, list[filepath]]
            - session_to_radscore: dict[session, single_radscore]
    """
    session_to_lines: DefaultDict[str, list] = defaultdict(list)
    session_to_filepaths: DefaultDict[str, list] = defaultdict(list)
    session_to_radscore: DefaultDict[str, list] = defaultdict(list)

    for session, line, filepath, rad_score in zip(
        metadata_session_list,
        metadata_line_list,
        filepath_list,
        metadata_radscore_list,
    ):
        if (
            pd.notna(session)
            and pd.notna(line)
            and pd.notna(filepath)
            and pd.notna(rad_score)
        ):
            if is_numeric_string(rad_score):
                session_to_lines[session].append(line)
                session_to_filepaths[session].append(filepath)
                session_to_radscore[session].append(rad_score)

    session_to_lines = dict(session_to_lines)
    session_to_filepaths = dict(session_to_filepaths)
    session_to_radscore = dict(session_to_radscore)

    for session in session_to_radscore:
        session_to_radscore.update(
            {session: list(set(session_to_radscore[session]))[0]}
        )

    return session_to_lines, session_to_filepaths, session_to_radscore
import os
from collections import defaultdict
from typing import List, Tuple

import pandas as pd


REQUIRED_METADATA_COLUMNS = {"filename", "rad_score", "session"}


def data_prepare(folder_path: str) -> Tuple[List[str], List, List]:
    """Read SymBrain-style metadata and return absolute file paths."""
    metadata_csv_path = os.path.join(folder_path, "metadata.csv")
    metadata = pd.read_csv(metadata_csv_path)

    missing_cols = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing_cols:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing_cols)}")

    metadata_filename_list = metadata["filename"].tolist()
    metadata_radscore_list = metadata["rad_score"].tolist()
    metadata_session_list = metadata["session"].tolist()

    metadata_filename_list = [
        filename if os.path.isabs(str(filename)) else os.path.join(folder_path, str(filename))
        for filename in metadata_filename_list
    ]

    return metadata_filename_list, metadata_radscore_list, metadata_session_list


def data_rearrange(
    metadata_session_list: list,
    filepath_list: list,
    metadata_radscore_list: list,
):
    """Group image paths and radiology scores by imaging session."""
    session_to_items = defaultdict(list)

    for session, filepath, rad_score in zip(
        metadata_session_list,
        filepath_list,
        metadata_radscore_list,
    ):
        if pd.isna(session) or pd.isna(filepath) or pd.isna(rad_score):
            continue

        try:
            rad_score_str = str(int(float(rad_score))) if float(rad_score).is_integer() else str(rad_score)
        except Exception as exc:
            print(f"[WARN] line parsing failed: session={session}, filepath={filepath}, error={exc}")
            continue

        session_to_items[session].append({
            "filepath": filepath,
            "rad_score": rad_score_str,
        })

    session_to_filepaths = {}
    session_to_radscore = {}

    for session, items in session_to_items.items():
        session_to_filepaths[session] = [item["filepath"] for item in items]

        unique_rads = sorted(set(item["rad_score"] for item in items))
        if len(unique_rads) > 1:
            print(f"[WARN] Multiple rad_scores in session={session}: {unique_rads}")

        session_to_radscore[session] = unique_rads[0]

    return session_to_filepaths, session_to_radscore

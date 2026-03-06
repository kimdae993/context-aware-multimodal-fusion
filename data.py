from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from sklearn.model_selection import train_test_split

from data_read import data_prepare, data_rearrange
from extra import set_seed
from options import args_parser


def data_setting() -> Tuple[List[int], List[List[str]]]:
    """
    Prepare paired T1w/T2w file lists and corresponding radiology scores.

    Expected directory structure:
        <current working directory>/T1w
        <current working directory>/T2w

    Returns:
        radscore_list:
            List of integer radiology scores (only 1 or 5).
        filepath_list:
            List of concatenated file-path lists for each session.
            Each entry contains:
                [T1w file paths..., T2w file paths...]
    """
    t1w_folder_path = os.path.join(os.getcwd(), "T1w")
    t2w_folder_path = os.path.join(os.getcwd(), "T2w")

    if not os.path.isdir(t1w_folder_path):
        raise FileNotFoundError(f"T1w folder not found: {t1w_folder_path}")
    if not os.path.isdir(t2w_folder_path):
        raise FileNotFoundError(f"T2w folder not found: {t2w_folder_path}")

    (
        t1w_metadata_filename_list,
        t1w_metadata_line_list,
        t1w_metadata_radscore_list,
        t1w_metadata_session_list,
    ) = data_prepare(t1w_folder_path)

    (
        t2w_metadata_filename_list,
        t2w_metadata_line_list,
        t2w_metadata_radscore_list,
        t2w_metadata_session_list,
    ) = data_prepare(t2w_folder_path)

    (
        t1w_session_to_lines,
        t1w_session_to_filepaths,
        t1w_session_to_radscore,
    ) = data_rearrange(
        t1w_metadata_session_list,
        t1w_metadata_line_list,
        t1w_metadata_filename_list,
        t1w_metadata_radscore_list,
    )

    (
        t2w_session_to_lines,
        t2w_session_to_filepaths,
        t2w_session_to_radscore,
    ) = data_rearrange(
        t2w_metadata_session_list,
        t2w_metadata_line_list,
        t2w_metadata_filename_list,
        t2w_metadata_radscore_list,
    )

    # Keep only sessions available in T1w and collect paired T1w/T2w file paths
    t1w_t2w_file_concat_list_dict = {session: [] for session in t1w_session_to_filepaths}

    for session in t1w_session_to_filepaths:
        if session not in t2w_session_to_filepaths:
            continue

        if t1w_session_to_radscore[session] == "1" or t1w_session_to_radscore[session] == "5":
            t1w_t2w_file_concat_list_dict[session].extend(t1w_session_to_filepaths[session])
            t1w_t2w_file_concat_list_dict[session].extend(t2w_session_to_filepaths[session])

    # Remove sessions with empty file lists
    t1w_t2w_file_concat_list_dict = {
        session: file_list
        for session, file_list in t1w_t2w_file_concat_list_dict.items()
        if len(file_list) > 0
    }

    radscore_list: List[int] = []
    filepath_list: List[List[str]] = []

    for session in t1w_t2w_file_concat_list_dict:
        radscore_list.append(int(t1w_session_to_radscore[session]))
        filepath_list.append(t1w_t2w_file_concat_list_dict[session])

    return radscore_list, filepath_list


def data_split(
    data_list: Sequence,
    label_list: Sequence,
    seed: int,
):
    """
    Split data into train / validation / test sets using stratified sampling.

    Split ratios:
        - test: 20%
        - validation: 20% of the remaining training set

    Args:
        data_list: Input samples.
        label_list: Corresponding labels.
        seed: Random seed used for reproducibility.

    Returns:
        ((X_train, y_train), (X_validation, y_validation), (X_test, y_test))
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data_list,
        label_list,
        test_size=0.2,
        random_state=seed,
        stratify=label_list,
    )

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def train_test_data_split(
    data_list: Sequence,
    label_list: Sequence,
    seed: int,
):
    """
    Split data into train / test sets using stratified sampling.

    Args:
        data_list: Input samples.
        label_list: Corresponding labels.
        seed: Random seed used for reproducibility.

    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data_list,
        label_list,
        test_size=0.2,
        random_state=seed,
        stratify=label_list,
    )

    return (x_train, y_train), (x_test, y_test)


def main() -> None:
    """
    Example entry point for preparing the paired dataset.
    """
    args = args_parser()
    set_seed(args.seed)

    radscore_list, filepath_list = data_setting()

    if len(radscore_list) == 0 or len(filepath_list) == 0:
        print("No valid paired samples were found.")
        return

    print(f"Number of samples: {len(filepath_list)}")
    print(f"First radiology score: {radscore_list[0]}")
    print(f"First file-path list: {filepath_list[0]}")


if __name__ == "__main__":
    main()
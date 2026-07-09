import os
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from data_read import data_prepare, data_rearrange


def _validate_mode(mode: str) -> None:
    if mode not in {"1vs5", "1vs234vs5"}:
        raise ValueError("mode must be either '1vs5' or '1vs234vs5'.")


def _validate_data_dir(path: str, modality_name: str) -> None:
    metadata_path = os.path.join(path, "metadata.csv")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{modality_name} directory does not exist: {path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.csv was not found in {modality_name} directory: {metadata_path}")


def data_setting(
    T1w_data_foler_path: str = "INPUT_YOUR_T1W_FOLDER_PATH",
    T2w_data_foler_path: str = "INPUT_YOUR_T2W_FOLDER_PATH",
    mode: str = "1vs5",
    num_slices: int = 3,
) -> Tuple[List[int], List[List[str]]]:
    """Build paired T1w/T2w session-level samples.

    Each returned sample contains ``num_slices`` T1w file paths followed by
    ``num_slices`` T2w file paths. The function keeps only sessions available
    in both modalities. In ``1vs5`` mode, radiology scores 2, 3, and 4 are
    excluded from the binary task.
    """
    _validate_mode(mode)
    _validate_data_dir(T1w_data_foler_path, "T1w")
    _validate_data_dir(T2w_data_foler_path, "T2w")

    T1w_metadata_filename_list, T1w_metadata_radscore_list, T1w_metadata_session_list = data_prepare(T1w_data_foler_path)
    T2w_metadata_filename_list, T2w_metadata_radscore_list, T2w_metadata_session_list = data_prepare(T2w_data_foler_path)

    T1w_session_to_filepaths, T1w_session_to_radscore = data_rearrange(
        T1w_metadata_session_list,
        T1w_metadata_filename_list,
        T1w_metadata_radscore_list,
    )
    T2w_session_to_filepaths, _ = data_rearrange(
        T2w_metadata_session_list,
        T2w_metadata_filename_list,
        T2w_metadata_radscore_list,
    )

    radscore_list: List[int] = []
    filepath_list: List[List[str]] = []

    common_sessions = sorted(set(T1w_session_to_filepaths).intersection(T2w_session_to_filepaths))
    skipped_sessions = 0

    for session in common_sessions:
        rad_score = int(T1w_session_to_radscore[session])

        if mode == "1vs5" and rad_score not in {1, 5}:
            continue

        t1_paths = list(T1w_session_to_filepaths[session])
        t2_paths = list(T2w_session_to_filepaths[session])

        if len(t1_paths) != num_slices or len(t2_paths) != num_slices:
            skipped_sessions += 1
            continue

        radscore_list.append(rad_score)
        filepath_list.append(t1_paths + t2_paths)

    if len(radscore_list) == 0:
        raise RuntimeError(
            "No valid paired T1w/T2w sessions were found. Check metadata.csv, "
            "session IDs, mode, and num_slices."
        )

    if skipped_sessions > 0:
        print(f"[WARN] Skipped {skipped_sessions} sessions with an unexpected number of slices.")

    return radscore_list, filepath_list


def data_split(data_list, label_list, SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        data_list,
        label_list,
        test_size=0.2,
        random_state=SEED,
        stratify=label_list,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train,
    )

    return (X_train, y_train), (X_validation, y_validation), (X_test, y_test)


def train_test_data_split(data_list, label_list, SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        data_list,
        label_list,
        test_size=0.2,
        random_state=SEED,
        stratify=label_list,
    )

    return (X_train, y_train), (X_test, y_test)

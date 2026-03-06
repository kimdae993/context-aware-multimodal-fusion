from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader_check(dataloader: Any) -> None:
    """
    Print basic shape information for a dataloader batch.

    Expected batch format:
        (data, labels)

    Args:
        dataloader: PyTorch dataloader.
    """
    data_iter = iter(dataloader)
    data, labels = next(data_iter)

    print(f"Data: {data.shape}")
    print(f"Modality 1: {data[:, 0, :].shape}")
    print(f"Modality 2: {data[:, 1, :].shape}")
    print(f"Rad score: {labels.shape}\n{labels}\n")


def data_check(dataset: Any) -> None:
    """
    Print basic shape information for the first dataset sample.

    Expected dataset item format:
        (data, label)

    Args:
        dataset: PyTorch-style dataset.
    """
    print(f"Data[0]: {dataset[0][0].shape}")
    print(f"Label[0]: {dataset[0][1].shape}")


def save_experiment_status(
    experiment_status_txt_path: str,
    seed: int,
    train_bs: int,
    inference_bs: int,
    pos_weight: float,
    criterion: str,
    lr: float,
    wd: float,
    optimizer: str,
    epochs: int,
    max_grad_norm: float,
    thr: float,
) -> None:
    """
    Save experiment configuration to a text file.

    Args:
        experiment_status_txt_path: Output text file path.
        seed: Random seed.
        train_bs: Training batch size.
        inference_bs: Inference batch size.
        pos_weight: Positive class weight.
        criterion: Loss function name.
        lr: Learning rate.
        wd: Weight decay.
        optimizer: Optimizer name.
        epochs: Number of epochs.
        max_grad_norm: Gradient clipping norm.
        thr: Decision threshold.
    """
    output_path = Path(experiment_status_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"SEED: {seed}",
        f"Train Batch Size: {train_bs}",
        f"Inference Batch Size: {inference_bs}",
        f"Pos Weight: {pos_weight}",
        f"Criterion: {criterion}",
        f"Learning Rate: {lr}",
        f"Weight Decay: {wd}",
        "",
        f"Optimizer: {optimizer}",
        "",
        f"Epochs: {epochs}",
        f"Max Grad Norm: {max_grad_norm}",
        f"Threshold: {thr}",
    ]

    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def save_csv(
    results_folder: str,
    train_history: Mapping[str, Sequence[float]],
    validation_history: Mapping[str, Sequence[float]],
) -> None:
    """
    Save training/validation metric histories as CSV files.

    Args:
        results_folder: Root results directory.
        train_history: Dictionary of training metric histories.
        validation_history: Dictionary of validation metric histories.
    """
    file_name = {
        "loss": "Loss",
        "acc": "Accuracy",
        "bal_acc": "Balanced_Accuracy",
        "prec": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "per_class_acc": "Per_Class_Accuracy",
    }

    csv_folder = Path(results_folder) / "CSV"
    csv_folder.mkdir(parents=True, exist_ok=True)

    metrics = list(train_history.keys())

    for metric in metrics:
        if metric == "per_class_acc":
            continue

        if metric not in validation_history:
            raise KeyError(f"Metric '{metric}' is missing from validation_history.")

        df = pd.DataFrame(
            {
                "Train": train_history[metric],
                "Validation": validation_history[metric],
            }
        )
        df.to_csv(csv_folder / f"{file_name[metric]}.csv", index=False)


def save_plt(
    results_folder: str,
    train_history: Mapping[str, Sequence[float]],
    validation_history: Mapping[str, Sequence[float]],
) -> None:
    """
    Save training/validation metric histories as line plots.

    Args:
        results_folder: Root results directory.
        train_history: Dictionary of training metric histories.
        validation_history: Dictionary of validation metric histories.
    """
    file_name = {
        "loss": "Loss",
        "acc": "Accuracy",
        "bal_acc": "Balanced_Accuracy",
        "prec": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "per_class_acc": "Per_Class_Accuracy",
    }

    graph_folder = Path(results_folder) / "Graph"
    graph_folder.mkdir(parents=True, exist_ok=True)

    metrics = list(train_history.keys())

    for metric in metrics:
        if metric == "per_class_acc":
            continue

        if metric not in validation_history:
            raise KeyError(f"Metric '{metric}' is missing from validation_history.")

        plt.figure()
        plt.title(file_name[metric].replace("_", " "))
        plt.plot(range(len(train_history[metric])), train_history[metric], label="Train")
        plt.plot(
            range(len(validation_history[metric])),
            validation_history[metric],
            label="Validation",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(graph_folder / f"{file_name[metric]}.jpg")
        plt.close()
# train_kfold.py
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from test import eval_model


def _extract_labels(dataset) -> np.ndarray:
    """
    Extract labels for stratified K-fold splitting.

    This function first tries to use a dataset-level `labels` attribute,
    and falls back to reading labels sample-by-sample.

    Returns:
        NumPy array of shape [N].
    """
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if torch.is_tensor(labels):
            return labels.detach().cpu().numpy().astype(int)
        return np.asarray(labels, dtype=int)

    extracted_labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        _, _, y = sample
        extracted_labels.append(int(y))

    return np.asarray(extracted_labels, dtype=int)


def train_one_fold(
    fold_id: int,
    epochs: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion,
    threshold: float,
    best_model_save_path: str,
):
    """
    Train a single fold and track the best model by validation balanced accuracy.

    Args:
        fold_id: Fold index starting from 1.
        epochs: Number of training epochs.
        model: Model instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        device: Target device.
        optimizer: Optimizer.
        criterion: Loss function.
        threshold: Decision threshold for evaluation.
        best_model_save_path: Output path for the best checkpoint.

    Returns:
        (
            train_history,
            validation_history,
            best_val_bal_acc,
            best_state_dict,
            final_state_dict,
        )
    """
    if epochs <= 0:
        raise ValueError(f"`epochs` must be positive, got {epochs}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"`threshold` must be in [0, 1], got {threshold}.")

    train_history = {
        "loss": [],
        "acc": [],
        "bal_acc": [],
        "prec": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "per_class_acc": [],
    }

    validation_history = {
        "loss": [],
        "acc": [],
        "bal_acc": [],
        "prec": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "per_class_acc": [],
    }

    best_val_bal_acc = 0.0
    best_state_dict = None
    best_opt_state_dict = None  # kept for compatibility / future reuse

    best_model_path = Path(best_model_save_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()

        for batch in train_loader:
            t1, t2, label = batch

            t1 = t1.to(device)
            t2 = t2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(t1, t2).view(-1)
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()

        # Evaluate train / validation at the end of each epoch
        train_metrics = eval_model(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            thr=threshold,
            desc=f"[Fold {fold_id}] Train epoch {epoch} Inference",
            verbose=False,
            save_dir=None,
            prefix=None,
            save_plots=False,
            save_detail_csv=False,
            compute_ci=False,
        )

        val_metrics = eval_model(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            thr=threshold,
            desc=f"[Fold {fold_id}] Valid epoch {epoch} Inference",
            verbose=True,
            save_dir=None,
            prefix=None,
            save_plots=False,
            save_detail_csv=False,
            compute_ci=False,
        )

        if train_metrics.keys() != val_metrics.keys():
            raise ValueError("Train and validation metric keys do not match.")

        for metric_name in train_metrics:
            train_history[metric_name].append(train_metrics[metric_name])
            validation_history[metric_name].append(val_metrics[metric_name])

        # Save the fold-best model by validation balanced accuracy
        if val_metrics["bal_acc"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["bal_acc"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_opt_state_dict = copy.deepcopy(optimizer.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": best_opt_state_dict,
                    "fold": fold_id,
                    "epoch": epoch,
                    "best_val_bal_acc": best_val_bal_acc,
                },
                best_model_path,
            )

    final_state_dict = copy.deepcopy(model.state_dict())

    return (
        train_history,
        validation_history,
        best_val_bal_acc,
        best_state_dict,
        final_state_dict,
    )


def train_kfold_model(
    k: int,
    dataset,
    device: torch.device,
    model_fn: Callable[[], torch.nn.Module],
    optimizer_fn: Callable[[Any], torch.optim.Optimizer],
    criterion,
    threshold: float,
    epochs: int,
    batch_size: int,
    best_model_dir: str,
    shuffle: bool = True,
    random_state: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Train a model using stratified K-fold cross-validation.

    Args:
        k: Number of folds.
        dataset: Dataset returning either (t1, t2, label) or (t1, t2, label, paths).
        device: Target device.
        model_fn: Function returning a new model instance.
        optimizer_fn: Function that receives model parameters and returns an optimizer.
        criterion: Loss function.
        threshold: Decision threshold used during evaluation.
        epochs: Number of epochs per fold.
        batch_size: Dataloader batch size.
        best_model_dir: Directory for saving best checkpoints.
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed for K-fold splitting.
        num_workers: Number of dataloader workers.
        pin_memory: Whether to enable pinned memory in dataloaders.

    Returns:
        List of fold result dictionaries.
    """
    if k <= 1:
        raise ValueError(f"`k` must be greater than 1, got {k}.")
    if epochs <= 0:
        raise ValueError(f"`epochs` must be positive, got {epochs}.")
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"`threshold` must be in [0, 1], got {threshold}.")
    if len(dataset) == 0:
        raise ValueError("`dataset` is empty.")

    best_model_dir_path = Path(best_model_dir)
    best_model_dir_path.mkdir(parents=True, exist_ok=True)

    labels = _extract_labels(dataset)

    skf = StratifiedKFold(
        n_splits=k,
        shuffle=shuffle,
        random_state=random_state,
    )

    all_fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        print("=" * 50)
        print(f"[Fold {fold_id}/{k}] train size = {len(train_idx)}, val size = {len(val_idx)}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())

        best_model_save_path = best_model_dir_path / f"fold{fold_id}_best.pth"

        (
            train_hist,
            val_hist,
            best_val_bal_acc,
            best_state_dict,
            final_state_dict,
        ) = train_one_fold(
            fold_id=fold_id,
            epochs=epochs,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            threshold=threshold,
            best_model_save_path=str(best_model_save_path),
        )

        fold_result = {
            "fold": fold_id,
            "train_history": train_hist,
            "val_history": val_hist,
            "best_val_bal_acc": best_val_bal_acc,
            "best_model_state_dict": best_state_dict,
            "final_model_state_dict": final_state_dict,
            "best_model_path": str(best_model_save_path),
        }
        all_fold_results.append(fold_result)

    best_bal_accs = [result["best_val_bal_acc"] for result in all_fold_results]
    print("=" * 50)
    print(f"[K-Fold Summary] mean best val_bal_acc = {np.mean(best_bal_accs):.4f}")
    print(f"per fold best val_bal_acc = {[f'{score:.4f}' for score in best_bal_accs]}")

    return all_fold_results
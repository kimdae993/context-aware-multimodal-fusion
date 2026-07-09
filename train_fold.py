import copy
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from test_model import eval_model_by_loss_type


def train_one_fold(
    fold_id,
    epochs,
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    criterion,
    loss_type,
    num_classes,
    threshold,
    best_model_save_path,
):
    train_history = {
        "loss": [],
        "acc": [],
        "bal_acc": [],
        "prec": [],
        "recall": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "mcc": [],
        "per_class_acc": [],
    }

    validation_history = {key: [] for key in train_history}

    best_val_score = float("-inf")
    best_epoch = None
    best_state_dict = None
    standard_metric = "mcc"

    for epoch in range(1, epochs + 1):
        model.train()

        for t1, t2, label in train_loader:
            t1, t2 = t1.to(device), t2.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(t1, t2)

            if loss_type == "binary_bce":
                logits = logits.view(-1)
                loss = criterion(logits, label.float())
            elif loss_type == "multiclass_ce":
                if logits.dim() != 2 or logits.size(1) != num_classes:
                    raise ValueError(
                        "CrossEntropyLoss expects logits with shape [B, num_classes], "
                        f"got {tuple(logits.shape)} with num_classes={num_classes}."
                    )
                loss = criterion(logits, label.long())
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            loss.backward()
            optimizer.step()

        train_metrics = eval_model_by_loss_type(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            loss_type=loss_type,
            num_classes=num_classes,
            thr=threshold,
            desc=f"[Fold {fold_id}] Train epoch {epoch}",
            verbose=False,
        )

        val_metrics = eval_model_by_loss_type(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            loss_type=loss_type,
            num_classes=num_classes,
            thr=threshold,
            desc=f"[Fold {fold_id}] Valid epoch {epoch}",
            verbose=True,
            save_dir=None,
            prefix=None,
            save_plots=False,
            save_detail_csv=False,
        )

        for metric in train_metrics:
            train_history[metric].append(train_metrics[metric])
            validation_history[metric].append(val_metrics[metric])

        val_score = val_metrics[standard_metric]
        if val_score is None or np.isnan(val_score):
            val_score_for_selection = float("-inf")
        else:
            val_score_for_selection = float(val_score)

        if best_state_dict is None or val_score_for_selection > best_val_score:
            best_val_score = val_score_for_selection
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "fold": fold_id,
                    "epoch": best_epoch,
                    "best_val_bal_acc": best_val_score,  # backward-compatible key
                    "best_val_score": best_val_score,
                    "best_metric_name": standard_metric,
                    "loss_type": loss_type,
                    "num_classes": num_classes,
                },
                best_model_save_path,
            )

    final_state_dict = copy.deepcopy(model.state_dict())

    return (
        train_history,
        validation_history,
        best_val_score,
        best_state_dict,
        final_state_dict,
    )


def train_kfold_model(
    k,
    dataset,
    device,
    model_fn,
    optimizer_fn,
    criterion,
    loss_type,
    threshold,
    epochs,
    batch_size,
    best_model_dir,
    shuffle=True,
    random_state=42,
    num_workers=4,
):
    """Train models with stratified k-fold cross-validation.

    Class weights are intentionally computed outside this function and reused
    across folds. This preserves the global-train-pool weighting protocol.
    """
    os.makedirs(best_model_dir, exist_ok=True)
    criterion = criterion.to(device)

    if hasattr(dataset, "label"):
        labels = dataset.label.detach().cpu().numpy().astype(int)
    else:
        labels = np.asarray([int(dataset[i][2]) for i in range(len(dataset))])

    num_classes = int(labels.max()) + 1
    print(f"num_classes: {num_classes}")
    print(f"loss_type: {loss_type}")

    if num_classes < 2:
        raise ValueError("At least two classes are required.")
    if num_classes == 2 and loss_type != "binary_bce":
        raise ValueError("num_classes == 2 requires loss_type='binary_bce'.")
    if num_classes > 2 and loss_type != "multiclass_ce":
        raise ValueError("num_classes > 2 requires loss_type='multiclass_ce'.")

    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    all_fold_results = []
    pin_memory = device.type == "cuda"

    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        print("=" * 50)
        print(f"[Fold {fold_id}/{k}] train size = {len(train_idx)}, val size = {len(val_idx)}")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        best_model_save_path = os.path.join(best_model_dir, f"fold{fold_id}_best.pth")

        (
            train_hist,
            val_hist,
            best_val_score,
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
            loss_type=loss_type,
            num_classes=num_classes,
            threshold=threshold,
            best_model_save_path=best_model_save_path,
        )

        fold_result = {
            "fold": fold_id,
            "train_history": train_hist,
            "val_history": val_hist,
            "best_val_bal_acc": best_val_score,  # backward-compatible key
            "best_val_score": best_val_score,
            "best_metric_name": "mcc",
            "best_model_state_dict": best_state_dict,
            "final_model_state_dict": final_state_dict,
            "best_model_path": best_model_save_path,
            "loss_type": loss_type,
            "num_classes": num_classes,
        }
        all_fold_results.append(fold_result)

    best_scores = [fr["best_val_score"] for fr in all_fold_results]
    print("=" * 50)
    print(f"[K-Fold Summary] mean best val_mcc = {np.mean(best_scores):.4f}")
    print(f"per fold best val_mcc = {[f'{b:.4f}' for b in best_scores]}")

    return all_fold_results

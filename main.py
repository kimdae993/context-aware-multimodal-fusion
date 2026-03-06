from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data import data_setting, data_split
from extra import save_csv, save_experiment_status, save_plt, set_seed
from img_dataset import TwinMultiImageDataset
from img_model import SymBrainRadModel
from options import args_parser
from test import eval_model
from train_fold import train_kfold_model


def build_image_transform() -> transforms.Compose:
    """
    Build the image transform used for training and evaluation.
    """
    return transforms.Compose(
        [
            transforms.Resize((290, 290)),
            transforms.ToTensor(),
        ]
    )


def build_model(d_model: int, device: torch.device, num_classes: int = 1) -> SymBrainRadModel:
    """
    Create a model instance and move it to the target device.
    """
    return SymBrainRadModel(d_model=d_model, num_classes=num_classes).to(device)


def build_optimizer(
    params,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Build the optimizer used for training.
    """
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def main() -> None:
    args = args_parser()
    set_seed(seed=args.seed)

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    experiment_time = time.strftime("%Y%m%d-%H%M%S")
    results_folder = Path("results") / experiment_time
    results_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------
    radscore_list, filepath_list = data_setting()

    if len(radscore_list) == 0 or len(filepath_list) == 0:
        raise RuntimeError("No valid samples were returned by `data_setting()`.")

    img_train, img_validation, img_test = data_split(
        data_list=filepath_list,
        label_list=radscore_list,
        seed=args.seed,
    )

    img_transform = build_image_transform()

    # Test dataset / dataloader
    img_test_ds = TwinMultiImageDataset(
        img_test[0],
        img_test[1],
        transform=img_transform,
    )
    inference_bs = args.bs
    img_test_loader = DataLoader(
        img_test_ds,
        batch_size=inference_bs,
        shuffle=False,
    )

    # Full training dataset for k-fold training: train + validation
    train_paths = img_train[0] + img_validation[0]
    train_labels = img_train[1] + img_validation[1]
    full_train_ds = TwinMultiImageDataset(
        train_paths,
        train_labels,
        transform=img_transform,
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Save model structure
    # ------------------------------------------------------------------
    model_for_log = build_model(
        d_model=args.D,
        device=device,
        num_classes=args.num_classes,
    )

    model_txt_path = results_folder / "model.txt"
    with model_txt_path.open("w", encoding="utf-8") as model_txt:
        model_txt.write(str(model_for_log))

    print(model_for_log)

    # ------------------------------------------------------------------
    # Compute class weight for BCEWithLogitsLoss
    # ------------------------------------------------------------------
    labels = torch.as_tensor(full_train_ds.labels, dtype=torch.long)
    num_classes = int(labels.max().item()) + 1

    if num_classes != 2:
        raise ValueError(
            "BCEWithLogitsLoss is intended here for binary classification with labels {0, 1}."
        )

    class_counts = torch.bincount(labels, minlength=num_classes)
    neg_count = class_counts[0]
    pos_count = class_counts[1].clamp_min(1)

    pos_weight = (neg_count / pos_count).to(torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    max_grad_norm = args.grad_norm

    # Dummy optimizer for experiment logging only
    optimizer_for_log = build_optimizer(
        model_for_log.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    experiment_status_txt_path = results_folder / "experiment_status.txt"
    save_experiment_status(
        str(experiment_status_txt_path),
        args.seed,
        args.bs,
        inference_bs,
        pos_weight,
        criterion,
        lr,
        wd,
        optimizer_for_log,
        epochs,
        max_grad_norm,
        args.thr,
    )

    del optimizer_for_log
    del model_for_log

    # ------------------------------------------------------------------
    # K-fold training
    # ------------------------------------------------------------------
    k_folds = args.fold
    checkpoints_dir = results_folder / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    all_fold_results = train_kfold_model(
        k=k_folds,
        dataset=full_train_ds,
        device=device,
        model_fn=lambda: build_model(d_model=args.D, device=device),
        optimizer_fn=lambda params: build_optimizer(params, lr=lr, weight_decay=wd),
        criterion=criterion,
        threshold=args.thr,
        epochs=epochs,
        batch_size=args.bs,
        best_model_dir=str(checkpoints_dir),
    )

    # ------------------------------------------------------------------
    # Save fold-wise training logs
    # ------------------------------------------------------------------
    for fold_result in all_fold_results:
        fold_id = fold_result["fold"]
        fold_folder = results_folder / f"fold{fold_id}"
        fold_folder.mkdir(parents=True, exist_ok=True)

        train_history = fold_result["train_history"]
        val_history = fold_result["val_history"]

        save_csv(str(fold_folder), train_history, val_history)
        save_plt(str(fold_folder), train_history, val_history)

    # ------------------------------------------------------------------
    # Select best fold
    # ------------------------------------------------------------------
    best_idx = int(np.argmax([result["best_val_bal_acc"] for result in all_fold_results]))
    best_fold_result = all_fold_results[best_idx]

    print(
        f"Best fold: {best_fold_result['fold']} "
        f"(val_bal_acc={best_fold_result['best_val_bal_acc']:.4f})"
    )
    print(f"Best checkpoint path: {best_fold_result['best_model_path']}")

    encoder = build_model(d_model=args.D, device=device)

    # ------------------------------------------------------------------
    # Evaluate the best validation-balanced-accuracy model on the test set
    # ------------------------------------------------------------------
    encoder.load_state_dict(best_fold_result["best_model_state_dict"])

    test_result = eval_model(
        encoder,
        img_test_loader,
        device,
        criterion,
        desc="Test (best val_bal_acc model of best fold)",
        save_dir=str(results_folder),
        prefix="test_best",
        save_plots=True,
        save_detail_csv=True,
        ci_seed=args.seed,
        save_attention=True,
        max_attn_save=20,
    )

    # ------------------------------------------------------------------
    # Save the best model checkpoint for reuse
    # ------------------------------------------------------------------
    best_model_save_path = results_folder / "best_val_bal_acc_model.pt"
    torch.save(
        {
            "model_state_dict": best_fold_result["best_model_state_dict"],
            "fold": best_fold_result["fold"],
            "best_val_bal_acc": best_fold_result["best_val_bal_acc"],
        },
        best_model_save_path,
    )

    print(f"Saved best model checkpoint: {best_model_save_path}")


if __name__ == "__main__":
    main()
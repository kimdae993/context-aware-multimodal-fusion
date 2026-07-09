import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data import data_setting, data_split
from extra import (
    save_csv,
    save_experiment_status,
    save_per_class_csv,
    save_per_class_plt,
    save_plt,
    set_seed,
)
from img_dataset import TwinMultiImageDataset
from img_model import SymBrainRadModel
from options import args_parser
from test_model import eval_model_by_loss_type
from train_fold import train_kfold_model


def _format_metric_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def _write_test_result(f, title, result):
    f.write(f"----- {title} -----\n")
    for metric, value in result.items():
        if metric == "per_class_acc":
            f.write("per_class_acc:\n")
            for class_idx in sorted(value.keys()):
                class_acc = value[class_idx]
                if isinstance(class_acc, float) and np.isnan(class_acc):
                    f.write(f"  class {class_idx}: nan\n")
                else:
                    f.write(f"  class {class_idx}: {class_acc:.4f}\n")
        else:
            f.write(f"{metric}:\t{_format_metric_value(value)}\n")
    f.write("\n")


def _build_criterion(labels: torch.Tensor, num_classes: int):
    """Build one global criterion from the train+validation pool.

    This intentionally does not compute fold-specific class weights.
    """
    class_counts = torch.bincount(labels, minlength=num_classes)

    if num_classes == 2:
        neg_count = class_counts[0]
        pos_count = class_counts[1].clamp_min(1)
        pos_weight = (neg_count / pos_count).to(torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_type = "binary_bce"
        criterion_for_log = f"BCEWithLogitsLoss(pos_weight={float(pos_weight):.6f})"
        print("Using BCEWithLogitsLoss")
        print(f"pos_weight: {float(pos_weight):.4f}")
        return criterion, loss_type, criterion_for_log, pos_weight, None

    class_counts_safe = class_counts.clamp_min(1)
    class_weights = labels.numel() / (num_classes * class_counts_safe)
    class_weights = class_weights.to(torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss_type = "multiclass_ce"
    criterion_for_log = f"CrossEntropyLoss(weight={class_weights.detach().cpu().tolist()})"
    print("Using CrossEntropyLoss")
    print(f"class_weights: {class_weights.detach().cpu().tolist()}")
    return criterion, loss_type, criterion_for_log, None, class_weights


def _make_model(args, num_classes, device):
    model_num_classes = 1 if num_classes == 2 else num_classes
    return SymBrainRadModel(
        d_model=args.D,
        num_slices=args.num_slices,
        slice_nhead=args.slice_nhead,
        slice_layers=args.slice_layers,
        slice_dropout=args.slice_dropout,
        mlp_ratio=args.mlp_ratio,
        modal_nhead=args.modal_nhead,
        modal_layers=args.modal_layers,
        modal_dropout=args.modal_dropout,
        cls_dropout=args.cls_dropout,
        num_classes=model_num_classes,
    ).to(device)


def _empty_metric_history():
    return {
        "loss": [],
        "acc": [],
        "bal_acc": [],
        "prec": [],
        "recall": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
        "mcc": [],
        "roc_auc": [],
        "pr_auc": [],
    }


def main():
    args = args_parser()
    set_seed(seed=args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    experiment_time = time.strftime("%Y%m%d-%H%M%S")
    experiment_mode = args.mode
    results_folder = os.path.join(args.out_dir, experiment_mode, experiment_time)
    os.makedirs(results_folder, exist_ok=True)

    radscore_list, filepath_list = data_setting(
        T1w_data_foler_path=args.t1_dir,
        T2w_data_foler_path=args.t2_dir,
        mode=experiment_mode,
        num_slices=args.num_slices,
    )

    img_train, img_validation, img_test = data_split(
        data_list=filepath_list,
        label_list=radscore_list,
        SEED=args.seed,
    )

    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    train_paths = img_train[0] + img_validation[0]
    train_labels = img_train[1] + img_validation[1]

    full_train_ds = TwinMultiImageDataset(
        train_paths,
        train_labels,
        transform=img_transform,
        mode=experiment_mode,
        num_slices=args.num_slices,
    )
    img_test_ds = TwinMultiImageDataset(
        img_test[0],
        img_test[1],
        transform=img_transform,
        mode=experiment_mode,
        num_slices=args.num_slices,
    )

    pin_memory = device.type == "cuda"
    img_test_loader = DataLoader(
        img_test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    labels = full_train_ds.label.detach().cpu().long()
    num_classes = int(labels.max().item()) + 1
    if num_classes < 2:
        raise ValueError("At least two classes are required.")

    label_counts = Counter(full_train_ds.label.tolist())
    print(f"Train/validation label counts: {dict(label_counts)}")
    print(f"Test label counts: {dict(Counter(img_test_ds.label.tolist()))}")
    print(f"num_classes: {num_classes}")

    criterion, loss_type, criterion_for_log, pos_weight, class_weights = _build_criterion(
        labels=labels,
        num_classes=num_classes,
    )

    encoder_for_log = _make_model(args, num_classes=num_classes, device=device)
    with open(os.path.join(results_folder, "model.txt"), "w") as model_txt:
        model_txt.write(str(encoder_for_log))
    print(encoder_for_log)

    lr = args.lr
    wd = args.wd
    optimizer_for_log = torch.optim.AdamW(encoder_for_log.parameters(), lr=lr, weight_decay=wd)

    save_experiment_status(
        experiment_status_txt_path=os.path.join(results_folder, "experiment_status.txt"),
        seed=args.seed,
        train_bs=args.bs,
        inference_bs=args.bs,
        criterion=criterion_for_log,
        lr=lr,
        wd=wd,
        optimizer=optimizer_for_log,
        epochs=args.epochs,
        thr=args.thr,
        loss_type=loss_type,
        num_classes=num_classes,
        pos_weight=pos_weight,
        class_weights=class_weights,
    )

    del optimizer_for_log
    del encoder_for_log

    def build_model():
        return _make_model(args, num_classes=num_classes, device=device)

    def build_optimizer(params):
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    checkpoints_dir = os.path.join(results_folder, "checkpoints")
    all_fold_results = train_kfold_model(
        k=args.fold,
        dataset=full_train_ds,
        device=device,
        model_fn=build_model,
        optimizer_fn=build_optimizer,
        criterion=criterion,
        threshold=args.thr,
        epochs=args.epochs,
        batch_size=args.bs,
        best_model_dir=checkpoints_dir,
        loss_type=loss_type,
        random_state=args.seed,
        num_workers=args.num_workers,
    )

    criterion = criterion.to(device)

    for state_key in ["best_model_state_dict", "final_model_state_dict"]:
        test_fold_result = _empty_metric_history()
        result_tag = state_key.replace("_state_dict", "")

        for fold_result in all_fold_results:
            fold_id = fold_result["fold"]
            fold_folder = os.path.join(results_folder, f"fold{fold_id}")
            os.makedirs(fold_folder, exist_ok=True)

            train_history = fold_result["train_history"]
            val_history = fold_result["val_history"]

            save_csv(fold_folder, train_history, val_history)
            save_per_class_csv(
                fold_folder,
                train_history["per_class_acc"],
                file_name="Train_Class_Accuracy",
                mode=args.mode,
            )
            save_per_class_csv(
                fold_folder,
                val_history["per_class_acc"],
                file_name="Validation_Class_Accuracy",
                mode=args.mode,
            )
            save_plt(fold_folder, train_history, val_history)
            save_per_class_plt(
                fold_folder,
                train_history["per_class_acc"],
                file_name="Train_Class_Accuracy",
            )
            save_per_class_plt(
                fold_folder,
                val_history["per_class_acc"],
                file_name="Validation_Class_Accuracy",
            )

            model = _make_model(args, num_classes=num_classes, device=device)
            model.load_state_dict(fold_result[state_key])

            fold_test_result = eval_model_by_loss_type(
                model=model,
                loader=img_test_loader,
                device=device,
                criterion=criterion,
                loss_type=loss_type,
                num_classes=num_classes,
                thr=args.thr,
                desc=f"Fold {fold_id} test ({result_tag})",
                verbose=True,
                save_dir=results_folder,
                prefix=f"fold{fold_id}_{result_tag}_test",
                save_plots=False,
                save_detail_csv=False,
                seed=args.seed,
            )

            with open(os.path.join(fold_folder, f"test_result_{result_tag}.txt"), "w") as f:
                _write_test_result(f, f"Test Result ({result_tag})", fold_test_result)

            for metric in test_fold_result:
                test_fold_result[metric].append(fold_test_result[metric])

        mean_std_test_fold_result = {}
        for metric, values in test_fold_result.items():
            mean_std_test_fold_result[f"{metric}_mean"] = float(np.nanmean(values))
            mean_std_test_fold_result[f"{metric}_std"] = float(np.nanstd(values, ddof=1))

        with open(os.path.join(results_folder, f"test_{result_tag}_fold_mean_std_result.txt"), "w") as f:
            _write_test_result(f, "Fold Mean & Std Result", mean_std_test_fold_result)

    best_idx = int(np.argmax([fr["best_val_score"] for fr in all_fold_results]))
    best_fold_result = all_fold_results[best_idx]
    print(
        f"Best fold: {best_fold_result['fold']} "
        f"(val_{best_fold_result['best_metric_name']}={best_fold_result['best_val_score']:.4f})"
    )
    print(f"Best checkpoint path: {best_fold_result['best_model_path']}")

    encoder = _make_model(args, num_classes=num_classes, device=device)
    encoder.load_state_dict(best_fold_result["best_model_state_dict"])

    test_result_best = eval_model_by_loss_type(
        model=encoder,
        loader=img_test_loader,
        device=device,
        criterion=criterion,
        loss_type=loss_type,
        num_classes=num_classes,
        thr=args.thr,
        desc="Test (best validation-MCC model of best fold)",
        verbose=True,
        save_dir=os.path.join(results_folder, "Best_result"),
        prefix="test_best",
        save_plots=True,
        save_detail_csv=True,
        seed=args.seed,
        save_gradcam=args.save_gradcam,
        gradcam_alpha=0.6,
        gradcam_colormap="jet",
        gradcam_mask_background=False,
        gradcam_mask_threshold=0.03,
        gradcam_mask_closing_iterations=2,
        gradcam_mask_dilation_iterations=1,
        gradcam_cam_cutoff=0.05,
        gradcam_cam_gamma=0.70,
        gradcam_save_mask=False,
    )

    test_result_save_txt_path = os.path.join(results_folder, "test_result.txt")
    with open(test_result_save_txt_path, "w") as f:
        f.write(f"loss_type: {loss_type}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"best_fold: {best_fold_result['fold']}\n")
        f.write(f"best_metric_name: {best_fold_result['best_metric_name']}\n")
        f.write(f"best_val_score: {best_fold_result['best_val_score']:.4f}\n")
        f.write(f"best_checkpoint_path: {best_fold_result['best_model_path']}\n\n")
        _write_test_result(
            f,
            "Test Result (best validation-MCC model of best fold)",
            test_result_best,
        )

    best_model_save_path = os.path.join(results_folder, "best_val_mcc_model.pt")
    torch.save(
        {
            "model_state_dict": best_fold_result["best_model_state_dict"],
            "fold": best_fold_result["fold"],
            "best_val_score": best_fold_result["best_val_score"],
            "best_metric_name": best_fold_result["best_metric_name"],
            "loss_type": loss_type,
            "num_classes": num_classes,
            "model_num_classes": 1 if num_classes == 2 else num_classes,
        },
        best_model_save_path,
    )

    print(f"Test result saved to: {test_result_save_txt_path}")
    print(f"Best model saved to: {best_model_save_path}")


if __name__ == "__main__":
    main()

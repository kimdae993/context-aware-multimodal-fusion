import numpy as np
import random, torch
import matplotlib.pyplot as plt
import os
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def save_experiment_status(
    experiment_status_txt_path,
    seed,
    train_bs,
    inference_bs,
    criterion,
    lr,
    wd,
    optimizer,
    epochs,
    thr=None,
    loss_type=None,
    num_classes=None,
    pos_weight=None,
    class_weights=None,
):
    with open(experiment_status_txt_path, "w") as f:
        f.write(f"SEED: {seed}\n")
        f.write(f"Train Batch Size: {train_bs}\n")
        f.write(f"Inference Batch Size: {inference_bs}\n")

        if loss_type is not None:
            f.write(f"Loss Type: {loss_type}\n")

        if num_classes is not None:
            f.write(f"Num Classes: {num_classes}\n")

        if pos_weight is not None:
            f.write(f"Pos Weight: {pos_weight}\n")

        if class_weights is not None:
            if torch.is_tensor(class_weights):
                class_weights_to_write = class_weights.detach().cpu().tolist()
            else:
                class_weights_to_write = class_weights
            f.write(f"Class Weights: {class_weights_to_write}\n")

        f.write(f"Criterion: {criterion}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Weight Decay: {wd}\n\n")
        f.write(f"Optimizer: {optimizer}\n\n")
        f.write(f"Epochs: {epochs}\n")

        if thr is not None:
            f.write(f"Threshold: {thr}\n")
    
    
    
def save_csv(results_folder, train_history, validation_history):
    file_name = {
        "loss": "Loss",
        "acc": "Accuracy",
        "bal_acc": "Balanced_Accuracy",
        "prec": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "mcc": "MCC",
        "per_class_acc": "Per_Class_Accuracy",
    }

    csv_folder = f"{results_folder}/CSV"
    os.makedirs(csv_folder, exist_ok=True)

    metrics = list(train_history.keys())

    for metric in metrics:
        if metric == "per_class_acc":
            continue

        df = pd.DataFrame({
            "Train": train_history[metric],
            "Validation": validation_history[metric],
        })

        save_name = file_name.get(metric, metric)
        df.to_csv(f"{csv_folder}/{save_name}.csv", index_label="epoch")

def save_per_class_csv(results_folder, train_per_class_acc, file_name, mode):
    csv_folder = f"{results_folder}/CSV"
    os.makedirs(csv_folder, exist_ok=True)
    
    labels = list(train_per_class_acc[0].keys())
    epochs = range(len(train_per_class_acc))
    labels_acc = {label:[] for label in labels}

    for label_acc in train_per_class_acc:
        for label in labels:
            labels_acc[label].append(label_acc[label])
    
    if mode == "1vs5":
        csv_colums = ["Rad score 1", "Rad score 5"]
        csv_values = [labels_acc[0], labels_acc[1]]
    elif mode == "1vs234vs5":
        csv_colums = ["Rad score 1", "Rad score 2/3/4", "Rad score 5"]
        csv_values = [labels_acc[0], labels_acc[1], labels_acc[2]]
    csv_data = {column: [] for column in csv_colums}
    for i in range(len(csv_colums)):
        csv_data[csv_colums[i]] = csv_values[i]
    
    df = pd.DataFrame(csv_data, index=epochs, columns=csv_colums)
    df.to_csv(f"{csv_folder}/{file_name}.csv")

def save_plt(results_folder, train_history, validation_history):
    file_name = {
        "loss": "Loss",
        "acc": "Accuracy",
        "bal_acc": "Balanced_Accuracy",
        "prec": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "mcc": "MCC",
        "per_class_acc": "Per_Class_Accuracy",
    }

    graph_folder = f"{results_folder}/Graph"
    os.makedirs(graph_folder, exist_ok=True)

    metrics = list(train_history.keys())

    for metric in metrics:
        if metric == "per_class_acc":
            continue

        save_name = file_name.get(metric, metric)

        plt.figure()
        plt.title(save_name.replace("_", " "))

        plt.plot(
            range(1, len(train_history[metric]) + 1),
            train_history[metric],
            label="Train",
        )

        plt.plot(
            range(1, len(validation_history[metric]) + 1),
            validation_history[metric],
            label="Validation",
        )

        plt.xlabel("Epoch")
        plt.ylabel(save_name.replace("_", " "))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{graph_folder}/{save_name}.jpg", dpi=200)
        plt.close()
            
def save_per_class_plt(
    results_folder,
    per_class_acc_history,
    file_name,
    class_names=None,
):
    """
    per_class_acc_history 예시:
    [
        {0: 0.8, 1: 0.7, 2: 0.6},
        {0: 0.9, 1: 0.6, 2: 0.7},
        ...
    ]
    """
    graph_folder = f"{results_folder}/Graph"
    os.makedirs(graph_folder, exist_ok=True)

    if len(per_class_acc_history) == 0:
        return

    labels = sorted(per_class_acc_history[0].keys())

    if class_names is None:
        class_names = {label: f"Class {label}" for label in labels}

    epochs = range(1, len(per_class_acc_history) + 1)

    plt.figure()
    plt.title(file_name.replace("_", " "))

    for label in labels:
        values = [
            epoch_acc.get(label, np.nan)
            for epoch_acc in per_class_acc_history
        ]

        plt.plot(
            epochs,
            values,
            label=class_names.get(label, f"Class {label}"),
        )

    plt.xlabel("Epoch")
    plt.ylabel("Per-class Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{graph_folder}/{file_name}.jpg", dpi=200)
    plt.close()    
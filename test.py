# test.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# -----------------------------
# Plot helpers
# -----------------------------
def _plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str,
) -> None:
    """
    Save an ROC curve figure.

    If only one class is present in `y_true`, the figure is still saved,
    but the ROC curve itself is not computed.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    plt.figure()

    if len(np.unique(y_true)) < 2:
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title} (ROC undefined: only one class present)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str,
) -> None:
    """
    Save a confusion matrix figure for binary classification.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["0", "1"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def _save_eval_detail_csv(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """
    Save per-sample evaluation details as CSV.

    Columns:
        - paths
        - y_true
        - y_score
        - y_pred
        - type (TP / TN / FP / FN)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    def conf_type(t: int, p: int) -> str:
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 0:
            return "TN"
        if t == 0 and p == 1:
            return "FP"
        return "FN"

    types = np.array([conf_type(t, p) for t, p in zip(y_true, y_pred)], dtype=object)

    import csv

    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["paths", "y_true", "y_score", "y_pred", "type"])
        for t, s, pr, tp in zip(y_true, y_score, y_pred, types):
            writer.writerow([int(t), float(s), int(pr), tp])


# -----------------------------
# Metric helpers
# -----------------------------
def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute PR-AUC safely for binary classification.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def _safe_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Compute standard binary classification metrics safely.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return (
            float("nan"),  # precision
            float("nan"),  # recall
            float("nan"),  # f1
            float("nan"),  # balanced_accuracy
            float("nan"),  # roc_auc
        )

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)

    return float(precision), float(recall), float(f1), float(bal_acc), float(roc_auc)


def _nan_percentile_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute a percentile confidence interval while ignoring NaNs.
    """
    values_np = np.asarray(values, dtype=float)

    if np.all(np.isnan(values_np)):
        return float("nan"), float("nan")

    lo = np.nanpercentile(values_np, 100 * (alpha / 2))
    hi = np.nanpercentile(values_np, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _bootstrap_ci_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr: float,
    per_sample_loss: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for evaluation metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)

    if n == 0:
        return {}

    rng = np.random.default_rng(seed)

    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]

    samples = {
        "loss": [],
        "acc": [],
        "bal_acc": [],
        "prec": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    for _ in range(n_boot):
        if stratified and (idx0.size > 0) and (idx1.size > 0):
            sample0 = rng.choice(idx0, size=idx0.size, replace=True)
            sample1 = rng.choice(idx1, size=idx1.size, replace=True)
            bootstrap_idx = np.concatenate([sample0, sample1])
        else:
            bootstrap_idx = rng.integers(0, n, size=n)

        yt = y_true[bootstrap_idx]
        ys = y_score[bootstrap_idx]
        yp = (ys >= thr).astype(int)

        if per_sample_loss is not None and len(per_sample_loss) == n:
            samples["loss"].append(float(np.mean(per_sample_loss[bootstrap_idx])))
        else:
            samples["loss"].append(float("nan"))

        samples["acc"].append(float(np.mean(yt == yp)) if len(yt) > 0 else float("nan"))

        prec, rec, f1, bal_acc, roc_auc = _safe_metrics(yt, yp, ys)
        pr_auc = _safe_pr_auc(yt, ys)

        samples["prec"].append(prec)
        samples["recall"].append(rec)
        samples["f1"].append(f1)
        samples["bal_acc"].append(bal_acc)
        samples["roc_auc"].append(roc_auc)
        samples["pr_auc"].append(pr_auc)

    return {key: _nan_percentile_ci(values, alpha=alpha) for key, values in samples.items()}


# -----------------------------
# Attention helpers
# -----------------------------
def _save_attn_matrix(attn_2d: np.ndarray, save_path: str, title: Optional[str] = None) -> None:
    """
    Save a 2D attention matrix as an image.
    """
    plt.figure()
    plt.imshow(attn_2d)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _attn_to_2d(attn_obj: Any) -> Optional[np.ndarray]:
    """
    Convert various attention formats to a 2D NumPy array.

    Supported formats:
        - list of tensors shaped [B, heads, T, T]
        - tensor shaped [B, heads, T, T]
        - tensor shaped [B, T, T]
        - tensor shaped [T, T]
        - NumPy array shaped [T, T]
    """
    if attn_obj is None:
        return None

    if isinstance(attn_obj, (list, tuple)):
        if len(attn_obj) == 0:
            return None
        attn_obj = attn_obj[-1]

    if torch.is_tensor(attn_obj):
        attn = attn_obj.detach()

        if attn.dim() == 4:
            attn = attn[0].mean(dim=0)  # [B, heads, T, T] -> first sample + head mean
        elif attn.dim() == 3:
            attn = attn[0]              # [B, T, T] -> first sample
        elif attn.dim() == 2:
            pass
        else:
            return None

        return attn.cpu().numpy()

    if isinstance(attn_obj, np.ndarray) and attn_obj.ndim == 2:
        return attn_obj

    return None


def _serialize_batch_paths(batch_paths: Any, batch_size: int) -> List[str]:
    """
    Convert collated path information from a dataloader batch into
    per-sample pipe-joined strings.

    Expected outputs:
        - ["a|b|c|d|e|f", ...] with length `batch_size`
    """
    if batch_paths is None:
        return [""] * batch_size

    if isinstance(batch_paths, np.ndarray):
        batch_paths = batch_paths.tolist()

    if isinstance(batch_paths, (list, tuple)):
        if len(batch_paths) == 0:
            return [""] * batch_size

        first = batch_paths[0]

        # Case 1: already per-sample list, length == batch_size
        if len(batch_paths) == batch_size:
            if isinstance(first, (list, tuple)):
                return ["|".join(map(str, sample_paths)) for sample_paths in batch_paths]
            if isinstance(first, str):
                return [str(path_str) for path_str in batch_paths]

        # Case 2: default_collate may transpose lists of strings
        if isinstance(first, (list, tuple)) and len(first) == batch_size:
            return ["|".join(map(str, sample_paths)) for sample_paths in zip(*batch_paths)]

        # Fallback: try generic transpose
        try:
            transposed = list(zip(*batch_paths))
            if len(transposed) == batch_size:
                return ["|".join(map(str, sample_paths)) for sample_paths in transposed]
        except TypeError:
            pass

    return [str(batch_paths)] * batch_size


# ============================================================
# BCEWithLogitsLoss-based evaluation + optional attention export
# ============================================================
def eval_model(
    model,
    loader,
    device,
    criterion,
    thr: float = 0.5,
    desc: str = "Eval",
    verbose: bool = True,
    save_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    save_plots: bool = False,
    save_detail_csv: bool = True,
    compute_ci: bool = True,
    ci_alpha: float = 0.05,
    ci_n_boot: int = 2000,
    ci_seed: int = 42,
    ci_stratified: bool = True,
    save_attention: bool = False,
    max_attn_save: int = 20,
) -> Dict[str, Any]:
    """
    Evaluate a binary classifier trained with BCEWithLogitsLoss.

    Expected loader output:
        - (t1, t2, label, paths), or
        - (t1, t2, label)

    Shapes:
        - t1, t2: [B, K, C, H, W]
        - label: [B]

    Model behavior:
        - when `save_attention=True`:
            model(t1, t2, return_aux=True, return_attn=True) -> (logits, aux)
        - otherwise:
            model(t1, t2) -> logits
    """
    if not 0.0 <= thr <= 1.0:
        raise ValueError(f"`thr` must be in [0, 1], got {thr}.")
    if ci_n_boot <= 0:
        raise ValueError(f"`ci_n_boot` must be positive, got {ci_n_boot}.")
    if max_attn_save < 0:
        raise ValueError(f"`max_attn_save` must be >= 0, got {max_attn_save}.")

    model.eval()

    loss_sum = 0.0
    seen = 0

    y_true_tensors: List[torch.Tensor] = []
    y_score_tensors: List[torch.Tensor] = []
    per_sample_loss_tensors: List[torch.Tensor] = []

    reduction = getattr(criterion, "reduction", "mean")
    crit_weight = getattr(criterion, "weight", None)
    crit_pos_weight = getattr(criterion, "pos_weight", None)

    if save_attention and (save_dir is not None):
        attn_dir = os.path.join(save_dir, "attn_maps")
        os.makedirs(attn_dir, exist_ok=True)
    else:
        attn_dir = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            t1, t2, label = batch

            t1 = t1.to(device)
            t2 = t2.to(device)
            label = label.to(device).view(-1)

            if save_attention:
                logits, aux = model(t1, t2, return_aux=True, return_attn=True)
            else:
                logits = model(t1, t2)
                aux = None

            logits = logits.view(-1)

            batch_size = label.size(0)
            seen += batch_size

            # Loss
            loss = criterion(logits, label.float())
            if reduction == "mean":
                loss_sum += loss.item() * batch_size
            elif reduction == "sum":
                loss_sum += loss.item()
            else:  # reduction == "none"
                loss_sum += loss.sum().item()

            # Per-sample loss for bootstrap CI
            weight = crit_weight
            pos_weight = crit_pos_weight
            if isinstance(weight, torch.Tensor) and weight.device != logits.device:
                weight = weight.to(logits.device)
            if isinstance(pos_weight, torch.Tensor) and pos_weight.device != logits.device:
                pos_weight = pos_weight.to(logits.device)

            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits,
                label.float(),
                weight=weight,
                pos_weight=pos_weight,
                reduction="none",
            )
            per_sample_loss_tensors.append(per_sample_loss)

            # Probabilities
            probs = torch.sigmoid(logits)

            y_score_tensors.append(probs)
            y_true_tensors.append(label)

            # Save attention maps for the first sample of selected batches
            if (
                save_attention
                and attn_dir is not None
                and batch_idx < max_attn_save
                and aux is not None
            ):
                t1_map = _attn_to_2d(aux.get("t1_attn", None))
                t2_map = _attn_to_2d(aux.get("t2_attn", None))
                modal_map = _attn_to_2d(aux.get("modal_attn", None))

                base = prefix if prefix is not None else desc.replace(" ", "_")

                if t1_map is not None:
                    _save_attn_matrix(
                        t1_map,
                        os.path.join(attn_dir, f"{base}_t1_attn_{batch_idx}.png"),
                        title="T1 slice attention",
                    )
                if t2_map is not None:
                    _save_attn_matrix(
                        t2_map,
                        os.path.join(attn_dir, f"{base}_t2_attn_{batch_idx}.png"),
                        title="T2 slice attention",
                    )
                if modal_map is not None:
                    _save_attn_matrix(
                        modal_map,
                        os.path.join(attn_dir, f"{base}_modal_attn_{batch_idx}.png"),
                        title="Modality attention",
                    )

    # Concatenate outputs
    if len(y_score_tensors) > 0:
        y_score_np = torch.cat(y_score_tensors).detach().cpu().numpy()
        y_true_np = torch.cat(y_true_tensors).detach().cpu().numpy()
        per_sample_loss_np = torch.cat(per_sample_loss_tensors).detach().cpu().numpy()
    else:
        y_score_np = np.array([])
        y_true_np = np.array([])
        per_sample_loss_np = np.array([])

    y_pred_np = (y_score_np >= thr).astype(int)

    final_loss = loss_sum / max(seen, 1)
    acc = float((y_true_np == y_pred_np).mean()) if seen > 0 else 0.0

    pr_auc = _safe_pr_auc(y_true_np, y_score_np)
    precision, recall, f1, bal_acc, roc_auc = _safe_metrics(y_true_np, y_pred_np, y_score_np)

    # Save plots and CSV
    if (save_plots or save_detail_csv) and (save_dir is not None) and (y_true_np.size > 0):
        os.makedirs(save_dir, exist_ok=True)
        base = prefix if prefix is not None else desc.replace(" ", "_")

        if save_plots:
            roc_path = os.path.join(save_dir, f"{base}_roc_curve.png")
            cm_path = os.path.join(save_dir, f"{base}_confusion_matrix.png")

            _plot_roc_curve(y_true_np, y_score_np, roc_path, title=f"ROC Curve - {desc}")
            _plot_confusion_matrix(
                y_true_np,
                y_pred_np,
                cm_path,
                title=f"Confusion Matrix - {desc}",
            )

        if save_detail_csv:
            csv_path = os.path.join(save_dir, f"{base}_eval_detail.csv")
            _save_eval_detail_csv(y_true_np, y_score_np, y_pred_np, csv_path)

    # Bootstrap CI
    ci = None
    if compute_ci and y_true_np.size > 0:
        ci = _bootstrap_ci_all_metrics(
            y_true=y_true_np,
            y_score=y_score_np,
            thr=thr,
            per_sample_loss=per_sample_loss_np if per_sample_loss_np.size > 0 else None,
            n_boot=ci_n_boot,
            alpha=ci_alpha,
            seed=ci_seed,
            stratified=ci_stratified,
        )

    if verbose:
        print(f"----- {desc} Results -----")
        print(f"Loss:              {final_loss:.4f}")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"F1-score:          {f1:.4f}")
        print(f"ROC-AUC:           {roc_auc:.4f}")
        print(f"PR-AUC:            {pr_auc:.4f}")

        if ci is not None:
            print(f"\n----- {desc} 95% CI (bootstrap, n={ci_n_boot}, stratified={ci_stratified}) -----")
            print(f"Loss CI:              [{ci['loss'][0]:.4f}, {ci['loss'][1]:.4f}]")
            print(f"Accuracy CI:          [{ci['acc'][0]:.4f}, {ci['acc'][1]:.4f}]")
            print(f"Balanced Acc CI:      [{ci['bal_acc'][0]:.4f}, {ci['bal_acc'][1]:.4f}]")
            print(f"Precision CI:         [{ci['prec'][0]:.4f}, {ci['prec'][1]:.4f}]")
            print(f"Recall CI:            [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")
            print(f"F1 CI:                [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]")
            print(f"ROC-AUC CI:           [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]")
            print(f"PR-AUC CI:            [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][1]:.4f}]")

    output = {
        "loss": final_loss,
        "acc": acc,
        "bal_acc": bal_acc,
        "prec": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    if ci is not None:
        output["ci"] = ci

    return output


# ============================================================
# Test-only wrapper
# ============================================================
def test_model_bce(model, test_loader, device, criterion, thr: float = 0.5) -> Dict[str, Any]:
    """
    Convenience wrapper for test evaluation.
    """
    return eval_model(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        thr=thr,
        desc="Test",
        verbose=True,
    )
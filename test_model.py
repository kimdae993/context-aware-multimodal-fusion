from sklearn.metrics import (
    precision_recall_curve, auc,
    precision_score, recall_score, balanced_accuracy_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, matthews_corrcoef,
)
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# PIL: Grad-CAM 업샘플링에 사용 (선택적 의존성)
try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# SciPy: foreground mask 후처리에 사용 (선택적 의존성)
try:
    from scipy import ndimage as _ndi
    _SCIPY_AVAILABLE = True
except ImportError:
    _ndi = None
    _SCIPY_AVAILABLE = False

# ============================================================
# 기존 헬퍼 함수 (변경 없음)
# ============================================================

def _safe_pr_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def _plot_roc_curve(y_true, y_score, save_path: str, title: str):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
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


def _safe_binary_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    if len(y_true) == 0:
        return (float("nan"),) * 8
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        bal_acc = float("nan")
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception:
        roc_auc = float("nan")
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")
    return precision, recall, sensitivity, specificity, f1, bal_acc, roc_auc, mcc


def _plot_confusion_matrix(y_true, y_pred, save_path: str, title: str, num_classes: int = 2):
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = [str(i) for i in labels]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def _save_eval_detail_csv(y_true, y_score, y_pred, save_path: str):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = np.asarray(y_pred)
    if y_score.ndim == 1:
        data = np.stack([y_true, y_score, y_pred], axis=1)
        header = "y_true,y_score,y_pred"
        fmt = ["%d", "%.8f", "%d"]
    else:
        num_classes = y_score.shape[1]
        data = np.concatenate(
            [y_true.reshape(-1, 1), y_score, y_pred.reshape(-1, 1)], axis=1)
        score_cols = ",".join([f"y_score_{i}" for i in range(num_classes)])
        header = f"y_true,{score_cols},y_pred"
        fmt = ["%d"] + ["%.8f"] * num_classes + ["%d"]
    np.savetxt(save_path, data, delimiter=",", header=header, comments="", fmt=fmt)


def _safe_multiclass_metrics(y_true, y_pred, y_score, num_classes: int):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    labels = list(range(num_classes))
    if len(y_true) == 0:
        return (float("nan"),) * 9
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    sensitivity = recall
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_spec = []
    total = cm.sum()
    for c in labels:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        if (tn + fp) > 0:
            per_class_spec.append(tn / (tn + fp))
    specificity = float(np.mean(per_class_spec)) if per_class_spec else float("nan")
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        bal_acc = float("nan")
    try:
        roc_auc = roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")
    try:
        y_true_onehot = np.eye(num_classes)[y_true]
        pr_auc = average_precision_score(y_true_onehot, y_score, average="macro")
    except Exception:
        pr_auc = float("nan")
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = float("nan")
    return precision, recall, sensitivity, specificity, f1, bal_acc, roc_auc, pr_auc, mcc

def _normalize_2d(arr: np.ndarray) -> np.ndarray:
    """[H, W] float 배열을 [0, 1]로 min-max 정규화합니다."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize_2d(cam: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    [h, w] float 배열을 [H, W]로 업샘플합니다.
    PIL 사용 가능 시 bilinear, 없으면 numpy nearest-neighbor로 대체합니다.
    """
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)

    if cam.shape == (H, W):
        return cam

    if _PIL_AVAILABLE:
        # float 값을 uint8로 바꾸는 과정에서 약간의 양자화가 발생할 수 있으나,
        # 시각화용 업샘플링에는 충분합니다.
        pil = _PILImage.fromarray((cam * 255).astype(np.uint8))
        pil = pil.resize((W, H), _PILImage.BILINEAR)
        return np.asarray(pil, dtype=np.float32) / 255.0

    row_idx = np.round(
        np.linspace(0, cam.shape[0] - 1, H)
    ).astype(int)
    col_idx = np.round(
        np.linspace(0, cam.shape[1] - 1, W)
    ).astype(int)
    return cam[np.ix_(row_idx, col_idx)]


def _largest_connected_component_numpy(mask: np.ndarray) -> np.ndarray:
    """
    SciPy가 없을 때 사용하는 8-neighborhood 최대 연결 성분 추출 함수.
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    best_component: list[tuple[int, int]] = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            component = []

            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))

                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if len(component) > len(best_component):
                best_component = component

    output = np.zeros_like(mask, dtype=bool)
    for r, c in best_component:
        output[r, c] = True

    return output


def _fill_holes_numpy(mask: np.ndarray) -> np.ndarray:
    """
    SciPy가 없을 때 이미지 경계에서 접근 가능한 배경을 flood-fill하여
    내부 hole을 채웁니다.
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    background = ~mask
    reachable = np.zeros_like(background, dtype=bool)
    stack = []

    for r in range(H):
        for c in (0, W - 1):
            if background[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    for c in range(W):
        for r in (0, H - 1):
            if background[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while stack:
        cr, cc = stack.pop()
        for dr, dc in neighbors:
            nr, nc = cr + dr, cc + dc
            if (
                0 <= nr < H
                and 0 <= nc < W
                and background[nr, nc]
                and not reachable[nr, nc]
            ):
                reachable[nr, nc] = True
                stack.append((nr, nc))

    holes = background & ~reachable
    return mask | holes


def _make_foreground_mask(
    raw_norm: np.ndarray,
    threshold: float = 0.03,
    closing_iterations: int = 2,
    dilation_iterations: int = 1,
) -> np.ndarray:
    """
    검은 MRI 배경을 제거하기 위한 foreground mask를 생성합니다.

    처리 과정:
      1. 정규화 영상에서 threshold보다 큰 픽셀 선택
      2. 가장 큰 연결 성분만 유지
      3. 내부 hole 채움
      4. closing 및 dilation으로 경계를 보완

    주의:
      이 mask는 intensity 기반 foreground/head mask입니다.
      전문적인 skull-stripping 결과와 동일한 'brain mask'는 아닙니다.
    """
    raw_norm = np.asarray(raw_norm, dtype=np.float32)
    raw_norm = np.nan_to_num(
        raw_norm,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    raw_norm = np.clip(raw_norm, 0.0, 1.0)

    threshold = float(np.clip(threshold, 0.0, 1.0))
    mask = raw_norm > threshold

    if not np.any(mask):
        return np.ones_like(raw_norm, dtype=bool)

    if _SCIPY_AVAILABLE:
        structure = _ndi.generate_binary_structure(rank=2, connectivity=2)

        labeled, num_features = _ndi.label(
            mask,
            structure=structure,
        )
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0
            largest_label = int(component_sizes.argmax())
            mask = labeled == largest_label

        if closing_iterations > 0:
            mask = _ndi.binary_closing(
                mask,
                structure=structure,
                iterations=int(closing_iterations),
            )

        mask = _ndi.binary_fill_holes(mask)

        if dilation_iterations > 0:
            mask = _ndi.binary_dilation(
                mask,
                structure=structure,
                iterations=int(dilation_iterations),
            )
    else:
        mask = _largest_connected_component_numpy(mask)
        mask = _fill_holes_numpy(mask)

    return np.asarray(mask, dtype=bool)


def _save_slice_triplet(
    save_dir: str,
    fname_base: str,
    raw_hw: np.ndarray,
    cam_hw: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    mask_background: bool = True,
    mask_threshold: float = 0.03,
    mask_closing_iterations: int = 2,
    mask_dilation_iterations: int = 1,
    cam_cutoff: float = 0.05,
    cam_gamma: float = 0.70,
    save_mask: bool = False,
) -> None:
    """
    슬라이스 하나에 대해 raw / cam / overlay를 저장합니다.

    mask_background=True이면 intensity 기반 foreground mask 바깥을
    완전히 검은색으로 표시합니다.

    foreground 내부에서는 고정 alpha overlay를 적용하므로,
    낮은 CAM 값(파란색)도 포함한 전체 기여도 범위를 그대로 표시합니다.
    foreground 바깥만 완전히 검은색으로 처리합니다.

    주의:
      배경 masking은 시각화에만 적용됩니다.
      모델 입력, Grad-CAM 계산 및 정량 분석용 CAM에는 적용하지 마십시오.
    """
    os.makedirs(save_dir, exist_ok=True)

    raw_hw = np.asarray(raw_hw, dtype=np.float32)
    if raw_hw.ndim != 2:
        raise ValueError(
            f"raw_hw는 [H,W]여야 합니다. 현재 shape={raw_hw.shape}"
        )

    H, W = raw_hw.shape
    raw_norm = _normalize_2d(raw_hw)
    cam_up = _resize_2d(cam_hw, H, W)
    cam_up = np.clip(cam_up, 0.0, 1.0)

    if mask_background:
        foreground_mask = _make_foreground_mask(
            raw_norm=raw_norm,
            threshold=mask_threshold,
            closing_iterations=mask_closing_iterations,
            dilation_iterations=mask_dilation_iterations,
        )
    else:
        foreground_mask = np.ones((H, W), dtype=bool)

    # foreground 내부에서는 CAM 전체 범위를 그대로 유지합니다.
    # 낮은 기여도(파란색)도 제거하지 않습니다.
    cam_masked = cam_up * foreground_mask.astype(np.float32)

    try:
        cmap_fn = plt.get_cmap(colormap)
    except ValueError:
        cmap_fn = plt.get_cmap("jet")

    heatmap_rgb = cmap_fn(cam_masked)[..., :3].astype(np.float32)
    raw_rgb = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)

    alpha = float(np.clip(alpha, 0.0, 1.0))

    # 배경은 검은색으로 두고, foreground 내부에만 고정 alpha overlay를 적용합니다.
    # cam_cutoff와 cam_gamma는 기존 호출과의 호환성을 위해 인자로 남겨 두지만,
    # foreground 내부의 낮은 CAM을 보존하기 위해 여기서는 사용하지 않습니다.
    overlay = np.zeros_like(raw_rgb, dtype=np.float32)
    overlay[foreground_mask] = (
        (1.0 - alpha) * raw_rgb[foreground_mask]
        + alpha * heatmap_rgb[foreground_mask]
    )

    # foreground 밖은 raw / CAM / overlay 모두 완전히 검은색으로 설정합니다.
    raw_rgb[~foreground_mask] = 0.0
    heatmap_rgb[~foreground_mask] = 0.0
    overlay[~foreground_mask] = 0.0

    raw_masked = raw_norm * foreground_mask.astype(np.float32)

    plt.imsave(
        os.path.join(save_dir, f"{fname_base}_raw.png"),
        raw_masked,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    plt.imsave(
        os.path.join(save_dir, f"{fname_base}_cam.png"),
        heatmap_rgb,
    )
    plt.imsave(
        os.path.join(save_dir, f"{fname_base}_overlay.png"),
        np.clip(overlay, 0.0, 1.0),
    )

    if save_mask:
        plt.imsave(
            os.path.join(save_dir, f"{fname_base}_foreground_mask.png"),
            foreground_mask.astype(np.uint8),
            cmap="gray",
            vmin=0,
            vmax=1,
        )


def _compute_dual_gradcam(
    model,
    t1_single: torch.Tensor,
    t2_single: torch.Tensor,
    device,
    loss_type: str,
    target_class: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    단일 샘플(B=1)에 대해 T1·T2 각 슬라이스의 Grad-CAM을 계산합니다.

    대상 레이어: t{1,2}_cross_slice_encoder.backbone.features[10]
                 (마지막 ReLU, AdaptiveAvgPool2d 직전)

    슬라이스 처리 방식에 따라 두 가지 hook 시나리오를 자동으로 구분합니다.
      - 벡터화 (view/reshape):
          hook 1회 호출, feature map shape = [S, C, H_cam, W_cam]
      - 슬라이스 루프:
          hook S회 호출, 각 shape = [1, C, H_cam, W_cam]

    retain_grad() 방식으로 per-slice 그래디언트를 정확히 수집합니다.

    Args:
        t1_single : [1, S, 1, H, W]  CPU 텐서
        t2_single : [1, S, 1, H, W]  CPU 텐서
        loss_type : "binary_bce" | "multiclass_ce"

    Returns:
        t1_cams : S개의 numpy 배열 리스트 (각 [H_cam, W_cam], 0-1 정규화)
        t2_cams : S개의 numpy 배열 리스트
    """
    model.eval()

    t1_single = t1_single.to(device)
    t2_single = t2_single.to(device)

    t1_fmaps: list[torch.Tensor] = []
    t2_fmaps: list[torch.Tensor] = []

    def _fwd_t1(module, inp, out):
        out.retain_grad()
        t1_fmaps.append(out)

    def _fwd_t2(module, inp, out):
        out.retain_grad()
        t2_fmaps.append(out)

    h1 = model.t1_cross_slice_encoder.backbone.features[10].register_forward_hook(_fwd_t1)
    h2 = model.t2_cross_slice_encoder.backbone.features[10].register_forward_hook(_fwd_t2)

    try:
        model.zero_grad()
        with torch.enable_grad():
            logits = model(t1_single, t2_single)

            if loss_type == "binary_bce":
                logit = logits.view(-1)[0]

                if target_class == 1:
                    # Severe-supporting attribution
                    score = logit
                elif target_class == 0:
                    # Normal-supporting attribution
                    # 단일-logit BCE에는 별도 normal logit이 없으므로
                    # severe logit의 음수를 normal score로 사용
                    score = -logit
                else:
                    raise ValueError(
                        f"Binary target_class는 0 또는 1이어야 합니다: "
                        f"{target_class}"
                    )

            elif loss_type == "multiclass_ce":
                if target_class < 0 or target_class >= logits.size(1):
                    raise ValueError(
                        f"target_class 범위 오류: {target_class}, "
                        f"num_classes={logits.size(1)}"
                    )

                # Softmax 확률이 아니라 해당 class의 logit을 직접 사용
                score = logits[0, target_class]

            else:
                raise ValueError(f"지원하지 않는 loss_type입니다: {loss_type}")

            score.backward()

    finally:
        h1.remove()
        h2.remove()

    def _fmaps_to_cams(fmaps: list[torch.Tensor]) -> list[np.ndarray]:
        """
        캡처된 feature map 리스트를 Grad-CAM 배열 리스트로 변환합니다.

        fmaps의 길이로 처리 방식을 구분합니다:
          len == 1 → 벡터화: fmaps[0].shape = [S, C, Hc, Wc]
          len == S → 루프:   fmaps[i].shape = [1, C, Hc, Wc]
        """
        if not fmaps:
            return []

        if len(fmaps) == 1:
            fmap = fmaps[0]
            grad = fmap.grad if fmap.grad is not None else torch.zeros_like(fmap)
            slices_a = [fmap[s].detach() for s in range(fmap.shape[0])]
            slices_g = [grad[s].detach()  for s in range(fmap.shape[0])]
        else:
            slices_a = [f[0].detach() for f in fmaps]
            slices_g = [
                (f.grad[0].detach() if f.grad is not None else torch.zeros_like(f[0]))
                for f in fmaps
            ]

        cams: list[np.ndarray] = []
        for a, g in zip(slices_a, slices_g):
            # a, g : [C, Hc, Wc]
            weights = g.mean(dim=(1, 2))                       # [C]
            cam = (weights[:, None, None] * a).sum(dim=0)      # [Hc, Wc]
            cam = torch.relu(cam)
            mn, mx = cam.min(), cam.max()
            cam = (cam - mn) / (mx - mn + 1e-8)
            cams.append(cam.cpu().numpy())

        return cams

    t1_cams = _fmaps_to_cams(t1_fmaps)
    t2_cams = _fmaps_to_cams(t2_fmaps)

    return t1_cams, t2_cams


def _run_gradcam_and_save(
    cases: dict[str, list],
    model,
    device,
    loss_type: str,
    save_dir: str,
    base: str,
    alpha: float = 0.5,
    colormap: str = "jet",
    mask_background: bool = True,
    mask_threshold: float = 0.03,
    mask_closing_iterations: int = 2,
    mask_dilation_iterations: int = 1,
    cam_cutoff: float = 0.05,
    cam_gamma: float = 0.70,
    save_mask: bool = False,
    verbose: bool = True,
) -> None:
    """
    TP·TN·FP·FN (또는 multiclass의 Correct·Wrong) 케이스별로
    Grad-CAM 이미지를 저장합니다.

    저장 디렉터리 구조:
        {save_dir}/gradcam/{base}_{case_label}/case_{idx:04d}/
            t1_s0_raw.png     # T1w 슬라이스 0 원본
            t1_s0_cam.png     # T1w 슬라이스 0 Grad-CAM heatmap
            t1_s0_overlay.png # T1w 슬라이스 0 overlay
            t1_s1_raw.png
            ...
            t2_s0_raw.png     # T2w 슬라이스 0 원본
            t2_s0_cam.png
            t2_s0_overlay.png
            ...

    Args:
        cases    : {"TP": [(t1,t2,y_true,y_pred,y_score), ...], "TN": [...], ...}
        base     : 파일명 접두어 (prefix 또는 desc)
        alpha    : overlay 혼합 비율
        colormap : jet, hot, turbo 등 matplotlib colormap 이름
    """
    total = sum(len(v) for v in cases.values())
    if total == 0:
        return

    done = 0
    for case_label, sample_list in cases.items():
        if not sample_list:
            continue

        case_root = os.path.join(save_dir, "gradcam", f"{base}_{case_label}")

        for idx, (t1_s, t2_s, y_true, y_pred, _y_score) in enumerate(sample_list):
            case_dir = os.path.join(case_root, f"case_{idx:04d}")

            # Grad-CAM 계산
            try:
                target_class = int(y_pred)

                t1_cams, t2_cams = _compute_dual_gradcam(
                    model=model,
                    t1_single=t1_s,
                    t2_single=t2_s,
                    device=device,
                    loss_type=loss_type,
                    target_class=target_class,
                )
            except Exception as e:
                if verbose:
                    print(f"\n  [Grad-CAM 경고] {case_label}/case_{idx:04d} 스킵: {e}")
                done += 1
                continue

            S = t1_s.shape[1]
            H, W = t1_s.shape[3], t1_s.shape[4]
            fallback = np.zeros((max(1, H // 4), max(1, W // 4)), dtype=np.float32)

            # 슬라이스별 3종 이미지 저장
            for s in range(S):
                raw_t1 = t1_s[0, s, 0].numpy()                    # [H, W]
                raw_t2 = t2_s[0, s, 0].numpy()                    # [H, W]
                cam_t1 = t1_cams[s] if s < len(t1_cams) else fallback
                cam_t2 = t2_cams[s] if s < len(t2_cams) else fallback

                _save_slice_triplet(
                    save_dir=case_dir,
                    fname_base=f"t1_s{s}",
                    raw_hw=raw_t1,
                    cam_hw=cam_t1,
                    alpha=alpha,
                    colormap=colormap,
                    mask_background=mask_background,
                    mask_threshold=mask_threshold,
                    mask_closing_iterations=mask_closing_iterations,
                    mask_dilation_iterations=mask_dilation_iterations,
                    cam_cutoff=cam_cutoff,
                    cam_gamma=cam_gamma,
                    save_mask=save_mask,
                )
                _save_slice_triplet(
                    save_dir=case_dir,
                    fname_base=f"t2_s{s}",
                    raw_hw=raw_t2,
                    cam_hw=cam_t2,
                    alpha=alpha,
                    colormap=colormap,
                    mask_background=mask_background,
                    mask_threshold=mask_threshold,
                    mask_closing_iterations=mask_closing_iterations,
                    mask_dilation_iterations=mask_dilation_iterations,
                    cam_cutoff=cam_cutoff,
                    cam_gamma=cam_gamma,
                    save_mask=save_mask,
                )

            done += 1
            if verbose:
                print(
                    f"\r  [Grad-CAM] {done:4d}/{total:4d}  "
                    f"({case_label}: case_{idx:04d})",
                    end="", flush=True,
                )

    if verbose:
        print(f"\r  [Grad-CAM] {done}/{total} 케이스 저장 완료.                              ")


# ============================================================
# 메인 평가 함수 (Grad-CAM 기능 추가)
# ============================================================

def eval_model_by_loss_type(
    model,
    loader,
    device,
    criterion,
    loss_type: str,
    num_classes: int,
    thr: float = 0.5,
    desc: str = "Eval",
    verbose: bool = True,
    save_dir: str | None = None,
    prefix: str = None,
    save_plots: bool = False,
    save_detail_csv: bool = False,
    seed: int = 42,
    # ── NEW: Grad-CAM ──────────────────────────────────────
    save_gradcam: bool = False,
    gradcam_alpha: float = 0.5,
    gradcam_colormap: str = "jet",
    gradcam_mask_background: bool = True,
    gradcam_mask_threshold: float = 0.03,
    gradcam_mask_closing_iterations: int = 2,
    gradcam_mask_dilation_iterations: int = 1,
    gradcam_cam_cutoff: float = 0.05,
    gradcam_cam_gamma: float = 0.70,
    gradcam_save_mask: bool = False,
    # ───────────────────────────────────────────────────────
):
    """
    loss_type에 따라 BCEWithLogitsLoss / CrossEntropyLoss를 모두 지원하는 평가 함수.

    loss_type:
        - "binary_bce"
        - "multiclass_ce"

    Grad-CAM 옵션 (save_gradcam=True 시 활성화):
        save_gradcam    : TP·TN·FP·FN 각 케이스의 Grad-CAM 이미지를 저장합니다.
                          save_dir 필수.
        gradcam_alpha   : overlay 혼합 비율 (기본 0.5 = heatmap 50% + raw 50%)
        gradcam_colormap: Grad-CAM 컬러맵 (기본 "jet", "hot", "turbo" 등 사용 가능)

    저장 구조 (save_gradcam=True):
        {save_dir}/gradcam/{prefix}_{TP|TN|FP|FN}/case_{idx:04d}/
            t1_s0_raw.png, t1_s0_cam.png, t1_s0_overlay.png
            t1_s1_raw.png, ...
            t2_s0_raw.png, t2_s0_cam.png, t2_s0_overlay.png
            ...
    """
    model.eval()
    criterion = criterion.to(device)

    loss_sum = 0.0
    seen = 0

    y_true_tensors = []
    y_pred_tensors = []
    y_score_tensors = []
    per_sample_loss_tensors = []

    reduction = getattr(criterion, "reduction", "mean")

    # ── NEW: Grad-CAM 케이스 수집 딕셔너리 ──────────────────
    gradcam_cases: dict[str, list] = {}
    if save_gradcam:
        if loss_type == "binary_bce":
            gradcam_cases = {"TP": [], "TN": [], "FP": [], "FN": []}
        # multiclass_ce: 케이스 키를 동적으로 추가
    # ─────────────────────────────────────────────────────────

    with torch.no_grad():
        for t1, t2, label in loader:
            # t1 = [B, S, 1, H, W]
            # t2 = [B, S, 1, H, W]
            t1, t2 = t1.to(device), t2.to(device)
            label = label.to(device)

            logits = model(t1, t2)
            bs = label.size(0)
            seen += bs

            if loss_type == "binary_bce":
                logits = logits.view(-1)
                loss = criterion(logits, label.float())

                bce_weight = getattr(criterion, "weight", None)
                bce_pos_weight = getattr(criterion, "pos_weight", None)
                per_sample_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, label.float(),
                    weight=bce_weight.to(device) if bce_weight is not None else None,
                    pos_weight=bce_pos_weight.to(device) if bce_pos_weight is not None else None,
                    reduction="none",
                )
                probs = torch.sigmoid(logits)
                preds = (probs >= thr).long()
                y_score_tensors.append(probs)

            elif loss_type == "multiclass_ce":
                assert logits.dim() == 2, (
                    f"CrossEntropyLoss 사용 시 logits는 [B, C] 형태여야 합니다. "
                    f"현재 logits.shape={tuple(logits.shape)}"
                )
                assert logits.size(1) == num_classes, (
                    f"모델 출력 클래스 수와 num_classes가 다릅니다. "
                    f"logits.size(1)={logits.size(1)}, num_classes={num_classes}"
                )
                loss = criterion(logits, label.long())

                ce_weight = getattr(criterion, "weight", None)
                ce_ignore_index = getattr(criterion, "ignore_index", -100)
                ce_label_smoothing = getattr(criterion, "label_smoothing", 0.0)
                per_sample_loss = torch.nn.functional.cross_entropy(
                    logits, label.long(),
                    weight=ce_weight.to(device) if ce_weight is not None else None,
                    ignore_index=ce_ignore_index,
                    reduction="none",
                    label_smoothing=ce_label_smoothing,
                )
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_score_tensors.append(probs)

            else:
                raise ValueError(f"지원하지 않는 loss_type입니다: {loss_type}")

            if reduction == "mean":
                loss_sum += loss.item() * bs
            elif reduction == "sum":
                loss_sum += loss.item()
            else:
                loss_sum += loss.sum().item()

            y_true_tensors.append(label)
            y_pred_tensors.append(preds)
            per_sample_loss_tensors.append(per_sample_loss)

            # ── NEW: TP·TN·FP·FN 케이스 수집 ──────────────────
            if save_gradcam and save_dir is not None:
                for i in range(bs):
                    yt = int(label[i].item())
                    yp = int(preds[i].item())

                    # 각 샘플을 [1, S, 1, H, W] CPU 텐서로 저장
                    t1_i = t1[i:i + 1].cpu()
                    t2_i = t2[i:i + 1].cpu()

                    if loss_type == "binary_bce":
                        ys = float(probs[i].item())
                        if   yt == 1 and yp == 1: key = "TP"
                        elif yt == 0 and yp == 0: key = "TN"
                        elif yt == 0 and yp == 1: key = "FP"
                        else:                     key = "FN"

                    else:  # multiclass_ce
                        ys = probs[i].cpu().numpy()
                        key = "Correct" if yt == yp else f"Wrong_true{yt}_pred{yp}"
                        if key not in gradcam_cases:
                            gradcam_cases[key] = []

                    gradcam_cases[key].append((t1_i, t2_i, yt, yp, ys))
            # ───────────────────────────────────────────────────

    # ── NEW: Grad-CAM 이미지 저장 ──────────────────────────────
    if save_gradcam and save_dir is not None and any(gradcam_cases.values()):
        base_name = prefix if prefix is not None else desc.replace(" ", "_")
        if verbose:
            total_cases = sum(len(v) for v in gradcam_cases.values())
            print(f"[Grad-CAM] 총 {total_cases}개 케이스 이미지 저장 시작...")
        _run_gradcam_and_save(
            cases=gradcam_cases,
            model=model,
            device=device,
            loss_type=loss_type,
            save_dir=save_dir,
            base=base_name,
            alpha=gradcam_alpha,
            colormap=gradcam_colormap,
            mask_background=gradcam_mask_background,
            mask_threshold=gradcam_mask_threshold,
            mask_closing_iterations=gradcam_mask_closing_iterations,
            mask_dilation_iterations=gradcam_mask_dilation_iterations,
            cam_cutoff=gradcam_cam_cutoff,
            cam_gamma=gradcam_cam_gamma,
            save_mask=gradcam_save_mask,
            verbose=verbose,
        )
    # ─────────────────────────────────────────────────────────────

    if len(y_true_tensors) > 0:
        y_true_np = torch.cat(y_true_tensors).detach().cpu().numpy()
        y_pred_np = torch.cat(y_pred_tensors).detach().cpu().numpy()
        y_score_np = torch.cat(y_score_tensors).detach().cpu().numpy()
        per_sample_loss_np = torch.cat(per_sample_loss_tensors).detach().cpu().numpy()
    else:
        y_true_np = np.array([])
        y_pred_np = np.array([])
        y_score_np = np.array([])
        per_sample_loss_np = np.array([])

    final_loss = loss_sum / max(seen, 1)
    acc = (y_true_np == y_pred_np).mean() if seen > 0 else 0.0

    if loss_type == "binary_bce":
        pr_auc = _safe_pr_auc(y_true_np, y_score_np)
        precision, recall, sensitivity, specificity, f1, bal_acc, roc_auc, mcc = \
            _safe_binary_metrics(y_true_np, y_pred_np, y_score_np)

    elif loss_type == "multiclass_ce":
        precision, recall, sensitivity, specificity, f1, bal_acc, roc_auc, pr_auc, mcc = \
            _safe_multiclass_metrics(y_true_np, y_pred_np, y_score_np, num_classes=num_classes)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    per_class_acc = {}
    for c in labels:
        class_count = cm[c, :].sum()
        per_class_acc[c] = cm[c, c] / class_count if class_count > 0 else float("nan")

    if (save_plots or save_detail_csv) and save_dir is not None and y_true_np.size > 0:
        os.makedirs(save_dir, exist_ok=True)
        base = prefix if prefix is not None else desc.replace(" ", "_")
        if save_plots:
            cm_path = os.path.join(save_dir, f"{base}_confusion_matrix.png")
            _plot_confusion_matrix(y_true_np, y_pred_np, cm_path,
                                   title=f"Confusion Matrix - {desc}",
                                   num_classes=num_classes)
            if loss_type == "binary_bce":
                roc_path = os.path.join(save_dir, f"{base}_roc_curve.png")
                _plot_roc_curve(y_true_np, y_score_np, roc_path,
                                title=f"ROC Curve - {desc}")
        if save_detail_csv:
            csv_path = os.path.join(save_dir, f"{base}_eval_detail.csv")
            _save_eval_detail_csv(y_true_np, y_score_np, y_pred_np, csv_path)

    if verbose:
        print(f"----- {desc} Results -----")
        print(f"Loss:              {final_loss:.4f}")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"Sensitivity:       {sensitivity:.4f}")
        if np.isnan(specificity):
            print("Specificity:       nan")
        else:
            print(f"Specificity:       {specificity:.4f}")
        print(f"F1-score:          {f1:.4f}")
        if np.isnan(mcc):
            print("MCC:               nan")
        else:
            print(f"MCC:               {mcc:.4f}")
        if np.isnan(roc_auc):
            print("ROC-AUC:           nan")
        else:
            print(f"ROC-AUC:           {roc_auc:.4f}")
        if np.isnan(pr_auc):
            print("PR-AUC:            nan")
        else:
            print(f"PR-AUC:            {pr_auc:.4f}")
        print(f"Per-class acc:     {per_class_acc}")

    result = {
        "loss": final_loss,
        "acc": acc,
        "bal_acc": bal_acc,
        "prec": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "per_class_acc": per_class_acc,
    }

    return result
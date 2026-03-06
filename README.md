# SymBrainRad: Paired T1/T2 Slice-Based Binary Classification for Radiology Score Prediction

This repository provides a PyTorch-based pipeline for binary classification of **Radiology score 1 vs 5** using paired **T1-weighted (T1w)** and **T2-weighted (T2w)** brain MRI slices.  
The model independently encodes slice sequences from each modality, learns inter-modality relationships, and performs final classification. The repository also includes auxiliary scripts for **evaluation logging**, **FP/FN analysis**, **attention visualization**, and **Grad-CAM visualization**. 

---

## 1. Key Features

- Binary classification using **paired multi-slice T1/T2** inputs
- Training with `BCEWithLogitsLoss`
- **Held-out test split + K-fold training on the remaining data**
- Saving fold-wise training logs (CSV and plots)
- Selecting the fold with the best validation balanced accuracy and evaluating it on the test set
- Saving evaluation outputs to `eval_detail.csv`
- Exporting attention maps and generating paper-ready figures
- Selecting representative FP/FN/TP/TN cases and generating error analysis CSVs
- Summarizing modality attention statistics and producing boxplots
- Visualizing T1/T2 slice importance using Grad-CAM

---

## 2. Model Overview

The full model is organized as follows:

1. **SliceBackbone**  
   Encodes each 2D slice with a CNN to produce a slice embedding.

2. **CrossSliceEncoder**  
   Encodes the slice sequence within a single modality (T1 or T2) using attention.  
   It can optionally return slice-depth attention weights and transformer-style self-attention weights.

3. **ModalityRelationEncoder**  
   Takes T1 and T2 modality representations and models inter-modality interactions to produce a relation-aware feature.

4. **RadClassifier**  
   Uses the final fused feature to output a binary logit.

The complete pipeline is combined in `SymBrainRadModel`.

---

## 3. Data Format

The current codebase assumes the following directory structure:

```text
project_root/
├── T1w/
│   ├── metadata.csv
│   └── ... image files ...
├── T2w/
│   ├── metadata.csv
│   └── ... image files ...
└── ... python scripts ...
```

### Required columns in `metadata.csv`

Each `T1w/metadata.csv` and `T2w/metadata.csv` must contain the following columns:

- `filename`
- `line`
- `rad_score`
- `session`

### Current preprocessing assumptions

The code assumes the following:

- Only **rad score 1** and **rad score 5** are used for final training.
- Each sample must contain **exactly 6 image paths**:
  - the first 3 paths: T1 slices
  - the last 3 paths: T2 slices
- Label mapping is defined as:
  - `rad_score == 1` → class `0`
  - `rad_score == 5` → class `1`

In other words, the current dataset class (`TwinMultiImageDataset`) expects inputs in the following format:

```text
[
  T1_slice1, T1_slice2, T1_slice3,
  T2_slice1, T2_slice2, T2_slice3
]
```

---

## 4. Training and Evaluation Design

The main training script follows this procedure:

1. Split the full dataset into `train / validation / test`
2. Merge `train + validation` to form a **full training set**
3. Perform **Stratified K-fold** training on this full training set
4. Save the model with the best **validation balanced accuracy** within each fold
5. Select the fold with the overall best validation balanced accuracy
6. Evaluate the best model from that fold on the **held-out test set**

Therefore, the evaluation design is:

- **Test set**: held out from the start
- **Train/Validation**: handled using K-fold cross-validation

---

## 5. Example Repository Structure

Below is an example of the main files in the current codebase:

```text
.
├── classifier.py
├── data.py
├── data_read.py
├── extra.py
├── img_cross_slice_encoder.py
├── img_dataset.py
├── img_model.py
├── img_slice_backbone.py
├── modality_relation_encoder.py
├── options.py
├── test.py
├── train_kfold.py
├── analyze_fpfn.py
├── select_cases.py
├── export_selected_attention.py
├── make_attn_figure.py
└── ... (Grad-CAM and other analysis scripts)
```

### Core modules

- `img_slice_backbone.py`  
  CNN backbone for slice-level encoding

- `img_cross_slice_encoder.py`  
  Intra-modality slice attention encoder

- `modality_relation_encoder.py`  
  Inter-modality relation encoder

- `classifier.py`  
  Final binary classifier

- `img_model.py`  
  Definition of the complete model (`SymBrainRadModel`)

- `img_dataset.py`  
  Paired T1/T2 multi-slice dataset

- `data_read.py`, `data.py`  
  Metadata loading, session-based grouping, and dataset splitting

- `test.py`  
  Evaluation utilities, ROC/confusion matrix export, `eval_detail.csv` creation, bootstrap CI computation

- `train_kfold.py`  
  Fold-wise training and K-fold training loop

---

## 6. CLI Options

The main CLI options are defined in `options.py`.

| Argument | Description | Default |
|---|---|---:|
| `--seed` | random seed | `42` |
| `--bs` | batch size | `4` |
| `--epochs` | number of training epochs | `50` |
| `--lr` | learning rate | `3e-4` |
| `--wd` | weight decay | `5e-4` |
| `--grad_norm` | gradient clipping norm | `1.0` |
| `--thr` | binary threshold | `0.5` |
| `--fold` | number of folds for K-fold | `5` |
| `--num_classes` | output dimension | `1` |
| `--D` / `--d-model` | hidden dimension | `128` |
| `--use_cls` | whether to use CLS-token-based processing (optional) | `False` |

---

## 7. Training

The name of the main training entry script may differ depending on your local setup. The example below assumes that the entry file is named `train.py`. If your actual filename is different, replace it accordingly.

```bash
python train.py \
  --seed 42 \
  --bs 4 \
  --epochs 50 \
  --lr 3e-4 \
  --wd 5e-4 \
  --grad_norm 1.0 \
  --thr 0.5 \
  --fold 5 \
  --D 128
```

After training starts, outputs will be saved under `results/<timestamp>/`.

---

## 8. Output Directory Structure

An example output directory created after training is shown below:

```text
results/20260210-161302/
├── model.txt
├── experiment_status.txt
├── best_val_bal_acc_model.pt
├── test_best_eval_detail.csv
├── test_best_roc_curve.png
├── test_best_confusion_matrix.png
├── checkpoints/
│   ├── fold1_best.pth
│   ├── fold2_best.pth
│   └── ...
├── fold1/
│   ├── CSV/
│   └── Graph/
├── fold2/
│   ├── CSV/
│   └── Graph/
└── ...
```

### Main outputs

- `model.txt`  
  Saved model architecture

- `experiment_status.txt`  
  Saved experiment configuration, including seed, batch size, loss, optimizer, etc.

- `best_val_bal_acc_model.pt`  
  Best model from the fold with the highest validation balanced accuracy

- `test_best_eval_detail.csv`  
  Per-sample prediction results for the test set

- `fold*/CSV/*.csv`  
  Metric logs for training and validation

- `fold*/Graph/*.jpg`  
  Metric curves for training and validation

---

## 9. Evaluation Result File (`eval_detail.csv`)

The `*_eval_detail.csv` file generated during evaluation is the main input for downstream error analysis and visualization.

Required columns:

- `paths`
- `y_true`
- `y_score`
- `y_pred`
- `type`

Here, `type` is one of:

- `TP`
- `TN`
- `FP`
- `FN`

The `paths` field is a serialized string in which multiple image paths are joined with `|`.

Example:

```text
./T1w/a.png|./T1w/b.png|./T1w/c.png|./T2w/d.png|./T2w/e.png|./T2w/f.png
```

---

## 10. Error Analysis (FP/FN Analysis)

`analyze_fpfn.py` automatically generates CSV files for FP/FN/TP/TN cases and extracts representative examples from `eval_detail.csv`.

### Example usage

```bash
python analyze_fpfn.py \
  --results-dir results/20260210-161302 \
  --threshold 0.5 \
  --topk 30
```

### Main outputs

```text
results/<run>/fpfn_analysis/<eval_detail_name>/
├── all_with_analysis.csv
├── TP.csv
├── TN.csv
├── FP.csv
├── FN.csv
├── FN_confident_top30.csv
├── FP_confident_top30.csv
├── FN_near_thr_top30.csv
├── FP_near_thr_top30.csv
├── FN_by_confidence_top30.csv
├── FP_by_confidence_top30.csv
└── report.txt
```

### Analysis criteria

- **confident error**: incorrect predictions made with high confidence
- **near-threshold error**: ambiguous errors near the decision threshold
- **confidence**: confidence with respect to the predicted class

---

## 11. Representative Case Selection

`select_cases.py` selects representative cases for each confusion-matrix type and generates `selected_cases.csv`.

### Example usage

```bash
python select_cases.py
```

Default selection rule:

- `TP`, `FP`: samples with higher `y_score` are prioritized
- `TN`, `FN`: samples with lower `y_score` are prioritized

Output file:

```text
results/<run>/selected_cases.csv
```

---

## 12. Attention Map Export

`export_selected_attention.py` exports attention maps for the representative samples listed in `selected_cases.csv`.

### Example usage

```bash
python export_selected_attention.py
```

Output:

```text
results/<run>/attn_selected/
├── 00_TP_..._t1.png
├── 00_TP_..._t2.png
├── 00_TP_..._modal.png
└── ...
```

Each file corresponds to:

- `*_t1.png`: T1 slice attention
- `*_t2.png`: T2 slice attention
- `*_modal.png`: modality attention (2×2)

> Note: the `D_MODEL` value used in this script must exactly match the `--D` value used during training.

---

## 13. Attention Figure Generation

`make_attn_figure.py` uses `selected_cases.csv` and `attn_selected/` to generate a paper-ready 4×3 attention figure.

### Example usage

```bash
python make_attn_figure.py
```

Output:

```text
results/<run>/figure_attn_4x3.png
```

By default, the figure contains one example each for `TP / FN / FP / TN`, arranged with the following three columns:

1. T1 slice attention
2. T2 slice attention
3. modality attention

---

## 14. Modality Attention Statistics

Using `test_best_eval_detail.csv` and a saved checkpoint, you can compute modality attention statistics for each sample. The script generates the following files:

- `modality_attn_stats.csv`
- `modality_attn_summary.csv`
- `modality_attn_box_by_type.png`
- `modality_attn_box_by_true.png`

Example analyses include:

- T1 / T2 modality importance
- T2 reliance ratio
- boxplots by TP/FN/FP/TN
- boxplots by true label

This script must also use the **same `d_model` setting as used during training**.

---

## 15. Grad-CAM Visualization

The Grad-CAM script parses session IDs and radiology scores from `test_best_eval_detail.csv` or `test_final_eval_detail.csv`, and generates Grad-CAM overlay figures for three T1 slices and three T2 slices.

### Output

```text
results/<run>/gradcam_figs/
├── gradcam_radscore1_sessionXXXX.png
├── gradcam_radscore5_sessionYYYY.png
└── match_debug.txt
```

### Summary of the procedure

- Extract `session_id`, `radscore`, and `modality` from the `paths` column
- Find sessions that contain both T1 and T2
- Select representative samples for the target radiology scores (default: 1 and 5)
- Generate a figure containing:
  - T1 raw
  - T1 Grad-CAM
  - T2 raw
  - T2 Grad-CAM

---

## 16. Individual Image Grid Rendering

A helper script is available for saving 2×3 image grids for individual samples. It is typically used with FP/FN analysis result CSVs such as `FN_confident_top30.csv`.

Example:

```bash
python render_case_grids.py \
  --csv results/20260210-161302/fpfn_analysis/test_best_eval_detail/FN_confident_top30.csv \
  --out_dir results/20260210-161302/fn_confident_grids \
  --max_n 30
```

> Replace the script filename with the actual filename used in your repository.

---

## 17. Evaluation Function Summary

`eval_model_bce()` in `test.py` provides the following:

- computation of loss, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
- ROC curve export
- confusion matrix export
- per-sample detail CSV export
- bootstrap confidence interval computation
- optional attention map export

`test_model_bce()` is a thin wrapper around this function for test evaluation.

---

## 18. Reproducibility and Notes

### Reproducibility

`set_seed()` fixes the following:

- Python random
- NumPy random
- PyTorch random
- CUDA random
- cuDNN deterministic / benchmark settings

### Notes

1. The `d_model` used during training and the `D_MODEL` used in analysis scripts must match exactly.
2. The current dataset class assumes **exactly 6 image paths (3 T1 + 3 T2)** per sample.
3. Some visualization scripts assume the `results/<timestamp>/...` output structure.
4. The `paths` column must be a `|`-joined serialized string.
5. Attention and Grad-CAM scripts assume that the model supports `return_aux=True` and `return_attn=True`.

---

## 19. Quick Start

The most basic execution flow is as follows:

```bash
# 1) Train
python main.py --seed <SEED> --bs <BatchSize> --epochs <NumEpochs> --fold <NumFolds> --D <HiddenDimension>

# 2) FP/FN analysis
python analyze_fpfn.py --results-dir results/20260210-161302 --threshold 0.5 --topk 30

# 3) Select representative cases
python select_cases.py

# 4) Export attention maps
python export_selected_attention.py

# 5) Generate a paper-ready attention figure
python make_attn_figure.py
```

# Context-Aware Multimodal Slice-Sequence Fusion

PyTorch implementation of a context-aware multimodal slice-sequence fusion model for T1w/T2w neonatal MRI radiology-score classification.

The repository contains the proposed model implementation, stratified k-fold training/evaluation code, metric logging, and optional Grad-CAM visualization utilities.

## Repository structure

```text
.
├── main.py                         # Training/evaluation entry point
├── options.py                      # Command-line arguments
├── data.py                         # Session-level T1w/T2w pairing and splitting
├── data_read.py                    # metadata.csv parsing utilities
├── img_dataset.py                  # Paired multi-slice dataset
├── img_model.py                    # Full proposed model
├── img_slice_backbone.py           # Slice-wise CNN backbone
├── img_cross_slice_encoder.py      # Positional + depth-wise + cross-slice encoder
├── modality_relation_encoder.py    # Modality embedding + interaction + relation fusion
├── classifier.py                   # MLP classifier
├── train_fold.py                   # Stratified k-fold training
├── test_model.py                   # Evaluation metrics and Grad-CAM utilities
├── extra.py                        # Logging and plotting utilities
├── smoke_test.py                   # Synthetic forward-pass test
├── requirements.txt
└── .gitignore
```

## Data format

Data are not included in this repository.

Prepare two modality directories, one for T1w and one for T2w. Each directory must contain a `metadata.csv` file with the following columns:

```text
filename,rad_score,session
```

Each session should have three T1w slices and three T2w slices. The code builds each session-level sample as:

```text
[T1_slice1, T1_slice2, T1_slice3, T2_slice1, T2_slice2, T2_slice3]
```

The file paths in `filename` may be relative to the modality directory or absolute paths.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if needed.

## Smoke test

```bash
python smoke_test.py
```

## Training examples

Binary normal-versus-severe task using radiology scores 1 and 5:

```bash
python main.py \
  --t1_dir /path/to/T1w \
  --t2_dir /path/to/T2w \
  --mode 1vs5 \
  --epochs 50 \
  --fold 5 \
  --bs 4 \
  --D 128
```

Three-class task using score 1, scores 2/3/4, and score 5:

```bash
python main.py \
  --t1_dir /path/to/T1w \
  --t2_dir /path/to/T2w \
  --mode 1vs234vs5 \
  --epochs 50 \
  --fold 5 \
  --bs 4 \
  --D 128
```

For CPU-only smoke runs or constrained machines, use:

```bash
python main.py ... --no_cuda --num_workers 0 --num_threads 1
```

## Outputs

Experiment outputs are saved under:

```text
results/<mode>/<YYYYMMDD-HHMMSS>/
```

The output directory contains:

- `model.txt`: model architecture
- `experiment_status.txt`: training configuration
- `checkpoints/`: fold-specific best checkpoints
- `fold*/CSV/`: metric CSV files
- `fold*/Graph/`: metric plots
- `test_*_fold_mean_std_result.txt`: fold-averaged test results
- `Best_result/`: final selected test result files and plots
- `best_val_mcc_model.pt`: selected model checkpoint

## Loss weighting protocol

Class weights are computed once from the full training/validation pool after the independent test split, then reused across all cross-validation folds.

- Binary task: `BCEWithLogitsLoss(pos_weight=N_negative/N_positive)`
- Three-class task: `CrossEntropyLoss(weight=N/(C*N_c))`

Fold-specific class-weight recomputation is intentionally not applied in this code version.

## Model summary

For each modality, the model applies:

1. slice-wise CNN feature extraction,
2. learnable positional embedding,
3. depth-wise self-attention over ordered slice tokens,
4. cross-slice transformer encoding,
5. mean pooling into a modality representation.

T1w and T2w representations are then augmented with learnable modality embeddings, processed by a modality interaction transformer, and fused using:

```text
[T1w, T2w, T1w - T2w, T1w * T2w]
```

The fused representation is passed to an MLP classifier.

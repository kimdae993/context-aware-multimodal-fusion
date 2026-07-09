import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Context-aware multimodal slice-sequence fusion for "
            "T1w/T2w neonatal MRI radiology-score classification."
        )
    )

    # Data and output paths
    parser.add_argument(
        "--t1_dir", type=str, default="INPUT_YOUR_T1W_FOLDER_PATH",
        help="Directory containing T1w slice images and metadata.csv."
    )
    parser.add_argument(
        "--t2_dir", type=str, default="INPUT_YOUR_T2W_FOLDER_PATH",
        help="Directory containing T2w slice images and metadata.csv."
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results",
        help="Root directory for experiment outputs."
    )

    # Experiment setup
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument(
        '--mode', type=str, default="1vs5", choices=["1vs5", "1vs234vs5"],
        help=(
            "Classification task. "
            "1vs5: score 1 vs score 5; "
            "1vs234vs5: score 1 vs scores 2/3/4 vs score 5."
        )
    )
    parser.add_argument('--fold', type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument('--img_size', type=int, default=290, help="Input image size after resizing.")
    parser.add_argument('--num_slices', type=int, default=3, help="Number of slices per modality.")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader worker count.")

    # Training
    parser.add_argument('--bs', type=int, default=4, help="Batch size for training and inference.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--wd', type=float, default=5e-4, help="Weight decay.")
    parser.add_argument('--thr', type=float, default=0.5, help="Binary decision threshold.")
    parser.add_argument('--no_cuda', action="store_true", help="Force CPU execution even when CUDA is available.")
    parser.add_argument('--num_threads', type=int, default=None, help="Optional torch CPU thread count.")

    # Model hyperparameters
    parser.add_argument('--D', type=int, default=128, help="Latent feature dimension.")
    parser.add_argument('--slice_nhead', type=int, default=2, help="Number of slice-attention heads.")
    parser.add_argument('--slice_layers', type=int, default=1, help="Number of cross-slice transformer layers.")
    parser.add_argument('--slice_dropout', type=float, default=0.1, help="Dropout in slice encoder.")
    parser.add_argument('--modal_nhead', type=int, default=2, help="Number of modality-attention heads.")
    parser.add_argument('--modal_layers', type=int, default=1, help="Number of modality transformer layers.")
    parser.add_argument('--modal_dropout', type=float, default=0.3, help="Dropout in modality interaction encoder.")
    parser.add_argument('--cls_dropout', type=float, default=0.3, help="Dropout in classifier.")
    parser.add_argument('--mlp_ratio', type=int, default=4, help="Transformer feed-forward expansion ratio.")

    # Optional outputs
    parser.add_argument('--save_gradcam', action="store_true", help="Save Grad-CAM images for the selected best model.")

    return parser.parse_args()

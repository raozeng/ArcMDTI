# -*- coding:utf-8 -*-
import argparse
import os
from RunModel import run_5_fold_cv
from run_cold_start import run_5_fold_cv as run_cold_cv


def main():
    parser = argparse.ArgumentParser(description="ArcMDTI Unified SCI Training Pipeline")
    parser.add_argument("dataset_pos", nargs="?", default=None, help="Dataset name (e.g., Enzyme, GPCRs, ion_channel, Davis, KIBA)")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="Dataset name (e.g., Enzyme, GPCRs, ion_channel, Davis, KIBA)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-f", "--fold", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("-e", "--epochs", type=int, default=0, help="Override max training epochs (default: 100 from config.py)")
    parser.add_argument("--patience", type=int, default=0, help="Override early-stopping patience (default: 25 from config.py)")
    parser.add_argument("--batch_size", type=int, default=0, help="Override default batch size")
    parser.add_argument(
        "--mode",
        choices=["random", "cold_drug", "cold_target", "cold-drug", "cold-target"],
        default="random",
    )
    args = parser.parse_args()

    raw_dataset = args.dataset or args.dataset_pos
    if not raw_dataset:
        parser.error("Please specify a dataset name (e.g., `python main.py Davis` or `python main.py --dataset Davis`).")

    if args.epochs > 0:
        os.environ["ARCMDTI_EPOCHS"] = str(args.epochs)
    if args.patience > 0:
        os.environ["ARCMDTI_PATIENCE"] = str(args.patience)

    dataset_name = raw_dataset.replace(".txt", "")
    dataset_path = f"./DataSets/{dataset_name}.txt"

    if not os.path.exists(dataset_path):
        print(f"Error: File not found at {dataset_path}")
        return

    b_size = args.batch_size if args.batch_size > 0 else None
    ep_override = args.epochs if args.epochs > 0 else None
    mode_norm = args.mode.replace("-", "_")
    if mode_norm == "random":
        run_5_fold_cv(dataset_path, k_folds=args.fold, seed=args.seed, custom_batch_size=b_size, custom_epochs=ep_override)
    else:
        run_cold_cv(dataset_path, split_mode=mode_norm, k_folds=args.fold, seed=args.seed, custom_batch_size=b_size, custom_epochs=ep_override)


if __name__ == "__main__":
    main()


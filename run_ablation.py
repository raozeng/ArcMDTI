# -*- coding:utf-8 -*-
"""
ArcMDTI Fair & Comprehensive Ablation Study Runner (code_new/run_ablation.py)
=============================================================================
Fixes over legacy run_ablation.py:
1. Uses `load_secondary_structure_dict(dataset_path, all_lines=all_lines)` so dataset-specific
   SS pickles (e.g. Davis_ss.pkl, KIBA_ss.pkl, >protein0 keys) are properly aligned.
2. Removes artificial epoch truncation: all variants are trained under a consistent, fair
   optimization schedule.
3. Computes and saves all 8 metrics (AUC, AUPR, ACC, F1, MCC, Pre, Sn, Sp) per fold and summary.
4. Supports full module ablations, encoder ablations, and loss/margin head comparisons.
"""

import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")

from model import AMCADTI_V2
from config import hyperparameter
from LossFunction import FocalLoss
from utils.DataSetsFunction import CustomDataSet, collate_fn, load_secondary_structure_dict
from utils.EarlyStoping import EarlyStopping
from utils.EpochLoggerAndResume import (
    run_arcmdti_fold_training,
    write_run_status,
    aggregate_all_epoch_histories,
)
from RunModel import (
    DEVICE,
    setup_seed,
    load_trusted_checkpoint,
    compute_all_metrics,
    eval_model,
    train_epoch,
)

ALL_VARIANTS = [
    "full",
    "no_ss",
    "no_crossattn",
    "no_gate",
    "no_arcface",
    "no_focalloss",
    "ce",
    "cosface",
    "sphereface",
    "direct_margin",
    "cnn_only",
    "no_bilstm",
    "no_transformer",
]


def run_ablation_variant(
    dataset_path,
    variant,
    k_folds=5,
    seed=42,
    run_only_one_fold=False,
    selected_folds=None,
    skip_completed=True,
    custom_epochs=None,
):
    setup_seed(seed)
    hp = hyperparameter()
    hp.select_dataset_config(dataset_path)
    if custom_epochs:
        hp.Epoch = int(custom_epochs)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = np.array([l.strip() for l in f if l.strip()])

    SS_DICT = load_secondary_structure_dict(dataset_path, all_lines=all_lines)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    dataset_name = os.path.basename(dataset_path).split(".")[0]
    base_save_dir = os.path.join(".", "result", f"Ablation_{dataset_name}_Seed{seed}", variant)
    os.makedirs(base_save_dir, exist_ok=True)
    exp_tag = f"Ablation_{dataset_name}_{variant}_Seed{seed}"

    print(f"\n[Ablation Study] Variant: {variant} | Dataset: {dataset_name} | Device: {DEVICE}", flush=True)

    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)
    all_fold_results = []
    selected_folds = set(selected_folds) if selected_folds else None

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_lines)):
        fold_no = fold + 1
        if selected_folds is not None and fold_no not in selected_folds:
            continue

        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_no}")
        os.makedirs(fold_save_dir, exist_ok=True)
        npz_path = os.path.join(fold_save_dir, "ROC_PRC_data.npz")

        if skip_completed and os.path.exists(npz_path):
            print(f"[RESUME] Fold {fold_no} already completed ({npz_path}). Loading metrics...", flush=True)
            data = np.load(npz_path)
            metrics = compute_all_metrics(data["y_true"], data["y_scores"])
            all_fold_results.append(metrics)
            if run_only_one_fold:
                break
            continue

        print(f"\n{'=' * 20} Fold {fold_no} / {k_folds} ({variant}) {'=' * 20}", flush=True)
        train_lines = all_lines[train_idx].tolist()
        val_lines = all_lines[val_idx].tolist()

        train_loader = DataLoader(CustomDataSet(train_lines), batch_size=hp.Batch_size, shuffle=True, collate_fn=my_collate)
        val_loader = DataLoader(CustomDataSet(val_lines), batch_size=hp.Batch_size, shuffle=False, collate_fn=my_collate)

        is_no_focal = variant in ("no_focalloss", "ce")
        if variant == "no_focalloss":
            model_ablation_type = "full"
        elif variant == "ce":
            model_ablation_type = "no_arcface"
        else:
            model_ablation_type = variant

        model = AMCADTI_V2(hp, ablation=model_ablation_type).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=hp.Learning_rate, weight_decay=hp.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hp.scheduler_T0, T_mult=hp.scheduler_Tmult
        )

        if is_no_focal:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = FocalLoss(alpha=hp.focal_alpha, gamma=hp.focal_gamma)

        stopper = EarlyStopping(savepath=fold_save_dir, patience=hp.Patience)

        final_fold_res, _, _ = run_arcmdti_fold_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            stopper=stopper,
            hp=hp,
            fold_no=fold_no,
            k_folds=k_folds,
            fold_save_dir=fold_save_dir,
            base_save_dir=base_save_dir,
            exp_tag=exp_tag,
            device=DEVICE,
            train_epoch_fn=train_epoch,
            eval_model_fn=eval_model,
            load_ckpt_fn=load_trusted_checkpoint,
            fold_record_path=npz_path,
        )
        all_fold_results.append(final_fold_res)

        if run_only_one_fold:
            print("Fast mode enabled: stopping after 1 fold.", flush=True)
            break

    df = pd.DataFrame(all_fold_results)
    if df.empty:
        return {}

    df.index = [f"Fold_{i + 1}" for i in range(len(all_fold_results))]
    df.to_csv(os.path.join(base_save_dir, f"{dataset_name}_{variant}_per_fold.csv"), index_label="Fold")

    write_run_status(
        base_save_dir,
        stage=exp_tag,
        fold=len(all_fold_results),
        total_folds=k_folds,
        epoch=hp.Epoch,
        total_epochs=hp.Epoch,
        best_val_auc=float(df["AUC"].mean()),
        best_epoch=-1,
        state="completed",
    )
    aggregate_all_epoch_histories("./result")

    res_dict = df.mean().to_dict()
    for col in df.columns:
        res_dict[f"{col}_std"] = float(df[col].std()) if len(df) > 1 else 0.0
    res_dict["n_folds"] = len(df)
    return res_dict


def main():
    parser = argparse.ArgumentParser(description="ArcMDTI Comprehensive Ablation Study")
    parser.add_argument("--datasets", nargs="+", default=["Davis", "KIBA"], help="Datasets to run on")
    parser.add_argument("--fast", action="store_true", help="Run only 1 fold instead of full 5 folds")
    parser.add_argument("--skip_full", action="store_true", help="Skip the 'full' variant if already trained")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=ALL_VARIANTS,
        help="Run only selected ablation variants",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Run selected 1-based fold numbers")
    parser.add_argument("--start_fold", type=int, default=None, help="Run folds from this 1-based fold through 5")
    parser.add_argument("--skip_completed", action="store_true", default=True, help="Skip folds with ROC_PRC_data.npz")
    parser.add_argument("--epochs", type=int, default=0, help="Override max epochs")
    args = parser.parse_args()

    default_variants = ["full", "no_arcface", "no_crossattn", "no_ss", "no_focalloss", "no_gate", "ce", "cosface"]
    variants = args.variants if args.variants is not None else default_variants
    variants = [v for v in variants if not (args.skip_full and v == "full")]
    print(f"Variants to run: {variants}")

    selected_folds = None
    if args.folds is not None:
        selected_folds = args.folds
    elif args.start_fold is not None:
        selected_folds = list(range(args.start_fold, 6))

    overall_results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./result/ablation_summary", exist_ok=True)

    for dataset in args.datasets:
        dataset_path = f"./DataSets/{dataset}.txt"
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset {dataset_path} not found. Skipping...")
            continue

        for variant in variants:
            res_dict = run_ablation_variant(
                dataset_path=dataset_path,
                variant=variant,
                k_folds=5,
                run_only_one_fold=args.fast,
                selected_folds=selected_folds,
                skip_completed=args.skip_completed,
                custom_epochs=args.epochs if args.epochs > 0 else None,
            )
            if not res_dict:
                continue
            res_dict["Dataset"] = dataset
            res_dict["Variant"] = variant
            overall_results.append(res_dict)

            temp_df = pd.DataFrame(overall_results)
            temp_df.to_csv(f"./result/ablation_summary/ablation_progress_{timestamp}.csv", index=False)

    if overall_results:
        final_df = pd.DataFrame(overall_results)
        final_csv_path = f"./result/ablation_summary/final_ablation_results_{timestamp}.csv"
        final_df.to_csv(final_csv_path, index=False)
        final_df.to_csv("./result/ablation_summary/final_ablation_results_latest.csv", index=False)
        print(f"\nAblation Study Complete! Summary saved to {final_csv_path}")


if __name__ == "__main__":
    main()


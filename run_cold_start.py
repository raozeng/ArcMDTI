# -*- coding:utf-8 -*-
"""
ArcMDTI Cold-Start Evaluation Runner (code_new/run_cold_start.py)
=================================================================
Evaluates ArcMDTI under Cold-Drug and Cold-Target 5-fold cross-validation splits.
Computes all 8 metrics (AUC, AUPR, ACC, F1, MCC, Pre, Sn, Sp) and exports OOF predictions.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")

from model import AMCADTI_V2
from config import hyperparameter
from LossFunction import FocalLoss
from utils.ColdStartSplit import get_cold_start_folds
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
    validate_features,
    build_oof_records_for_fold,
)
from plot_best_features import generate_sci_ready_dr_plots


def calculate_metrics(y_true, y_scores):
    return compute_all_metrics(y_true, y_scores)


def run_5_fold_cv(dataset_path, split_mode="cold_drug", k_folds=5, seed=42, custom_batch_size=None, custom_epochs=None):
    setup_seed(seed)
    split_mode = split_mode.replace("-", "_")
    hp = hyperparameter()
    hp.select_dataset_config(dataset_path)
    if custom_batch_size:
        hp.Batch_size = int(custom_batch_size)
        print(f"[REDUCE] Using custom Batch Size: {hp.Batch_size}")
    if custom_epochs:
        hp.Epoch = int(custom_epochs)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = np.array([l.strip() for l in f if l.strip()])

    SS_DICT = load_secondary_structure_dict(dataset_path, all_lines=all_lines)

    all_fold_results = []
    oof_records = []
    dataset_name = os.path.basename(dataset_path).split(".")[0]
    base_save_dir = f"./result/{dataset_name}_{split_mode.upper()}_CV_Seed{seed}"
    os.makedirs(base_save_dir, exist_ok=True)
    exp_tag = f"{dataset_name}_{split_mode.upper()}_Seed{seed}"

    print(f"\n[SCI-Pro Mode] {k_folds}-Fold {split_mode.upper()} CV on {dataset_name} | Device: {DEVICE}", flush=True)

    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)

    if split_mode in ("cold_drug", "cold_target"):
        fold_splits = get_cold_start_folds(all_lines, split_mode=split_mode, k_folds=k_folds, seed=seed)
    else:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(all_lines))

    final_summary_csv = os.path.join(base_save_dir, f"{dataset_name}_final_summary.csv")
    if os.path.exists(final_summary_csv):
        print(f"\n[SKIP] Dataset {dataset_name} with {split_mode} already finished. Found: {final_summary_csv}", flush=True)
        return

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_no = fold + 1
        print(f"\n{'=' * 20} Fold {fold_no} / {k_folds} {'=' * 20}", flush=True)
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_no}")
        os.makedirs(fold_save_dir, exist_ok=True)

        fold_record_path = os.path.join(fold_save_dir, "ROC_PRC_data.npz")
        if os.path.exists(fold_record_path):
            print(f"[RESUME] Loading existing results for Fold {fold_no} from {fold_record_path}", flush=True)
            data = np.load(fold_record_path)
            metrics = compute_all_metrics(data["y_true"], data["y_scores"])
            all_fold_results.append(metrics)
            oof_records.extend(
                build_oof_records_for_fold(all_lines, val_idx, data["y_true"], data["y_scores"], dataset_name, fold_no)
            )
            continue

        train_lines = all_lines[train_idx].tolist()
        val_lines = all_lines[val_idx].tolist()

        train_loader = DataLoader(CustomDataSet(train_lines), batch_size=hp.Batch_size, shuffle=True, collate_fn=my_collate)
        val_loader = DataLoader(CustomDataSet(val_lines), batch_size=hp.Batch_size, shuffle=False, collate_fn=my_collate)

        model = AMCADTI_V2(hp).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=hp.Learning_rate, weight_decay=hp.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hp.scheduler_T0, T_mult=hp.scheduler_Tmult
        )
        criterion = FocalLoss(alpha=hp.focal_alpha, gamma=hp.focal_gamma)

        stopper = EarlyStopping(savepath=fold_save_dir, patience=hp.Patience)

        final_fold_res, y_true_fold, y_scores_fold = run_arcmdti_fold_training(
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
            fold_record_path=fold_record_path,
        )
        all_fold_results.append(final_fold_res)
        oof_records.extend(
            build_oof_records_for_fold(all_lines, val_idx, y_true_fold, y_scores_fold, dataset_name, fold_no)
        )

    df = pd.DataFrame(all_fold_results)
    metric_cols = ["AUC", "AUPR", "ACC", "F1", "MCC", "Pre", "Sn", "Sp"]

    print("\n" + "*" * 20 + " SCI FINAL 5-FOLD REPORT " + "*" * 20)
    summary_data = {}
    for col in metric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:6}: {mean_val:.4f} ± {std_val:.4f}")
        summary_data[col] = f"{mean_val:.4f} ± {std_val:.4f}"

    df.index = [f"Fold_{i + 1}" for i in range(len(all_fold_results))]
    fold_csv_path = os.path.join(base_save_dir, f"{dataset_name}_per_fold_metrics.csv")
    df.to_csv(fold_csv_path, index_label="Fold")
    print(f"\n[SAVED] Per-fold metrics saved to: {fold_csv_path}")

    summary_df = pd.DataFrame([summary_data], index=["Mean ± Std"])
    summary_csv_path = os.path.join(base_save_dir, f"{dataset_name}_final_summary.csv")
    summary_df.to_csv(summary_csv_path, index_label="Stats")
    print(f"[SAVED] Final summary saved to: {summary_csv_path}")

    if oof_records:
        df_oof = pd.DataFrame(oof_records).sort_values("LineIdx").reset_index(drop=True)
        oof_csv_path = os.path.join(base_save_dir, f"oof_predictions_{dataset_name}.csv")
        df_oof.to_csv(oof_csv_path, index=False)
        print(f"[SAVED] OOF predictions saved to: {oof_csv_path}", flush=True)

    write_run_status(
        base_save_dir,
        stage=exp_tag,
        fold=k_folds,
        total_folds=k_folds,
        epoch=hp.Epoch,
        total_epochs=hp.Epoch,
        best_val_auc=float(df["AUC"].mean()),
        best_epoch=-1,
        state="completed",
    )
    aggregate_all_epoch_histories("./result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ArcMDTI with Cold-Start Splits")
    parser.add_argument("--dataset", type=str, default="Enzyme.txt", help="Dataset filename in DataSets folder")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="cold_drug",
        choices=["random", "cold_drug", "cold_target"],
        help="CV split strategy",
    )
    parser.add_argument("--batch_size", type=int, default=0, help="Override default batch size")
    parser.add_argument("--epochs", type=int, default=0, help="Override max epochs")
    args = parser.parse_args()

    dataset_name = args.dataset
    if not dataset_name.endswith(".txt"):
        dataset_name += ".txt"
    dataset_path = dataset_name if os.path.exists(dataset_name) else f"./DataSets/{dataset_name}"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    b_size = args.batch_size if args.batch_size > 0 else None
    ep = args.epochs if args.epochs > 0 else None
    run_5_fold_cv(dataset_path, split_mode=args.split_mode, k_folds=5, seed=42, custom_batch_size=b_size, custom_epochs=ep)


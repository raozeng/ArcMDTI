# -*- coding:utf-8 -*-
"""
ArcMDTI Hyperparameter Sensitivity Analysis (code_new/run_sensitivity.py)
=========================================================================
Systematically evaluates the sensitivity of ArcMDTI to:
1. Additive angular margin penalty m in {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
2. Hypersphere feature scale s in {8.0, 15.0, 24.0, 32.0}
3. Focal Loss gamma in {0.0, 0.5, 1.0, 2.0}

Saves full fold-level and summary CSVs and generates publication-ready plots
(`paper_new/sci_plots_ablation/Sensitivity_Margin_Scale.pdf`).
"""

import os
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
import matplotlib.pyplot as plt

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
from RunModel import DEVICE, setup_seed, load_trusted_checkpoint, compute_all_metrics, eval_model, train_epoch

MARGIN_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SCALE_GRID = [8.0, 15.0, 24.0, 32.0]


def run_single_setting(dataset_path, s_val, m_val, gamma_val=1.0, k_folds=5, fast=False, seed=42, custom_epochs=None):
    setup_seed(seed)
    hp = hyperparameter()
    hp.select_dataset_config(dataset_path)
    if custom_epochs:
        hp.Epoch = int(custom_epochs)

    dataset_name = os.path.basename(dataset_path).split(".")[0]
    tag = f"s{s_val:.1f}_m{m_val:.2f}_g{gamma_val:.1f}"
    save_dir = os.path.join("result", f"Sensitivity_{dataset_name}_Seed{seed}", tag)
    os.makedirs(save_dir, exist_ok=True)
    exp_tag = f"Sensitivity_{dataset_name}_{tag}_Seed{seed}"

    summary_file = os.path.join(save_dir, "summary.csv")
    if os.path.exists(summary_file):
        df_old = pd.read_csv(summary_file)
        if not df_old.empty:
            print(f"[SKIP] Sensitivity {dataset_name} {tag} already completed.", flush=True)
            return df_old.iloc[0].to_dict()

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = np.array([l.strip() for l in f if l.strip()])

    SS_DICT = load_secondary_structure_dict(dataset_path, all_lines=all_lines)
    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_lines)):
        fold_no = fold + 1
        fold_dir = os.path.join(save_dir, f"fold_{fold_no}")
        os.makedirs(fold_dir, exist_ok=True)
        npz_path = os.path.join(fold_dir, "ROC_PRC_data.npz")

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            fold_metrics.append(compute_all_metrics(data["y_true"], data["y_scores"]))
            if fast:
                break
            continue

        train_loader = DataLoader(CustomDataSet(all_lines[train_idx].tolist()), batch_size=hp.Batch_size, shuffle=True, collate_fn=my_collate)
        val_loader = DataLoader(CustomDataSet(all_lines[val_idx].tolist()), batch_size=hp.Batch_size, shuffle=False, collate_fn=my_collate)

        model = AMCADTI_V2(hp, ablation="full", arcface_s=s_val, arcface_m=m_val).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=hp.Learning_rate, weight_decay=hp.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hp.scheduler_T0, T_mult=hp.scheduler_Tmult)
        criterion = FocalLoss(alpha=hp.focal_alpha, gamma=gamma_val)

        stopper = EarlyStopping(savepath=fold_dir, patience=hp.Patience)

        res, _, _ = run_arcmdti_fold_training(
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
            fold_save_dir=fold_dir,
            base_save_dir=save_dir,
            exp_tag=exp_tag,
            device=DEVICE,
            train_epoch_fn=train_epoch,
            eval_model_fn=eval_model,
            load_ckpt_fn=load_trusted_checkpoint,
            fold_record_path=npz_path,
        )
        fold_metrics.append(res)

        if fast:
            break

    df = pd.DataFrame(fold_metrics)
    row = {
        "Dataset": dataset_name,
        "s": s_val,
        "m": m_val,
        "gamma": gamma_val,
        "n_folds": len(df),
    }
    for col in ["AUC", "AUPR", "ACC", "F1", "MCC", "Pre", "Sn", "Sp"]:
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_std"] = float(df[col].std()) if len(df) > 1 else 0.0

    pd.DataFrame([row]).to_csv(summary_file, index=False)
    write_run_status(
        save_dir,
        stage=exp_tag,
        fold=len(df),
        total_folds=k_folds,
        epoch=hp.Epoch,
        total_epochs=hp.Epoch,
        best_val_auc=float(row["AUC_mean"]),
        best_epoch=-1,
        state="completed",
    )
    aggregate_all_epoch_histories("./result")
    return row


def plot_sensitivity_curves(df, out_dir="sci_plots_ablation"):
    os.makedirs(out_dir, exist_ok=True)
    datasets = df["Dataset"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)
    colors = ["#c1272d", "#1f77b4", "#2b7a78", "#c96a20"]

    # Panel A: Angular Margin m sweep (at s=15.0)
    ax = axes[0]
    sub_m = df[np.isclose(df["s"], 15.0)].sort_values("m")
    for idx, ds in enumerate(datasets):
        d_sub = sub_m[sub_m["Dataset"] == ds]
        if d_sub.empty:
            continue
        c = colors[idx % len(colors)]
        ax.plot(d_sub["m"], d_sub["AUC_mean"], marker="o", lw=2.2, color=c, label=f"{ds} (AUC)")
        ax.plot(d_sub["m"], d_sub["F1_mean"], marker="s", lw=1.8, ls="--", color=c, alpha=0.85, label=f"{ds} (F1)")
    ax.axvline(0.15, color="#555555", ls=":", lw=1.2, label="Default $m=0.15$")
    ax.set_xlabel("Additive Angular Margin $m$ (rad)", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("(a) Sensitivity to Angular Margin $m$ ($s=15.0$)", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8.5, loc="best")

    # Panel B: Scale factor s sweep (at m=0.15)
    ax = axes[1]
    sub_s = df[np.isclose(df["m"], 0.15)].sort_values("s")
    for idx, ds in enumerate(datasets):
        d_sub = sub_s[sub_s["Dataset"] == ds]
        if d_sub.empty:
            continue
        c = colors[idx % len(colors)]
        ax.plot(d_sub["s"], d_sub["AUC_mean"], marker="o", lw=2.2, color=c, label=f"{ds} (AUC)")
        ax.plot(d_sub["s"], d_sub["F1_mean"], marker="s", lw=1.8, ls="--", color=c, alpha=0.85, label=f"{ds} (F1)")
    ax.axvline(15.0, color="#555555", ls=":", lw=1.2, label="Default $s=15.0$")
    ax.set_xlabel("Hypersphere Scale Factor $s$", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("(b) Sensitivity to Hypersphere Scale $s$ ($m=0.15$)", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8.5, loc="best")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"Sensitivity_Margin_Scale.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensitivity figure to {out_dir}/Sensitivity_Margin_Scale.pdf")


def main():
    parser = argparse.ArgumentParser(description="ArcMDTI Hyperparameter Sensitivity Analysis")
    parser.add_argument("--datasets", nargs="+", default=["Enzyme", "Davis"], help="Datasets for sensitivity sweep")
    parser.add_argument("--fast", action="store_true", help="Run 1 fold per setting for rapid sweep")
    parser.add_argument("--epochs", type=int, default=0, help="Override max epochs")
    args = parser.parse_args()

    rows = []
    os.makedirs("sci_plots_ablation", exist_ok=True)
    ep = args.epochs if args.epochs > 0 else None

    for ds in args.datasets:
        path = f"./DataSets/{ds}.txt"
        if not os.path.exists(path):
            print(f"[SKIP] Dataset {path} not found.")
            continue
        # 1. Sweep m with s=15.0
        for m_val in MARGIN_GRID:
            rows.append(run_single_setting(path, s_val=15.0, m_val=m_val, fast=args.fast, custom_epochs=ep))
        # 2. Sweep s with m=0.15
        for s_val in SCALE_GRID:
            if abs(s_val - 15.0) < 1e-6:
                continue
            rows.append(run_single_setting(path, s_val=s_val, m_val=0.15, fast=args.fast, custom_epochs=ep))

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("sci_plots_ablation/Sensitivity_Summary.csv", index=False)
        plot_sensitivity_curves(df, out_dir="sci_plots_ablation")


if __name__ == "__main__":
    main()


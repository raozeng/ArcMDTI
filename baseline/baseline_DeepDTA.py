# -*- coding:utf-8 -*-
import os
import sys
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import random
import argparse
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import KFold
from utils.ColdStartSplit import get_cold_start_folds
from utils.EpochLoggerAndResume import write_run_status

import torch.utils._pytree as _pytree

_raw_reg = getattr(_pytree, "_register_pytree_node", getattr(_pytree, "register_pytree_node", None))


def robust_register_patch(cls, to_iter, from_iter, **kwargs):
    if _raw_reg is not None:
        return _raw_reg(cls, to_iter, from_iter)


_pytree.register_pytree_node = _pytree._register_pytree_node = robust_register_patch
sys.modules["torch.utils._pytree"] = _pytree

try:
    from DeepPurpose import DTI as models
    from DeepPurpose import utils
    from DeepPurpose.utils import data_process
    HAS_DEEPPURPOSE = True
except Exception:
    HAS_DEEPPURPOSE = False
    from baseline.pytorch_baseline_fallback import train_and_predict_fallback


def setup_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_your_data(path):
    drugs, prots, labels, lines = [], [], [], []
    if not os.path.exists(path):
        return None, None, None, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            try:
                if len(p) >= 5:
                    lbl = int(float(p[4]))
                    drugs.append(p[2])
                    prots.append(p[3])
                    labels.append(lbl)
                    lines.append(line.strip())
                elif len(p) == 3:
                    lbl = int(float(p[2]))
                    drugs.append(p[0])
                    prots.append(p[1])
                    labels.append(lbl)
                    lines.append(line.strip())
            except ValueError:
                continue
    return np.array(drugs), np.array(prots), np.array(labels), np.array(lines)


def calculate_metrics(y_true, y_pred_prob):
    y_pred_prob = np.array(y_pred_prob, dtype=float)
    y_true = np.array(y_true, dtype=int)
    y_pred_label = (y_pred_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_pred_label)
    pre = precision_score(y_true, y_pred_label, zero_division=0)
    f1 = f1_score(y_true, y_pred_label, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_label)

    conf = confusion_matrix(y_true, y_pred_label, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel() if conf.size == 4 else (0, 0, 0, 0)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": auc,
        "AUPRC": auprc,
        "Acc": acc,
        "F1": f1,
        "Pre": pre,
        "Sn": sn,
        "Sp": sp,
        "MCC": mcc,
    }


def save_outputs(df_res, summary_lines, df_oof, dataset, split_mode):
    dirs = [f"./result/Baseline_DeepDTA_{dataset}_{split_mode.upper()}"]
    if split_mode == "random":
        dirs.append(f"./result/Baseline_DeepDTA_{dataset}")
    for save_dir in dirs:
        os.makedirs(save_dir, exist_ok=True)
        df_res.to_csv(os.path.join(save_dir, "Detailed_Fold_Results.csv"))
        with open(os.path.join(save_dir, "Final_Summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"DeepDTA Baseline on {dataset} ({split_mode})\n")
            f.writelines([l + "\n" for l in summary_lines])
        df_oof.to_csv(os.path.join(save_dir, f"oof_predictions_{dataset}.csv"), index=False)


def run_experiment():
    parser = argparse.ArgumentParser(description="DeepDTA Baseline")
    parser.add_argument("dataset", help="Dataset name (e.g., GPCRs)")
    parser.add_argument("-f", "--fold", type=int, default=5, help="Number of folds")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=["random", "cold_drug", "cold_target"],
        help="CV split strategy",
    )
    args = parser.parse_args()

    setup_seed(args.seed)
    ds_clean = args.dataset.replace(".txt", "")
    data_path = f"./DataSets/{ds_clean}.txt"

    X_d, X_p, y, all_lines = load_your_data(data_path)
    if X_d is None:
        print(f"Error: Training file not found at {data_path}")
        return

    base_save_dir = f"./result/Baseline_DeepDTA_{ds_clean}_{args.split_mode.upper()}"
    os.makedirs(base_save_dir, exist_ok=True)
    exp_tag = f"Baseline_DeepDTA_{ds_clean}_{args.split_mode.upper()}"
    summary_file = os.path.join(base_save_dir, "Final_Summary.txt")
    oof_file = os.path.join(base_save_dir, f"oof_predictions_{ds_clean}.csv")
    if os.path.exists(summary_file) and os.path.exists(oof_file):
        print(f"[SKIP] DeepDTA ({ds_clean}, {args.split_mode}) already completed: {summary_file}", flush=True)
        return

    all_fold_results = []
    oof_records = []

    if args.split_mode == "random":
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        fold_splits = list(kf.split(X_d))
    else:
        fold_splits = get_cold_start_folds(all_lines, split_mode=args.split_mode, k_folds=args.fold, seed=args.seed)

    print(f"Starting DeepDTA Baseline | Dataset: {ds_clean} | Mode: {args.split_mode} | {args.fold}-Fold CV", flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        fold_no = fold_idx + 1
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_no}")
        os.makedirs(fold_save_dir, exist_ok=True)
        fold_npz = os.path.join(fold_save_dir, "ROC_PRC_data.npz")

        d_tr, p_tr, y_tr = X_d[train_idx], X_p[train_idx], y[train_idx]
        d_val, p_val, y_val = X_d[val_idx], X_p[val_idx], y[val_idx]

        if os.path.exists(fold_npz):
            print(f"[RESUME] Fold {fold_no}/{args.fold} already completed ({fold_npz}). Loading...", flush=True)
            npz_data = np.load(fold_npz)
            y_pred = npz_data["y_scores"]
            fold_metrics = calculate_metrics(y_val, y_pred)
            all_fold_results.append(fold_metrics)
            for i, idx_val in enumerate(val_idx):
                oof_records.append(
                    {
                        "Dataset": ds_clean,
                        "Fold": fold_no,
                        "LineIdx": int(idx_val),
                        "True_Label": int(y_val[i]),
                        "Probability": float(y_pred[i]),
                        "Prediction": 1 if float(y_pred[i]) >= 0.5 else 0,
                    }
                )
            continue

        print(f"\n>>>>>>> Fold {fold_no}/{args.fold} <<<<<<<", flush=True)
        write_run_status(
            base_save_dir,
            stage=exp_tag,
            fold=fold_no,
            total_folds=args.fold,
            epoch=0,
            total_epochs=args.epochs,
            best_val_auc=0.0,
            best_epoch=0,
            state="running",
        )

        if HAS_DEEPPURPOSE:
            train = data_process(d_tr, p_tr, y_tr, "CNN", "CNN", split_method="no_split")
            val = data_process(d_val, p_val, y_val, "CNN", "CNN", split_method="no_split")

            config = utils.generate_config(
                drug_encoding="CNN",
                target_encoding="CNN",
                result_folder=fold_save_dir,
                cls_hidden_dims=[1024, 1024, 512],
                train_epoch=args.epochs,
                LR=0.001,
                batch_size=32,
                cnn_drug_filters=[32, 64, 96],
                cnn_drug_kernels=[4, 6, 8],
                cnn_target_filters=[32, 64, 96],
                cnn_target_kernels=[4, 8, 12],
                hidden_dim_drug=256,
                hidden_dim_protein=256,
            )

            model = models.model_initialize(**config)
            model.train(train, val, val)
            try:
                model.save_model(fold_save_dir)
            except Exception:
                pass

            y_pred = np.array(model.predict(val))
        else:
            print("[INFO] Using native PyTorch DeepDTA implementation (DeepPurpose not installed)...", flush=True)
            y_pred = train_and_predict_fallback(
                d_tr, p_tr, y_tr, d_val, p_val, y_val, arch="DeepDTA", epochs=args.epochs, lr=5e-4, batch_size=64
            )
        np.savez(fold_npz, y_true=np.asarray(y_val, dtype=int), y_scores=y_pred)
        fold_metrics = calculate_metrics(y_val, y_pred)
        all_fold_results.append(fold_metrics)
        print(f"Fold {fold_no} Result: AUC={fold_metrics['AUC']:.4f}, ACC={fold_metrics['Acc']:.4f}", flush=True)

        for i, idx_val in enumerate(val_idx):
            oof_records.append(
                {
                    "Dataset": ds_clean,
                    "Fold": fold_no,
                    "LineIdx": int(idx_val),
                    "True_Label": int(y_val[i]),
                    "Probability": float(y_pred[i]),
                    "Prediction": 1 if float(y_pred[i]) >= 0.5 else 0,
                }
            )

    df_res = pd.DataFrame(all_fold_results)
    summary_lines = []
    for col in ["AUC", "AUPRC", "Acc", "F1", "Pre", "Sn", "Sp", "MCC"]:
        mean_val = df_res[col].mean()
        std_val = df_res[col].std()
        line = f"{col:6}: {mean_val:.4f} ± {std_val:.4f}"
        print(line, flush=True)
        summary_lines.append(line)

    df_oof = pd.DataFrame(oof_records).sort_values("LineIdx").reset_index(drop=True)
    save_outputs(df_res, summary_lines, df_oof, ds_clean, args.split_mode)
    write_run_status(
        base_save_dir,
        stage=exp_tag,
        fold=args.fold,
        total_folds=args.fold,
        epoch=args.epochs,
        total_epochs=args.epochs,
        best_val_auc=float(df_res["AUC"].mean()),
        best_epoch=-1,
        state="completed",
    )


if __name__ == "__main__":
    run_experiment()



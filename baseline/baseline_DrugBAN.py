# -*- coding:utf-8 -*-
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
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
from utils.EpochLoggerAndResume import (
    atomic_torch_save,
    load_checkpoint_safely,
    truncate_fold_logs_upto_epoch,
    append_fold_epoch_log,
    sync_unified_epoch_logs,
    write_run_status,
    aggregate_all_epoch_histories,
)
import time

SMILES_CHAR = {
    "#": 1, "(": 2, ")": 3, "-": 4, "/": 5, "1": 6, "2": 7, "3": 8, "4": 9, "5": 10,
    "6": 11, "7": 12, "8": 13, "=": 14, "@": 15, "B": 16, "C": 17, "F": 18, "H": 19,
    "I": 20, "N": 21, "O": 22, "P": 23, "S": 24, "[": 25, "\\": 26, "]": 27, "a": 28,
    "c": 29, "e": 30, "i": 31, "l": 32, "n": 33, "o": 34, "r": 35, "s": 36,
}
AMINO_ACID_CHAR = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "X": 21,
}


def smiles_to_int(smiles, max_len=100):
    res = [SMILES_CHAR.get(char, 0) for char in smiles]
    if len(res) > max_len:
        res = res[:max_len]
    else:
        res = res + [0] * (max_len - len(res))
    return np.array(res, dtype=np.int64)


def seq_to_int(seq, max_len=1000):
    res = [AMINO_ACID_CHAR.get(char, 0) for char in seq]
    if len(res) > max_len:
        res = res[:max_len]
    else:
        res = res + [0] * (max_len - len(res))
    return np.array(res, dtype=np.int64)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, glim=2):
        super(BANLayer, self).__init__()
        self.glim = glim
        self.v_proj = nn.Linear(v_dim, h_dim)
        self.q_proj = nn.Linear(q_dim, h_dim)
        self.dropout = nn.Dropout(0.1)
        self.h_out = h_out
        self.h_dim = h_dim

    def forward(self, v, q):
        v = self.v_proj(v)
        q = self.q_proj(q)
        att_map = torch.matmul(v, q.transpose(1, 2))
        att_map = F.softmax(att_map.view(v.size(0), -1), dim=1).view(v.size(0), v.size(1), q.size(1))
        v_att = torch.matmul(att_map, q)
        joint = v * v_att
        return joint.mean(dim=1)


class DrugBAN(nn.Module):
    def __init__(self):
        super(DrugBAN, self).__init__()
        self.drug_embed = nn.Embedding(len(SMILES_CHAR) + 1, 64)
        self.drug_cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.prot_embed = nn.Embedding(len(AMINO_ACID_CHAR) + 1, 64)
        self.prot_cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.ban = BANLayer(v_dim=256, q_dim=256, h_dim=256, h_out=2)
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, d, p):
        d_feat = self.drug_embed(d).transpose(1, 2)
        d_feat = self.drug_cnn(d_feat).transpose(1, 2)
        p_feat = self.prot_embed(p).transpose(1, 2)
        p_feat = self.prot_cnn(p_feat).transpose(1, 2)
        joint = self.ban(d_feat, p_feat)
        out = self.mlp(joint)
        return torch.sigmoid(out).view(-1)


class DTIDataset(Dataset):
    def __init__(self, drugs, prots, labels):
        self.drugs = [smiles_to_int(d) for d in drugs]
        self.prots = [seq_to_int(p) for p in prots]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.drugs[idx]),
            torch.LongTensor(self.prots[idx]),
            torch.FloatTensor([self.labels[idx]]),
        )


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
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
        "AUC": float(auc),
        "AUPRC": float(auprc),
        "AUPR": float(auprc),
        "Acc": float(acc),
        "ACC": float(acc),
        "F1": float(f1),
        "Pre": float(pre),
        "Sn": float(sn),
        "Sp": float(sp),
        "MCC": float(mcc),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def save_outputs(df_res, summary_lines, df_oof, dataset, split_mode):
    dirs = [f"./result/Baseline_DrugBAN_{dataset}_{split_mode.upper()}"]
    if split_mode == "random":
        dirs.append(f"./result/Baseline_DrugBAN_{dataset}")
    for save_dir in dirs:
        os.makedirs(save_dir, exist_ok=True)
        df_res.to_csv(os.path.join(save_dir, "Detailed_Fold_Results.csv"))
        with open(os.path.join(save_dir, "Final_Summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"DrugBAN Baseline on {dataset} ({split_mode})\n")
            f.writelines([l + "\n" for l in summary_lines])
        df_oof.to_csv(os.path.join(save_dir, f"oof_predictions_{dataset}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "cold_drug", "cold_target"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(42)

    ds_clean = args.dataset.replace(".txt", "")
    data_path = f"./DataSets/{ds_clean}.txt"
    X_d, X_p, y, all_lines = load_your_data(data_path)
    if X_d is None:
        return

    base_save_dir = f"./result/Baseline_DrugBAN_{ds_clean}_{args.split_mode.upper()}"
    os.makedirs(base_save_dir, exist_ok=True)
    exp_tag = f"Baseline_DrugBAN_{ds_clean}_{args.split_mode.upper()}"

    summary_file = os.path.join(base_save_dir, "Final_Summary.txt")
    oof_file = os.path.join(base_save_dir, f"oof_predictions_{ds_clean}.csv")
    if os.path.exists(summary_file) and os.path.exists(oof_file):
        print(f"[SKIP] DrugBAN ({ds_clean}, {args.split_mode}) already completed: {summary_file}", flush=True)
        return

    if args.split_mode == "random":
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_splits = list(kf.split(X_d))
    else:
        fold_splits = get_cold_start_folds(all_lines, split_mode=args.split_mode, k_folds=5, seed=42)

    all_fold_results, oof_records = [], []
    print(f"DrugBAN Baseline | Dataset: {ds_clean} | Mode: {args.split_mode} | Device: {device}", flush=True)

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_no = fold + 1
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_no}")
        os.makedirs(fold_save_dir, exist_ok=True)
        fold_npz = os.path.join(fold_save_dir, "ROC_PRC_data.npz")

        if os.path.exists(fold_npz):
            print(f"[RESUME] Fold {fold_no}/5 already completed ({fold_npz}). Loading...", flush=True)
            npz_data = np.load(fold_npz)
            trues = npz_data["y_true"]
            probs = npz_data["y_scores"]
            metrics = calculate_metrics(trues, probs)
            all_fold_results.append(metrics)
            for i, idx in enumerate(val_idx):
                oof_records.append(
                    {
                        "Dataset": ds_clean,
                        "Fold": fold_no,
                        "LineIdx": int(idx),
                        "True_Label": int(trues[i]),
                        "Probability": float(probs[i]),
                        "Prediction": 1 if probs[i] >= 0.5 else 0,
                    }
                )
            continue

        print(f"Fold {fold_no}/5", flush=True)
        train_loader = DataLoader(DTIDataset(X_d[train_idx], X_p[train_idx], y[train_idx]), batch_size=64, shuffle=True)
        val_loader = DataLoader(DTIDataset(X_d[val_idx], X_p[val_idx], y[val_idx]), batch_size=64)

        model = DrugBAN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        criterion = nn.BCELoss()

        best_auc = 0.0
        best_epoch = 0
        start_epoch = 1
        epochs_no_improve = 0
        best_ckpt_path = os.path.join(fold_save_dir, "valid_best_checkpoint.pth")
        latest_ckpt_path = os.path.join(fold_save_dir, f"checkpoint_latest_fold_{fold_no}.pt")
        prev_ckpt_path = os.path.join(fold_save_dir, f"checkpoint_prev_fold_{fold_no}.pt")

        ckpt, ckpt_src = load_checkpoint_safely(fold_save_dir, fold_no, device)
        if ckpt is not None:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = int(ckpt["epoch"]) + 1
                best_auc = float(ckpt.get("best_auc", 0.0))
                best_epoch = int(ckpt.get("best_epoch", 0))
                epochs_no_improve = int(ckpt.get("stopper_counter", 0))
                truncate_fold_logs_upto_epoch(fold_save_dir, int(ckpt["epoch"]))
                sync_unified_epoch_logs(base_save_dir, exp_tag)
                print(
                    f"[RESUME] DrugBAN Fold {fold_no} resumed from {os.path.basename(ckpt_src)} at Epoch {ckpt['epoch']} "
                    f"(Best AUC={best_auc:.4f})",
                    flush=True,
                )
            except Exception as e:
                print(f"[WARNING] Failed to load DrugBAN checkpoint: {e}. Starting Fold {fold_no} fresh.", flush=True)
                start_epoch = 1

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            model.train()
            tr_losses, tr_probs, tr_trues = [], [], []
            for d, p, label in train_loader:
                d, p, label = d.to(device), p.to(device), label.to(device)
                optimizer.zero_grad()
                pred = model(d, p)
                loss = criterion(pred, label.view(-1))
                loss.backward()
                optimizer.step()
                tr_losses.append(float(loss.item()))
                tr_probs.extend(pred.detach().cpu().numpy().flatten())
                tr_trues.extend(label.detach().cpu().numpy().flatten())

            train_metrics = calculate_metrics(np.array(tr_trues), np.array(tr_probs))
            train_metrics["Loss"] = float(np.mean(tr_losses)) if tr_losses else 0.0

            model.eval()
            val_losses, val_probs, val_trues = [], [], []
            with torch.no_grad():
                for d, p, label in val_loader:
                    d, p, label = d.to(device), p.to(device), label.to(device)
                    pred = model(d, p)
                    v_loss = criterion(pred, label.view(-1))
                    val_losses.append(float(v_loss.item()))
                    val_probs.extend(pred.cpu().numpy().flatten())
                    val_trues.extend(label.cpu().numpy().flatten())

            val_metrics = calculate_metrics(np.array(val_trues), np.array(val_probs))
            val_metrics["Loss"] = float(np.mean(val_losses)) if val_losses else 0.0
            elapsed = time.time() - t0

            is_best = False
            if val_metrics["AUC"] > best_auc:
                best_auc = val_metrics["AUC"]
                best_epoch = epoch
                epochs_no_improve = 0
                is_best = True
                atomic_torch_save(model.state_dict(), best_ckpt_path)
            else:
                epochs_no_improve += 1

            row_dict = {
                "fold": fold_no,
                "epoch": epoch,
                "lr": 5e-5,
                "train_loss": f"{train_metrics['Loss']:.6f}",
                "train_auc": f"{train_metrics['AUC']:.6f}",
                "train_aupr": f"{train_metrics['AUPR']:.6f}",
                "train_acc": f"{train_metrics['ACC']:.6f}",
                "train_f1": f"{train_metrics['F1']:.6f}",
                "train_mcc": f"{train_metrics['MCC']:.6f}",
                "train_pre": f"{train_metrics['Pre']:.6f}",
                "train_sn": f"{train_metrics['Sn']:.6f}",
                "train_sp": f"{train_metrics['Sp']:.6f}",
                "train_tp": train_metrics["TP"],
                "train_tn": train_metrics["TN"],
                "train_fp": train_metrics["FP"],
                "train_fn": train_metrics["FN"],
                "val_loss": f"{val_metrics['Loss']:.6f}",
                "val_auc": f"{val_metrics['AUC']:.6f}",
                "val_aupr": f"{val_metrics['AUPR']:.6f}",
                "val_acc": f"{val_metrics['ACC']:.6f}",
                "val_f1": f"{val_metrics['F1']:.6f}",
                "val_mcc": f"{val_metrics['MCC']:.6f}",
                "val_pre": f"{val_metrics['Pre']:.6f}",
                "val_sn": f"{val_metrics['Sn']:.6f}",
                "val_sp": f"{val_metrics['Sp']:.6f}",
                "val_tp": val_metrics["TP"],
                "val_tn": val_metrics["TN"],
                "val_fp": val_metrics["FP"],
                "val_fn": val_metrics["FN"],
                "best_val_auc": f"{best_auc:.6f}",
                "best_epoch": best_epoch,
                "is_best": 1 if is_best else 0,
                "epochs_no_improve": epochs_no_improve,
                "epoch_time_sec": f"{elapsed:.2f}",
            }
            append_fold_epoch_log(fold_save_dir, row_dict)
            sync_unified_epoch_logs(base_save_dir, exp_tag)
            write_run_status(
                base_save_dir,
                stage=exp_tag,
                fold=fold_no,
                total_folds=5,
                epoch=epoch,
                total_epochs=args.epochs,
                best_val_auc=best_auc,
                best_epoch=best_epoch,
                state="running",
            )

            print(
                f"[Fold {fold_no}/5] Ep {epoch:>3}/{args.epochs:<3} | "
                f"TrLoss: {train_metrics['Loss']:.4f} TrAUC: {train_metrics['AUC']:.4f} | "
                f"ValLoss: {val_metrics['Loss']:.4f} ValAUC: {val_metrics['AUC']:.4f} "
                f"ValAUPR: {val_metrics['AUPR']:.4f} ValF1: {val_metrics['F1']:.4f} | "
                f"BestAUC: {best_auc:.4f} (Ep {best_epoch}) {'*' if is_best else ''}",
                flush=True,
            )

            atomic_torch_save(
                {
                    "fold": fold_no,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "best_epoch": best_epoch,
                    "stopper_counter": epochs_no_improve,
                },
                latest_ckpt_path,
                backup_prev_path=prev_ckpt_path,
            )

        if os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for d, p, label in val_loader:
                d, p = d.to(device), p.to(device)
                pred = model(d, p)
                probs.extend(pred.cpu().numpy().flatten())
                trues.extend(label.numpy().flatten())

        probs_arr = np.array(probs)
        trues_arr = np.array(trues)
        np.savez(fold_npz, y_true=trues_arr, y_scores=probs_arr)

        for p_clean in (latest_ckpt_path, prev_ckpt_path):
            if os.path.exists(p_clean):
                try:
                    os.remove(p_clean)
                except OSError:
                    pass

        metrics = calculate_metrics(trues_arr, probs_arr)
        all_fold_results.append(metrics)
        for i, idx in enumerate(val_idx):
            oof_records.append(
                {
                    "Dataset": ds_clean,
                    "Fold": fold_no,
                    "LineIdx": int(idx),
                    "True_Label": int(trues_arr[i]),
                    "Probability": float(probs_arr[i]),
                    "Prediction": 1 if probs_arr[i] >= 0.5 else 0,
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
        fold=5,
        total_folds=5,
        epoch=args.epochs,
        total_epochs=args.epochs,
        best_val_auc=float(df_res["AUC"].mean()),
        best_epoch=-1,
        state="completed",
    )
    aggregate_all_epoch_histories("./result")


if __name__ == "__main__":
    main()



# -*- coding: utf-8 -*-
"""
Unified Per-Epoch Full Metrics Logger & Atomic Checkpoint Resume Manager
========================================================================
Modeled after DNAPred_Luo (`model_plus/train.py` & `model_plus/run.py`):
1. Logs complete per-epoch Train & Validation metrics (Loss, AUC, AUPR, ACC, F1, MCC,
   Pre, Sn, Sp, TP, TN, FP, FN, LR, Best_AUC, Best_Epoch, Time) to:
   - `<base_save_dir>/fold_<k>/epoch_metrics_fold_<k>.csv`
   - `<base_save_dir>/fold_<k>/epoch_metrics_fold_<k>.jsonl`
   - `<base_save_dir>/epoch_metrics_all_folds.csv`
   - `<base_save_dir>/train_log.txt` (and `./log/<exp_name>_train_log.txt`)
2. Atomically saves per-epoch checkpoints (`checkpoint_latest_fold_<k>.pt` with automatic
   backup `checkpoint_prev_fold_<k>.pt`, plus `valid_best_checkpoint.pth` and `last_checkpoint.pth`)
   using `.tmp` + `os.replace` so power outages or process kills never corrupt model files.
3. Automatically resumes from the exact interrupted fold and epoch on restart.
"""

import csv
import json
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


EPOCH_CSV_COLUMNS = [
    "fold",
    "epoch",
    "lr",
    "train_loss",
    "train_auc",
    "train_aupr",
    "train_acc",
    "train_f1",
    "train_mcc",
    "train_pre",
    "train_sn",
    "train_sp",
    "train_tp",
    "train_tn",
    "train_fp",
    "train_fn",
    "val_loss",
    "val_auc",
    "val_aupr",
    "val_acc",
    "val_f1",
    "val_mcc",
    "val_pre",
    "val_sn",
    "val_sp",
    "val_tp",
    "val_tn",
    "val_fp",
    "val_fn",
    "best_val_auc",
    "best_epoch",
    "is_best",
    "epochs_no_improve",
    "epoch_time_sec",
]


def compute_detailed_metrics(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    if len(y_true) == 0:
        return {
            "AUC": 0.5, "AUPR": 0.0, "ACC": 0.0, "F1": 0.0, "MCC": 0.0,
            "Pre": 0.0, "Sn": 0.0, "Sp": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        }
    y_pred = (y_scores >= 0.5).astype(int)

    auc_val = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.5
    aupr_val = float(average_precision_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel() if conf.size == 4 else (0, 0, 0, 0)
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    sn = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    sp = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "AUC": auc_val,
        "AUPR": aupr_val,
        "ACC": acc,
        "F1": f1,
        "MCC": mcc,
        "Pre": pre,
        "Sn": sn,
        "Sp": sp,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def atomic_torch_save(obj, target_path, backup_prev_path=None):
    """Atomically saves a PyTorch object via a temporary file + os.replace."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    if backup_prev_path and os.path.exists(target_path):
        try:
            shutil.copy2(target_path, backup_prev_path)
        except Exception:
            pass
    tmp_path = target_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def atomic_json_save(payload, target_path):
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def write_run_status(base_save_dir, status=None, extra=None, **kwargs):
    status_val = status or kwargs.pop("state", "in_progress")
    payload = {
        "pipeline": "ArcMDTI_code_new",
        "status": status_val,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    if kwargs:
        payload.update(kwargs)
    atomic_json_save(payload, os.path.join(base_save_dir, "run_status.json"))


def load_checkpoint_safely(latest_path_or_dir, prev_path_or_fold, legacy_path_or_device=None, device=None):
    """
    Attempts to load the latest checkpoint; falls back to prev_path or legacy_path if needed.
    Supports both:
      - load_checkpoint_safely(fold_save_dir, fold_no, device)
      - load_checkpoint_safely(latest_path, prev_path, legacy_path, device)
    """
    if isinstance(prev_path_or_fold, int):
        fold_save_dir = latest_path_or_dir
        fold_no = prev_path_or_fold
        dev = legacy_path_or_device
        latest_path = os.path.join(fold_save_dir, f"checkpoint_latest_fold_{fold_no}.pt")
        prev_path = os.path.join(fold_save_dir, f"checkpoint_prev_fold_{fold_no}.pt")
        legacy_path = os.path.join(fold_save_dir, "last_checkpoint.pth")
    else:
        latest_path = latest_path_or_dir
        prev_path = prev_path_or_fold
        legacy_path = legacy_path_or_device
        dev = device

    for candidate in (latest_path, prev_path, legacy_path):
        if candidate and os.path.exists(candidate):
            try:
                try:
                    ckpt = torch.load(candidate, map_location=dev, weights_only=False)
                except TypeError:
                    ckpt = torch.load(candidate, map_location=dev)
                if isinstance(ckpt, dict) and "epoch" in ckpt and "model_state_dict" in ckpt:
                    return ckpt, candidate
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint {candidate}: {e}", flush=True)
    return None, None


def append_fold_epoch_log(fold_save_dir, row_dict):
    """Appends a single epoch row to per-fold CSV and JSONL files."""
    os.makedirs(fold_save_dir, exist_ok=True)
    fold_no = int(row_dict.get("fold", 1))
    fold_csv_path = os.path.join(fold_save_dir, f"epoch_metrics_fold_{fold_no}.csv")
    fold_jsonl_path = os.path.join(fold_save_dir, f"epoch_metrics_fold_{fold_no}.jsonl")
    if not os.path.exists(fold_csv_path):
        with open(fold_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
    with open(fold_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
        writer.writerow({k: row_dict.get(k, "") for k in EPOCH_CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    with open(fold_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def truncate_fold_logs_upto_epoch(fold_csv_or_dir, fold_jsonl_or_max_epoch, max_epoch=None):
    """Ensures per-fold CSV and JSONL logs only retain completed epochs <= max_epoch on resume."""
    if max_epoch is None and isinstance(fold_jsonl_or_max_epoch, int):
        fold_dir = fold_csv_or_dir
        max_ep = fold_jsonl_or_max_epoch
        # Find fold_metrics files in fold_dir
        csv_candidates = [
            os.path.join(fold_dir, f) for f in os.listdir(fold_dir) if f.endswith(".csv") and "epoch_metrics" in f
        ] if os.path.exists(fold_dir) else []
        jsonl_candidates = [
            os.path.join(fold_dir, f) for f in os.listdir(fold_dir) if f.endswith(".jsonl") and "epoch_metrics" in f
        ] if os.path.exists(fold_dir) else []
        fold_csv_path = csv_candidates[0] if csv_candidates else os.path.join(fold_dir, "epoch_metrics_fold_1.csv")
        fold_jsonl_path = jsonl_candidates[0] if jsonl_candidates else os.path.join(fold_dir, "epoch_metrics_fold_1.jsonl")
    else:
        fold_csv_path = fold_csv_or_dir
        fold_jsonl_path = fold_jsonl_or_max_epoch
        max_ep = max_epoch

    if os.path.exists(fold_csv_path):
        kept_rows = []
        try:
            with open(fold_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get("epoch", 0)) <= max_ep:
                        kept_rows.append(row)
        except Exception as e:
            print(f"[WARNING] Could not parse {fold_csv_path} during resume cleanup: {e}", flush=True)

        tmp_csv = fold_csv_path + ".tmp"
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
            writer.writeheader()
            for r in kept_rows:
                writer.writerow({k: r.get(k, "") for k in EPOCH_CSV_COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_csv, fold_csv_path)

    if os.path.exists(fold_jsonl_path):
        kept_lines = []
        try:
            with open(fold_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        obj = json.loads(line_s)
                        if int(obj.get("epoch", 0)) <= max_ep:
                            kept_lines.append(line_s)
                    except Exception:
                        continue
        except Exception as e:
            print(f"[WARNING] Could not parse {fold_jsonl_path} during resume cleanup: {e}", flush=True)

        tmp_jsonl = fold_jsonl_path + ".tmp"
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for l in kept_lines:
                f.write(l + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_jsonl, fold_jsonl_path)


def sync_unified_epoch_logs(base_save_dir, k_folds=5, exp_tag=None):
    """
    Synchronizes `epoch_metrics_all_folds.csv` and `train_log.txt` across all folds in
    `base_save_dir`, and mirrors `train_log.txt` to `./log/<exp_tag>_train_log.txt`.
    """
    if isinstance(k_folds, str) and exp_tag is None:
        exp_tag = k_folds
        k_folds = 5
    all_rows = []
    for fold_no in range(1, k_folds + 1):
        fold_csv = os.path.join(base_save_dir, f"fold_{fold_no}", f"epoch_metrics_fold_{fold_no}.csv")
        if not os.path.exists(fold_csv):
            fold_csv_alt = os.path.join(base_save_dir, f"epoch_metrics_fold_{fold_no}.csv")
            if os.path.exists(fold_csv_alt):
                fold_csv = fold_csv_alt
        if os.path.exists(fold_csv):
            try:
                with open(fold_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_rows.append(row)
            except Exception:
                pass

    if not all_rows:
        return

    all_csv_path = os.path.join(base_save_dir, "epoch_metrics_all_folds.csv")
    tmp_all_csv = all_csv_path + ".tmp"
    with open(tmp_all_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in EPOCH_CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_all_csv, all_csv_path)

    unified_log_path = os.path.join(base_save_dir, "train_log.txt")
    tmp_log = unified_log_path + ".tmp"
    with open(tmp_log, "w", encoding="utf-8") as f:
        header_cols = [
            "Fold", "Epoch", "LR",
            "Train_Loss", "Train_AUC", "Train_AUPR", "Train_ACC", "Train_F1", "Train_MCC", "Train_Pre", "Train_Sn", "Train_Sp",
            "Val_Loss", "Val_AUC", "Val_AUPR", "Val_ACC", "Val_F1", "Val_MCC", "Val_Pre", "Val_Sn", "Val_Sp",
            "Best_Val_AUC", "Best_Epoch", "Is_Best", "No_Improve", "Time_s",
        ]
        f.write("\t".join(header_cols) + "\n")
        for r in all_rows:
            vals = [
                str(r.get("fold", "")),
                str(r.get("epoch", "")),
                str(r.get("lr", "")),
                str(r.get("train_loss", "")),
                str(r.get("train_auc", "")),
                str(r.get("train_aupr", "")),
                str(r.get("train_acc", "")),
                str(r.get("train_f1", "")),
                str(r.get("train_mcc", "")),
                str(r.get("train_pre", "")),
                str(r.get("train_sn", "")),
                str(r.get("train_sp", "")),
                str(r.get("val_loss", "")),
                str(r.get("val_auc", "")),
                str(r.get("val_aupr", "")),
                str(r.get("val_acc", "")),
                str(r.get("val_f1", "")),
                str(r.get("val_mcc", "")),
                str(r.get("val_pre", "")),
                str(r.get("val_sn", "")),
                str(r.get("val_sp", "")),
                str(r.get("best_val_auc", "")),
                str(r.get("best_epoch", "")),
                str(r.get("is_best", "")),
                str(r.get("epochs_no_improve", "")),
                str(r.get("epoch_time_sec", "")),
            ]
            f.write("\t".join(vals) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_log, unified_log_path)

    if exp_tag:
        os.makedirs("./log", exist_ok=True)
        shutil.copy2(unified_log_path, os.path.join("./log", f"{exp_tag}_train_log.txt"))
        shutil.copy2(all_csv_path, os.path.join("./log", f"{exp_tag}_epoch_metrics.csv"))


def write_fold_best_metrics(fold_txt_path, fold_no, best_epoch, best_metrics):
    if not best_metrics:
        return
    tmp_path = fold_txt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"--- Fold {fold_no} Best Results (Epoch {best_epoch}) ---\n")
        for k in ["AUC", "AUPR", "ACC", "F1", "MCC", "Pre", "Sn", "Sp"]:
            if k in best_metrics:
                f.write(f"{k}: {best_metrics[k]:.6f}\n")
        for k in ["TP", "TN", "FP", "FN"]:
            if k in best_metrics:
                f.write(f"{k}: {best_metrics[k]}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, fold_txt_path)


def train_and_eval_epoch_arcmdti(model, train_loader, val_loader, optimizer, criterion, device, grad_clip=1.0):
    """
    Runs 1 training epoch and 1 validation pass for ArcMDTI, returning:
      (train_loss_avg, train_metrics, val_loss_avg, val_metrics)
    """
    model.train()
    train_losses = []
    train_true, train_scores = [], []

    for batch in train_loader:
        if batch is None:
            continue
        d, p, ps, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(d, p, ps, labels=labels)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append(float(loss.item()))

        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            train_true.extend(labels.detach().cpu().numpy())
            train_scores.extend(probs.cpu().numpy())

    train_loss_avg = float(np.mean(train_losses)) if train_losses else 0.0
    train_metrics = compute_detailed_metrics(train_true, train_scores)

    model.eval()
    val_losses = []
    val_true, val_scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            d, p, ps, labels = [x.to(device) for x in batch]
            logits = model(d, p, ps, labels=None)
            loss = criterion(logits, labels)
            val_losses.append(float(loss.item()))
            probs = torch.softmax(logits, dim=1)[:, 1]
            val_true.extend(labels.cpu().numpy())
            val_scores.extend(probs.cpu().numpy())

    val_loss_avg = float(np.mean(val_losses)) if val_losses else 0.0
    val_metrics = compute_detailed_metrics(val_true, val_scores)

    return train_loss_avg, train_metrics, val_loss_avg, val_metrics


def record_epoch_and_save_checkpoint(
    fold_save_dir,
    base_save_dir,
    fold_no,
    k_folds,
    epoch,
    max_epochs,
    patience,
    current_lr,
    train_loss_avg,
    train_metrics,
    val_loss_avg,
    val_metrics,
    best_val_auc,
    best_epoch,
    best_metrics,
    is_best,
    epochs_no_improve,
    epoch_elapsed,
    model,
    optimizer,
    scheduler,
    exp_tag=None,
):
    """
    Appends the current epoch's metrics to CSV/JSONL/TXT logs and atomically saves
    both the best model and the resumable latest checkpoint.
    """
    os.makedirs(fold_save_dir, exist_ok=True)
    fold_csv_path = os.path.join(fold_save_dir, f"epoch_metrics_fold_{fold_no}.csv")
    fold_jsonl_path = os.path.join(fold_save_dir, f"epoch_metrics_fold_{fold_no}.jsonl")
    fold_txt_path = os.path.join(fold_save_dir, f"fold_{fold_no}_best_metrics.txt")

    best_ckpt_path = os.path.join(fold_save_dir, "valid_best_checkpoint.pth")
    latest_model_path = os.path.join(fold_save_dir, f"model_latest_fold_{fold_no}.pth")
    ckpt_latest_path = os.path.join(fold_save_dir, f"checkpoint_latest_fold_{fold_no}.pt")
    ckpt_prev_path = os.path.join(fold_save_dir, f"checkpoint_prev_fold_{fold_no}.pt")
    legacy_last_ckpt_path = os.path.join(fold_save_dir, "last_checkpoint.pth")

    if not os.path.exists(fold_csv_path):
        with open(fold_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    if is_best:
        atomic_torch_save(model.state_dict(), best_ckpt_path)
        write_fold_best_metrics(fold_txt_path, fold_no, best_epoch, best_metrics)

    row_dict = {
        "fold": fold_no,
        "epoch": epoch,
        "lr": f"{current_lr:.8f}",
        "train_loss": f"{train_loss_avg:.6f}",
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
        "val_loss": f"{val_loss_avg:.6f}",
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
        "best_val_auc": f"{best_val_auc:.6f}",
        "best_epoch": best_epoch,
        "is_best": int(is_best),
        "epochs_no_improve": epochs_no_improve,
        "epoch_time_sec": f"{epoch_elapsed:.3f}",
    }

    with open(fold_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
        writer.writerow(row_dict)
        f.flush()
        os.fsync(f.fileno())

    jsonl_record = {
        "fold": fold_no,
        "epoch": epoch,
        "lr": current_lr,
        "train": {"loss": train_loss_avg, **train_metrics},
        "val": {"loss": val_loss_avg, **val_metrics},
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "is_best": bool(is_best),
        "epochs_no_improve": epochs_no_improve,
        "epoch_time_sec": round(epoch_elapsed, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(fold_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    sync_unified_epoch_logs(base_save_dir, k_folds=k_folds, exp_tag=exp_tag)

    will_stop = (epoch >= max_epochs) or (epochs_no_improve >= patience)

    atomic_torch_save(model.state_dict(), latest_model_path)
    ckpt_payload = {
        "fold": fold_no,
        "epoch": epoch,
        "completed": bool(will_stop),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_auc": best_val_auc,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epochs_no_improve": epochs_no_improve,
        "stopper_counter": epochs_no_improve,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if os.path.exists(ckpt_latest_path):
        try:
            shutil.copy2(ckpt_latest_path, ckpt_prev_path)
        except Exception:
            pass
    atomic_torch_save(ckpt_payload, ckpt_latest_path)
    atomic_torch_save(ckpt_payload, legacy_last_ckpt_path)

    write_run_status(
        base_save_dir,
        status="in_progress",
        extra={
            "current_fold": fold_no,
            "current_epoch": epoch,
            "max_epochs": max_epochs,
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
        },
    )

    best_tag = " [*BEST*]" if is_best else ""
    print(
        f"Fold {fold_no}/{k_folds} | Ep {epoch:03d}/{max_epochs} | LR: {current_lr:.2e} | "
        f"Train [Loss: {train_loss_avg:.4f}, AUC: {train_metrics['AUC']:.4f}, AUPR: {train_metrics['AUPR']:.4f}, "
        f"ACC: {train_metrics['ACC']:.4f}, F1: {train_metrics['F1']:.4f}, MCC: {train_metrics['MCC']:.4f}] | "
        f"Val [Loss: {val_loss_avg:.4f}, AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}, "
        f"ACC: {val_metrics['ACC']:.4f}, F1: {val_metrics['F1']:.4f}, MCC: {val_metrics['MCC']:.4f}, "
        f"Pre: {val_metrics['Pre']:.4f}, Sn: {val_metrics['Sn']:.4f}, Sp: {val_metrics['Sp']:.4f}] | "
        f"Best_AUC: {best_val_auc:.4f} (Ep {best_epoch}){best_tag} ({epoch_elapsed:.1f}s)",
        flush=True,
    )


def aggregate_all_epoch_histories(result_root="./result", output_csv="./result/all_epochs_training_history.csv"):
    """
    Scans all experiment directories under `result_root` for `epoch_metrics_all_folds.csv`
    and combines them into a single master CSV (`result/all_epochs_training_history.csv`).
    """
    if not os.path.exists(result_root):
        return 0
    dfs = []
    for root, _, files in os.walk(result_root):
        if "epoch_metrics_all_folds.csv" in files:
            fpath = os.path.join(root, "epoch_metrics_all_folds.csv")
            try:
                df = pd.read_csv(fpath)
                if not df.empty:
                    exp_name = os.path.relpath(root, result_root)
                    df.insert(0, "experiment", exp_name)
                    dfs.append(df)
            except Exception:
                pass
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        master_df.to_csv(output_csv, index=False)
        return len(master_df)
    return 0


def run_arcmdti_fold_training(
    fold_no,
    k_folds,
    fold_save_dir,
    base_save_dir,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    max_epochs=100,
    patience=25,
    grad_clip=1.0,
    exp_tag=None,
    force_retrain=False,
    skip_completed=True,
    hp=None,
    fold_record_path=None,
    **kwargs,
):
    """
    Executes a single fold of ArcMDTI training with full per-epoch Train/Val metric
    logging (CSV/JSONL/TXT) and atomic per-epoch checkpointing & automatic interruption resume.
    Returns: (final_fold_metrics_dict, y_true_array, y_scores_array)
    """
    if hp is not None:
        max_epochs = int(getattr(hp, "Epoch", max_epochs))
        patience = int(getattr(hp, "Patience", patience))
        grad_clip = float(getattr(hp, "grad_clip", grad_clip))

    os.makedirs(fold_save_dir, exist_ok=True)
    npz_path = fold_record_path if fold_record_path else os.path.join(fold_save_dir, "ROC_PRC_data.npz")
    best_ckpt_path = os.path.join(fold_save_dir, "valid_best_checkpoint.pth")
    ckpt_latest_path = os.path.join(fold_save_dir, f"checkpoint_latest_fold_{fold_no}.pt")
    ckpt_prev_path = os.path.join(fold_save_dir, f"checkpoint_prev_fold_{fold_no}.pt")
    legacy_last_ckpt_path = os.path.join(fold_save_dir, "last_checkpoint.pth")
    fold_csv_path = os.path.join(fold_save_dir, f"epoch_metrics_fold_{fold_no}.csv")
    fold_jsonl_path = os.path.join(fold_save_dir, f"epoch_metrics_fold_{fold_no}.jsonl")
    fold_txt_path = os.path.join(fold_save_dir, f"fold_{fold_no}_best_metrics.txt")

    if not force_retrain and skip_completed and os.path.exists(npz_path) and os.path.exists(best_ckpt_path):
        print(f"[RESUME] Fold {fold_no}/{k_folds} already completed. Loading {npz_path}...", flush=True)
        data = np.load(npz_path)
        metrics = compute_detailed_metrics(data["y_true"], data["y_scores"])
        sync_unified_epoch_logs(base_save_dir, k_folds=k_folds, exp_tag=exp_tag)
        return metrics, data["y_true"], data["y_scores"]

    start_epoch = 1
    best_val_auc = -1.0
    best_epoch = 0
    best_metrics = {}
    epochs_no_improve = 0

    if not force_retrain:
        ckpt, loaded_from = load_checkpoint_safely(
            ckpt_latest_path, ckpt_prev_path, legacy_last_ckpt_path, device
        )
        if ckpt is not None:
            completed_epoch = int(ckpt.get("epoch", 0))
            if ckpt.get("completed", False) and os.path.exists(best_ckpt_path) and os.path.exists(npz_path):
                data = np.load(npz_path)
                metrics = compute_detailed_metrics(data["y_true"], data["y_scores"])
                print(
                    f"[RESUME] Fold {fold_no}/{k_folds} already completed at Epoch {completed_epoch} "
                    f"(Best Val AUC: {metrics['AUC']:.4f}). Skipping fold training.",
                    flush=True,
                )
                sync_unified_epoch_logs(base_save_dir, k_folds=k_folds, exp_tag=exp_tag)
                return metrics, data["y_true"], data["y_scores"]
            elif completed_epoch > 0:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                best_val_auc = float(ckpt.get("best_val_auc", ckpt.get("best_auc", -1.0)))
                best_epoch = int(ckpt.get("best_epoch", completed_epoch))
                best_metrics = ckpt.get("best_metrics", {})
                epochs_no_improve = int(ckpt.get("epochs_no_improve", ckpt.get("stopper_counter", 0)))
                start_epoch = completed_epoch + 1
                truncate_fold_logs_upto_epoch(fold_csv_path, fold_jsonl_path, completed_epoch)
                sync_unified_epoch_logs(base_save_dir, k_folds=k_folds, exp_tag=exp_tag)
                print(
                    f"[RESUME] Restored Fold {fold_no}/{k_folds} from {os.path.basename(loaded_from)} "
                    f"(Completed Epoch {completed_epoch}/{max_epochs}, Best Val AUC: {best_val_auc:.4f} at Ep {best_epoch}, "
                    f"No-improve: {epochs_no_improve}/{patience}). Continuing at Epoch {start_epoch}...",
                    flush=True,
                )

    if start_epoch == 1:
        with open(fold_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
        with open(fold_jsonl_path, "w", encoding="utf-8") as f:
            pass

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_loss_avg, train_metrics, val_loss_avg, val_metrics = train_and_eval_epoch_arcmdti(
            model, train_loader, val_loader, optimizer, criterion, device, grad_clip=grad_clip
        )
        if scheduler is not None:
            scheduler.step()

        epoch_elapsed = time.time() - t0
        is_best = False
        if val_metrics["AUC"] > best_val_auc:
            is_best = True
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        record_epoch_and_save_checkpoint(
            fold_save_dir=fold_save_dir,
            base_save_dir=base_save_dir,
            fold_no=fold_no,
            k_folds=k_folds,
            epoch=epoch,
            max_epochs=max_epochs,
            patience=patience,
            current_lr=current_lr,
            train_loss_avg=train_loss_avg,
            train_metrics=train_metrics,
            val_loss_avg=val_loss_avg,
            val_metrics=val_metrics,
            best_val_auc=best_val_auc,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            is_best=is_best,
            epochs_no_improve=epochs_no_improve,
            epoch_elapsed=epoch_elapsed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            exp_tag=exp_tag,
        )

        if epochs_no_improve >= patience:
            print(
                f"[Early Stopping] Fold {fold_no}/{k_folds} stopped at Epoch {epoch} "
                f"(Best Val AUC: {best_val_auc:.4f} at Epoch {best_epoch}).",
                flush=True,
            )
            break

    # Load best fold checkpoint and evaluate final validation curves & OOF
    if os.path.exists(best_ckpt_path):
        try:
            state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state)

    model.eval()
    y_true_fold, y_scores_fold = [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            d, p, ps, labels = [x.to(device) for x in batch]
            logits = model(d, p, ps, labels=None)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true_fold.extend(labels.cpu().numpy())
            y_scores_fold.extend(probs.cpu().numpy())

    y_true_fold = np.asarray(y_true_fold, dtype=int)
    y_scores_fold = np.asarray(y_scores_fold, dtype=float)
    final_fold_res = compute_detailed_metrics(y_true_fold, y_scores_fold)

    fpr, tpr, _ = roc_curve(y_true_fold, y_scores_fold) if len(np.unique(y_true_fold)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
    p_pts, r_pts, _ = precision_recall_curve(y_true_fold, y_scores_fold) if len(np.unique(y_true_fold)) > 1 else (np.array([0, 1]), np.array([1, 0]), None)
    tmp_npz = npz_path + ".tmp.npz"
    np.savez(
        tmp_npz,
        fpr=fpr,
        tpr=tpr,
        precision=p_pts,
        recall=r_pts,
        y_true=y_true_fold,
        y_scores=y_scores_fold,
    )
    os.replace(tmp_npz, npz_path)

    write_fold_best_metrics(fold_txt_path, fold_no, best_epoch, final_fold_res)

    # Mark checkpoint as completed
    ckpt, _ = load_checkpoint_safely(ckpt_latest_path, ckpt_prev_path, legacy_last_ckpt_path, device)
    if ckpt is not None and not ckpt.get("completed", False):
        ckpt["completed"] = True
        atomic_torch_save(ckpt, ckpt_latest_path)
        atomic_torch_save(ckpt, legacy_last_ckpt_path)

    sync_unified_epoch_logs(base_save_dir, k_folds=k_folds, exp_tag=exp_tag)
    return final_fold_res, y_true_fold, y_scores_fold


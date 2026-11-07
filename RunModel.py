# -*- coding:utf-8 -*-
"""
ArcMDTI Standard 5-Fold Cross-Validation Runner (code_new/RunModel.py)
======================================================================
Improvements over legacy RunModel.py:
1. Computes all 8 evaluation metrics per fold: AUC, AUPR (PRC), ACC, F1, MCC, Pre, Sn, Sp.
2. Automatically exports out-of-fold predictions (`oof_predictions_<dataset>.csv`) during CV.
3. Supports both fold-level resume (`ROC_PRC_data.npz`) and epoch-level resume (`last_checkpoint.pth`).
4. Extracts 256-D hyperspherical features and geodesic angles (theta_0, theta_1) from the best fold.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    roc_curve,
    precision_score,
    auc,
)
from sklearn.decomposition import PCA

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
from plot_best_features import generate_sci_ready_dr_plots

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_trusted_checkpoint(path, map_location=DEVICE):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def compute_all_metrics(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= 0.5).astype(int)

    auc_val = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    aupr_val = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel() if conf.size == 4 else (0, 0, 0, 0)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": float(auc_val),
        "AUPR": float(aupr_val),
        "ACC": float(acc),
        "F1": float(f1),
        "MCC": float(mcc),
        "Pre": float(pre),
        "Sn": float(sn),
        "Sp": float(sp),
    }


def eval_model(model, loader, save_curves_path=None, return_raw=False):
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            d, p, ps, labels = [x.to(DEVICE) for x in batch]
            logits = model(d, p, ps, labels=None)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    metrics = compute_all_metrics(y_true, y_scores)

    if save_curves_path:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        p_pts, r_pts, _ = precision_recall_curve(y_true, y_scores)
        np.savez(
            save_curves_path,
            fpr=fpr,
            tpr=tpr,
            precision=p_pts,
            recall=r_pts,
            y_true=y_true,
            y_scores=y_scores,
        )

    if return_raw:
        return metrics, y_true, y_scores
    return metrics


def train_epoch(model, loader, optimizer, criterion, grad_clip=1.0):
    model.train()
    train_loss = []
    for batch in tqdm(loader, desc="Training", leave=False, file=sys.stdout, ncols=80, mininterval=10.0):
        if batch is None:
            continue
        d, p, ps, labels = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()
        logits = model(d, p, ps, labels=labels)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_loss.append(loss.item())

    return float(np.mean(train_loss)) if train_loss else 0.0


def validate_features(features, labels, save_dir, dataset_name):
    """Quantitative and visual PCA check of learned 256-D representations."""
    pos_mask = labels == 1
    neg_mask = labels == 0

    if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
        pos_feats = features[pos_mask]
        neg_feats = features[neg_mask]

        pos_mean = pos_feats.mean(axis=0)
        neg_mean = neg_feats.mean(axis=0)
        center_dist = float(np.linalg.norm(pos_mean - neg_mean))

        pos_var = float(pos_feats.var(axis=0).mean())
        neg_var = float(neg_feats.var(axis=0).mean())
        within_var = (pos_var + neg_var) / 2.0

        separability = center_dist / np.sqrt(within_var) if within_var > 0 else 0.0

        print(f"\n[Feature Quality] {dataset_name}: Center Dist={center_dist:.4f}, Within Var={within_var:.4f}, Ratio={separability:.4f}")

        pca = PCA(n_components=2, random_state=42)
        pca_embedding = pca.fit_transform(features)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#C44E52" if l == 1 else "#4C72B0" for l in labels]
        ax.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=colors, s=10, alpha=0.5)

        pos_center_pca = pca.transform(pos_mean.reshape(1, -1))
        neg_center_pca = pca.transform(neg_mean.reshape(1, -1))
        ax.scatter(
            pos_center_pca[0, 0], pos_center_pca[0, 1],
            c="red", s=180, marker="X", edgecolors="black", linewidth=1.8, label="Positive Center"
        )
        ax.scatter(
            neg_center_pca[0, 0], neg_center_pca[0, 1],
            c="blue", s=180, marker="X", edgecolors="black", linewidth=1.8, label="Negative Center"
        )

        ax.set_xlabel("PCA Dimension 1", fontsize=12)
        ax.set_ylabel("PCA Dimension 2", fontsize=12)
        ax.set_title(f"PCA Visualization - {dataset_name} (Separability: {separability:.3f})", fontsize=13)
        ax.legend()

        pca_save_path = os.path.join(save_dir, f"PCA_SanityCheck_{dataset_name}.png")
        plt.savefig(pca_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return separability
    return 0.0


def build_oof_records_for_fold(all_lines, val_idx, y_true, y_scores, dataset_name, fold_no):
    records = []
    for i, line_idx in enumerate(val_idx):
        parts = all_lines[line_idx].split()
        drug_id, prot_id = (parts[0], parts[1]) if len(parts) >= 5 else ("N/A", "N/A")
        prob = float(y_scores[i])
        records.append({
            "Dataset": dataset_name,
            "Fold": int(fold_no),
            "LineIdx": int(line_idx),
            "Drug_ID": drug_id,
            "Protein_ID": prot_id,
            "True_Label": int(y_true[i]),
            "Probability": prob,
            "Prediction": 1 if prob >= 0.5 else 0,
        })
    return records


def run_5_fold_cv(dataset_path, k_folds=5, seed=42, custom_batch_size=None, custom_epochs=None, force_retrain=False):
    setup_seed(seed)
    hp = hyperparameter()
    hp.select_dataset_config(dataset_path)
    if custom_batch_size:
        hp.Batch_size = int(custom_batch_size)
    if custom_epochs:
        hp.Epoch = int(custom_epochs)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = np.array([l.strip() for l in f if l.strip()])

    SS_DICT = load_secondary_structure_dict(dataset_path, all_lines=all_lines)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_fold_results = []
    oof_records = []

    dataset_name = os.path.basename(dataset_path).split(".")[0]
    base_save_dir = f"./result/{dataset_name}_CV_Seed{seed}"
    os.makedirs(base_save_dir, exist_ok=True)
    exp_tag = f"WarmStart_{dataset_name}_Seed{seed}"

    print(f"\n[ArcMDTI 5-Fold Warm-Start CV] Dataset: {dataset_name} | Folds: {k_folds} | Device: {DEVICE}", flush=True)
    write_run_status(base_save_dir, status="in_progress", extra={"dataset": dataset_name, "mode": "warm_start"})

    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_lines)):
        fold_no = fold + 1
        print(f"\n{'=' * 20} Fold {fold_no} / {k_folds} ({dataset_name}) {'=' * 20}", flush=True)
        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_no}")
        os.makedirs(fold_save_dir, exist_ok=True)

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

        final_fold_res, y_true_fold, y_scores_fold = run_arcmdti_fold_training(
            fold_no=fold_no,
            k_folds=k_folds,
            fold_save_dir=fold_save_dir,
            base_save_dir=base_save_dir,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=DEVICE,
            max_epochs=hp.Epoch,
            patience=hp.Patience,
            grad_clip=hp.grad_clip,
            exp_tag=exp_tag,
            force_retrain=force_retrain,
            skip_completed=not force_retrain,
        )
        all_fold_results.append(final_fold_res)
        oof_records.extend(
            build_oof_records_for_fold(all_lines, val_idx, y_true_fold, y_scores_fold, dataset_name, fold_no)
        )

    df = pd.DataFrame(all_fold_results)
    metric_cols = ["AUC", "AUPR", "ACC", "F1", "MCC", "Pre", "Sn", "Sp"]

    print("\n" + "★" * 20 + f" SCI FINAL {k_folds}-FOLD REPORT ({dataset_name}) " + "★" * 20, flush=True)
    summary_data = {}
    for col in metric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:6}: {mean_val:.4f} ± {std_val:.4f}", flush=True)
        summary_data[col] = f"{mean_val:.4f} ± {std_val:.4f}"

    df.index = [f"Fold_{i + 1}" for i in range(len(all_fold_results))]
    fold_csv_path = os.path.join(base_save_dir, f"{dataset_name}_per_fold_metrics.csv")
    df.to_csv(fold_csv_path, index_label="Fold")

    summary_df = pd.DataFrame([summary_data], index=["Mean ± Std"])
    summary_csv_path = os.path.join(base_save_dir, f"{dataset_name}_final_summary.csv")
    summary_df.to_csv(summary_csv_path, index_label="Stats")

    if oof_records:
        df_oof = pd.DataFrame(oof_records).sort_values("LineIdx").reset_index(drop=True)
        oof_csv_path = os.path.join(base_save_dir, f"oof_predictions_{dataset_name}.csv")
        df_oof.to_csv(oof_csv_path, index=False)
        print(f"Saved OOF predictions to: {oof_csv_path}", flush=True)

    write_run_status(base_save_dir, status="completed", extra={"dataset": dataset_name, "mode": "warm_start"})
    aggregate_all_epoch_histories()

    # ==================== Best-Fold Feature & Angle Extraction ====================
    best_fold_idx = int(np.argmax(df["AUC"].values))
    best_ckpt = os.path.join(base_save_dir, f"fold_{best_fold_idx + 1}", "valid_best_checkpoint.pth")
    if os.path.exists(best_ckpt):
        print(f"\nBest Fold Identified: Fold {best_fold_idx + 1}. Extracting latent manifold & hyperspherical angles...")
        all_data_loader = DataLoader(
            CustomDataSet(all_lines.tolist()), batch_size=hp.Batch_size, collate_fn=my_collate, shuffle=False
        )
        best_model = AMCADTI_V2(hp).to(DEVICE)
        best_model.load_state_dict(load_trusted_checkpoint(best_ckpt, map_location=DEVICE))
        best_model.eval()

        best_feats, best_labs, best_cosines, best_angles = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(all_data_loader, desc="Extracting Full-Space Features", file=sys.stdout, ncols=80):
                if batch is None:
                    continue
                d, p, ps, labels = [x.to(DEVICE) for x in batch]
                _, extracted_feat, attn_info = best_model(d, p, ps, labels=None, return_attn=True)
                best_feats.append(extracted_feat.cpu().numpy())
                best_labs.append(labels.cpu().numpy())
                if attn_info["cosines"] is not None:
                    best_cosines.append(attn_info["cosines"].cpu().numpy())
                    best_angles.append(attn_info["angles_deg"].cpu().numpy())

        features_matrix = np.concatenate(best_feats)
        labels_vector = np.concatenate(best_labs)
        cosines_matrix = np.concatenate(best_cosines) if best_cosines else None
        angles_matrix = np.concatenate(best_angles) if best_angles else None

        validate_features(features_matrix, labels_vector, base_save_dir, dataset_name)
        try:
            generate_sci_ready_dr_plots(
                features_matrix,
                labels_vector,
                save_dir=base_save_dir,
                dataset_name=dataset_name,
                angles_deg=angles_matrix,
                cosines=cosines_matrix,
            )
        except Exception as e:
            print(f"[WARNING] Visualization encountered an issue: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ArcMDTI with 5-Fold Cross-Validation")
    parser.add_argument("dataset", type=str, nargs="?", default="Enzyme.txt", help="Dataset filename")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=0, help="Override max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
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
    run_5_fold_cv(dataset_path, k_folds=args.folds, seed=args.seed, custom_batch_size=b_size, custom_epochs=ep)


# -*- coding:utf-8 -*-
"""
Out-of-Fold (OOF) Prediction Recovery Script for ArcMDTI (code_new/predict_oof.py)
==================================================================================
Reconstructs identical 5-fold splits and loads fold-specific checkpoints to generate
`oof_predictions_<dataset>.csv` if not already produced during `RunModel.py`.
Uses `load_secondary_structure_dict` for accurate SS alignment.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AMCADTI_V2
from config import hyperparameter
from utils.DataSetsFunction import CustomDataSet, collate_fn, load_secondary_structure_dict
from RunModel import DEVICE, load_trusted_checkpoint

DATASETS = [
    {"name": "Enzyme",      "txt": "DataSets/Enzyme.txt",      "result_dir": "result/Enzyme_CV_Seed42"},
    {"name": "GPCRs",       "txt": "DataSets/GPCRs.txt",       "result_dir": "result/GPCRs_CV_Seed42"},
    {"name": "ion_channel", "txt": "DataSets/ion_channel.txt", "result_dir": "result/ion_channel_CV_Seed42"},
    {"name": "Davis",       "txt": "DataSets/Davis.txt",       "result_dir": "result/Davis_CV_Seed42"},
    {"name": "KIBA",        "txt": "DataSets/KIBA.txt",        "result_dir": "result/KIBA_CV_Seed42"},
]

K_FOLDS = 5
SEED = 42


def run_oof_for_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    txt_path = dataset_cfg["txt"]
    result_dir = dataset_cfg["result_dir"]

    out_csv = os.path.join(result_dir, f"oof_predictions_{name}.csv")
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"[SKIP] OOF predictions already exist for {name}: {out_csv}")
        return out_csv

    if not os.path.exists(txt_path):
        print(f"[SKIP] Dataset file missing: {txt_path}")
        return None

    hp = hyperparameter()
    hp.select_dataset_config(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        all_lines = np.array([l.strip() for l in f if l.strip()])

    SS_DICT = load_secondary_structure_dict(txt_path, all_lines=all_lines)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)

    all_records = []
    for fold, (_, val_idx) in enumerate(kf.split(all_lines)):
        fold_num = fold + 1
        ckpt_path = os.path.join(result_dir, f"fold_{fold_num}", "valid_best_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Fold {fold_num}: missing checkpoint {ckpt_path}")
            continue

        model = AMCADTI_V2(hp).to(DEVICE)
        model.load_state_dict(load_trusted_checkpoint(ckpt_path, map_location=DEVICE))
        model.eval()

        val_lines = all_lines[val_idx].tolist()
        val_loader = DataLoader(
            CustomDataSet(val_lines),
            batch_size=hp.Batch_size,
            shuffle=False,
            collate_fn=my_collate,
            num_workers=0,
        )

        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Fold {fold_num}/{K_FOLDS}", leave=False, file=sys.stdout, ncols=80):
                if batch is None:
                    continue
                d, p, ps, labels = batch
                d, p, ps = d.to(DEVICE), p.to(DEVICE), ps.to(DEVICE)
                logits = model(d, p, ps, return_cl=False)
                probs = torch.softmax(logits, dim=1)[:, 1]
                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        for i, line_idx in enumerate(val_idx):
            parts = all_lines[line_idx].split()
            drug_id, prot_id = (parts[0], parts[1]) if len(parts) >= 5 else ("N/A", "N/A")
            prob = float(fold_probs[i])
            all_records.append({
                "Dataset": name,
                "Fold": fold_num,
                "LineIdx": int(line_idx),
                "Drug_ID": drug_id,
                "Protein_ID": prot_id,
                "True_Label": int(fold_labels[i]),
                "Probability": prob,
                "Prediction": 1 if prob >= 0.5 else 0,
            })

    if not all_records:
        return None

    os.makedirs(result_dir, exist_ok=True)
    df = pd.DataFrame(all_records).sort_values("LineIdx").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] OOF predictions saved to: {out_csv}")
    return out_csv


def main():
    for cfg in DATASETS:
        run_oof_for_dataset(cfg)


if __name__ == "__main__":
    main()


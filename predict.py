# -*- coding:utf-8 -*-
"""
5-Fold Ensemble Inference Script for ArcMDTI (code_new/predict.py)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AMCADTI_V2
from config import hyperparameter
from utils.DataSetsFunction import CustomDataSet, collate_fn, load_secondary_structure_dict
from RunModel import DEVICE, load_trusted_checkpoint


def process_and_predict_ensemble(model_dir, data_path, out_csv_path, folds=5):
    hp = hyperparameter()
    hp.select_dataset_config(data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        data_list = [line.strip() for line in f if line.strip()]

    SS_DICT = load_secondary_structure_dict(data_path, all_lines=data_list)
    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)

    test_loader = DataLoader(
        CustomDataSet(data_list),
        batch_size=hp.Batch_size,
        shuffle=False,
        collate_fn=my_collate,
        num_workers=0,
    )

    probs_by_fold = []
    for fold in range(1, folds + 1):
        fold_model_path = os.path.join(model_dir, f"fold_{fold}", "valid_best_checkpoint.pth")
        if not os.path.exists(fold_model_path):
            print(f"[WARN] Fold {fold} checkpoint missing: {fold_model_path}")
            continue

        model = AMCADTI_V2(hp).to(DEVICE)
        model.load_state_dict(load_trusted_checkpoint(fold_model_path, map_location=DEVICE))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting Fold {fold}", leave=False, file=sys.stdout, ncols=80):
                if batch is None:
                    continue
                drug_seq, prot_seq, prot_ss, _ = batch
                drug_seq, prot_seq, prot_ss = drug_seq.to(DEVICE), prot_seq.to(DEVICE), prot_ss.to(DEVICE)
                logits = model(drug_seq, prot_seq, prot_ss, return_cl=False)
                probs = torch.softmax(logits, dim=1)[:, 1]
                fold_probs.extend(probs.cpu().numpy())

        probs_by_fold.append(fold_probs)

    if not probs_by_fold:
        print("Error: No fold checkpoints found.")
        return

    probs_by_fold = np.array(probs_by_fold)
    ensemble_probs = np.mean(probs_by_fold, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    results = []
    for i, line in enumerate(data_list):
        parts = line.split()
        if len(parts) >= 5:
            drug_id, prot_id, smiles, seq, label = parts[0], parts[1], parts[2], parts[3], parts[4]
        elif len(parts) >= 3:
            drug_id, prot_id = "N/A", "N/A"
            smiles, seq, label = parts[0], parts[1], parts[2]
        else:
            continue

        row = {
            "Drug_ID": drug_id,
            "Protein_ID": prot_id,
            "SMILES": smiles,
            "Protein_Seq": seq,
            "True_Label": label,
        }
        for fold_idx in range(len(probs_by_fold)):
            row[f"Prob_Fold_{fold_idx + 1}"] = f"{probs_by_fold[fold_idx][i]:.6f}"
        row["Prob_Ensemble"] = f"{ensemble_probs[i]:.6f}"
        row["Prediction"] = int(ensemble_preds[i])
        results.append(row)

    pd.DataFrame(results).to_csv(out_csv_path, index=False)
    print(f"Saved 5-fold ensemble predictions to: {out_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python predict.py <model_dir> <test_txt_path> <out_csv_path>")
        sys.exit(1)
    process_and_predict_ensemble(sys.argv[1], sys.argv[2], sys.argv[3])


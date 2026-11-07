# -*- coding:utf-8 -*-
"""
Cold-Start K-Fold Split Generator (Cold-Drug & Cold-Target)
===========================================================
Ensures strictly disjoint drug sets (cold_drug) or protein target sets (cold_target)
between training and validation folds for realistic out-of-distribution evaluation.
"""

import numpy as np
from sklearn.model_selection import KFold


def get_cold_start_folds(all_lines, split_mode="cold_drug", k_folds=5, seed=42):
    """
    Generate cold-start drug or target K-Fold splits.
    split_mode: 'cold_drug' or 'cold_target'
    Returns a list of (train_idx, val_idx) tuples.
    """
    np.random.seed(seed)

    drugs = []
    targets = []

    for line in all_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            smiles, seq = parts[2], parts[3]
        else:
            smiles, seq = parts[0], parts[1]
        drugs.append(smiles)
        targets.append(seq)

    drugs = np.array(drugs)
    targets = np.array(targets)

    unique_drugs = np.unique(drugs)
    unique_targets = np.unique(targets)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []

    if split_mode in ("cold_drug", "cold-drug"):
        for _, val_d_idx in kf.split(unique_drugs):
            val_drugs = set(unique_drugs[val_d_idx])
            train_idx, val_idx = [], []
            for i, d in enumerate(drugs):
                if d in val_drugs:
                    val_idx.append(i)
                else:
                    train_idx.append(i)
            folds.append((np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)))

    elif split_mode in ("cold_target", "cold-target"):
        for _, val_p_idx in kf.split(unique_targets):
            val_targets = set(unique_targets[val_p_idx])
            train_idx, val_idx = [], []
            for i, p in enumerate(targets):
                if p in val_targets:
                    val_idx.append(i)
                else:
                    train_idx.append(i)
            folds.append((np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)))

    else:
        raise ValueError(f"split_mode must be 'cold_drug' or 'cold_target', got: {split_mode}")

    return folds


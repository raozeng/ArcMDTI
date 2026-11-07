# -*- coding:utf-8 -*-
"""
Dataset Loading, Tokenization, and Secondary Structure Alignment Utilities
==========================================================================
Fixes:
1. Remaps CHARISOSMISET indices 65..68 ('m', 'u', 't', 'y') to unused slots 25..28
   and clamps token IDs strictly within [0, vocab_size - 1] to prevent CUDA device asserts.
2. Gracefully resolves dataset-specific secondary structure (.pkl / _ss.pkl) dictionaries
   and aligns >proteinK keys with dataset protein IDs and sequences.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 9, "B": 42, "E": 43,
    "D": 44, "G": 45, "F": 10, "I": 11, "H": 46, "K": 47, "M": 48, "L": 12,
    "O": 13, "N": 14, "P": 15, "S": 16, "R": 49, "U": 50, "T": 17, "W": 51,
    "V": 52, "Y": 53, "[": 54, "Z": 55, "]": 56, "_": 57, "a": 58, "c": 18,
    "b": 59, "e": 60, "d": 61, "g": 62, "f": 19, "i": 63, "h": 64, "m": 25,
    "l": 20, "o": 21, "n": 22, "s": 23, "r": 24, "u": 26, "t": 27, "y": 28,
}

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25,
}

# DSSP 8-state to 3-state physical mapping:
# Coil/Loop (C, T, S, I) -> 1 | Helix (H, G) -> 2 | Beta-Strand (E, B) -> 3
SS_MAP = {"C": 1, "H": 2, "E": 3, "T": 1, "S": 1, "G": 2, "B": 3, "I": 1}


def resolve_ss_path(dataset_path):
    base_path = os.path.splitext(dataset_path)[0]
    dataset_name = os.path.basename(base_path)
    candidates = [
        base_path + ".pkl",
        base_path + "_ss.pkl",
        os.path.join("./DataSets", f"{dataset_name}.pkl"),
        os.path.join("./DataSets", f"{dataset_name}_ss.pkl"),
        "./DataSets/GPCRs.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def parse_dataset_line(item):
    parts = item.strip().split()
    if len(parts) >= 5:
        # Format: [Drug_ID, Prot_ID, SMILES, Prot_Seq, Label]
        return parts[1], parts[2], parts[3], parts[4]
    elif len(parts) >= 3:
        # Format: [SMILES, Prot_Seq, Label]
        smiles, seq, label_raw = parts[0], parts[1], parts[2]
        return seq, smiles, seq, label_raw
    raise ValueError(f"Invalid dataset line format: {item}")


def get_ss_from_map(ss_map_dict, pid, seq):
    if ss_map_dict:
        for key in (pid, f">{pid}", str(pid).lstrip(">"), seq, f">{seq}"):
            if key in ss_map_dict:
                return ss_map_dict[key]
    return "C" * len(seq)


def load_secondary_structure_dict(dataset_path, all_lines=None):
    ss_path = resolve_ss_path(dataset_path)
    ss_dict = {}
    if os.path.exists(ss_path):
        try:
            with open(ss_path, "rb") as f:
                ss_dict = pickle.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load SS pickle from {ss_path}: {e}")
            ss_dict = {}
    else:
        print(f"[WARNING] SS file {ss_path} not found; defaulting to coil ('C') priors.")

    if all_lines is not None and ss_dict:
        protein_keys = []
        for key in ss_dict:
            key_text = str(key)
            if key_text.startswith(">protein") and key_text[len(">protein"):].isdigit():
                protein_keys.append(key_text)

        if protein_keys:
            protein_keys.sort(key=lambda k: int(k[len(">protein"):]))
            for line, key in zip(all_lines, protein_keys):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                pid, _, seq, _ = parse_dataset_line(line)
                ss_value = ss_dict[key]
                ss_dict.setdefault(pid, ss_value)
                ss_dict.setdefault(f">{pid}", ss_value)
                ss_dict.setdefault(seq, ss_value)

    return ss_dict


def label_smiles(line, max_len=100, vocab_size=65):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_len]):
        idx = CHARISOSMISET.get(ch, 0)
        x[i] = idx if 0 <= idx < vocab_size else 0
    return x


def label_sequence(line, max_len=1000, vocab_size=26):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_len]):
        idx = CHARPROTSET.get(ch, 0)
        x[i] = idx if 0 <= idx < vocab_size else 0
    return x


def label_ss(line, max_len=1000, vocab_size=5):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_len]):
        idx = SS_MAP.get(ch, 0)
        x[i] = idx if 0 <= idx < vocab_size else 0
    return x


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


def load_ss_from_fasta(path):
    if not os.path.exists(path):
        return {}

    ss_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]
            elif current_id:
                if current_id in ss_dict:
                    ss_dict[current_id] += line
                else:
                    ss_dict[current_id] = line
    return ss_dict


def collate_fn(batch_data, ss_map_dict, max_prot_len=1000, max_drug_len=100):
    drug_list, prot_list, ss_list, label_list = [], [], [], []

    for item in batch_data:
        parts = item.strip().split()
        if len(parts) < 3:
            continue

        pid, smiles, seq, label_raw = parse_dataset_line(item)

        drug_vec = label_smiles(smiles, max_drug_len)
        prot_vec = label_sequence(seq, max_prot_len)
        ss_str = get_ss_from_map(ss_map_dict, pid, seq)
        ss_vec = label_ss(ss_str, max_prot_len)

        try:
            label_value = int(float(label_raw))
            label_value = 1 if label_value >= 1 else 0
        except Exception:
            label_value = 0

        drug_list.append(torch.tensor(drug_vec, dtype=torch.long))
        prot_list.append(torch.tensor(prot_vec, dtype=torch.long))
        ss_list.append(torch.tensor(ss_vec, dtype=torch.long))
        label_list.append(label_value)

    if len(drug_list) == 0:
        return None

    return (
        torch.stack(drug_list),
        torch.stack(prot_list),
        torch.stack(ss_list),
        torch.tensor(label_list, dtype=torch.long),
    )


import os
import py_compile
import unittest
from pathlib import Path

from utils.ColdStartSplit import get_cold_start_folds


class PipelineSmokeTests(unittest.TestCase):
    def test_all_python_files_compile_cleanly(self):
        root = Path(__file__).resolve().parent.parent
        py_files = list(root.rglob("*.py"))
        self.assertGreater(len(py_files), 15)
        for py_file in py_files:
            py_compile.compile(str(py_file), doraise=True)

    def test_cold_start_splits_are_strictly_entity_disjoint(self):
        # Generate synthetic DTI pairs with 20 drugs and 15 targets
        synthetic_data = []
        for d in range(20):
            for t in range(15):
                if (d + t) % 3 == 0:
                    label = (d + t) % 2
                    synthetic_data.append(f"DRUG_{d} TARGET_{t} CCO_{d} ACDEFGH_{t} {label}")

        # 1. Cold-Drug: zero drug SMILES overlap between train and test across all 5 folds
        drug_folds = get_cold_start_folds(synthetic_data, split_mode="cold_drug", k_folds=5, seed=42)
        self.assertEqual(len(drug_folds), 5)
        for train_idx, val_idx in drug_folds:
            train_drugs = {synthetic_data[i].split()[2] for i in train_idx}
            test_drugs = {synthetic_data[i].split()[2] for i in val_idx}
            self.assertEqual(len(train_drugs & test_drugs), 0)

        # 2. Cold-Target: zero target sequence overlap between train and test across all 5 folds
        target_folds = get_cold_start_folds(synthetic_data, split_mode="cold_target", k_folds=5, seed=42)
        self.assertEqual(len(target_folds), 5)
        for train_idx, val_idx in target_folds:
            train_targets = {synthetic_data[i].split()[3] for i in train_idx}
            test_targets = {synthetic_data[i].split()[3] for i in val_idx}
            self.assertEqual(len(train_targets & test_targets), 0)


if __name__ == "__main__":
    unittest.main()

import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def import_dataset_utils_with_fake_torch():
    fake_torch = types.ModuleType("torch")
    fake_torch.long = np.int64
    fake_torch.tensor = lambda value, dtype=None: np.array(value, dtype=dtype)
    fake_torch.stack = lambda values: np.stack(values)

    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")

    class FakeDataset:
        pass

    fake_data.Dataset = FakeDataset
    fake_utils.data = fake_data
    fake_torch.utils = fake_utils

    sys.modules["torch"] = fake_torch
    sys.modules["torch.utils"] = fake_utils
    sys.modules["torch.utils.data"] = fake_data

    sys.modules.pop("utils.DataSetsFunction", None)
    return importlib.import_module("utils.DataSetsFunction")


class SecondaryStructureAlignmentTests(unittest.TestCase):
    def test_collate_fn_matches_secondary_structure_key_with_gt_prefix(self):
        dataset_utils = import_dataset_utils_with_fake_torch()

        _, _, ss_tensor, _ = dataset_utils.collate_fn(
            ["CHEMBL1 P12345 CCO ACDE 1"],
            ss_map_dict={">P12345": "HEC"},
            max_prot_len=4,
        )

        self.assertEqual(ss_tensor[0].tolist(), [2, 3, 1, 0])

    def test_resolve_ss_path_prefers_dataset_ss_pkl_before_fallback(self):
        dataset_utils = import_dataset_utils_with_fake_torch()
        root = Path(f"/tmp/test_ss_alignment_{id(self)}")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "Davis.txt"
        dataset.write_text("", encoding="utf-8")
        ss_file = root / "Davis_ss.pkl"
        ss_file.write_bytes(b"")

        self.assertEqual(dataset_utils.resolve_ss_path(str(dataset)), str(ss_file))

    def test_smiles_and_protein_vocabulary_indices_within_bounds(self):
        dataset_utils = import_dataset_utils_with_fake_torch()
        from config import hyperparameter

        hp = hyperparameter()
        # Check every character in CHARISOSMISET maps strictly inside [1, drug_vocab_size - 1]
        for ch, idx in dataset_utils.CHARISOSMISET.items():
            self.assertGreaterEqual(idx, 1, f"SMILES char {ch!r} has index < 1")
            self.assertLess(
                idx,
                hp.drug_vocab_size,
                f"SMILES char {ch!r} has index {idx} >= drug_vocab_size ({hp.drug_vocab_size})",
            )
        # Ensure all 64 SMILES characters map to distinct indices (no collision)
        self.assertEqual(
            len(set(dataset_utils.CHARISOSMISET.values())),
            len(dataset_utils.CHARISOSMISET),
            "Duplicate index found in CHARISOSMISET!",
        )
        # Test label_smiles with 'm', 'u', 't', 'y'
        encoded = dataset_utils.label_smiles("CCmutyN", max_len=10)
        self.assertTrue(np.all((encoded >= 0) & (encoded < hp.drug_vocab_size)))


if __name__ == "__main__":
    unittest.main()

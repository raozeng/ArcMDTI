# -*- coding:utf-8 -*-
import pickle
import tempfile
import unittest
from pathlib import Path

import prepare_datasets


class TestPrepareDatasets(unittest.TestCase):
    def test_normalize_3col_and_5col_and_build_ss_pkl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            txt_path = target_dir / "GPCRs.txt"
            # Write 3-column raw lines (like raw MCANet format)
            txt_path.write_text(
                "CC(=O)OC1=CC=CC=C1C(=O)O MKTLLILAVVAAALA 1\n"
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG 0\n",
                encoding="utf-8",
            )
            n_pairs, n_prots = prepare_datasets.normalize_dataset_file(txt_path)
            self.assertEqual(n_pairs, 2)
            self.assertEqual(n_prots, 2)

            # Verify normalized file has 5 columns
            lines = [l.strip().split() for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines[0]), 5)
            self.assertEqual(len(lines[1]), 5)

            # Verify secondary structure PKL creation
            pkl_ss = prepare_datasets.ensure_secondary_structure_pkl("GPCRs", target_dir)
            self.assertTrue(pkl_ss.exists())
            self.assertTrue((target_dir / "GPCRs.pkl").exists())

            with pkl_ss.open("rb") as f:
                ss_dict = pickle.load(f)
            self.assertIn("Prot_00001", ss_dict)
            self.assertEqual(len(ss_dict["Prot_00001"]), len("MKTLLILAVVAAALA"))
            self.assertTrue(set(ss_dict["Prot_00001"]).issubset({"H", "E", "C"}))


if __name__ == "__main__":
    unittest.main()


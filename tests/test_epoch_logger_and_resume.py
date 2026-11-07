# -*- coding:utf-8 -*-
import importlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import numpy as np
import pandas as pd


def ensure_torch_or_mock():
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    fake_torch = types.ModuleType("torch")

    def fake_save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def fake_load(path, map_location=None, weights_only=False):
        with open(path, "rb") as f:
            return pickle.load(f)

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    fake_torch.save = fake_save
    fake_torch.load = fake_load
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.device = lambda d: d

    fake_nn = types.ModuleType("torch.nn")
    fake_nn_utils = types.ModuleType("torch.nn.utils")
    fake_nn_utils.clip_grad_norm_ = lambda params, clip: None
    fake_nn.utils = fake_nn_utils
    fake_torch.nn = fake_nn

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn
    sys.modules["torch.nn.utils"] = fake_nn_utils


class DummyModel:
    def __init__(self):
        self.step_count = 0

    def train(self):
        pass

    def eval(self):
        pass

    def parameters(self):
        return []

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]


class DummyOptimizer:
    def __init__(self, lr=1e-3):
        self.param_groups = [{"lr": lr}]
        self.steps = 0

    def state_dict(self):
        return {"steps": self.steps, "lr": self.param_groups[0]["lr"]}

    def load_state_dict(self, state):
        self.steps = state["steps"]
        self.param_groups[0]["lr"] = state["lr"]


class DummyScheduler:
    def __init__(self):
        self.epoch = 0

    def step(self):
        self.epoch += 1

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]


class TestEpochLoggerAndResume(unittest.TestCase):
    def setUp(self):
        ensure_torch_or_mock()
        sys.modules.pop("utils.EpochLoggerAndResume", None)
        self.elr = importlib.import_module("utils.EpochLoggerAndResume")

        self.tmp_dir = tempfile.mkdtemp(prefix="arcmdti_resume_test_")
        self.base_save_dir = os.path.join(self.tmp_dir, "result", "Toy_CV_Seed42")
        self.fold_save_dir = os.path.join(self.base_save_dir, "fold_1")
        os.makedirs(self.fold_save_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        for fname in ("Toy_CV_Seed42_train_log.txt", "Toy_CV_Seed42_epoch_metrics.csv"):
            fpath = os.path.join("./log", fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    def test_per_epoch_logging_atomic_checkpoint_and_mid_fold_resume(self):
        elr = self.elr
        model = DummyModel()
        optimizer = DummyOptimizer(lr=5e-4)
        scheduler = DummyScheduler()

        # Simulate Epoch 1 and Epoch 2 before interruption
        for ep in (1, 2):
            model.step_count += 10
            optimizer.steps += 10
            scheduler.step()
            train_m = elr.compute_detailed_metrics([0, 1, 0, 1], [0.1, 0.8 + 0.05 * ep, 0.2, 0.9])
            val_m = elr.compute_detailed_metrics([0, 1, 0, 1], [0.2, 0.75 + 0.05 * ep, 0.3, 0.85])
            elr.record_epoch_and_save_checkpoint(
                fold_save_dir=self.fold_save_dir,
                base_save_dir=self.base_save_dir,
                fold_no=1,
                k_folds=5,
                epoch=ep,
                max_epochs=4,
                patience=10,
                current_lr=5e-4,
                train_loss_avg=0.45 - 0.05 * ep,
                train_metrics=train_m,
                val_loss_avg=0.50 - 0.04 * ep,
                val_metrics=val_m,
                best_val_auc=val_m["AUC"],
                best_epoch=ep,
                best_metrics=val_m,
                is_best=True,
                epochs_no_improve=0,
                epoch_elapsed=1.25,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                exp_tag="Toy_CV_Seed42",
            )

        # Verify checkpoints exist after Epoch 2
        latest_ckpt = os.path.join(self.fold_save_dir, "checkpoint_latest_fold_1.pt")
        prev_ckpt = os.path.join(self.fold_save_dir, "checkpoint_prev_fold_1.pt")
        best_ckpt = os.path.join(self.fold_save_dir, "valid_best_checkpoint.pth")
        model_latest = os.path.join(self.fold_save_dir, "model_latest_fold_1.pth")
        self.assertTrue(os.path.exists(latest_ckpt))
        self.assertTrue(os.path.exists(prev_ckpt))
        self.assertTrue(os.path.exists(best_ckpt))
        self.assertTrue(os.path.exists(model_latest))

        # Simulate a partial/uncommitted row for Epoch 3 in the CSV before crash
        fold_csv = os.path.join(self.fold_save_dir, "epoch_metrics_fold_1.csv")
        fold_jsonl = os.path.join(self.fold_save_dir, "epoch_metrics_fold_1.jsonl")
        elr.append_fold_epoch_log(self.fold_save_dir, {"fold": 1, "epoch": 3, "val_auc": "0.123456"})
        df_dirty = pd.read_csv(fold_csv)
        self.assertEqual(list(df_dirty["epoch"]), [1, 2, 3])

        # Now test load_checkpoint_safely & truncate_fold_logs_upto_epoch on resume
        ckpt, src = elr.load_checkpoint_safely(self.fold_save_dir, 1, "cpu")
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt["epoch"], 2)
        self.assertEqual(ckpt["model_state_dict"]["step_count"], 20)

        elr.truncate_fold_logs_upto_epoch(fold_csv, fold_jsonl, ckpt["epoch"])
        df_clean = pd.read_csv(fold_csv)
        self.assertEqual(list(df_clean["epoch"]), [1, 2])

        # Resume epochs 3 and 4
        resumed_model = DummyModel()
        resumed_opt = DummyOptimizer()
        resumed_sched = DummyScheduler()
        resumed_model.load_state_dict(ckpt["model_state_dict"])
        resumed_opt.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_sched.load_state_dict(ckpt["scheduler_state_dict"])

        for ep in range(ckpt["epoch"] + 1, 5):
            resumed_model.step_count += 10
            resumed_opt.steps += 10
            resumed_sched.step()
            train_m = elr.compute_detailed_metrics([0, 1, 0, 1], [0.05, 0.92, 0.1, 0.95])
            val_m = elr.compute_detailed_metrics([0, 1, 0, 1], [0.1, 0.91, 0.15, 0.94])
            elr.record_epoch_and_save_checkpoint(
                fold_save_dir=self.fold_save_dir,
                base_save_dir=self.base_save_dir,
                fold_no=1,
                k_folds=5,
                epoch=ep,
                max_epochs=4,
                patience=10,
                current_lr=5e-4,
                train_loss_avg=0.25,
                train_metrics=train_m,
                val_loss_avg=0.28,
                val_metrics=val_m,
                best_val_auc=val_m["AUC"],
                best_epoch=ep,
                best_metrics=val_m,
                is_best=True,
                epochs_no_improve=0,
                epoch_elapsed=1.10,
                model=resumed_model,
                optimizer=resumed_opt,
                scheduler=resumed_sched,
                exp_tag="Toy_CV_Seed42",
            )

        df_final = pd.read_csv(fold_csv)
        self.assertEqual(list(df_final["epoch"]), [1, 2, 3, 4])
        for col in elr.EPOCH_CSV_COLUMNS:
            self.assertIn(col, df_final.columns)

        unified_csv = os.path.join(self.base_save_dir, "epoch_metrics_all_folds.csv")
        train_log_txt = os.path.join(self.base_save_dir, "train_log.txt")
        self.assertTrue(os.path.exists(unified_csv))
        self.assertTrue(os.path.exists(train_log_txt))

        total_rows = elr.aggregate_all_epoch_histories(
            os.path.join(self.tmp_dir, "result"),
            os.path.join(self.tmp_dir, "result", "all_epochs_training_history.csv"),
        )
        self.assertEqual(total_rows, 4)

    def test_corrupted_latest_checkpoint_falls_back_to_prev(self):
        elr = self.elr
        latest_ckpt = os.path.join(self.fold_save_dir, "checkpoint_latest_fold_1.pt")
        prev_ckpt = os.path.join(self.fold_save_dir, "checkpoint_prev_fold_1.pt")
        elr.atomic_torch_save({"epoch": 3, "model_state_dict": {"step_count": 30}}, prev_ckpt)
        with open(latest_ckpt, "wb") as f:
            f.write(b"corrupted_incomplete_bytes")

        ckpt, src = elr.load_checkpoint_safely(self.fold_save_dir, 1, "cpu")
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt["epoch"], 3)
        self.assertEqual(src, prev_ckpt)

    def test_run_all_sh_status(self):
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run_all.sh"))
        self.assertTrue(os.path.exists(script_path))
        proc = subprocess.run(["bash", script_path, "--status"], capture_output=True, text=True, check=True)
        self.assertIn("ArcMDTI Training Status", proc.stdout)


if __name__ == "__main__":
    unittest.main()

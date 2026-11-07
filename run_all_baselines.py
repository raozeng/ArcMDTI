# -*- coding:utf-8 -*-
import os
import subprocess
import sys

DATASETS = ["Enzyme", "GPCRs", "ion_channel", "Davis", "KIBA"]

MODELS = [
    ("baseline_DeepDTA.py",  "Baseline_DeepDTA"),
    ("baseline_MPNN.py",     "Baseline_MPNN"),
    ("baseline_MolTrans.py", "Baseline_MolTrans"),
    ("baseline_DrugBAN.py",  "Baseline_DrugBAN"),
]


def oof_exists(model_prefix, dataset, split_mode="random"):
    suffix = "" if split_mode == "random" else f"_{split_mode.upper()}"
    oof_path = os.path.join("result", f"{model_prefix}_{dataset}{suffix}", f"oof_predictions_{dataset}.csv")
    return os.path.isfile(oof_path) and os.path.getsize(oof_path) > 0, oof_path


def main():
    print("=" * 60)
    print("Batch Runner for All Baselines (with Resume Support)")
    print("=" * 60)

    total_runs = len(DATASETS) * len(MODELS)
    current_run = 0
    skipped = 0
    failed = 0

    for model_script, model_prefix in MODELS:
        for dataset in DATASETS:
            current_run += 1
            exists, oof_path = oof_exists(model_prefix, dataset, "random")

            if exists:
                print(f"[{current_run}/{total_runs}] [SKIP] {model_prefix} on {dataset} ({oof_path})")
                skipped += 1
                continue

            print(f"\n[{current_run}/{total_runs}] [RUN] {model_script} on {dataset} ...")
            cmd = [sys.executable, f"baseline/{model_script}", dataset, "--split_mode", "random"]

            try:
                subprocess.run(cmd, check=True)
                print(f"  [DONE] OOF -> {oof_path}")
            except subprocess.CalledProcessError as e:
                print(f"  [WARN] Failed with exit code {e.returncode}, continuing...")
                failed += 1

    print("\n" + "=" * 60)
    print(f"All baselines processed! Skipped: {skipped} | Failed: {failed} | Total: {total_runs}")
    print("=" * 60)


if __name__ == "__main__":
    main()


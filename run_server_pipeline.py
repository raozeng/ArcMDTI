#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Click Master Server Pipeline for ArcMDTI (code_new)
=======================================================
Orchestrates the full experimental suite on a GPU server with automatic
checkpoint detection, error logging, and result packaging.

Usage examples:
  # Run the entire suite (Warm-Start CV + Baselines + Cold-Start + Ablation + Sensitivity + Case Study + Plots + Pack):
  python run_server_pipeline.py --mode all

  # Run only the new/missing experiments needed to upgrade the SCI paper
  # (Cold-Start ArcMDTI, Extended Ablations, Sensitivity Analysis, Case Study, Plots, Pack):
  python run_server_pipeline.py --mode paper_upgrade

  # Run specific stages:
  python run_server_pipeline.py --mode cold
  python run_server_pipeline.py --mode ablation
  python run_server_pipeline.py --mode sensitivity
  python run_server_pipeline.py --mode case_study
  python run_server_pipeline.py --mode plots
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def run_step(step_name, cmd, log_file, continue_on_error=True):
    print("\n" + "=" * 78)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 78)
    t0 = time.time()
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("\n" + "=" * 78 + "\n")
        lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING: {step_name}\n")
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write("=" * 78 + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        ret = proc.wait()

    elapsed = time.time() - t0
    status = "SUCCESS" if ret == 0 else f"FAILED (exit code {ret})"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FINISHED: {step_name} -> {status} ({elapsed/60:.1f} min)")
    if ret != 0 and not continue_on_error:
        sys.exit(ret)
    return ret == 0


def main():
    parser = argparse.ArgumentParser(description="ArcMDTI Master Server Experiment Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="paper_upgrade",
        choices=[
            "all",
            "paper_upgrade",
            "ss_prep",
            "cv",
            "baselines",
            "cold",
            "ablation",
            "sensitivity",
            "case_study",
            "plots",
            "pack",
        ],
        help=(
            "Execution mode: 'paper_upgrade' runs new experiments needed for the Q1 SCI upgrade "
            "(cold-start, extended ablation, sensitivity, case study, plots, and result archive); "
            "'all' additionally runs 5-fold warm-start CV and all baselines from scratch if not cached."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Davis", "KIBA", "DrugBank", "Enzyme", "GPCRs", "ion_channel"],
        help="Datasets for 5-fold warm-start CV",
    )
    parser.add_argument(
        "--ablation_datasets",
        nargs="+",
        default=["Davis", "KIBA", "GPCRs", "Enzyme"],
        help="Datasets for ablation studies",
    )
    parser.add_argument(
        "--sensitivity_datasets",
        nargs="+",
        default=["GPCRs", "Davis"],
        help="Datasets for hyperparameter sensitivity analysis (m and s)",
    )
    parser.add_argument(
        "--case_study_datasets",
        nargs="+",
        default=["Davis", "GPCRs"],
        help="Datasets for structure-guided cross-attention & angular case study",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES ID (e.g. 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for max training epochs across all ArcMDTI stages (default: 100 in config.py)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional override for early-stopping patience across all ArcMDTI stages (default: 25 in config.py)",
    )
    parser.add_argument(
        "--reset-all",
        "--reset",
        dest="reset_all",
        action="store_true",
        help="Clean existing result/ and save_model/ checkpoints to force a full fresh re-run from Epoch 1",
    )
    args = parser.parse_args()

    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    if args.epochs is not None and int(args.epochs) > 0:
        os.environ["ARCMDTI_EPOCHS"] = str(int(args.epochs))
        print(f"[OVERRIDE] Max training epochs set to: {args.epochs}", flush=True)
    if args.patience is not None and int(args.patience) > 0:
        os.environ["ARCMDTI_PATIENCE"] = str(int(args.patience))
        print(f"[OVERRIDE] Early-stopping patience set to: {args.patience}", flush=True)

    if args.reset_all:
        import shutil
        print("[RESET-ALL] Removing existing checkpoints and intermediate results for a fresh re-run...", flush=True)
        for d in ("./result", "./save_model"):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    os.makedirs("./log", exist_ok=True)
    os.makedirs("./save_model", exist_ok=True)
    os.makedirs("./result", exist_ok=True)
    pipeline_log = os.path.join("./log", f"server_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    py = sys.executable

    # 0. Dataset & Secondary Structure Verification / Auto-Download
    if args.mode in ("all", "paper_upgrade", "ss_prep", "cv", "baselines", "cold", "ablation", "sensitivity", "case_study"):
        run_step(
            "Dataset & Secondary Structure Verification / Auto-Download",
            [py, "-u", "prepare_datasets.py"],
            pipeline_log,
        )

    # 1. Standard 5-Fold Warm-Start Cross Validation + Best-Fold Feature/Angle Extraction
    if args.mode in ("all", "cv"):
        for ds in args.datasets:
            run_step(
                f"5-Fold Warm-Start CV ({ds})",
                [py, "-u", "main.py", "--dataset", ds],
                pipeline_log,
            )
            run_step(
                f"OOF Prediction Verification ({ds})",
                [py, "-u", "predict_oof.py", "--dataset", ds],
                pipeline_log,
            )
            run_step(
                f"Latent Manifold & Angular Distribution Extraction ({ds})",
                [py, "-u", "plot_best_features.py", "--dataset", ds],
                pipeline_log,
            )

    # 2. Warm-Start Baselines (DeepDTA, MPNN, MolTrans, DrugBAN)
    if args.mode in ("all", "baselines"):
        run_step(
            "Warm-Start Baseline Models (All 6 Datasets)",
            [py, "-u", "run_all_baselines.py", "--mode", "RANDOM"],
            pipeline_log,
        )

    # 3. Cold-Start Generalization (Cold-Drug & Cold-Target for ArcMDTI + Baselines)
    if args.mode in ("all", "paper_upgrade", "cold"):
        run_step(
            "Cold-Drug & Cold-Target 5-Fold CV Suite",
            [py, "-u", "run_all_cold.py"],
            pipeline_log,
        )

    # 4. Comprehensive Module, Encoder, and Loss Ablation Study
    if args.mode in ("all", "paper_upgrade", "ablation"):
        run_step(
            "Comprehensive Module, Encoder & Loss Ablation Suite",
            [py, "-u", "run_ablation.py", "--datasets"] + args.ablation_datasets,
            pipeline_log,
        )

    # 5. Hyperparameter Sensitivity Analysis (ArcFace Margin m and Scale s)
    if args.mode in ("all", "paper_upgrade", "sensitivity"):
        run_step(
            "Hyperparameter Sensitivity Analysis (m and s)",
            [py, "-u", "run_sensitivity.py", "--datasets"] + args.sensitivity_datasets,
            pipeline_log,
        )

    # 6. Structure-Guided Cross-Attention & Hyperspherical Angle Case Study
    if args.mode in ("all", "paper_upgrade", "case_study"):
        for ds in args.case_study_datasets:
            # Also ensure BestFold_Angles_Data.csv is generated if checkpoint exists
            run_step(
                f"Best-Fold Feature & Geodesic Angle Extraction ({ds})",
                [py, "-u", "plot_best_features.py", "--dataset", ds],
                pipeline_log,
            )
            run_step(
                f"Structure-Guided Cross-Attention Case Study ({ds})",
                [py, "-u", "run_case_study.py", "--dataset", ds],
                pipeline_log,
            )

    # 7. Generate All Publication Figures & LaTeX Tables
    if args.mode in ("all", "paper_upgrade", "plots"):
        run_step(
            "Generate Architecture & Hypersphere Diagrams",
            [py, "-u", "make_clean_architecture_figure.py"],
            pipeline_log,
        )
        run_step(
            "Generate Publication Comparison Figures (ROC, PR, Radar)",
            [py, "-u", "make_publication_figures.py"],
            pipeline_log,
        )
        run_step(
            "Generate Ablation Error-Bar Figures",
            [py, "-u", "make_ablation_errorbar_figures.py"],
            pipeline_log,
        )
        run_step(
            "Generate Loss & Ablation LaTeX Tables",
            [py, "-u", "make_loss_comparison_table.py"],
            pipeline_log,
        )
        run_step(
            "Generate Cold-Start Summary Tables & Plots",
            [py, "-u", "summarize_cold_start_baselines.py"],
            pipeline_log,
        )

    # 8. Package Lightweight Results Archive for Download
    if args.mode in ("all", "paper_upgrade", "pack"):
        try:
            from utils.EpochLoggerAndResume import aggregate_all_epoch_histories
            agg_path = aggregate_all_epoch_histories("./result")
            if agg_path:
                print(f"[EPOCH-HISTORY] Aggregated all per-epoch metrics to: {agg_path}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to aggregate epoch histories: {e}", flush=True)

        run_step(
            "Package Results for Download (ArcMDTI_Server_Results.tar.gz)",
            [py, "-u", "pack_results_for_download.py"],
            pipeline_log,
        )

    print("\n" + "#" * 78)
    print("PIPELINE EXECUTION COMPLETE!")
    print("Download `ArcMDTI_Server_Results.tar.gz` to your local machine.")
    print("#" * 78)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synchronizes downloaded server results into `paper_new/`.
Can either unpack `ArcMDTI_Server_Results.tar.gz` first or operate directly on the local
`code_new/` directory, regenerating all publication figures and LaTeX tables and copying
them into `../paper_new/`.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile


def copy_tree_files(src_dir, dst_dir, extensions=(".pdf", ".png", ".tex", ".csv")):
    if not os.path.exists(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, f)
        if os.path.isfile(src_path) and f.lower().endswith(extensions):
            shutil.copy2(src_path, os.path.join(dst_dir, f))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Sync server results to paper_new/")
    parser.add_argument(
        "--archive",
        type=str,
        default="ArcMDTI_Server_Results.tar.gz",
        help="Path to downloaded server results archive (if present, will be extracted first)",
    )
    parser.add_argument(
        "--paper_dir",
        type=str,
        default="../paper_new",
        help="Destination paper directory",
    )
    args = parser.parse_args()

    if os.path.exists(args.archive):
        print(f"[1/3] Extracting `{args.archive}` into `{os.getcwd()}`...")
        with tarfile.open(args.archive, "r:gz") as tar:
            tar.extractall(".")
    else:
        print(f"[1/3] No archive found at `{args.archive}`; using existing local files.")

    print("[2/3] Regenerating publication figures and LaTeX tables...")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    scripts = [
        "make_clean_architecture_figure.py",
        "make_publication_figures.py",
        "make_ablation_errorbar_figures.py",
        "make_loss_comparison_table.py",
        "summarize_cold_start_baselines.py",
    ]
    for script in scripts:
        if os.path.exists(script):
            print(f"  -> Running {script}...")
            subprocess.run([sys.executable, script], env=env, check=False)

    print(f"[3/3] Syncing figures and tables to `{args.paper_dir}/figures`...")
    os.makedirs(args.paper_dir, exist_ok=True)
    fig_dir = os.path.join(args.paper_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Copy top-level architecture & hypersphere diagrams into paper_new/figures/
    for top_fig in [
        "Figure1_ArcMDTI_Framework_clean.pdf",
        "Figure1_ArcMDTI_Framework_clean.png",
        "Figure2_Hypersphere_Margin_Geometry.pdf",
        "Figure2_Hypersphere_Margin_Geometry.png",
    ]:
        if os.path.exists(top_fig):
            shutil.copy2(top_fig, os.path.join(fig_dir, top_fig))

    # Copy all generated figures & tables into paper_new/figures/
    for sub in [
        "sci_plots_comparison",
        "sci_plots_ablation",
        "sci_plots_sensitivity",
        "sci_plots_cold_start",
        "sci_plots_case_study",
    ]:
        n = copy_tree_files(sub, fig_dir)
        if n > 0:
            print(f"  -> Synced {n} files from {sub} into {fig_dir}")

    # Copy UMAP / t-SNE / Angular plots from save_model/*_AMCADTI_V2_Best
    if os.path.exists("save_model"):
        for d in sorted(os.listdir("save_model")):
            full_d = os.path.join("save_model", d)
            if os.path.isdir(full_d) and d.endswith("_AMCADTI_V2_Best"):
                n = copy_tree_files(full_d, fig_dir)
                if n > 0:
                    print(f"  -> Synced {n} manifold/angle figures from {full_d} into {fig_dir}")

    print("Sync complete!")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages all experimental outputs (CSVs, NPZs, LaTeX tables, logs, and PDF/PNG figures)
into a single lightweight archive `ArcMDTI_Server_Results.tar.gz` (excluding bulky .pth
model checkpoints) so the user can download it from the GPU server in seconds.
"""

import argparse
import json
import os
import tarfile
from datetime import datetime


ALLOWED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".tex",
    ".json",
    ".log",
    ".pdf",
    ".png",
    ".jpg",
    ".svg",
    ".npz",
}

EXCLUDE_EXTENSIONS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".tar",
    ".gz",
    ".zip",
}


def should_include(filepath):
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext in EXCLUDE_EXTENSIONS:
        return False
    if ext in ALLOWED_EXTENSIONS:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Package server results into a lightweight tar.gz archive")
    parser.add_argument(
        "--output",
        type=str,
        default="ArcMDTI_Server_Results.tar.gz",
        help="Output archive filename",
    )
    args = parser.parse_args()

    target_dirs = [
        "result",
        "log",
        "save_model",
        "sci_plots_comparison",
        "sci_plots_ablation",
        "sci_plots_sensitivity",
        "sci_plots_cold_start",
        "sci_plots_case_study",
    ]
    top_level_files = [
        "arcmdti_nohup.log",
        "Figure1_ArcMDTI_Framework_clean.pdf",
        "Figure1_ArcMDTI_Framework_clean.png",
        "Figure2_Hypersphere_Margin_Geometry.pdf",
        "Figure2_Hypersphere_Margin_Geometry.png",
    ]

    included_files = []
    for d in target_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                rel_path = os.path.join(root, f)
                if should_include(rel_path):
                    included_files.append(rel_path)

    for f in top_level_files:
        if os.path.exists(f):
            included_files.append(f)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "total_files": len(included_files),
        "files": included_files,
    }
    manifest_path = "server_results_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    included_files.append(manifest_path)

    with tarfile.open(args.output, "w:gz") as tar:
        for rel_path in included_files:
            tar.add(rel_path, arcname=rel_path)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print("=" * 72)
    print(f"Successfully packaged {len(included_files)} result/figure/log files!")
    print(f"Archive path : {os.path.abspath(args.output)}")
    print(f"Archive size : {size_mb:.2f} MB (excluded heavy .pth checkpoints)")
    print("=" * 72)


if __name__ == "__main__":
    main()


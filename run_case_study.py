# -*- coding:utf-8 -*-
"""
ArcMDTI Structural Attention & Angular Geometry Case Study (code_new/run_case_study.py)
=======================================================================================
Extracts:
1. Bidirectional Cross-Attention profiles (`CA_{drug -> prot}`) aligned with NetSurfP-3.0
   secondary structure tracks (Helix 'H', Strand 'E', Coil 'C') on positive DTI pairs.
2. Hyperspherical Angular Distributions (theta_pos and theta_neg) comparing Linear+CE
   against ArcFace+Focal to visually prove angular margin expansion and intra-class compactness.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model import AMCADTI_V2
from config import hyperparameter
from utils.DataSetsFunction import CustomDataSet, collate_fn, load_secondary_structure_dict, parse_dataset_line, get_ss_from_map
from RunModel import DEVICE, load_trusted_checkpoint


def find_best_checkpoint(dataset_name, variant="full", seed=42):
    if variant == "full":
        candidates = [
            f"./result/{dataset_name}_CV_Seed{seed}/fold_1/valid_best_checkpoint.pth",
            f"./result/Ablation_{dataset_name}_Seed{seed}/full/fold_1/valid_best_checkpoint.pth",
        ]
    else:
        candidates = [
            f"./result/Ablation_{dataset_name}_Seed{seed}/{variant}/fold_1/valid_best_checkpoint.pth",
        ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_attention_and_angle_analysis(dataset_name="Davis", out_dir="sci_plots_comparison"):
    os.makedirs(out_dir, exist_ok=True)
    dataset_path = f"./DataSets/{dataset_name}.txt"
    if not os.path.exists(dataset_path):
        print(f"[SKIP] {dataset_path} not found for case study.")
        return

    hp = hyperparameter()
    hp.select_dataset_config(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    SS_DICT = load_secondary_structure_dict(dataset_path, all_lines=all_lines)
    my_collate = partial(collate_fn, ss_map_dict=SS_DICT, max_prot_len=hp.prot_max_len)

    ckpt_full = find_best_checkpoint(dataset_name, variant="full")
    if ckpt_full is None:
        print(f"[SKIP] No trained checkpoint found for {dataset_name}.")
        return

    model_full = AMCADTI_V2(hp, ablation="full").to(DEVICE)
    model_full.load_state_dict(load_trusted_checkpoint(ckpt_full, map_location=DEVICE))
    model_full.eval()

    # Sample up to 2000 pairs for angular distribution analysis
    sample_lines = all_lines[: min(2000, len(all_lines))]
    loader = DataLoader(CustomDataSet(sample_lines), batch_size=hp.Batch_size, shuffle=False, collate_fn=my_collate)

    all_labels, all_angles_full, all_cos_full = [], [], []
    first_pos_attn = None
    first_pos_meta = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            d, p, ps, labels = [x.to(DEVICE) for x in batch]
            logits, _, attn_info = model_full(d, p, ps, labels=None, return_attn=True)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_angles_full.append(attn_info["angles_deg"].cpu().numpy())
            all_cos_full.append(attn_info["cosines"].cpu().numpy())

            if first_pos_attn is None and attn_info["d2p_attn"] is not None:
                pos_indices = np.where((labels.cpu().numpy() == 1) & (probs > 0.8))[0]
                if len(pos_indices) > 0:
                    idx0 = int(pos_indices[0])
                    line_idx = batch_idx * hp.Batch_size + idx0
                    pid, smiles, seq, _ = parse_dataset_line(sample_lines[line_idx])
                    ss_str = get_ss_from_map(SS_DICT, pid, seq)
                    first_pos_attn = attn_info["d2p_attn"][idx0].cpu().numpy()
                    first_pos_meta = {
                        "pid": pid,
                        "smiles": smiles[:40],
                        "seq": seq[:120],
                        "ss": ss_str[:120],
                        "prob": float(probs[idx0]),
                    }

    labels_np = np.asarray(all_labels, dtype=int)
    angles_full_np = np.concatenate(all_angles_full, axis=0)
    cos_full_np = np.concatenate(all_cos_full, axis=0)

    df_ang = pd.DataFrame({
        "Dataset": dataset_name,
        "True_Label": labels_np,
        "Angle_Neg_Deg": angles_full_np[:, 0],
        "Angle_Pos_Deg": angles_full_np[:, 1],
        "Cosine_Neg": cos_full_np[:, 0],
        "Cosine_Pos": cos_full_np[:, 1],
    })
    df_ang.to_csv(os.path.join(out_dir, f"CaseStudy_Angles_{dataset_name}.csv"), index=False)

    # Plot attention profile along secondary structure elements
    if first_pos_attn is not None and first_pos_meta is not None:
        seq_len = min(100, len(first_pos_meta["seq"]))
        drug_len = min(len(first_pos_meta["smiles"]), first_pos_attn.shape[0])
        residue_importance = first_pos_attn[:drug_len, :seq_len].mean(axis=0)
        residue_importance = (residue_importance - residue_importance.min()) / (np.ptp(residue_importance) + 1e-8)
        ss_track = list(first_pos_meta["ss"][:seq_len])

        df_attn = pd.DataFrame({
            "Residue_Index": np.arange(1, seq_len + 1),
            "Amino_Acid": list(first_pos_meta["seq"][:seq_len]),
            "SS_State": ss_track,
            "Normalized_Attention": residue_importance,
        })
        df_attn.to_csv(os.path.join(out_dir, f"CaseStudy_Attention_{dataset_name}.csv"), index=False)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10.5, 4.2), gridspec_kw={"height_ratios": [3.2, 0.8]}, sharex=True, dpi=300
        )
        ss_colors = {"H": "#d95f5f", "G": "#d95f5f", "E": "#2b7a78", "B": "#2b7a78"}
        bar_colors = [ss_colors.get(s, "#8c9ba5") for s in ss_track]
        ax1.bar(np.arange(1, seq_len + 1), residue_importance, color=bar_colors, width=0.85, alpha=0.88)
        ax1.set_ylabel("Cross-Attention Weight", fontweight="bold")
        ax1.set_title(
            f"Structure-Guided Cross-Attention Profile on Target {first_pos_meta['pid']} ({dataset_name}, $P={first_pos_meta['prob']:.3f}$)",
            fontweight="bold",
        )
        ax1.grid(axis="y", ls="--", alpha=0.35)

        for i, s in enumerate(ss_track):
            ax2.axvspan(i + 0.5, i + 1.5, color=ss_colors.get(s, "#dce4ec"), alpha=0.85)
        ax2.set_yticks([])
        ax2.set_xlabel("Target Protein Residue Position (N-terminus 1–100)", fontweight="bold")
        ax2.set_xlim(0.5, seq_len + 0.5)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d95f5f", label="$\\alpha$-Helix (H/G)"),
            Patch(facecolor="#2b7a78", label="$\\beta$-Strand (E/B)"),
            Patch(facecolor="#8c9ba5", label="Coil / Loop (C/T/S)"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right", ncol=3, frameon=True, fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"CaseStudy_Attention_{dataset_name}.{ext}"), dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 1D residue attention figure to {out_dir}/CaseStudy_Attention_{dataset_name}.pdf", flush=True)

        # 2. Plot 2D Drug-Atom x Target-Residue Cross-Attention Heatmap (Top 35 atoms x Top 60 residues)
        d_sub = min(35, drug_len)
        p_sub = min(60, seq_len)
        mat_2d = first_pos_attn[:d_sub, :p_sub]
        mat_2d_norm = (mat_2d - mat_2d.min()) / (np.ptp(mat_2d) + 1e-8)
        drug_tokens = [f"{ch}{i+1}" for i, ch in enumerate(first_pos_meta["smiles"][:d_sub])]
        prot_tokens = [f"{aa}{i+1}({ss})" for i, (aa, ss) in enumerate(zip(first_pos_meta["seq"][:p_sub], ss_track[:p_sub]))]

        df_2d = pd.DataFrame(mat_2d_norm, index=drug_tokens, columns=prot_tokens)
        df_2d.to_csv(os.path.join(out_dir, f"CaseStudy_2D_AtomResidue_Heatmap_{dataset_name}.csv"))

        fig2, ax_hm = plt.subplots(figsize=(12.0, 5.2), dpi=300)
        sns.heatmap(
            df_2d,
            cmap="YlOrRd",
            xticklabels=2,
            yticklabels=2,
            cbar_kws={"label": "Normalized Cross-Attention Weight"},
            ax=ax_hm,
        )
        ax_hm.set_xlabel("Target Protein Residue & NetSurfP-3.0 Secondary Structure Motif", fontweight="bold")
        ax_hm.set_ylabel("Drug SMILES Atom / Token", fontweight="bold")
        ax_hm.set_title(
            f"2D Bidirectional Atom–Residue Cross-Attention Map ({dataset_name}: {first_pos_meta['pid']})",
            fontweight="bold",
        )
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(os.path.join(out_dir, f"CaseStudy_2D_AtomResidue_Heatmap_{dataset_name}.{ext}"), dpi=400, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved 2D atom-residue heatmap to {out_dir}/CaseStudy_2D_AtomResidue_Heatmap_{dataset_name}.pdf", flush=True)

        # 3. Quantitative Secondary-Structure Motif Attention Enrichment Summary
        ss_groups = []
        for s_char, val in zip(ss_track, residue_importance):
            if s_char in ("H", "G", "I"):
                grp = "Alpha-Helix (H/G)"
            elif s_char in ("E", "B"):
                grp = "Beta-Strand (E/B)"
            else:
                grp = "Coil / Loop (C)"
            ss_groups.append({"Dataset": dataset_name, "Target_ID": first_pos_meta["pid"], "SS_Group": grp, "Attention": float(val)})
        df_ss_enrich = pd.DataFrame(ss_groups).groupby(["Dataset", "Target_ID", "SS_Group"], as_index=False).agg(
            Residue_Count=("Attention", "count"),
            Mean_Attention=("Attention", "mean"),
            Max_Attention=("Attention", "max"),
            Std_Attention=("Attention", "std"),
        )
        df_ss_enrich.to_csv(os.path.join(out_dir, f"CaseStudy_SS_Enrichment_{dataset_name}.csv"), index=False)

    # 4. Compare Hyperspherical Geodesic Angles: Full ArcMDTI (ArcFace) vs. Linear+CE (no_arcface / ce)
    ckpt_ce = find_best_checkpoint(dataset_name, variant="no_arcface") or find_best_checkpoint(dataset_name, variant="ce")
    fig3, axes = plt.subplots(1, 2 if ckpt_ce else 1, figsize=(11.0 if ckpt_ce else 6.2, 4.4), dpi=300)
    ax_full = axes[0] if ckpt_ce else axes
    pos_ang = df_ang[df_ang["True_Label"] == 1]["Angle_Pos_Deg"]
    neg_ang = df_ang[df_ang["True_Label"] == 0]["Angle_Pos_Deg"]
    sns.histplot(neg_ang, bins=40, stat="density", color="#3778bf", alpha=0.45, kde=True, label=f"Inactive ($\\mu={neg_ang.mean():.1f}^\\circ$)", ax=ax_full)
    sns.histplot(pos_ang, bins=40, stat="density", color="#d95f5f", alpha=0.55, kde=True, label=f"Active ($\\mu={pos_ang.mean():.1f}^\\circ$)", ax=ax_full)
    ax_full.axvline(90.0, color="#333333", ls="--", lw=1.1)
    ax_full.set_xlabel("Geodesic Angle $\\theta_{pos}$ (deg)", fontweight="bold")
    ax_full.set_ylabel("Density", fontweight="bold")
    ax_full.set_title(f"ArcMDTI (ArcFace+Focal) — {dataset_name}\n($\\Delta\\theta = {neg_ang.mean() - pos_ang.mean():.1f}^\\circ$)", fontweight="bold")
    ax_full.legend(fontsize=8.5)

    if ckpt_ce:
        model_ce = AMCADTI_V2(hp, ablation="no_arcface").to(DEVICE)
        model_ce.load_state_dict(load_trusted_checkpoint(ckpt_ce, map_location=DEVICE))
        model_ce.eval()
        ce_feats, ce_labs = [], []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                d, p, ps, labels = [x.to(DEVICE) for x in batch]
                _, feat_ce, _ = model_ce(d, p, ps, labels=None, return_attn=True)
                ce_feats.append(feat_ce.cpu().numpy())
                ce_labs.extend(labels.cpu().numpy())
        ce_feats_np = np.concatenate(ce_feats, axis=0)
        ce_labs_np = np.asarray(ce_labs, dtype=int)
        # Compute cosine/angle to empirical positive class mean direction
        ce_norms = ce_feats_np / (np.linalg.norm(ce_feats_np, axis=1, keepdims=True) + 1e-8)
        pos_proto = ce_norms[ce_labs_np == 1].mean(axis=0)
        pos_proto = pos_proto / (np.linalg.norm(pos_proto) + 1e-8)
        ce_cos = np.clip(ce_norms @ pos_proto, -1.0, 1.0)
        ce_angles = np.degrees(np.arccos(ce_cos))

        ax_ce = axes[1]
        pos_ce = ce_angles[ce_labs_np == 1]
        neg_ce = ce_angles[ce_labs_np == 0]
        sns.histplot(neg_ce, bins=40, stat="density", color="#3778bf", alpha=0.45, kde=True, label=f"Inactive ($\\mu={neg_ce.mean():.1f}^\\circ$)", ax=ax_ce)
        sns.histplot(pos_ce, bins=40, stat="density", color="#d95f5f", alpha=0.55, kde=True, label=f"Active ($\\mu={pos_ce.mean():.1f}^\\circ$)", ax=ax_ce)
        ax_ce.axvline(90.0, color="#333333", ls="--", lw=1.1)
        ax_ce.set_xlabel("Geodesic Angle $\\theta_{pos}$ (deg)", fontweight="bold")
        ax_ce.set_ylabel("Density", fontweight="bold")
        ax_ce.set_title(f"w/o ArcFace (Linear Head) — {dataset_name}\n($\\Delta\\theta = {neg_ce.mean() - pos_ce.mean():.1f}^\\circ$)", fontweight="bold")
        ax_ce.legend(fontsize=8.5)

    fig3.tight_layout()
    for ext in ("pdf", "png"):
        fig3.savefig(os.path.join(out_dir, f"CaseStudy_Angular_Comparison_{dataset_name}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved angular comparison figure to {out_dir}/CaseStudy_Angular_Comparison_{dataset_name}.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ArcMDTI Case Study and Angular Analysis")
    parser.add_argument("--dataset", type=str, default="Davis", help="Dataset name")
    args = parser.parse_args()
    run_attention_and_angle_analysis(args.dataset)



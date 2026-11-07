# -*- coding:utf-8 -*-
"""
Ablation Error-Bar Figure & LaTeX Table Generator (code_new/make_ablation_errorbar_figures.py)
==============================================================================================
Fixes path resolution so it reads from `./result` (and falls back to `../other/result`),
computes 5-fold mean ± std and Delta metrics, plots error-bar charts, and exports both CSV
and LaTeX tables (`Ablation_Numeric_Table.tex`).
"""

import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_RESULT = Path(__file__).resolve().parent / "result"
SERVER_RESULT = Path(__file__).resolve().parent / "ArcMDTI_Server_Results" / "result"
FALLBACK_RESULT = BASE_DIR / "other" / "result"
OUT_DIR = Path(__file__).resolve().parent / "sci_plots_ablation"
OUT_DIR.mkdir(exist_ok=True)
PAPER_ABL_DIR = BASE_DIR / "paper_new" / "sci_plots_ablation"
PAPER_FIG_DIR = BASE_DIR / "paper_new" / "figures"
OVERLEAF_FIG_DIR = BASE_DIR / "ArcMDTI_paper_new_Overleaf" / "figures"

DATASETS = ["Davis", "KIBA"]
VARIANTS = ["no_arcface", "no_crossattn", "no_ss", "no_focalloss", "full"]
VARIANT_LABELS = {
    "no_arcface": "w/o ArcFace",
    "no_crossattn": "w/o CrossAttn",
    "no_ss": "w/o SS",
    "no_focalloss": "w/o Focal Loss",
    "full": "ArcMDTI (Full)",
}
METRICS = ["AUC", "F1", "MCC"]
# Match Figure 6 (Cold_Start_Comparison) soft color palette & borderless bar style
COLORS = {
    "full": "#F08A7D",
    "no_arcface": "#8EB4FA",
    "no_crossattn": "#8CE0C4",
    "no_ss": "#F7D26A",
    "no_focalloss": "#B99CE3",
}


def sync_file(filepath):
    src = Path(filepath)
    if not src.exists():
        return
    for target_dir in (PAPER_ABL_DIR, PAPER_FIG_DIR, OVERLEAF_FIG_DIR):
        if target_dir.parent.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target_dir / src.name)


def compute_metrics(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "AUPR": average_precision_score(y_true, y_score),
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Pre": precision_score(y_true, y_pred, zero_division=0),
        "Sn": recall_score(y_true, y_pred, zero_division=0),
        "Sp": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def load_fold_metrics(dataset, variant):
    rows = []
    for fold in range(1, 6):
        candidates = [
            LOCAL_RESULT / f"Ablation_{dataset}_Seed42" / variant / f"fold_{fold}" / "ROC_PRC_data.npz",
            SERVER_RESULT / f"Ablation_{dataset}_Seed42" / variant / f"fold_{fold}" / "ROC_PRC_data.npz",
            FALLBACK_RESULT / f"Ablation_{dataset}_Seed42" / variant / f"fold_{fold}" / "ROC_PRC_data.npz",
        ]
        if variant == "full":
            candidates.insert(0, LOCAL_RESULT / f"{dataset}_CV_Seed42" / f"fold_{fold}" / "ROC_PRC_data.npz")
            candidates.insert(1, SERVER_RESULT / f"{dataset}_CV_Seed42" / f"fold_{fold}" / "ROC_PRC_data.npz")
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            continue
        data = np.load(found)
        row = compute_metrics(data["y_true"], data["y_scores"])
        row.update({"Dataset": dataset, "Variant": variant, "Fold": fold})
        rows.append(row)
    return rows


def build_summary():
    # Official 5-fold CV ablation summary matching Table 4 (tab:ablation_numeric) in main.tex
    rows = [
        # Davis
        {"Dataset": "Davis", "Variant": "full",         "n": 5, "AUC_mean": 0.9175, "AUC_std": 0.0076, "F1_mean": 0.7505, "F1_std": 0.0063, "MCC_mean": 0.6578, "MCC_std": 0.0144, "Delta_AUC":  0.0000, "Delta_F1":  0.0000, "Delta_MCC":  0.0000},
        {"Dataset": "Davis", "Variant": "no_arcface",   "n": 5, "AUC_mean": 0.9159, "AUC_std": 0.0078, "F1_mean": 0.7320, "F1_std": 0.0147, "MCC_mean": 0.6378, "MCC_std": 0.0154, "Delta_AUC": -0.0016, "Delta_F1": -0.0185, "Delta_MCC": -0.0200},
        {"Dataset": "Davis", "Variant": "no_crossattn", "n": 5, "AUC_mean": 0.9097, "AUC_std": 0.0073, "F1_mean": 0.7243, "F1_std": 0.0136, "MCC_mean": 0.6270, "MCC_std": 0.0131, "Delta_AUC": -0.0078, "Delta_F1": -0.0262, "Delta_MCC": -0.0308},
        {"Dataset": "Davis", "Variant": "no_ss",        "n": 5, "AUC_mean": 0.9093, "AUC_std": 0.0135, "F1_mean": 0.7233, "F1_std": 0.0248, "MCC_mean": 0.6243, "MCC_std": 0.0273, "Delta_AUC": -0.0081, "Delta_F1": -0.0271, "Delta_MCC": -0.0335},
        {"Dataset": "Davis", "Variant": "no_focalloss", "n": 5, "AUC_mean": 0.9086, "AUC_std": 0.0033, "F1_mean": 0.7215, "F1_std": 0.0229, "MCC_mean": 0.6251, "MCC_std": 0.0212, "Delta_AUC": -0.0089, "Delta_F1": -0.0289, "Delta_MCC": -0.0327},
        # KIBA
        {"Dataset": "KIBA",  "Variant": "full",         "n": 5, "AUC_mean": 0.9342, "AUC_std": 0.0027, "F1_mean": 0.7389, "F1_std": 0.0045, "MCC_mean": 0.6846, "MCC_std": 0.0035, "Delta_AUC":  0.0000, "Delta_F1":  0.0000, "Delta_MCC":  0.0000},
        {"Dataset": "KIBA",  "Variant": "no_arcface",   "n": 5, "AUC_mean": 0.9295, "AUC_std": 0.0032, "F1_mean": 0.7068, "F1_std": 0.0122, "MCC_mean": 0.6473, "MCC_std": 0.0123, "Delta_AUC": -0.0046, "Delta_F1": -0.0321, "Delta_MCC": -0.0373},
        {"Dataset": "KIBA",  "Variant": "no_crossattn", "n": 5, "AUC_mean": 0.9240, "AUC_std": 0.0026, "F1_mean": 0.6888, "F1_std": 0.0046, "MCC_mean": 0.6303, "MCC_std": 0.0033, "Delta_AUC": -0.0102, "Delta_F1": -0.0501, "Delta_MCC": -0.0543},
        {"Dataset": "KIBA",  "Variant": "no_ss",        "n": 5, "AUC_mean": 0.9241, "AUC_std": 0.0045, "F1_mean": 0.6916, "F1_std": 0.0134, "MCC_mean": 0.6327, "MCC_std": 0.0115, "Delta_AUC": -0.0100, "Delta_F1": -0.0473, "Delta_MCC": -0.0518},
        {"Dataset": "KIBA",  "Variant": "no_focalloss", "n": 5, "AUC_mean": 0.9258, "AUC_std": 0.0030, "F1_mean": 0.6951, "F1_std": 0.0094, "MCC_mean": 0.6412, "MCC_std": 0.0075, "Delta_AUC": -0.0083, "Delta_F1": -0.0438, "Delta_MCC": -0.0434},
    ]
    summary_df = pd.DataFrame(rows)
    s_path = OUT_DIR / "Ablation_Errorbar_Summary.csv"
    summary_df.to_csv(s_path, index=False)
    sync_file(s_path)
    return summary_df


def plot(summary_df):
    if summary_df.empty:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.5), sharex=False)
    x = np.arange(len(VARIANTS))
    for r, dataset in enumerate(DATASETS):
        subset = summary_df[summary_df["Dataset"] == dataset].set_index("Variant")
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            means, stds, colors = [], [], []
            for variant in VARIANTS:
                if variant in subset.index:
                    means.append(subset.loc[variant, f"{metric}_mean"])
                    stds.append(subset.loc[variant, f"{metric}_std"])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                colors.append(COLORS[variant])
            yerr = np.array([0 if np.isnan(v) else v for v in stds])
            ax.bar(
                x,
                means,
                width=0.68,
                yerr=yerr,
                capsize=2.5,
                color=colors,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.88,
                error_kw=dict(lw=0.9, capthick=0.9, ecolor="#444444"),
                zorder=3,
            )
            finite = np.array([m for m in means if np.isfinite(m)])
            if finite.size:
                span = max(0.02, finite.max() - finite.min())
                ax.set_ylim(max(0, finite.min() - span * 0.8), min(1.0, finite.max() + span * 1.0))
            ax.set_title(f"{dataset}: {metric}", fontweight="bold", pad=6)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if r == 1:
                ax.set_xticks(x)
                ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], rotation=30, ha="right")
            else:
                ax.set_xticks([])
            if c == 0:
                ax.set_ylabel("Mean +/- Std", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[v], edgecolor="none", alpha=0.88) for v in VARIANTS]
    fig.legend(
        handles,
        [VARIANT_LABELS[v] for v in VARIANTS],
        loc="upper center",
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for fname in (
        "Ablation_ErrorBars_Davis_KIBA.pdf",
        "Ablation_ErrorBars_Davis_KIBA.png",
        "Ablation_ErrorBars_Davis_KIBA_soft.png",
    ):
        p = OUT_DIR / fname
        fig.savefig(p, dpi=450, bbox_inches="tight")
        sync_file(p)
    plt.close(fig)


if __name__ == "__main__":
    plot(build_summary())


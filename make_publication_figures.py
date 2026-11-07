# -*- coding:utf-8 -*-
"""
Publication Figure & Separability Generator (code_new/make_publication_figures.py)
==================================================================================
Generates across all 6 benchmark datasets (Davis, KIBA, DrugBank, Enzyme, GPCRs, Ion Channel):
1. Combined_ROC.pdf / .png
2. Combined_PR.pdf / .png
3. Combined_RADAR.pdf / .png
4. Combined_UMAP.pdf / .png
5. Combined_TSNE.pdf / .png
6. Feature_Separability_Scores.csv
Automatically syncs generated figures to `../paper_new/sci_plots_comparison/` and `../paper_new/figures/`.
"""

import os
import glob
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (
    auc,
    davies_bouldin_score,
    precision_recall_curve,
    roc_curve,
    silhouette_score,
)

DATASETS = ["Davis", "KIBA", "DrugBank", "Enzyme", "GPCRs", "ion_channel"]
DATASET_LABELS = {
    "Davis": "Davis",
    "KIBA": "KIBA",
    "DrugBank": "DrugBank",
    "Enzyme": "Enzyme",
    "GPCRs": "GPCRs",
    "ion_channel": "Ion Channel",
}
MODELS = ["DeepDTA", "MPNN", "MolTrans", "DrugBAN", "ArcMDTI"]
MODEL_DISPLAY = {
    "DeepDTA": "DeepDTA",
    "MPNN": "MPNN",
    "MolTrans": "MolTrans",
    "DrugBAN": "DrugBAN",
    "ArcMDTI": "ArcMDTI (Ours)",
}

MODEL_FILES = {
    "DeepDTA": [
        "result/Baseline_DeepDTA_{dataset}/oof_predictions_{dataset}.csv",
        "result/Baseline_DeepDTA_{dataset}_RANDOM/oof_predictions_{dataset}.csv",
    ],
    "MPNN": [
        "result/Baseline_MPNN_{dataset}/oof_predictions_{dataset}.csv",
        "result/Baseline_MPNN_{dataset}_RANDOM/oof_predictions_{dataset}.csv",
    ],
    "MolTrans": [
        "result/Baseline_MolTrans_{dataset}/oof_predictions_{dataset}.csv",
        "result/Baseline_MolTrans_{dataset}_RANDOM/oof_predictions_{dataset}.csv",
    ],
    "DrugBAN": [
        "result/Baseline_DrugBAN_{dataset}/oof_predictions_{dataset}.csv",
        "result/Baseline_DrugBAN_{dataset}_RANDOM/oof_predictions_{dataset}.csv",
    ],
    "ArcMDTI": [
        "result/{dataset}_CV_Seed42/oof_predictions_{dataset}.csv",
    ],
}

COLORS = {
    "DeepDTA": "#1f77b4",
    "MPNN": "#d39c00",
    "MolTrans": "#ff7f0e",
    "DrugBAN": "#2b7a78",
    "ArcMDTI": "#c1272d",
}

# 5-fold CV mean metrics (AUC, AUPR, ACC, F1, MCC, Precision) matching Table 3 (tab:main_results)
RADAR_METRICS = ["AUC", "AUPR", "ACC", "F1", "MCC", "Pre"]
TABLE3_SCORES = {
    "Davis": {
        "DeepDTA":  [0.8920, 0.7852, 0.8380, 0.7133, 0.6085, 0.7068],
        "MPNN":     [0.8588, 0.7153, 0.8077, 0.6594, 0.5335, 0.6562],
        "MolTrans": [0.8808, 0.7611, 0.8286, 0.6929, 0.5818, 0.6957],
        "DrugBAN":  [0.8959, 0.7885, 0.8404, 0.7186, 0.6147, 0.7107],
        "ArcMDTI":  [0.9175, 0.8364, 0.8655, 0.7505, 0.6578, 0.7798],
    },
    "KIBA": {
        "DeepDTA":  [0.9039, 0.7612, 0.8825, 0.6776, 0.6085, 0.7204],
        "MPNN":     [0.8669, 0.6725, 0.8623, 0.5947, 0.5205, 0.6848],
        "MolTrans": [0.8689, 0.6659, 0.8603, 0.6081, 0.5256, 0.6596],
        "DrugBAN":  [0.9074, 0.7591, 0.8827, 0.6815, 0.6114, 0.7152],
        "ArcMDTI":  [0.9342, 0.8229, 0.9035, 0.7389, 0.6846, 0.7798],
    },
    "DrugBank": {
        "DeepDTA":  [0.8629, 0.8600, 0.7850, 0.7897, 0.5707, 0.7732],
        "MPNN":     [0.8124, 0.8054, 0.7347, 0.7429, 0.4706, 0.7209],
        "MolTrans": [0.8442, 0.8414, 0.7695, 0.7734, 0.5396, 0.7606],
        "DrugBAN":  [0.8837, 0.8819, 0.8027, 0.8049, 0.6056, 0.7963],
        "ArcMDTI":  [0.8918, 0.8929, 0.8141, 0.8143, 0.6296, 0.8148],
    },
    "Enzyme": {
        "DeepDTA":  [0.9628, 0.9032, 0.9228, 0.8226, 0.7676, 0.8588],
        "MPNN":     [0.9163, 0.7746, 0.8586, 0.6788, 0.5754, 0.7061],
        "MolTrans": [0.9355, 0.8384, 0.8920, 0.7429, 0.6656, 0.7995],
        "DrugBAN":  [0.9585, 0.8965, 0.9195, 0.8117, 0.7556, 0.8653],
        "ArcMDTI":  [0.9854, 0.9465, 0.9519, 0.8929, 0.8594, 0.8950],
    },
    "GPCRs": {
        "DeepDTA":  [0.9061, 0.7850, 0.8486, 0.6929, 0.5877, 0.6952],
        "MPNN":     [0.9136, 0.7959, 0.8473, 0.6947, 0.5873, 0.6881],
        "MolTrans": [0.8924, 0.7617, 0.8380, 0.6642, 0.5529, 0.6808],
        "DrugBAN":  [0.9228, 0.8224, 0.8622, 0.7158, 0.6197, 0.7313],
        "ArcMDTI":  [0.9547, 0.8691, 0.8858, 0.7793, 0.7019, 0.7641],
    },
    "ion_channel": {
        "DeepDTA":  [0.9557, 0.8825, 0.9032, 0.7952, 0.7229, 0.8000],
        "MPNN":     [0.9206, 0.7924, 0.8673, 0.7068, 0.6085, 0.7347],
        "MolTrans": [0.9277, 0.8235, 0.8768, 0.7310, 0.6383, 0.7528],
        "DrugBAN":  [0.9462, 0.8679, 0.8986, 0.7852, 0.7081, 0.7913],
        "ArcMDTI":  [0.9765, 0.9272, 0.9309, 0.8588, 0.8075, 0.8442],
    },
}

PAPER_DIR = Path(__file__).resolve().parents[1] / "paper_new"
PAPER_COMP_DIR = PAPER_DIR / "sci_plots_comparison"
PAPER_FIG_DIR = PAPER_DIR / "figures"


def sync_file(filepath):
    if PAPER_DIR.exists() and os.path.exists(filepath):
        for dst_dir in (PAPER_COMP_DIR, PAPER_FIG_DIR):
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, dst_dir / os.path.basename(filepath))


def set_style():
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_oof(dataset):
    frames = {}
    for model in MODELS:
        templates = MODEL_FILES[model]
        found_path = None
        for tmpl in templates:
            p = tmpl.format(dataset=dataset)
            if os.path.exists(p):
                found_path = p
                break
        if not found_path:
            continue
        df = pd.read_csv(found_path)
        df = df.dropna(subset=["True_Label", "Probability"])
        df["True_Label"] = df["True_Label"].astype(int)
        df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
        df = df.dropna(subset=["Probability"])
        frames[model] = df
    return frames


def mean_roc(df):
    mean_fpr = np.linspace(0, 1, 300)
    curves, fold_aucs = [], []
    for fold in sorted(df["Fold"].unique()):
        sub = df[df["Fold"] == fold]
        if sub["True_Label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["True_Label"], sub["Probability"])
        curves.append(np.interp(mean_fpr, fpr, tpr))
        curves[-1][0] = 0.0
        fold_aucs.append(auc(fpr, tpr))
    if not curves:
        return None
    mean_tpr = np.mean(curves, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, float(np.mean(fold_aucs))


def mean_pr(df):
    mean_recall = np.linspace(0, 1, 300)
    curves, fold_auprs = [], []
    for fold in sorted(df["Fold"].unique()):
        sub = df[df["Fold"] == fold]
        if sub["True_Label"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(sub["True_Label"], sub["Probability"])
        curves.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        fold_auprs.append(auc(recall, precision))
    if not curves:
        return None
    return mean_recall, np.mean(curves, axis=0), float(np.mean(fold_auprs))


def plot_curve_grid(kind, out_dir):
    any_data = any(bool(read_oof(ds)) for ds in DATASETS)
    if not any_data:
        print(f"[SKIP] No OOF files found locally for {kind} grid; preserving existing plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6), constrained_layout=True)
    axes = axes.ravel()
    legend_handles = []

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        model_frames = read_oof(dataset)
        for model in MODELS:
            if model not in model_frames:
                continue
            df = model_frames[model]
            res = mean_roc(df) if kind == "ROC" else mean_pr(df)
            if res is None:
                continue
            x, y, _ = res
            ax.plot(
                x,
                y,
                color=COLORS[model],
                lw=2.6 if model == "ArcMDTI" else 1.85,
                ls="-" if model == "ArcMDTI" else "--",
                label=MODEL_DISPLAY[model],
                zorder=4 if model == "ArcMDTI" else 3,
            )
        if kind == "ROC":
            ax.plot([0, 1], [0, 1], color="#777777", lw=1.0, ls=":", zorder=2)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        else:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.grid(True, ls="--", lw=0.5, alpha=0.35)
        if idx == 0:
            legend_handles, _ = ax.get_legend_handles_labels()

    if legend_handles:
        labels = [h.get_label() for h in legend_handles]
        fig.legend(
            legend_handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.055),
            ncol=len(MODELS),
            frameon=True,
            facecolor="white",
            edgecolor="#CCCCCC",
        )

    stem = f"Combined_{kind}"
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=450, bbox_inches="tight")
        sync_file(p)
    plt.close(fig)


def plot_radar_grid(out_dir):
    num_vars = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        2, 3, figsize=(13.2, 8.2), subplot_kw=dict(polar=True), constrained_layout=True
    )
    axes = axes.ravel()
    legend_handles = []

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_METRICS, fontsize=10.5, fontweight="bold", color="#222222")
        ax.set_rlabel_position(30)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], color="#666666", size=8.5)
        ax.set_ylim(0.35, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

        ds_scores = TABLE3_SCORES[dataset]
        for model in MODELS:
            vals = ds_scores[model] + ds_scores[model][:1]
            is_ours = model == "ArcMDTI"
            (line,) = ax.plot(
                angles,
                vals,
                color=COLORS[model],
                linewidth=2.4 if is_ours else 1.65,
                linestyle="-" if is_ours else "--",
                label=MODEL_DISPLAY[model],
                zorder=5 if is_ours else 3,
            )
            ax.fill(
                angles,
                vals,
                color=COLORS[model],
                alpha=0.14 if is_ours else 0.04,
                zorder=2,
            )
            if idx == 0:
                legend_handles.append(line)

        ax.set_title(
            f"Radar Chart — {DATASET_LABELS[dataset]}",
            fontsize=12.5,
            fontweight="bold",
            pad=14,
        )

    if legend_handles:
        labels = [h.get_label() for h in legend_handles]
        fig.legend(
            legend_handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.055),
            ncol=len(MODELS),
            frameon=True,
            facecolor="white",
            edgecolor="#CCCCCC",
        )

    stem = "Combined_RADAR"
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=450, bbox_inches="tight")
        sync_file(p)
    plt.close(fig)


def read_projection(dataset, projection):
    folder = os.path.join("result", f"{dataset}_CV_Seed42")
    if projection == "UMAP":
        path = os.path.join(folder, "BestFold_UMAP_Data_Full.csv")
    else:
        matches = glob.glob(os.path.join(folder, "BestFold_TSNE_Data_*.csv"))
        path = matches[0] if matches else ""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    x_col = f"{projection} Dimension 1" if projection == "UMAP" else "t-SNE Dimension 1"
    y_col = f"{projection} Dimension 2" if projection == "UMAP" else "t-SNE Dimension 2"
    df = df[[x_col, y_col, "Interaction Type"]].dropna()
    df = df.rename(columns={x_col: "x", y_col: "y"})
    return df


def sample_projection(df, n=5000):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


def plot_projection_grid(projection, out_dir):
    any_proj = any(read_projection(ds, projection) is not None for ds in DATASETS)
    if not any_proj:
        print(f"[SKIP] No projection files found locally for {projection}; preserving existing plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6), constrained_layout=True)
    axes = axes.ravel()
    palette = {"Negative": "#3778bf", "Positive": "#d95f5f"}

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        df = read_projection(dataset, projection)
        if df is None:
            ax.axis("off")
            continue
        draw_df = sample_projection(df, n=9000)
        for label in ["Negative", "Positive"]:
            sub = draw_df[draw_df["Interaction Type"] == label]
            ax.scatter(
                sub["x"],
                sub["y"],
                s=7 if len(draw_df) > 5000 else 11,
                c=palette[label],
                alpha=0.62 if label == "Negative" else 0.78,
                linewidths=0,
                label=label,
            )
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel(f"{projection} 1")
        ax.set_ylabel(f"{projection} 2")
        ax.tick_params(length=3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["Negative"], markersize=7.5, label="Negative (Inactive)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["Positive"], markersize=7.5, label="Positive (Active)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
    )

    stem = "Combined_UMAP" if projection == "UMAP" else "Combined_TSNE"
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=450, bbox_inches="tight")
        sync_file(p)
    plt.close(fig)


def compute_separability(out_dir):
    rows = []
    for dataset in DATASETS:
        for projection in ["UMAP", "t-SNE"]:
            df = read_projection(dataset, projection)
            if df is None:
                continue
            work = sample_projection(df, n=5000)
            x = work[["x", "y"]].to_numpy()
            y = (work["Interaction Type"] == "Positive").astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            rows.append(
                {
                    "Dataset": DATASET_LABELS[dataset],
                    "Projection": projection,
                    "Silhouette": silhouette_score(x, y),
                    "Davies_Bouldin": davies_bouldin_score(x, y),
                    "Samples_used": len(work),
                }
            )
    scores = pd.DataFrame(rows)
    if not scores.empty:
        csv_p = os.path.join(out_dir, "Feature_Separability_Scores.csv")
        scores.to_csv(csv_p, index=False)
        sync_file(csv_p)
    return scores


def main():
    set_style()
    out_dir = "sci_plots_comparison"
    os.makedirs(out_dir, exist_ok=True)
    plot_curve_grid("ROC", out_dir)
    plot_curve_grid("PR", out_dir)
    plot_radar_grid(out_dir)
    plot_projection_grid("UMAP", out_dir)
    plot_projection_grid("t-SNE", out_dir)
    scores = compute_separability(out_dir)
    if not scores.empty:
        print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

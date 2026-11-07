# -*- coding:utf-8 -*-
"""
Feature Dimensionality Reduction, Angular Distribution, and Manifold Analysis
=============================================================================
Generates publication-quality UMAP, t-SNE, and Hyperspherical Angular Distribution
plots and exports full coordinate/angle CSVs for downstream combined figures.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")


def generate_sci_ready_dr_plots(
    features,
    labels,
    save_dir="./result/Enzyme_CV_Seed42",
    dataset_name=None,
    angles_deg=None,
    cosines=None,
):
    """
    Generate and save SCI-grade UMAP, t-SNE, and Angular Distribution plots and CSVs.
    """
    os.makedirs(save_dir, exist_ok=True)
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.normpath(save_dir)).split("_")[0]

    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    label_map = {0: "Negative", 1: "Positive"}
    text_labels = np.array([label_map.get(int(l), "Negative") for l in labels])

    print("-" * 50)
    print(f"[Manifold Analysis] Dataset: {dataset_name} | Total samples: {features.shape[0]}, Dim: {features.shape[1]}")
    print(f"   Positives: {np.sum(labels == 1)}, Negatives: {np.sum(labels == 0)}")

    # ==========================================
    # Module 0: Hyperspherical Angular Distribution
    # ==========================================
    if angles_deg is not None and cosines is not None:
        if torch.is_tensor(angles_deg):
            angles_deg = angles_deg.detach().cpu().numpy()
        if torch.is_tensor(cosines):
            cosines = cosines.detach().cpu().numpy()
        df_angles = pd.DataFrame({
            "True_Label": labels.astype(int),
            "Interaction Type": text_labels,
            "Cosine_Neg_Proto": cosines[:, 0],
            "Cosine_Pos_Proto": cosines[:, 1],
            "Angle_Neg_Deg": angles_deg[:, 0],
            "Angle_Pos_Deg": angles_deg[:, 1],
        })
        df_angles.to_csv(os.path.join(save_dir, "BestFold_Angles_Data.csv"), index=False)
        plot_angular_distribution(df_angles, os.path.join(save_dir, f"BestFold_Angular_Dist_{dataset_name}.pdf"), dataset_name)

    # ==========================================
    # Module A: UMAP Projection
    # ==========================================
    print("\n[1/2] Running UMAP projection...")
    start_time = time.time()
    try:
        import umap.umap_ as umap
        umap_reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            n_jobs=1,
        )
        umap_embeddings = umap_reducer.fit_transform(features)
        print(f"UMAP completed in {time.time() - start_time:.2f}s")

        df_umap = pd.DataFrame({
            "UMAP Dimension 1": umap_embeddings[:, 0],
            "UMAP Dimension 2": umap_embeddings[:, 1],
            "Interaction Type": text_labels,
        })
        df_umap.to_csv(os.path.join(save_dir, "BestFold_UMAP_Data_Full.csv"), index=False)
        plot_dr_scatter(
            df_umap,
            x_col="UMAP Dimension 1",
            y_col="UMAP Dimension 2",
            save_path=os.path.join(save_dir, f"BestFold_UMAP_RepSpaceAnalysis_{dataset_name}.pdf"),
        )
        print("UMAP plot & CSV saved successfully.")
    except Exception as e:
        print(f"[WARNING] UMAP failed ({e}), using PCA fallback for UMAP CSV...")
        pca_fallback = PCA(n_components=2, random_state=42).fit_transform(features)
        df_umap = pd.DataFrame({
            "UMAP Dimension 1": pca_fallback[:, 0],
            "UMAP Dimension 2": pca_fallback[:, 1],
            "Interaction Type": text_labels,
        })
        df_umap.to_csv(os.path.join(save_dir, "BestFold_UMAP_Data_Full.csv"), index=False)

    # ==========================================
    # Module B: t-SNE Projection
    # ==========================================
    print("\n[2/2] Running t-SNE projection...")
    t0 = time.time()
    try:
        n_samples = features.shape[0]
        use_sampling = n_samples > 3000

        if use_sampling:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, 3000, replace=False)
            sampled_features = features[sample_idx]
            sampled_labels = text_labels[sample_idx]
        else:
            sampled_features = features
            sampled_labels = text_labels

        n_pca = min(30, sampled_features.shape[1], sampled_features.shape[0])
        pca = PCA(n_components=n_pca, random_state=42)
        pca_features = pca.fit_transform(sampled_features)

        perplexity = max(5, min(30, (sampled_features.shape[0] - 1) // 3))
        max_iter = 500

        tsne = TSNE(
            n_components=2,
            init="pca",
            perplexity=perplexity,
            learning_rate="auto",
            max_iter=max_iter,
            random_state=42,
            n_jobs=1,
            verbose=0,
            angle=0.5,
            early_exaggeration=12.0,
        )

        tsne_embeddings = tsne.fit_transform(pca_features)
        print(f"t-SNE completed in {time.time() - t0:.2f}s")

        save_suffix = "Sampled" if use_sampling else "Full"
        df_tsne = pd.DataFrame({
            "t-SNE Dimension 1": tsne_embeddings[:, 0],
            "t-SNE Dimension 2": tsne_embeddings[:, 1],
            "Interaction Type": sampled_labels,
        })
        df_tsne.to_csv(os.path.join(save_dir, f"BestFold_TSNE_Data_{save_suffix}.csv"), index=False)
        plot_dr_scatter(
            df_tsne,
            x_col="t-SNE Dimension 1",
            y_col="t-SNE Dimension 2",
            save_path=os.path.join(save_dir, f"BestFold_TSNE_RepSpaceAnalysis_{dataset_name}.pdf"),
        )
        print("t-SNE plot & CSV saved successfully.")
    except Exception as e:
        print(f"[WARNING] t-SNE failed: {e}")

    print("-" * 50)


def plot_dr_scatter(df, x_col, y_col, save_path):
    palette = {"Positive": "#D65F5F", "Negative": "#4878D0"}
    plt.figure(figsize=(8, 6), dpi=300)

    sns.scatterplot(
        data=df[df["Interaction Type"] == "Negative"],
        x=x_col,
        y=y_col,
        color=palette["Negative"],
        label="Negative",
        s=30,
        alpha=0.5,
        edgecolor=None,
        zorder=1,
    )
    sns.scatterplot(
        data=df[df["Interaction Type"] == "Positive"],
        x=x_col,
        y=y_col,
        color=palette["Positive"],
        label="Positive",
        s=35,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.2,
        zorder=2,
    )

    plt.xlabel(x_col, fontsize=14, fontweight="bold")
    plt.ylabel(y_col, fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Interaction Type", title_fontsize=12, fontsize=11, loc="best")
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_angular_distribution(df_angles, save_path, dataset_name):
    """Plot the angular distribution theta to the positive class prototype on S^(d-1)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    pos_angles = df_angles[df_angles["True_Label"] == 1]["Angle_Pos_Deg"]
    neg_angles = df_angles[df_angles["True_Label"] == 0]["Angle_Pos_Deg"]

    sns.histplot(
        neg_angles,
        bins=45,
        stat="density",
        color="#3778bf",
        alpha=0.45,
        kde=True,
        label=f"Inactive Pairs (mean $\\theta_{{pos}}={neg_angles.mean():.1f}^\\circ$)",
        ax=ax,
    )
    sns.histplot(
        pos_angles,
        bins=45,
        stat="density",
        color="#d95f5f",
        alpha=0.55,
        kde=True,
        label=f"Active Pairs (mean $\\theta_{{pos}}={pos_angles.mean():.1f}^\\circ$)",
        ax=ax,
    )
    ax.axvline(90.0, color="#333333", linestyle="--", linewidth=1.2, label="Orthogonal Boundary ($90^\\circ$)")
    ax.set_xlabel("Geodesic Angle $\\theta_{pos}$ to Active Class Prototype (Degrees)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(f"Hyperspherical Angular Distribution — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_and_plot_for_dataset(dataset_name, seed=42):
    """Standalone helper to extract 256-D features & geodesic angles from the best fold checkpoint."""
    import argparse
    from functools import partial
    from torch.utils.data import DataLoader
    from config import hyperparameter
    from model import AMCADTI_V2
    from utils.DataSetsFunction import CustomDataSet, collate_fn, load_secondary_structure_dict

    ds_clean = dataset_name.replace(".txt", "")
    dataset_path = f"./DataSets/{ds_clean}.txt"
    if not os.path.exists(dataset_path):
        print(f"[SKIP] Dataset not found at {dataset_path}")
        return

    base_save_dir = f"./result/{ds_clean}_CV_Seed{seed}"
    if not os.path.exists(base_save_dir):
        print(f"[SKIP] Result directory not found: {base_save_dir}")
        return

    # Find best fold checkpoint
    best_ckpt = None
    best_auc = -1.0
    for fold_no in range(1, 6):
        cand_ckpt = os.path.join(base_save_dir, f"fold_{fold_no}", "valid_best_checkpoint.pth")
        cand_npz = os.path.join(base_save_dir, f"fold_{fold_no}", "ROC_PRC_data.npz")
        if os.path.exists(cand_ckpt):
            if os.path.exists(cand_npz):
                try:
                    from sklearn.metrics import roc_auc_score
                    d = np.load(cand_npz)
                    auc_val = float(roc_auc_score(d["y_true"], d["y_scores"]))
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_ckpt = cand_ckpt
                except Exception:
                    if best_ckpt is None:
                        best_ckpt = cand_ckpt
            elif best_ckpt is None:
                best_ckpt = cand_ckpt

    if best_ckpt is None:
        print(f"[SKIP] No valid_best_checkpoint.pth found in {base_save_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = hyperparameter()
    hp.select_dataset_config(dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    ss_dict = load_secondary_structure_dict(dataset_path, all_lines=all_lines)
    my_collate = partial(collate_fn, ss_map_dict=ss_dict, max_prot_len=hp.prot_max_len)

    # Sample up to 5000 pairs (stratified/shuffled) for fast manifold & angle extraction
    if len(all_lines) > 5000:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(all_lines), size=5000, replace=False))
        sample_lines = [all_lines[i] for i in idx]
    else:
        sample_lines = all_lines
    loader = DataLoader(CustomDataSet(sample_lines), batch_size=hp.Batch_size, shuffle=False, collate_fn=my_collate)

    model = AMCADTI_V2(hp).to(device)
    try:
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    feats, labs, cosines_list, angles_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            d, p, ps, labels = [x.to(device) for x in batch]
            _, feat, attn_info = model(d, p, ps, labels=None, return_attn=True)
            feats.append(feat.cpu().numpy())
            labs.append(labels.cpu().numpy())
            if attn_info.get("cosines") is not None:
                cosines_list.append(attn_info["cosines"].cpu().numpy())
                angles_list.append(attn_info["angles_deg"].cpu().numpy())

    if not feats:
        return
    features_matrix = np.concatenate(feats, axis=0)
    labels_vector = np.concatenate(labs, axis=0)
    cosines_matrix = np.concatenate(cosines_list, axis=0) if cosines_list else None
    angles_matrix = np.concatenate(angles_list, axis=0) if angles_list else None

    generate_sci_ready_dr_plots(
        features_matrix,
        labels_vector,
        save_dir=base_save_dir,
        dataset_name=ds_clean,
        angles_deg=angles_matrix,
        cosines=cosines_matrix,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and Plot Best-Fold Latent Manifold & Geodesic Angles")
    parser.add_argument("--dataset", type=str, default="Enzyme", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    extract_and_plot_for_dataset(args.dataset, seed=args.seed)



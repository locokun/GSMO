#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========== 0) 环境变量：务必在 import torch 之前 ==========
import os
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Deterministic training for GSMO on Triplet_Omics_Data")
    p.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES index, e.g. '0' or '6'")
    p.add_argument("--seed", type=int, default=100, help="global random seed (保持与原notebook一致)")
    p.add_argument("--data_dir", type=str, default="data/Triplet_Omics_Data", help="dataset folder")
    p.add_argument("--out_dir", type=str, default="outputs_triplet", help="where to save checkpoints/figures")
    p.add_argument("--max_epochs", type=int, default=2000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--check_interval", type=int, default=200)
    p.add_argument("--nmi_threshold", type=float, default=0.98)
    p.add_argument("--n_clusters_eval", type=int, default=5, help="kmeans clusters for evaluation (notebook里是5)")
    p.add_argument("--no_plots", action="store_true", help="skip plotting to speed up")
    p.add_argument("--load_ckpt", type=str, default="", help="path to load an existing model state_dict")
    return p.parse_args()

args = parse_args()

# 设备可见性需在 import torch 前设置
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# RNG & 数值稳定相关
os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.makedirs(args.out_dir, exist_ok=True)

# ========== 1) 常用库（此处才 import torch） ==========
import random
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sp
import hashlib

import torch
from gsmo.utils import preprocess
from gsmo import GSMO, GSMOConfig

from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    mutual_info_score,
    v_measure_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.metrics import pairwise_distances

# 禁用 TF32，开启确定性算法
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

fix_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}, GPU name: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
print(f"[INFO] Seed: {args.seed}")

# ========== 2) 读数据 ==========
print(f"[INFO] Loading AnnData from {args.data_dir}")
rna_path = os.path.join(args.data_dir, "adata_RNA.h5ad")
atac_path = os.path.join(args.data_dir, "adata_ATAC.h5ad")
adt_path = os.path.join(args.data_dir, "adata_ADT.h5ad")
gt_path = os.path.join(args.data_dir, "ground_truth_clusters.csv")

rna = sc.read(rna_path)
atac = sc.read(atac_path)
protein = sc.read(adt_path)

# 备份空间坐标（训练与评估都用这一份）
assert "spatial" in rna.obsm, "RNA AnnData 缺少 obsm['spatial']"
spatial_coords = rna.obsm["spatial"].copy()

# var名唯一化并预处理（与notebook一致）
rna.var_names_make_unique()
atac.var_names_make_unique()
protein.var_names_make_unique()

print("[INFO] Preprocessing modalities ...")
rna = preprocess(rna, modality="rna")
atac = preprocess(atac, modality="atac", n_dim=1000)
protein = preprocess(protein, modality="protein")

'''# 把 .X 转为 np.ndarray（若是稀疏）
def to_array(X):
    return X.toarray() if sp.issparse(X) else (np.asarray(X) if not isinstance(X, np.ndarray) else X)

rna_X = to_array(rna.X)
atac_X = to_array(atac.X)
protein_X = to_array(protein.X)
'''
# ========== 3) 构建模型（cfg=presets v2）并（可选）载入 ckpt ==========
features_dict = {
    "rna": rna,
    "protein": protein,
    "atac": atac,
}
initial_weights = [1.0, 1.0, 1.0]
cfg = GSMOConfig.presets("triplet")
cfg.checkpoint_name = os.path.join(args.data_dir, f"best_model_train.pth")

model = GSMO(
    features_dict,
    spatial_coords=spatial_coords,
    device=device,
    cfg=cfg,
    trainable_weights=True,
    initial_weights=initial_weights,
)

if args.load_ckpt:
    print(f"[INFO] Loading checkpoint: {args.load_ckpt}")
    state = torch.load(args.load_ckpt, map_location=device)
    model.load_state_dict(state)

# ========== 4) 训练 ==========
print("[INFO] Start training ...")
model.train_model(
    max_epochs=args.max_epochs,
    patience=args.patience,
    check_interval=args.check_interval,
    nmi_threshold=args.nmi_threshold,
)

# —— 训练内部已 load 回 best ckpt，这里打印一个“指纹”以便两次运行核对
def tensor_fingerprint(t):
    a = t.detach().float().cpu().view(-1)[:1024].numpy().tobytes()
    return hashlib.md5(a).hexdigest()

with torch.no_grad():
    for n, p in model.named_parameters():
        print(f"[FP] {n}: {tensor_fingerprint(p)}")
        break

# 复制内部 best ckpt 到输出目录
internal_ckpt = os.path.join(args.data_dir, f"best_model_train.pth")
if os.path.exists(internal_ckpt):
    dst = os.path.join(args.out_dir, f"best_model_seed{args.seed}.pth")
    try:
        import shutil
        shutil.copyfile(internal_ckpt, dst)
        print(f"[INFO] Copied internal ckpt to {dst}")
    except Exception as e:
        print(f"[WARN] Failed to copy ckpt: {e}")

# ========== 5) 导出 embeddings，并用“锁死”的评估路径做聚类 ==========
print("[INFO] Export embeddings at best ckpt ...")
emb = model.get_embeddings()
np.save(os.path.join(args.out_dir, f"emb_at_best_seed{args.seed}.npy"), emb)

def kmeans_deterministic(X, n_clusters, seed=100):
    kwargs = dict(n_clusters=n_clusters, n_init=10, init="k-means++", random_state=seed)
    try:
        km = KMeans(algorithm="lloyd", **kwargs)  # sklearn>=1.3
    except TypeError:
        km = KMeans(**kwargs)                     # 旧版本
    with threadpool_limits(limits=1, user_api="blas"):
        return km.fit_predict(X)

clusters = kmeans_deterministic(emb, n_clusters=args.n_clusters_eval, seed=100)

# 稳定重标号（按簇中心的字典序），保证配色与图像一致
def relabel_stable(labels, X):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    centers = np.stack([X[labels == c].mean(axis=0) for c in uniq], axis=0)
    order = np.lexsort(np.array(centers).T[::-1])  # 多维字典序
    mapping = {int(uniq[i]): int(i) for i in order}
    return np.array([mapping[int(z)] for z in labels])

clusters_stable = relabel_stable(clusters, emb)
np.save(os.path.join(args.out_dir, f"clusters_at_best_seed{args.seed}.npy"), clusters_stable)

# ========== 6) 监督指标（若提供 GT） ==========
gt_csv = os.path.join(args.data_dir, "ground_truth_clusters.csv")
if os.path.exists(gt_csv):
    gt = pd.read_csv(gt_csv)["cluster"].values
    # 假设顺序与当前 AnnData 一致；若不是，请在此处显式对齐索引
    scores = {
        "ARI": adjusted_rand_score(gt, clusters_stable),
        "Homogeneity": homogeneity_score(gt, clusters_stable),
        "Mutual Info": mutual_info_score(gt, clusters_stable),
        "V-measure": v_measure_score(gt, clusters_stable),
        "AMI": adjusted_mutual_info_score(gt, clusters_stable),
        "NMI": normalized_mutual_info_score(gt, clusters_stable),
    }
    print("=== Clustering Evaluation Metrics (deterministic path) ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
else:
    print("[INFO] No ground_truth_clusters.csv found; skip supervised metrics.")

# ========== 7) 无监督指标（Jaccard / Moran's I）==========
# 这里用预处理后的矩阵与 emb 对比（与notebook一致）
def compute_jaccard_similarity(Z_m, Z_e, k=50):
    dist_m = pairwise_distances(Z_m)
    dist_e = pairwise_distances(Z_e)
    n = Z_m.shape[0]
    jaccard_scores = []
    # 顺序固定，不用任何随机
    for i in range(n):
        neighbors_m = set(np.argsort(dist_m[i])[1:k+1])
        neighbors_e = set(np.argsort(dist_e[i])[1:k+1])
        inter = len(neighbors_m & neighbors_e)
        union = len(neighbors_m | neighbors_e)
        jaccard_scores.append(inter / union if union > 0 else 0.0)
    return np.array(jaccard_scores), float(np.mean(jaccard_scores))

jm, jm_mean = compute_jaccard_similarity(rna, emb, k=50)
print(f"Mean Jaccard RNA Similarity: {jm_mean:.4f}")
jm, jm_mean = compute_jaccard_similarity(atac, emb, k=50)
print(f"Mean Jaccard ATAC Similarity: {jm_mean:.4f}")
jm, jm_mean = compute_jaccard_similarity(protein, emb, k=50)
print(f"Mean Jaccard protein Similarity: {jm_mean:.4f}")

# Moran's I（与 notebook 相同实现）
import squidpy as sq
def compute_morans_I(labels, spatial_coords, label_name="cluster_label"):
    adata = sc.AnnData(X=np.zeros((len(labels), 1)))
    adata.obs[label_name] = pd.Categorical([str(x) for x in labels])
    adata.obsm["spatial"] = spatial_coords.astype(float)
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=label_name, attr="obs")
    return float(adata.uns["moranI"].loc[label_name, "I"])

morans_I = compute_morans_I(clusters_stable, spatial_coords)
print(f"Moran's I score: {morans_I:.4f}")

# ========== 8) （可选）绘图（用 matplotlib，直接从 .obsm['spatial'] 画） ==========
if not args.no_plots:
    print("[INFO] Plotting ...")
    matplotlib.rcParams["figure.dpi"] = 200

    # 读取原始 RNA，用其 obsm['spatial']
    rna_plot = sc.read(rna_path)
    assert "spatial" in rna_plot.obsm, "adata_RNA.h5ad 缺少 obsm['spatial']"
    coords = np.asarray(rna_plot.obsm["spatial"])
    assert coords.shape[1] == 2, "obsm['spatial'] 形状应为 [N,2]"

    # 颜色盘（与你 notebook 相同）
    from matplotlib.colors import ListedColormap
    plot_color = [
        '#f58231', '#3cb44b', '#ffe119', '#4363d8', '#e6194b', '#911eb4', '#46f0f0', '#f032e6', 
        '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', 
        '#000075', '#000000', '#808080', '#ffffff', '#fffac8', '#D1D1D1'
    ]

    # 用我们前面算好的 clusters_stable，确保配色稳定
    labels_for_plot = np.asarray(clusters_stable).astype(int)
    num_clusters = int(np.unique(labels_for_plot).size)
    if num_clusters > len(plot_color):
        raise ValueError(f"Cluster number {num_clusters} exceeds palette size {len(plot_color)}.")

    cmap = ListedColormap(plot_color[:num_clusters])

    # 画图（与 notebook 一致）
    fig, ax = plt.subplots(figsize=(10, 8))
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=labels_for_plot, cmap=cmap, s=80)
    cbar = plt.colorbar(sca, ax=ax, label="Cluster")
    ax.set_title("Spatial Distribution of Clusters")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir, f"triplet_clusters_seed{args.seed}.png")
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    print(f"[INFO] Saved {fig_path}")

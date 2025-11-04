import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import scipy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph
from PIL import Image
import random

def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell using Seurat-style CLR"""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def tfidf_transform(adata):
    """TF-IDF transformation for sparse ATAC matrix"""
    from sklearn.preprocessing import normalize

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    idf = np.log(1 + X.shape[0] / (1 + np.sum(X > 0, axis=0)))  # inverse document freq
    tf = normalize(X, norm='l1', axis=1)  # term frequency
    tfidf = tf * idf
    return tfidf

def lsi_from_tfidf(adata, n_components=3000):
    """Perform TF-IDF + TruncatedSVD to compute LSI"""
    from sklearn.decomposition import TruncatedSVD
    tfidf = tfidf_transform(adata)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi = svd.fit_transform(tfidf)
    return lsi

def preprocess(adata, modality, n_dim=1024, save_path=None):
    """
    Preprocess adata based on modality.
    If save_path is provided, save the processed adata to a .h5ad file.
    """
    adata.var_names_make_unique()

    if modality == 'rna':
        sc.pp.filter_genes(adata, min_cells=10)

        # 只在有 "vf_vst_counts_variable" 标志的情况下筛选高变基因
        if 'vf_vst_counts_variable' in adata.var.columns:
            hvg_mask = adata.var['vf_vst_counts_variable'].values.astype(bool)
            adata = adata[:, hvg_mask]

        if not hasattr(adata, 'raw') or adata.raw is None:
            sc.pp.log1p(adata)
        
        # 保护机制：在scale之前过滤掉标准差为0的基因
        X_dense = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        gene_std = np.std(X_dense, axis=0)
        non_zero_std_genes = np.where(gene_std > 1e-6)[0]  # 保留标准差大于很小阈值的基因
        adata = adata[:, non_zero_std_genes]  # 只保留这些基因

        sc.pp.scale(adata)  # 现在scale不会炸了

        # 最保险：scale完再清理一遍NaN（虽然应该没有了）
        X_dense = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=1e5, neginf=-1e5)

        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ Processed RNA data saved to {save_path}")
        
        return X_dense

    elif modality in ['atac', 'histone', 'H3K27ac', 'H3K27me3', 'H3K4me3']:
        if 'X_lsi' in adata.obsm:
            lsi = adata.obsm['X_lsi']
            if lsi.shape[1] >= n_dim:
                if save_path is not None:
                    adata.write_h5ad(save_path)
                    print(f"✅ ATAC/histone data (existing LSI) saved to {save_path}")
                return lsi[:, :n_dim]
            else:
                print(f"⚠️ obsm['X_lsi'] has only {lsi.shape[1]} dims, recomputing LSI.")
        else:
            print("⚠️ X_lsi not found, performing LSI in Python from count matrix.")

        lsi = lsi_from_tfidf(adata, n_components=n_dim)
        adata.obsm['X_lsi'] = lsi

        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ Recomputed and saved ATAC/histone data with new LSI to {save_path}")

        return lsi

    elif modality == 'protein':
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        sc.pp.scale(adata)

        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ Protein data saved to {save_path}")

        return adata.X
    elif modality == "metabolite":
        sc.pp.log1p(adata)
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        if save_path is not None:
            adata.write_h5ad(save_path)
            print(f"✅ metabolite data saved to {save_path}")
        return adata.X

    else:
        raise ValueError(f"Unsupported modality: {modality}")




def calculate_affinity(X1, sig=30, sparse = False, neighbors = 100):
  if not sparse:
    dist1 = pairwise_distances(X1)
    a1 = np.exp(-1*(dist1**2)/(2*(sig**2)))
    return a1
  else:
    dist1 = kneighbors_graph(X1, n_neighbors = neighbors, mode='distance')
    dist1.data = np.exp(-1*(dist1.data**2)/(2*(sig**2)))
    dist1.eliminate_zeros()
    return dist1

def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)



def cmap_tab30(x):
    n_base = 20
    n_max = 30
    brightness = 0.7
    brightness = (brightness,) * 3 + (1.0,)
    isin_base = (x < n_base)[..., np.newaxis]
    isin_extended = ((x >= n_base) * (x < n_max))[..., np.newaxis]
    isin_beyond = (x >= n_max)[..., np.newaxis]
    color = (
        isin_base * cmap_tab20(x)
        + isin_extended * cmap_tab20(x-n_base) * brightness
        + isin_beyond * (0.0, 0.0, 0.0, 1.0))
    return color


def cmap_tab70(x):
    cmap_base = cmap_tab30
    brightness = 0.5
    brightness = np.array([brightness] * 3 + [1.0])
    color = [
        cmap_base(x),  # same as base colormap
        1 - (1 - cmap_base(x-20)) * brightness,  # brighter
        cmap_base(x-20) * brightness,  # darker
        1 - (1 - cmap_base(x-40)) * brightness**2,  # even brighter
        cmap_base(x-40) * brightness**2,  # even darker
        [0.0, 0.0, 0.0, 1.0],  # black
        ]
    x = x[..., np.newaxis]
    isin = [
        (x < 30),
        (x >= 30) * (x < 40),
        (x >= 40) * (x < 50),
        (x >= 50) * (x < 60),
        (x >= 60) * (x < 70),
        (x >= 70)]
    color_out = np.sum(
            [isi * col for isi, col in zip(isin, color)],
            axis=0)
    return color_out


def plot(clusters,locs):
  locs['2'] = locs['2'].astype('int')
  locs['3'] = locs['3'].astype('int')
  im1 = np.empty((locs['2'].max()+1, locs['3'].max()+1))
  im1[:] = np.nan
  im1[locs['2'],locs['3']] = clusters
  im2 = cmap_tab70(im1.astype('int'))
  im2[np.isnan(im1)] = 1
  im3 = Image.fromarray((im2 * 255).astype(np.uint8))
  return im3

def plot_on_histology(clusters, locs, im, scale, s=10):
  locs = locs*scale
  locs = locs.round().astype('int')
  im = im[(locs['4'].min()-10):(locs['4'].max()+10),(locs['5'].min()-10):(locs['5'].max()+10)]
  locs = locs-locs.min()+10
  cmap1 = mcolors.ListedColormap([cmap_tab70(np.array(i)) for i in range(len(np.unique(clusters)))])
  plt.imshow(im, alpha=0.7); 
  plot = plt.scatter(x=locs['5'], y=locs['4'], c = clusters, cmap=cmap1, s=s); 
  plt.axis('off'); 
  # plt.savefig('/opt/user2/1/gsmo/images/tutorial-uni/unimodel4.png')
  return plot

def set_random_seed(seed=100):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)


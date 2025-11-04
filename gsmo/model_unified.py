# -*- coding: utf-8 -*-
"""
Unified GSMO implementation that reproduces three historical versions via a
single codebase controlled by a small set of configuration switches.

How to use
----------
1) Pick one of the built-in presets ("v1", "v2", "v3") to exactly match
   behavior of your first/second/third code versions per-dataset, e.g.:

    cfg = GSMOConfig.presets("v1")
    model = GSMO(features_dict, spatial_coords, device=device, cfg=cfg)

2) Or customize any switch in GSMOConfig to compose your own variant.

Repro-critical settings (matching your 3 versions):
- k_neighbors, hidden_dims rule, losses + coefficients, warm-up schedule,
  graph contrast margin, DEC & balance penalties, cluster-weight consistency,
  image reconstruction scale, ARI grid, default K for kmeans, checkpoint name.

This file contains no external dependencies beyond your original imports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Optional heavy deps used only when requested
import scanpy as sc  # noqa: F401
import anndata  # noqa: F401
import pandas as pd  # noqa: F401


# v1:model、v2:cluster、v3:2025
# ---------------------- Config ----------------------
@dataclass
class GSMOConfig:
    # Topology / data
    embed_dim: int = 128
    k_neighbors: int = 15
    # Hidden dims rule
    max_hidden_layers: int = 2  # v1:2, v2:2, v3:3

    # Loss toggles
    use_graph_contrast: bool = True
    graph_margin: float = 2.0  # v1:2.0, v2:3.0
    graph_warmup_epochs: int = 300  # v1:300, v2:200
    coef_graph: float = 0.04  # v1:0.04, v2:0.05, v3:0 (if disabled)

    use_dec_compactness: bool = True
    coef_dec: float = 0.4  # v1:0.4, v2:0.2, v3:0

    use_balance_penalty: bool = True
    coef_balance: float = 0.05  # v1/v2:0.05

    use_spatial_consistency: bool = False
    coef_spatial_consistency: float = 0.02  # v3:0.02

    # Cross-modality losses
    coef_contrastive: float = 0.2  # v1:0.2, v2:0.3, v3:0.3
    coef_align: float = 0.3  # v1:0.3, v2:0.5, v3:0.7

    # Weight regularization
    coef_weight_reg: float = 0.01

    # Image modality reconstruction scale (v1 special case 0.01)
    image_recon_scale: float = 1.0

    # Cluster-weight consistency (v2)
    use_cluster_weight_consistency: bool = False
    coef_cluster_weight: float = 0.1
    cluster_weight_warmup_epochs: int = 200

    # Training / evaluation
    default_kmeans_k: int = 14  # v1:14, v2:5, v3:15
    ari_grid: Tuple[int, int] = (8, 13)  # range(start, stop) as in range()
    nmi_threshold: float = 0.90
    checkpoint_name: str = "best_model.pth"

    # Attention / fusion
    use_conditional_attention: bool = True
    use_spatial_encoding_in_fusion: bool = False

    @staticmethod
    def presets(name: str) -> "GSMOConfig":
        name = name.lower().strip()
        if name == "triplet": # triplet
            return GSMOConfig(
                k_neighbors=6,
                max_hidden_layers=2,
                use_graph_contrast=True,
                graph_margin=3.0,
                graph_warmup_epochs=200,
                coef_graph=0.05,
                use_dec_compactness=True,
                coef_dec=0.2,
                use_balance_penalty=True,
                use_spatial_consistency=False,
                coef_spatial_consistency=0.0,
                coef_contrastive=0.3,
                coef_align=0.5,
                coef_weight_reg=0.01,
                image_recon_scale=1.0,
                use_cluster_weight_consistency=True,
                coef_cluster_weight=0.1,
                cluster_weight_warmup_epochs=200,
                default_kmeans_k=5,
                ari_grid=(6, 9),
                checkpoint_name="best_model.pth",
                use_conditional_attention=True,
                use_spatial_encoding_in_fusion=False,
            )


# ---------------------- Building blocks ----------------------
class ImageAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        emb = self.encoder(x)
        rec = self.decoder(emb)
        return emb, rec

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.encoder(x)


class MLP_AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 128):
        super().__init__()
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, output_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        prev = output_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        rec = self.decoder(emb)
        return emb, rec

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ModalityWeighting(nn.Module):
    def __init__(self, num_modalities: int, trainable_weights: bool = True, initial_weights: Optional[List[float]] = None):
        super().__init__()
        if initial_weights is None:
            init = torch.ones(num_modalities)
        else:
            init = torch.tensor(initial_weights, dtype=torch.float)
        if trainable_weights:
            self.weights = nn.Parameter(init)
        else:
            self.register_buffer("weights", init)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, embeddings_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.softmax(self.weights)
        weighted = sum(w * e for w, e in zip(weights, embeddings_list))
        return weighted, weights


class ModalityFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, use_spatial_encoding: bool = True):
        super().__init__()
        self.use_spatial_encoding = use_spatial_encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=2
        )
        self.attention_fc = nn.Linear(embed_dim, 1)

    def forward(self, embeddings_list: List[torch.Tensor], spatial_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.stack(embeddings_list, dim=1)  # [B, M, D]
        if self.use_spatial_encoding and spatial_coords is not None:
            x = x + self.get_spatial_encoding(spatial_coords, x.size(-1)).unsqueeze(1)
        x = self.transformer(x)
        attn = torch.softmax(self.attention_fc(x), dim=1)  # [B, M, 1]
        fused = torch.sum(x * attn, dim=1)
        return fused

    def get_spatial_encoding(self, coords: torch.Tensor, embed_dim: int) -> torch.Tensor:
        mlp = nn.Sequential(nn.Linear(2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)).to(coords.device)
        return mlp(coords)


class ConditionalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_modalities: int):
        super().__init__()
        self.attention_fc = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, num_modalities))

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.attention_fc(fused_emb), dim=-1)


# ---------------------- Main model ----------------------
class GSMO(nn.Module):
    def __init__(
        self,
        features_dict: Dict[str, np.ndarray],
        spatial_coords: np.ndarray,
        device: str = "cpu",
        k_neighbors: Optional[int] = None,
        embed_dim: int = 128,
        trainable_weights: bool = True,
        initial_weights: Optional[List[float]] = None,
        cfg: Optional[GSMOConfig] = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.modality_names = list(features_dict.keys())
        self.num_modalities = len(self.modality_names)
        self.cfg = cfg or GSMOConfig()
        # Allow legacy ctor args to override cfg basics if provided
        if k_neighbors is not None:
            self.cfg.k_neighbors = k_neighbors
        if embed_dim != self.cfg.embed_dim:
            self.cfg.embed_dim = embed_dim

        # Standardize and move to device
        self.features: Dict[str, torch.Tensor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        for modality, feat in features_dict.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(feat)
            self.features[modality] = torch.tensor(scaled, dtype=torch.float32, device=self.device)
            self.scalers[modality] = scaler
        self.spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32, device=self.device)

        # Encoders & fusion decoders
        self.encoders = nn.ModuleDict()
        self.fusion_decoders = nn.ModuleDict()
        for modality, feat in self.features.items():
            in_dim = feat.shape[1]
            if modality == "image":
                self.encoders[modality] = ImageAutoEncoder(input_dim=in_dim, output_dim=self.cfg.embed_dim).to(self.device)
                self.fusion_decoders[modality] = nn.Sequential(nn.Linear(self.cfg.embed_dim, 32), nn.ReLU(), nn.Linear(32, in_dim)).to(self.device)
            else:
                hidden_dims = self._compute_hidden_dims(in_dim)
                self.encoders[modality] = MLP_AutoEncoder(in_dim, hidden_dims, output_dim=self.cfg.embed_dim).to(self.device)
                self.fusion_decoders[modality] = (
                    nn.Sequential(nn.Linear(self.cfg.embed_dim, hidden_dims[-1]), nn.ReLU(), nn.Linear(hidden_dims[-1], in_dim)).to(self.device)
                )

        # Weighting & fusion
        self.modality_weighting = ModalityWeighting(self.num_modalities, trainable_weights, initial_weights).to(self.device)
        self.fusion_layer = ModalityFusion(embed_dim=self.cfg.embed_dim, num_heads=4, use_spatial_encoding=self.cfg.use_spatial_encoding_in_fusion).to(self.device)

        # Spatial graph
        self.spatial_adj = self._compute_spatial_adj(self.spatial_coords.detach().cpu().numpy(), self.cfg.k_neighbors).to(self.device)

        # Optimizer
        encoder_params = [p for enc in self.encoders.values() for p in enc.parameters()]
        decoder_params = [p for dec in self.fusion_decoders.values() for p in dec.parameters()]
        weighting_params = list(self.modality_weighting.parameters())
        fusion_params = list(self.fusion_layer.parameters())
        self.optimizer = optim.AdamW(encoder_params + decoder_params + weighting_params + fusion_params, lr=1e-3, weight_decay=1e-4)

        # Conditional attention module (optionally used in training)
        self.cond_attention = (ConditionalAttention(embed_dim=self.cfg.embed_dim, num_modalities=self.num_modalities).to(self.device)
                                if self.cfg.use_conditional_attention else None)

    # ---------------- utils ----------------
    def _compute_hidden_dims(self, input_dim: int) -> List[int]:
        # Reproduces v1/v2 (max 2) and v3 (max 3) behavior via cfg.max_hidden_layers
        num_layers = min(self.cfg.max_hidden_layers, max(1, int(np.log2(max(input_dim / self.cfg.embed_dim, 1)))))
        hidden = [min(1024, input_dim // 2)]
        for i in range(1, num_layers):
            if i == 1:
                hidden.append(min(512, hidden[-1] // 2))
            else:
                hidden.append(min(256, hidden[-1] // 2))
        return hidden


    def _compute_spatial_adj(self, spatial_coords: np.ndarray, k_neighbors: int) -> torch.Tensor:
        adj = kneighbors_graph(spatial_coords, n_neighbors=k_neighbors, mode="connectivity", include_self=False)
        return torch.tensor(adj.toarray(), dtype=torch.float32)


    def _get_hidden_dims(self, modality: str, input_dim: int) -> List[int]:
        # If user provided exact per-modality hidden dims, use them
        if self.cfg.override_hidden_dims and modality in self.cfg.override_hidden_dims:
            return list(self.cfg.override_hidden_dims[modality])
        return self._compute_hidden_dims(input_dim)

    # ---------------- losses ----------------
    def _graph_contrastive_loss(self, emb: torch.Tensor, adj: torch.Tensor, margin: float) -> torch.Tensor:
        adj_sparse = adj.to_sparse().coalesce()
        row, col = adj_sparse.indices()
        pos_dist = torch.norm(emb[row] - emb[col], dim=1).mean()
        rand_idx = torch.randperm(col.size(0), device=emb.device)
        neg_dist = torch.norm(emb[row] - emb[col[rand_idx]], dim=1)
        neg = F.relu(margin - neg_dist).mean()
        return pos_dist + neg

    def _dec_compactness_loss(self, emb: torch.Tensor, n_clusters: int) -> torch.Tensor:
        with torch.no_grad():
            km = KMeans(n_clusters=n_clusters, random_state=0).fit(emb.detach().cpu().numpy())
            centers = torch.tensor(km.cluster_centers_, device=emb.device, dtype=emb.dtype)
        dist = torch.cdist(emb, centers) ** 2
        q = 1.0 / (1.0 + dist)
        q = q / q.sum(dim=1, keepdim=True)
        p = (q ** 2) / q.sum(dim=0, keepdim=True)
        p = p / p.sum(dim=1, keepdim=True)
        return F.kl_div(q.log(), p, reduction="batchmean")

    def _balance_penalty(self, labels: np.ndarray, K: int) -> torch.Tensor:
        hist = torch.bincount(torch.tensor(labels, device=self.device), minlength=K).float()
        probs = hist / hist.sum()
        entropy = -(probs * (probs + 1e-9).log()).sum()
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float, device=self.device))
        return 1.0 - entropy / max_entropy

    def _spatial_consistency_loss(self, emb: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_sparse = adj.to_sparse().coalesce()
        row, col = adj_sparse.indices()
        diff = torch.norm(emb[row] - emb[col], dim=1)
        return diff.mean()

    def _cluster_assign(self, emb: torch.Tensor, n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(emb.detach().cpu().numpy())
        labels = torch.tensor(km.labels_, device=self.device, dtype=torch.long)
        masks = F.one_hot(labels, num_classes=n_clusters).float().T  # (K,N)
        return labels, masks

    def _cluster_weight_consistency(self, weights: torch.Tensor, cluster_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cluster_w = cluster_masks @ weights / (cluster_masks.sum(dim=1, keepdim=True) + 1e-9)  # (K,M)
        diff = weights.unsqueeze(0) - cluster_w.unsqueeze(1)  # (K,N,M)
        intra = ((cluster_masks.unsqueeze(-1) * diff ** 2).sum() / (cluster_masks.sum() + 1e-9))
        grand = cluster_w.mean(dim=0, keepdim=True)
        inter = ((cluster_w - grand) ** 2).mean()
        return intra, inter

    def _compute_loss(
        self,
        embeddings_list: List[torch.Tensor],
        reconstructed_list: List[torch.Tensor],
        fused: torch.Tensor,
        weights_batch_or_vec: torch.Tensor,
        epoch: int,
        n_clusters: int,
        cluster_specific_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Reconstruction: with v1 image scaling option
        recon_auto, recon_fusion = 0.0, 0.0
        for mod, rec in zip(self.modality_names, reconstructed_list):
            scale = self.cfg.image_recon_scale if mod == "image" else 1.0
            recon_auto = recon_auto + scale * F.mse_loss(rec, self.features[mod])
            recon_fusion = recon_fusion + scale * F.mse_loss(self.fusion_decoders[mod](fused), self.features[mod])

        # Cross-modality losses
        contrastive = sum(F.mse_loss(embeddings_list[i], embeddings_list[j]) for i in range(self.num_modalities) for j in range(i + 1, self.num_modalities))
        align = sum(F.mse_loss(fused, e) for e in embeddings_list) / self.num_modalities

        total = (recon_auto + recon_fusion) \
            + self.cfg.coef_contrastive * contrastive \
            + self.cfg.coef_align * align

        # Graph contrast with warmup
        if self.cfg.use_graph_contrast:
            warm = min(1.0, epoch / float(max(self.cfg.graph_warmup_epochs, 1)))
            total = total + warm * self.cfg.coef_graph * self._graph_contrastive_loss(fused, self.spatial_adj, margin=self.cfg.graph_margin)

        # DEC compactness
        if self.cfg.use_dec_compactness:
            total = total + self.cfg.coef_dec * self._dec_compactness_loss(fused, n_clusters=n_clusters)

        # Balance penalty via KMeans labels (no GT labels)
        if self.cfg.use_balance_penalty:
            km_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(fused.detach().cpu().numpy())
            total = total + self.cfg.coef_balance * self._balance_penalty(km_labels, n_clusters)

        # Spatial consistency (v3)
        if self.cfg.use_spatial_consistency:
            total = total + self.cfg.coef_spatial_consistency * self._spatial_consistency_loss(fused, self.spatial_adj)

        # Weight regularization
        total = total + self.cfg.coef_weight_reg * torch.sum((weights_batch_or_vec - 1 / self.num_modalities) ** 2)

        # Cluster-weight consistency (v2)
        if self.cfg.use_cluster_weight_consistency:
            labels, masks = self._cluster_assign(fused, n_clusters=n_clusters)
            L_intra, L_inter = self._cluster_weight_consistency(weights_batch_or_vec, masks)
            alpha = min(1.0, epoch / float(max(self.cfg.cluster_weight_warmup_epochs, 1)))
            total = total + self.cfg.coef_cluster_weight * ((1 - alpha) * L_intra + alpha * L_inter)

        if cluster_specific_weight is not None:
            total = total + 0.1 * torch.sum((weights_batch_or_vec - cluster_specific_weight) ** 2)
        return total

    # ---------------- training & inference ----------------
    def train_model(
        self,
        max_epochs: int = 2000,
        patience: int = 300,
        check_interval: int = 50,
        nmi_threshold: Optional[float] = None,
        use_ari_as_metric: bool = False,
        true_labels: Optional[np.ndarray] = None,
        use_conditional_attention: Optional[bool] = None,
        use_spatial_encoding: Optional[bool] = None,
        cluster_specific_weight: Optional[torch.Tensor] = None,
        n_clusters_for_losses: Optional[int] = None,
    ) -> None:
        from sklearn.metrics import normalized_mutual_info_score as nmi

        best_loss = float("inf")
        patience_counter = 0
        prev_clusters = None
        best_metric = -float("inf")
        best_epoch = -1
        eps = 1e-12

        if nmi_threshold is None:
            nmi_threshold = self.cfg.nmi_threshold
        if use_conditional_attention is None:
            use_conditional_attention = self.cfg.use_conditional_attention
        if use_spatial_encoding is None:
            use_spatial_encoding = self.cfg.use_spatial_encoding_in_fusion
        if n_clusters_for_losses is None:
            n_clusters_for_losses = self.cfg.default_kmeans_k

        for epoch in tqdm(range(max_epochs), desc="Training"):
            self.optimizer.zero_grad()

            embeddings_list: List[torch.Tensor] = []
            reconstructed_list: List[torch.Tensor] = []
            for modality in self.modality_names:
                enc = self.encoders[modality]
                emb, rec = enc(self.features[modality])
                embeddings_list.append(emb)
                reconstructed_list.append(rec)

            fused = self.fusion_layer(embeddings_list, self.spatial_coords if use_spatial_encoding else None)

            if use_conditional_attention:
                weights = self.cond_attention(fused)  # [N,M]
            else:
                # v2 reproduces per-batch expansion of vector weights
                weights_vec = self.modality_weighting.weights
                if weights_vec.dim() == 1:
                    weights = weights_vec.expand(fused.size(0), -1)
                else:
                    weights = weights_vec

            loss = self._compute_loss(
                embeddings_list,
                reconstructed_list,
                fused,
                weights,
                epoch,
                n_clusters=n_clusters_for_losses,
                cluster_specific_weight=cluster_specific_weight,
            )

            loss.backward()
            self.optimizer.step()

            if epoch % 20 == 0:
                with torch.no_grad():
                    print(f"Epoch [{epoch+1}/{max_epochs}] Loss={loss.item():.4f}; Weights(mean)={weights.mean(dim=0).tolist()}")

            # Clustering stability check (KMeans over current embeddings)
            if epoch % check_interval == 0:
                current_clusters = self.cluster_kmeans(n_clusters=n_clusters_for_losses)
                if prev_clusters is not None:
                    cluster_nmi = nmi(prev_clusters, current_clusters)
                    print(f"Epoch {epoch}, Clustering NMI: {cluster_nmi:.4f}")
                    if cluster_nmi >= nmi_threshold:
                        print("Clustering stabilized, stopping early.")
                        self._save_checkpoint()
                        break
                prev_clusters = current_clusters

            # ARI-based early stopping (evaluation only)
            if epoch % check_interval == 0 and true_labels is not None:
                df_embedding = fused.detach().cpu().numpy()
                start, stop = self.cfg.ari_grid
                rec_ari = []
                for n in range(start, stop):
                    clusters = self.cluster_kmeans(embeddings=df_embedding, n_clusters=n)
                    rec_ari.append(ari_score(true_labels, clusters))
                max_ari = max(rec_ari) if len(rec_ari) else -1.0
                print(f"Epoch {epoch}, Max ARI: {max_ari:.4f}")
                if (max_ari > best_metric + eps) or (abs(max_ari - best_metric) <= eps and epoch > best_epoch):
                    best_metric = max_ari
                    best_epoch = epoch
                    self._save_checkpoint()
                if max_ari >= nmi_threshold:
                    print("ARI stable, early stop.")
                    break

            # Loss-based early stop
            if not use_ari_as_metric:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    self._save_checkpoint()
                else:
                    patience_counter += 1
            if patience_counter >= patience:
                print("Loss plateau, early stop.")
                break

        self._load_checkpoint()

    def _save_checkpoint(self) -> None:
        torch.save(self.state_dict(), self.cfg.checkpoint_name)

    def _load_checkpoint(self) -> None:
        if os.path.exists(self.cfg.checkpoint_name):
            self.load_state_dict(torch.load(self.cfg.checkpoint_name, map_location=self.device))

    # ---------- embeddings & clustering ----------
    @torch.no_grad()
    def get_embeddings(self) -> np.ndarray:
        self.eval()
        emb_list = [self.encoders[m].get_embeddings(self.features[m]) for m in self.modality_names]
        fused = self.fusion_layer(emb_list)
        return fused.detach().cpu().numpy()

    def cluster_kmeans(self, embeddings: Optional[np.ndarray] = None, n_clusters: Optional[int] = None) -> np.ndarray:
        if embeddings is None:
            embeddings = self.get_embeddings()
        if n_clusters is None:
            n_clusters = self.cfg.default_kmeans_k
        return KMeans(n_clusters=n_clusters, random_state=100).fit_predict(embeddings)

    def cluster_leiden(self, n_neighbors: int = 15, resolution: float = 1.0) -> np.ndarray:
        emb = self.get_embeddings()
        adata = anndata.AnnData(X=emb)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.leiden(adata, resolution=resolution)
        return np.array(adata.obs['leiden'].astype(int))

    def cluster_hdbscan(self, min_cluster_size: int = 10) -> np.ndarray:
        import hdbscan
        emb = self.get_embeddings()
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb)

    def cluster_spectral(self, n_clusters: int = 10) -> np.ndarray:
        from sklearn.cluster import SpectralClustering
        emb = self.get_embeddings()
        return SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit_predict(emb)

    def cluster_gaussian(self, n_components: int = 10) -> np.ndarray:
        from sklearn.mixture import GaussianMixture
        emb = self.get_embeddings()
        return GaussianMixture(n_components=n_components, random_state=100).fit_predict(emb)

    def cluster_umap_hdbscan(self, n_neighbors: int = 15, min_cluster_size: int = 10) -> np.ndarray:
        import umap
        import hdbscan
        emb = self.get_embeddings()
        reduced = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, metric='cosine').fit_transform(emb)
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)


# ---------------------- Example ----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Dummy data
    features_dict = {
        'rna': np.random.rand(100, 19751),
        'image': np.random.rand(100, 1536),
        'protein': np.random.rand(100, 131),
    }
    spatial_coords = np.random.rand(100, 2)

    # Choose preset to reproduce a historical version:
    cfg = GSMOConfig.presets("v1")  # or "v2", "v3"
    model = GSMO(features_dict, spatial_coords=spatial_coords, device=device, cfg=cfg)
    model.train_model()
    np.save('emb_multimodal.npy', model.get_embeddings())

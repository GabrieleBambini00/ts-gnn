"""
Full TS-GNN (Temporal Sheaf Graph Neural Network) Architecture.

Phase 2D: Assembles sheaf diffusion + GRU temporal evolution + FiLM conditioning
into the complete model that maps (temporal_graphs, allele_embedding) -> predictions.

Architecture flow at each time step t:
1. Project input features to stalk space: x = proj(X(t))
2. Evolve restriction maps via GRU: R(t+1) = GRU(R(t), context)
3. Apply FiLM conditioning from allele: R'(t+1) = gamma * R(t+1) + beta
4. Compute connection Laplacian: L_F = sheaf_laplacian(R'(t+1))
5. Sheaf diffusion (multiple steps): x = diffuse(x, L_F)
6. Project back to expression space: X_hat(t+1) = proj_out(x)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt

from .sheaf import SheafDiffusionLayer
from .sheaf_vectorized import VectorizedSheafDiffusion, NormalizedVectorizedSheafDiffusion
from .temporal import TemporalRestrictionEvolution
from .film import AlleleFiLM, ZeroFiLM

logger = logging.getLogger(__name__)


class TSGNN(nn.Module):
    """
    Temporal Sheaf Graph Neural Network for allele-conditioned GRN rewiring.

    Args:
        num_nodes: N, number of genes/TFs.
        num_edges: E, number of regulatory edges.
        stalk_dim: d, sheaf stalk dimension.
        input_dim: Dimension of input gene features.
        esm_dim: Dimension of ESM-2 allele embeddings.
        conditioning_dim: FiLM conditioning latent dimension.
        edge_index: (2, E) directed edge tensor.
        num_diffusion_steps: Sheaf diffusion iterations per time step.
        use_allele_conditioning: If False, disables FiLM (ablation).
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        input_dim: int,
        esm_dim: int,
        conditioning_dim: int,
        edge_index: torch.Tensor,
        num_diffusion_steps: int = 3,
        use_allele_conditioning: bool = True,
        use_vectorized: bool = True,
        use_gradient_checkpointing: bool = False,
        use_normalized_laplacian: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.d = stalk_dim
        self.input_dim = input_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.use_allele_conditioning = use_allele_conditioning
        self.use_vectorized = use_vectorized
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_normalized_laplacian = use_normalized_laplacian

        self.register_buffer("edge_index", edge_index)

        # Input projection: gene expression -> stalk space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, stalk_dim * 2),
            nn.ReLU(),
            nn.Linear(stalk_dim * 2, stalk_dim),
        )

        # Sheaf diffusion layer — normalized variant keeps eigenvalues in [0,2]
        # which improves conditioning and gradient flow (Bodnar et al. 2022)
        if use_vectorized and use_normalized_laplacian:
            self.sheaf_layer = NormalizedVectorizedSheafDiffusion(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        elif use_vectorized:
            self.sheaf_layer = VectorizedSheafDiffusion(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )
        else:
            self.sheaf_layer = SheafDiffusionLayer(
                num_nodes=num_nodes,
                num_edges=num_edges,
                stalk_dim=stalk_dim,
                edge_index=edge_index,
            )

        # Temporal evolution of restriction maps
        self.temporal_evolution = TemporalRestrictionEvolution(
            stalk_dim=stalk_dim,
        )

        # FiLM allele conditioning
        modulation_dim = 2 * stalk_dim * stalk_dim
        if use_allele_conditioning:
            self.allele_film = AlleleFiLM(
                esm_dim=esm_dim,
                conditioning_dim=conditioning_dim,
                modulation_dim=modulation_dim,
            )
        else:
            self.allele_film = ZeroFiLM()

        # Output projection: stalk space -> expression prediction
        self.output_proj = nn.Sequential(
            nn.Linear(stalk_dim, stalk_dim * 2),
            nn.ReLU(),
            nn.Linear(stalk_dim * 2, input_dim),
        )

        # Initial restriction maps (learned starting point)
        self.initial_maps = nn.Parameter(
            torch.empty(num_edges, 2, stalk_dim, stalk_dim)
        )
        nn.init.orthogonal_(self.initial_maps.data.reshape(-1, stalk_dim, stalk_dim)
                            .reshape(num_edges * 2, stalk_dim, stalk_dim))
        self.initial_maps.data = self.initial_maps.data.reshape(num_edges, 2, stalk_dim, stalk_dim)

    def forward(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
        edge_weights_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass over the temporal graph sequence.

        Args:
            node_features_seq: (K, N, input_dim) temporal node features.
            allele_embedding: (esm_dim,) or (1, esm_dim) ESM-2 embedding.
            edge_weights_seq: Optional (K, E) temporal edge weights.

        Returns:
            predictions: List of K tensors, each (N, input_dim).
            restriction_maps_trajectory: List of K tensors, each (E, 2, d, d).
            sheaf_laplacians: List of K tensors, each (N*d, N*d).
        """
        K = node_features_seq.shape[0]
        predictions = []
        maps_trajectory = []
        laplacians = []

        # Start with initial restriction maps
        current_maps = self.initial_maps.clone()

        def _step(current_maps_inner, node_feat_t, allele_emb_inner):
            # 1. Project input features to stalk space
            x = self.input_proj(node_feat_t)  # (N, d)

            # 2. Evolve restriction maps via GRU
            new_maps = self.temporal_evolution(
                current_maps_inner, x, self.edge_index
            )

            # 3. Apply FiLM allele conditioning
            E, d = new_maps.shape[0], self.d
            maps_flat = new_maps.reshape(E, -1)  # (E, 2*d*d)
            maps_modulated = self.allele_film(allele_emb_inner, maps_flat)  # (E, 2*d*d)
            cmaps = maps_modulated.reshape(E, 2, d, d)

            # 4. Compute connection Laplacian
            L_F = self.sheaf_layer.compute_connection_laplacian(cmaps)

            # 5. Sheaf diffusion
            for _ in range(self.num_diffusion_steps):
                x = self.sheaf_layer.diffuse(x, L_F)

            # 6. Project back to expression space
            x_pred = self.output_proj(x)  # (N, input_dim)
            return x_pred, cmaps, L_F

        for t in range(K):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory on long sequences
                x_pred, current_maps, L_F = grad_ckpt(
                    _step, current_maps, node_features_seq[t], allele_embedding,
                    use_reentrant=False
                )
            else:
                x_pred, current_maps, L_F = _step(
                    current_maps, node_features_seq[t], allele_embedding
                )

            maps_trajectory.append(current_maps.detach().clone())
            laplacians.append(L_F)
            predictions.append(x_pred)

        return predictions, maps_trajectory, laplacians

    def decompose_rewiring(
        self,
        node_features_seq: torch.Tensor,
        allele_embedding: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Decompose rewiring into context + allele-specific components.

        L_F(t) = L_F^context(t) + Delta_L_F^allele(t)

        Runs forward twice:
        1. With allele embedding (full model)
        2. With zero allele embedding (context only)

        Returns:
            context_laplacians: List of K Laplacians from context-only model.
            allele_laplacians: List of K full Laplacians.
            delta_laplacians: List of K (allele - context) perturbation Laplacians.
        """
        # Full model
        _, _, full_laplacians = self.forward(node_features_seq, allele_embedding)

        # Context only (zero allele embedding)
        zero_allele = torch.zeros_like(allele_embedding)
        _, _, context_laplacians = self.forward(node_features_seq, zero_allele)

        # Delta: allele-specific perturbation
        delta_laplacians = [
            full_L - ctx_L
            for full_L, ctx_L in zip(full_laplacians, context_laplacians)
        ]

        return context_laplacians, full_laplacians, delta_laplacians

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per module."""
        counts = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
        counts["initial_maps"] = self.initial_maps.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


def create_tsgnn_from_config(config: dict, edge_index: torch.Tensor) -> TSGNN:
    """Factory function to create TS-GNN from a config dict."""
    # num_nodes = number of genes (N), distinct from input_dim (feature size per gene)
    num_nodes = config.get("data", {}).get("n_genes", config["model"]["input_dim"])
    model_cfg = config.get("model", {})
    return TSGNN(
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
        stalk_dim=model_cfg["stalk_dim"],
        input_dim=model_cfg["input_dim"],
        esm_dim=model_cfg["esm_dim"],
        conditioning_dim=model_cfg["conditioning_dim"],
        edge_index=edge_index,
        num_diffusion_steps=model_cfg["num_diffusion_steps"],
        use_allele_conditioning=True,
        use_gradient_checkpointing=model_cfg.get(
            "use_gradient_checkpointing", model_cfg["stalk_dim"] >= 8
        ),
        use_normalized_laplacian=model_cfg.get("use_normalized_laplacian", True),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Test with synthetic data
    N, E, d, input_dim, K = 50, 200, 4, 50, 10
    esm_dim = 1280

    edge_index = torch.randint(0, N, (2, E))
    model = TSGNN(
        num_nodes=N,
        num_edges=E,
        stalk_dim=d,
        input_dim=input_dim,
        esm_dim=esm_dim,
        conditioning_dim=128,
        edge_index=edge_index,
        num_diffusion_steps=3,
    )

    # Synthetic temporal data
    node_features_seq = torch.randn(K, N, input_dim)
    allele_emb = torch.randn(esm_dim)

    predictions, maps_traj, laplacians = model(node_features_seq, allele_emb)

    print(f"Model created: {N} nodes, {E} edges, d={d}, K={K}")
    print(f"Predictions: {len(predictions)} steps, each {predictions[0].shape}")
    print(f"Maps trajectory: {len(maps_traj)} steps, each {maps_traj[0].shape}")
    print(f"Laplacians: {len(laplacians)} steps, each {laplacians[0].shape}")

    params = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Test decomposition
    ctx_L, full_L, delta_L = model.decompose_rewiring(node_features_seq, allele_emb)
    print(f"\nRewiring decomposition:")
    print(f"  Context Laplacian norm: {ctx_L[0].norm():.4f}")
    print(f"  Full Laplacian norm: {full_L[0].norm():.4f}")
    print(f"  Delta (allele-specific) norm: {delta_L[0].norm():.4f}")

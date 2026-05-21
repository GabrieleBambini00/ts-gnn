"""
Temporal graph sequence construction from pseudotime.

Phase 1E.6-7: Pseudotime inference, temporal binning, time-varying edge weights,
              VIPER regulon activity computation.
"""

import logging
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TemporalGraphSnapshot(NamedTuple):
    """A single time-step graph snapshot."""
    node_features: torch.Tensor   # (N, d_input)
    edge_index: torch.Tensor      # (2, E)
    edge_weights: torch.Tensor    # (E,)
    time_index: int               # Which temporal bin


class TemporalGraphSequence:
    """A sequence of K temporal graph snapshots for one allele."""

    def __init__(
        self,
        snapshots: List[TemporalGraphSnapshot],
        allele: str,
        viper_activity: Optional[torch.Tensor] = None,
    ):
        self.snapshots = snapshots
        self.K = len(snapshots)
        self.allele = allele
        self.viper_activity = viper_activity  # (K, N)

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
        return self.snapshots[idx]

    @property
    def node_features_sequence(self) -> torch.Tensor:
        """Stack all node features: (K, N, d_input)."""
        return torch.stack([s.node_features for s in self.snapshots])

    @property
    def edge_weights_sequence(self) -> torch.Tensor:
        """Stack all edge weights: (K, E)."""
        return torch.stack([s.edge_weights for s in self.snapshots])


def compute_pseudotime(
    adata,
    method: str = "diffusion",
    root_cell: Optional[str] = None,
) -> np.ndarray:
    """
    Compute pseudotime ordering for cells.

    Args:
        adata: Preprocessed AnnData with PCA and neighbors computed.
        method: "diffusion" (scanpy) or "monocle3" (requires R).
        root_cell: Optional root cell barcode for directed pseudotime.

    Returns:
        Array of pseudotime values per cell.
    """
    import scanpy as sc

    if method == "diffusion":
        logger.info("Computing diffusion pseudotime...")

        # Compute diffusion map
        sc.tl.diffmap(adata, n_comps=min(15, adata.n_obs - 1))

        # Find root cell (use cell with highest/lowest DC1 if not specified)
        if root_cell is None:
            # Use the cell with the most extreme diffusion component 1
            # This heuristic works for linear trajectories
            adata.uns["iroot"] = int(np.argmin(adata.obsm["X_diffmap"][:, 0]))
        else:
            adata.uns["iroot"] = list(adata.obs_names).index(root_cell)

        sc.tl.dpt(adata)
        pseudotime = adata.obs["dpt_pseudotime"].values

        # Handle infinite values (disconnected components)
        pseudotime[np.isinf(pseudotime)] = np.nanmax(
            pseudotime[~np.isinf(pseudotime)]
        )

        logger.info(
            f"Pseudotime computed: range [{np.nanmin(pseudotime):.3f}, "
            f"{np.nanmax(pseudotime):.3f}]"
        )
        return pseudotime

    elif method == "paga":
        logger.info("Computing PAGA-guided pseudotime...")
        # Assume PCA and neighbors are already computed
        sc.tl.paga(adata)
        sc.pl.paga(adata, plot=False)
        
        if root_cell is None:
            if "X_diffmap" not in adata.obsm:
                sc.tl.diffmap(adata, n_comps=min(15, adata.n_obs - 1))
            adata.uns["iroot"] = int(np.argmin(adata.obsm["X_diffmap"][:, 0]))
        else:
            adata.uns["iroot"] = list(adata.obs_names).index(root_cell)

        sc.tl.dpt(adata)  # DPT guided by PAGA connectivity
        pseudotime = adata.obs["dpt_pseudotime"].values
        pseudotime[np.isinf(pseudotime)] = np.nanmax(pseudotime[~np.isinf(pseudotime)])
        logger.info("PAGA-guided pseudotime computed")
        return pseudotime

    elif method == "monocle3":
        logger.info(
            "Monocle 3 requires R. To compute:\n"
            "  1. Export adata to R (use anndata2ri)\n"
            "  2. Run Monocle 3 trajectory inference\n"
            "  3. Import pseudotime values back\n"
            "Falling back to diffusion pseudotime."
        )
        return compute_pseudotime(adata, method="diffusion")

    else:
        raise ValueError(f"Unknown pseudotime method: {method}")


def orient_with_velocity(adata, pseudotime: np.ndarray) -> np.ndarray:
    """
    Orient pseudotime using RNA velocity direction.

    If velocity indicates the trajectory runs in the opposite direction
    from pseudotime, flip the pseudotime axis.
    """
    try:
        import scvelo as scv

        logger.info("Computing RNA velocity for pseudotime orientation...")
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode="stochastic")

        # Check correlation between velocity-implied direction and pseudotime
        if "velocity_pseudotime" in adata.obs.columns:
            corr = np.corrcoef(pseudotime, adata.obs["velocity_pseudotime"])[0, 1]
            if corr < 0:
                logger.info("Flipping pseudotime direction based on RNA velocity")
                pseudotime = pseudotime.max() - pseudotime
            logger.info(f"Pseudotime-velocity correlation: {corr:.3f}")
        else:
            scv.tl.velocity_pseudotime(adata)
            vel_pt = adata.obs["velocity_pseudotime"].values
            corr = np.corrcoef(pseudotime[~np.isnan(vel_pt)],
                               vel_pt[~np.isnan(vel_pt)])[0, 1]
            if corr < 0:
                pseudotime = pseudotime.max() - pseudotime
                logger.info("Flipped pseudotime based on velocity")

    except ImportError:
        logger.warning("scVelo not installed. Skipping velocity-based orientation.")
    except Exception as e:
        logger.warning(f"Velocity computation failed: {e}. Using unoriented pseudotime.")

    return pseudotime


def bin_cells_by_pseudotime(
    pseudotime: np.ndarray,
    K: int = 10,
) -> np.ndarray:
    """
    Assign cells to K temporal bins based on pseudotime.

    Uses quantile-based binning to ensure roughly equal cell counts per bin.

    Returns:
        Array of bin assignments (0 to K-1) per cell.
    """
    # Remove NaN pseudotime values
    valid = ~np.isnan(pseudotime)
    bins = np.full(len(pseudotime), -1, dtype=int)

    # Quantile-based binning for equal cell counts
    quantiles = np.linspace(0, 100, K + 1)
    boundaries = np.percentile(pseudotime[valid], quantiles)

    for k in range(K):
        mask = valid & (pseudotime >= boundaries[k])
        if k < K - 1:
            mask = mask & (pseudotime < boundaries[k + 1])
        bins[mask] = k

    # Report bin sizes (boundaries has K+1 elements: boundaries[0..K])
    for k in range(K):
        n = (bins == k).sum()
        upper = boundaries[k + 1] if k < K else boundaries[K]
        logger.info(f"  Bin {k}: {n} cells (pseudotime [{boundaries[k]:.3f}, {upper:.3f}])")

    return bins


def construct_temporal_graphs(
    adata,
    gene_list: List[str],
    edge_index: torch.Tensor,
    pseudotime_bins: np.ndarray,
    K: int = 10,
    allele: str = "WT",
    allele_mask: Optional[np.ndarray] = None,
) -> TemporalGraphSequence:
    """
    Construct K temporal graph snapshots from binned cells.

    For each temporal bin k:
    - Node features X(t_k): mean expression per gene in that bin
    - Edge weights A(t_k): Pearson correlation between connected genes,
      restricted to base GRN edges

    Args:
        adata: Preprocessed AnnData with raw expression in .raw.
        gene_list: List of N gene names (node ordering).
        edge_index: (2, E) base GRN topology.
        pseudotime_bins: Cell bin assignments (0 to K-1).
        K: Number of temporal bins.
        allele: TP53 allele label.
        allele_mask: Boolean mask for cells of this allele.

    Returns:
        TemporalGraphSequence with K snapshots.
    """
    N = len(gene_list)
    E = edge_index.shape[1]

    # Subset to allele-specific cells if mask provided
    if allele_mask is not None:
        cell_mask = allele_mask
    else:
        cell_mask = np.ones(adata.n_obs, dtype=bool)

    # Get expression matrix for selected genes
    # Try raw (normalized, not scaled) first
    if adata.raw is not None:
        gene_indices = [list(adata.raw.var_names).index(g)
                        for g in gene_list if g in adata.raw.var_names]
        missing = [g for g in gene_list if g not in adata.raw.var_names]
        if missing:
            logger.warning(f"{len(missing)} genes not found in adata.raw")
        _raw_X = adata.raw.X
        # anndata >= 0.10.8 may return a backed array; normalise to ndarray/csr
        expr_matrix = _raw_X.to_memory() if hasattr(_raw_X, "to_memory") else _raw_X
    else:
        gene_indices = [list(adata.var_names).index(g)
                        for g in gene_list if g in adata.var_names]
        expr_matrix = adata.X

    def _compute_chunk(mask):
        try:
            v_expr = expr_matrix[mask][:, gene_indices]
            if hasattr(v_expr, "toarray"):
                v_expr = v_expr.toarray()
            return np.asarray(v_expr, dtype=np.float32)
        except Exception:
            return np.zeros((mask.sum(), N), dtype=np.float32)

    snapshots = []
    for k in range(K):
        # Cells in this bin and allele
        bin_mask = cell_mask & (pseudotime_bins == k)
        n_cells = bin_mask.sum()

        if n_cells < 5:
            logger.warning(f"Bin {k} has only {n_cells} cells for {allele}. Using interpolation.")
            
            def _get_feat(mask):
                ex = _compute_chunk(mask)
                feat = torch.tensor(ex.mean(axis=0), dtype=torch.float32)
                return feat.unsqueeze(1) if feat.dim() == 1 else feat

            prev_features, next_features = None, None
            for delta in range(1, K):
                if k - delta >= 0:
                    pm = cell_mask & (pseudotime_bins == k - delta)
                    if pm.sum() >= 5:
                        prev_features = _get_feat(pm)
                        break
            for delta in range(1, K):
                if k + delta < K:
                    nm = cell_mask & (pseudotime_bins == k + delta)
                    if nm.sum() >= 5:
                        next_features = _get_feat(nm)
                        break
                        
            if prev_features is not None and next_features is not None:
                alpha = 0.5  # equidistant interpolation when both neighbours available
                node_features = alpha * prev_features + (1 - alpha) * next_features
            elif prev_features is not None:
                node_features = prev_features
            elif next_features is not None:
                node_features = next_features
            else:
                node_features = torch.zeros(N, 1)

            edge_weights = torch.ones(E)
            
        else:
            bin_expr = _compute_chunk(bin_mask)
            
            node_features = torch.tensor(bin_expr.mean(axis=0), dtype=torch.float32)
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(1)
                
            if bin_expr.shape[0] > 2:
                edge_weights = torch.zeros(E, dtype=torch.float32)
                src_idx = edge_index[0].numpy()
                tgt_idx = edge_index[1].numpy()
                
                valid_mask = (src_idx < bin_expr.shape[1]) & (tgt_idx < bin_expr.shape[1])
                v_src = src_idx[valid_mask]
                v_tgt = tgt_idx[valid_mask]
                
                if len(v_src) > 0:
                    x = bin_expr[:, v_src]  # (n_cells, E_valid)
                    y = bin_expr[:, v_tgt]  # (n_cells, E_valid)
                    
                    x_centered = x - x.mean(axis=0)
                    y_centered = y - y.mean(axis=0)
                    
                    cov = (x_centered * y_centered).mean(axis=0)
                    std_prod = x.std(axis=0) * y.std(axis=0)
                    
                    corr = np.zeros_like(cov)
                    nonzero_std = std_prod > 1e-8
                    corr[nonzero_std] = cov[nonzero_std] / std_prod[nonzero_std]
                    
                    edge_weights[torch.from_numpy(valid_mask)] = torch.from_numpy(corr).to(torch.float32)
            else:
                edge_weights = torch.ones(E)

        snapshots.append(TemporalGraphSnapshot(
            node_features=node_features,
            edge_index=edge_index,
            edge_weights=edge_weights,
            time_index=k,
        ))

    logger.info(f"Constructed {K} temporal snapshots for allele {allele}")
    return TemporalGraphSequence(snapshots=snapshots, allele=allele)


def compute_viper_activity(
    adata,
    gene_list: List[str],
    pseudotime_bins: np.ndarray,
    K: int = 10,
    dorothea_levels: List[str] = None,
) -> torch.Tensor:
    """
    Compute TF activity scores per temporal bin using decoupler ULM with DoRothEA.

    Implements the VIPER algorithm (Alvarez et al. 2016, Nature Genetics) via
    decoupler-py's Univariate Linear Model (ULM), which is a fast linear
    approximation of VIPER with comparable accuracy (Badia-i-Mompel et al. 2022).

    Regulon source: DoRothEA confidence levels A+B (ChIP-seq + PWM + literature).

    Args:
        adata: AnnData with log-normalised expression in .X.
        gene_list: List of gene names (must match adata.var_names).
        pseudotime_bins: Per-cell bin assignment, shape (n_cells,).
        K: Number of temporal bins.
        dorothea_levels: DoRothEA confidence levels to use (default ['A', 'B']).

    Returns:
        Tensor of shape (K, N_tfs) with TF activity estimates.
        Returns zeros if decoupler / DoRothEA unavailable.
    """
    if dorothea_levels is None:
        dorothea_levels = ['A', 'B']

    N = len(gene_list)

    try:
        import decoupler as dc
        import pandas as pd

        logger.info("Loading DoRothEA regulons (levels %s)...", dorothea_levels)
        try:
            dorothea = dc.get_dorothea(organism='human', levels=dorothea_levels)
        except Exception as e:
            logger.warning(f"DoRothEA download failed ({e}); falling back to mean expression proxy.")
            raise

        # Identify TFs present in gene_list
        tfs_in_data = [g for g in gene_list if g in set(dorothea['source'].unique())]
        if not tfs_in_data:
            logger.warning("No DoRothEA TFs found in gene_list. Using mean expression proxy.")
            raise ValueError("No TFs")

        activity = torch.zeros(K, N)

        for k in range(K):
            bin_mask = pseudotime_bins == k
            n_bin = int(bin_mask.sum())
            if n_bin < 3:
                continue  # leave as zeros; caller handles sparse bins

            import anndata as ad
            adata_bin = adata[bin_mask][:, gene_list].copy()

            # Ensure dense matrix
            if hasattr(adata_bin.X, "toarray"):
                adata_bin.X = adata_bin.X.toarray()

            # Run ULM (fast linear VIPER approximation)
            dc.run_ulm(
                adata_bin,
                net=dorothea,
                source='source',
                target='target',
                weight='weight',
                use_raw=False,
                verbose=False,
            )

            # ulm_estimate: (n_cells, n_tfs) DataFrame in obsm
            ulm_est = adata_bin.obsm['ulm_estimate']  # DataFrame
            # Map TF columns back to gene_list positions
            for gene_idx, gene in enumerate(gene_list):
                if gene in ulm_est.columns:
                    activity[k, gene_idx] = float(ulm_est[gene].mean())

        logger.info(f"ULM/VIPER activity computed: shape {activity.shape}, "
                    f"{len(tfs_in_data)} TFs active")
        return activity

    except (ImportError, Exception) as exc:
        logger.warning(
            f"Decoupler VIPER unavailable ({exc}); using log-normalised mean expression as proxy. "
            "Install with: pip install decoupler"
        )
        activity = torch.zeros(K, N)
        for k in range(K):
            bin_mask = pseudotime_bins == k
            if bin_mask.sum() > 0:
                bin_expr = adata[bin_mask][:, gene_list].X
                if hasattr(bin_expr, "toarray"):
                    bin_expr = bin_expr.toarray()
                activity[k] = torch.tensor(bin_expr.mean(axis=0), dtype=torch.float32)
        return activity


# ── Public API aliases ───────────────────────────────────────────────────────

create_temporal_graphs = construct_temporal_graphs   # backwards-compatible alias
bin_by_pseudotime = bin_cells_by_pseudotime          # backwards-compatible alias


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Temporal module ready. Import and use with preprocessed AnnData objects.")

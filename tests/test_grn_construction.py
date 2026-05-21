"""Tests for GRN construction module."""
import numpy as np
import pytest
import torch
from tsgnn.data.grn_construction import (
    compute_spearman_weights,
    construct_base_grn,
    _generate_synthetic_grn,
)


def test_construct_base_grn_raises_without_data():
    """construct_base_grn must raise RuntimeError if no regulatory databases are found."""
    gene_list = [f"GENE_{i}" for i in range(50)]
    gene_list[0] = "TP53"
    with pytest.raises(RuntimeError, match="No prior regulatory edges"):
        construct_base_grn(gene_list)


def test_synthetic_grn_dimensions():
    """_generate_synthetic_grn returns correct edge format and no self-loops."""
    N = 50
    target_edges = 200
    edges = _generate_synthetic_grn(N, target_edges=target_edges)
    assert isinstance(edges, dict)
    assert len(edges) > 0
    # No self-loops
    for (src, tgt) in edges:
        assert src != tgt, f"Self-loop at node {src}"
    # All indices in valid range
    for (src, tgt) in edges:
        assert 0 <= src < N and 0 <= tgt < N


def test_no_self_loops():
    """Synthetic GRN must not contain self-loops."""
    gene_list = [f"GENE_{i}" for i in range(20)]
    edges = _generate_synthetic_grn(len(gene_list), target_edges=100)
    for (src, tgt) in edges:
        assert src != tgt, "Self-loop found in GRN"


# ---------------------------------------------------------------------------
# Task 2.1 — A = M ⊙ R biological coherence tests
# ---------------------------------------------------------------------------

def _make_test_fixtures(n_cells: int = 80, rng_seed: int = 0):
    """
    Build a synthetic 4-gene setup with KNOWN mask and KNOWN correlations.

    Genes: G0, G1, G2, G3
    Mask M (prior edges):
        G0 → G1 (allowed, strongly positive correlation)
        G0 → G2 (allowed, strongly negative correlation — repressor)
        G1 → G3 (allowed, weak correlation)
        G0 → G3 NOT in mask  ← must be zero in A

    Expression layout (n_cells × 4):
        G0: random signal
        G1: +0.95 * G0  (strong positive, rho ≈ +0.95)
        G2: -0.95 * G0  (strong negative, rho ≈ -0.95)
        G3:  0.10 * G0  (weak positive,  rho ≈ +0.10)
    """
    rng = np.random.default_rng(rng_seed)
    base = rng.standard_normal(n_cells).astype(np.float32)
    noise = rng.standard_normal((n_cells, 4)).astype(np.float32) * 0.05

    expr = np.column_stack([
        base,
        0.95 * base + noise[:, 1],   # G1 ≈ +0.95 corr with G0
        -0.95 * base + noise[:, 2],  # G2 ≈ -0.95 corr with G0
        0.10 * base + noise[:, 3],   # G3 ≈ +0.10 corr with G0
    ])  # shape (n_cells, 4)

    # Mask M: edges (0→1), (0→2), (1→3)  — G0→G3 deliberately ABSENT
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
    gene_list = ["G0", "G1", "G2", "G3"]
    return expr, edge_index, gene_list


def test_spearman_output_zero_outside_mask():
    """A = M ⊙ R: entries for gene-pairs NOT in M must be exactly zero.

    The mask has 3 edges (0→1, 0→2, 1→3).  The pair 0→3 is NOT in the mask,
    so it must not appear in the output weight tensor at all (it is absent from
    edge_index, hence absent from the weight vector).  We verify indirectly by
    checking the weight for each edge is only non-zero for allowed pairs.
    """
    expr, edge_index, gene_list = _make_test_fixtures()
    weights = compute_spearman_weights(expr, edge_index, gene_list)

    # Shape matches number of prior edges
    assert weights.shape == (edge_index.shape[1],), (
        f"Expected {edge_index.shape[1]} weights, got {weights.shape}"
    )

    # Build dense 4×4 adjacency from weights — only masked positions filled
    n = 4
    A_dense = torch.zeros(n, n)
    for k in range(edge_index.shape[1]):
        src = edge_index[0, k].item()
        tgt = edge_index[1, k].item()
        A_dense[src, tgt] = weights[k]

    # 0→3 not in mask, must stay zero
    assert A_dense[0, 3].item() == 0.0, (
        f"A[G0,G3] should be 0 (not in mask), got {A_dense[0, 3].item()}"
    )
    # 3→0 not in mask, must stay zero
    assert A_dense[3, 0].item() == 0.0, (
        f"A[G3,G0] should be 0 (not in mask), got {A_dense[3, 0].item()}"
    )


def test_spearman_preserves_negative_correlations():
    """A = M ⊙ R: a strongly negatively-correlated allowed edge must yield a NEGATIVE weight.

    G2 = -0.95 * G0 ⟹ rho(G0, G2) ≈ -0.95.
    The prior mask includes edge 0→2, so A[0,2] must be negative (repressor preserved).
    """
    expr, edge_index, gene_list = _make_test_fixtures()
    weights = compute_spearman_weights(expr, edge_index, gene_list)

    # Edge index 1 = (0→2)
    w_repressor = weights[1].item()
    assert w_repressor < -0.80, (
        f"Expected strong negative weight for repressive edge (rho ≈ -0.95), got {w_repressor:.4f}. "
        "Negative correlations (repressors) must not be clipped or abs-valued."
    )


def test_spearman_magnitude_tracks_correlation_strength():
    """A = M ⊙ R: |A[strong edge]| > |A[weak edge]| within allowed edges.

    Edge 0→1 has rho ≈ +0.95 (strong).
    Edge 1→3 has rho ≈ +0.10 (weak, G3 barely correlated with G1).
    The stronger edge must have larger absolute weight.
    """
    expr, edge_index, gene_list = _make_test_fixtures()
    weights = compute_spearman_weights(expr, edge_index, gene_list)

    w_strong = abs(weights[0].item())   # edge 0: 0→1 strong
    w_weak = abs(weights[2].item())     # edge 2: 1→3 weak

    assert w_strong > w_weak, (
        f"Expected |A[0→1]|={w_strong:.4f} > |A[1→3]|={w_weak:.4f}. "
        "Edge magnitude should track correlation strength."
    )


def test_spearman_fallback_uses_prior_weights_without_expr():
    """Backward-compat: when expr_matrix=None, construct_base_grn uses fixed prior weights.

    We inject a minimal synthetic prior via scenic_regulons (bypasses real DB files)
    and verify that:
    - The call succeeds (no RuntimeError).
    - metadata['uses_spearman'] is False.
    - edge_weight is a float tensor with positive entries (the fixed 1.0 prior weight).
    """
    gene_list = ["TP53", "CDKN1A", "MDM2"]
    # Minimal SCENIC regulon: TP53 → CDKN1A and TP53 → MDM2
    scenic = {"TP53": ["CDKN1A", "MDM2"]}

    edge_index, edge_weight, meta = construct_base_grn(
        gene_list,
        scenic_regulons=scenic,
        expr_matrix=None,   # ← no expression data → fallback path
    )

    assert not meta["uses_spearman"], (
        "uses_spearman should be False when no expr_matrix is provided"
    )
    assert edge_weight.dtype == torch.float32
    assert (edge_weight > 0).all(), (
        "Prior weights should all be positive (fixed 1.0 SCENIC weight)"
    )

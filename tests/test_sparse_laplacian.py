"""
Task 3.1 — Sparse Sheaf Laplacian tests.

Tests:
  T1 — Numerical equivalence: sparse L_F @ x equals dense L_F @ x
       across stalk_dim ∈ {1, 2, 4} and multiple random graphs.
  T2 — Memory comparison: sparse stores fewer values than (Nd)² for a
       representative graph (N=500, d=4, realistic edge count).
  T3 — Differentiability: backward pass through sparse_diffusion produces
       finite gradients on the restriction maps.
  T4 — Sparse Laplacian symmetry: coalesced COO tensor is symmetric.
  T5 — forward(use_sparse=True) output equals forward(use_sparse=False).
"""

import pytest
import torch
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_layer(N: int, E: int, d: int, seed: int = 0) -> VectorizedSheafDiffusion:
    torch.manual_seed(seed)
    edge_index = torch.randint(0, N, (2, E))
    return VectorizedSheafDiffusion(N, E, d, edge_index)


def _random_maps(layer: VectorizedSheafDiffusion, seed: int = 99) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(layer.num_edges, 2, layer.d, layer.d)


# ── T1: Numerical equivalence ─────────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 4])
def test_sparse_matvec_equals_dense_matvec(d: int):
    """Sparse L_F @ x must equal dense L_F @ x, atol=1e-5, across stalk dims."""
    torch.manual_seed(42 + d)
    N, E = 20, 35
    layer = _make_layer(N, E, d, seed=42 + d)
    maps = _random_maps(layer, seed=7 + d)

    # Dense path
    L_dense = layer.compute_connection_laplacian_vectorized(maps)
    x = torch.randn(N * d)
    Lx_dense = L_dense @ x

    # Sparse path
    L_sparse = layer.compute_sparse_laplacian(maps)
    Lx_sparse = torch.sparse.mm(L_sparse, x.unsqueeze(-1)).squeeze(-1)

    max_diff = (Lx_dense - Lx_sparse).abs().max().item()
    assert torch.allclose(Lx_dense, Lx_sparse, atol=1e-5), (
        f"Sparse matvec != dense matvec for d={d}. Max diff: {max_diff:.2e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_matvec_multiple_graphs_d4(seed: int):
    """Numerical equivalence on multiple random graph topologies, d=4."""
    N, E, d = 15, 30, 4
    layer = _make_layer(N, E, d, seed=seed)
    maps = _random_maps(layer, seed=seed + 50)

    L_dense = layer.compute_connection_laplacian_vectorized(maps)
    x = torch.randn(N * d)
    Lx_dense = L_dense @ x

    L_sparse = layer.compute_sparse_laplacian(maps)
    Lx_sparse = torch.sparse.mm(L_sparse, x.unsqueeze(-1)).squeeze(-1)

    max_diff = (Lx_dense - Lx_sparse).abs().max().item()
    assert torch.allclose(Lx_dense, Lx_sparse, atol=1e-5), (
        f"Mismatch for graph seed={seed}, d={d}. Max diff: {max_diff:.2e}"
    )


# ── T2: Memory comparison ──────────────────────────────────────────────────────

def test_sparse_stores_fewer_values_than_dense():
    """Sparse representation stores < (Nd)² values for a sparse graph.

    Representative size: N=500, d=4 → dense has (2000)² = 4_000_000 elements.
    With E=3000 edges (realistic GRN sparsity ~0.024): sparse stores
    4 * E * d² = 4 * 3000 * 16 = 192_000 values (before deduplication).
    After coalesce(), unique (row,col) pairs ≤ 192_000.

    We assert strictly fewer than (Nd)² values and report the ratio.
    """
    N, d, E = 500, 4, 3000
    Nd = N * d
    dense_count = Nd * Nd  # 4_000_000

    layer = _make_layer(N, E, d, seed=123)
    maps = _random_maps(layer, seed=456)

    L_sparse = layer.compute_sparse_laplacian(maps)
    # .values() on a coalesced COO gives the deduplicated non-zero values
    sparse_count = L_sparse.values().numel()

    ratio = dense_count / sparse_count
    # Print concrete numbers for the test report
    print(
        f"\n[memory] N={N}, d={d}, E={E}: "
        f"dense={dense_count:,} elems ({dense_count*4/1e6:.1f} MB fp32), "
        f"sparse={sparse_count:,} elems ({sparse_count*4/1e6:.2f} MB fp32), "
        f"ratio={ratio:.1f}x"
    )

    assert sparse_count < dense_count, (
        f"Expected sparse ({sparse_count}) < dense ({dense_count})"
    )
    # Sanity bound: can't have more non-zeros than the four scatter slots
    assert sparse_count <= 4 * E * d * d, (
        f"After coalesce, {sparse_count} > 4*E*d²={4*E*d*d}"
    )


# ── T3: Differentiability ──────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 4])
def test_sparse_diffusion_gradients_finite(d: int):
    """Backward through sparse_diffusion produces finite gradients on restriction maps."""
    N, E = 10, 20
    torch.manual_seed(0)
    edge_index = torch.randint(0, N, (2, E))
    layer = VectorizedSheafDiffusion(N, E, d, edge_index)

    x = torch.randn(N, d, requires_grad=False)
    # restriction_maps already a leaf Parameter — gradients flow through it
    layer.restriction_maps.requires_grad_(True)

    L_sparse = layer.compute_sparse_laplacian()  # uses layer.restriction_maps
    x_out = layer.sparse_diffusion(x, L_sparse)

    loss = x_out.sum()
    loss.backward()

    grad = layer.restriction_maps.grad
    assert grad is not None, "No gradient on restriction_maps after sparse backward"
    assert torch.isfinite(grad).all(), (
        f"Non-finite gradients in restriction_maps for d={d}: "
        f"nan={grad.isnan().sum().item()}, inf={grad.isinf().sum().item()}"
    )


# ── T4: Sparse Laplacian symmetry ─────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 4])
def test_sparse_laplacian_symmetric(d: int):
    """Coalesced sparse Laplacian must equal its transpose (dense comparison)."""
    layer = _make_layer(15, 25, d, seed=5)
    maps = _random_maps(layer, seed=10)

    L_sparse = layer.compute_sparse_laplacian(maps)
    L_dense_from_sparse = L_sparse.to_dense()

    assert torch.allclose(L_dense_from_sparse, L_dense_from_sparse.T, atol=1e-5), (
        f"Sparse-derived dense L_F not symmetric for d={d}. "
        f"Max asym: {(L_dense_from_sparse - L_dense_from_sparse.T).abs().max():.2e}"
    )


# ── T5: forward(use_sparse=True) matches forward(use_sparse=False) ─────────────

@pytest.mark.parametrize("d", [2, 4])
def test_forward_sparse_matches_dense(d: int):
    """forward(use_sparse=True) output equals forward(use_sparse=False) for same input."""
    N, E = 12, 22
    torch.manual_seed(d)
    edge_index = torch.randint(0, N, (2, E))
    layer = VectorizedSheafDiffusion(N, E, d, edge_index)

    x = torch.randn(N, d)
    maps = _random_maps(layer, seed=d + 100)

    with torch.no_grad():
        out_dense, _ = layer.forward(x, restriction_maps=maps, num_steps=2, use_sparse=False)
        out_sparse, _ = layer.forward(x, restriction_maps=maps, num_steps=2, use_sparse=True)

    max_diff = (out_dense - out_sparse).abs().max().item()
    assert torch.allclose(out_dense, out_sparse, atol=1e-5), (
        f"forward(use_sparse=True) != forward(use_sparse=False) for d={d}. "
        f"Max diff: {max_diff:.2e}"
    )

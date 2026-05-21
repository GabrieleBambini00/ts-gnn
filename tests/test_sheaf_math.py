"""
Test 8.1 — Matematica del Laplaciano Sheaf.

Verifica:
- L_F è positive semi-definite (PSD)
- Implementazione vettorizzata == loop
- Per d=1 e mappe identità, L_F si riduce al Laplaciano del grafo
- Diffusione non esplode per d grande

Extended with five rigorous mathematical property tests (Task 1.2):
  P1 – Symmetry:                L_F == L_F^T
  P2 – Positive semi-definite:  all eigenvalues >= -1e-5
  P3 – Graph Laplacian reduction: identity restriction maps reproduce L_graph
  P4 – Vectorized == loop:      torch.allclose over stalk_dim in {1, 2, 4}
  P5 – Energy non-increasing:   sheaf Dirichlet energy E(x) = x^T L_F x does
       not increase under the linearised diffusion step x <- x - alpha * L_F x
       with alpha = 0.01 (chosen small enough that the gradient-flow property
       holds cleanly without any weight matrix or nonlinearity).
"""
import torch
import pytest
from tsgnn.utils import set_global_seed


# ── helpers ────────────────────────────────────────────────────────────────

def _make_edge_index(edges):
    """Lista di tuple (src, tgt) → (2, E) tensor."""
    src, tgt = zip(*edges)
    return torch.tensor([list(src), list(tgt)], dtype=torch.long)


def _import_layers(N, E, d, edge_index):
    from tsgnn.model.sheaf import SheafDiffusionLayer
    from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion
    loop = SheafDiffusionLayer(N, E, d, edge_index)
    vec  = VectorizedSheafDiffusion(N, E, d, edge_index)
    return loop, vec


# ── Test 1: PSD ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 4])
def test_laplacian_psd(d):
    """L_F deve avere tutti gli autovalori >= -epsilon (PSD)."""
    N, E = 5, 6
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)]
    edge_index = _make_edge_index(edges)
    _, vec = _import_layers(N, E, d, edge_index)

    torch.manual_seed(0)
    maps = torch.randn(E, 2, d, d)
    L = vec.compute_connection_laplacian_vectorized(maps)

    eigvals = torch.linalg.eigvalsh(L)
    min_eig = eigvals.min().item()
    assert min_eig >= -1e-4, (
        f"L_F non PSD per d={d}: autovalore minimo = {min_eig:.6f}"
    )


# ── Test 2: vectorized == loop ───────────────────────────────────────────────

@pytest.mark.parametrize("d", [1, 2, 4])
def test_vectorized_matches_loop(d):
    """P4 — Vectorized Laplacian equals loop Laplacian for stalk_dim in {1, 2, 4}.

    Property: compute_connection_laplacian_vectorized(maps) must equal
    compute_connection_laplacian(maps) (the loop reference) element-wise,
    up to atol=1e-5 floating-point tolerance.  Parametrized over stalk_dim
    {1, 2, 4} and two random graphs to ensure the scatter-add indexing is
    correct for every block size used in practice.
    """
    torch.manual_seed(42)
    N, E = 6, 8
    edge_index = torch.randint(0, N, (2, E))
    loop_layer, vec_layer = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    L_loop = loop_layer.compute_connection_laplacian(maps)
    L_vec  = vec_layer.compute_connection_laplacian_vectorized(maps)

    assert torch.allclose(L_loop, L_vec, atol=1e-5), (
        f"Mismatch loop vs vectorized per d={d}. "
        f"Max diff: {(L_loop - L_vec).abs().max():.2e}"
    )


# ── Test 3: riduzione a graph Laplacian ─────────────────────────────────────

def test_laplacian_reduces_to_graph_laplacian_d1():
    """For d=1 and identity restriction maps on a triangle, verify sheaf structure.

    Mathematical note: the sheaf Laplacian uses the formula
        L_F[v,v] = Σ_e F_{v,e}^T F_{v,e}     (diagonal blocks)
        L_F[u,v] = -F_{u,e}^T F_{v,e}          (off-diagonal blocks)

    For d=1, F = scalar. With 6 directed edges (each undirected edge split into
    two directed edges), each node v has degree 4 in the directed graph
    (2 out + 2 in), so L_F[v,v] = 4 and L_F[u,v] = -2 for adjacent u,v.
    This equals 2 * L_graph for the underlying undirected graph.

    We verify this factor-2 relationship explicitly.
    """
    # Triangle: every undirected edge represented as two directed edges
    edges = [(0,1),(1,0),(1,2),(2,1),(2,0),(0,2)]
    N, E, d = 3, 6, 1
    edge_index = _make_edge_index(edges)
    _, vec = _import_layers(N, E, d, edge_index)

    # Identity maps (scalar 1.0 for every directed arc)
    maps = torch.ones(E, 2, 1, 1)
    L_F = vec.compute_connection_laplacian_vectorized(maps)

    # Underlying undirected graph Laplacian (triangle: each node has degree 2)
    A = torch.zeros(N, N)
    for s, t in edges:
        A[s, t] = 1.0
    D = torch.diag(A.sum(dim=1))
    L_graph = D - A

    # With bidirectional directed edges, L_F = 2 * L_graph
    assert torch.allclose(L_F, 2.0 * L_graph, atol=1e-5), (
        f"Expected L_F = 2 * L_graph for bidirectional triangle with identity maps.\n"
        f"L_F:\n{L_F}\n2*L_graph:\n{2.0 * L_graph}"
    )



# ── Test 4: diffusione stabile ───────────────────────────────────────────────

@pytest.mark.parametrize("d", [2, 4, 8])
def test_diffusion_numerical_stability(d):
    """x dopo diffusione non deve esplodere (no NaN, no Inf, norma ragionevole)."""
    torch.manual_seed(7)
    N, E = 10, 15
    edge_index = torch.randint(0, N, (2, E))
    _, vec = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d) * 0.1   # piccole mappe per stabilità
    L_F  = vec.compute_connection_laplacian_vectorized(maps)
    x    = torch.randn(N, d)

    x_out = vec.diffuse(x, L_F)

    assert not torch.isnan(x_out).any(), "NaN dopo diffusione"
    assert not torch.isinf(x_out).any(), "Inf dopo diffusione"
    assert x_out.norm().item() < 1e4,    f"Norma esplosiva: {x_out.norm():.2e}"


# ── Test 5 (legacy): simmetria del Laplaciano ───────────────────────────────

def test_laplacian_symmetry():
    """P1 — L_F is symmetric (L = L^T) for arbitrary restriction maps.

    The construction guarantees symmetry: off-diagonal block (u,v) is
    -F_{u,e}^T F_{v,e} while block (v,u) is the transpose.  This test
    verifies symmetry on the vectorized implementation with random maps.
    """
    torch.manual_seed(1)
    N, E, d = 5, 7, 2
    edge_index = torch.randint(0, N, (2, E))
    _, vec = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    L = vec.compute_connection_laplacian_vectorized(maps)

    assert torch.allclose(L, L.T, atol=1e-5), (
        f"L_F non simmetrico. Max asimmetria: {(L - L.T).abs().max():.2e}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Task 1.2 — Five rigorous mathematical property tests
# ════════════════════════════════════════════════════════════════════════════

# ── P1 (strengthened): Symmetry for multiple random graphs ──────────────────

@pytest.mark.parametrize("seed,N,E,d", [
    (10, 5, 8, 1),
    (11, 8, 12, 2),
    (12, 6, 10, 4),
])
def test_p1_symmetry_arbitrary_maps(seed, N, E, d):
    """P1 — L_F is symmetric for arbitrary restriction maps, all stalk dims.

    Verifies torch.allclose(L, L.T, atol=1e-5) on both the loop and the
    vectorized implementations using different random graphs and stalk dims.
    """
    set_global_seed(seed)
    edge_index = torch.randint(0, N, (2, E))
    loop_layer, vec_layer = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)

    L_loop = loop_layer.compute_connection_laplacian(maps)
    assert torch.allclose(L_loop, L_loop.T, atol=1e-5), (
        f"Loop L_F not symmetric: seed={seed}, d={d}, "
        f"max_asym={(L_loop - L_loop.T).abs().max():.2e}"
    )

    L_vec = vec_layer.compute_connection_laplacian_vectorized(maps)
    assert torch.allclose(L_vec, L_vec.T, atol=1e-5), (
        f"Vectorized L_F not symmetric: seed={seed}, d={d}, "
        f"max_asym={(L_vec - L_vec.T).abs().max():.2e}"
    )


# ── P2 (strengthened): Positive semi-definiteness via eigvalsh ──────────────

@pytest.mark.parametrize("seed,d", [(20, 1), (21, 2), (22, 4)])
def test_p2_psd_eigvalsh(seed, d):
    """P2 — L_F is positive semi-definite: min eigenvalue >= -1e-5.

    The coboundary form L_F = B^T B guarantees PSD analytically; this test
    confirms it empirically via torch.linalg.eigvalsh on random restriction
    maps.  Tolerance -1e-5 accounts for floating-point rounding.
    """
    set_global_seed(seed)
    N, E = 7, 10
    edge_index = torch.randint(0, N, (2, E))
    _, vec = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    L = vec.compute_connection_laplacian_vectorized(maps)

    min_eig = torch.linalg.eigvalsh(L).min().item()
    assert min_eig >= -1e-5, (
        f"L_F not PSD: seed={seed}, d={d}, min_eigenvalue={min_eig:.6f}"
    )


# ── P3: Graph Laplacian reduction — loop implementation helper test ──────────

def test_p3_loop_reduces_to_graph_laplacian_via_helper():
    """P3 — verify_sheaf_reduces_to_graph_laplacian helper passes as pytest test.

    The module-level helper in sheaf.py checks that identity restriction maps
    reproduce the standard graph Laplacian.  This test wraps it as a proper
    pytest assertion so CI captures any regression.

    Implementation note: the helper uses networkx for the reference Laplacian;
    it asserts max block-diagonal error < 1e-5 for a cycle graph.
    """
    from tsgnn.model.sheaf import verify_sheaf_reduces_to_graph_laplacian
    # Runs for two (num_nodes, stalk_dim) pairs to cover d=2 and d=4.
    verify_sheaf_reduces_to_graph_laplacian(num_nodes=5, stalk_dim=2)
    verify_sheaf_reduces_to_graph_laplacian(num_nodes=6, stalk_dim=4)


def test_p3_vectorized_block_identity_structure():
    """P3 — with identity maps, each d×d off-diagonal block of L_F equals
    -(connectivity) * I_d, and diagonal blocks equal degree * I_d.

    For a simple path graph (directed, single direction per edge), node v's
    diagonal block must equal deg(v) * I_d.  The off-diagonal block (u,v)
    for adjacent u,v must equal -I_d.  This is the true block-identity
    reduction test for the vectorized implementation, beyond the d=1 scalar
    check already in test_laplacian_reduces_to_graph_laplacian_d1.
    """
    # Path: 0→1, 1→2, 2→3 — only forward edges (single direction).
    edges = [(0, 1), (1, 2), (2, 3)]
    N, E, d = 4, 3, 2
    edge_index = _make_edge_index(edges)
    _, vec = _import_layers(N, E, d, edge_index)

    # Identity restriction maps
    maps = torch.zeros(E, 2, d, d)
    for e in range(E):
        for s in range(2):
            maps[e, s] = torch.eye(d)

    L = vec.compute_connection_laplacian_vectorized(maps)

    I_d = torch.eye(d)

    # Expected degrees (directed: u contributes to diag_src, v to diag_tgt)
    # edge (0,1): node0 += I (src), node1 += I (tgt)
    # edge (1,2): node1 += I (src), node2 += I (tgt)
    # edge (2,3): node2 += I (src), node3 += I (tgt)
    expected_deg = {0: 1, 1: 2, 2: 2, 3: 1}  # total incident contributions

    for v, deg in expected_deg.items():
        block = L[v*d:(v+1)*d, v*d:(v+1)*d]
        assert torch.allclose(block, float(deg) * I_d, atol=1e-5), (
            f"Diagonal block for node {v}: expected {deg}*I, got\n{block}"
        )

    # Off-diagonal blocks for adjacent nodes should be -I (off-diag is -F_u^T F_v = -I^T I = -I)
    for u, v in edges:
        block_uv = L[u*d:(u+1)*d, v*d:(v+1)*d]
        block_vu = L[v*d:(v+1)*d, u*d:(u+1)*d]
        assert torch.allclose(block_uv, -I_d, atol=1e-5), (
            f"Off-diagonal block ({u},{v}): expected -I, got\n{block_uv}"
        )
        assert torch.allclose(block_vu, -I_d, atol=1e-5), (
            f"Off-diagonal block ({v},{u}): expected -I, got\n{block_vu}"
        )


# ── P4 (extended): Vectorized == loop for stalk_dim in {1, 2, 4} ─────────────
# (The existing test_vectorized_matches_loop already covers this after the
#  parametrize fix above — this additional test uses a second random graph
#  to improve coverage.)

@pytest.mark.parametrize("d", [1, 2, 4])
def test_p4_vectorized_equals_loop_second_graph(d):
    """P4 — Vectorized Laplacian equals loop Laplacian on a second random graph.

    Uses a different graph topology (denser, larger) than test_vectorized_matches_loop
    to confirm the scatter-add indexing is correct across topologies and block sizes.
    Tolerance: atol=1e-5 (single-precision floating point).
    """
    set_global_seed(100 + d)
    N, E = 10, 18
    edge_index = torch.randint(0, N, (2, E))
    loop_layer, vec_layer = _import_layers(N, E, d, edge_index)

    maps = torch.randn(E, 2, d, d)
    L_loop = loop_layer.compute_connection_laplacian(maps)
    L_vec = vec_layer.compute_connection_laplacian_vectorized(maps)

    max_diff = (L_loop - L_vec).abs().max().item()
    assert torch.allclose(L_loop, L_vec, atol=1e-5), (
        f"Vectorized != loop for d={d} on second graph. Max diff: {max_diff:.2e}"
    )


# ── P5: Energy non-increasing under linearised sheaf diffusion ───────────────

@pytest.mark.parametrize("seed,d", [(30, 1), (31, 2), (32, 4)])
def test_p5_energy_non_increasing(seed, d):
    """P5 — Sheaf Dirichlet energy E(x) = x^T L_F x does not increase under
    the linearised diffusion step x <- x - alpha * L_F_flat x.

    Mathematical justification
    --------------------------
    E(x') = (x - alpha * L x)^T L (x - alpha * L x)
           = x^T L x  - 2*alpha * x^T L^2 x + alpha^2 * x^T L^3 x
           = E(x) - alpha * x^T L^2 x * (2 - alpha * lambda_max)

    Since L is PSD, x^T L^2 x >= 0.  The bracketed factor is positive when
    alpha < 2 / lambda_max.  We choose alpha = 0.01 which is conservatively
    below this bound for reasonable random restriction maps.

    Note: we test the PURE linearised step (no weight matrix, no activation,
    no learned bias) because the production diffuse() in sheaf_vectorized.py
    applies a learned weight W and residual scaling alpha (itself a learnable
    parameter), which breaks strict monotonicity for arbitrary W.  The
    linearised step isolates the gradient-flow property of the Laplacian.
    """
    set_global_seed(seed)
    N, E = 8, 12
    edge_index = torch.randint(0, N, (2, E))
    _, vec = _import_layers(N, E, d, edge_index)

    # Small restriction maps so lambda_max stays well below 1/0.01 = 100
    maps = torch.randn(E, 2, d, d) * 0.3
    L = vec.compute_connection_laplacian_vectorized(maps)

    x = torch.randn(N, d)
    x_flat = x.reshape(-1)  # (N*d,)

    # Dirichlet energy before step
    energy_before = (x_flat @ L @ x_flat).item()

    # Linearised diffusion step: x <- x - alpha * L x
    alpha = 0.01  # conservative step; bound: alpha < 2/lambda_max
    x_flat_new = x_flat - alpha * (L @ x_flat)

    # Dirichlet energy after step
    energy_after = (x_flat_new @ L @ x_flat_new).item()

    assert energy_after <= energy_before + 1e-6, (
        f"Energy INCREASED under linearised diffusion: seed={seed}, d={d}, "
        f"E_before={energy_before:.6f}, E_after={energy_after:.6f}, "
        f"delta={energy_after - energy_before:.2e}.  "
        f"This indicates alpha={alpha} exceeds 2/lambda_max; reduce maps scale."
    )

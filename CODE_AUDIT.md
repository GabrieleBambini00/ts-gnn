# TS-GNN Code Audit Report
*Generated: 2026-04-05 — Full Codebase Review*

## Files Analyzed (21 source files, ~5,500 LOC)

| Module | File | LOC | Purpose |
|--------|------|-----|---------|
| **model/** | `sheaf.py` | 322 | Core Sheaf Diffusion Layer (Bodnar et al. NeurIPS 2022) |
| | `sheaf_vectorized.py` | 357 | O(E·d²) vectorized Laplacian (zero Python loops) |
| | `tsgnn.py` | 283 | Full TS-GNN assembly (Sheaf + GRU + FiLM) |
| | `temporal.py` | 123 | GRU evolution of restriction maps |
| | `film.py` | 152 | FiLM allele conditioning from ESM-2 |
| | `baselines.py` | 629 | 5 baselines (CellOracle, Dictys, EvolveGCN, No-Allele, TemporalGAT) |
| | `base_model.py` | ~100 | Base model abstract class |
| | `registry.py` | ~180 | Model registry/factory |
| **data/** | `download.py` | 416 | Dataset acquisition (GEO, Figshare, cBioPortal) |
| | `preprocess.py` | 248 | scRNA-seq QC, normalization, HVG, Harmony |
| | `grn_construction.py` | 253 | Multi-layer GRN from SCENIC + RegNetwork + TargetGeneReg + STRING |
| | `temporal.py` | 354 | Pseudotime binning, temporal graph snapshots, VIPER |
| | `allele.py` | 241 | TP53 allele sequences, ESM-2 embedding generation |
| | `allele_embeddings.py` | ~80 | Standalone embedding generator script |
| **training/** | `trainer.py` | 503 | Full training loop with early stopping, AMP, wandb |
| | `loss.py` | 301 | 4-term composite loss (L_expr + L_topo + L_regulon + L_sparse) |
| **evaluation/** | `metrics.py` | 497 | 5 evaluation metrics (zero-shot, enrichment, ChIP, DepMap, permutation) |
| | `ablation.py` | ~270 | Ablation study runner |
| | `benchmarks.py` | ~260 | Benchmark comparison table generator |
| **config** | `config.py` | 172 | Typed @dataclass config with cross-validation |
| **infra** | `configs/default.yaml` | 82 | Central hyperparameter config |

---

## Evaluation: 10 Parameters (Score 1–100)

### 1. Mathematical Correctness of Sheaf Laplacian — 95/100 ✅
**Strengths:**
- `sheaf.py` implements the *exact* Bodnar et al. NeurIPS 2022 formulation: L_F[u,v] = −F_{u,e}^T F_{v,e}
- Orthogonal initialization of restriction maps ensures well-conditioned starting matrix
- `sheaf_vectorized.py` proves correctness via `verify_vectorized_matches_loop()` against the reference implementation
- Spectral diagnostics (gap, condition number, PSD check) are built-in
- Identity maps correctly reduce to standard graph Laplacian (verified in `verify_sheaf_reduces_to_graph_laplacian`)

**Minor issue:** `sheaf.py` L160–L206 `compute_connection_laplacian_efficient` is a dead copy of the loop version, NOT actually efficient. It's misleading naming. The *real* vectorized version is in `sheaf_vectorized.py`.

### 2. Biological Coherence of GRN Construction — 72/100 ⚠️ BELOW THRESHOLD
**Strengths:**
- Multi-layer evidence integration (SCENIC > RegNetwork > TargetGeneReg > STRING) with correct priority ordering
- Signed edges: TargetGeneReg edges carry ±1.5 weights encoding activation/repression
- Synthetic fallback uses scale-free (Barabási-Albert) topology, which is biologically realistic

**Critical issues:**
- **No Spearman correlation masking ($A = M \odot R$)**: The `construct_base_grn()` function uses fixed prior weights (1.0 / 0.5 / 1.5) but NEVER computes data-driven Spearman correlation to weight edges dynamically. The `tp53_gnn_project.md` spec *explicitly requires* this.
- **STRING PPI integration is stubbed**: Line 188 just logs "deferred to data download" — never actually implemented
- **TargetGeneReg loader expects `p53_targets.csv`** but `download.py` saves as `p53_targets_benchmark.xlsx` — filename mismatch will cause silent failure

### 3. Preprocessing Pipeline Alignment with Literature — 91/100 ✅
**Strengths:**
- Follows the exact Scanpy canonical pipeline: filter cells/genes → MT QC → Scrublet doublets → CPM → log1p → HVG → PCA → Harmony → UMAP
- `seurat_v3` flavor for HVG selection (current best practice for droplet-based data)
- Harmony batch correction correctly uses PCA space
- Malignant cell identification includes heuristic cell-type filtering

**Minor issue:** HVG selection with `seurat_v3` requires raw counts layer, which the code correctly handles through the `layer="counts"` parameter.

### 4. Temporal Architecture Soundness — 93/100 ✅
**Strengths:**
- GRU-based restriction map evolution is correctly edge-batched (shared GRU across all edges)
- Context concatenation (src + tgt node features) gives the GRU proper local information
- LayerNorm after GRU output prevents gradient explosion over long temporal sequences
- Pseudotime uses quantile-based binning (equal cell counts per bin)
- RNA velocity orientation via scVelo is correctly implemented as an optional refinement

**Minor issue:** `data/temporal.py` L282–L290 computes edge weights via a Python loop over E edges — this is O(E·n_cells) and will be extremely slow for E > 5000. Should use vectorized correlation.

### 5. Loss Function Design — 94/100 ✅
**Strengths:**
- 4-term composite loss is well-motivated and matches the proposal exactly
- Differentiable Spearman via soft-ranking (temperature-controlled sigmoid) is mathematically sound
- Sparsity loss uses `torch.autograd.grad` for true gradient-based L1 — this is elegant
- Topological preservation uses hinge loss (Fused Lasso analogy), correct for penalizing rapid rewiring

**No issues above threshold.**

### 6. Scalability & Performance — 78/100 ⚠️ BELOW THRESHOLD
**Strengths:**
- `sheaf_vectorized.py` is a genuine O(E·d²) implementation with precomputed scatter indices — this is publication-quality engineering
- Mixed precision (AMP) is properly implemented in the trainer
- Gradient clipping prevents explosion

**Critical issues:**
- **Dense Laplacian**: The connection Laplacian L_F is stored as a dense (N·d × N·d) matrix. For N=500, d=4 this is 2000×2000 = 16MB per snapshot, ×K×batch = **significant VRAM pressure**. Sparse CSR format would reduce memory by 10-50×.
- **Edge weight computation in temporal.py**: Python loop over all edges is O(E·n_cells) — will bottleneck at preprocessing time
- `sheaf.py` still has the O(E) Python loop version used by default in `tsgnn.py` (line 261 calls `compute_connection_laplacian`, not the vectorized version)

### 7. Code Quality & Engineering Standards — 88/100 ⚠️ BELOW THRESHOLD
**Strengths:**
- Consistent logging throughout all modules
- Type hints on most function signatures
- Docstrings with mathematical notation (LaTeX-style)
- `config.py` uses @dataclass with cross-config validation — excellent pattern
- Model `count_parameters()` method for reproducibility reporting

**Issues:**
- `sheaf.py` L160–L206 is dead code (misleadingly named "efficient" but identical logic to the loop version)
- `allele.py` L22–L30 `TP53_WT_SEQUENCE` is a **wrong/truncated sequence** that's never used — the actual sequence is `TP53_SEQUENCE` on L33. Having both is confusing and error-prone.
- `download.py` mixes `print()` and `logger.info()` inconsistently in `download_all()`
- The `data/__init__.py` only exports 2 symbols — most of the data module is not importable via `from tsgnn.data import ...`

### 8. Test Coverage — 90/100 ✅
**Strengths:**
- 13 test files covering: baselines, config, data loading, FiLM, integration, loss, metrics, registry, sheaf, temporal, trainer, tsgnn, vectorized
- `conftest.py` provides shared fixtures
- Integration test (`test_integration.py`) tests the full forward+backward pass

**At threshold.** Could benefit from parametric tests across stalk_dim values.

### 9. Experiment Reproducibility — 92/100 ✅
**Strengths:**
- `default.yaml` centralizes ALL hyperparameters (seed, grid, thresholds)
- `TSGNNConfig.from_dict()` ensures configs are validated before use
- Checkpointing saves model state, optimizer state, AND config
- `AblationConfig` defines complete grid for systematic experiments

**Minor issue:** No `torch.manual_seed()` / `np.random.seed()` enforcement at training startup — the seed is in config but never actually set by the trainer.

### 10. Colab/Drive Portability — 85/100 ⚠️ BELOW THRESHOLD
**Strengths:**
- `download.py` now uses `_resolve_data_dir()` with 3-level fallback
- `Colab_Data_Downloader.ipynb` exists for one-click setup
- Atomic downloads with `.tmp` prevent corruption

**Issues:**
- `grn_construction.py` L17: `DATA_DIR = Path(__file__).resolve().parents[3] / "data"` — hardcoded relative depth that breaks if the package is installed via pip or moved
- `preprocess.py` L17: Same hardcoded `parents[3]` pattern
- `allele.py` L18: Same pattern
- These 3 files do NOT use `_resolve_data_dir()` from `download.py` — inconsistent path resolution

---

## Summary Table

| # | Parameter | Score | Status |
|---|-----------|-------|--------|
| 1 | Mathematical Correctness (Sheaf) | **95** | ✅ |
| 2 | Biological Coherence (GRN) | **72** | ❌ Fix needed |
| 3 | Preprocessing Literature Alignment | **91** | ✅ |
| 4 | Temporal Architecture | **93** | ✅ |
| 5 | Loss Function Design | **94** | ✅ |
| 6 | Scalability & Performance | **78** | ❌ Fix needed |
| 7 | Code Quality & Engineering | **88** | ❌ Fix needed |
| 8 | Test Coverage | **90** | ✅ |
| 9 | Reproducibility | **92** | ✅ |
| 10 | Colab/Drive Portability | **85** | ❌ Fix needed |
| | **Weighted Average** | **87.8** | |

**4 parameters below 90 — fixes follow below.**

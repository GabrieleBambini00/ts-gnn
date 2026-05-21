# Post-Mortem: Verified Sheaf Integration for TS-GNN

**Date:** 2026-05-21
**Branch:** `sheaf-verified-integration`
**Commits:** 13 (90211c0 → HEAD)
**Net change:** +5,058 lines across 24 files

---

## 1. What happened: the Ravasio leakage failure

Tommaso Ravasio's master thesis (scRNAseq-GNN-binary-tp53 / multiclass-tp53)
achieved near-perfect test accuracy on TP53 mutation classification. The cause
was **cell-line leakage**: the same cell line appeared in both train and test
splits. Because cell lines from the same lineage share expression signatures
independent of TP53 status, the model learned to identify cell lines rather
than TP53-dependent transcriptional programs.

### How Phase 0.3 guards this

`src/tsgnn/data/splits.py` now provides:

- **`GroupDisjointSplitter`** — wraps `sklearn.model_selection.GroupShuffleSplit`
  to split by an arbitrary group key (cell line, patient ID, batch).
- **`assert_no_group_leakage(train_groups, test_groups)`** — raises
  `LeakageError` if any group ID appears on both sides.
- **Unit tests** (`tests/test_splits.py`, 233 lines) verify the assertion
  catches injected leaks and passes on clean grouped splits.

**Recommendation:** the leakage assertion should be a mandatory CI gate on any
training pipeline run. It costs zero compute and prevents the single most
damaging failure mode in biological ML.

---

## 2. Three diverged copies — process failure

At integration start, three copies of the codebase existed:

| Copy | Location | Status |
|------|----------|--------|
| `ts-gnn/` | Canonical | Under git, now verified |
| `ts-gnn local/` | Near-duplicate | Had HPC-specific path tweaks, missing some test files |
| `ts-gnn-chatgpt/` | ChatGPT-assisted fork | Diverged configs, extra experimental code |

None had version control. None was authoritative. Differences were documented
in `DIVERGENCE_REPORT.md` at Phase 0.1.

### Root cause

No single-source-of-truth discipline. Each exploration session (local HPC,
ChatGPT pair-programming, manual edits) forked a new copy instead of branching.
With no git history, it was impossible to determine which copy held the latest
correct version of any given file.

### Mitigation applied

- `ts-gnn/` declared canonical; git initialized with full history.
- Other copies frozen (read-only reference); unique files documented.
- All subsequent work on `sheaf-verified-integration` branch.

### Recommendation

Never work on unversioned copies. Use git branches, even for "quick
experiments." The cost of `git checkout -b experiment` is zero; the cost of
reconciling three diverged trees is hours.

---

## 3. Synthetic-data silent fallback — bug class

The codebase contained a pattern where data-loading functions silently fell
back to generating synthetic data when real files were missing (hardcoded
`parents[3]` paths). This meant:

- Tests could pass on synthetic data while real-data loading was broken.
- A researcher could train a model on synthetic data without knowing it.
- Results were not reproducible across machines (different path depths).

### Fixes applied

- **Task 0.2:** `set_global_seed()` ensures reproducible synthetic data
  generation when it does occur.
- **Task 2.2:** All three modules (`grn_construction.py`, `preprocess.py`,
  `allele.py`) now route through `download._resolve_data_dir()` instead of
  hardcoded `parents[3]`.
- **Tests** (`tests/test_data_dir.py`, 202 lines) verify consistent resolution
  from arbitrary working directories.

### Recommendation

Silent fallback to synthetic data should emit a **warning** (at minimum) or
raise an error in production mode. Add a `config.allow_synthetic_fallback`
flag that defaults to `False` in production pipelines.

---

## 4. Audit gaps — review discipline

The CODE_AUDIT.md scored the codebase 87.8/100 with 4 areas failing:

| # | Issue | Phase fix | Status |
|---|-------|-----------|--------|
| 1 | Dead `*_efficient` function in sheaf.py | 1.1 | ✅ Verified dead; vectorized path is default |
| 2 | GRN A = M ⊙ R missing Spearman correlation | 2.1 | ✅ Test suite validates masking + sign preservation |
| 6 | Dense Laplacian O(N²d²) memory | 3.1 | ✅ Sparse COO path: 38.9× reduction (97.4% saved) |
| 9 | Seed set but never applied | 0.2 | ✅ `set_global_seed()` + bit-identical verification |
| 10 | Hardcoded `parents[3]` data paths | 2.2 | ✅ Consistent `_resolve_data_dir()` |

### What the gaps reveal

These were not subtle bugs. They were missed because:

1. **No test suite enforcing math invariants.** The sheaf Laplacian has five
   testable properties (symmetry, PSD, identity-map reduction, vectorized==loop,
   Dirichlet energy). None were tested before Phase 1.2.
2. **No integration tests for the data pipeline.** Path resolution was never
   tested from a clean working directory.
3. **No reproducibility test.** The seed was set in config but never wired
   through — and no test verified determinism.

### Recommendation

Every mathematical module should ship with property tests. Every data pipeline
should have a smoke test from a clean state. These are cheap to write and catch
the exact class of bugs that slip through code review.

---

## 5. Concrete recommendations

### CI gates (ordered by impact)

1. **Leakage assertion** — `assert_no_group_leakage()` on every split.
   Cost: <1ms. Prevents the Ravasio failure mode entirely.

2. **Sheaf math properties** — P1–P5 from `test_sheaf_math.py`.
   Cost: ~2s. Catches silent Laplacian corruption.

3. **Reproducibility check** — Two 3-epoch runs produce identical loss.
   Cost: ~10s. Catches unseeded randomness.

4. **Data-dir resolution** — `test_data_dir.py` from arbitrary CWD.
   Cost: <1s. Catches hardcoded path regressions.

### Process rules

- **Single source of truth:** one repo, one branch per experiment. No copy-folders.
- **No silent fallback:** synthetic data requires explicit opt-in.
- **Property tests for math:** if a module computes a mathematical object with
  known invariants, those invariants are tested.
- **Grouped splits for biological data:** default splitter is group-aware;
  ungrouped splitting requires explicit justification.

---

## 6. Summary of verified integration

| Phase | Task | Key deliverable | Verified by |
|-------|------|----------------|-------------|
| 0.1 | Git baseline | Single canonical repo | `git log`, clean status |
| 0.2 | Reproducibility | `set_global_seed()` | Bit-identical 3-epoch runs |
| 0.3 | Leakage guard | `GroupDisjointSplitter` | Leak injection test |
| 1.1 | Dead code cleanup | Vectorized Laplacian default | Code path audit |
| 1.2 | Math property tests | P1–P5 test suite | 5/5 properties passing |
| 2.1 | A = M ⊙ R | Spearman-masked GRN | Sign + mask tests |
| 2.2 | Data-dir consistency | `_resolve_data_dir()` | Cross-CWD tests |
| 3.1 | Sparse Laplacian | COO construction | 38.9× memory reduction, 15/15 tests |

**Out of scope (HPC-gated):** Real-data training runs on scRNA-seq datasets.
The infrastructure is verified; the data pipeline is ready; actual benchmarks
require multi-GB data and GPU compute.

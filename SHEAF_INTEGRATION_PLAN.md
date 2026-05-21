# Verified Sheaf Integration for TS-GNN — Execution Plan (Chain-of-Verification)

## Reality check (grounding)

The sheaf is **already implemented**. `ts-gnn/` is a Temporal Sheaf GNN
(sheaf diffusion + GRU + FiLM, Bodnar et al. NeurIPS 2022 formulation):
`model/sheaf.py`, `model/sheaf_vectorized.py`, full trainer, 18 test files.

So "implement sheaf on top of the project" is **not** writing sheaf from
zero. It is: make the existing sheaf integration **trustworthy** — fix the
gaps `CODE_AUDIT.md` already flagged (87.8/100, 4 areas failing), eliminate
the leakage risk inherited from Ravasio's thesis, and consolidate the three
diverged copies. Then post-mortem.

Out of session scope: real-data training/benchmark runs (multi-GB scRNA-seq,
HPC). Those are gated on data and flagged in tasks, not executed here.

## Method: Chain-of-Verification (CoVe)

Each phase ends with a **CoVe checkpoint**: a fixed list of verification
questions that must be answered with *executed evidence* (test output, diff,
command result) — not assertions — before the phase is accepted. A checkpoint
failing sends the phase back for fixes.

## Canonical source

`ts-gnn/` is declared canonical. `ts-gnn local/` and `ts-gnn-chatgpt/` are
**not modified** (non-destructive). Their divergences are documented only.

---

## Phase 0 — Baseline & guardrails

### Task 0.1 — Canonical repo + git baseline
- `git init` in `ts-gnn/`, add a `.gitignore` covering `data/`, `checkpoints/`,
  `__pycache__/`, `.pytest_cache/`, `*.pt`, notebooks output.
- Create branch `sheaf-verified-integration`. Initial commit of current tree.
- Write `DIVERGENCE_REPORT.md`: enumerate what differs in the two other copies
  (from memory obs 61/62) so nothing unique is silently lost.
- **Verify:** `git status` clean; branch is `sheaf-verified-integration`;
  `pytest --collect-only` succeeds (tests discoverable).

### Task 0.2 — Reproducibility guardrail
- Audit issue #9: seed is in config but never applied. Add a `set_global_seed()`
  that seeds `torch`, `numpy`, `random`, `torch.cuda`, and sets
  `cudnn.deterministic`. Call it at trainer startup from `config.seed`.
- **Verify:** two trainer runs (3 epochs, synthetic data) produce
  bit-identical loss sequences.

### Task 0.3 — Leakage guardrail (Ravasio lesson)
- Ravasio's thesis hit 100% test accuracy from cell-line leakage. The spec
  (`tp53_gnn_project.md` §3.3) requires splitting by cell line / patient.
- Implement a grouped splitter (`GroupShuffleSplit` by cell-line/patient id)
  and an `assert_no_group_leakage(train, test)` that raises if any group id
  appears on both sides.
- **Verify:** assertion catches an injected leak (test); a correct grouped
  split passes; unit test green.

### CoVe Checkpoint 0
1. Is there exactly one canonical tree under version control? (show `git log`)
2. Are the other two copies' unique files documented? (show DIVERGENCE_REPORT)
3. Does a fixed seed give reproducible loss? (show two runs)
4. Does the leakage assertion fail on a leaked split and pass on a clean one?
   (show test output)

---

## Phase 1 — Sheaf core verification & cleanup

### Task 1.1 — Remove dead code, default to vectorized Laplacian
- Audit #1/#6: `sheaf.py` `compute_connection_laplacian_efficient` is a
  misleadingly-named dead copy of the loop version. Remove it.
- `tsgnn.py` currently calls the O(E) Python-loop Laplacian. Switch the default
  path to `sheaf_vectorized.py`'s O(E·d²) implementation.
- **Verify:** `tsgnn.py` forward pass uses vectorized path; loop and vectorized
  Laplacians are numerically equal (`verify_vectorized_matches_loop`).

### Task 1.2 — Sheaf math property-test suite
- Add property tests covering: (a) L_F symmetric, (b) L_F positive
  semi-definite, (c) identity restriction maps → standard graph Laplacian,
  (d) vectorized == loop across `stalk_dim ∈ {1,2,4}`, (e) one diffusion step
  is Dirichlet-energy non-increasing for small step size.
- **Verify:** all property tests pass; show `pytest tests/test_sheaf_math.py`.

### CoVe Checkpoint 1
1. Is the dead `*_efficient` function gone? (show diff)
2. Does the assembled model use the vectorized Laplacian? (show code path)
3. Do all five sheaf math properties hold? (show test output)

---

## Phase 2 — Biological coherence (GRN A = M ⊙ R)

### Task 2.1 — Spearman-masked adjacency
- Audit #2: spec explicitly requires `A = M ⊙ R` (prior mask ⊙ data-driven
  Spearman correlation) but `construct_base_grn()` uses only fixed prior
  weights — no correlation. Implement the element-wise masking, retaining
  signed/negative correlations (repressors).
- **Verify:** on synthetic expression + known mask, output edges = mask edges
  only; negative correlations survive; sign matches correlation sign.

### Task 2.2 — Consistent data-dir resolution
- Audit #10: `grn_construction.py`, `preprocess.py`, `allele.py` use a hardcoded
  `parents[3]` path. Route all three through `download._resolve_data_dir()`.
- **Verify:** modules import and resolve data dir correctly from an arbitrary
  working directory (test).

### CoVe Checkpoint 2
1. Does the GRN now multiply prior mask by Spearman correlation? (show diff)
2. Are negative (repressor) correlations preserved? (show test)
3. Do all three modules resolve the data dir consistently? (show test)

---

## Phase 3 — Scalability

### Task 3.1 — Sparse sheaf Laplacian
- Audit #6: L_F is a dense (N·d × N·d) matrix. Provide a sparse (CSR/COO)
  construction path for the diffusion matvec.
- **Verify:** sparse result matches dense within tolerance; report measured
  memory reduction.

### CoVe Checkpoint 3
1. Does sparse L_F match dense numerically? (show test)
2. What is the measured memory reduction? (show numbers)

---

## Phase 4 — Post-mortem

### Task 4.1 — Write POST_MORTEM.md
Cover: the Ravasio leakage failure mode and how Phase 0.3 guards it; the
three-diverged-copies process failure; the synthetic-data silent-fallback bug
class; what the audit gaps revealed about review discipline; concrete
recommendations (single source of truth, CI gate, leakage test in CI).

---

## Execution

Subagent-driven: fresh implementer per task → spec-compliance review →
code-quality review → next task. Working dir: `ts-gnn/`.

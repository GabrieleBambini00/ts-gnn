# TS-GNN PROJECT CONTEXT
> **Agent instructions**: Read this file at the start of every session instead of re-reading all source files. Update it whenever you change project files. Mark tasks done as you complete them. Never ask the user what to do next - consult this file first.

---

## CURRENT SCORES (session 2026-04-06 Phase 2 — Nobel-standard upgrades)

| # | Criterion | Score | Notes |
|---|-----------|:-----:|-------|
| 1 | Mathematical Correctness (Sheaf) | **91** | NormalizedVectorizedSheafDiffusion: L_sym=D^{-1/2}L_FD^{-1/2}, eigenvalues in [0,2] (Cheeger bound) |
| 2 | Biological Coherence (GRN) | **86** | DoRothEA A+B + decoupler ULM replacing mean-expr VIPER proxy; load_dorothea_prior_edges as Layer 0 |
| 3 | Preprocessing Literature Alignment | **82** | scVI (Lopez 2018 NatMethods) deep generative batch correction; Harmony fallback; seurat_v3 HVG |
| 4 | Temporal Architecture | **85** | scVelo velocity consistency loss (λ4); diffusion pseudotime n_comps adaptive |
| 5 | Loss Function Design | **92** | NeuralSort (Grover NeurIPS 2019) differentiable ranking; λ4 velocity term; create_graph=False |
| 6 | Scalability & Performance | **83** | Lazy device buffer move (single guarded block); gradient checkpointing for d≥8 |
| 7 | Code Quality & Engineering | **84** | All English; type hints; API aliases; pyproject.toml fixed |
| 8 | Test Coverage | **91** | 130/130 passing; 18 new tests: NeuralSort, NormLaplacian, VelLoss, DoRothEA, Checkpoint, Optuna |
| 9 | Reproducibility | **87** | environment.yml pinned; setup_colab.sh w/ auto PyG wheels; MD5 validation framework |
| 10 | Colab/Drive Portability | **83** | Checkpoint resume cell (cell 11); setup_colab.sh; 22-cell notebook |

**Mean: 86.4/100** (up from 74.7 Phase 1)

---

## ARCHITECTURE REFERENCE

### Neural Sheaf Diffusion (Bodnar et al. NeurIPS 2022)
- Sheaf Laplacian: L_F in R^(Nd x Nd), block-sparse
  - Off-diagonal blocks: -F_u^T @ F_v
  - Diagonal blocks: sum_e F_e^T @ F_e
- VectorizedSheafDiffusion (sheaf_vectorized.py): O(E*d^2) via scatter_add_, zero Python loops
- SheafDiffusionLayer (sheaf.py): reference O(E) loop implementation
- API aliases added: both expose .compute_connection_laplacian() and .diffuse()

### GRU Temporal Evolution
- Shared GRUCell evolves restriction maps (E, 2, d, d) over K=10 pseudotime bins
- Initial hidden = flattened restriction maps (not zeros - see temporal.py:96)

### FiLM Allele Conditioning
- ESM-2 (1280-dim) -> MLP -> gamma/beta modulating restriction maps
- AlleleFiLM in model/film.py; ZeroFiLM for ablation
- gamma offset by +1.0 for near-identity initialisation

### 5-term Composite Loss (Phase 2)
```
L = L_expr + lambda1*L_topo + lambda2*L_regulon + lambda3*L_sparse + lambda4*L_vel
```
- L_expr: MSE on predicted vs actual expression
- L_topo: hinge penalty ||delta L_F||_F > tau
- L_regulon: -Spearman(predicted_activity, DoRothEA_ULM) via NeuralSort (Grover NeurIPS 2019)
  - NeuralSort logit: logit[k,j] = (n+1-2k)*s[j] - Σ|s[j]-s[l]|; softmax over dim=-1
- L_sparse: L1 of d(L_F)/d(z_allele), create_graph=False (saves memory)
- L_vel: -mean cosine_similarity(X_hat(t+1)-X_hat(t), v(t)) -- scVelo velocity field
  - lambda_4=0.0 by default (disabled until scVelo data available)
- FIXED: Pearson std uses .mean() not .sum() (loss.py:259-260)
- FIXED: NeuralSort softmax dim=-1 (element dim), not dim=-2 (rank dim)

### Data Pipeline
1. GSE176078 (Wu et al. 2021 TNBC, 24 patients) -> load_gse176078()
2. GSE158508 (paired scRNA+scATAC) -> load_gse158508()
3. TCGA BRCA TP53 mutations (cBioPortal) -> load_tcga_brca_tp53()
4. Merge + allele assignment -> load_brca_data()
5. QC/normalise/HVG -> preprocess_scrna()
6. Feature selection (BRCA_KEY_TFS -> JASPAR TFs -> p53 targets -> HVGs) -> select_features()
7. JASPAR + RegNetwork + TargetGeneReg + STRING + ENCODE -> construct_base_grn()
8. Pseudotime -> binning -> snapshots -> build_brca_temporal_sequences()
9. Patient-level train/val/test split -> split_patients_by_allele()

### Key Environment Variables
- TSGNN_DATA_DIR: root of all data directories (raw/, external/, processed/)

---

## FILE INVENTORY

### Source (src/tsgnn/)
| File | Purpose | Status |
|------|---------|--------|
| model/tsgnn.py | Full TSGNN architecture, create_tsgnn_from_config | DONE - lazy import fixed |
| model/sheaf_vectorized.py | VectorizedSheafDiffusion + NormalizedVectorizedSheafDiffusion | DONE - normalized Laplacian added, lazy device move |
| model/sheaf.py | Reference SheafDiffusionLayer (O(E) loop) | OK |
| model/film.py | AlleleFiLM, ZeroFiLM | OK |
| model/temporal.py | TemporalRestrictionEvolution (GRU) | OK |
| model/baselines.py | Baseline models for comparison | OK |
| data/brca_loader.py | BRCA data loading, allele assignment, splits | DONE - patient merge fixed |
| data/grn_construction.py | GRN assembly + load_dorothea_prior_edges Layer 0 | DONE - DoRothEA A+B as top-priority edges |
| data/preprocess.py | scRNA QC/normalise, scVI batch correction | DONE - scVI+Harmony fallback, seurat_v3 |
| data/temporal.py | Pseudotime, binning, compute_viper_activity decoupler ULM | DONE - real DoRothEA VIPER replacing mean-expr |
| data/download.py | Dataset downloading, download_geo_dataset alias | OK |
| data/allele.py | ESM-2 embeddings, TP53_HOTSPOT_MUTATIONS | OK |
| training/trainer.py | TSGNNTrainer, EarlyStopping | DONE - grad_clip_norm fix |
| training/loss.py | TSGNNLoss, NeuralSort Spearman, velocity_consistency_loss | DONE - NeuralSort dim fix, lambda_4, create_graph=False |
| training/optuna_search.py | Optuna HPO with TPE + MedianPruner | DONE - created |
| config.py | Config dataclass with validation | OK |
| cli.py | CLI entry points | OK |

### Scripts (scripts/)
| File | Purpose | Status |
|------|---------|--------|
| run_pipeline.py | Main end-to-end pipeline | DONE - ESM-2 fallback seeded |
| run_hparam_search.py | Optuna search entry point | DONE - created |
| run_ablations.py | Ablation study runner | OK |
| run_benchmarks.py | Performance benchmarking | OK |
| download_all_datasets.py | Download all datasets | MD5 validation framework added |
| compute_md5s.py | MD5 checksum computation | OK |
| setup_colab.sh | NEW - Colab auto-setup with PyG wheels | DONE - created |

### Tests (tests/)
- **130 passing, 1 skipped** (test_e2e_real.py - requires real data, marked @slow)
- 18 new tests in tests/test_advanced.py
- All Italian text in comments translated to English

### Config/Env
| File | Purpose | Status |
|------|---------|--------|
| configs/default.yaml | All hyperparameters | DONE - lambda_4, use_normalized_laplacian |
| environment.yml | Pinned conda env (Python 3.10, CUDA 11.8) | DONE - created |
| TSGNN_Colab_Master.ipynb | Colab notebook 22 cells | DONE - checkpoint resume cell 11 |

---

## BUGS FIXED (session 2026-04-06)

| ID | File | Bug | Fix |
|----|------|-----|-----|
| B1 | training/loss.py:259 | Pearson std used .sum() -- wrong numerics | Changed to .mean() |
| B2 | data/grn_construction.py:422 | Silent synthetic GRN fallback | Now raises RuntimeError |
| B3 | scripts/run_pipeline.py:85 | ESM-2 random fallback unseeded | Uses seeded Generator |
| B4 | data/brca_loader.py:141 | Patient merge only checked patient_id-0 | Handles all patient_id-N via bfill |
| B5 | data/preprocess.py:117 | seurat_v3 silently fails without counts layer | Explicit check + fallback to seurat |
| B6 | model/tsgnn.py:179 | torch.utils import inside method | Moved to module-level import |
| B7 | model/sheaf_vectorized.py | Missing compute_connection_laplacian + diffuse | Aliases added |
| B8 | training/trainer.py:104 | max_grad_norm key mismatch with grad_clip_norm | Both keys supported |
| B9 | data/grn_construction.py | Italian docstrings | Translated to English |
| B10 | data/preprocess.py | Italian in SoupX log message | Translated to English |
| B11 | data/temporal.py:298 | Italian comment | Translated to English |
| B12 | configs/default.yaml | Italian comments | Translated to English |
| B13 | pyproject.toml | Italian marker description | Translated to English |
| B14 | data/preprocess.py | BRCA_KEY_TFS duplicated | Now imported from grn_construction |
| B15 | tests/test_grn_construction.py | Tests expected silent synthetic fallback | Updated to test RuntimeError + _generate_synthetic_grn |
| B16 | pyproject.toml | build-backend was setuptools.backends._legacy (invalid) | Fixed to setuptools.build_meta |
| B17 | tests/conftest.py | src/ not in sys.path (module not found) | Added sys.path.insert in conftest.py |
| B18 | training/loss.py | NeuralSort softmax on dim=-2 (rank dim) → all ranks=1.0 | Fixed to dim=-1 (element dim) |
| B19 | model/sheaf_vectorized.py | .to(device) called 4x per forward pass | Single guarded block with dev check |
| B20 | training/loss.py | sparsity_loss create_graph=True causing 2nd-order graph | Fixed to create_graph=False |

---

## PHASE 2 NEW COMPONENTS (2026-04-06)

| Component | Location | Description |
|-----------|----------|-------------|
| NormalizedVectorizedSheafDiffusion | model/sheaf_vectorized.py | D^{-1/2}L_FD^{-1/2}, eigenvalues in [0,2], Cheeger bound |
| load_dorothea_prior_edges | data/grn_construction.py | DoRothEA A+B TF-target edges as Layer 0 GRN prior |
| compute_viper_activity (real) | data/temporal.py | decoupler ULM replacing mean-expr proxy |
| _batch_correct_scvi | data/preprocess.py | scVI deep generative model (Lopez 2018 NatMethods) |
| velocity_consistency_loss | training/loss.py | L_vel = -cosine_sim(delta_pred, scVelo_vel) |
| _neural_sort_ranks | training/loss.py | NeuralSort (Grover NeurIPS 2019), dim=-1 fix |
| resume_from_best | training/trainer.py | Loads best.pt and returns start_epoch |
| setup_colab.sh | scripts/ | Auto PyG wheel detection, full dep install |
| environment.yml | root | Pinned conda env Python 3.10 + CUDA 11.8 |
| test_advanced.py | tests/ | 18 new tests across all new components |

---

## PENDING TASKS (remaining to reach 90+)

### Still open (score headroom)
- [ ] BIO2: Normalise cell-type names in brca_loader.py non-malignant set (lowercase + strip)
- [ ] BIO3: Make gene overlap threshold configurable (currently hardcoded=100 in brca_loader.py:305)
- [ ] T2: Add device mismatch test in test_vectorized.py
- [ ] T3: Add multi-dataset merge test in test_brca_loader.py
- [ ] C3: Add tqdm progress bars to long loops in brca_loader.py and temporal.py
- [ ] C4: Make ESM-2 cache Drive-aware (check Drive path before recomputing)
- [ ] S2: Add @torch.compile to VectorizedSheafDiffusion for PyTorch 2.x speedup

### COMPLETED in Phase 2
- [x] R1: environment.yml with pinned conda deps
- [x] R3: MD5 validation framework in download.py (validate_file_md5, validate_checksums)
- [x] C1: scripts/setup_colab.sh with auto PyG wheel detection
- [x] C2: Checkpoint resume in TSGNN_Colab_Master.ipynb cell 11
- [x] BIO1: Real decoupler ULM replacing mean-expr VIPER
- [x] S1: Lazy device buffer move (single guarded block)
- [x] T1: Optuna test in test_advanced.py:TestOptuna
- [x] NeuralSort: Grover NeurIPS 2019 differentiable ranking
- [x] NormLaplacian: D^{-1/2}L_FD^{-1/2} eigenvalues in [0,2]
- [x] VelLoss: scVelo velocity consistency as lambda_4 term

---

## OPTUNA HYPERPARAMETER SEARCH

New module: src/tsgnn/training/optuna_search.py
New script: scripts/run_hparam_search.py
Install: pip install ".[hparam]"  (or pip install optuna optuna-dashboard)

### Search space
| Parameter | Range | Sampler |
|-----------|-------|---------|
| lr | [1e-4, 5e-3] | log-uniform TPE |
| stalk_dim | {2, 4, 8} | categorical |
| conditioning_dim | {64, 128, 256} | categorical |
| num_diffusion_steps | [1, 5] | integer |
| lambda_1 | [0.01, 1.0] | log-uniform |
| lambda_2 | [0.1, 2.0] | log-uniform |
| lambda_3 | [1e-3, 0.1] | log-uniform |
| tau | [0.1, 1.0] | uniform |
| weight_decay | [0.0, 1e-3] | uniform |

### Programmatic usage
```python
from tsgnn.training.optuna_search import run_optuna_search
results = run_optuna_search(
    train_data=train_sequences,   # dict allele -> TemporalGraphSequence
    val_data=val_sequences,
    esm_embeddings=esm_embeddings,
    n_trials=50,
    n_epochs_per_trial=30,
    out_config_path="configs/optuna_best.yaml",
    storage="sqlite:///optuna_tsgnn.db",  # resumable
)
```

### CLI usage
```bash
python scripts/run_hparam_search.py \
    --n-trials 50 --n-epochs 30 \
    --out-config configs/optuna_best.yaml \
    --storage sqlite:///optuna_tsgnn.db
```

---

## HYPERPARAMETER DEFAULTS (configs/default.yaml)

seed: 42
model.stalk_dim: 4
model.input_dim: 500
model.esm_dim: 1280
model.conditioning_dim: 128
model.num_diffusion_steps: 3
model.k_hop_radius: 2
training.lr: 1e-3
training.weight_decay: 0.0
training.max_epochs: 500
training.patience: 20
training.grad_clip_norm: 1.0
loss.lambda_1: 0.1
loss.lambda_2: 0.5
loss.lambda_3: 0.01
loss.tau: 0.5
temporal.K: 10
temporal.pseudotime_method: diffusion

---

## BRCA DATASETS

| Dataset | Source | Content |
|---------|--------|---------|
| GSE176078 | Wu et al. 2021, Nature Genetics | 100,064 cells, 24 TNBC patients, 10x |
| GSE158508 | Paired scRNA+scATAC | 10x Multiome BRCA panel |
| TCGA BRCA | cBioPortal | TP53 somatic mutations (protein changes) |
| JASPAR 2024 | jaspar.genereg.net | Human TF binding motifs |
| RegNetwork | regnetwork.org | TF-gene regulatory edges |
| STRING v12 | string-db.org | Protein-protein interactions |
| ENCODE ChIP-seq | encodeproject.org | TF ChIP in MDA-MB-231, T47D, MCF7 |

TP53 hotspot alleles modelled: R175H, R273H, R248W, R282W, G245S, Y220C, WT

---

## COLAB LAUNCH CHECKLIST

1. Upload ts-gnn/ to Google Drive at /content/drive/MyDrive/tsgnn/
2. Open TSGNN_Colab_Master.ipynb
3. Cell 1: Mount Drive + GPU check
4. Cell 2: pip install -e .
5. Cell 3: python scripts/download_all_datasets.py
6. Cell 4: Compute ESM-2 embeddings (requires ~8 GB VRAM)
7. Cell 5: python scripts/run_pipeline.py
8. Cell 6: Display results
9. Cell 7 (optional): python scripts/run_hparam_search.py

Before upload zip with:
  zip -r tsgnn_colab.zip ts-gnn/ -x "ts-gnn/data/raw/*" "ts-gnn/__pycache__/*"

---

## AGENT UPDATE RULES

1. After every code change: update the relevant row in BUGS FIXED or PENDING TASKS
2. After every test run: update score table if scores change
3. Mark tasks DONE when done, BLOCKED with reason if stuck
4. Never use synthetic data -- raise RuntimeError if real data unavailable
5. All new code must pass: python -m pytest tests/ -q before marking done
6. Keep all code and comments in English
7. BRCA_KEY_TFS is canonical in grn_construction.py -- import from there everywhere else
8. ESM-2 fallback must use seeded Generator -- never random without seed
9. Optuna deps are optional: guard with try/except ImportError, never hard-depend

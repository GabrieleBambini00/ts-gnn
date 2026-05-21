# Graph Report - .  (2026-04-20)

## Corpus Check
- 74 files · ~41,264,692 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 895 nodes · 1532 edges · 58 communities detected
- Extraction: 65% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 532 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `TSGNN` - 143 edges
2. `TSGNNLoss` - 74 edges
3. `EvolveGCN` - 60 edges
4. `TemporalGAT` - 58 edges
5. `TSGNNTrainer` - 56 edges
6. `AlleleFiLM` - 42 edges
7. `ZeroFiLM` - 39 edges
8. `VectorizedSheafDiffusion` - 38 edges
9. `SheafDiffusionLayer` - 36 edges
10. `NormalizedVectorizedSheafDiffusion` - 36 edges

## Surprising Connections (you probably didn't know these)
- `Unit tests for the Neural Sheaf Diffusion layer.` --uses--> `SheafDiffusionLayer`  [INFERRED]
  tests\test_sheaf.py → src\tsgnn\model\sheaf.py
- `With identity restriction maps, sheaf Laplacian should reduce to graph Laplacian` --uses--> `SheafDiffusionLayer`  [INFERRED]
  tests\test_sheaf.py → src\tsgnn\model\sheaf.py
- `Diffusion should include residual connection.` --uses--> `SheafDiffusionLayer`  [INFERRED]
  tests\test_sheaf.py → src\tsgnn\model\sheaf.py
- `Can pass external restriction maps.` --uses--> `SheafDiffusionLayer`  [INFERRED]
  tests\test_sheaf.py → src\tsgnn\model\sheaf.py
- `Tests for the vectorized sheaf Laplacian implementation.` --uses--> `VectorizedSheafDiffusion`  [INFERRED]
  tests\test_vectorized.py → ultimate\sheaf_vectorized.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (89): CellOracleBaseline, create_baseline(), create_tsgnn_no_allele(), DictysBaseline, GATLayer, Baseline Models for Fair Comparison with TS-GNN.  Phase 2E: Implements 5 baselin, Wrapper around Dictys for time-resolved GRN inference.      Dictys (Wang et al.,, Run Dictys dynamic GRN inference.          Returns:             Dict with 'tempo (+81 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (92): EvolveGCN, EvolveGCN baseline: GCNConv with GRU-evolved parameters.      Replaces sheaf dif, Temporal Graph Attention Network baseline.      Replaces sheaf diffusion with mu, TemporalGAT, Composite loss for TS-GNN training.      Args:         lambda_1: Weight for topo, TSGNNLoss, create_default_metric_registry(), create_default_model_registry() (+84 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (59): _objective(), print_study_summary(), Optuna hyperparameter optimisation for TS-GNN.  Searches over:   - Learning rate, Run Optuna hyperparameter search for TS-GNN.      Args:         train_data: Dict, Print a formatted summary of an Optuna study., Build a config dict from Optuna trial suggestions., Optuna objective: returns best validation loss for a trial., run_optuna_search() (+51 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (26): cli_download(), cli_evaluate(), cli_train(), cli_validate(), TS-GNN CLI entry points.  Provides command-line interfaces:     tsgnn-train, Verifica che tutti i file necessari esistano e abbiano dimensioni corrette., Run the TS-GNN training pipeline., Download all required datasets for TS-GNN. (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (35): download_all(), download_brca_scrna(), download_crc_scrna(), download_depmap(), download_encode_chipseq(), _download_file(), download_geo_dataset(), download_jaspar() (+27 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (29): _aggregate_seeds(), BenchmarkRunner, _flatten_metrics(), _latex_escape(), Baseline Comparison Benchmarking for TS-GNN.  Phase 4C: Runs all baseline compar, Generate LaTeX comparison table (Table 1 format from the paper).          Return, Flatten nested metric dicts to a single level., Aggregate metrics across seeds. (+21 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (18): AblationRunner, _aggregate_results(), _latex_escape(), Ablation Study Runner for TS-GNN.  Phase 4B: Systematic ablation experiments acr, Ablation 2: FiLM conditioning vs. no conditioning., Ablation 3: Stalk dimension d in {2, 4, 8}., Ablation 4: k-hop radius in {1, 2, 3}., Ablation 5: Temporal resolution K in {5, 10, 20}. (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (29): Unit tests for evaluation metrics., Identical scores → rho ≈ 1., Should handle < 3 valid data points., Identical alleles → high p-value (not distinguishable).      Theoretical: if del, Highly different alleles → low p-value (distinguishable).      With laps_b scale, Should return top_k edges with correct count., Hub scores should have shape (N,) with positive values., Identical predicted and observed edges → r ≈ 1.0. (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (6): create_default_metric_registry(), create_default_model_registry(), MetricRegistry, ModelRegistry, DummyModel, Tests for the dependency-injected registries.

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (22): model(), model_no_allele(), model_params(), Unit tests for the full TS-GNN architecture., Full forward-backward pass should complete without errors., The allele-specific delta Laplacians should be nonzero.      Theoretical justifi, Canonical small-scale model parameters for testing., Without allele conditioning, delta Laplacians should be zero.      Theoretical: (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (16): AblationConfig, DataConfig, from_dict(), LossConfig, ModelConfig, Typed configuration schema for TS-GNN.  REPLACES the raw dict-based config patte, Configuration for ablation study parameters., Top-level configuration combining all sub-configs. (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (21): _build_string_id_map(), compute_spearman_weights(), construct_base_grn(), _generate_synthetic_grn(), load_dorothea_prior_edges(), load_regnetwork(), load_scenic_regulons(), load_string_ppi() (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.1
Nodes (16): NamedTuple, bin_cells_by_pseudotime(), compute_pseudotime(), compute_viper_activity(), construct_temporal_graphs(), orient_with_velocity(), GRU Temporal Evolution of Restriction Maps.  Phase 2B: Implements the GRU that e, Orient pseudotime using RNA velocity direction.      If velocity indicates the t (+8 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (14): condition_number(), Fully Vectorized Sheaf Connection Laplacian.  REPLACES the O(E) Python loop in s, Build the (Nd, Nd) sheaf connection Laplacian with ZERO Python loops.          U, Build the (Nd, Nd) sheaf connection Laplacian with ZERO Python loops.          U, One step of sheaf diffusion: x_{k+1} = x_k - α·L_F·(x_k·W + b)          Residual, Full sheaf diffusion forward pass.          Args:             node_features: (N,, One step of sheaf diffusion: x_{k+1} = x_k - α·L_F·(x_k·W + b)          Residual, Full sheaf diffusion forward pass.          Args:             node_features: (N, (+6 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (19): annotate_brca_subtype(), build_brca_temporal_sequences(), _extract_gse176078(), load_brca_data(), load_gse158508(), load_gse176078(), _load_patient_dir(), load_tcga_brca_tp53() (+11 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (13): _differentiable_spearman(), _neural_sort_ranks(), _pearson_correlation(), Composite 4-Term Loss Function for TS-GNN Training.  Phase 3A: L = L_expr + lamb, L_regulon = -(1/N) * sum_v rho(a_hat_v, a_VIPER_v)          Negative correlation, L_sparse = ||dL_F / dz_allele||_1          L1 penalty on allele-specific perturb, L_vel = -mean cosine_similarity(X_hat(t+1) - X_hat(t), v(t))          Penalises, Compute composite loss.          Returns:             total_loss: Scalar loss fo (+5 more)

### Community 16 - "Community 16"
Cohesion: 0.16
Nodes (13): assign_alleles_to_cells(), create_mutant_sequence(), generate_esm2_embeddings(), _generate_placeholder_embeddings(), get_structural_class(), load_esm2_embeddings(), TP53 allele extraction and ESM-2 embedding generation.  Phase 1D: Generate allel, Generate random placeholder embeddings for development without ESM-2. (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.2
Nodes (11): _apply_soupx_if_available(), _batch_correct_scvi(), identify_malignant_cells(), preprocess_scrna(), scRNA-seq preprocessing pipeline for TS-GNN.  Phase 1E.1-2: Quality control, nor, Full scRNA-seq preprocessing pipeline.      Steps:     1. Basic QC filtering, Apply SoupX ambient RNA correction if pre-computed counts are available.     If, Identify malignant cells via copy number inference.      Adds adata.obs["is_mali (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.2
Nodes (11): _extract_primary_value(), format_ablation_table(), format_comparison_table(), format_results_table(), print_results_summary(), LaTeX-ready results table generator for TS-GNN evaluation.  Produces publication, Generate a LaTeX table comparing TS-GNN against baselines.      Args:         mo, Generate a LaTeX table from evaluation results.      Args:         results: Dict (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.17
Nodes (2): test_all_baselines_return_three_lists(), test_registry_has_all_baselines()

### Community 20 - "Community 20"
Cohesion: 0.17
Nodes (7): Unit tests for the Neural Sheaf Diffusion layer., With identity restriction maps, sheaf Laplacian should reduce to graph Laplacian, Diffusion should include residual connection., Can pass external restriction maps., test_diffusion_residual(), test_external_maps(), test_identity_maps_reduce_to_graph_laplacian()

### Community 21 - "Community 21"
Cohesion: 0.24
Nodes (8): ABC, AbstractGRNModel, forward(), predict(), Abstract base class for all GRN models in TS-GNN.  Enforces a uniform interface, Abstract base class for Gene Regulatory Network models.      All models — TS-GNN, Count trainable parameters by module.          Returns:             Dict mapping, Human-readable model summary.

### Community 22 - "Community 22"
Cohesion: 0.18
Nodes (1): Tests for the vectorized sheaf Laplacian implementation.

### Community 23 - "Community 23"
Cohesion: 0.2
Nodes (7): medium_graph(), Shared test fixtures for TS-GNN test suite., Small graph for fast tests: 10 nodes, 25 edges, stalk_dim=2., Medium graph: 30 nodes, 80 edges, stalk_dim=4., K=3 temporal sequence for small graph., small_graph(), temporal_sequence()

### Community 24 - "Community 24"
Cohesion: 0.22
Nodes (3): criterion(), Unit tests for the composite loss function., test_forward_returns_components()

### Community 25 - "Community 25"
Cohesion: 0.25
Nodes (7): analyze_regulatory_mode(), plot_restriction_map_eigenvalues(), plot_restriction_map_heatmap(), Restriction Map Analysis and Visualization.  Phase 5.2: Visualize learned restri, Plot heatmaps of restriction maps for a specific edge over time.      Shows how, Classify each edge as activating, repressing, or mixed based on     restriction, Visualize eigenvalue distribution of restriction maps over time.      Negative e

### Community 26 - "Community 26"
Cohesion: 0.25
Nodes (7): plot_allele_embedding_space(), plot_grn_network(), plot_therapeutic_window(), GRN and Allele Embedding Space Visualization.  Phase 5.3: - t-SNE/UMAP of ESM-2, Visualize GRN as a directed network graph.      Args:         edge_index: (2, E), Visualize therapeutic window from transient bottleneck analysis.      Shows cent, Visualize ESM-2 allele embeddings in 2D.      Colors by structural class (contac

### Community 27 - "Community 27"
Cohesion: 0.25
Nodes (7): identify_bottleneck_timepoints(), plot_differential_rewiring(), plot_rewiring_trajectory(), Rewiring Trajectory Visualization.  Phase 5.1: Animate the GRN evolving over pse, Plot differential rewiring between two alleles.      Highlights edges that diffe, Identify time points with maximal betweenness centrality change.      These repr, Plot GRN rewiring trajectory over pseudotime.      Creates a multi-panel figure

### Community 28 - "Community 28"
Cohesion: 0.25
Nodes (7): Tests for GRN construction module., _generate_synthetic_grn returns correct edge format and no self-loops., Synthetic GRN must not contain self-loops., construct_base_grn must raise RuntimeError if no regulatory databases are found., test_construct_base_grn_raises_without_data(), test_no_self_loops(), test_synthetic_grn_dimensions()

### Community 29 - "Community 29"
Cohesion: 0.36
Nodes (7): module(), Unit tests for the GRU temporal evolution module., PAGA pseudotime non deve contenere valori infiniti dopo la correzione., test_different_from_input(), test_gradient_flow(), test_output_shape(), test_paga_pseudotime_no_infs()

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (1): Contract tests for data loading modules.  These test that the public interfaces

### Community 31 - "Community 31"
Cohesion: 0.6
Nodes (4): load_data_for_search(), main(), parse_args(), Load real BRCA data and build train/val splits for hyperparameter search.     Mi

### Community 32 - "Community 32"
Cohesion: 0.83
Nodes (3): main(), mkdirs(), validate()

### Community 33 - "Community 33"
Cohesion: 0.5
Nodes (3): Test BRCA data loader con file mock., Verifica mapping proteinChange → allele name., test_tcga_allele_mapping()

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (1): Test end-to-end su dati BRCA reali (skip se file non presenti).

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Create config from a flat dictionary (e.g., loaded from YAML).

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Stack all node features: (K, N, d_input).

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Stack all edge weights: (K, E).

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Forward pass through the model.          Args:             node_features_seq: (K

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Inference-time prediction (no grad, returns predictions only).          Args:

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Compute the spectral gap (λ₂ - λ₁) of the sheaf Laplacian.          Theory: The

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Compute the condition number κ(L_F) = λ_max / λ₂.          Theory: High conditio

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Comprehensive Laplacian verification suite.          Checks:         1. Symmetry

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Create config from a flat dictionary (e.g., loaded from YAML).

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Compute the spectral gap (λ₂ - λ₁) of the sheaf Laplacian.          Theory: The

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Compute the condition number κ(L_F) = λ_max / λ₂.          Theory: High conditio

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Comprehensive Laplacian verification suite.          Checks:         1. Symmetry

## Knowledge Gaps
- **213 isolated node(s):** `Load real BRCA data and build train/val splits for hyperparameter search.     Mi`, `Typed configuration schema for TS-GNN.  REPLACES the raw dict-based config patte`, `Configuration for data loading and preprocessing.`, `Configuration for TS-GNN model architecture.`, `Configuration for the composite loss function.` (+208 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 35`** (2 nodes): `diagnose_download.py`, `test_url()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (2 nodes): `allele_embeddings.py`, `generate_tp53_embeddings()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (2 nodes): `debug_brca.py`, `download_file()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `generate_notebook.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `test_api.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `test_geo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `test_urls.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `test_url_status.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `sync_down.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `sync_up.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `compute_md5s.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Create config from a flat dictionary (e.g., loaded from YAML).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Stack all node features: (K, N, d_input).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Stack all edge weights: (K, E).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Forward pass through the model.          Args:             node_features_seq: (K`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Inference-time prediction (no grad, returns predictions only).          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Compute the spectral gap (λ₂ - λ₁) of the sheaf Laplacian.          Theory: The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Compute the condition number κ(L_F) = λ_max / λ₂.          Theory: High conditio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Comprehensive Laplacian verification suite.          Checks:         1. Symmetry`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Create config from a flat dictionary (e.g., loaded from YAML).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Compute the spectral gap (λ₂ - λ₁) of the sheaf Laplacian.          Theory: The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Compute the condition number κ(L_F) = λ_max / λ₂.          Theory: High conditio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Comprehensive Laplacian verification suite.          Checks:         1. Symmetry`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TSGNN` connect `Community 1` to `Community 0`, `Community 8`, `Community 2`, `Community 9`?**
  _High betweenness centrality (0.230) - this node is a cross-community bridge._
- **Why does `TS-GNN Visualization Utilities.` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 8`, `Community 21`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `VectorizedSheafDiffusion` connect `Community 0` to `Community 1`, `Community 13`, `Community 22`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 136 inferred relationships involving `TSGNN` (e.g. with `Run All Baseline Comparisons.  Usage:     python scripts/run_benchmarks.py --con` and `End-to-End TS-GNN Pipeline.  Phase 6: Executes the complete pipeline from raw da`) actually correct?**
  _`TSGNN` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `TSGNNLoss` (e.g. with `TS-GNN Visualization Utilities.` and `EarlyStopping`) actually correct?**
  _`TSGNNLoss` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `EvolveGCN` (e.g. with `Run All Baseline Comparisons.  Usage:     python scripts/run_benchmarks.py --con` and `TSGNN`) actually correct?**
  _`EvolveGCN` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `TemporalGAT` (e.g. with `Run All Baseline Comparisons.  Usage:     python scripts/run_benchmarks.py --con` and `TSGNN`) actually correct?**
  _`TemporalGAT` has 53 INFERRED edges - model-reasoned connections that need verification._
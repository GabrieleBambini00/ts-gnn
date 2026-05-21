# TS-GNN API Reference

## Core Model

### `TSGNN(num_nodes, num_edges, stalk_dim, input_dim, esm_dim, conditioning_dim, edge_index, num_diffusion_steps=3, use_allele_conditioning=True)`

The main Temporal Sheaf Graph Neural Network model.

| Parameter | Type | Description |
|---|---|---|
| `num_nodes` | `int` | N — number of genes/TFs in the GRN |
| `num_edges` | `int` | E — number of directed regulatory edges |
| `stalk_dim` | `int` | d — sheaf stalk dimension (recommended: 2–8) |
| `input_dim` | `int` | Dimension of input gene expression features |
| `esm_dim` | `int` | Dimension of ESM-2 protein embeddings (1280 for ESM-2) |
| `conditioning_dim` | `int` | FiLM conditioning latent dimension |
| `edge_index` | `Tensor (2, E)` | Directed edge indices |
| `num_diffusion_steps` | `int` | Number of sheaf diffusion iterations per time step |
| `use_allele_conditioning` | `bool` | If `False`, disables FiLM (ablation mode) |

**Forward signature:**
```python
predictions, maps_trajectory, laplacians = model(node_features_seq, allele_embedding)
```

| Return | Shape | Description |
|---|---|---|
| `predictions` | `List[Tensor(N, input_dim)]` | K predicted expression profiles |
| `maps_trajectory` | `List[Tensor(E, 2, d, d)]` | K restriction map snapshots |
| `laplacians` | `List[Tensor(Nd, Nd)]` | K sheaf connection Laplacians |

**Key methods:**
- `decompose_rewiring(node_seq, allele_emb)` → context Laplacian, full Laplacian, delta (allele-specific perturbation)
- `count_parameters()` → `Dict[str, int]` with per-module breakdown

---

## Sheaf Layer

### `SheafDiffusionLayer(num_nodes, num_edges, stalk_dim, edge_index)`

Computes the sheaf connection Laplacian L_F and performs diffusion.

**Mathematical definition:**

$$L_F[u \cdot d : (u+1) \cdot d, \; v \cdot d : (v+1) \cdot d] = -F_{u,e}^\top F_{v,e}$$

$$L_F[u \cdot d : (u+1) \cdot d, \; u \cdot d : (u+1) \cdot d] \mathrel{+}= F_{u,e}^\top F_{u,e}$$

Where $F_{u,e} \in \mathbb{R}^{d \times d}$ are the learnable restriction maps.

**Key methods:**
- `compute_connection_laplacian(restriction_maps=None)` → `Tensor(Nd, Nd)`
- `sheaf_diffusion(x, L_F)` → `Tensor(N, d)` — one diffusion step

---

## Temporal Evolution

### `TemporalRestrictionEvolution(stalk_dim, hidden_dim=0)`

GRU that evolves restriction maps over pseudotime.

```python
new_maps = temporal_evolution(current_maps, node_features, edge_index)
# current_maps: (E, 2, d, d) → new_maps: (E, 2, d, d)
```

The GRU is **shared across all edges** for parameter efficiency.

---

## FiLM Conditioning

### `AlleleFiLM(esm_dim, conditioning_dim, modulation_dim)`

Feature-wise Linear Modulation from ESM-2 allele embeddings.

```python
modulated = film(allele_embedding, gru_output)
# Computes: γ(z) * x + β(z), where z = ESM-2 embedding
```

**Identity initialization:** γ=1, β=0 at init → starts as identity transform.

### `ZeroFiLM()`

Identity FiLM (always returns input unchanged). Used in ablation mode.

---

## Loss Function

### `TSGNNLoss(lambda_1=0.1, lambda_2=0.5, lambda_3=0.01, tau=0.5)`

Composite 4-term loss: $L = L_{expr} + \lambda_1 L_{topo} + \lambda_2 L_{regulon} + \lambda_3 L_{sparse}$

| Term | Description | Depends on |
|---|---|---|
| $L_{expr}$ | MSE between predicted and target expression | predictions, targets |
| $L_{topo}$ | Hinge loss on Laplacian change rate (temporal smoothness) | sheaf_laplacians |
| $L_{regulon}$ | Differentiable Spearman correlation with VIPER activity | predicted_activity, viper_activity |
| $L_{sparse}$ | Frobenius norm of ∂L_F/∂z_allele (locality constraint) | allele_embedding |

---

## Evaluation Metrics

| Function | Metric | Returns |
|---|---|---|
| `cross_cancer_zero_shot(pred, obs)` | Pearson r on cross-cancer edges | `{pearson_r, p_value}` |
| `allele_edge_enrichment(top, validated, total)` | Fisher exact test vs TargetGeneReg | `{fisher_p, odds_ratio}` |
| `chromatin_remodeler_concordance(hw, chip, total)` | Jaccard with ChIP-seq peaks | `{jaccard_index, fisher_p}` |
| `depmap_dependency_correlation(hub, depmap)` | Spearman ρ with CRISPR screens | `{spearman_rho, p_value}` |
| `rewiring_distinguishability(L_a, L_b)` | Permutation test on Frobenius norm | `{p_value, effect_size}` |

---

## Training

### `TSGNNTrainer(model, config, device=None, checkpoint_dir="checkpoints", use_wandb=False)`

```python
results = trainer.train(train_data, val_data, esm_embeddings, viper_activity=None)
```

Features: early stopping, gradient clipping, mixed precision, wandb logging, tqdm progress bars.

---

## Baselines

| Class | Description |
|---|---|
| `EvolveGCN` | GCN with GRU-evolved weights (no sheaf) |
| `TemporalGAT` | Multi-head GAT + GRU + FiLM (no sheaf) |
| `create_tsgnn_no_allele()` | TS-GNN with FiLM disabled (ablation) |
| `CellOracleBaseline` | Wrapper for CellOracle in silico perturbation |
| `DictysBaseline` | Wrapper for Dictys dynamic GRN inference |

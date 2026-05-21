# TS-GNN Architecture

## System Overview

```mermaid
graph TD
    subgraph "Input"
        A["scRNA-seq (K time bins)"] --> B["Node Features (K×N×input_dim)"]
        C["ESM-2 Embedding (1280)"] --> D["Allele Conditioning"]
    end

    subgraph "TS-GNN Core"
        B --> E["Input Projection (→ stalk space)"]
        E --> F["GRU Temporal Evolution"]
        F --> G["Restriction Maps (E×2×d×d)"]
        D --> H["FiLM: γ(z)·R + β(z)"]
        G --> H
        H --> I["Sheaf Laplacian L_F"]
        I --> J["Sheaf Diffusion (× num_steps)"]
        E --> J
        J --> K["Output Projection"]
    end

    subgraph "Output"
        K --> L["Predicted Expression (K×N×input_dim)"]
        I --> M["Laplacian Trajectory (K)"]
        G --> N["Maps Trajectory (K)"]
    end

    subgraph "Loss"
        L --> O["L_expr (MSE)"]
        M --> P["L_topo (Temporal Smoothness)"]
        L --> Q["L_regulon (VIPER Correlation)"]
        I --> R["L_sparse (∂L/∂z Frobenius)"]
        O --> S["Total Loss"]
        P --> S
        Q --> S
        R --> S
    end
```

## Data Flow Pipeline

```mermaid
flowchart LR
    subgraph "Phase 1: Data"
        A1[GEO Download] --> A2[QC + Normalize]
        A2 --> A3[Pseudotime Binning]
        A3 --> A4["Temporal Graphs (K bins)"]
    end

    subgraph "Phase 1C: Priors"
        B1[JASPAR Motifs] --> B2[Multi-Layer GRN]
        B3[STRING PPI] --> B2
        B4[RegNetwork] --> B2
    end

    subgraph "Phase 1D: Embeddings"
        C1[TP53 Sequences] --> C2[ESM-2 Model]
        C2 --> C3["Per-Allele Embeddings"]
    end

    A4 --> D[TS-GNN Training]
    B2 --> D
    C3 --> D

    D --> E[Evaluation]
    E --> F1[Zero-Shot Test]
    E --> F2[Fisher Enrichment]
    E --> F3[ChIP-seq Overlap]
    E --> F4[DepMap Correlation]
    E --> F5[Permutation Test]
```

## Component Interactions

```mermaid
classDiagram
    class AbstractGRNModel {
        <<abstract>>
        +forward(node_seq, allele_emb)
        +count_parameters()
        +predict(node_seq, allele_emb)
        +summary()
    }

    class TSGNN {
        -SheafDiffusionLayer sheaf_layer
        -TemporalRestrictionEvolution temporal
        -AlleleFiLM allele_film
        +forward()
        +decompose_rewiring()
    }

    class EvolveGCN {
        -GRUCell weight_gru
        +forward()
    }

    class TemporalGAT {
        -GATLayer[] attn_layers
        -AlleleFiLM allele_film
        +forward()
    }

    AbstractGRNModel <|-- TSGNN
    AbstractGRNModel <|-- EvolveGCN
    AbstractGRNModel <|-- TemporalGAT

    class SheafDiffusionLayer {
        -restriction_maps: Parameter(E,2,d,d)
        +compute_connection_laplacian()
        +sheaf_diffusion(x, L_F)
    }

    class AlleleFiLM {
        +forward(z_allele, x): γ·x + β
        +get_conditioning_vector(z)
    }

    class TemporalRestrictionEvolution {
        -GRUCell gru
        +forward(maps, features, edge_index)
    }

    TSGNN *-- SheafDiffusionLayer
    TSGNN *-- AlleleFiLM
    TSGNN *-- TemporalRestrictionEvolution
```

## Key Design Decisions

### 1. FiLM vs Hypernetwork

**Chosen: FiLM** — Feature-wise Linear Modulation

| | FiLM | Hypernetwork |
|---|---|---|
| Parameters | O(k) — 2 vectors per allele | O(k·p) — full weight matrix |
| Expressivity | Per-element scaling + shift | Arbitrary weight generation |
| Stability | Identity init (γ=1, β=0) | Hard to initialize stably |
| Interpretability | γ shows which dimensions matter | Black box |

**Rationale:** FiLM is sufficient because allele conditioning modulates *existing* restriction map dynamics rather than generating entirely new ones. The identity initialization ensures the model starts with the context-only Laplacian and learns allele-specific perturbations incrementally.

### 2. Stalk Dimension d=4

- d=1: reduces to standard graph Laplacian (no sheaf benefit)
- d=2: minimal sheaf structure, may underfit heterophily
- **d=4: good balance** of expressivity (16 params per map) and efficiency
- d=8: 64 params per map, Laplacian is (4000×4000) for N=500

### 3. Sparsity Loss via Autograd

The sparsity loss ‖∂L_F/∂z_allele‖_F penalizes the norm of the Jacobian of the Laplacian w.r.t. the allele embedding. This is computed via:

```python
grad = torch.autograd.grad(L_F.sum(), allele_embedding, create_graph=True)
sparsity = grad[0].norm()
```

This enforces **k-hop locality**: the allele-specific perturbation should affect only a local neighborhood, not the entire Laplacian.

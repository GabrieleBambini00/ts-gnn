"""
Typed configuration schema for TS-GNN.

REPLACES the raw dict-based config pattern with @dataclass + validation.
This makes configuration errors caught at initialization rather than
buried in training failures.

Impact:
  - Code Quality: dict → typed dataclass (IDE autocomplete, type checking)
  - Usability: informative error messages on invalid config combinations
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    K: int = 10                         # Number of pseudotime bins
    n_genes: int = 500                  # Number of genes (N)
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    max_pct_mito: float = 20.0
    batch_key: Optional[str] = None     # Batch correction key
    n_pcs: int = 50                     # PCA components
    data_dir: str = "data"

    def validate(self):
        assert self.K >= 2, f"K must be ≥ 2 (got {self.K}). Need at least 2 time steps."
        assert self.n_genes >= 10, f"n_genes must be ≥ 10 (got {self.n_genes})."
        assert 0 < self.max_pct_mito < 100, f"max_pct_mito must be in (0, 100)."


@dataclass
class ModelConfig:
    """Configuration for TS-GNN model architecture."""
    stalk_dim: int = 4                  # d: sheaf stalk dimension
    input_dim: int = 50                 # Gene feature dimension
    esm_dim: int = 1280                 # ESM-2 embedding dimension
    conditioning_dim: int = 128         # FiLM conditioning latent dim
    num_diffusion_steps: int = 3        # Sheaf diffusion iterations per t
    use_allele_conditioning: bool = True
    num_edges: int = 200                # E: number of regulatory edges

    def validate(self):
        assert self.stalk_dim >= 2, (
            f"stalk_dim must be ≥ 2 (got {self.stalk_dim}). "
            f"d=1 reduces sheaf to standard graph Laplacian (no geometric benefit)."
        )
        assert self.num_diffusion_steps >= 1, (
            f"num_diffusion_steps must be ≥ 1 (got {self.num_diffusion_steps})."
        )
        assert self.esm_dim > 0, "esm_dim must be positive."
        if self.stalk_dim > 8:
            import warnings
            warnings.warn(
                f"stalk_dim={self.stalk_dim} is unusually large. "
                f"Laplacian will be {self.input_dim * self.stalk_dim}×"
                f"{self.input_dim * self.stalk_dim} = "
                f"{(self.input_dim * self.stalk_dim)**2 / 1e6:.1f}M entries. "
                f"Consider d ∈ [2, 8] for efficiency.",
                stacklevel=2,
            )


@dataclass
class LossConfig:
    """Configuration for the composite loss function."""
    lambda_1: float = 0.1              # Topological preservation weight
    lambda_2: float = 0.5              # Regulon activity weight
    lambda_3: float = 0.01             # Sparsity weight
    tau: float = 0.5                   # Hinge threshold for topological loss

    def validate(self):
        for name, val in [("lambda_1", self.lambda_1), ("lambda_2", self.lambda_2),
                          ("lambda_3", self.lambda_3)]:
            assert val >= 0, f"{name} must be ≥ 0 (got {val})."
        assert self.tau > 0, f"tau must be > 0 (got {self.tau})."
        if self.lambda_2 > 0 and self.lambda_3 > 0:
            import warnings
            warnings.warn(
                "Both regulon (λ₂) and sparsity (λ₃) losses are active. "
                "Ensure VIPER activity data and allele embeddings with grad are provided.",
                stacklevel=2,
            )


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    lr: float = 1e-3
    max_epochs: int = 500
    patience: int = 20                  # Early stopping patience
    max_grad_norm: float = 1.0          # Gradient clipping
    weight_decay: float = 0.0
    mixed_precision: bool = True
    use_wandb: bool = False
    wandb_project: str = "tsgnn"
    run_name: Optional[str] = None

    def validate(self):
        assert self.lr > 0, f"lr must be > 0 (got {self.lr})."
        assert self.max_epochs >= 1, f"max_epochs must be ≥ 1."
        assert self.patience >= 1, f"patience must be ≥ 1."
        assert self.max_grad_norm > 0, f"max_grad_norm must be > 0."
        if self.lr > 0.1:
            import warnings
            warnings.warn(
                f"lr={self.lr} is very high for sheaf diffusion. "
                f"Consider lr ∈ [1e-4, 1e-2] to avoid gradient explosion.",
                stacklevel=2,
            )


@dataclass
class AblationConfig:
    """Configuration for ablation study parameters."""
    stalk_dims: List[int] = field(default_factory=lambda: [2, 4, 8])
    diffusion_steps: List[int] = field(default_factory=lambda: [1, 3, 5])
    conditioning_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class TSGNNConfig:
    """Top-level configuration combining all sub-configs."""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def validate(self):
        """Validate all sub-configs and cross-config constraints."""
        self.data.validate()
        self.model.validate()
        self.loss.validate()
        self.training.validate()

        # Cross-config validation
        if self.model.input_dim != self.data.n_genes:
            import warnings
            warnings.warn(
                f"model.input_dim ({self.model.input_dim}) ≠ data.n_genes "
                f"({self.data.n_genes}). These should typically match.",
                stacklevel=2,
            )

    @classmethod
    def from_dict(cls, d: dict) -> "TSGNNConfig":
        """Create config from a flat dictionary (e.g., loaded from YAML)."""
        config = cls(
            seed=d.get("seed", 42),
            data=DataConfig(**{k: v for k, v in d.get("data", {}).items()
                              if k in DataConfig.__dataclass_fields__}),
            model=ModelConfig(**{k: v for k, v in d.get("model", {}).items()
                               if k in ModelConfig.__dataclass_fields__}),
            loss=LossConfig(**{k: v for k, v in d.get("loss", {}).items()
                              if k in LossConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in d.get("training", {}).items()
                                     if k in TrainingConfig.__dataclass_fields__}),
        )
        config.validate()
        return config

    def to_dict(self) -> dict:
        """Serialize to dict for checkpoint saving / wandb logging."""
        from dataclasses import asdict
        return asdict(self)

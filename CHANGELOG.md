# Changelog

All notable changes to TS-GNN are documented in this file.

## [0.2.0] - 2026-03-26

### Added
- Fully vectorized sheaf Laplacian construction (`sheaf_vectorized.py`) — O(E·d²) with zero Python loops
- Abstract base class `AbstractGRNModel` enforcing uniform model interface at compile-time
- Typed configuration schema (`config.py`) with `@dataclass` + validation replacing raw dicts
- Dependency-injected `ModelRegistry` and `MetricRegistry` for plugin-based extensibility
- 13+ property-based and fuzz tests (`test_properties.py`) for mathematical invariant verification
- Synthetic proof-of-concept script (`run_synthetic_proof.py`) demonstrating full E2E pipeline
- Spectral diagnostic tools: `spectral_gap()`, `condition_number()`, `verify_laplacian_properties()`
- `ARCHITECTURE.md` with Mermaid diagrams (system overview, data flow, class diagram)
- `API_REFERENCE.md` with complete function signatures and tensor shapes
- GitHub Actions CI pipeline (`ci.yml`) with Python 3.10/3.11, lint, test, smoke test
- Docker Compose with 3 services: training, testing, demo

### Changed
- Pinned all dependency versions in `pyproject.toml` and `requirements.txt` for exact reproducibility
- Added `__all__` exports to all sub-package `__init__.py` files

## [0.1.0] - 2026-03-01

### Added
- Initial TS-GNN implementation with Sheaf Diffusion + GRU + FiLM
- 5 baseline models (CellOracle, Dictys, EvolveGCN, TemporalGAT, no-allele ablation)
- 5 evaluation metrics (zero-shot, enrichment, ChIP-seq, DepMap, distinguishability)
- Composite 4-term loss function
- Full data acquisition pipeline (8 sources)
- Dockerfile and Makefile

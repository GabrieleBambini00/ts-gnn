# Divergence Report — Three-Copy Situation

**Generated:** Phase 0, Task 0.1 of SHEAF_INTEGRATION_PLAN.md
**Date:** 2026-05-21

## Summary

Three diverged copies of the TS-GNN codebase exist on disk with no shared version history:

| Directory | Role |
|-----------|------|
| `ts-gnn/` | **CANONICAL** (this repo) |
| `ts-gnn local/` | HPC / SLURM variant |
| `ts-gnn-chatgpt/` | Data-API exploration variant |

`ts-gnn/` is declared canonical as of this commit. All future development happens here.

---

## Content unique to `ts-gnn local/` (HPC focus)

These items exist in `ts-gnn local/` but are absent from `ts-gnn/`:

| Item | Notes |
|------|-------|
| `environment_hpc.yml` | Conda env spec tuned for HPC cluster |
| `setup_hpc.sh` | HPC environment setup script |
| `submit_slurm.sh` | Single SLURM job submission script |
| `submit_slurm_array.sh` | SLURM array job: 27-run ablation (3×3×3 grid over `stalk_dim × k_hop × temporal_bins`) |
| `configs/hpc.yaml` | HPC-specific config overrides |
| `requirements.txt` | 573 B — has extra dependencies not in canonical `requirements.txt` (556 B) |
| `TSGNN_Colab_Master.ipynb` | Slightly diverged copy of the Colab notebook |

`ts-gnn local/` **lacks:** `GEMINI.md`, `.gemini/`, `graphify-out/`, `hpc/`.

**To port later:** SLURM submission scripts and `configs/hpc.yaml` should be integrated into `ts-gnn/hpc/` (directory already exists canonically). The extra `requirements.txt` dependencies should be reconciled.

---

## Content unique to `ts-gnn-chatgpt/` (Data-API exploration)

These items exist in `ts-gnn-chatgpt/` but are absent from `ts-gnn/`:

| Item | Notes |
|------|-------|
| `figures/` | Generated figures directory |
| `test_api.py` | Tests for data-download API surface |
| `test_geo.py` | Tests for geo/graph data loading |
| `test_url_status.py` | Tests verifying URL reachability |
| `test_urls.py` | URL reliability exploration tests |
| `requirements.txt` | 613 B — largest of the three; extra packages vs canonical |

`ts-gnn-chatgpt/` **lacks:** any SLURM/HPC files.

**Note:** The four test files (`test_api.py`, `test_geo.py`, `test_url_status.py`, `test_urls.py`) that live in the root of the canonical `ts-gnn/` were ported from this variant. They are already present here and committed in the baseline.

**To port later:** `figures/` generation logic and the extra `requirements.txt` dependencies should be reviewed for inclusion.

---

## Recommendations

1. **Archive** (do not delete) `ts-gnn local/` and `ts-gnn-chatgpt/`. Rename or move them to a clearly labeled archive location (e.g., `_archive/ts-gnn-local-hpc/` and `_archive/ts-gnn-chatgpt/`).
2. **Port HPC scripts** from `ts-gnn local/` into `ts-gnn/hpc/` in a dedicated later task.
3. **Reconcile requirements** by diffing all three `requirements.txt` files and merging needed extras into the canonical one.
4. **Do not work in** `ts-gnn local/` or `ts-gnn-chatgpt/` going forward — changes made there will be silently lost.

---

*This report was generated as part of Phase 0 baseline establishment. It records known divergences so nothing is silently discarded during consolidation.*

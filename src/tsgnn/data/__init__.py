"""TS-GNN Data Loading and Preprocessing."""

import os
from pathlib import Path

from tsgnn.data.splits import LeakageError, assert_no_group_leakage, group_split


def _resolve_data_dir() -> Path:
    """Resolve the project data directory with 3-level fallback.

    Priority:
    1. TSGNN_DATA_DIR environment variable (for custom deployments)
    2. Relative to this package's location (local workspace)
    3. Current working directory (Google Colab / Drive)
    """
    env_dir = os.environ.get("TSGNN_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        # Parents[3] = ts-gnn root from src/tsgnn/data/__init__.py
        return Path(__file__).resolve().parents[3] / "data"
    except (NameError, IndexError):
        return Path(os.getcwd()) / "data"


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"

__all__ = [
    "download", "preprocess", "allele", "grn_construction", "temporal",
    "brca_loader",
    "splits", "LeakageError", "assert_no_group_leakage", "group_split",
    "DATA_DIR", "RAW_DIR", "EXTERNAL_DIR", "_resolve_data_dir",
]

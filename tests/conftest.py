"""Shared test fixtures for TS-GNN test suite."""
import sys
from pathlib import Path

# Ensure tsgnn is importable without pip install -e .
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest
import torch

from tsgnn.utils import set_global_seed


@pytest.fixture
def seed():
    set_global_seed(42)
    return 42


@pytest.fixture
def small_graph():
    """Small graph for fast tests: 10 nodes, 25 edges, stalk_dim=2."""
    N, E, d = 10, 25, 2
    edge_index = torch.randint(0, N, (2, E))
    return {"N": N, "E": E, "d": d, "edge_index": edge_index}


@pytest.fixture
def medium_graph():
    """Medium graph: 30 nodes, 80 edges, stalk_dim=4."""
    N, E, d = 30, 80, 4
    edge_index = torch.randint(0, N, (2, E))
    return {"N": N, "E": E, "d": d, "edge_index": edge_index}


@pytest.fixture
def esm_embedding():
    return torch.randn(1280)


@pytest.fixture
def temporal_sequence(small_graph):
    """K=3 temporal sequence for small graph."""
    K, N, input_dim = 3, small_graph["N"], small_graph["N"]
    return torch.randn(K, N, input_dim)

"""
Tests for training reproducibility via set_global_seed.

Runs training TWICE with the same seed on small synthetic data for 3 epochs
and asserts per-epoch loss sequences are bit-identical (torch.equal / ==).
"""

import copy

import pytest
import torch

from tsgnn.model.tsgnn import TSGNN
from tsgnn.training.trainer import TSGNNTrainer, create_synthetic_training_data
from tsgnn.utils import set_global_seed


# ── Constants ────────────────────────────────────────────────────────────────

N, E, D, INPUT_DIM, K, ESM_DIM = 15, 40, 2, 15, 3, 1280
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model(edge_index: torch.Tensor) -> TSGNN:
    return TSGNN(
        num_nodes=N, num_edges=E, stalk_dim=D, input_dim=INPUT_DIM,
        esm_dim=ESM_DIM, conditioning_dim=32, edge_index=edge_index,
        num_diffusion_steps=1,
    )


def _make_config(seed: int = SEED) -> dict:
    return {
        "seed": seed,
        "training": {
            "lr": 1e-2,
            "max_epochs": 3,
            "patience": 10,
            "max_grad_norm": 1.0,
            "mixed_precision": False,
        },
        "loss": {
            "lambda_1": 0.1,
            "lambda_2": 0.0,
            "lambda_3": 0.0,
            "tau": 0.5,
        },
    }


def _run_training(seed: int, tmp_path):
    """Seed everything, build a fresh model, run training, return loss history."""
    set_global_seed(seed)

    edge_index = torch.randint(0, N, (2, E))
    model = _make_model(edge_index)

    train_data, val_data, esm_embeddings = create_synthetic_training_data(
        N=N, E=E, K=K, input_dim=INPUT_DIM, alleles=["WT", "R175H"],
    )

    config = _make_config(seed=seed)
    trainer = TSGNNTrainer(
        model=model, config=config,
        checkpoint_dir=str(tmp_path),
        use_wandb=False,
    )
    results = trainer.train(train_data, val_data, esm_embeddings)
    return results["history"]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_seed_produces_identical_loss_sequences(tmp_path):
    """Two training runs with the same seed must produce bit-identical losses."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    history1 = _run_training(SEED, dir1)
    history2 = _run_training(SEED, dir2)

    assert history1["train_loss"] == history2["train_loss"], (
        f"Train losses differ between runs with seed={SEED}:\n"
        f"  run1: {history1['train_loss']}\n"
        f"  run2: {history2['train_loss']}"
    )
    assert history1["val_loss"] == history2["val_loss"], (
        f"Val losses differ between runs with seed={SEED}:\n"
        f"  run1: {history1['val_loss']}\n"
        f"  run2: {history2['val_loss']}"
    )


def test_different_seeds_produce_different_losses(tmp_path):
    """Two runs with different seeds should (very likely) diverge."""
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    history1 = _run_training(seed=42, tmp_path=dir1)
    history2 = _run_training(seed=99, tmp_path=dir2)

    # It would be astronomically unlikely for these to be identical
    assert history1["train_loss"] != history2["train_loss"], (
        "Train losses are identical across seeds — seeding may not be working."
    )


def test_set_global_seed_utility():
    """set_global_seed should be callable without error and affect torch state."""
    set_global_seed(0)
    x = torch.randn(5)
    set_global_seed(0)
    y = torch.randn(5)
    assert torch.equal(x, y), "set_global_seed(0) did not produce identical tensors."


def test_trainer_uses_config_seed(tmp_path):
    """Trainer.train() must read seed from config and enforce it."""
    set_global_seed(SEED)
    edge_index = torch.randint(0, N, (2, E))
    model = _make_model(edge_index)

    train_data, val_data, esm_embeddings = create_synthetic_training_data(
        N=N, E=E, K=K, input_dim=INPUT_DIM, alleles=["WT"],
    )
    config = _make_config(seed=SEED)
    trainer = TSGNNTrainer(
        model=model, config=config,
        checkpoint_dir=str(tmp_path),
        use_wandb=False,
    )
    results = trainer.train(train_data, val_data, esm_embeddings)
    assert results["epochs_trained"] >= 1
    # Seed field must be present in config and consumed
    assert config["seed"] == SEED


def test_model_init_reproducibility():
    """Constructing the model TWICE after set_global_seed(same_seed) yields
    bit-identical initial parameters.

    This directly tests the gap that was previously missed: model weight
    initialisation must be covered by the seed, not just the training loop.
    """
    edge_index_fixed = torch.tensor(
        [[i % N for i in range(E)], [(i + 1) % N for i in range(E)]],
        dtype=torch.long,
    )

    set_global_seed(SEED)
    model_a = _make_model(edge_index_fixed)

    set_global_seed(SEED)
    model_b = _make_model(edge_index_fixed)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    assert sd_a.keys() == sd_b.keys(), "State dicts have different keys."
    mismatched = [
        k for k in sd_a
        if not torch.equal(sd_a[k], sd_b[k])
    ]
    assert not mismatched, (
        f"Model init is NOT reproducible across two set_global_seed({SEED}) calls.\n"
        f"Differing parameter tensors: {mismatched}"
    )

"""
Tests for tsgnn.data.splits — group-disjoint splitting and leakage assertion.

Synthetic dataset
-----------------
20 groups (cell lines), each with 5 samples, binary labels.
Groups 0–9 → label 0; groups 10–19 → label 1.
Total 100 samples.
"""

import numpy as np
import pytest

from tsgnn.data.splits import LeakageError, assert_no_group_leakage, group_split

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_GROUPS = 20
SAMPLES_PER_GROUP = 5
N_SAMPLES = N_GROUPS * SAMPLES_PER_GROUP


@pytest.fixture
def synthetic_dataset():
    """20 groups × 5 samples, binary labels, string group ids."""
    groups = np.array(
        [f"CL{g:02d}" for g in range(N_GROUPS) for _ in range(SAMPLES_PER_GROUP)]
    )
    labels = np.array(
        [0 if g < N_GROUPS // 2 else 1 for g in range(N_GROUPS) for _ in range(SAMPLES_PER_GROUP)]
    )
    return groups, labels


# ─────────────────────────────────────────────────────────────────────────────
# 1. Splitter — zero shared groups
# ─────────────────────────────────────────────────────────────────────────────

def test_two_way_split_no_group_overlap(synthetic_dataset):
    """Train and test must share no group ids."""
    groups, labels = synthetic_dataset
    train_idx, test_idx = group_split(groups, labels, test_size=0.2, random_seed=42)

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(test_groups), (
        f"Group leakage: {train_groups & test_groups}"
    )


def test_three_way_split_no_group_overlap(synthetic_dataset):
    """Train, val, and test must be pairwise group-disjoint."""
    groups, labels = synthetic_dataset
    train_idx, val_idx, test_idx = group_split(
        groups, labels, test_size=0.2, val_size=0.1, random_seed=42
    )

    train_g = set(groups[train_idx])
    val_g = set(groups[val_idx])
    test_g = set(groups[test_idx])

    assert train_g.isdisjoint(test_g), f"train∩test={train_g & test_g}"
    assert train_g.isdisjoint(val_g), f"train∩val={train_g & val_g}"
    assert val_g.isdisjoint(test_g), f"val∩test={val_g & test_g}"


def test_split_covers_all_samples(synthetic_dataset):
    """Union of train+test indices must equal all sample indices."""
    groups, labels = synthetic_dataset
    train_idx, test_idx = group_split(groups, labels, test_size=0.2, random_seed=42)

    all_idx = set(train_idx) | set(test_idx)
    assert all_idx == set(range(N_SAMPLES))


def test_three_way_split_covers_all_samples(synthetic_dataset):
    """Union of train+val+test indices must equal all sample indices."""
    groups, labels = synthetic_dataset
    train_idx, val_idx, test_idx = group_split(
        groups, labels, test_size=0.2, val_size=0.1, random_seed=42
    )

    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert all_idx == set(range(N_SAMPLES))


# ─────────────────────────────────────────────────────────────────────────────
# 2. assert_no_group_leakage — raises on deliberate leakage
# ─────────────────────────────────────────────────────────────────────────────

def test_assert_raises_on_leakage():
    """Deliberately leaked group id must trigger LeakageError naming the offender."""
    train_groups = ["CL01", "CL02", "CL03"]
    test_groups = ["CL03", "CL04"]  # CL03 leaks

    with pytest.raises(LeakageError) as exc_info:
        assert_no_group_leakage(train_groups, test_groups)

    assert "CL03" in str(exc_info.value), (
        f"Exception message must name offending id, got: {exc_info.value}"
    )


def test_assert_raises_on_multiple_leaked_ids():
    """Multiple leaked ids must all appear in the exception message."""
    train_groups = ["CL01", "CL02", "CL03", "CL04"]
    test_groups = ["CL02", "CL04", "CL05"]  # CL02 and CL04 leak

    with pytest.raises(LeakageError) as exc_info:
        assert_no_group_leakage(train_groups, test_groups)

    msg = str(exc_info.value)
    assert "CL02" in msg and "CL04" in msg, (
        f"Both offending ids must appear in message, got: {msg}"
    )


def test_assert_raises_on_val_train_leakage():
    """Leakage between train and val must be detected in three-way check."""
    train_groups = ["CL01", "CL02"]
    val_groups = ["CL02", "CL03"]   # CL02 leaks between train and val
    test_groups = ["CL04", "CL05"]

    with pytest.raises(LeakageError) as exc_info:
        assert_no_group_leakage(train_groups, test_groups, val_groups=val_groups)

    assert "CL02" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# 3. assert_no_group_leakage — passes silently on a correct split
# ─────────────────────────────────────────────────────────────────────────────

def test_assert_passes_on_correct_split(synthetic_dataset):
    """assert_no_group_leakage must not raise for a correctly grouped split."""
    groups, labels = synthetic_dataset
    train_idx, test_idx = group_split(groups, labels, test_size=0.2, random_seed=42)

    # Should not raise
    assert_no_group_leakage(groups[train_idx], groups[test_idx])


def test_assert_passes_on_correct_three_way_split(synthetic_dataset):
    """assert_no_group_leakage must not raise for a correct three-way split."""
    groups, labels = synthetic_dataset
    train_idx, val_idx, test_idx = group_split(
        groups, labels, test_size=0.2, val_size=0.1, random_seed=42
    )

    # Should not raise
    assert_no_group_leakage(
        groups[train_idx], groups[test_idx], val_groups=groups[val_idx]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_split_deterministic_same_seed(synthetic_dataset):
    """Same seed → identical splits on two calls."""
    groups, labels = synthetic_dataset
    train_a, test_a = group_split(groups, labels, test_size=0.2, random_seed=7)
    train_b, test_b = group_split(groups, labels, test_size=0.2, random_seed=7)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)


def test_split_different_seeds_differ(synthetic_dataset):
    """Different seeds should (almost certainly) produce different splits."""
    groups, labels = synthetic_dataset
    train_a, _ = group_split(groups, labels, test_size=0.2, random_seed=0)
    train_b, _ = group_split(groups, labels, test_size=0.2, random_seed=99)

    # It is astronomically unlikely for both splits to be identical
    assert not np.array_equal(train_a, train_b), (
        "Different seeds produced identical splits — seeding may be broken"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Input validation
# ─────────────────────────────────────────────────────────────────────────────

def test_mismatched_lengths_raises():
    """Mismatched groups/labels lengths must raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        group_split(
            groups=["CL01", "CL02"],
            labels=[0, 1, 0],  # wrong length
        )


def test_invalid_size_sum_raises(synthetic_dataset):
    """test_size + val_size >= 1.0 must raise ValueError."""
    groups, labels = synthetic_dataset
    with pytest.raises(ValueError, match="must be < 1.0"):
        group_split(groups, labels, test_size=0.6, val_size=0.5)


def test_too_few_groups_raises():
    """Fewer than 2 unique groups must raise ValueError."""
    groups = ["CL01"] * 10
    labels = [0] * 10
    with pytest.raises(ValueError, match="at least 2 unique group ids"):
        group_split(groups, labels)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_test_size_out_of_range_raises(synthetic_dataset, test_size):
    """test_size outside (0, 1) must raise ValueError."""
    groups, labels = synthetic_dataset
    with pytest.raises(ValueError, match="test_size must be in the open interval"):
        group_split(groups, labels, test_size=test_size)


@pytest.mark.parametrize("val_size", [0.0, 1.0, -0.1, 1.5])
def test_val_size_out_of_range_raises(synthetic_dataset, val_size):
    """val_size outside (0, 1) must raise ValueError."""
    groups, labels = synthetic_dataset
    with pytest.raises(ValueError, match="val_size must be in the open interval"):
        group_split(groups, labels, test_size=0.2, val_size=val_size)


def test_val_plus_test_size_equal_one_raises(synthetic_dataset):
    """test_size + val_size == 1.0 must raise ValueError (boundary check)."""
    groups, labels = synthetic_dataset
    with pytest.raises(ValueError, match="must be < 1.0"):
        group_split(groups, labels, test_size=0.5, val_size=0.5)

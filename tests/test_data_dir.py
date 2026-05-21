"""
Task 2.2 — Consistent data-directory resolution tests.

Verifies that all data modules resolve their data directory through the
same _resolve_data_dir() logic defined in tsgnn.data.__init__, and that
resolution respects TSGNN_DATA_DIR, falls back correctly, and is stable
regardless of the current working directory.

Design note:
    Module-level constants like DATA_DIR are evaluated once at import time.
    Re-importing tsgnn.data with a different env var would mutate cached
    module state and potentially contaminate other tests.  To avoid this,
    tests that need to vary TSGNN_DATA_DIR call _resolve_data_dir()
    *directly* (the function itself) after setting the env var via
    monkeypatch — rather than reloading modules.  This is safe because
    _resolve_data_dir() reads os.environ fresh on every call.
"""
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. _resolve_data_dir is importable and callable
# ---------------------------------------------------------------------------

def test_resolve_data_dir_callable():
    """_resolve_data_dir must be exported from tsgnn.data and be callable."""
    import tsgnn.data as data_pkg
    assert hasattr(data_pkg, "_resolve_data_dir"), (
        "_resolve_data_dir must be exported from tsgnn.data"
    )
    assert callable(data_pkg._resolve_data_dir)


# ---------------------------------------------------------------------------
# 2. TSGNN_DATA_DIR env var is honoured
# ---------------------------------------------------------------------------

def test_env_var_overrides_resolution(monkeypatch, tmp_path):
    """When TSGNN_DATA_DIR is set, _resolve_data_dir() must return exactly that path."""
    from tsgnn.data import _resolve_data_dir

    custom_dir = str(tmp_path / "custom_data")
    monkeypatch.setenv("TSGNN_DATA_DIR", custom_dir)

    resolved = _resolve_data_dir()
    assert resolved == Path(custom_dir), (
        f"Expected {custom_dir!r}, got {resolved!r}"
    )


def test_env_var_takes_priority_over_package_relative(monkeypatch, tmp_path):
    """TSGNN_DATA_DIR must beat the package-relative fallback."""
    from tsgnn.data import _resolve_data_dir

    custom_dir = str(tmp_path / "priority_data")
    monkeypatch.setenv("TSGNN_DATA_DIR", custom_dir)

    resolved = _resolve_data_dir()
    # Must not be the package-relative path
    assert resolved == Path(custom_dir), (
        f"Env var must take priority; got {resolved!r} instead of {custom_dir!r}"
    )


# ---------------------------------------------------------------------------
# 3. Fallback when env var is absent
# ---------------------------------------------------------------------------

def test_fallback_without_env_var(monkeypatch):
    """When TSGNN_DATA_DIR is absent, resolution falls back to package-relative or cwd."""
    from tsgnn.data import _resolve_data_dir

    monkeypatch.delenv("TSGNN_DATA_DIR", raising=False)
    resolved = _resolve_data_dir()

    assert isinstance(resolved, Path), "Fallback resolution must return a Path"
    assert resolved.name == "data", (
        f"Fallback data dir should end in 'data', got: {resolved}"
    )


# ---------------------------------------------------------------------------
# 4. All three modules use the centralised DATA_DIR / EXTERNAL_DIR
# ---------------------------------------------------------------------------

def test_grn_construction_uses_central_external_dir():
    """grn_construction must use EXTERNAL_DIR from tsgnn.data, not a hardcoded path."""
    import tsgnn.data as data_pkg
    import tsgnn.data.grn_construction as grn_mod

    assert grn_mod.EXTERNAL_DIR == data_pkg.EXTERNAL_DIR, (
        f"grn_construction.EXTERNAL_DIR ({grn_mod.EXTERNAL_DIR}) "
        f"!= tsgnn.data.EXTERNAL_DIR ({data_pkg.EXTERNAL_DIR})"
    )


def test_preprocess_uses_central_data_dir():
    """preprocess must use DATA_DIR from tsgnn.data, not a hardcoded path."""
    import tsgnn.data as data_pkg
    import tsgnn.data.preprocess as pre_mod

    assert pre_mod.DATA_DIR == data_pkg.DATA_DIR, (
        f"preprocess.DATA_DIR ({pre_mod.DATA_DIR}) "
        f"!= tsgnn.data.DATA_DIR ({data_pkg.DATA_DIR})"
    )


def test_allele_uses_central_data_dir():
    """allele must use DATA_DIR from tsgnn.data, not a hardcoded path."""
    import tsgnn.data as data_pkg
    import tsgnn.data.allele as allele_mod

    assert allele_mod.DATA_DIR == data_pkg.DATA_DIR, (
        f"allele.DATA_DIR ({allele_mod.DATA_DIR}) "
        f"!= tsgnn.data.DATA_DIR ({data_pkg.DATA_DIR})"
    )


# ---------------------------------------------------------------------------
# 5. Resolution is stable regardless of cwd (function-level, not constant)
# ---------------------------------------------------------------------------

def test_resolution_stable_regardless_of_cwd(monkeypatch, tmp_path):
    """
    When TSGNN_DATA_DIR is set, _resolve_data_dir() must return the same path
    regardless of the current working directory.

    This verifies there is no cwd-dependent side-channel in the resolution
    function itself (the parents[3] fallback only kicks in when __file__ is
    unavailable, and even then falls back to cwd/data — but the env-var path
    must be returned in preference to both).
    """
    from tsgnn.data import _resolve_data_dir

    custom_dir = str(tmp_path / "stable_data")
    monkeypatch.setenv("TSGNN_DATA_DIR", custom_dir)

    # Resolve from the original working directory
    resolved_orig = _resolve_data_dir()

    # Now move cwd to tmp_path (a different location)
    monkeypatch.chdir(tmp_path)

    # Must return the same path (env var overrides cwd)
    resolved_moved = _resolve_data_dir()

    assert resolved_orig == resolved_moved, (
        f"Resolution changed after chdir: {resolved_orig!r} → {resolved_moved!r}. "
        "Data-dir resolution must be cwd-independent when TSGNN_DATA_DIR is set."
    )


# ---------------------------------------------------------------------------
# 6. download module sees the same DATA_DIR as the package
# ---------------------------------------------------------------------------

def test_download_module_sees_same_data_dir():
    """download must import DATA_DIR from tsgnn.data, not recompute independently."""
    import tsgnn.data as data_pkg
    from tsgnn.data import download

    assert download.DATA_DIR == data_pkg.DATA_DIR, (
        f"download.DATA_DIR ({download.DATA_DIR}) "
        f"!= tsgnn.data.DATA_DIR ({data_pkg.DATA_DIR})"
    )


# ---------------------------------------------------------------------------
# 7. No parents[3] hardcode in target modules
# ---------------------------------------------------------------------------

def test_no_hardcoded_parents3_in_grn_construction():
    """grn_construction.py must not contain a hardcoded parents[3] pattern."""
    import inspect
    from tsgnn.data import grn_construction
    src = inspect.getsource(grn_construction)
    assert "parents[3]" not in src, (
        "grn_construction.py must not hardcode parents[3]; use tsgnn.data.EXTERNAL_DIR instead."
    )


def test_no_hardcoded_parents3_in_preprocess():
    """preprocess.py must not contain a hardcoded parents[3] pattern."""
    import inspect
    from tsgnn.data import preprocess
    src = inspect.getsource(preprocess)
    assert "parents[3]" not in src, (
        "preprocess.py must not hardcode parents[3]; use tsgnn.data.DATA_DIR instead."
    )


def test_no_hardcoded_parents3_in_allele():
    """allele.py must not contain a hardcoded parents[3] pattern."""
    import inspect
    from tsgnn.data import allele
    src = inspect.getsource(allele)
    assert "parents[3]" not in src, (
        "allele.py must not hardcode parents[3]; use tsgnn.data.DATA_DIR instead."
    )

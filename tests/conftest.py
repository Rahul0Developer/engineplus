"""Shared pytest fixtures.

Tests run against small synthetic C-MAPSS-shaped fixtures (tests/fixtures) so
the suite is fast, hermetic and has no network dependency. The real NASA data
in ./data is exercised by the smoke workflow, not unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    assert FIXTURES.exists(), f"missing test fixtures at {FIXTURES}"
    return FIXTURES


@pytest.fixture(scope="session")
def fd001_ds(fixtures_dir):
    from enginepulse.data.loader import load_dataset

    return load_dataset(fixtures_dir, "FD001")


@pytest.fixture(scope="session")
def fd002_ds(fixtures_dir):
    from enginepulse.data.loader import load_dataset

    return load_dataset(fixtures_dir, "FD002")


@pytest.fixture(scope="session")
def prepared_fd001(fd001_ds):
    """One shared PreparedData for FD001 — read-only use across tests."""
    from enginepulse.data.preprocessing import prepare_dataset

    return prepare_dataset(fd001_ds, window_len=8, max_rul=30, seed=42)


@pytest.fixture
def tiny_windows():
    """(n=16, w=8, f=5) random windows + targets, deterministic."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 8, 5)).astype("float32")
    y = rng.uniform(0, 30, size=16).astype("float32")
    return X, y

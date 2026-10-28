"""Shared helpers: seeding, logging, config loading, environment stamps."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from enginepulse.utils import (
    file_sha256,
    get_logger,
    git_hash,
    load_config,
    package_versions,
    set_seed,
    write_json,
)


def test_set_seed_makes_rng_reproducible():
    set_seed(11)
    a = np.random.rand(5)
    b = np.random.randn(3)
    set_seed(11)
    assert np.allclose(a, np.random.rand(5))
    assert np.allclose(b, np.random.randn(3))


def test_get_logger_is_child_of_package_namespace():
    log = get_logger("enginepulse.something")
    assert log.name == "enginepulse.something"
    # Calling twice must not stack duplicate handlers.
    n = len(log.handlers)
    get_logger("enginepulse.something")
    assert len(log.handlers) == n


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset: FD002\nwindow_len: 32\nmodel_params:\n  alpha: 0.5\n")
    cfg = load_config(p)
    assert cfg["dataset"] == "FD002" and cfg["model_params"]["alpha"] == 0.5


def test_load_config_missing_file_raises_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_git_hash_returns_string():
    h = git_hash()
    assert isinstance(h, str) and len(h) >= 4  # short hash or "unknown"


def test_package_versions_includes_numpy():
    v = package_versions()
    assert "numpy" in v


def test_write_json_creates_parents(tmp_path):
    out = tmp_path / "a" / "b" / "x.json"
    write_json(out, {"k": [1, 2]})
    assert "k" in out.read_text()


def test_file_sha256_stable(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"abc")
    assert file_sha256(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

"""Shared helpers: seeding, logging, config loading, git stamp."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if present) torch so runs repeat exactly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # deep extras are optional; classical path must not require torch


def get_logger(name: str) -> logging.Logger:
    """Consistent logger for CLI and library code."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML experiment config into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {p} must contain a mapping at the top level")
    return cfg


def git_hash() -> str:
    """Current commit hash, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    """Versions of the packages that actually change results."""
    mods = ["numpy", "pandas", "sklearn", "xgboost", "torch"]
    versions: dict[str, str] = {}
    for m in mods:
        try:
            mod = __import__(m)
            versions[m] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[m] = "not installed"
    return versions


def write_json(path: str | Path, payload: Any) -> None:
    """Dump JSON with sorted keys; used for result stamps."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

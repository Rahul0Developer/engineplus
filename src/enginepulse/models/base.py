"""The single model interface every EnginePulse model implements.

Design goals:
  * a benchmark harness can train/evaluate/save/load ANY model with the same
    five calls;
  * models consume PreparedData (already leakage-safe) and emit raw RUL-scale
    predictions, so metrics never have to know about normalisation;
  * adding your own model = subclass + register. That is the whole contract.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class RULModel(ABC):
    """Abstract base class for remaining-useful-life regressors."""

    #: human-readable name used in CLI/configs/results tables
    name: str = "base"

    def __init__(self, seed: int = 42, **params: Any) -> None:
        self.seed = seed
        self.params = params
        self.is_fitted_ = False

    # -- required API -------------------------------------------------------
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "RULModel":
        """Train on windows X (n, w, f) and targets y (n,) in RUL cycles."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted RUL (cycles) for each window; shape (n,)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist weights/state so serving can load without retraining."""

    @abstractmethod
    def load(self, path: str | Path) -> "RULModel":
        """Restore state previously written by ``save``."""

    # -- optional API (defaults keep simple models simple) ------------------
    def predict_quantiles(self, X: np.ndarray, quantiles: list[float]) -> dict[float, np.ndarray]:
        """Uncertainty-aware prediction. Default: no intervals (NaN)."""
        n = len(X)
        return {q: np.full(n, np.nan) for q in quantiles}

    def feature_importance(self) -> dict[str, float] | None:
        """Per-sensor importance if the model exposes one natively."""
        return None

    # -- shared helpers -----------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "params": self.params}

    def _write_meta(self, path: str | Path, extra: dict | None = None) -> None:
        """Write meta.json next to a model artifact.

        ``path`` may be the artifact file itself or its directory; either way
        meta.json lands in the same directory as the weights.
        """
        p = Path(path)
        directory = p.parent if p.suffix else p
        directory.mkdir(parents=True, exist_ok=True)
        meta = self.describe()
        if extra:
            meta.update(extra)
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def clip_rul(pred: np.ndarray, max_rul: float | None) -> np.ndarray:
    """Clamp predictions into [0, max_rul]; models share this convention.

    NaN/Inf are treated as invalid model output and floored to 0.0 so a
    degenerate prediction can never silently poison a metric computation.
    """
    arr = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(arr, 0.0, np.inf)
    if max_rul is not None:
        out = np.minimum(out, float(max_rul))
    return out

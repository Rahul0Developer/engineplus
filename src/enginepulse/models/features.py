"""Window -> flat feature vectors shared by the classical models.

Deliberately small and readable: per-sensor mean/std/min/max/last plus the
window-to-window slope (linear trend over time). These are the features that
make gradient boosting competitive with neural nets on this dataset — and
they are cheap enough to compute in the API hot path.
"""

from __future__ import annotations

import numpy as np


def window_features(X: np.ndarray) -> np.ndarray:
    """(n_windows, w, f) -> (n_windows, 6*f) float32 matrix."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"expected 3-D windows, got shape {X.shape}")
    n, w, f = X.shape
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)
    # OLS slope per sensor without a loop: cov(t, x)/var(t)
    tc = t - t.mean()
    xc = X - X.mean(axis=1, keepdims=True)
    slope = (tc[None, :, None] * xc).sum(axis=1) / float((tc * tc).sum())
    feats = np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X[:, -1, :],
            slope,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def feature_names(sensor_names: list[str]) -> list[str]:
    stats = ["mean", "std", "min", "max", "last", "slope"]
    return [f"{s}_{st}" for st in stats for s in sensor_names]

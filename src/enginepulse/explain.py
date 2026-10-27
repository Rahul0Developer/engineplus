"""Per-sensor explainability: permutation importance + native tree importances.

The question practitioners actually ask is "which sensors drive the RUL
estimate?" — so everything aggregates back to sensor columns, even though the
classical models see six statistical features per sensor (mean/std/min/max/
last/slope, in that block order — see models/features.py).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

from enginepulse.models.features import feature_names, window_features

# Number of statistics computed per sensor inside window_features().
_N_STATS = 6


def _sensor_of(feature_name: str, sensor_cols: list[str]) -> str | None:
    """Map 'slope_15' -> '15'. Longest-suffix match so multi-digit codes are safe."""
    best = None
    for s in sensor_cols:
        if feature_name.endswith(f"_{s}") and (best is None or len(s) > len(best)):
            best = s
    return best


class _WindowAdapter(BaseEstimator, RegressorMixin):
    """Adapts a window model (n,w,f) to sklearn's flat-feature (n, 6f) world.

    permutation_importance shuffles individual columns of the flat matrix.
    We rebuild a width-1 pseudo-window from the shuffled features before
    handing them to the wrapped model; sensitivity is still measured correctly
    because each sensor's stats move together only when ALL its columns are
    permuted — which is what we aggregate over afterwards.
    """

    def __init__(self, model) -> None:
        self.model = model

    def predict(self, F: np.ndarray) -> np.ndarray:
        n_sensors = F.shape[1] // _N_STATS
        # Use the 'last' statistic block as the single timestep; shapes stay valid.
        last = F[:, 4 * n_sensors : 5 * n_sensors]
        pseudo = last[:, None, :].astype(np.float32)
        return self.model.predict(pseudo)


def permutation_sensor_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sensor_cols: list[str],
    n_repeats: int = 8,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation importance (RMSE-based), summed back up to per-sensor values.

    Returns {sensor_code: importance}; larger means the model degrades more
    when that sensor's information is destroyed.
    """
    F = window_features(X).astype(np.float64)
    names = feature_names(sensor_cols)
    f_imp = permutation_importance(
        _WindowAdapter(model),
        F,
        np.asarray(y, dtype=np.float64),
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=seed,
    ).importances_mean

    out = {s: 0.0 for s in sensor_cols}
    for nm, val in zip(names, f_imp):
        s = _sensor_of(nm, sensor_cols)
        if s is not None:
            out[s] += float(val)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def native_sensor_importance(model, sensor_cols: list[str]) -> dict[str, float]:
    """Fold a model's native feature_importance() back onto sensors (XGBoost path)."""
    fi = model.feature_importance()
    if not fi:
        return {}
    out = {s: 0.0 for s in sensor_cols}
    matched = 0
    for feat, val in fi.items():
        s = _sensor_of(feat, sensor_cols)
        if s is not None:
            out[s] += float(val)
            matched += 1
    if matched == 0:
        return {}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def top_sensors(importances: dict[str, float], k: int = 8) -> list[tuple[str, float]]:
    """Sorted top-k; used by CLI output and the report."""
    return sorted(importances.items(), key=lambda kv: -kv[1])[:k]

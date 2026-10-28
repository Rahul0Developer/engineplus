"""Per-sensor explainability: permutation importance + native tree importances.

The question practitioners actually ask is "which sensors drive the RUL
estimate?" So importance is always aggregated back up to sensor columns.

Implementation note: we deliberately do NOT run sklearn's
permutation_importance on the flat 6-stats-per-sensor feature matrix that the
classical models consume. Shuffling one statistic column (say `std_15`) while
`mean_15` stays put produces feature combinations the model never saw in
training, which inflates importances with distribution-shift noise. Instead we
permute whole sensor channels in raw window space (n, w, f): every statistic
derived from that sensor moves together, exactly as it would in a real faulty
sensor reading. That is both more honest and simpler than adapting the flat
matrix back into pseudo-windows.
"""

from __future__ import annotations

import numpy as np

# Number of statistics computed per sensor inside window_features().
_N_STATS = 6


def _sensor_of(feature_name: str, sensor_cols: list[str]) -> str | None:
    """Map 'slope_15' -> '15'. Longest-suffix match so multi-digit codes are safe."""
    best = None
    for s in sensor_cols:
        if feature_name.endswith(f"_{s}") and (best is None or len(s) > len(best)):
            best = s
    return best


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(err**2)))


def permutation_sensor_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sensor_cols: list[str],
    n_repeats: int = 8,
    seed: int = 42,
) -> dict[str, float]:
    """Per-sensor permutation importance, measured as increase in RMSE.

    For each sensor channel we shuffle its rows across windows `n_repeats`
    times and record how much worse the (already fitted) model gets. Larger
    importance means the model degrades more when that sensor's information
    is destroyed. Evaluation happens on whichever (X, y) you pass — call it
    with validation windows, not training windows, or you measure
    memorisation instead of generalisation sensitivity.

    Returns {sensor_code: mean_importance}, sorted most important first.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 3 or X.shape[2] != len(sensor_cols):
        raise ValueError(
            f"expected windows of shape (n, w, {len(sensor_cols)}), got {X.shape}"
        )
    rng = np.random.default_rng(seed)
    baseline = _rmse(y, model.predict(X))

    out: dict[str, float] = {}
    for j, sensor in enumerate(sensor_cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
            deltas.append(_rmse(y, model.predict(Xp)) - baseline)
        out[sensor] = float(np.mean(deltas))
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

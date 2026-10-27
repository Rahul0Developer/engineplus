"""RUL metrics: RMSE, MAE and the official NASA asymmetric scoring function.

The NASA score (Saxena et al., 2008) penalises late predictions — saying an
engine has more life than it does — far harder than early ones, which matches
the operational risk intuition: you can always fly an engine again after a
conservative call, not after a optimistic one.

    d_i = predicted_rul_i - true_rul_i
    s_i = exp(-d_i / 13) - 1   if d_i < 0   (early prediction)
        = exp( d_i / 10) - 1   if d_i >= 0  (late prediction)
    score = sum_i s_i        (lower is better; 0 would be perfect)

Per-engine aggregation convention: exactly one prediction per test engine
(the last window), which is what ``LastWindowSampler`` guarantees upstream.
"""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    return float(np.mean(np.abs(pred - true)))


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Summed asymmetric exponential score (lower is better)."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    d = pred - true
    terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    # exp overflow on wildly wrong predictions should surface as inf, not crash.
    with np.errstate(over="ignore"):
        terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(terms))


def all_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Compute the standard trio in one call."""
    return {"rmse": rmse(pred, true), "mae": mae(pred, true), "nasa_score": nasa_score(pred, true)}


def error_by_life_stage(
    pred: np.ndarray, true: np.ndarray, threshold: float = 125.0
) -> dict[str, dict[str, float]]:
    """Split errors into 'far from failure' (true RUL above the cap knee) and
    'near failure'. Useful because models that look fine overall are usually
    worst right before the event that matters."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    out = {}
    for label, mask in {
        "far_from_failure": true > threshold,
        "near_failure": true <= threshold,
    }.items():
        if mask.sum() == 0:
            out[label] = {"n": 0, "rmse": float("nan"), "mae": float("nan")}
        else:
            out[label] = {
                "n": int(mask.sum()),
                "rmse": rmse(pred[mask], true[mask]),
                "mae": mae(pred[mask], true[mask]),
            }
    return out


def late_prediction_fraction(pred: np.ndarray, true: np.ndarray, tol: float = 0.0) -> float:
    """Share of engines where the model said 'more life than reality'."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean(pred - true > tol))

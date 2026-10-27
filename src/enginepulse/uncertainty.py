"""Prediction intervals for RUL estimates.

Two supported strategies (chosen by model capability, not by config magic):

- ``quantile``  : sklearn QuantileRegressor trained on window features. Cheap,
  well-calibrated-ish, only works on the classical feature path.
- ``mc_dropout``: Monte-Carlo dropout via the GRU model's predict_quantiles.

Everything here returns an IntervalSet so callers (API, trainer) share one shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enginepulse.models.features import window_features


@dataclass
class IntervalSet:
    """Lower/upper bounds per sample plus a nominal coverage level."""

    lower: np.ndarray
    upper: np.ndarray
    point: np.ndarray
    level: float  # e.g. 0.90 for a 90% interval

    def widths(self) -> np.ndarray:
        return self.upper - self.lower

    def empirical_coverage(self, truth: np.ndarray) -> float:
        """Fraction of truths inside the interval — the honest calibration check."""
        truth = np.asarray(truth, dtype=float)
        return float(np.mean((truth >= self.lower) & (truth <= self.upper)))


def residual_quantile_intervals(
    y_val_true: "np.ndarray",
    y_val_pred: "np.ndarray",
    point_preds: "np.ndarray",
    level: float = 0.90,
    max_rul: float | None = 125.0,
) -> IntervalSet:
    """Conformal-style intervals from validation residuals.

    We take the (1-level)/2 and (1+level)/2 quantiles of val errors and shift
    them onto each test prediction. Assumes homoscedastic errors — a known
    simplification; MC-dropout (GRU) is the model-aware alternative.
    """
    resid = np.asarray(y_val_pred, dtype=float) - np.asarray(y_val_true, dtype=float)
    if len(resid) < 20:
        raise ValueError("need >=20 validation windows for residual quantiles")
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lower = point_preds + np.quantile(resid, lo_q)
    upper = point_preds + np.quantile(resid, hi_q)
    if max_rul is not None:
        lower, upper = np.clip(lower, 0.0, max_rul), np.clip(upper, 0.0, max_rul)
    lower = np.minimum(lower, upper)
    return IntervalSet(lower=lower, upper=upper, point=np.asarray(point_preds), level=level)


def mc_dropout_intervals(model, X_new: np.ndarray, level: float = 0.90) -> IntervalSet:
    """Intervals from a model that implements predict_quantiles (GRU w/ MC dropout)."""
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    q = model.predict_quantiles(X_new, [lo_q, hi_q])
    lower, upper = q[lo_q], q[hi_q]
    if np.isnan(lower).any():
        raise ValueError(
            "model does not provide quantiles; enable mc_dropout on the GRU "
            "or use quantile_intervals() for the classical path"
        )
    point = model.predict(X_new)
    return IntervalSet(lower=lower, upper=upper, point=point, level=level)

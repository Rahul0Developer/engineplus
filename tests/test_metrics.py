"""Metric correctness, especially the NASA asymmetric score."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from enginepulse.metrics import (
    all_metrics,
    error_by_life_stage,
    late_prediction_fraction,
    mae,
    nasa_score,
    rmse,
)


def test_perfect_predictions_score_zero():
    t = np.array([0.0, 12.5, 99.0])
    assert nasa_score(t, t) == pytest.approx(0.0)
    assert rmse(t, t) == 0.0
    assert mae(t, t) == 0.0


def test_late_error_penalised_harder_than_early():
    true = np.array([50.0])
    early = nasa_score(np.array([40.0]), true)   # d = -10 -> exp(10/13)-1
    late = nasa_score(np.array([60.0]), true)    # d = +10 -> exp(1)-1
    assert late > early
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late == pytest.approx(math.e - 1)


def test_formula_matches_reference_implementation():
    rng = np.random.default_rng(3)
    pred = rng.uniform(0, 125, 200)
    true = rng.uniform(0, 125, 200)
    ref = 0.0
    for p, t in zip(pred, true):
        d = p - t
        ref += math.exp(-d / 13) - 1 if d < 0 else math.exp(d / 10) - 1
    assert nasa_score(pred, true) == pytest.approx(ref, rel=1e-9)


def test_empty_arrays_score_zero_and_raise_on_mismatch():
    assert nasa_score(np.array([]), np.array([])) == 0.0
    with pytest.raises(ValueError, match="shape mismatch"):
        nasa_score(np.array([1.0, 2.0]), np.array([1.0]))


def test_extreme_error_overflows_to_inf_not_crash():
    s = nasa_score(np.array([1e6]), np.array([0.0]))
    assert s == float("inf")


def test_all_metrics_keys_and_values():
    m = all_metrics(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert set(m) == {"rmse", "mae", "nasa_score"}
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(2.0)


def test_error_by_life_stage_splits_at_threshold():
    pred = np.array([100.0, 5.0])
    true = np.array([200.0, 10.0])
    out = error_by_life_stage(pred, true, threshold=125.0)
    assert out["far_from_failure"]["n"] == 1
    assert out["near_failure"]["n"] == 1
    assert out["far_from_failure"]["rmse"] == pytest.approx(100.0)
    assert out["near_failure"]["mae"] == pytest.approx(5.0)


def test_error_by_life_stage_empty_bucket_is_nan():
    out = error_by_life_stage(np.array([1.0]), np.array([1.0]), threshold=125.0)
    assert out["far_from_failure"]["n"] == 0
    assert math.isnan(out["far_from_failure"]["rmse"])


def test_late_prediction_fraction():
    pred = np.array([10.0, 10.0, 10.0, 5.0])
    true = np.array([5.0, 5.0, 10.0, 10.0])
    assert late_prediction_fraction(pred, true) == pytest.approx(0.5)
    # exact ties are NOT counted as late (pred - true > 0)
    assert late_prediction_fraction(np.array([7.0]), np.array([7.0])) == 0.0


@given(
    d=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_nasa_score_is_asymmetric_and_nonneg_late(d):
    """Property: every late error costs at least as much as the mirrored early one."""
    s_late = nasa_score(np.array([100.0 + abs(d)]), np.array([100.0]))
    s_early = nasa_score(np.array([100.0 - abs(d)]), np.array([100.0]))
    assert s_late >= 0.0 and s_early >= 0.0
    assert s_late >= s_early  # asymmetry is the whole point of the metric

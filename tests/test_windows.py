"""Window construction: boundary safety, labels, samplers, empty cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enginepulse.data.windows import (
    FullWindowSampler,
    LastWindowSampler,
    TailWindowsSampler,
    build_windows,
)


def _toy_frame():
    """Two engines with different lengths; 'rul' = remaining rows after time t."""
    rows = []
    for unit, T in [(1, 10), (2, 6)]:
        for t in range(1, T + 1):
            rows.append({"unit": unit, "time": t, "x": float(t), "y": float(unit * 100 + t),
                         "rul": float(T - t)})
    return pd.DataFrame(rows)


FEATS = ["x", "y"]


def test_windows_never_cross_engine_boundary():
    ws = build_windows(_toy_frame(), FEATS, 4, FullWindowSampler())
    # A cross-boundary window would mix y values from two engines; check the
    # last-timestep feature equals the window's own unit marker.
    for i in range(len(ws)):
        last_y = ws.X[i, -1, 1]
        first_y = ws.X[i, 0, 1]
        unit = ws.units[i]
        assert int(last_y) // 100 == unit and int(first_y) // 100 == unit


def test_window_counts_are_exact():
    ws = build_windows(_toy_frame(), FEATS, 4, FullWindowSampler())
    # engine1: 10-4+1 = 7 windows; engine2: 6-4+1 = 3
    assert len(ws) == 10
    assert (ws.units == 1).sum() == 7 and (ws.units == 2).sum() == 3


def test_label_is_rul_of_last_timestep():
    w = 4
    ws = build_windows(_toy_frame(), FEATS, w, FullWindowSampler())
    for i in range(len(ws)):
        unit = int(ws.units[i])
        t_last = int(ws.X[i, -1, 1] % 100)  # y encoded as unit*100 + t
        expected_rul = (10 if unit == 1 else 6) - t_last
        assert ws.y[i] == pytest.approx(expected_rul)


def test_last_window_sampler_gives_one_per_engine():
    ws = build_windows(_toy_frame(), FEATS, 4, LastWindowSampler())
    assert len(ws) == 2
    assert sorted(ws.units.tolist()) == [1, 2]
    # final window of engine 1 ends at its last cycle -> label 0
    eng1 = ws.X[ws.units == 1][0]
    assert int(eng1[-1, 1]) == 110  # unit 1, time 10
    assert ws.y[ws.units == 1][0] == 0.0


def test_tail_sampler_limits_count():
    ws = build_windows(_toy_frame(), FEATS, 4, TailWindowsSampler(k=2))
    assert (ws.units == 1).sum() == 2
    assert (ws.units == 2).sum() == 2


def test_short_engine_produces_no_windows():
    df = pd.DataFrame([{"unit": 5, "time": 1, "x": 1.0, "y": 5.0, "rul": 0.0}])
    ws = build_windows(df, FEATS, 4, LastWindowSampler())
    assert len(ws) == 0
    assert ws.X.shape == (0, 4, 2)


def test_subset_preserves_alignment():
    ws = build_windows(_toy_frame(), FEATS, 4, FullWindowSampler())
    sub = ws.subset(ws.units == 2)
    assert len(sub) == 3
    assert (sub.units == 2).all()
    assert sub.X.shape[0] == len(sub.y) == len(sub.times)


def test_missing_rul_col_zerofills():
    df = _toy_frame().drop(columns=["rul"])
    ws = build_windows(df, FEATS, 4, LastWindowSampler(), rul_col=None)
    assert (ws.y == 0).all()


@pytest.mark.parametrize("w", [1, 2, 4, 9, 10])
def test_hypothesis_style_sweep_over_window_lengths(w):
    """Property: count == sum(max(T_i - w + 1, 0)) and every window is finite."""
    df = _toy_frame()
    ws = build_windows(df, FEATS, w, FullWindowSampler())
    assert len(ws) == max(10 - w + 1, 0) + max(6 - w + 1, 0)
    assert np.isfinite(ws.X).all()

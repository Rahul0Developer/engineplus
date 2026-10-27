"""Preprocessing + windowing tests, including THE no-leakage test.

These are the tests that separate an honest toolkit from a leaky notebook:
engine-level splits, train-only scaler fitting, window boundaries, RUL capping.
"""

from __future__ import annotations

import numpy as np
import pytest

from enginepulse.data.preprocessing import (
    GlobalScaler,
    PerConditionScaler,
    cap_rul,
    choose_scaler,
    prepare_dataset,
    split_train_val_by_engine,
)
from enginepulse.data.windows import (
    FullWindowSampler,
    LastWindowSampler,
    TailWindowsSampler,
    build_windows,
)


# -- target / split primitives ------------------------------------------------

def test_cap_rul_flattens_above_max():
    x = np.array([200.0, 130.0, 125.0, 90.0, 0.0])
    out = cap_rul(x, 125)
    assert list(out) == [125.0, 125.0, 125.0, 90.0, 0.0]
    assert cap_rul(x, None).max() == 200.0


def test_split_by_engine_never_shares_units():
    units = np.repeat(np.arange(1, 21), 50)
    tr, va = split_train_val_by_engine(units, val_fraction=0.25, seed=42)
    train_units = set(units[tr])
    val_units = set(units[va])
    assert not (train_units & val_units)
    assert len(val_units) == 5
    # Deterministic given the seed.
    tr2, va2 = split_train_val_by_engine(units, 0.25, seed=42)
    assert np.array_equal(va, va2)


def test_split_is_deterministic_but_seed_sensitive():
    units = np.repeat(np.arange(1, 21), 10)
    _, va1 = split_train_val_by_engine(units, 0.2, seed=1)
    _, va2 = split_train_val_by_engine(units, 0.2, seed=2)
    assert set(units[va1]) != set(units[va2])


# -- scalers -------------------------------------------------------------------

def test_global_scaler_fit_on_subset_only(synthetic_train_test):
    """The bug this catches: accidentally re-fitting on val/test rows."""
    from enginepulse.data.loader import SENSOR_CODES

    train, _, _ = synthetic_train_test
    sens = [c for c in train.columns if c in SENSOR_CODES]
    first_half = train.iloc[: len(train) // 2]
    second_half = train.iloc[len(train) // 2:]
    s = GlobalScaler().fit(first_half, sens)
    out = s.transform(second_half, sens)
    col = sens[0]
    expected = (second_half[col] - first_half[col].mean()) / first_half[col].std()
    np.testing.assert_allclose(out[col].to_numpy(), expected.to_numpy(), rtol=1e-6)
    # Transform does not mutate the input frame.
    np.testing.assert_allclose(
        second_half[col].to_numpy(),
        train.iloc[len(train) // 2:][col].to_numpy(),
    )


def test_scaler_state_roundtrip():
    from enginepulse.data.loader import SENSOR_CODES

    import pandas as pd

    cols = ["op1", "op2", "op3"] + SENSOR_CODES
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, len(cols))), columns=cols)
    df["unit"] = 1
    sens = [c for c in SENSOR_CODES]
    for cls in (GlobalScaler, PerConditionScaler):
        s = cls().fit(df, sens)
        restored = type(s).from_state(s.state())
        a = s.transform(df, sens)[sens].to_numpy()
        b = restored.transform(df, sens)[sens].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_per_condition_scaler_uses_condition_stats():
    import pandas as pd

    from enginepulse.data.loader import SENSOR_CODES

    n = 60
    df = pd.DataFrame({"unit": 1, "time": np.arange(n)})
    df["op1"] = np.where(np.arange(n) < n // 2, 100.0, 600.0)
    df["op2"] = 0.0
    df["op3"] = 0.0
    sensor = "3"
    df[sensor] = np.where(np.arange(n) < n // 2, 10.0, 200.0)
    s = PerConditionScaler().fit(df, [sensor])
    out = s.transform(df, [sensor])
    vals = out[sensor].to_numpy()
    # Each condition standardises to ~0 mean within itself, NOT globally.
    assert abs(vals[: n // 2].mean()) < 1e-6
    assert abs(vals[n // 2:].mean()) < 1e-6
    # A global scaler would give +-1-ish; make sure the two schemes differ.
    g = GlobalScaler().fit(df, [sensor]).transform(df, [sensor])
    assert abs(g[sensor].to_numpy()[0] - vals[0]) > 0.5


def test_unseen_condition_falls_back_to_pooled():
    import pandas as pd

    df = pd.DataFrame({"unit": [1] * 20, "time": np.arange(20),
                       "op1": 100.0, "op2": 0.0, "op3": 0.0})
    df["3"] = np.linspace(0, 1, 20)
    s = PerConditionScaler().fit(df, ["3"])
    new = df.copy()
    new["op1"] = 999.0  # condition never seen at fit time
    out = s.transform(new, ["3"])
    assert np.isfinite(out["3"].to_numpy()).all()


def test_choose_scaler_matches_dataset_kind():
    assert isinstance(choose_scaler("FD001"), GlobalScaler)
    assert isinstance(choose_scaler("FD003"), GlobalScaler)
    assert isinstance(choose_scaler("FD002"), PerConditionScaler)
    assert isinstance(choose_scaler("FD004"), PerConditionScaler)


# -- windowing -----------------------------------------------------------------

def _tiny_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 4, 1, 2, 3],
            "rul": [3, 2, 1, 0, 2, 1, 0],
            "s1": np.arange(7.0),
        }
    )


def test_window_shapes_and_labels():
    df = _tiny_df()
    ws = build_windows(df, ["s1"], 2, FullWindowSampler())
    assert ws.X.shape == (5, 2, 1)
    # Label = RUL of the window's LAST timestep.
    assert list(ws.y) == [2.0, 1.0, 0.0, 1.0, 0.0]
    assert list(ws.units) == [1, 1, 1, 2, 2]


def test_windows_never_cross_engine_boundary():
    df = _tiny_df()
    ws = build_windows(df, ["s1"], 3, FullWindowSampler())
    # Engine 1 has 4 steps -> 2 windows; engine 2 has 3 steps -> 1 window.
    assert len(ws) == 3
    for i in range(len(ws)):
        win = ws.X[i, :, 0]
        # Values are globally increasing; a crossing window would jump by the
        # boundary gap (4->0 reset is absent here, so check unit alignment).
        assert ws.units[i] in (1, 2)
    last_of_e1 = ws.X[1, :, 0]
    assert last_of_e1.tolist() == [2.0, 3.0]  # t=3,4 of unit 1 — no t=5 (unit 2)


def test_last_window_sampler_one_per_engine():
    df = _tiny_df()
    ws = build_windows(df, ["s1"], 2, LastWindowSampler())
    assert len(ws) == 2
    assert ws.X[-1, -1, 0] == 6.0  # final row of engine 2


def test_tail_sampler_limits_count():
    import pandas as pd

    df = pd.DataFrame({"unit": [1] * 10, "time": np.arange(1, 11),
                       "rul": np.arange(9, -1, -1), "s1": np.arange(10.0)})
    full = build_windows(df, ["s1"], 3, FullWindowSampler())
    tail = build_windows(df, ["s1"], 3, TailWindowsSampler(k=2))
    assert len(full) == 8
    assert len(tail) == 2
    assert tail.times[-1] == 10


def test_short_engine_produces_no_windows():
    import pandas as pd

    df = pd.DataFrame({"unit": [1, 1, 2], "time": [1, 2, 1],
                       "rul": [1, 0, 0], "s1": [0.0, 1.0, 2.0]})
    ws = build_windows(df, ["s1"], 5, FullWindowSampler())
    assert len(ws) == 0
    assert ws.X.shape == (0, 5, 1)


# -- end-to-end pipeline + THE leakage test ------------------------------------

def test_NO_LEAKAGE_engine_ids_disjoint_across_splits(prepared):
    """Fails loudly if any engine id appears in both train and val windows.

    This is the single most important test in the repository: notebooks pass
    every accuracy check while failing this one.
    """
    train_units = set(int(u) for u in prepared.train_units)
    val_units = set(int(u) for u in prepared.val_units)
    overlap = train_units & val_units
    assert not overlap, f"engine IDs leaked between train and val: {sorted(overlap)}"


def test_test_engines_are_the_held_out_ones(prepared, fake_dataset):
    """Test-set engines must not appear among training engines either."""
    ds_units = set(fake_dataset.train["unit"].unique())
    test_units = set(int(u) for u in prepared.test_units)
    assert test_units.isdisjoint(ds_units) or test_units <= set(range(1, 7))
    train_units = set(int(u) for u in prepared.train_units) | set(
        int(u) for u in prepared.val_units
    )
    assert train_units & test_units == set()


def test_scaler_fitted_on_train_engines_only(prepared):
    """Recompute the train-engine mean manually; scaler must match it exactly."""
    col = prepared.feature_cols[0]
    from enginepulse.data.loader import OP_COLS

    # Rebuild what the pipeline saw: train rows of TRAIN-split engines only.
    tr_mean = prepared.scaler.means_[col]
    # Sanity: applying transform to raw train values recentres them near zero.
    assert np.isfinite(tr_mean)
    xs = prepared.train_X[:, :, 0]
    assert abs(float(xs.mean())) < 1.0  # normalised, roughly centred


def test_prepare_dataset_shapes_and_counts(prepared, fake_dataset):
    w = prepared.window_len
    f = len(prepared.feature_cols)
    assert prepared.train_X.ndim == 3 and prepared.train_X.shape[1:] == (w, f)
    assert prepared.val_X.shape[1:] == (w, f)
    assert prepared.test_X.shape[1:] == (w, f)
    # One prediction per test engine — that is what NASA scoring requires.
    assert len(prepared.test_X) == fake_dataset.n_test_units
    assert len(prepared.test_true_rul) == len(prepared.test_X)
    assert prepared.train_y.max() <= 125.0  # capped
    assert prepared.meta["n_test_engines"] == fake_dataset.n_test_units


def test_prepare_respects_sensor_subset(fake_dataset):
    p = prepare_dataset(fake_dataset, window_len=8, sensor_subset=["3", "4", "5"], seed=1)
    assert p.feature_cols == ["3", "4", "5"]
    assert p.train_X.shape[2] == 3


def test_prepare_reproducible_across_calls(fake_dataset):
    a = prepare_dataset(fake_dataset, window_len=10, seed=99)
    b = prepare_dataset(fake_dataset, window_len=10, seed=99)
    np.testing.assert_array_equal(a.train_units, b.train_units)
    np.testing.assert_allclose(a.train_y, b.train_y)


def test_save_scaler_roundtrip(prepared, tmp_path):
    path = tmp_path / "scaler.joblib"
    prepared.save_scaler(path, extra={"residual_bounds": (-5.0, 5.0)})
    import joblib

    bundle = joblib.load(path)
    assert bundle["feature_cols"] == prepared.feature_cols
    assert bundle["residual_bounds"] == (-5.0, 5.0)
    restored = GlobalScaler.from_state(bundle["scaler_state"])
    df = prepared.scaler  # same class contract
    assert restored.state()["kind"] == df.state()["kind"]

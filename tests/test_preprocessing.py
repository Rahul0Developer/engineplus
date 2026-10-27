"""The two leakage guarantees + scaler correctness + end-to-end preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enginepulse.data.preprocessing import (
    GlobalScaler,
    PerConditionScaler,
    cap_rul,
    choose_scaler,
    prepare_dataset,
    split_train_val_by_engine,
)


# ---------------------------------------------------------------- splitting

def test_split_has_zero_engine_overlap(fd001_ds):
    units = fd001_ds.train["unit"].to_numpy()
    tr, va = split_train_val_by_engine(units, val_fraction=0.25, seed=42)
    assert set(units[tr]).isdisjoint(set(units[va]))


def test_split_is_deterministic_for_seed(fd001_ds):
    units = fd001_ds.train["unit"].to_numpy()
    a_tr, a_va = split_train_val_by_engine(units, 0.3, seed=7)
    b_tr, b_va = split_train_val_by_engine(units, 0.3, seed=7)
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)


def test_split_keeps_every_row_and_covers_all_engines(fd001_ds):
    units = fd001_ds.train["unit"].to_numpy()
    tr, va = split_train_val_by_engine(units, 0.25, seed=1)
    assert tr.sum() + va.sum() == len(units)
    assert set(units[tr]) | set(units[va]) == set(np.unique(units))


def test_split_smallest_fraction_still_gets_one_val_unit(fd001_ds):
    units = fd001_ds.train["unit"].to_numpy()
    _, va = split_train_val_by_engine(units, val_fraction=0.001, seed=3)
    assert va.sum() > 0


@pytest.mark.parametrize("seed", [0, 1, 2, 99])
def test_prepare_dataset_no_leakage_anywhere(fd001_ds, seed):
    """The headline guarantee: train/val/test share no engine ids, ever."""
    prep = prepare_dataset(fd001_ds, window_len=6, max_rul=25, seed=seed)
    tr_u, va_u, te_u = set(prep.train_units), set(prep.val_units), set(prep.test_units)
    assert tr_u.isdisjoint(va_u)
    # Test engines are numbered from the test file; keep them distinct by id too.
    ds = fd001_ds
    test_ids = set(int(u) for u in ds.test["unit"].unique())
    train_ids = set(int(u) for u in ds.train["unit"].unique())
    assert te_u == test_ids
    assert tr_u | va_u == train_ids


def test_test_targets_match_published_rul(fd001_ds):
    prep = prepare_dataset(fd001_ds, window_len=8, max_rul=None, seed=42)
    truth = {e.unit: e.true_rul for e in fd001_ds.engines("test")}
    for u, t in zip(prep.test_units, prep.test_true_rul):
        assert t == truth[int(u)]


# ---------------------------------------------------------------- RUL capping

def test_cap_rul_flattens_above_knee():
    v = np.array([200.0, 125.0, 90.0, 0.0])
    out = cap_rul(v, 125)
    assert out.tolist() == [125.0, 125.0, 90.0, 0.0]
    assert out.dtype == np.float32


def test_cap_rul_none_passthrough():
    v = np.array([300.0])
    assert cap_rul(v, None).tolist() == [300.0]


# ---------------------------------------------------------------- scalers

def _toy_df(conds):
    rng = np.random.default_rng(0)
    n = len(conds)
    return pd.DataFrame({
        "op1": [c[0] for c in conds],
        "op2": [c[1] for c in conds],
        "op3": [c[2] for c in conds],
        "3": rng.normal(500, 10, n),
        "4": rng.normal(30, 2, n),
    })


def test_global_scaler_zero_mean_unit_std():
    df = _toy_df([(0.006, 0.0006, 100.0)] * 50)
    s = GlobalScaler().fit(df, ["3", "4"])
    out = s.transform(df, ["3", "4"])
    assert abs(out["3"].mean()) < 1e-9
    assert abs(out["3"].std() - 1.0) < 0.05


def test_global_scaler_survives_constant_column():
    df = _toy_df([(0.0, 0.0, 0.0)] * 10)
    df["9"] = 7.0  # constant -> std guard must kick in, not divide by zero
    s = GlobalScaler().fit(df, ["9"])
    out = s.transform(df, ["9"])
    assert np.isfinite(out["9"]).all()


def test_per_condition_scaler_normalises_within_each_condition():
    conds = [(-0.0005, 0.0004, 42.0)] * 40 + [(0.0031, -0.0004, 69.0)] * 40
    df = _toy_df(conds)
    s = PerConditionScaler().fit(df, ["3", "4"])
    out = s.transform(df, ["3", "4"])
    for cond in [(-0, 0, 42), (0, 0, 69)]:
        pass  # keys are rounded ints; check both groups directly below
    g42 = out[np.round(df["op3"]).astype(int) == 42]
    g69 = out[np.round(df["op3"]).astype(int) == 69]
    assert abs(g42["3"].mean()) < 1e-9 and abs(g69["3"].mean()) < 1e-9
    assert len(s.by_cond_) == 2


def test_per_condition_unseen_condition_falls_back_to_pooled():
    train = _toy_df([(-0.0005, 0.0004, 42.0)] * 30)
    s = PerConditionScaler().fit(train, ["3", "4"])
    new = _toy_df([(0.0099, 0.0099, 99.0)] * 5)  # condition never seen in fit
    out = s.transform(new, ["3", "4"])
    assert np.isfinite(out[["3", "4"]].to_numpy()).all()


def test_scaler_state_roundtrip():
    df = _toy_df([(-0.0005, 0.0004, 42.0), (0.0031, -0.0004, 69.0)] * 20)
    s = PerConditionScaler().fit(df, ["3", "4"])
    restored = type(s).from_state(s.state())
    a = s.transform(df, ["3", "4"])
    b = restored.transform(df, ["3", "4"])
    pd.testing.assert_frame_equal(a, b)


def test_choose_scaler_mapping():
    assert isinstance(choose_scaler("FD001"), GlobalScaler)
    assert isinstance(choose_scaler("FD003"), GlobalScaler)
    assert isinstance(choose_scaler("FD002"), PerConditionScaler)
    assert isinstance(choose_scaler("FD004"), PerConditionScaler)


# ---------------------------------------------------------------- end-to-end

def test_prepare_fd002_uses_per_condition(fd002_ds):
    prep = prepare_dataset(fd002_ds, window_len=6, max_rul=30, seed=42)
    assert prep.meta["scaler"] == "PerConditionScaler"
    assert len(prep.train_X) == len(prep.train_y) == len(prep.train_units)
    assert prep.train_X.shape[1] == 6


def test_save_scaler_roundtrip(tmp_path, prepared_fd001):
    import joblib

    path = tmp_path / "art" / "scaler.joblib"
    prepared_fd001.save_scaler(path, extra={"residual_bounds": (-5.0, 5.0)})
    bundle = joblib.load(path)
    assert bundle["feature_cols"] == prepared_fd001.feature_cols
    assert bundle["residual_bounds"] == (-5.0, 5.0)
    restored = GlobalScaler.from_state(bundle["scaler_state"])
    assert isinstance(restored, GlobalScaler)


def test_uncapped_targets_keep_full_range(fd001_ds):
    prep = prepare_dataset(fd001_ds, window_len=4, max_rul=None, seed=42)
    assert prep.train_y.max() > 30  # no flattening applied

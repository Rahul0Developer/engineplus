"""Shared fixtures.

Everything here works WITHOUT the C-MAPSS download: synthetic data is shaped
exactly like the real thing (unit/time/op cols/21 sensor cols, space-separated)
so the pipeline under test is the same code path a real run takes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

OP_COLS = ["op1", "op2", "op3"]
SENSOR_CODES = [str(i) for i in range(1, 22)]


def make_engine_table(
    rng: np.random.Generator,
    units: list[int],
    n_steps_range: tuple[int, int],
    n_conditions: int = 1,
) -> pd.DataFrame:
    """Build a raw-format dataframe: unit, time, 3 op settings, 21 sensors.

    Sensors 0..5 are constant/near-constant (like the real dead positions),
    the rest carry a linear degradation trend + noise so regressors have
    something real to learn.
    """
    rows = []
    for u in units:
        n = int(rng.integers(*n_steps_range))
        cond = int(rng.integers(0, n_conditions))
        base = rng.normal(size=16) * 5.0
        drift = rng.normal(size=16) * 0.5 + 0.5  # per-engine degradation rate
        t = np.arange(1, n + 1)
        sens = np.zeros((n, 21))
        live = base[None, :] + np.outer(t / max(n, 1), drift) + rng.normal(scale=0.1, size=(n, 16))
        # Positions 2, 11, 17, 20, 21 (1-based) stay flat, matching the real data.
        sens[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 18]] = live[:, :15]
        sens[:, 1] = 500.0  # constant
        sens[:, 10] = 0.0   # constant
        sens[:, 19] = 0.0
        sens[:, 20] = 0.0
        # A few columns with only two distinct values -> near-constant.
        sens[:, 17] = (t % 2).astype(float)
        ops = np.tile([100.0 + cond, 500.0 + 100.0 * cond, 0.02 + 0.01 * cond], (n, 1))
        for i in range(n):
            rows.append([u, i + 1, *ops[i], *sens[i]])
    return pd.DataFrame(
        rows, columns=["unit", "time"] + OP_COLS + SENSOR_CODES, dtype=float
    )


@pytest.fixture()
def synthetic_train_test():
    """(train_df, test_df, rul_by_unit) for 20 train engines / 6 test engines."""
    rng = np.random.default_rng(7)
    train_units = list(range(1, 21))
    test_units = list(range(1, 7))
    train = make_engine_table(rng, train_units, (40, 90))
    test = make_engine_table(rng, test_units, (40, 90))
    true_rul = {u: int(rng.integers(5, 60)) for u in test_units}
    return train, test, true_rul


@pytest.fixture()
def fake_dataset(synthetic_train_test, tmp_path):
    """Write a Dataset object into the on-disk C-MAPSS text format and load it."""
    from enginepulse.data.loader import load_dataset

    train, test, true_rul = synthetic_train_test
    d = tmp_path / "data"
    d.mkdir()
    name = "FDX01"

    def dump(df: pd.DataFrame, path):
        # Loader reads exactly [unit, time, op1..3, 21 sensors]; drop any 'rul'.
        cols = ["unit", "time", "op1", "op2", "op3"] + SENSOR_CODES
        df[cols].to_csv(path, sep=" ", header=False, index=False)

    dump(train, d / f"train_{name}.txt")
    dump(test, d / f"test_{name}.txt")
    (d / f"RUL_{name}.txt").write_text("\n".join(str(true_rul[u]) for u in sorted(true_rul)))

    ds = load_dataset(d, name)
    ds.test_rul_by_unit = true_rul  # convenience for tests
    return ds


@pytest.fixture()
def prepared(fake_dataset):
    """PreparedData from the fake dataset with small windows."""
    from enginepulse.data.preprocessing import GlobalScaler, prepare_dataset

    return prepare_dataset(
        fake_dataset, window_len=10, max_rul=125, val_fraction=0.2, seed=42,
        scaler=GlobalScaler(),
    )

"""Loader tests: schema, RUL unravelling, dead-sensor dropping, error paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enginepulse.data.loader import (
    OP_COLS,
    SENSOR_CODES,
    add_train_rul,
    drop_constant_sensors,
    load_dataset,
)


def test_load_dataset_shape(fd001_ds):
    assert list(fd001_ds.train.columns[:5]) == ["unit", "time", "rul", "op1", "op2"]
    kept = [c for c in fd001_ds.train.columns if c in SENSOR_CODES]
    assert len(kept) == 16, "positions 2,11,17,20,21 should be dropped"
    assert fd001_ds.n_train_units == 12
    assert fd001_ds.n_test_units == 6


def test_constant_sensors_dropped_are_the_usual_five(fd001_ds):
    combined = pd.concat([fd001_ds.train, fd001_ds.test], ignore_index=True)
    present = set(combined.columns)
    # 1-based positions 2, 11, 17, 20, 21 map to codes '1','10','16','19','20'.
    for gone in ("1", "10", "16", "19", "20"):
        assert gone not in present


def test_train_rul_unravels_to_zero_at_failure(fd001_ds):
    for eng in fd001_ds.engines("train"):
        last = eng.df["rul"].iloc[-1]
        assert last == 0
        # strictly decreasing by 1 per cycle
        ruls = eng.df.sort_values("time")["rul"].to_numpy()
        assert np.array_equal(ruls, np.arange(len(ruls))[::-1])


def test_test_engines_carry_published_true_rul(fd001_ds):
    engs = fd001_ds.engines("test")
    assert all(e.true_rul is not None for e in engs)
    assert engs[0].true_rul == int(fd001_ds.test_rul[0])


def test_add_train_rul_on_raw_frame(fixtures_dir):
    raw = pd.read_csv(
        fixtures_dir / "train_FD001.txt", sep=r"\s+", header=None,
        names=["unit", "time"] + OP_COLS + SENSOR_CODES, dtype=float,
    )
    out = add_train_rul(raw)
    assert "rul" in out.columns
    assert out.loc[out["unit"] == 1, "rul"].iloc[-1] == 0


def test_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="enginepulse download"):
        load_dataset(tmp_path, "FD001")


def test_rul_length_mismatch_raises(fixtures_dir, tmp_path):
    import shutil

    for f in ("train_FD001.txt", "test_FD001.txt"):
        shutil.copy(fixtures_dir / f, tmp_path / f)
    (tmp_path / "RUL_FD001.txt").write_text("10\n20\n")  # far too few entries
    with pytest.raises(ValueError, match="file mismatch"):
        load_dataset(tmp_path, "FD001")


def test_drop_constant_sensors_records_attrs():
    df = pd.DataFrame({
        "unit": [1, 1, 1],
        "time": [1, 2, 3],
        "3": [1.0, 2.0, 3.0],   # informative
        "4": [0.0, 0.0, 0.0],   # dead
    })
    out = drop_constant_sensors(df)
    assert "3" in out.columns and "4" not in out.columns
    assert out.attrs["dropped_sensors"] == ["4"]


def test_dataset_without_rul_file(tmp_path, fixtures_dir):
    import shutil

    shutil.copy(fixtures_dir / "train_FD001.txt", tmp_path / "train_FD001.txt")
    shutil.copy(fixtures_dir / "test_FD001.txt", tmp_path / "test_FD001.txt")
    ds = load_dataset(tmp_path, "FD001")
    assert ds.test_rul is None
    assert all(e.true_rul is None for e in ds.engines("test"))

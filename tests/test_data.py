"""Loader + download-helper tests (synthetic data, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enginepulse.data import download as dl
from enginepulse.data.loader import (
    SENSOR_CODES,
    add_train_rul,
    drop_constant_sensors,
    load_dataset,
)


def test_add_train_rul_is_last_minus_current_time():
    df = pd.DataFrame(
        {"unit": [1, 1, 1, 2, 2], "time": [1, 2, 3, 1, 2]}, dtype=float
    )
    out = add_train_rul(df)
    # Engine 1 ends at t=3 -> RULs 2,1,0 ; engine 2 ends at t=2 -> 1,0
    assert list(out["rul"]) == [2.0, 1.0, 0.0, 1.0, 0.0]


def test_drop_constant_sensors_removes_flat_columns(synthetic_train_test):
    train, test, _ = synthetic_train_test
    # Same combined-statistics path the loader uses: a sensor can look dead
    # in one short engine yet be alive elsewhere, so we judge on both splits.
    combined = pd.concat([train, test], ignore_index=True)
    out = drop_constant_sensors(combined)
    kept = [c for c in out.columns if c in SENSOR_CODES]
    # Positions 2, 11, 17, 20, 21 (1-based) are flat in the fixture.
    assert "1" not in kept and "10" not in kept and "16" not in kept
    assert "19" not in kept and "20" not in kept
    assert len(kept) == 16
    assert out.attrs["dropped_sensors"] == ["1", "10", "16", "19", "20"]


def test_load_dataset_shapes_and_ground_truth(fake_dataset):
    ds = fake_dataset
    assert ds.name == "FDX01"
    assert ds.n_train_units == 20
    assert ds.n_test_units == 6
    assert ds.test_rul is not None and len(ds.test_rul) == 6
    assert "rul" in ds.train.columns
    # Engines iterate in unit order with true RUL attached to test engines.
    eng = ds.engines("test")
    assert [e.unit for e in eng] == list(range(1, 7))
    assert eng[0].true_rul == int(ds.test_rul[0])


def test_load_dataset_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="enginepulse download"):
        load_dataset(tmp_path, "FD001")


def test_rul_file_length_mismatch_raises(synthetic_train_test, tmp_path):
    train, test, _ = synthetic_train_test
    d = tmp_path
    cols = ["unit", "time", "op1", "op2", "op3"] + SENSOR_CODES
    train[cols].to_csv(d / "train_FD999.txt", sep=" ", header=False, index=False)
    test[cols].to_csv(d / "test_FD999.txt", sep=" ", header=False, index=False)
    (d / "RUL_FD999.txt").write_text("5\n6\n")  # 2 entries vs 6 engines
    with pytest.raises(ValueError, match="file mismatch"):
        load_dataset(d, "FD999")


def test_expected_files_and_prepare_dir_errors(tmp_path):
    assert dl._expected_files("FD001") == [
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"
    ]
    with pytest.raises(FileNotFoundError, match="enginepulse download"):
        dl.prepare_data_dir(tmp_path)


def test_extract_zip_flattens_inner_dirs(tmp_path):
    import zipfile
    from pathlib import Path

    src = tmp_path / "inner" / "subdir"
    src.mkdir(parents=True)
    (src / "train_FDXXX.txt").write_text("1 1 0 0 0 " + "0 " * 21 + "\n")
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(src / "train_FDXXX.txt", "inner/subdir/train_FDXXX.txt")
    dest = tmp_path / "out"
    dest.mkdir()
    dl._extract_zip(zpath, dest)
    assert (dest / "train_FDXXX.txt").exists()


def test_verify_rejects_truncated_file(synthetic_train_test, tmp_path):
    train, test, true_rul = synthetic_train_test
    cols = ["unit", "time", "op1", "op2", "op3"] + SENSOR_CODES
    for name, df in (("train", train), ("test", test)):
        df.to_csv(tmp_path / f"{name}_FD001.txt", sep=" ", header=False, index=False)
    (tmp_path / "RUL_FD001.txt").write_text("\n".join(str(x) for x in true_rul.values()))
    with pytest.raises(dl.DownloadError, match="expected"):
        dl._verify(tmp_path, ("FD001",))  # row counts won't match the official ones


def test_fallback_embed_is_deterministic_and_normalised():
    from enginepulse.docqa.ingest import fallback_embed

    a = fallback_embed(["borescope inspection every 400 hours"])
    b = fallback_embed(["borescope inspection every 400 hours"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5

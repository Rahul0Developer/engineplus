"""Trainer + benchmark tests.

These run the real train/evaluate/save path on the small FD001 fixture so a
broken artifact layout or metric plumbing fails loudly here, not in CI at
3 a.m. the day after a refactor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from enginepulse.train.benchmark import make_table, run_benchmark
from enginepulse.train.trainer import TrainConfig, infer_latency_ms, train_and_evaluate


def _dump_raw_format(ds, d: Path) -> None:
    """Write a loaded Dataset back out in raw NASA txt layout."""
    from enginepulse.data.loader import OP_COLS, SENSOR_CODES

    d.mkdir(parents=True, exist_ok=True)
    # load_dataset() has already dropped the dead sensor columns, so dump the
    # surviving ones — the loader will re-validate them against its own rule.
    kept = [c for c in ds.train.columns if c in SENSOR_CODES]
    cols = ["unit", "time"] + OP_COLS + kept
    ds.train[cols].to_csv(d / f"train_{ds.name}.txt", sep=" ", header=False, index=False)
    ds.test[cols].to_csv(d / f"test_{ds.name}.txt", sep=" ", header=False, index=False)
    (d / f"RUL_{ds.name}.txt").write_text(
        "\n".join(str(ds.test_rul_by_unit[u]) for u in sorted(ds.test_rul_by_unit))
    )


@pytest.fixture()
def fdx01_dir(fake_dataset, tmp_path):
    """A data dir holding the synthetic dataset in raw C-MAPSS text format.

    train_and_evaluate loads from disk itself, so trainer tests need real
    files — not just the in-memory Dataset the fake_dataset fixture returns.
    """
    d = tmp_path / "cmatss_data"
    _dump_raw_format(fake_dataset, d)
    return d


def test_train_and_evaluate_writes_stamped_result(fdx01_dir, tmp_path):
    """End-to-end on the synthetic dataset written to disk by the fixture."""
    out = tmp_path / "results"
    cfg = TrainConfig(
        dataset="FDX01", model="baseline", seed=3, window_len=6, max_rul=50,
        data_dir=str(fdx01_dir), output_dir=str(out), save_artifacts=True,
    )
    res = train_and_evaluate(cfg)

    assert set(res["metrics"]) >= {"rmse", "mae", "nasa_score"}
    assert isinstance(res["git"], str)  # stamped commit hash (or "unknown")
    assert "numpy" in res["versions"]
    saved = out / "FDX01_baseline_seed3.json"
    assert saved.exists()
    doc = json.loads(saved.read_text())
    assert doc["seed"] == 3

    # Artifacts: scaler bundle + model dir + meta with serving context.
    art = out / "artifacts" / "FDX01_baseline_seed3"
    assert (art / "model.joblib").exists()
    meta = json.loads((art / "meta.json").read_text())
    assert meta["window_len"] == 6
    assert len(meta["feature_cols"]) == 16  # 21 sensors minus the five dead ones


def test_train_creates_interval_bounds_for_serving(fdx01_dir, tmp_path):
    out = tmp_path / "res"
    cfg = TrainConfig(dataset="FDX01", model="ridge", seed=1, window_len=6,
                      max_rul=50, data_dir=str(fdx01_dir), output_dir=str(out))
    train_and_evaluate(cfg)
    import joblib

    bundle = joblib.load(out / "artifacts" / "FDX01_ridge_seed1" / "scaler.joblib")
    assert "residual_bounds" in bundle
    lo, hi = bundle["residual_bounds"]
    assert lo <= hi


def test_infer_latency_returns_percentiles(prepared):
    from enginepulse.models.classical import ConstantBaseline

    m = ConstantBaseline().fit(prepared.train_X, prepared.train_y)
    lat = infer_latency_ms(m, prepared.test_X, n_reps=3)
    assert 0 < lat["p50_ms"] <= lat["p95_ms"] * 10  # sane ordering


def test_run_benchmark_skips_missing_extras_gracefully(fdx01_dir, tmp_path):
    results = run_benchmark(
        models=["baseline", "not-a-model"], datasets=["FDX01"], seeds=[0],
        base={"window_len": 6, "max_rul": 50},
        data_dir=str(fdx01_dir), output_dir=str(tmp_path / "res"),
    )
    # bad name must not crash the run; baseline must be there
    assert [r["model"] for r in results] == ["baseline"]


def test_make_table_renders_markdown(tmp_path):
    rows = [{
        "dataset": "FD001", "model": "ridge",
        "metrics": {"rmse": 12.0, "mae": 9.0, "nasa_score": 40.0},
        "latency": {"p50_ms": 0.2, "p95_ms": 0.5},
    }]
    md = make_table(rows)
    assert "| FD001 | ridge |" in md
    assert "RMSE" in md

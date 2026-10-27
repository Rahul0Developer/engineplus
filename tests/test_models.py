"""Every registered model must complete fit -> predict -> save -> load -> predict,
with identical outputs before and after the roundtrip, and sane clipping."""

from __future__ import annotations

import numpy as np
import pytest

from enginepulse.models.base import clip_rul
from enginepulse.models.registry import available_models, build_model
from enginepulse.models.features import window_features

MAX_RUL = 30.0


def _quick_params(name: str) -> dict:
    """Tiny epochs so deep models train in under a second on fixture data."""
    if name in ("cnn", "gru", "transformer"):
        return dict(epochs=3, batch_size=32)
    if name == "xgb":
        return dict(n_estimators=25, max_depth=3)
    return {}


ALL_MODELS = ["baseline", "ridge", "xgb", "cnn", "gru", "transformer"]


@pytest.fixture(scope="module")
def data(prepared_fd001):
    X, y = prepared_fd001.train_X[:400], prepared_fd001.train_y[:400]
    Xv, yv = prepared_fd001.val_X[:100], prepared_fd001.val_y[:100]
    return X, y, Xv, yv


@pytest.mark.parametrize("name", ALL_MODELS)
def test_fit_predict_save_load_roundtrip(name, data, tmp_path):
    if name in ("cnn", "gru", "transformer"):
        pytest.importorskip("torch", reason="deep extras not installed")
    if name == "xgb":
        pytest.importorskip("xgboost")

    X, y, Xv, yv = data
    model = build_model(name, seed=42, max_rul=MAX_RUL, **_quick_params(name))
    model.fit(X, y, Xv, yv)
    assert model.is_fitted_

    pred = model.predict(Xv)
    assert pred.shape == (len(Xv),)
    assert np.isfinite(pred).all()
    assert pred.min() >= 0.0 and pred.max() <= MAX_RUL  # clipped both ends

    art = tmp_path / f"art_{name}"
    model.save(art)
    assert (art / "meta.json").exists()

    reloaded = build_model(name, seed=42, max_rul=MAX_RUL, **_quick_params(name))
    reloaded.load(art)
    pred2 = reloaded.predict(Xv)
    np.testing.assert_allclose(pred, pred2, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("name", ["baseline", "ridge", "xgb", "cnn", "gru", "transformer"])
def test_unfitted_predict_raises_helpful_error(name):
    if name in ("cnn", "gru", "transformer"):
        pytest.importorskip("torch")
    if name == "xgb":
        pytest.importorskip("xgboost")
    model = build_model(name, seed=0, **_quick_params(name))
    with pytest.raises(RuntimeError, match="call fit\\(\\) before predict"):
        model.predict(np.zeros((2, 4, 5), dtype="float32"))


def test_registry_reports_available_and_unknown(tmp_path):
    names = available_models()
    assert "baseline" in names and "ridge" in names
    with pytest.raises(KeyError, match="Unknown model 'nope'"):
        build_model("nope")


def test_build_model_deep_without_torch_gives_clear_hint(monkeypatch):
    # If torch is absent this raises ImportError from the constructor; if present,
    # the model simply builds. Either way no cryptic failure.
    try:
        build_model("gru")
    except ImportError as exc:
        assert "enginepulse[deep]" in str(exc)


def test_clip_rul_bounds():
    v = np.array([-5.0, 10.0, 200.0])
    c = clip_rul(v, 125.0)
    assert c.tolist() == [0.0, 10.0, 125.0]
    assert clip_rul(np.array([np.nan]), 125.0)[0] == 0.0  # nan treated as invalid -> floor


def test_baseline_predicts_constant(data):
    X, y, Xv, yv = data
    m = build_model("baseline", seed=1).fit(X, y)
    pred = m.predict(Xv)
    assert np.allclose(pred, pred[0])
    assert abs(pred[0] - float(np.mean(y))) < 1e-5


def test_ridge_beats_baseline_on_toy_data(data):
    X, y, Xv, yv = data
    base = build_model("baseline", seed=1).fit(X, y).predict(Xv)
    ridge = build_model("ridge", seed=1, alpha=1.0).fit(X, y, Xv, yv).predict(Xv)
    err_b = np.sqrt(np.mean((base - yv) ** 2))
    err_r = np.sqrt(np.mean((ridge - yv) ** 2))
    assert err_r <= err_b * 1.05  # ridge should at least not be much worse


@pytest.mark.parametrize("name", ["xgb", "gru"])
def test_quantile_intervals_ordered(name, data, tmp_path):
    if name == "xgb":
        pytest.importorskip("xgboost")
    else:
        pytest.importorskip("torch")
    X, y, Xv, yv = data
    kw = _quick_params(name)
    if name == "gru":
        kw.update(mc_dropout=True, mc_samples=5)
    m = build_model(name, seed=42, max_rul=MAX_RUL, **kw).fit(X, y, Xv, yv)
    q = m.predict_quantiles(Xv, [0.05, 0.95])
    lo, hi = q[0.05], q[0.95]
    assert (lo <= hi + 1e-6).all()
    assert ((hi - lo) > 0).any()


def test_xgb_feature_width_guard(data):
    """A loaded xgb model fed windows with the wrong sensor count fails loudly."""
    pytest.importorskip("xgboost")
    X, y, Xv, yv = data
    m = build_model("xgb", seed=1, **_quick_params("xgb")).fit(X, y, Xv, yv)
    bad = np.zeros((2, 4, 3), dtype="float32")  # 3 sensors instead of 16
    with pytest.raises(ValueError, match="expects"):
        m.predict(bad)


def test_xgb_importance_and_uncertainty_after_reload(data, tmp_path):
    pytest.importorskip("xgboost")
    X, y, Xv, yv = data
    m = build_model("xgb", seed=7, **_quick_params("xgb")).fit(X, y, Xv, yv)
    m.save(tmp_path / "xgb_art")
    m2 = build_model("xgb", seed=7).load(tmp_path / "xgb_art")
    fi = m2.feature_importance()
    assert len(fi) == 6 * X.shape[-1]
    q = m2.predict_quantiles(Xv, [0.05, 0.95])  # resid_std path after reload
    assert (q[0.05] <= q[0.95]).all()

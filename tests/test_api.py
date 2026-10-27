"""End-to-end API tests with FastAPI's TestClient.

A real ridge model is trained on the hermetic FD001 fixture and saved into a
temp artifact directory, so the registry/normalisation/prediction path under
test is byte-for-byte the production path — no mocks around the parts that
matter.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from enginepulse.api.app import create_app

WINDOW_LEN = 8


@pytest.fixture(scope="module")
def models_dir(prepared_fd001, tmp_path_factory):
    """Train + persist one ridge artifact exactly like `enginepulse train` does."""
    from enginepulse.models.registry import build_model

    art = tmp_path_factory.mktemp("models") / f"FD001_ridge_seed{42}"
    model = build_model("ridge", seed=42, max_rul=30)
    model.fit(prepared_fd001.train_X, prepared_fd001.train_y,
              prepared_fd001.val_X, prepared_fd001.val_y)
    model.save(art)
    prepared_fd001.save_scaler(art / "scaler.joblib",
                               extra={"residual_bounds": (-4.0, 4.0)})
    meta = art / "meta.json"
    import json

    m = json.loads(meta.read_text())
    m.update({"window_len": WINDOW_LEN, "max_rul": 30,
              "feature_cols": prepared_fd001.feature_cols, "dataset": "FD001"})
    meta.write_text(json.dumps(m))
    return art.parent


@pytest.fixture(scope="module")
def client(models_dir):
    app = create_app(models_dir=models_dir)
    with TestClient(app) as c:  # context manager fires startup -> load_all()
        yield c


@pytest.fixture(scope="module")
def raw_window(fd001_ds):
    """One genuine raw (window_len x n_sensors) slice from the fixture data."""
    df = fd001_ds.engines("train")[0].df
    from enginepulse.data.loader import SENSOR_CODES
    cols = [c for c in df.columns if c in SENSOR_CODES]
    arr = df[cols].to_numpy(dtype=np.float32)
    return arr[:WINDOW_LEN]


def test_health_reports_version_and_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_models_loaded"] == 1
    assert isinstance(body["version"], str)


def test_models_lists_loaded_key(client):
    r = client.get("/models")
    assert r.status_code == 200
    info = r.json()[0]
    assert info["key"].startswith("FD001_ridge_seed")
    assert info["window_len"] == WINDOW_LEN
    assert info["n_features"] > 0


def test_predict_returns_clipped_rul(client, raw_window):
    X = raw_window[None, :, :]
    r = client.post("/predict", json={"windows": X.tolist()})
    assert r.status_code == 200, r.text
    p = r.json()["predictions"][0]["rul_cycles"]
    assert 0.0 <= p <= 30.0


def test_predict_matches_direct_model_call(client, models_dir, raw_window):
    """Serving path must reproduce offline predictions (scaler roundtrip)."""
    from enginepulse.data.preprocessing import FeatureScaler
    from enginepulse.models.registry import build_model
    import joblib

    bundle = joblib.load(next(models_dir.glob("*/scaler.joblib")))
    scaler = FeatureScaler.from_state(bundle["scaler_state"])
    feature_cols = bundle["feature_cols"]
    model = build_model("ridge", seed=42, max_rul=30)
    model.load(next(models_dir.glob("*")))

    import pandas as pd

    X = raw_window[None]
    df = pd.DataFrame(X.reshape(-1, len(feature_cols)), columns=feature_cols)
    Xn = scaler.transform(df, feature_cols).to_numpy(np.float32).reshape(X.shape)
    direct = float(model.predict(Xn)[0])

    served = client.post("/predict", json={"windows": X.tolist()}).json()
    assert served["predictions"][0]["rul_cycles"] == pytest.approx(direct, abs=1e-4)


def test_predict_with_interval_uses_residual_bounds(client, raw_window):
    X = raw_window[None]
    r = client.post("/predict", json={"windows": X.tolist(), "with_interval": True})
    pred = r.json()["predictions"][0]
    assert pred["lower_90"] is not None and pred["upper_90"] is not None
    assert pred["lower_90"] <= pred["rul_cycles"] <= pred["upper_90"] + 1e-6


def test_predict_wrong_shape_is_422_with_hint(client, raw_window):
    bad = np.zeros((1, 3, raw_window.shape[1]), dtype=np.float32)  # wrong window len
    r = client.post("/predict", json={"windows": bad.tolist()})
    assert r.status_code == 422
    assert "does not match model context" in r.json()["detail"]


def test_predict_unknown_model_is_404(client, raw_window):
    r = client.post("/predict", json={"windows": raw_window[None].tolist()},
                    params={"model": "nope"})
    assert r.status_code == 404
    assert "Unknown model" in r.json()["detail"]


def test_ragged_batch_rejected_by_schema(client, raw_window):
    w = raw_window.tolist()
    w[0] = w[0][:-1]  # make it ragged
    r = client.post("/predict", json={"windows": [w]})
    assert r.status_code == 422


def test_ask_without_docqa_is_503(client):
    r = client.post("/ask", json={"question": "how do I calibrate sensor 2?"})
    assert r.status_code == 503
    assert "DocQA not configured" in r.json()["detail"]


def test_auth_enforced_when_key_configured(models_dir):
    app = create_app(models_dir=models_dir, api_key="secret123")
    with TestClient(app) as c:
        assert c.get("/models").status_code == 401
        assert c.get("/models", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/models", headers={"X-API-Key": "secret123"}).status_code == 200
        # health stays open for liveness probes
        assert c.get("/health").status_code == 200


def test_empty_registry_gives_helpful_error(tmp_path):
    app = create_app(models_dir=tmp_path / "missing")
    with TestClient(app) as c:
        assert c.get("/health").json()["n_models_loaded"] == 0
        r = c.post("/predict", json={"windows": [[[1.0, 2.0]]]})
        assert r.status_code == 404
        assert "No models loaded" in r.json()["detail"]


def test_broken_artifact_does_not_kill_service(tmp_path, prepared_fd001):
    """One corrupt artifact dir must be skipped; valid ones still load."""
    good = tmp_path / "FD001_ridge_seed1"
    _write_good_artifact(good, prepared_fd001)
    broken = tmp_path / "FD001_gru_seed2"
    broken.mkdir()
    (broken / "meta.json").write_text("{not json")

    app = create_app(models_dir=tmp_path)
    with TestClient(app) as c:
        assert c.get("/health").json()["n_models_loaded"] == 1


def _write_good_artifact(art, prep):
    import json

    from enginepulse.models.registry import build_model

    m = build_model("ridge", seed=1, max_rul=30)
    m.fit(prep.train_X, prep.train_y, prep.val_X, prep.val_y)
    m.save(art)
    prep.save_scaler(art / "scaler.joblib")
    meta = json.loads((art / "meta.json").read_text())
    meta.update({"window_len": WINDOW_LEN, "max_rul": 30,
                 "feature_cols": prep.feature_cols, "dataset": "FD001"})
    (art / "meta.json").write_text(json.dumps(meta))

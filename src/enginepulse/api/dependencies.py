"""Dependency-injection container for the API.

The app never imports global state; a ModelRegistry instance is constructed at
startup (or injected by tests) and handed to routes via FastAPI dependencies.
That is what keeps serving code testable and lets the same app run with zero,
one, or fifty models loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from enginepulse.data.preprocessing import FeatureScaler
from enginepulse.models.registry import build_model
from enginepulse.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """One model + the preprocessing context needed to feed it raw windows."""

    key: str
    dataset: str
    seed: int
    window_len: int
    n_features: int
    feature_cols: list[str]
    scaler: FeatureScaler
    model: object
    max_rul: float | None
    residual_bounds: tuple[float, float] | None = None  # (lo_delta, hi_delta) from val

    def normalise(self, X: np.ndarray) -> np.ndarray:
        """Apply the frozen train-time scaler to a raw-sensor window batch.

        The scaler works on DataFrames keyed by sensor column; we rebuild one
        per call (cheap at serving batch sizes, explicit over clever).
        """
        import pandas as pd

        df = pd.DataFrame(X.reshape(-1, len(self.feature_cols)), columns=self.feature_cols)
        out = self.scaler.transform(df, self.feature_cols)
        return out.to_numpy(dtype=np.float32).reshape(X.shape)


class ModelRegistry:
    """Loads artifacts produced by `enginepulse train` from a directory once."""

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self._models: dict[str, LoadedModel] = {}
        self.default_key: str | None = None

    def load_all(self) -> None:
        if not self.models_dir.exists():
            logger.warning("Models dir %s does not exist; starting with no models.", self.models_dir)
            return
        for art in sorted(p for p in self.models_dir.iterdir() if p.is_dir()):
            try:
                lm = self._load_one(art)
                self._models[lm.key] = lm
                if self.default_key is None:
                    self.default_key = lm.key
            except Exception as exc:
                # A broken artifact must not take the whole service down.
                logger.error("Failed to load %s: %s", art, exc)
        logger.info("Loaded %d model(s) from %s", len(self._models), self.models_dir)

    def _load_one(self, art: Path) -> LoadedModel:
        meta = json.loads((art / "meta.json").read_text())
        stem = art.name  # e.g. FD001_gru_seed42
        parts = stem.rsplit("_seed", 1)
        ds_model = parts[0].split("_", 1)
        dataset, model_name = ds_model[0], ds_model[1]
        seed = int(parts[1]) if len(parts) > 1 else int(meta.get("seed", 0))

        scaler_bundle = joblib.load(art / "scaler.joblib")
        scaler = FeatureScaler.from_state(scaler_bundle["scaler_state"])
        feature_cols = scaler_bundle["feature_cols"]

        # max_rul is a constructor arg for every model that uses it; passing it
        # here keeps the served clipping identical to training-time clipping.
        model = build_model(model_name, seed=seed, max_rul=meta.get("max_rul"))
        model.load(art)

        bounds = scaler_bundle.get("residual_bounds")
        return LoadedModel(
            residual_bounds=tuple(bounds) if bounds else None,
            key=stem,
            dataset=dataset,
            seed=seed,
            window_len=int(meta.get("window_len", 0)) or _infer_window(art),
            n_features=len(feature_cols),
            feature_cols=feature_cols,
            scaler=scaler,
            model=model,
            max_rul=meta.get("max_rul"),
        )

    def get(self, key: str | None) -> LoadedModel:
        if not self._models:
            raise LookupError(
                "No models loaded. Train one first: `enginepulse train --model ridge` "
                f"(artifacts go to {self.models_dir})."
            )
        key = key or self.default_key
        if key not in self._models:
            raise LookupError(f"Unknown model '{key}'. Loaded: {sorted(self._models)}")
        return self._models[key]

    def keys(self) -> list[str]:
        return sorted(self._models)


def _infer_window(art: Path) -> int:
    """Older artifacts may lack window_len in meta; fall back gracefully."""
    m = art / "meta.json"
    if m.exists():
        return int(json.loads(m.read_text()).get("window_len", 24))
    return 24

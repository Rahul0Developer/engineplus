"""Leakage-free preprocessing: engine-level splits, scaling, RUL capping.

The two rules this module exists to enforce:

1. Splits are by ENGINE ID, never by row. An engine's entire trajectory lives
   in exactly one of train/val/test.
2. Every scaler is fit on TRAINING engines only. Validation and test engines
   are transformed with those frozen statistics — the same thing you would
   have if you were really deploying to unseen engines.

Normalization schemes are pluggable (strategy pattern): ``GlobalScaler`` for
single-condition subsets (FD001/FD003) and ``PerConditionScaler`` for the
multi-condition subsets (FD002/FD004), where sensor levels shift with the
operating setting and a global scale washes out the degradation signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from enginepulse.data.loader import OP_COLS, SENSOR_CODES


def cap_rul(values: np.ndarray, max_rul: int | None) -> np.ndarray:
    """Piecewise-linear target: flatten everything above ``max_rul``."""
    if max_rul is None:
        return values.astype(np.float32)
    return np.minimum(values, float(max_rul)).astype(np.float32)


def split_train_val_by_engine(
    units: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly partition UNIQUE engine ids into train/val index masks.

    Returns boolean masks over rows such that no engine id appears in both.
    Deterministic given the seed.
    """
    unique_units = np.unique(units)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(unique_units) * val_fraction)))
    val_units = set(rng.choice(unique_units, size=n_val, replace=False).tolist())
    is_val = np.isin(units, list(val_units))
    return ~is_val, is_val


class FeatureScaler(ABC):
    """Fit-on-train-only transformation of the feature columns."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler": ...

    @abstractmethod
    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame: ...

    def fit_transform(self, df, feature_cols):
        return self.fit(df, feature_cols).transform(df, feature_cols)

    def state(self) -> dict:
        """Serialisable form so API/CLI can reload without joblib pickles."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict) -> "FeatureScaler":
        impl = {"global": GlobalScaler, "per_condition": PerConditionScaler}[state["kind"]]
        return impl._restore(state)

    @staticmethod
    def _restore(state: dict) -> "FeatureScaler":  # pragma: no cover - overridden
        raise NotImplementedError


class GlobalScaler(FeatureScaler):
    """Standardise each sensor with train-set mean/std."""

    def __init__(self) -> None:
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}

    def fit(self, df, feature_cols):
        for c in feature_cols:
            v = df[c].to_numpy(dtype=float)
            self.means_[c] = float(v.mean())
            std = float(v.std())
            self.stds_[c] = std if std > 1e-8 else 1.0
        return self

    def transform(self, df, feature_cols):
        out = df.copy()
        for c in feature_cols:
            out[c] = (out[c] - self.means_[c]) / self.stds_[c]
        return out

    def inverse_transform(self, X, feature_cols):
        """Undo scaling on a raw array whose last axis matches ``feature_cols``.

        Serving needs this to report window values back in sensor units; the
        pair lives together so nobody forgets one half of it.
        """
        X = np.asarray(X, dtype=float)
        mu = np.array([self.means_[c] for c in feature_cols], dtype=float)
        sd = np.array([self.stds_[c] for c in feature_cols], dtype=float)
        return X * sd + mu

    def state(self):
        return {"kind": "global", "means": self.means_, "stds": self.stds_}

    @staticmethod
    def _restore(state):
        s = GlobalScaler()
        s.means_, s.stds_ = state["means"], state["stds"]
        return s


class PerConditionScaler(FeatureScaler):
    """Standardise sensors within each operating condition separately.

    Conditions are defined by rounding the three op settings to integers —
    FD002/FD004 fly six discrete profiles, and the rounded triple identifies
    them reliably (verified against distinct-op counts during development).
    Statistics come from training engines only; unseen conditions fall back
    to the pooled train statistics so serving never crashes on a new profile.
    """

    def __init__(self) -> None:
        self.by_cond_: dict[tuple[int, int, int], dict[str, tuple[float, float]]] = {}
        self.pooled_: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _cond_key(row_op: np.ndarray) -> tuple[int, int, int]:
        return (int(round(row_op[0])), int(round(row_op[1])), int(round(row_op[2])))

    def fit(self, df, feature_cols):
        ops = df[OP_COLS].to_numpy(dtype=float)
        keys = np.array([self._cond_key(o) for o in ops])
        for cond in np.unique(keys, axis=0):
            mask = (keys == cond).all(axis=1)
            stats = {}
            for c in feature_cols:
                v = df.loc[mask, c].to_numpy(dtype=float)
                std = float(v.std())
                stats[c] = (float(v.mean()), std if std > 1e-8 else 1.0)
            self.by_cond_[tuple(cond)] = stats
        self.pooled_ = {
            c: (float(df[c].mean()), max(float(df[c].std()), 1e-8)) for c in feature_cols
        }
        return self

    def transform(self, df, feature_cols):
        out = df.copy()
        ops = out[OP_COLS].to_numpy(dtype=float)
        idx = np.arange(len(out))
        # Group rows by condition once instead of per-row dict lookups.
        key_arr = np.array([self._cond_key(o) for o in ops])
        for cond in np.unique(key_arr, axis=0):
            mask = (key_arr == cond).all(axis=1)
            stats = self.by_cond_.get(tuple(cond), self.pooled_)
            arr = out.loc[mask, feature_cols].to_numpy(dtype=float)
            mu = np.array([stats[c][0] for c in feature_cols])
            sd = np.array([stats[c][1] for c in feature_cols])
            out.loc[mask, feature_cols] = (arr - mu) / sd
        del idx
        return out

    def state(self):
        return {
            "kind": "per_condition",
            "by_cond": {str(k): v for k, v in self.by_cond_.items()},
            "pooled": self.pooled_,
        }

    @staticmethod
    def _restore(state):
        s = PerConditionScaler()
        import ast

        s.by_cond_ = {ast.literal_eval(k): v for k, v in state["by_cond"].items()}
        s.pooled_ = state["pooled"]
        return s


@dataclass
class PreparedData:
    """Everything a model needs after leakage-safe preprocessing."""

    dataset: str
    feature_cols: list[str]
    scaler: FeatureScaler
    window_len: int
    max_rul: int | None
    train_X: np.ndarray
    train_y: np.ndarray
    train_units: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    val_units: np.ndarray
    test_X: np.ndarray  # last window per test engine
    test_true_rul: np.ndarray
    test_units: np.ndarray
    meta: dict = field(default_factory=dict)

    def save_scaler(self, path: str | Path, extra: dict | None = None) -> None:
        """Persist scaler + column order so serving matches training exactly.

        `extra` lets the trainer freeze small serving-time context (e.g.
        validation-residual bounds for prediction intervals) into the bundle.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bundle = {"feature_cols": self.feature_cols, "scaler_state": self.scaler.state()}
        if extra:
            bundle.update(extra)
        joblib.dump(bundle, path)


def choose_scaler(dataset_name: str) -> FeatureScaler:
    """FD002/FD004 have multiple operating conditions; others do not."""
    if dataset_name in ("FD002", "FD004"):
        return PerConditionScaler()
    return GlobalScaler()


def prepare_dataset(
    ds,
    window_len: int = 24,
    max_rul: int | None = 125,
    val_fraction: float = 0.15,
    seed: int = 42,
    scaler: FeatureScaler | None = None,
    sensor_subset: list[str] | None = None,
) -> PreparedData:
    """Full pipeline: drop dead sensors -> split by engine -> fit scaler -> window.

    Order matters and is deliberate:
      * the train/val engine split happens BEFORE any statistic is computed;
      * the scaler sees only training-engine rows;
      * windows are built last, per engine, so they cannot straddle boundaries;
      * the test set is each test engine's final window (one prediction per
        engine), matching how NASA's ground truth is published.
    """
    from enginepulse.data.windows import (
        FullWindowSampler,
        LastWindowSampler,
        build_windows,
    )

    sens = [c for c in ds.train.columns if c in SENSOR_CODES]
    if sensor_subset:
        sens = [c for c in sens if c in sensor_subset]

    train_df = ds.train[["unit", "time", "rul"] + OP_COLS + sens].copy()
    test_df = ds.test[["unit", "time"] + OP_COLS + sens].copy()

    # 1) Engine-level train/val split (rows untouched until after fitting).
    train_units = train_df["unit"].to_numpy()
    tr_mask, val_mask = split_train_val_by_engine(train_units, val_fraction, seed)

    # 2) Fit scaler on TRAIN engines only.
    if scaler is None:
        scaler = choose_scaler(ds.name)
    scaler.fit(train_df.loc[tr_mask], sens)

    train_norm = scaler.transform(train_df, sens)
    test_norm = scaler.transform(test_df, sens)

    # 3) Cap the RUL target (piecewise linear) on the normalised frames.
    train_norm["rul"] = cap_rul(train_norm["rul"].to_numpy(), max_rul)

    # 4) Windows. Train sampler on all training-engine windows; then split
    #    the resulting windows by the engine-level masks computed above.
    full = build_windows(train_norm, sens, window_len, FullWindowSampler())
    is_val_unit = np.isin(full.units, train_units[val_mask])
    val_ws = full.subset(is_val_unit)
    tr_ws = full.subset(~is_val_unit)

    test_ws = build_windows(test_norm, sens, window_len, LastWindowSampler(), rul_col=None)
    # Attach each test engine's published ground-truth RUL to its last window.
    true_by_unit = {e.unit: e.true_rul for e in ds.engines("test")}
    test_true = np.array([true_by_unit[int(u)] for u in test_ws.units], dtype=np.float32)

    return PreparedData(
        dataset=ds.name,
        feature_cols=sens,
        scaler=scaler,
        window_len=window_len,
        max_rul=max_rul,
        train_X=tr_ws.X,
        train_y=tr_ws.y,
        train_units=tr_ws.units,
        val_X=val_ws.X,
        val_y=val_ws.y,
        val_units=val_ws.units,
        test_X=test_ws.X,
        test_true_rul=test_true,
        test_units=test_ws.units,
        meta={
            "n_train_windows": len(tr_ws),
            "n_val_windows": len(val_ws),
            "n_test_engines": len(test_ws),
            "dropped_sensors": [c for c in SENSOR_CODES if c not in sens],
            "scaler": type(scaler).__name__,
        },
    )

"""Classical regressors behind the RULModel interface: baseline, ridge, XGBoost."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from enginepulse.models.base import RULModel, clip_rul
from enginepulse.models.features import feature_names, window_features
from enginepulse.models.registry import register


@register("baseline")
class ConstantBaseline(RULModel):
    """Predicts the training-mean RUL for every window.

    Not a joke — it is the number every honest benchmark must print next to
    the fancy models, because on capped-RUL data it is surprisingly strong.
    """

    name = "baseline"

    def fit(self, X, y, X_val=None, y_val=None):
        self.const_ = float(np.mean(y))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        return np.full(len(X), self.const_, dtype=np.float32)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"const": self.const_}, p / "model.joblib")
        self._write_meta(p)

    def load(self, path):
        state = joblib.load(Path(path) / "model.joblib")
        self.const_ = state["const"]
        self.is_fitted_ = True
        return self


@register("ridge")
class RidgeModel(RULModel):
    """Ridge regression on window statistics (see features.py)."""

    name = "ridge"

    def __init__(self, seed: int = 42, alpha: float = 1.0, max_rul: float | None = 125.0, **kw):
        super().__init__(seed=seed, alpha=alpha, max_rul=max_rul, **kw)
        self.alpha = alpha
        self.max_rul = max_rul
        self.model_: Ridge | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0

    def _prep(self, X):
        F = window_features(X)
        return ((F - self.x_mean_) / self.x_std_).astype(np.float64)

    def fit(self, X, y, X_val=None, y_val=None):
        F = window_features(X).astype(np.float64)
        self.x_mean_ = F.mean(axis=0)
        self.x_std_ = np.maximum(F.std(axis=0), 1e-8)
        Fs = (F - self.x_mean_) / self.x_std_
        self.y_mean_ = float(y.mean())
        self.y_std_ = max(float(y.std()), 1e-8)
        ys = (y - self.y_mean_) / self.y_std_
        self.model_ = Ridge(alpha=self.alpha, random_state=self.seed)
        self.model_.fit(Fs, ys)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        raw = self.model_.predict(self._prep(X)) * self.y_std_ + self.y_mean_
        return clip_rul(raw, self.max_rul)

    def feature_importance(self):
        if self.model_ is None:
            return None
        return dict(zip([f"w{i}" for i in range(len(self.model_.coef_))], map(float, self.model_.coef_)))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "x_mean": self.x_mean_,
                "x_std": self.x_std_,
                "y_mean": self.y_mean_,
                "y_std": self.y_std_,
            },
            p / "model.joblib",
        )
        self._write_meta(p)

    def load(self, path):
        state = joblib.load(Path(path) / "model.joblib")
        self.model_ = state["model"]
        self.x_mean_, self.x_std_ = state["x_mean"], state["x_std"]
        self.y_mean_, self.y_std_ = state["y_mean"], state["y_std"]
        self.is_fitted_ = True
        return self


@register("xgb")
class XGBModel(RULModel):
    """XGBoost on the same engineered features; early stopping on val if given."""

    name = "xgb"

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        max_rul: float | None = 125.0,
        **kw,
    ):
        super().__init__(
            seed=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_rul=max_rul,
            **kw,
        )
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("xgb model needs `pip install 'enginepulse[xgb]'`") from exc
        self.model_ = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=seed,
            tree_method="hist",
            eval_metric="rmse",
            # xgb>=2.0 wants early stopping configured on the estimator, not fit().
            early_stopping_rounds=30,
        )
        self.max_rul = max_rul
        self.sensor_names_: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):
        F = window_features(X)
        kwargs = {}
        if X_val is not None and len(X_val):
            kwargs["eval_set"] = [(window_features(X_val), y_val)]
        else:
            # No validation split -> disable the constructor's early stopping.
            self.model_.set_params(early_stopping_rounds=None)
        self.model_.fit(F, y, verbose=False, **kwargs)
        if y_val is not None and len(y_val):
            resid = self.predict(X_val) - np.asarray(y_val, dtype=float)
            self.resid_std_ = float(np.std(resid))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        return clip_rul(self.model_.predict(window_features(X)), self.max_rul)

    def predict_quantiles(self, X, quantiles):
        """Booster-level quantile approximation via per-tree spread is not
        available cheaply; fall back to an empirical residual std around the
        point prediction. Honest, clearly-labelled uncertainty, not a fake one.
        """
        point = self.predict(X)
        resid = getattr(self, "resid_std_", None)
        if resid is None:
            raise RuntimeError("xgb: intervals need a fitted validation residual std")
        from scipy.stats import norm  # lazy: only needed for intervals

        out = {}
        for q in quantiles:
            z = norm.ppf(q)
            out[q] = np.clip(point + z * resid, 0.0, self.max_rul or np.inf)
        return out

    def feature_importance(self):
        imp = self.model_.feature_importances_
        names = self.sensor_names_ or feature_names([f"f{i}" for i in range(len(imp) // 6)])
        if len(names) != len(imp):
            names = [f"feat{i}" for i in range(len(imp))]
        return dict(zip(names, map(float, imp)))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(p / "model.ubj")
        self._write_meta(p, {"resid_std": getattr(self, "resid_std_", None)})

    def load(self, path):
        import xgboost as xgb

        p = Path(path)
        self.model_ = xgb.XGBRegressor()
        self.model_.load_model(p / "model.ubj")
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        self.resid_std_ = meta.get("resid_std")
        self.is_fitted_ = True
        return self




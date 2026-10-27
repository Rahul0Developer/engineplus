"""Train one model on one dataset, evaluate honestly, save a versioned result.

A "result" here is a small JSON document stamped with the git hash, package
versions, seed and full config — so any number in REPORT.md can be traced back
to the exact command that produced it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from enginepulse.data.loader import load_dataset
from enginepulse.data.preprocessing import prepare_dataset
from enginepulse.metrics import all_metrics, error_by_life_stage, late_prediction_fraction
from enginepulse.models.registry import build_model
from enginepulse.utils import get_logger, git_hash, package_versions, set_seed, write_json

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Everything needed to reproduce one experiment (mirrors the YAML configs)."""

    dataset: str = "FD001"
    model: str = "ridge"
    seed: int = 42
    window_len: int = 24
    max_rul: int | None = 125
    val_fraction: float = 0.15
    data_dir: str = "data"
    output_dir: str = "benchmarks/results"
    # Model-specific kwargs passed straight through to the constructor.
    model_params: dict[str, Any] = field(default_factory=dict)
    save_artifacts: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        extra = set(d) - known
        if extra:
            logger.warning("Ignoring unknown config keys: %s", sorted(extra))
        return cls(**{k: v for k, v in d.items() if k in known})


def infer_latency_ms(model, X: np.ndarray, n_reps: int = 5) -> dict[str, float]:
    """Time batch-1 CPU predictions; report p50/p95 in milliseconds.

    One window at a time is what an API actually does, so batching would lie.
    """
    lat = []
    for i in range(min(n_reps, len(X))):
        x1 = X[i : i + 1]
        t0 = time.perf_counter()
        model.predict(x1)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat_arr = np.asarray(lat)
    return {"p50_ms": float(np.percentile(lat_arr, 50)), "p95_ms": float(np.percentile(lat_arr, 95))}


def train_and_evaluate(cfg: TrainConfig) -> dict[str, Any]:
    """Run the full leakage-safe pipeline for one (dataset, model, seed)."""
    set_seed(cfg.seed)
    ds = load_dataset(cfg.data_dir, cfg.dataset)
    prep = prepare_dataset(
        ds,
        window_len=cfg.window_len,
        max_rul=cfg.max_rul,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    model = build_model(cfg.model, seed=cfg.seed, max_rul=cfg.max_rul, **cfg.model_params)
    t0 = time.perf_counter()
    model.fit(prep.train_X, prep.train_y, prep.val_X, prep.val_y)
    train_seconds = time.perf_counter() - t0

    pred = model.predict(prep.test_X)
    metrics = all_metrics(pred, prep.test_true_rul)
    metrics["late_prediction_fraction"] = late_prediction_fraction(pred, prep.test_true_rul)
    stage = error_by_life_stage(pred, prep.test_true_rul, threshold=float(cfg.max_rul or 125))
    latency = infer_latency_ms(model, prep.test_X)

    result: dict[str, Any] = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "seed": cfg.seed,
        "config": cfg.__dict__,
        "metrics": metrics,
        "error_by_life_stage": stage,
        "train_seconds": round(train_seconds, 3),
        "latency": latency,
        "n_train_windows": int(len(prep.train_X)),
        "n_val_windows": int(len(prep.val_X)),
        "n_test_engines": int(len(prep.test_X)),
        "prep_meta": prep.meta,
        "git": git_hash(),
        "versions": package_versions(),
    }

    # 90% interval coverage check, if the model offers quantiles at all.
    if cfg.model == "gru":
        try:
            q = model.predict_quantiles(prep.test_X, [0.05, 0.95])
            lo, hi = q[0.05], q[0.95]
            if not (np.isnan(lo).any() or np.isnan(hi).any()):
                cov = float(np.mean((prep.test_true_rul >= lo) & (prep.test_true_rul <= hi)))
                width = float(np.mean(hi - lo))
                result["uncertainty"] = {"coverage_90": cov, "mean_width_90": width}
        except RuntimeError as exc:
            logger.info("Skipping intervals: %s", exc)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.dataset}_{cfg.model}_seed{cfg.seed}"
    write_json(out_dir / f"{stem}.json", result)

    # Frozen 90% residual bounds from validation, so the API can serve
    # intervals for models without native quantile support.
    residual_bounds = None
    if len(prep.val_y) >= 20:
        val_pred = model.predict(prep.val_X)
        err = np.asarray(val_pred) - np.asarray(prep.val_y)
        residual_bounds = (float(np.quantile(err, 0.05)), float(np.quantile(err, 0.95)))

    if cfg.save_artifacts:
        art = out_dir / "artifacts" / stem
        prep.save_scaler(art / "scaler.joblib",
                         extra={"residual_bounds": residual_bounds})
        model.save(art)
        # Stamp serving-relevant context next to the weights so the API can
        # validate input shapes and apply the right scaler without retraining.
        _augment_artifact_meta(art, prep)
        logger.info("Saved artifacts to %s", art)

    logger.info(
        "%s/%s seed=%d -> RMSE %.2f MAE %.2f NASA %.1f (%.1fs)",
        cfg.dataset, cfg.model, cfg.seed,
        metrics["rmse"], metrics["mae"], metrics["nasa_score"], train_seconds,
    )
    return result


def _augment_artifact_meta(art: Path, prep) -> None:
    meta_path = art / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "window_len": prep.window_len,
        "max_rul": prep.max_rul,
        "feature_cols": prep.feature_cols,
        "dataset": prep.dataset,
    })
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

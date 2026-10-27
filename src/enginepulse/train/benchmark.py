"""Benchmark runner: many models x datasets x seeds -> one Markdown table.

`enginepulse benchmark` calls run_benchmark(); make_table() renders whatever
JSON results currently live in benchmarks/results/, so you can add one model's
results and regenerate the table without re-running everything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from enginepulse.train.trainer import TrainConfig, train_and_evaluate
from enginepulse.utils import get_logger

logger = get_logger(__name__)

# Models we attempt by default; missing extras are skipped with a warning,
# never a crash — a partial benchmark should still produce a table.
DEFAULT_MODELS = ["baseline", "ridge", "xgb", "cnn", "gru"]
DEFAULT_DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def run_benchmark(
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    base: dict | None = None,
    data_dir: str = "data",
    output_dir: str = "benchmarks/results",
) -> list[dict[str, Any]]:
    """Train/evaluate every combination and collect the result dicts."""
    models = models or DEFAULT_MODELS
    datasets = datasets or DEFAULT_DATASETS
    seeds = seeds or [42]
    base = base or {}

    results: list[dict[str, Any]] = []
    for ds in datasets:
        for m in models:
            for sd in seeds:
                cfg = TrainConfig.from_dict({**base, "dataset": ds, "model": m,
                                             "seed": sd, "data_dir": data_dir,
                                             "output_dir": output_dir})
                try:
                    res = train_and_evaluate(cfg)
                    results.append(res)
                except ImportError as exc:
                    logger.warning("Skipping %s on %s: %s", m, ds, exc)
                except Exception as exc:  # keep the benchmark resilient, but loud
                    logger.error("FAILED %s on %s seed %d: %s", m, ds, sd, exc)
    return results


def load_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Read every *_seed*.json result file in a directory."""
    out = []
    for p in sorted(Path(results_dir).glob("*_seed*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable result file %s", p)
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Mean/std across seeds per (dataset, model) for RMSE, MAE, NASA score."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in results:
        key = (r["dataset"], r["model"])
        g = grouped.setdefault(key, {"rmse": [], "mae": [], "nasa_score": []})
        for k in g:
            v = r["metrics"][k]
            if v is not None and math.isfinite(v):
                g[k].append(float(v))
    agg: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for key, g in grouped.items():
        agg[key] = {k: (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
                    for k, v in g.items()}
    return agg


_MODEL_ORDER = {"baseline": 0, "ridge": 1, "xgb": 2, "cnn": 3, "gru": 4, "transformer": 5}


def make_table(results: list[dict[str, Any]], title: str = "Benchmark results") -> str:
    """Render aggregated results as a Markdown table (mean ± std over seeds)."""
    agg = aggregate(results)
    lines = [
        f"## {title}",
        "",
        "| Dataset | Model | Seeds | RMSE (mean ± std) | MAE | NASA score |",
        "|---|---|---|---|---|---|",
    ]
    for (ds, m), vals in sorted(agg.items(), key=lambda kv: (kv[0][0], _MODEL_ORDER.get(kv[0][1], 9))):
        rm, mm, ns = vals["rmse"], vals["mae"], vals["nasa_score"]
        n_seeds = sum(1 for r in results if r["dataset"] == ds and r["model"] == m)
        lines.append(
            f"| {ds} | {m} | {n_seeds} | {rm[0]:.2f} ± {rm[1]:.2f} | "
            f"{mm[0]:.2f} ± {mm[1]:.2f} | {ns[0]:.1f} ± {ns[1]:.1f} |"
        )
    lines.append("")
    lines.append("_NASA score: lower is better; late (optimistic) predictions are penalised exponentially._")
    return "\n".join(lines)


def write_table(results: list[dict[str, Any]], path: str | Path, title: str = "Benchmark results") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(make_table(results, title=title) + "\n")
    return p

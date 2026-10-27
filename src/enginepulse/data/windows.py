"""Sliding-window construction that never crosses engine boundaries.

Two strategies are provided behind one interface (strategy pattern):

- ``LastWindowSampler`` (default, literature standard): the test set is each
  engine's *final* window only — you get exactly one prediction per engine,
  which is what the NASA score and the RUL_* ground truth are defined for.
- ``FullWindowSampler``: every window of every engine. Useful for training
  density and diagnostics, but scoring it against per-engine ground truth
  would double-count engines, so the benchmark pipeline does not use it for
  test evaluation.

Windows are produced with numpy stride tricks on a contiguous float32 matrix
(C-order), which keeps memory copies near zero even for FD004 (~61k rows).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowSet:
    """Aligned windows + metadata needed to keep splits honest."""

    X: np.ndarray  # (n_windows, window_len, n_features) float32
    y: np.ndarray  # (n_windows,) float32 — RUL at the window's LAST timestep
    units: np.ndarray  # (n_windows,) int — engine id each window came from
    times: np.ndarray  # (n_windows,) int — cycle index of the last timestep

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(self.X[mask], self.y[mask], self.units[mask], self.times[mask])


def _windows_for_engine(arr: np.ndarray, w: int) -> np.ndarray:
    """All length-w windows of a (T, F) array as a (T-w+1, w, F) view.

    Uses as_strided instead of sliding_window_view-free loops; no copy is made
    until the caller materialises it.
    """
    T = arr.shape[0]
    if T < w:
        return np.empty((0, w, arr.shape[1]), dtype=arr.dtype)
    strided = np.lib.stride_tricks.as_strided(
        arr,
        shape=(T - w + 1, w, arr.shape[1]),
        strides=(arr.strides[0], arr.strides[0], arr.strides[1]),
        writeable=False,
    )
    # Copy once into a clean contiguous block; downstream models expect safe
    # memory they can hold onto after the base array goes away.
    return np.array(strided, dtype=np.float32, copy=True)


class WindowSampler(ABC):
    """Decides WHICH windows of an engine become samples."""

    @abstractmethod
    def pick(self, n_steps: int, window_len: int) -> list[int]:
        """Return indices (into the per-engine window list) to keep."""


class FullWindowSampler(WindowSampler):
    """Every valid window (training default)."""

    def pick(self, n_steps: int, window_len: int) -> list[int]:
        n = max(n_steps - window_len + 1, 0)
        return list(range(n))


class LastWindowSampler(WindowSampler):
    """Only the final window (test-set convention)."""

    def pick(self, n_steps: int, window_len: int) -> list[int]:
        n = max(n_steps - window_len + 1, 0)
        return [n - 1] if n else []


class TailWindowsSampler(WindowSampler):
    """The k most recent windows; handy for latency studies."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def pick(self, n_steps: int, window_len: int) -> list[int]:
        n = max(n_steps - window_len + 1, 0)
        return list(range(max(n - self.k, 0), n))


def build_windows(
    df,
    feature_cols: list[str],
    window_len: int,
    sampler: WindowSampler,
    rul_col: str | None = "rul",
) -> WindowSet:
    """Build windows per engine from a dataframe with 'unit', 'time' columns.

    The RUL label of a window is the label of its LAST timestep — the value
    the model must predict at that moment. Windows never span two engines.
    """
    xs, ys, us, ts = [], [], [], []
    for unit in sorted(df["unit"].unique()):
        g = df[df["unit"] == unit].sort_values("time")
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        times = g["time"].to_numpy()
        all_w = _windows_for_engine(feats, window_len)
        if rul_col is not None and rul_col in g.columns:
            labels = g[rul_col].to_numpy(dtype=np.float32)
        else:
            labels = np.zeros(len(g), dtype=np.float32)
        # time is 1..T contiguous per engine, so the row index of the window's
        # last timestep is simply idx + window_len - 1.
        for idx in sampler.pick(len(g), window_len):
            last = idx + window_len - 1
            xs.append(all_w[idx])
            ys.append(labels[last])
            us.append(unit)
            ts.append(int(times[last]))
    return WindowSet(
        X=np.stack(xs) if xs else np.empty((0, window_len, len(feature_cols)), np.float32),
        y=np.asarray(ys, dtype=np.float32),
        units=np.asarray(us, dtype=np.int64),
        times=np.asarray(ts, dtype=np.int64),
    )

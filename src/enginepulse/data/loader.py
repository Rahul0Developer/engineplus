"""Pandas loaders for the raw C-MAPSS text files.

Column layout (per the dataset readme):
    unit index, time index, 3 operating settings, 21 sensor measurements.
Sensor columns are referenced by their official 1-based position ("1".."21");
positions 2, 11, 17, 20, 21 are constant in every subset and get dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Official short codes: sensor positions 1..21 as strings (keeps CSV headers stable).
SENSOR_CODES = [str(i) for i in range(1, 22)]

OP_COLS = ["op1", "op2", "op3"]

# Sensor positions (1-based) that are exactly zero/constant across all subsets.
CONSTANT_SENSOR_POSITIONS = (2, 11, 17, 20, 21)

# The same dead sensors expressed as column codes ("1".."21", i.e. position - 1).
# Code names are the stable, human-readable way to refer to a sensor; prefer
# these over raw positional indices when calling drop_constant_sensors directly.
CONSTANT_SENSOR_CODES = tuple(str(p - 1) for p in CONSTANT_SENSOR_POSITIONS)


@dataclass
class EngineData:
    """One engine's full run-to-failure trajectory."""

    unit: int
    df: pd.DataFrame  # columns: time, op1..op3, sensors..., plus 'rul' when known
    true_rul: int | None = None  # RUL of the LAST row (test engines); None for train


@dataclass
class Dataset:
    """A loaded C-MAPSS subset with per-engine access."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: np.ndarray | None = None  # ground truth RUL of each test engine's last row
    n_train_units: int = field(init=False)
    n_test_units: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_train_units = int(self.train["unit"].nunique())
        self.n_test_units = int(self.test["unit"].nunique())

    def engines(self, split: str) -> list[EngineData]:
        """Iterate engines of a split ('train' or 'test') in unit order."""
        df = self.train if split == "train" else self.test
        out: list[EngineData] = []
        for unit, g in df.groupby("unit", sort=True):
            true_rul = None
            if split == "test" and self.test_rul is not None:
                true_rul = int(self.test_rul[int(unit) - 1])
            out.append(EngineData(unit=int(unit), df=g.reset_index(drop=True), true_rul=true_rul))
        return out


def _read_table(path: Path) -> pd.DataFrame:
    cols = ["unit", "time"] + OP_COLS + SENSOR_CODES
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols, dtype=float)
    df["unit"] = df["unit"].astype(int)
    df["time"] = df["time"].astype(int)
    return df


def drop_constant_sensors(
    df: pd.DataFrame,
    min_unique: int = 3,
    dead_positions: tuple[int, ...] | None = None,
    dead_codes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Drop sensor columns that are constant or near-constant.

    A sensor is considered dead when it takes fewer than ``min_unique``
    distinct values across the whole table. This matches the usual C-MAPSS
    practice (positions 2, 11, 17, 20, 21 in 1-based numbering — codes
    '1', '10', '16', '19', '20' here — are always zero/flat). Removed codes
    are recorded in ``df.attrs['dropped_sensors']`` for reporting.

    ``dead_positions``, when given, is a set of official 1-based sensor
    positions to treat as dead *regardless* of their observed variance. The
    loader passes :data:`CONSTANT_SENSOR_POSITIONS` so the canonical flat
    columns are dropped even if a particular subset shows them varying (and
    so live sensors with only one or two measurements never sneak through).

    ``dead_codes`` is the code-name equivalent (column names "1".."21");
    prefer it over positional indices in tests and downstream code so the
    naming convention stays consistent with :data:`SENSOR_CODES`.
    """
    forced = {str(p - 1) for p in dead_positions} if dead_positions is not None else set()
    if dead_codes is not None:
        forced |= {str(c) for c in dead_codes}
    keep = [c for c in df.columns if c not in SENSOR_CODES]
    dropped = []
    for c in SENSOR_CODES:
        if c not in df.columns:
            continue
        if c in forced or df[c].nunique() < min_unique:
            dropped.append(c)
        else:
            keep.append(c)
    df.attrs["dropped_sensors"] = dropped
    return df[keep]


def load_dataset(data_dir: str | Path, name: str) -> Dataset:
    """Load e.g. 'FD001' from a directory containing the official txt files."""
    d = Path(data_dir)
    train_path = d / f"train_{name}.txt"
    test_path = d / f"test_{name}.txt"
    rul_path = d / f"RUL_{name}.txt"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run `enginepulse download` first.")

    train = _read_table(train_path)
    test = _read_table(test_path)

    # Ground-truth RUL file lists one number per test engine, in unit order.
    test_rul = None
    if rul_path.exists():
        test_rul = np.array([int(tok) for tok in rul_path.read_text().split()])
        n_engines = test["unit"].nunique()
        if len(test_rul) != n_engines:
            raise ValueError(
                f"{rul_path} has {len(test_rul)} entries but test file has "
                f"{n_engines} engines — file mismatch?"
            )

    # Unravelling (uncapped) RUL target on the training rows; the cap is applied
    # later in preprocessing so it stays configurable per experiment.
    train = add_train_rul(train)

    # Drop dead sensors consistently using BOTH splits' statistics so the kept
    # column set matches between train and test.
    combined = pd.concat([train, test], ignore_index=True)
    combined = drop_constant_sensors(combined, dead_positions=CONSTANT_SENSOR_POSITIONS)
    kept = [c for c in combined.columns if c in SENSOR_CODES]
    train = train[["unit", "time", "rul"] + OP_COLS + kept]
    test = test[["unit", "time"] + OP_COLS + kept]

    return Dataset(name=name, train=train, test=test, test_rul=test_rul)


def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Attach the unravelling (uncapped) RUL target to training rows.

    For an engine with N cycles, cycle t has RUL = N - t. The standard cap
    (piecewise-linear target) is applied in preprocessing.cap_rul().
    """
    df = train.copy()
    rul = np.zeros(len(df))
    times = df["time"].to_numpy()
    units = df["unit"].to_numpy()
    for u in np.unique(units):
        mask = units == u
        n = times[mask].max()
        rul[mask] = n - times[mask]
    df["rul"] = rul
    return df

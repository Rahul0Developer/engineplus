"""Model registry: name -> factory, populated by importing model modules.

Construction goes through the registry so configs stay plain YAML and the
benchmark never hard-codes an import graph. Deep models register themselves
only when torch is importable; asking for one without torch raises a clear
error pointing at the right extra.
"""

from __future__ import annotations

import importlib
from typing import Callable

from enginepulse.models.base import RULModel

_FACTORIES: dict[str, Callable[..., RULModel]] = {}


def register(name: str):
    def deco(factory: Callable[..., RULModel]) -> Callable[..., RULModel]:
        _FACTORIES[name] = factory
        return factory

    return deco


def _ensure_loaded() -> None:
    importlib.import_module("enginepulse.models.classical")
    try:
        importlib.import_module("enginepulse.models.deep")
    except ImportError:  # torch missing: deep models simply are not registered
        pass


# Model name -> pip extra that provides its dependency.
_EXTRAS = {"xgb": "xgb", "cnn": "deep", "gru": "deep", "transformer": "deep"}


def available_models() -> list[str]:
    """Names constructible in this environment right now."""
    _ensure_loaded()
    return sorted(_FACTORIES)


def build_model(name: str, **kwargs) -> RULModel:
    _ensure_loaded()
    if name not in _FACTORIES:
        extra = _EXTRAS.get(name)
        hint = f" Install the extra first: pip install 'enginepulse[{extra}]'" if extra else ""
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}.{hint}")
    return _FACTORIES[name](**kwargs)

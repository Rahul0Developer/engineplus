"""RUL models behind one interface. Import submodules to register them."""

from enginepulse.models.base import RULModel
from enginepulse.models.registry import available_models, build_model, register

__all__ = ["RULModel", "available_models", "build_model", "register"]

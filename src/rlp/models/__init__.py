"""Model factories and lightweight network definitions."""

from .base import TinyModel
from .fake_model import FakeModel, build_fake_model
from . import registry as _registry

__all__ = [
    "TinyModel",
    "FakeModel",
    "build_fake_model",
] + list(getattr(_registry, "__all__", []))

for _name in getattr(_registry, "__all__", []):
    globals()[_name] = getattr(_registry, _name)

del _name, _registry

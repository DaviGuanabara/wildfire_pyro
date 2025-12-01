from typing import Type
from wildfire_pyro.common.baselines.BaselineStrategy import BaselineStrategy

class BaselineRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, baseline_class):
        if name in cls._registry:
            raise ValueError(f"Baseline '{name}' is already registered.")
        cls._registry[name] = baseline_class

    @classmethod
    def get(cls, name: str) -> Type[BaselineStrategy]:
        if name not in cls._registry:
            raise ValueError(f"Baseline '{name}' is not registered.")
        return cls._registry[name]

def register_baseline(name: str):

    def decorator(cls):
        BaselineRegistry.register(name, cls)
        return cls
    return decorator

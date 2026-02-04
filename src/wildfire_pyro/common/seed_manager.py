import numpy as np
import torch
from typing import Optional
import hashlib


class SeedManager:
    def __init__(self, global_seed: int):
        self.global_seed = global_seed
        self.cache = {}

    def get_seed(self, key: str) -> int:
        full_key = f"{self.global_seed}-{key}"
        digest = hashlib.sha256(full_key.encode()).hexdigest()
        return int(digest, 16) % (2**32 - 1)

    def get_rng(self, key: str) -> np.random.Generator:
        if key not in self.cache:
            self.cache[key] = np.random.default_rng(self.get_seed(key))
        return self.cache[key]

    def get_torch_generator(self, key: str) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.get_seed(key))
        return g


# ✅ Singleton global
_seed_manager: Optional[SeedManager] = None


def configure_seed_manager(global_seed: int) -> SeedManager:
    global _seed_manager
    _seed_manager = SeedManager(global_seed)
    return _seed_manager


def get_seed_manager() -> SeedManager:
    if _seed_manager is None:
        raise RuntimeError(
            "SeedManager has not been configured yet. Set global seed first calling configure_seed_manager"
        )
    return _seed_manager


def get_global_seed() -> int:
    return _seed_manager.global_seed # type: ignore


def get_seed(key: str) -> int:
    seed_manager = get_seed_manager()
    return seed_manager.get_seed(key)

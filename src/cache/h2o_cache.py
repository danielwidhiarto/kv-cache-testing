"""KV cache with H2O (Heavy-Hitter Oracle) eviction."""

import torch
from typing import Optional, Tuple
from .base import KVCache
from ..policies.h2o import H2OPolicy


class H2OCache(KVCache):
    """KV cache that evicts tokens with lowest cumulative attention."""

    def __init__(self, max_size: int = 2048, heavy_ratio: float = 0.1):
        self._max_size = max_size
        self._keys: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._policy = H2OPolicy(heavy_ratio=heavy_ratio)

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> None:
        if self._keys is None:
            self._keys = key
            self._values = value
        else:
            self._keys = torch.cat([self._keys, key], dim=2)
            self._values = torch.cat([self._values, value], dim=2)

        # Evict if over capacity
        while self.size() > self._max_size:
            num_to_evict = self.size() - self._max_size
            indices = self._policy.select_evict(
                self._keys, self._values, attention_scores, num_to_evict
            )
            if indices.numel() == 0:
                break
            self.evict(indices)

    def evict(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        mask = torch.ones(self._keys.shape[2], dtype=torch.bool, device=self._keys.device)
        mask[indices] = False
        self._keys = self._keys[:, :, mask, :]
        self._values = self._values[:, :, mask, :]
        self._policy.on_evict(indices)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._keys is None:
            raise ValueError("Cache is empty")
        return self._keys, self._values

    def reset(self) -> None:
        self._keys = None
        self._values = None
        self._policy.reset()

    def size(self) -> int:
        if self._keys is None:
            return 0
        return self._keys.shape[2]

    @property
    def max_size(self) -> int:
        return self._max_size

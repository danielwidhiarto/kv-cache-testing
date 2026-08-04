"""Generic KV cache that delegates eviction decisions to an EvictionPolicy."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import KVCache
from ..policies.base import EvictionPolicy


class PolicyCache(KVCache):
    """A real, bounded KV cache for one transformer layer.

    The cache owns one policy instance. A separate instance must be used for
    every layer because attention statistics are layer-specific.
    """

    def __init__(self, max_size: int, policy: EvictionPolicy):
        self._max_size = max(1, int(max_size))
        self._policy = policy
        self._keys: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._evicted_tokens = 0

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> None:
        if key.shape[2] != value.shape[2]:
            raise ValueError("Key/value sequence lengths must match")

        if self._keys is None:
            self._keys = key
            self._values = value
        else:
            self._keys = torch.cat([self._keys, key], dim=2)
            self._values = torch.cat([self._values, value], dim=2)

        target_max_size = (
            self._policy.get_max_size(self._max_size)
            if hasattr(self._policy, "get_max_size")
            else self._max_size
        )

        # Attention scores correspond to the cache state *before* any eviction
        # in this append call. After the first eviction the cache shrinks but
        # the original attention tensor still has the old seq_len, so reusing it
        # causes size mismatch (e.g. 1423 vs 2023). Only pass it once.
        attn_for_evict = attention_scores
        while self.size() > target_max_size:
            overflow = self.size() - target_max_size
            indices = self._policy.select_evict(
                self._keys,
                self._values,
                attention_scores=attn_for_evict,
                num_to_evict=overflow,
            )
            attn_for_evict = None

            # A policy must not be able to leave an over-budget cache forever.
            # This fallback is only for invalid/incompatible policy settings;
            # it preserves the first token and removes the oldest middle tokens.
            if indices.numel() == 0:
                candidate = torch.arange(
                    1,
                    self.size(),
                    device=self._keys.device,
                    dtype=torch.long,
                )
                if candidate.numel() == 0:
                    candidate = torch.zeros(1, device=self._keys.device, dtype=torch.long)
                indices = candidate[:overflow]

            self.evict(indices)

    def evict(self, indices: torch.Tensor) -> None:
        if self._keys is None or indices.numel() == 0:
            return

        indices = torch.unique(indices.to(device=self._keys.device, dtype=torch.long))
        indices = indices[(indices >= 0) & (indices < self.size())]
        if indices.numel() == 0:
            return

        mask = torch.ones(self.size(), dtype=torch.bool, device=self._keys.device)
        mask[indices] = False
        self._keys = self._keys[:, :, mask, :]
        self._values = self._values[:, :, mask, :]
        self._policy.on_evict(indices)
        self._evicted_tokens += int(indices.numel())

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._keys is None:
            raise ValueError("Cache is empty")
        return self._keys, self._values

    def reset(self) -> None:
        self._keys = None
        self._values = None
        self._evicted_tokens = 0
        self._policy.reset()

    def size(self) -> int:
        return 0 if self._keys is None else int(self._keys.shape[2])

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def evicted_tokens(self) -> int:
        return self._evicted_tokens

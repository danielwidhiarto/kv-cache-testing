"""LRU (Least Recently Used) eviction policy."""

import torch
from typing import Optional
from .base import EvictionPolicy


class LRUPolicy(EvictionPolicy):
    """Evicts the least recently used tokens.

    Tracks access timestamps per token position.
    Tokens that haven't been attended to recently get evicted first.
    """

    def __init__(self):
        self._last_access: Optional[torch.Tensor] = None
        self._step: int = 0

    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        seq_len = key_cache.shape[2]

        # Initialize access tracking
        if self._last_access is None or self._last_access.shape[0] != seq_len:
            self._last_access = torch.zeros(seq_len, dtype=torch.long, device=key_cache.device)

        # Update access timestamps for tokens that received attention
        if attention_scores is not None:
            # attention_scores: [batch, num_heads, query_len, seq_len]
            # Average across batch, heads, query positions
            avg_attn = attention_scores.float().mean(dim=(0, 1, 2))  # [seq_len]
            accessed = avg_attn > avg_attn.median()
            self._last_access[accessed] = self._step

        self._step += 1

        # Evict tokens with oldest access timestamps
        _, oldest_indices = torch.topk(self._last_access.float(), num_to_evict, largest=False)
        return oldest_indices

    def reset(self):
        self._last_access = None
        self._step = 0

    @property
    def name(self) -> str:
        return "lru"

    @property
    def requires_attention_scores(self) -> bool:
        return True

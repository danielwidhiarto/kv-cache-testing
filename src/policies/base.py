"""Abstract base class for KV cache eviction policies."""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class EvictionPolicy(ABC):
    """Base class for eviction policies.

    An eviction policy decides which KV cache entries to remove
    when the cache reaches capacity.
    """

    @abstractmethod
    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        """Select indices to evict from the KV cache.

        Args:
            key_cache: Shape [batch, num_heads, seq_len, head_dim]
            value_cache: Shape [batch, num_heads, seq_len, head_dim]
            attention_scores: Optional attention scores for importance-based policies.
                Shape [batch, num_heads, query_len, seq_len]
            num_to_evict: Number of entries to evict.

        Returns:
            LongTensor of indices to evict, shape [num_to_evict]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""
        pass

    @property
    def requires_attention_scores(self) -> bool:
        """Whether this policy needs attention scores to make decisions."""
        return False

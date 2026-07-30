"""Abstract base class for KV cache implementations."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class KVCache(ABC):
    """Base class for KV cache with eviction support.

    Stores key-value pairs from transformer attention layers
    and applies eviction policies when capacity is reached.
    """

    @abstractmethod
    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Append new KV pairs to the cache.

        Args:
            key: Shape [batch, num_heads, new_len, head_dim]
            value: Shape [batch, num_heads, new_len, head_dim]
            attention_scores: Optional attention scores for importance tracking.
        """
        pass

    @abstractmethod
    def evict(self, indices: torch.Tensor) -> None:
        """Remove entries at given indices.

        Args:
            indices: LongTensor of positions to evict.
        """
        pass

    @abstractmethod
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current cached KV pairs.

        Returns:
            (key_cache, value_cache) each of shape [batch, num_heads, seq_len, head_dim]
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear the cache."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Current number of cached tokens."""
        pass

    @property
    @abstractmethod
    def max_size(self) -> int:
        """Maximum cache capacity in tokens."""
        pass

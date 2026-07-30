"""H2O (Heavy-Hitter Oracle) eviction policy.

Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
of Large Language Models" (Zhang et al., 2023)
"""

import torch
from typing import Optional
from .base import EvictionPolicy


class H2OPolicy(EvictionPolicy):
    """Evicts tokens with lowest cumulative attention scores.

    Maintains a running sum of attention scores per token position.
    Tokens with lowest cumulative attention ("heavy hitters" are kept)
    are evicted first.
    """

    def __init__(self, heavy_ratio: float = 0.1):
        """Initialize H2O policy.

        Args:
            heavy_ratio: Fraction of cache reserved for heavy hitters.
                E.g., 0.1 means 10% of cache is always reserved for
                the highest-attention tokens.
        """
        self.heavy_ratio = heavy_ratio
        self._cumulative_attn: Optional[torch.Tensor] = None

    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        seq_len = key_cache.shape[2]

        # Initialize cumulative attention
        if self._cumulative_attn is None:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=key_cache.device)
        elif self._cumulative_attn.shape[0] < seq_len:
            padding = torch.zeros(
                seq_len - self._cumulative_attn.shape[0],
                dtype=self._cumulative_attn.dtype,
                device=key_cache.device,
            )
            self._cumulative_attn = torch.cat([self._cumulative_attn, padding])
        elif self._cumulative_attn.shape[0] > seq_len:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=key_cache.device)

        # Accumulate attention scores
        if attention_scores is not None:
            # attention_scores: [batch, num_heads, query_len, seq_len]
            # Sum across batch, heads, query positions
            attn_sum = attention_scores.float().sum(dim=(0, 1, 2))  # [seq_len]
            self._cumulative_attn += attn_sum

        # Don't evict heavy hitters
        heavy_count = max(1, int(seq_len * self.heavy_ratio))
        if seq_len - num_to_evict < heavy_count:
            num_to_evict = max(0, seq_len - heavy_count)

        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long, device=key_cache.device)

        # Evict tokens with lowest cumulative attention
        _, lowest_indices = torch.topk(
            self._cumulative_attn, num_to_evict, largest=False
        )
        return lowest_indices

    def reset(self):
        self._cumulative_attn = None

    def on_evict(self, indices: torch.Tensor) -> None:
        if self._cumulative_attn is not None and indices.numel() > 0:
            keep = torch.ones(
                self._cumulative_attn.shape[0],
                dtype=torch.bool,
                device=self._cumulative_attn.device,
            )
            keep[indices.to(device=keep.device, dtype=torch.long)] = False
            self._cumulative_attn = self._cumulative_attn[keep]

    @property
    def name(self) -> str:
        return f"h2o_{self.heavy_ratio}"

    @property
    def requires_attention_scores(self) -> bool:
        return True

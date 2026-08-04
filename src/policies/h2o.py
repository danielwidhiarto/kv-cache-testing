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

    def __init__(self, heavy_ratio: float = 0.1, sink_size: int = 4, window_size: int = 64):
        """Initialize H2O policy.

        Args:
            heavy_ratio: Fraction of cache reserved for heavy hitters.
            sink_size: Number of initial tokens to protect (attention sinks).
            window_size: Number of recent tokens to protect.
        """
        self.heavy_ratio = heavy_ratio
        self.sink_size = sink_size
        self.window_size = window_size
        self._cumulative_attn: Optional[torch.Tensor] = None

    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        seq_len = key_cache.shape[2]
        device = key_cache.device

        # Protect sinks and window
        protected_end = self.sink_size
        window_start = max(self.sink_size, seq_len - self.window_size)

        evictable_count = window_start - protected_end
        if evictable_count <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        num_to_evict = min(num_to_evict, evictable_count)
        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # Initialize cumulative attention
        if self._cumulative_attn is None:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
        elif self._cumulative_attn.shape[0] < seq_len:
            padding = torch.zeros(
                seq_len - self._cumulative_attn.shape[0],
                dtype=self._cumulative_attn.dtype,
                device=device,
            )
            self._cumulative_attn = torch.cat([self._cumulative_attn, padding])
        elif self._cumulative_attn.shape[0] > seq_len:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)

        # Accumulate attention scores — guard mismatched size after eviction loop
        if attention_scores is not None:
            if attention_scores.shape[3] == self._cumulative_attn.shape[0]:
                attn_sum = attention_scores.float().sum(dim=(0, 1, 2))  # [seq_len]
                self._cumulative_attn += attn_sum
            elif attention_scores.shape[3] > self._cumulative_attn.shape[0]:
                # stale larger attention from pre-evict iteration — skip
                pass
            else:
                pass

        # Evict tokens with lowest cumulative attention in middle range only
        middle_attn = self._cumulative_attn[protected_end:window_start]
        _, lowest_local = torch.topk(middle_attn, num_to_evict, largest=False)
        return lowest_local + protected_end

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

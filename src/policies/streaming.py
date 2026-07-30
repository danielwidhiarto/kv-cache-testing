"""StreamingLLM eviction policy.

Based on: "Efficient Streaming Language Models with Attention Sinks"
(Xiao et al., ICLR 2024)

Keeps attention sink tokens (first N) + sliding window of recent tokens.
"""

import torch
from typing import Optional
from .base import EvictionPolicy


class StreamingPolicy(EvictionPolicy):
    """Keeps attention sinks + recent window.

    Attention sinks are the first few tokens that receive disproportionately
    high attention regardless of content. Combined with a sliding window
    of recent tokens, this enables stable streaming inference.
    """

    def __init__(self, sink_size: int = 4, window_size: int = 512):
        """Initialize StreamingLLM policy.

        Args:
            sink_size: Number of initial tokens to always keep (attention sinks).
            window_size: Number of recent tokens to keep in the sliding window.
        """
        self.sink_size = sink_size
        self.window_size = window_size

    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        seq_len = key_cache.shape[2]
        keep_size = self.sink_size + self.window_size

        if seq_len <= keep_size:
            return torch.tensor([], dtype=torch.long, device=key_cache.device)

        # Evict everything between sinks and window
        # Keep: [0..sink_size) and [seq_len - window_size..seq_len)
        # Evict: [sink_size..seq_len - window_size)
        evict_start = self.sink_size
        evict_end = seq_len - self.window_size

        # Only evict num_to_evict at a time from the middle
        evict_indices = torch.arange(
            evict_start,
            min(evict_start + num_to_evict, evict_end),
            dtype=torch.long,
            device=key_cache.device,
        )
        return evict_indices

    def reset(self):
        pass  # Stateless policy

    @property
    def name(self) -> str:
        return f"streaming_s{self.sink_size}_w{self.window_size}"

    @property
    def requires_attention_scores(self) -> bool:
        return False

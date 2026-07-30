"""SnapKV eviction policy.

Based on: "SnapKV: LLM Knows What You are Looking for Before Generation"
(Li et al., 2024)

Uses per-head attention patterns from an observation window to score
token importance, then retains the most important tokens.
"""

import torch
from typing import Optional
from .base import EvictionPolicy


class SnapPolicy(EvictionPolicy):
    """Per-head attention-based eviction using observation window.

    Scans attention patterns from the last few query positions
    (observation window) to identify which KV entries each head
    considers most important.
    """

    def __init__(self, observation_size: int = 32, retain_ratio: float = 0.5):
        """Initialize SnapKV policy.

        Args:
            observation_size: Number of recent query positions to observe.
            retain_ratio: Fraction of tokens to retain after eviction.
        """
        self.observation_size = observation_size
        self.retain_ratio = retain_ratio

    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        seq_len = key_cache.shape[2]
        device = key_cache.device

        if attention_scores is None:
            # Fallback: evict from the middle (not ideal, but safe)
            mid = seq_len // 2
            return torch.arange(
                max(0, mid - num_to_evict // 2),
                min(seq_len, mid + num_to_evict // 2 + 1),
                dtype=torch.long,
                device=device,
            )

        # attention_scores: [batch, num_heads, query_len, seq_len]
        # Use last observation_size query positions
        obs_size = min(self.observation_size, attention_scores.shape[2])
        obs_attn = attention_scores[:, :, -obs_size:, :]  # [batch, heads, obs, seq]

        # Per-head importance: sum across batch and observation positions
        head_importance = obs_attn.float().sum(dim=(0, 2))  # [heads, seq_len]

        # Aggregate across heads (mean)
        token_importance = head_importance.mean(dim=0)  # [seq_len]

        # Evict least important tokens
        _, evict_indices = torch.topk(token_importance, num_to_evict, largest=False)
        return evict_indices

    def reset(self):
        pass  # Stateless policy

    @property
    def name(self) -> str:
        return f"snap_o{self.observation_size}_r{self.retain_ratio}"

    @property
    def requires_attention_scores(self) -> bool:
        return True

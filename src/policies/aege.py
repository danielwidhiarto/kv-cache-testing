"""AEGE (Attention Entropy-Guided Eviction) policy.

Novel policy that uses attention entropy as an eviction signal.
Low-entropy positions = tokens everyone "agrees" are important → keep.
High-entropy positions = divided attention → evict first.

Combines:
- Attention sink protection (first K tokens, from StreamingLLM insight)
- Recency window (last W tokens)
- Entropy-weighted importance scoring for middle positions
"""

import torch
from typing import Optional
from .base import EvictionPolicy


class AEGEPolicy(EvictionPolicy):
    """Evicts tokens based on entropy-weighted attention importance.

    Score = cumulative_attention × (1 - entropy_weight × normalized_entropy)

    Low entropy → high score → keep
    High entropy → low score → evict first
    """

    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 64,
        entropy_weight: float = 0.3,
        temporal_decay: float = 0.95,
    ):
        self.sink_size = sink_size
        self.window_size = window_size
        self.entropy_weight = entropy_weight
        self.temporal_decay = temporal_decay
        self._cumulative_attn: Optional[torch.Tensor] = None
        self._entropy_history: Optional[torch.Tensor] = None

    def _compute_entropy(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy per key position.

        Args:
            attention_scores: [batch, num_heads, query_len, seq_len]

        Returns:
            Entropy per position, shape [seq_len]
        """
        # Mean attention weights across batch and heads -> [query_len, seq_len]
        mean_attn = attention_scores.float().mean(dim=(0, 1))

        # Normalize across key positions for each query token -> sum_k p_{q,k} = 1
        attn_probs = mean_attn / (mean_attn.sum(dim=-1, keepdim=True) + 1e-10)

        # Shannon entropy per query position over keys
        # H_q = - sum_k (p_{q,k} * log p_{q,k})
        per_query_entropy = -(attn_probs * (attn_probs + 1e-10).log()).sum(dim=-1)  # [query_len]

        # For per-key position entropy (how much attention variance/dispersion reaches this key)
        # Average key attention weight across queries, normalized as distribution
        key_weights = mean_attn.mean(dim=0)  # [seq_len]
        key_probs = key_weights / (key_weights.sum() + 1e-10)
        
        # Per-position entropy weighting
        per_pos_entropy = -(key_probs * (key_probs + 1e-10).log())  # [seq_len]

        return per_pos_entropy

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

        # Count evictable positions (middle only)
        evictable_count = window_start - protected_end
        if evictable_count <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        num_to_evict = min(num_to_evict, evictable_count)
        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # Initialize state
        if self._cumulative_attn is None:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._entropy_history = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._has_entropy = False
        elif self._cumulative_attn.shape[0] < seq_len:
            padding = torch.zeros(
                seq_len - self._cumulative_attn.shape[0],
                dtype=self._cumulative_attn.dtype,
                device=device,
            )
            self._cumulative_attn = torch.cat([self._cumulative_attn, padding])
            entropy_padding = torch.zeros(
                seq_len - self._entropy_history.shape[0],
                dtype=self._entropy_history.dtype,
                device=device,
            )
            self._entropy_history = torch.cat([self._entropy_history, entropy_padding])
        elif self._cumulative_attn.shape[0] > seq_len:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._entropy_history = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._has_entropy = False

        if attention_scores is not None:
            # Accumulate attention with temporal decay
            self._cumulative_attn *= self.temporal_decay
            attn_sum = attention_scores.float().sum(dim=(0, 1, 2))  # [seq_len]
            self._cumulative_attn += attn_sum

            # Compute entropy during prefill (query_len > 1) or first call without GPU-CPU sync
            query_len = attention_scores.shape[2]
            if query_len > 1 or not self._has_entropy:
                entropy = self._compute_entropy(attention_scores)
                self._entropy_history[:entropy.shape[0]] = entropy
                self._has_entropy = True
            else:
                self._entropy_history *= self.temporal_decay

        # Normalize scores for middle positions purely on GPU (no Python syncs)
        middle_start = protected_end
        middle_end = window_start

        middle_attn = self._cumulative_attn[middle_start:middle_end]
        middle_entropy = self._entropy_history[middle_start:middle_end]

        attn_min, attn_max = middle_attn.min(), middle_attn.max()
        attn_denom = (attn_max - attn_min).clamp_min(1e-10)
        norm_attn = (middle_attn - attn_min) / attn_denom

        ent_min, ent_max = middle_entropy.min(), middle_entropy.max()
        ent_denom = (ent_max - ent_min).clamp_min(1e-10)
        norm_entropy = (middle_entropy - ent_min) / ent_denom

        # Score: high attn + low entropy = keep
        # score = attn × (1 - entropy_weight × entropy)
        scores = norm_attn * (1.0 - self.entropy_weight * norm_entropy)

        # Evict lowest scores
        _, lowest_local = torch.topk(scores, num_to_evict, largest=False)
        evict_indices = lowest_local + middle_start  # offset to global positions

        return evict_indices

    def reset(self):
        self._cumulative_attn = None
        self._entropy_history = None
        self._has_entropy = False

    def on_evict(self, indices: torch.Tensor) -> None:
        if self._cumulative_attn is None or indices.numel() == 0:
            return
        keep = torch.ones(
            self._cumulative_attn.shape[0],
            dtype=torch.bool,
            device=self._cumulative_attn.device,
        )
        keep[indices.to(device=keep.device, dtype=torch.long)] = False
        self._cumulative_attn = self._cumulative_attn[keep]
        self._entropy_history = self._entropy_history[keep]

    @property
    def name(self) -> str:
        return f"aege_s{self.sink_size}_w{self.window_size}_ew{self.entropy_weight}"

    @property
    def requires_attention_scores(self) -> bool:
        return True

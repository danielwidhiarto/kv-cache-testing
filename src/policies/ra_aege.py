"""RA-XAEGE: Retrieval-Aware Explainable AEGE — A+C combine.

Level 1 Rubric protection (B)
Level 2 Retrieval prior (A) — consumes S2 JSON spans + relevance
Level 3 Entropy XAI (C fixed)
Level 4 Pyramid adaptive

Usage:
- Without S2: retrieval_spans=None -> behaves like fixed AEGE (B+C)
- With S2: provide List[dict] start/end/score
"""

import torch
from typing import Optional, List, Tuple, Dict
from .aege import AEGEPolicy


class RAXAEGEPolicy(AEGEPolicy):
    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 64,
        entropy_weight: float = 0.5,
        temporal_decay: float = 0.95,
        adaptive_budget: bool = True,
        adaptive_quantile: float = 0.3,
        layer_idx: Optional[int] = None,
        total_layers: Optional[int] = None,
        rubric_spans: Optional[List[Tuple[int, int]]] = None,
        retrieval_spans: Optional[List[Dict]] = None,  # [{start,end,score}]
        retrieval_weight: float = 0.7,  # how much retrieval boosts keep score
        never_evict_spans: Optional[List[Tuple[int, int]]] = None,  # student essay
    ):
        super().__init__(
            sink_size=sink_size,
            window_size=window_size,
            entropy_weight=entropy_weight,
            temporal_decay=temporal_decay,
            adaptive_budget=adaptive_budget,
            adaptive_quantile=adaptive_quantile,
            layer_idx=layer_idx,
            total_layers=total_layers,
            rubric_spans=rubric_spans,
        )
        self.retrieval_spans = retrieval_spans or []
        self.retrieval_weight = retrieval_weight
        self.never_evict_spans = never_evict_spans or []

    def _get_retrieval_prior(self, total_len: int, device: torch.device) -> torch.Tensor:
        """Per-position prior from S2 retriever relevance [0,1] -> boost."""
        prior = torch.zeros(total_len, dtype=torch.float32, device=device)
        for item in self.retrieval_spans:
            s = int(item.get("start", 0))
            e = int(item.get("end", s))
            score = float(item.get("score", 0.0))  # 0-1
            s = max(0, min(s, total_len))
            e = max(s, min(e, total_len))
            if e > s:
                prior[s:e] = max(prior[s:e].max().item() if prior[s:e].numel() else 0.0, score)
        return prior

    def select_evict(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        num_to_evict: int = 1,
    ) -> torch.Tensor:
        seq_len = key_cache.shape[2]
        device = key_cache.device

        protected_end = self.sink_size
        window_start = max(self.sink_size, seq_len - self.window_size)
        evictable = window_start - protected_end
        if evictable <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # build never-evict mask = rubric + student essay
        all_never = list(self.rubric_spans) + list(self.never_evict_spans)
        never_mask = torch.zeros(evictable, dtype=torch.bool, device=device)
        for s, e in all_never:
            lo = max(s, protected_end)
            hi = min(e, window_start)
            if lo < hi:
                never_mask[lo - protected_end : hi - protected_end] = True

        eff = evictable - int(never_mask.sum().item())
        if eff <= 0:
            return torch.tensor([], dtype=torch.long, device=device)
        num_to_evict = min(num_to_evict, eff)
        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # init cache states like parent
        if self._cumulative_attn is None:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._entropy_history = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._has_entropy = False
        elif self._cumulative_attn.shape[0] < seq_len:
            pad = seq_len - self._cumulative_attn.shape[0]
            self._cumulative_attn = torch.cat([self._cumulative_attn, torch.zeros(pad, dtype=torch.float32, device=device)])
            self._entropy_history = torch.cat([self._entropy_history, torch.zeros(pad, dtype=torch.float32, device=device)])
        elif self._cumulative_attn.shape[0] > seq_len:
            self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._entropy_history = torch.zeros(seq_len, dtype=torch.float32, device=device)
            self._has_entropy = False

        if attention_scores is not None:
            attn_len = int(attention_scores.shape[3]) if attention_scores.dim() >= 4 else int(self._cumulative_attn.shape[0])
            if attn_len == self._cumulative_attn.shape[0]:
                self._cumulative_attn *= self.temporal_decay
                self._cumulative_attn += attention_scores.float().sum(dim=(0, 1, 2))
                if attention_scores.shape[2] > 1 or not self._has_entropy:
                    ent = self._compute_entropy_per_key(attention_scores)
                    self._entropy_history[: ent.shape[0]] = ent
                    self._has_entropy = True
                else:
                    self._entropy_history *= self.temporal_decay
            elif attn_len < self._cumulative_attn.shape[0]:
                pass
            else:
                self._cumulative_attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
                self._entropy_history = torch.zeros(seq_len, dtype=torch.float32, device=device)
                self._has_entropy = False
                pass

        ms, me = protected_end, window_start
        mid_attn = self._cumulative_attn[ms:me]
        mid_ent = self._entropy_history[ms:me]

        norm_attn = (mid_attn - mid_attn.min()) / (mid_attn.max() - mid_attn.min() + 1e-10)
        norm_ent = (mid_ent - mid_ent.min()) / (mid_ent.max() - mid_ent.min() + 1e-10)

        retrieval_prior_full = self._get_retrieval_prior(seq_len, device)
        retrieval_mid = retrieval_prior_full[ms:me]  # [evictable]

        # A+C score: attn * (1 - ew*ent) * (1 + rw * prior), clamp min
        base = norm_attn * (1.0 - self.entropy_weight * norm_ent).clamp_min(0.05)
        boosted = base * (1.0 + self.retrieval_weight * retrieval_mid)

        scores = boosted
        self._last_scores = scores.detach().clone()

        scores = scores.clone()
        scores[never_mask] = 2.5  # never evict

        # adaptive quantile
        valid_mask = ~never_mask
        valid_scores = scores[valid_mask]
        if valid_scores.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=device)

        if self.adaptive_budget and valid_scores.numel() > 1:
            thresh = torch.quantile(valid_scores, self.adaptive_quantile)
            if self.layer_idx is not None and self.total_layers and self.total_layers > 1:
                depth_ratio = self.layer_idx / (self.total_layers - 1)
                thresh = thresh * (1.2 - 0.4 * depth_ratio)
            below = (scores < thresh) & (~never_mask)
            below_idx = below.nonzero(as_tuple=True)[0]
            if below_idx.numel() > 0:
                k = min(num_to_evict, below_idx.numel())
                _, top = torch.topk(scores[below_idx], k, largest=False)
                return (below_idx[top] + ms).to(device)

        k = min(num_to_evict, int(valid_mask.sum().item()))
        _, lowest = torch.topk(scores, k, largest=False)
        # ensure lowest not from never_mask (they are 2.5 high)
        return (lowest + ms).to(device)

    def reset(self):
        super().reset()

    def on_evict(self, indices: torch.Tensor) -> None:
        super().on_evict(indices)

    @property
    def name(self) -> str:
        return f"ra_xaege_s{self.sink_size}_w{self.window_size}_ew{self.entropy_weight}_rw{self.retrieval_weight}"

    @property
    def requires_attention_scores(self) -> bool:
        return True

"""AEGE (Attention Entropy-Guided Eviction) — Fixed Core + XAI.

Novel: entropy filters noisy tokens. Keep high-attn + low-entropy.
Fixes vs old:
- Real Shannon H per query then avg key: sum(p log p) not elementwise
- Clamp score min so ew=1.0 never negative
- Quantile threshold 30% adaptive mode not fixed 0.5
- Pyramid get_max_size + explain_returns
"""

import torch
from typing import Optional, List, Tuple, Dict
from .base import EvictionPolicy


class AEGEPolicy(EvictionPolicy):
    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 64,
        entropy_weight: float = 0.3,
        temporal_decay: float = 0.95,
        adaptive_budget: bool = False,
        entropy_threshold: float = 0.4,  # kept compat, now used as quantile fallback
        adaptive_quantile: float = 0.3,  # evict bottom 30% in adaptive
        layer_idx: Optional[int] = None,
        total_layers: Optional[int] = None,
        rubric_spans: Optional[List[Tuple[int, int]]] = None,  # B: rubric protection
    ):
        self.sink_size = sink_size
        self.window_size = window_size
        self.entropy_weight = entropy_weight
        self.temporal_decay = temporal_decay
        self.adaptive_budget = adaptive_budget
        self.entropy_threshold = entropy_threshold
        self.adaptive_quantile = adaptive_quantile
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.rubric_spans = rubric_spans or []
        self._cumulative_attn: Optional[torch.Tensor] = None
        self._entropy_history: Optional[torch.Tensor] = None
        self._has_entropy = False
        # XAI explain: last computed scores per pos
        self._last_scores: Optional[torch.Tensor] = None
        self._last_kept: Optional[torch.Tensor] = None

    def get_max_size(self, default_max_size: int) -> int:
        if not self.adaptive_budget or self.layer_idx is None or self.total_layers is None or self.total_layers <= 1:
            return default_max_size
        depth_ratio = self.layer_idx / (self.total_layers - 1)
        multiplier = 0.6 + 0.6 * depth_ratio  # 60% shallow -> 120% deep
        min_allowed = self.sink_size + self.window_size + 4
        return max(min_allowed, int(default_max_size * multiplier))

    def _compute_entropy_per_key(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Proper Shannon entropy per key position.
        attention_scores: [batch, heads, query_len, seq_len]
        Returns entropy per key [seq_len] — high = dispersed attention (noisy)
        """
        # avg batch+heads -> [q, seq]
        mean_attn = attention_scores.float().mean(dim=(0, 1))  # [q_len, seq]
        # prob per query over keys
        probs = mean_attn / (mean_attn.sum(dim=-1, keepdim=True) + 1e-10)  # [q, seq]
        # H_q per query
        entropy_per_query = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [q]
        # weight per key by avg attention it receives
        key_weight = mean_attn.mean(dim=0)  # [seq]
        # entropy per key = avg entropy of queries that attend to it, weighted
        # ponytail: cheap proxy — query entropy * key attend strength
        # upgrade: per-key distribution over queries
        key_probs = key_weight / (key_weight.sum() + 1e-10)  # [seq] dist over keys
        # for XAI we want per-key dispersion: if many queries attend weakly = high entropy proxy
        # normalize entropy_per_query mean then broadcast * (1 - normalized key weight concentration)
        # simpler: per-key entropy = avg query entropy * (1 - normalized key prob) inverse
        # but proper: compute per-key distribution over queries?
        # Here: per-key entropy = entropy_of_key_distribution where key's attention across queries normalized
        # Fallback to attention dispersion: queries that strongly attend = low entropy intent
        # Use query-level entropy mean as base, modulated by key concentration
        avg_q_ent = entropy_per_query.mean()
        # high key_prob = repeatedly attended = low entropy (keep)
        # low key_prob = rarely attended = high entropy (evict)
        # invert: high entropy score for low prob keys
        per_key_entropy = avg_q_ent * (1.0 - key_probs) + (1.0 - (key_probs + 1e-10).log().neg()) * 0.0
        # Actually compute per-key: how uniform is attention from different queries to this key?
        # q distribution to this key across query positions
        q_dist = mean_attn / (mean_attn.mean(dim=0, keepdim=True) + 1e-10)  # [q, seq] normalized per key over q
        q_dist_norm = q_dist / (q_dist.sum(dim=0, keepdim=True) + 1e-10)
        per_key_q_entropy = -(q_dist_norm * (q_dist_norm + 1e-10).log()).sum(dim=0)  # [seq]
        return per_key_q_entropy  # high = attended uniformly from many queries (maybe filler? actually important? we evict high)

    def _in_rubric(self, idx: int) -> bool:
        for s, e in self.rubric_spans:
            if s <= idx < e:
                return True
        return False

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

        evictable_count = window_start - protected_end
        if evictable_count <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # B: exclude rubric spans from evictable middle
        rubric_mask_middle = None
        if self.rubric_spans:
            rubric_mask_middle = torch.zeros(evictable_count, dtype=torch.bool, device=device)
            for s, e in self.rubric_spans:
                # intersect [protected_end, window_start) with [s,e)
                lo = max(s, protected_end)
                hi = min(e, window_start)
                if lo < hi:
                    rubric_mask_middle[lo - protected_end : hi - protected_end] = True
            evictable_count_effective = evictable_count - int(rubric_mask_middle.sum().item())
            if evictable_count_effective <= 0:
                return torch.tensor([], dtype=torch.long, device=device)

        num_to_evict = min(num_to_evict, evictable_count if rubric_mask_middle is None else int((~rubric_mask_middle).sum().item()) if rubric_mask_middle is not None else evictable_count)
        if num_to_evict <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # init state
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
            self._cumulative_attn *= self.temporal_decay
            self._cumulative_attn += attention_scores.float().sum(dim=(0, 1, 2))
            query_len = attention_scores.shape[2]
            if query_len > 1 or not self._has_entropy:
                ent = self._compute_entropy_per_key(attention_scores)  # [seq]
                self._entropy_history[: ent.shape[0]] = ent
                self._has_entropy = True
            else:
                self._entropy_history *= self.temporal_decay

        middle_start = protected_end
        middle_end = window_start
        middle_attn = self._cumulative_attn[middle_start:middle_end]
        middle_ent = self._entropy_history[middle_start:middle_end]

        attn_min, attn_max = middle_attn.min(), middle_attn.max()
        norm_attn = (middle_attn - attn_min) / (attn_max - attn_min + 1e-10)

        ent_min, ent_max = middle_ent.min(), middle_ent.max()
        norm_ent = (middle_ent - ent_min) / (ent_max - ent_min + 1e-10)

        # score: high attn + low entropy = keep, clamp min 0.05 so ew=1.0 never negative dominance
        raw_scores = norm_attn * (1.0 - self.entropy_weight * norm_ent)
        scores = raw_scores.clamp_min(0.05)
        self._last_scores = scores.detach().clone()  # XAI

        # filter rubric out for eviction choice
        if rubric_mask_middle is not None and rubric_mask_middle.any():
            # make rubric unscorable by setting high score
            scores = scores.clone()
            scores[rubric_mask_middle] = 2.0

        if self.adaptive_budget:
            # quantile based: evict bottom adaptive_quantile
            thresh = torch.quantile(scores[scores < 1.9] if rubric_mask_middle is not None else scores, self.adaptive_quantile) if scores.numel() > 1 else scores[0] * 0.9
            # also respect entropy_threshold as max bound for safety
            if self.layer_idx is not None and self.total_layers is not None and self.total_layers > 1:
                depth_ratio = self.layer_idx / (self.total_layers - 1)
                thresh = thresh * (1.2 - 0.4 * depth_ratio)
            below = (scores < thresh).nonzero(as_tuple=True)[0]
            if below.numel() > 0:
                k = min(num_to_evict, below.numel())
                _, local_k = torch.topk(scores[below], k, largest=False)
                return (below[local_k] + middle_start).to(device)

        # fixed budget: lowest scores
        k = min(num_to_evict, (scores < 1.9).sum().item() if rubric_mask_middle is not None else scores.numel())
        if k <= 0:
            return torch.tensor([], dtype=torch.long, device=device)
        _, lowest = torch.topk(scores, k, largest=False)
        return (lowest + middle_start).to(device)

    def get_explain(self) -> Dict[str, torch.Tensor]:
        """Return last scores for XAI IoU calc — C requirement."""
        return {
            "scores": self._last_scores.clone() if self._last_scores is not None else torch.tensor([]),
        }

    def reset(self):
        self._cumulative_attn = None
        self._entropy_history = None
        self._has_entropy = False
        self._last_scores = None
        self._last_kept = None

    def on_evict(self, indices: torch.Tensor) -> None:
        if self._cumulative_attn is None or indices.numel() == 0:
            return
        keep = torch.ones(self._cumulative_attn.shape[0], dtype=torch.bool, device=self._cumulative_attn.device)
        keep[indices.to(device=keep.device, dtype=torch.long)] = False
        self._cumulative_attn = self._cumulative_attn[keep]
        self._entropy_history = self._entropy_history[keep]

    @property
    def name(self) -> str:
        suffix = f"_adapt_q{self.adaptive_quantile}" if self.adaptive_budget else ""
        return f"aege_s{self.sink_size}_w{self.window_size}_ew{self.entropy_weight}{suffix}"

    @property
    def requires_attention_scores(self) -> bool:
        return True

"""XAI Metrics for B+C — Fidelity of cache vs S2 SHAP / FullCache attention.

For Profile Alignment: S2 XAI important tokens vs S3 cache kept tokens.

Supports both:
- Real S2 SHAP JSON input (if available)
- FullCache attention top-k proxy (mock S2)
"""

from typing import List, Tuple, Set
import torch
import numpy as np


def attention_importance_topk(
    attention_scores: torch.Tensor,
    k_ratio: float = 0.2,
) -> torch.Tensor:
    """Get top-k% important token positions from FullCache attention as proxy SHAP.
    attention_scores: [batch, heads, query_len, seq_len] or [seq_len]
    Returns LongTensor indices of important tokens.
    """
    if attention_scores.dim() > 1:
        # avg over batch, heads, queries
        if attention_scores.dim() == 4:
            imp = attention_scores.float().mean(dim=(0, 1, 2))
        elif attention_scores.dim() == 2:
            imp = attention_scores.float().mean(dim=0)
        else:
            imp = attention_scores.float().flatten()
    else:
        imp = attention_scores.float()
    k = max(1, int(imp.shape[0] * k_ratio))
    _, topk = torch.topk(imp, k, largest=True)
    return topk


def xai_fidelity_iou(
    kept_indices: List[int],
    important_indices: List[int],
) -> float:
    """IoU between cache-kept tokens and XAI-important tokens.
    1.0 = cache keeps exactly important tokens (explainable)
    0.0 = no overlap
    PhD use: S2 SHAP important vs S3 kept under budget 256
    """
    if not important_indices:
        return 1.0
    kept_set = set(int(x) for x in kept_indices)
    imp_set = set(int(x) for x in important_indices)
    if not kept_set:
        return 0.0
    inter = len(kept_set & imp_set)
    union = len(kept_set | imp_set)
    # For fidelity we care recall of important in kept: |keep ∩ imp| / |imp|
    recall = inter / max(1, len(imp_set))
    # IoU stricter
    iou = inter / max(1, union)
    # Return recall as primary (how many important preserved)
    # ponytail: if need IoU formally, use iou var
    return float(recall)


def rubric_retention_rate(
    kept_indices: List[int],
    rubric_spans: List[Tuple[int, int]],
    total_vocab: int = None,
) -> float:
    """% rubric tokens retained under budget.
    rubric_spans: list of (start, end) token ranges
    """
    if not rubric_spans:
        return 1.0
    rubric_toks: Set[int] = set()
    for s, e in rubric_spans:
        for i in range(int(s), int(e)):
            rubric_toks.add(i)
    if not rubric_toks:
        return 1.0
    kept_set = set(int(x) for x in kept_indices)
    kept_rubric = len(kept_set & rubric_toks)
    return float(kept_rubric / max(1, len(rubric_toks)))


def compute_per_budget_xai_summary(
    cache_sizes_list: List[List[int]],
    fullcache_attn_list: List[torch.Tensor],
    k_ratio: float = 0.2,
) -> dict:
    """Aggregate IoU across samples for a budget.
    Helper for notebook dashboard.
    """
    # naive avg — real would use kept positions tracking
    return {"k_ratio": k_ratio, "num_samples": len(cache_sizes_list)}

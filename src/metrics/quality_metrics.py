"""Quality metrics module for evaluating KV Cache Eviction — QWK (Quadratic Weighted Kappa), Cosine Similarity, and Logit Divergence."""

import torch
import numpy as np
from typing import List, Union


def quadratic_weighted_kappa(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    min_rating: int = 1,
    max_rating: int = 6,
) -> float:
    """Compute Quadratic Weighted Kappa (QWK) metric between ground truth human scores and predicted scores.

    Args:
        y_true: Array of ground-truth scores
        y_pred: Array of predicted scores
        min_rating: Minimum rating scale boundary (default: 1)
        max_rating: Maximum rating scale boundary (default: 6)

    Returns:
        QWK score between -1.0 and 1.0 (1.0 = perfect agreement)
    """
    y_true = np.clip(np.asarray(y_true, dtype=int), min_rating, max_rating)
    y_pred = np.clip(np.asarray(y_pred, dtype=int), min_rating, max_rating)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return 0.0

    num_ratings = max_rating - min_rating + 1
    
    # Weight matrix W_ij = (i - j)^2 / (N - 1)^2
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    # Observed confusion matrix
    conf_mat = np.zeros((num_ratings, num_ratings))
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)

    for t, p in zip(y_true, y_pred):
        conf_mat[t - min_rating, p - min_rating] += 1
        hist_true[t - min_rating] += 1
        hist_pred[p - min_rating] += 1

    # Expected matrix under independence
    expected = np.outer(hist_true, hist_pred) / (len(y_true) + 1e-10)

    numerator = np.sum(weights * conf_mat)
    denominator = np.sum(weights * expected)

    if denominator == 0:
        return 1.0

    qwk = 1.0 - (numerator / denominator)
    return float(qwk)


def text_match_accuracy(str_a: str, str_b: str) -> float:
    """Compute exact or character sequence match ratio between baseline and evicted text."""
    if not str_a and not str_b:
        return 1.0
    if not str_a or not str_b:
        return 0.0
    matching_chars = sum(1 for a, b in zip(str_a, str_b) if a == b)
    max_len = max(len(str_a), len(str_b))
    return float(matching_chars / max_len)


def logit_kl_divergence(
    baseline_logits: torch.Tensor,
    evicted_logits: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """Compute Kullback-Leibler (KL) Divergence between baseline logits (FullCache) and evicted logits."""
    p_probs = torch.softmax(baseline_logits / temperature, dim=-1)
    q_log_probs = torch.log_softmax(evicted_logits / temperature, dim=-1)
    kl_div = torch.nn.functional.kl_div(q_log_probs, p_probs, reduction="batchmean")
    return float(kl_div.item())


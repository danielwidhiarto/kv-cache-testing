"""Fast tests for the real past_key_values generation path."""

from types import SimpleNamespace

import torch

from src.cache.manager import CacheManager
from src.policies.aege import AEGEPolicy


class ToyCausalLM(torch.nn.Module):
    """Small deterministic model exposing the HF cache contract."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(eos_token_id=99)
        self.num_layers = 2
        self.num_heads = 2
        self.head_dim = 4
        self.vocab_size = 8

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_attentions=False,
        use_cache=True,
        past_key_values=None,
        position_ids=None,
        **kwargs,
    ):
        batch_size, query_length = input_ids.shape
        old = [None] * self.num_layers if past_key_values is None else past_key_values
        new_past = []
        attentions = []

        for layer_idx in range(self.num_layers):
            key = input_ids.float().view(batch_size, 1, query_length, 1)
            key = key.expand(batch_size, self.num_heads, query_length, self.head_dim) + layer_idx
            value = key + 0.5
            if old[layer_idx] is not None:
                key = torch.cat([old[layer_idx][0], key], dim=2)
                value = torch.cat([old[layer_idx][1], value], dim=2)
            new_past.append((key, value))

            if output_attentions:
                score = torch.ones(
                    batch_size,
                    self.num_heads,
                    query_length,
                    key.shape[2],
                )
                attentions.append(torch.softmax(score, dim=-1))

        logits = torch.zeros(batch_size, query_length, self.vocab_size)
        logits[..., 1] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=tuple(new_past),
            attentions=tuple(attentions) if output_attentions else None,
        )


def test_full_cache_uses_native_past_without_evictions():
    manager = CacheManager(ToyCausalLM(), policy=None, max_cache_size=4, device=torch.device("cpu"))
    output = manager.generate_with_cache(torch.tensor([[1, 2, 3, 4, 5, 6]]), max_new_tokens=4)
    metrics = manager.get_metrics()

    assert output.shape == (1, 10)
    assert metrics["prefill_count"] == 1
    assert metrics["total_evictions"] == 0
    assert metrics["decode_model_calls"] == 3


def test_aege_respects_budget_and_evicts_per_layer():
    manager = CacheManager(
        ToyCausalLM(),
        policy=AEGEPolicy(sink_size=1, window_size=2),
        max_cache_size=4,
        device=torch.device("cpu"),
    )
    output = manager.generate_with_cache(torch.tensor([[1, 2, 3, 4, 5, 6]]), max_new_tokens=4)
    metrics = manager.get_metrics()

    assert output.shape == (1, 10)
    assert all(size <= 4 for size in metrics["cache_sizes"])
    assert metrics["total_evictions"] > 0
    assert metrics["peak_cache_tokens"] == 8

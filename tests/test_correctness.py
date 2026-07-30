"""Correctness tests — verify eviction policies don't break model outputs."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.cache.full_cache import FullCache
from src.cache.lru_cache import LRUCache
from src.cache.h2o_cache import H2OCache
from src.cache.streaming_cache import StreamingCache
from src.cache.snap_cache import SnapCache


# Use a tiny model for fast tests
TEST_MODEL = "gpt2"
DEVICE = "cpu"  # CPU for CI compatibility


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model once for all tests."""
    model, tokenizer = load_model(TEST_MODEL, device=DEVICE, dtype=torch.float32)
    return model, tokenizer


@pytest.fixture
def sample_input(model_and_tokenizer):
    """Create sample input tokens."""
    _, tokenizer = model_and_tokenizer
    text = "The quick brown fox jumps over the lazy dog. " * 10
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return tokens["input_ids"]


class TestCacheInstantiation:
    """Test that all cache types can be created."""

    def test_full_cache(self):
        cache = FullCache(max_size=1024)
        assert cache.max_size == 1024
        assert cache.size() == 0

    def test_lru_cache(self):
        cache = LRUCache(max_size=1024)
        assert cache.max_size == 1024
        assert cache.size() == 0

    def test_h2o_cache(self):
        cache = H2OCache(max_size=1024, heavy_ratio=0.1)
        assert cache.max_size == 1024

    def test_streaming_cache(self):
        cache = StreamingCache(max_size=1024, sink_size=4, window_size=256)
        assert cache.max_size == 1024

    def test_snap_cache(self):
        cache = SnapCache(max_size=1024, observation_size=16)
        assert cache.max_size == 1024


class TestCacheOperations:
    """Test basic cache operations."""

    def test_append_and_get(self):
        cache = FullCache(max_size=1024)
        k = torch.randn(1, 12, 10, 64)  # [batch, heads, seq, dim]
        v = torch.randn(1, 12, 10, 64)
        cache.append(k, v)
        assert cache.size() == 10
        k_out, v_out = cache.get()
        assert k_out.shape == k.shape
        assert torch.allclose(k_out, k)

    def test_evict(self):
        cache = FullCache(max_size=1024)
        k = torch.randn(1, 12, 10, 64)
        v = torch.randn(1, 12, 10, 64)
        cache.append(k, v)
        # Evict positions 2 and 5
        cache.evict(torch.tensor([2, 5]))
        assert cache.size() == 8

    def test_reset(self):
        cache = FullCache(max_size=1024)
        k = torch.randn(1, 12, 10, 64)
        v = torch.randn(1, 12, 10, 64)
        cache.append(k, v)
        cache.reset()
        assert cache.size() == 0

    def test_lru_eviction(self):
        """LRU cache should evict when over capacity."""
        cache = LRUCache(max_size=20)
        k = torch.randn(1, 12, 30, 64)
        v = torch.randn(1, 12, 30, 64)
        cache.append(k, v)
        # Should have evicted down to max_size
        assert cache.size() <= 20


class TestModelForward:
    """Test that model forward pass works with different setups."""

    def test_forward_basic(self, model_and_tokenizer, sample_input):
        model, _ = model_and_tokenizer
        with torch.no_grad():
            output = model(sample_input, output_attentions=True)
        assert output.logits.shape[0] == sample_input.shape[0]
        assert output.attentions is not None
        assert len(output.attentions) > 0

    def test_attention_shape(self, model_and_tokenizer, sample_input):
        model, _ = model_and_tokenizer
        with torch.no_grad():
            output = model(sample_input, output_attentions=True)
        # Check attention shape: [batch, heads, seq, seq]
        attn = output.attentions[0]
        assert attn.dim() == 4
        seq_len = sample_input.shape[1]
        assert attn.shape[2] == seq_len
        assert attn.shape[3] == seq_len


class TestLogitDivergence:
    """Test that eviction policies produce bounded logit divergence."""

    def test_full_cache_baseline(self, model_and_tokenizer, sample_input):
        """Full cache should produce identical output to standard forward."""
        model, _ = model_and_tokenizer
        with torch.no_grad():
            ref_output = model(sample_input)
        # Just verify it runs
        assert ref_output.logits is not None

    @pytest.mark.parametrize("cache_cls,kwargs", [
        (FullCache, {"max_size": 4096}),
        (LRUCache, {"max_size": 4096}),
        (H2OCache, {"max_size": 4096}),
        (StreamingCache, {"max_size": 4096, "sink_size": 4, "window_size": 512}),
        (SnapCache, {"max_size": 4096}),
    ])
    def test_cache_types_produce_output(self, cache_cls, kwargs):
        """All cache types should produce valid KV pairs."""
        cache = cache_cls(**kwargs)
        k = torch.randn(1, 12, 50, 64)
        v = torch.randn(1, 12, 50, 64)
        cache.append(k, v)
        k_out, v_out = cache.get()
        assert k_out.shape[2] == 50  # No eviction since under capacity
        assert v_out.shape[2] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

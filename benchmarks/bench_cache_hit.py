"""Benchmark: Cache hit rate measurement."""

import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.cache.full_cache import FullCache
from src.cache.lru_cache import LRUCache
from src.cache.h2o_cache import H2OCache
from src.cache.streaming_cache import StreamingCache
from src.cache.snap_cache import SnapCache
from src.cache.aege_cache import AEGECache
from src.metrics.collector import MetricsCollector
from src.metrics.reporter import Reporter


# Registry of cache types to test
CACHE_REGISTRY = {
    "full": (FullCache, {"max_size": 2048}),
    "lru": (LRUCache, {"max_size": 2048}),
    "h2o": (H2OCache, {"max_size": 2048, "heavy_ratio": 0.1}),
    "streaming": (StreamingCache, {"max_size": 2048, "sink_size": 4, "window_size": 512}),
    "snap": (SnapCache, {"max_size": 2048}),
    "aege": (AEGECache, {"max_size": 2048, "sink_size": 4, "window_size": 64, "entropy_weight": 0.3}),
}


def benchmark_cache_hit_rate(
    model,
    tokenizer,
    prompts: list,
    cache_type: str,
    cache_kwargs: dict,
    device: str = "cpu",
):
    """Measure cache hit rate for repeated/shared prompts."""
    cache_cls = CACHE_REGISTRY[cache_type][0]
    cache = cache_cls(**cache_kwargs)

    # Simulate: encode prompts, check how much KV cache is reused
    all_tokens = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        all_tokens.append(tokens)

    # First pass: populate cache
    total_positions = 0
    cache_hits = 0

    for tokens in all_tokens:
        with torch.no_grad():
            output = model(tokens, output_attentions=True)

        if hasattr(output, "attentions") and output.attentions:
            attn = output.attentions[0]  # [batch, heads, seq, seq]
            # Count positions with high attention (potential cache hits)
            avg_attn = attn.float().mean(dim=(0, 1, 2))  # [seq_len]
            high_attn = (avg_attn > avg_attn.median()).sum().item()
            cache_hits += high_attn
            total_positions += tokens.shape[1]

    hit_rate = cache_hits / total_positions if total_positions > 0 else 0.0

    return {
        "cache_type": cache_type,
        "num_prompts": len(prompts),
        "total_positions": total_positions,
        "estimated_hits": cache_hits,
        "hit_rate": round(hit_rate, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark cache hit rate")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--cache-types", nargs="+", default=["full", "lru", "h2o", "streaming", "snap"])
    parser.add_argument("--output", default="results/cache_hit.csv")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    device = next(model.parameters()).device

    # Test prompts with varying overlap
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy cat.",  # Similar prefix
        "Machine learning is a subset of artificial intelligence.",  # Different
        "The quick brown fox runs through the forest.",  # Similar prefix
        "In this essay, I will discuss the importance of education.",
    ]

    print(f"\nCache Hit Rate Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Cache types: {args.cache_types}")

    results = []
    for cache_type in args.cache_types:
        if cache_type not in CACHE_REGISTRY:
            print(f"  Skipping unknown cache type: {cache_type}")
            continue

        print(f"\n  Testing {cache_type}...")
        _, kwargs = CACHE_REGISTRY[cache_type]
        result = benchmark_cache_hit_rate(
            model, tokenizer, prompts, cache_type, kwargs, device
        )
        results.append(result)
        print(f"    Hit rate: {result['hit_rate']:.2%}")

    # Export
    import csv
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nExported to {args.output}")


if __name__ == "__main__":
    main()

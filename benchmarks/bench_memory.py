"""Benchmark: Memory profiling for different cache sizes."""

import sys
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model, estimate_memory
from src.metrics.collector import MetricsCollector
from src.metrics.reporter import Reporter


def benchmark_memory(
    model,
    tokenizer,
    input_length: int,
    cache_sizes: list,
    warmup: int = 2,
    measure: int = 3,
    device: str = "cpu",
):
    """Benchmark memory usage with different cache configurations."""
    text = "The quick brown fox jumps over the lazy dog. " * (input_length // 10 + 1)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=input_length,
    )
    input_ids = tokens["input_ids"].to(device)

    results = []

    # Baseline: model memory only
    model_info = estimate_memory(model)
    print(f"\nModel memory: {model_info['param_size_mb']:.1f} MB ({model_info['param_count']:,} params)")

    for cache_size in cache_sizes:
        collector = MetricsCollector()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model(input_ids, output_attentions=True)

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measure
        for i in range(measure):
            collector.start_step()
            with torch.no_grad():
                output = model(input_ids, output_attentions=True)
            collector.end_step(cache_size=cache_size, tokens_generated=1)

        metrics = collector.finalize(
            policy_name=f"cache_{cache_size}",
            model_name=model.config.name_or_path or "unknown",
            max_cache_size=cache_size,
            input_length=input_length,
        )
        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory usage")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--cache-sizes", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument("--output", default="results/memory.csv", help="Output CSV path")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    device = next(model.parameters()).device

    results = benchmark_memory(
        model, tokenizer, args.input_length, args.cache_sizes,
        args.warmup, args.measure, device
    )

    Reporter.print_comparison(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Reporter.to_csv(results, args.output)


if __name__ == "__main__":
    main()

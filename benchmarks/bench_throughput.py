"""Benchmark: Throughput (tokens/sec) under different conditions."""

import sys
import time
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.metrics.collector import MetricsCollector
from src.metrics.reporter import Reporter


def benchmark_throughput(
    model,
    tokenizer,
    input_length: int,
    max_new_tokens: int = 50,
    batch_sizes: list = [1],
    warmup: int = 1,
    measure: int = 3,
    device: str = "cpu",
):
    """Benchmark generation throughput."""
    results = []

    for batch_size in batch_sizes:
        text = "The quick brown fox jumps over the lazy dog. " * (input_length // 10 + 1)
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=input_length,
            padding="max_length",
        )
        input_ids = tokens["input_ids"].repeat(batch_size, 1).to(device)
        attention_mask = tokens["attention_mask"].repeat(batch_size, 1).to(device)

        collector = MetricsCollector()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                )

        # Measure generation
        for i in range(measure):
            collector.start_step()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            new_tokens = output_ids.shape[1] - input_ids.shape[1]
            collector.end_step(
                cache_size=output_ids.shape[1],
                tokens_generated=new_tokens * batch_size,
            )

        metrics = collector.finalize(
            policy_name=f"batch_{batch_size}",
            model_name=model.config.name_or_path or "unknown",
            max_cache_size=input_length + max_new_tokens,
            input_length=input_length,
        )
        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark throughput")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument("--output", default="results/throughput.csv")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    device = next(model.parameters()).device

    results = benchmark_throughput(
        model, tokenizer, args.input_length, args.max_new_tokens,
        args.batch_sizes, args.warmup, args.measure, device
    )

    Reporter.print_comparison(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Reporter.to_csv(results, args.output)


if __name__ == "__main__":
    main()

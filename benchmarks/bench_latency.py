"""Benchmark: Latency (TTFT + ITL) across sequence lengths."""

import sys
import time
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.metrics.collector import MetricsCollector
from src.metrics.reporter import Reporter


def benchmark_forward_latency(
    model,
    tokenizer,
    input_length: int,
    batch_size: int = 1,
    warmup: int = 2,
    measure: int = 5,
    device: str = "cpu",
):
    """Benchmark forward pass latency for a given input length."""
    # Create input
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
            model(input_ids, attention_mask=attention_mask, output_attentions=True)

    # Measure
    for i in range(measure):
        collector.start_step()
        with torch.no_grad():
            output = model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        collector.end_step(cache_size=input_length, tokens_generated=1)

    return collector.finalize(
        policy_name="standard_forward",
        model_name=model.config.name_or_path or "unknown",
        max_cache_size=input_length,
        input_length=input_length,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark forward latency")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=5)
    parser.add_argument("--output", default="results/latency.csv", help="Output CSV path")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    device = next(model.parameters()).device

    all_metrics = []
    for length in args.input_lengths:
        print(f"\nBenchmarking input_length={length}...")
        metrics = benchmark_forward_latency(
            model, tokenizer, length, args.batch_size, args.warmup, args.measure, device
        )
        Reporter.print_summary(metrics)
        all_metrics.append(metrics)

    Reporter.print_comparison(all_metrics)

    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Reporter.to_csv(all_metrics, args.output)
    Reporter.to_json(all_metrics, args.output.replace(".csv", ".json"))


if __name__ == "__main__":
    main()

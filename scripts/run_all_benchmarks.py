"""Run all benchmarks and export results."""

import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from benchmarks.bench_latency import benchmark_forward_latency
from benchmarks.bench_memory import benchmark_memory
from benchmarks.bench_throughput import benchmark_throughput
from src.metrics.reporter import Reporter


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run all KV cache benchmarks")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer iterations)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model or config["model"]["name"]
    device = args.device or config["model"].get("device")

    # Adjust for quick mode
    warmup = 1 if args.quick else config["benchmark"]["warmup_steps"]
    measure = 2 if args.quick else config["benchmark"]["measure_steps"]

    print("=" * 60)
    print("  KV Cache Testing Framework — Full Benchmark Suite")
    print("=" * 60)
    print(f"  Model: {model_name}")
    print(f"  Device: {device or 'auto'}")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(model_name, device=device)
    device = next(model.parameters()).device

    # === 1. Latency Benchmark ===
    print("\n" + "=" * 60)
    print("  [1/3] Latency Benchmark")
    print("=" * 60)
    latency_metrics = []
    for length in config["benchmark"]["input_lengths"]:
        print(f"\n  Input length: {length}")
        metrics = benchmark_forward_latency(
            model, tokenizer, length,
            batch_size=config["benchmark"]["batch_size"],
            warmup=warmup, measure=measure, device=device,
        )
        Reporter.print_summary(metrics)
        latency_metrics.append(metrics)

    Reporter.print_comparison(latency_metrics)

    # === 2. Memory Benchmark ===
    print("\n" + "=" * 60)
    print("  [2/3] Memory Benchmark")
    print("=" * 60)
    memory_metrics = benchmark_memory(
        model, tokenizer,
        input_length=config["benchmark"]["input_lengths"][-1],
        cache_sizes=config["cache"]["max_sizes"],
        warmup=warmup, measure=measure, device=device,
    )
    Reporter.print_comparison(memory_metrics)

    # === 3. Throughput Benchmark ===
    print("\n" + "=" * 60)
    print("  [3/3] Throughput Benchmark")
    print("=" * 60)
    throughput_metrics = benchmark_throughput(
        model, tokenizer,
        input_length=config["benchmark"]["input_lengths"][0],
        max_new_tokens=config["benchmark"]["max_new_tokens"],
        batch_sizes=[1, 2, 4],
        warmup=warmup, measure=measure, device=device,
    )
    Reporter.print_comparison(throughput_metrics)

    # === Export All Results ===
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model_name.replace("/", "_")

    Reporter.to_csv(latency_metrics, str(output_dir / f"latency_{model_tag}_{timestamp}.csv"))
    Reporter.to_csv(memory_metrics, str(output_dir / f"memory_{model_tag}_{timestamp}.csv"))
    Reporter.to_csv(throughput_metrics, str(output_dir / f"throughput_{model_tag}_{timestamp}.csv"))

    print("\n" + "=" * 60)
    print("  All benchmarks complete!")
    print(f"  Results saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Report and export metrics."""

import json
import csv
from typing import List, Dict, Any
from pathlib import Path
from .collector import ExperimentMetrics


class Reporter:
    """Formats and exports experiment metrics."""

    @staticmethod
    def print_summary(metrics: ExperimentMetrics):
        """Print a formatted summary table."""
        d = metrics.to_dict()
        print("\n" + "=" * 60)
        print(f"  KV Cache Experiment Results")
        print("=" * 60)
        print(f"  Policy:          {d['policy']}")
        print(f"  Model:           {d['model']}")
        print(f"  Max Cache Size:  {d['max_cache_size']} tokens")
        print(f"  Input Length:    {d['input_length']} tokens")
        print("-" * 60)
        print(f"  Avg Latency:     {d['avg_latency_ms']:.2f} ms/step")
        print(f"  Total Latency:   {d['total_latency_ms']:.2f} ms")
        print(f"  Throughput:      {d['tokens_per_second']:.2f} tokens/sec")
        print(f"  Peak Memory:     {d['peak_memory_mb']:.2f} MB")
        print(f"  Total Evictions: {d['total_evictions']}")
        print(f"  Steps:           {d['num_steps']}")
        print("=" * 60 + "\n")

    @staticmethod
    def print_comparison(all_metrics: List[ExperimentMetrics]):
        """Print a comparison table of multiple experiments."""
        if not all_metrics:
            print("No metrics to compare.")
            return

        print("\n" + "=" * 90)
        print(f"  KV Cache Policy Comparison")
        print("=" * 90)
        print(f"  {'Policy':<25} {'Latency(ms)':<12} {'Tok/s':<10} {'Memory(MB)':<12} {'Evictions':<10}")
        print("-" * 90)

        for m in all_metrics:
            d = m.to_dict()
            print(f"  {d['policy']:<25} {d['avg_latency_ms']:<12.2f} {d['tokens_per_second']:<10.2f} "
                  f"{d['peak_memory_mb']:<12.2f} {d['total_evictions']:<10}")

        print("=" * 90 + "\n")

    @staticmethod
    def to_csv(all_metrics: List[ExperimentMetrics], path: str):
        """Export metrics to CSV."""
        if not all_metrics:
            return

        fieldnames = list(all_metrics[0].to_dict().keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m.to_dict())

        print(f"Exported {len(all_metrics)} experiments to {path}")

    @staticmethod
    def to_json(all_metrics: List[ExperimentMetrics], path: str):
        """Export metrics to JSON."""
        data = [m.to_dict() for m in all_metrics]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(all_metrics)} experiments to {path}")

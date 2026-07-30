"""Google Colab runner for the AES KV-cache eviction benchmark.

The underlying benchmark performs real prefill/decode using past_key_values and
applies eviction per transformer layer.
"""

import argparse
import os
import sys

# Ensure current working directory is in sys.path
sys.path.append(os.getcwd())

from benchmarks.bench_aes import run_aes_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AES Benchmark on Google Colab GPU (Scale C)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Instruct model name")
    parser.add_argument("--num-samples", type=int, default=14, help="Number of samples (e.g., 14 = 2 samples per topic)")
    parser.add_argument("--max-cache-size", type=int, default=768, help="Max KV cache budget (e.g., 768 for ~50% eviction)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and policy-order seed")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unmeasured warm-up runs per sample")
    args = parser.parse_args()

    print("==================================================")
    print("🚀 Launching Google Colab AES Experiment — PhD AEGE Benchmark")
    print(f"• Target Model    : {args.model}")
    print(f"• Total Samples   : {args.num_samples}")
    print(f"• Cache Budget    : {args.max_cache_size} tokens")
    print(f"• Max New Tokens  : {args.max_new_tokens}")
    print("==================================================")

    # Execute benchmark
    sys.argv = [
        "bench_aes.py",
        "--model", args.model,
        "--num-samples", str(args.num_samples),
        "--max-cache-size", str(args.max_cache_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
        "--warmup-runs", str(args.warmup_runs),
    ]
    run_aes_benchmark()

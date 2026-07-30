"""
Google Colab Execution Script for Long-Context AES Benchmark
Scale C Configuration: prompt 4096 tokens + cache budget 256 tokens (~87% eviction rate)
Supports Instruct LLMs (Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, Llama-3-8B-Instruct)
Computes true Quadratic Weighted Kappa (QWK) against Human Ratings
"""

import argparse
import os
import sys

# Ensure current working directory is in sys.path
sys.path.append(os.getcwd())

from benchmarks.bench_aes import run_aes_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AES Benchmark on Google Colab GPU (Scale C)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Instruct model name")
    parser.add_argument("--num-samples", type=int, default=14, help="Number of samples (e.g., 14 = 2 samples per topic)")
    parser.add_argument("--max-cache-size", type=int, default=256, help="Max KV cache budget (Scale C: 256 for ~87% eviction)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    args = parser.parse_args()

    print("==================================================")
    print("🚀 Launching Google Colab AES Experiment — SCALE C")
    print(f"• Target Model    : {args.model}")
    print(f"• Total Samples   : {args.num_samples}")
    print(f"• Max Token Length: 4096 tokens")
    print(f"• Cache Budget    : {args.max_cache_size} tokens (~{round((1 - args.max_cache_size/2048)*100)}% est. eviction)")
    print(f"• Max New Tokens  : {args.max_new_tokens}")
    print("• Expected Speedup: ~5-7x vs FullCache baseline")
    print("==================================================")

    # Execute benchmark
    sys.argv = [
        "bench_aes.py",
        "--model", args.model,
        "--num-samples", str(args.num_samples),
        "--max-cache-size", str(args.max_cache_size),
        "--max-new-tokens", str(args.max_new_tokens)
    ]
    run_aes_benchmark()


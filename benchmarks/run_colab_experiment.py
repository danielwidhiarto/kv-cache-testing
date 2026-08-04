"""Google Colab runner for AES KV-cache benchmark v1 and v2 RA-XAEGE A+C."""

import argparse
import os
import sys

sys.path.append(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AES Benchmark v1 or v2 RA-XAEGE on Colab")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-samples", type=int, default=14)
    parser.add_argument("--max-cache-size", type=int, default=768)
    parser.add_argument("--max-cache-sizes", type=int, nargs="+", default=None, help="Sweep for v2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--v2", action="store_true", help="Use RA-XAEGE v2 bench (A+C: retrieval+feedback)")
    parser.add_argument("--retrieval-k", type=int, default=3)
    args = parser.parse_args()

    if args.v2 or args.max_cache_sizes:
        from benchmarks.bench_aes_v2_raxaege import run_benchmark
        # map args to sys.argv style expected by v2 parser
        sys_argv = ["bench_aes_v2_raxaege.py", "--model", args.model, "--num-samples", str(args.num_samples), "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed), "--retrieval-k", str(args.retrieval_k)]
        if args.max_cache_sizes:
            sys_argv += ["--max-cache-sizes"] + [str(x) for x in args.max_cache_sizes]
        else:
            sys_argv += ["--max-cache-size", str(args.max_cache_size)]
        sys.argv = sys_argv
        run_benchmark()
    else:
        from benchmarks.bench_aes import run_aes_benchmark
        print("🚀 Launching AES v1 (fair window 64 fixed)")
        sys.argv = ["bench_aes.py", "--model", args.model, "--num-samples", str(args.num_samples), "--max-cache-size", str(args.max_cache_size), "--max-new-tokens", str(args.max_new_tokens), "--seed", str(args.seed), "--warmup-runs", str(args.warmup_runs)]
        run_aes_benchmark()

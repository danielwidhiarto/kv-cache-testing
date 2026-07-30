"""PhD Dissertation Multi-Budget Experiment Runner.

Runs a multi-tier cache budget sweep (e.g., 512, 768, 1280 tokens) with Qwen2.5-7B-Instruct
to generate complete comparative results for paper tables and charts.
"""

import os
import sys
import subprocess
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(description="PhD Dissertation Multi-Tier KV Cache Eviction Sweep")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model path/name")
    parser.add_argument("--num-samples", type=int, default=14, help="Number of AES samples")
    parser.add_argument("--cache-budgets", nargs="+", type=int, default=[512, 768, 1280], help="List of cache budget sizes to sweep")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--output-dir", type=str, default="results/phd_sweep", help="Directory for experiment outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("==================================================")
    print("🎓 PhD Dissertation AEGE Benchmark Sweep")
    print(f"• Model        : {args.model}")
    print(f"• Samples      : {args.num_samples}")
    print(f"• Cache Budgets: {args.cache_budgets}")
    print(f"• Output Dir   : {args.output_dir}")
    print("==================================================")

    for budget in args.cache_budgets:
        print(f"\n🚀 Running Sweep Stage for Cache Budget = {budget} tokens...")
        cmd = [
            sys.executable,
            "benchmarks/bench_aes.py",
            "--model", args.model,
            "--num-samples", str(args.num_samples),
            "--max-cache-size", str(budget),
            "--max-new-tokens", str(args.max_new_tokens),
            "--output-dir", os.path.join(args.output_dir, f"budget_{budget}"),
            "--seed", str(args.seed),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"❌ Sweep for budget {budget} failed with code {res.returncode}")
        else:
            print(f"✅ Completed budget = {budget} tokens!")

    print(f"\n🎉 All sweeps completed! Check output files in: {args.output_dir}")

if __name__ == "__main__":
    main()

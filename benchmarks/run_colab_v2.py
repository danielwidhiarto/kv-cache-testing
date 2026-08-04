"""Colab wrapper for RA-XAEGE A+C sweep — copy github friendly."""
import argparse, sys, os
sys.path.append(os.getcwd())

if __name__ == "__main__":
    # forward all args to v2 bench
    from benchmarks.bench_aes_v2_raxaege import run_benchmark
    run_benchmark()

# KV Cache Testing Framework

Framework untuk menguji dan membandingkan berbagai KV cache eviction policies pada model LLM.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Run all benchmarks (quick mode)
```bash
python scripts/run_all_benchmarks.py --quick
```

### Run specific benchmark
```bash
# Latency
python benchmarks/bench_latency.py --model gpt2 --input-lengths 128 256 512

# Memory
python benchmarks/bench_memory.py --model gpt2 --cache-sizes 256 512 1024

# Throughput
python benchmarks/bench_throughput.py --model gpt2 --batch-sizes 1 2 4

# Cache hit rate
python benchmarks/bench_cache_hit.py --model gpt2
```

### Run tests
```bash
pytest tests/ -v
```

### Run the AES eviction benchmark (Colab GPU)

The AES runner now performs one prefill, keeps the model's real
`past_key_values`, and applies the selected policy independently per layer.
Run the benchmark after installing the requirements and extracting the ASAP 2.0
CSV:

```bash
python benchmarks/bench_aes.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 100 \
  --max-cache-size 512 \
  --max-new-tokens 20 \
  --seed 42 \
  --output-dir results
```

The resulting `results/aes_benchmark.csv` includes `actual_evicted_tokens`,
`peak_cache_tokens`, measured `ttft_sec`/`itl_ms`, and `predicted_score`.
Compare `QWK(policy, human)` with `QWK(FullCache, human)` and report their
difference; do not treat exact output matching alone as an accuracy metric.

## Eviction Policies

| Policy | Description | Reference |
|--------|-------------|-----------|
| `FullCache` | No eviction (baseline) | — |
| `LRUCache` | Least Recently Used | Classic |
| `H2OCache` | Heavy-Hitter Oracle | Zhang et al., 2023 |
| `StreamingCache` | Attention Sink + Window | Xiao et al., ICLR 2024 |
| `SnapCache` | Per-head attention-based | Li et al., 2024 |

## Metrics Collected

- **Latency**: TTFT (Time-to-First-Token), ITL (Inter-Token Latency)
- **Memory**: Peak GPU/CPU memory usage
- **Throughput**: Tokens per second
- **Cache Hit Rate**: KV cache reuse effectiveness
- **Eviction Count**: Total evictions per experiment

## Project Structure

```
kv-cache-testing/
├── src/
│   ├── cache/          # Cache implementations
│   ├── policies/       # Eviction policies
│   ├── metrics/        # Metrics collection & reporting
│   └── utils/          # Model loading utilities
├── tests/              # Correctness tests
├── benchmarks/         # Performance benchmarks
├── config/             # Experiment configurations
├── scripts/            # Runner scripts
└── results/            # Output directory
```

## Configuration

Edit `config/default.yaml` to customize:
- Model selection
- Cache sizes to test
- Which policies to enable
- Benchmark parameters
- Output format

## Adding a New Policy

1. Create `src/policies/my_policy.py`:
```python
from .base import EvictionPolicy

class MyPolicy(EvictionPolicy):
    def select_evict(self, key_cache, value_cache, attention_scores=None, num_to_evict=1):
        # Return indices to evict
        ...

    @property
    def name(self):
        return "my_policy"
```

2. Create `src/cache/my_cache.py` that uses the policy
3. Add to `benchmarks/bench_cache_hit.py` registry

## PhD Research Context

This framework is designed for research on KV cache eviction policies,
specifically for the intersection with Automated Essay Scoring (AES).

Next steps:
- [ ] Add more eviction policies (PyramidKV, KIVI, KVP)
- [ ] Integrate AES dataset (ASAP 2.0)
- [ ] Add correctness evaluation (logit divergence)
- [ ] Add per-head analysis
- [ ] Add visualization (matplotlib plots)

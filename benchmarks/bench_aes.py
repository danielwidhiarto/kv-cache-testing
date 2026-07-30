"""Benchmark script for evaluating KV Cache Eviction Policies on AES (Automated Essay Scoring) prompts."""

import os
import sys
import time
import argparse
import random
import re
import pandas as pd
import torch
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.policies.aege import AEGEPolicy
from src.policies.h2o import H2OPolicy
from src.policies.streaming import StreamingPolicy
from src.policies.lru import LRUPolicy
from src.cache.manager import CacheManager
from src.utils.aes_loader import AESDatasetLoader
from src.metrics.quality_metrics import quadratic_weighted_kappa


def parse_args():
    parser = argparse.ArgumentParser(description="AES Long-Context KV Cache Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")

    parser.add_argument("--num-samples", type=int, default=14, help="Number of AES essay samples to test")
    parser.add_argument("--max-cache-size", type=int, default=768, help="Max KV cache size budget")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate per sample")
    parser.add_argument("--dataset-path", type=str, default="dataset/ASAP2_train_sourcetexts.csv", help="Path to ASAP 2.0 CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save CSV benchmark output")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and policy-order seed")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unmeasured warm-up runs per sample")
    return parser.parse_args()


def get_policy(name: str, cache_budget: int):
    if name == "full":
        return None
    elif name == "aege":
        return AEGEPolicy(sink_size=4, window_size=min(64, cache_budget // 2), entropy_weight=0.3)
    elif name == "h2o":
        return H2OPolicy(heavy_ratio=0.1)
    elif name == "streaming":
        return StreamingPolicy(sink_size=4, window_size=min(256, cache_budget - 4))
    elif name == "lru":
        return LRUPolicy()
    else:
        raise ValueError(f"Unknown policy: {name}")


def extract_score(text: str) -> Optional[int]:
    """Extract numeric score (1-6) from model output with fallback patterns."""
    if not text:
        return None
    
    # 1. Direct "Score: X" or "score : X" (case-insensitive)
    match = re.search(r"score\s*:\s*([1-6])\b", str(text), re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 2. "Score X/6" or "score of X"
    match = re.search(r"score\s*(?:of|is|=)?\s*([1-6])\s*(?:/|out of\s*6)?", str(text), re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 3. Last standalone digit 1-6 near end of text
    digits = re.findall(r"\b([1-6])\b", str(text))
    if digits:
        return int(digits[-1])

    return None


def run_aes_benchmark():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"==================================================")
    print(f"🚀 Running AES KV Cache Benchmark")
    print(f"• Model: {args.model}")
    print(f"• Device: {device}")
    print(f"• Dtype: {dtype}")
    print(f"• Max Cache Budget: {args.max_cache_size} tokens")
    print(f"• Samples Count: {args.num_samples}")
    print(f"==================================================")

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)

    # Detect if this is an instruct/chat model
    is_instruct = any(kw in args.model.lower() for kw in ["instruct", "chat", "-it", "qwen3-", "gemma-", "mistral"])
    print(f"• Model Type: {'Instruct/Chat (Chat Template ON)' if is_instruct else 'Base Causal (Text Continuation)'}")

    # Load Dataset Samples
    loader = AESDatasetLoader(csv_path=args.dataset_path)
    samples = loader.get_samples(num_samples=args.num_samples, seed=args.seed)
    print(f"✓ Loaded {len(samples)} long-context samples from ASAP 2.0 dataset.")

    policies_to_test = ["full", "aege", "h2o", "streaming", "lru"]
    benchmark_records: List[Dict[str, Any]] = []

    for s_idx, sample in enumerate(samples, 1):
        prompt_text = sample["formatted_prompt"]

        # For instruct models: use chat template so model follows the instruction properly
        if is_instruct and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(device)
        else:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=3072).to(device)




        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        human_score = sample["score"]
        prompt_name = sample["prompt_name"]

        removed_tokens = max(0, seq_len - args.max_cache_size)
        removed_pct = (removed_tokens / seq_len * 100) if seq_len > 0 else 0.0

        print(f"\n--- Sample [{s_idx}/{len(samples)}] (Prompt Topic: '{prompt_name}', Len: {seq_len} tokens, Removed: {removed_tokens} ({removed_pct:.1f}%)) ---", flush=True)


        # Warm up the exact generation path before measuring latency.
        for _ in range(args.warmup_runs):
            warmup_manager = CacheManager(
                model=model,
                policy=None,
                max_cache_size=args.max_cache_size,
                device=torch.device(device),
            )
            with torch.no_grad():
                warmup_manager.generate_with_cache(input_ids=input_ids, max_new_tokens=1)
            warmup_manager.cleanup()

        policy_order = policies_to_test.copy()
        random.Random(args.seed + s_idx).shuffle(policy_order)

        for policy_name in policy_order:
            policy = get_policy(policy_name, cache_budget=args.max_cache_size)
            manager = CacheManager(model=model, policy=policy, max_cache_size=args.max_cache_size, device=torch.device(device))

            start_time = time.perf_counter()
            with torch.no_grad():
                output_ids = manager.generate_with_cache(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                )
            elapsed_time = time.perf_counter() - start_time

            generated_text = tokenizer.decode(output_ids[0][seq_len:], skip_special_tokens=True).strip()
            metrics = manager.get_metrics()
            manager.cleanup()

            actual_generated_tokens = max(0, output_ids.shape[1] - seq_len)
            decode_time = metrics.get("decode_latency_sec", 0.0)
            ttft_sec = round(metrics.get("prefill_latency_sec", 0.0), 4)
            itl_ms = round((decode_time / max(1, metrics.get("decode_model_calls", 0))) * 1000, 2)
            throughput_tok_sec = round(
                actual_generated_tokens / decode_time, 2
            ) if decode_time > 0 else 0.0
            actual_evicted_tokens = int(metrics.get("total_evictions", 0))
            predicted_score = extract_score(generated_text)

            record = {
                "sample_idx": s_idx,
                "prompt_name": prompt_name,
                "human_score": human_score,
                "prompt_tokens": seq_len,
                "max_cache_size": args.max_cache_size,
                "removed_tokens": removed_tokens,
                "removed_pct": round(removed_pct, 1),
                "generated_tokens": args.max_new_tokens,
                "actual_generated_tokens": actual_generated_tokens,
                "policy": policy_name if policy else "FullCache",
                "model": args.model,
                "dtype": str(dtype),
                "device": device,
                "seed": args.seed,
                "latency_sec": round(elapsed_time, 4),
                "ttft_sec": ttft_sec,
                "itl_ms": itl_ms,
                "throughput_tok_sec": throughput_tok_sec,
                "step_count": metrics.get("step_count", 0),
                "actual_evicted_tokens": actual_evicted_tokens,
                "peak_cache_tokens": metrics.get("peak_cache_tokens", 0),
                "final_cache_tokens_per_layer": ":".join(str(x) for x in metrics.get("cache_sizes", [])),
                "predicted_score": predicted_score,
                # Keep the full output for exact-match measurement. The preview
                # is only for readable logs/tables.
                "generated_output": generated_text,
                "generated_output_preview": generated_text[:100],
            }

            benchmark_records.append(record)


            print(f"  • Policy: {record['policy']:<12} | Time: {elapsed_time:.3f}s | Throughput: {throughput_tok_sec:.1f} tok/s | Output: '{generated_text[:30]}...'")

    df_results = pd.DataFrame(benchmark_records)
    out_csv = os.path.join(args.output_dir, "aes_benchmark.csv")
    df_results.to_csv(out_csv, index=False)

    print(f"\n==================================================")
    print(f"📊 Per-Prompt Evaluation Summary")
    print(f"==================================================")
    for p_topic in df_results["prompt_name"].unique():
        sub_df = df_results[df_results["prompt_name"] == p_topic]
        print(f"\n🏷️ Prompt Category: '{p_topic}'")
        for p_name in policies_to_test:
            pol_label = p_name if p_name != "full" else "FullCache"
            pol_df = sub_df[sub_df["policy"] == pol_label]
            if not pol_df.empty:
                avg_time = pol_df["latency_sec"].mean()
                avg_tput = pol_df["throughput_tok_sec"].mean()
                rem_tok = pol_df["removed_tokens"].iloc[0]
                rem_pct = pol_df["removed_pct"].iloc[0]
                print(f"  • Policy: {pol_label:<12} | Avg Time: {avg_time:.3f}s | Throughput: {avg_tput:.1f} tok/s | Removed: {rem_tok} ({rem_pct:.1f}%)")


    print(f"\n==================================================")
    print(f"📊 Accuracy & QWK Evaluation Summary (vs FullCache Baseline)")
    print(f"==================================================")
    
    baseline_df = df_results[df_results["policy"] == "FullCache"]
    
    for p_name in policies_to_test:
        pol_label = p_name if p_name != "full" else "FullCache"
        sub_df = df_results[df_results["policy"] == pol_label]
        if not sub_df.empty:
            # Use the explicitly parsed score, never an arbitrary digit from prose.
            y_true = sub_df["human_score"].values
            y_pred = sub_df["predicted_score"].dropna().astype(int).tolist()
            
            # Compare output text match with FullCache baseline
            exact_matches = 0
            for idx, r in sub_df.iterrows():
                sample_idx = r["sample_idx"]
                base_row = baseline_df[baseline_df["sample_idx"] == sample_idx]
                if not base_row.empty:
                    base_out = base_row["generated_output"].values[0]
                    if r["generated_output"] == base_out:
                        exact_matches += 1
                
            match_pct = (exact_matches / len(sub_df) * 100) if len(sub_df) > 0 else 0.0
            
            if len(y_pred) == len(y_true):
                from src.metrics.quality_metrics import quadratic_weighted_kappa
                qwk_score = quadratic_weighted_kappa(y_true, y_pred)
                qwk_str = f"{qwk_score:.4f}"
            else:
                qwk_str = "N/A (Text Continuation / No explicit 1-6 score)"

            print(f"• Policy: {pol_label:<12} | Output Match vs FullCache: {match_pct:.1f}% | QWK Score: {qwk_str}")

    print(f"\n✅ Benchmark completed successfully! Results saved to: {out_csv}")





if __name__ == "__main__":
    run_aes_benchmark()

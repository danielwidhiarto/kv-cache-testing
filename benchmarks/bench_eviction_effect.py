"""Benchmark: Effect of KV cache eviction on model output.

Honest approach: eviction = remove tokens from input.
1. Run model on FULL input → reference logits (last token prediction)
2. For each policy, select which tokens to KEEP
3. Run model on TRUNCATED input (kept tokens only)
4. Compare next-token predictions

This directly measures: "If we evict these tokens, does the model
still predict the same next token?"
"""

import sys
import time
import torch
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.policies.lru import LRUPolicy
from src.policies.h2o import H2OPolicy
from src.policies.streaming import StreamingPolicy
from src.policies.snap import SnapPolicy
from src.policies.aege import AEGEPolicy


POLICY_REGISTRY = {
    "lru": lambda: LRUPolicy(),
    "h2o_0.1": lambda: H2OPolicy(heavy_ratio=0.1),
    "h2o_0.2": lambda: H2OPolicy(heavy_ratio=0.2),
    "streaming_4_256": lambda: StreamingPolicy(sink_size=4, window_size=256),
    "streaming_4_512": lambda: StreamingPolicy(sink_size=4, window_size=512),
    "snap": lambda: SnapPolicy(observation_size=32),
    "aege": lambda: AEGEPolicy(sink_size=4, window_size=64, entropy_weight=0.3),
    "aege_0.5": lambda: AEGEPolicy(sink_size=4, window_size=64, entropy_weight=0.5),
}


def select_tokens_to_keep(policy, attention_scores, seq_len, keep_size, device):
    """Use eviction policy to select which token indices to keep.

    Returns:
        LongTensor of indices to KEEP, sorted by position.
    """
    num_to_evict = max(0, seq_len - keep_size)

    if num_to_evict == 0:
        return torch.arange(seq_len, device=device)

    # Get eviction indices from policy
    dummy_k = torch.randn(1, 1, seq_len, 1, device=device)
    dummy_v = torch.randn(1, 1, seq_len, 1, device=device)

    evict_indices = policy.select_evict(dummy_k, dummy_v, attention_scores, num_to_evict)

    # Convert to keep indices
    all_indices = set(range(seq_len))
    evict_set = set(evict_indices.cpu().tolist()) if evict_indices.numel() > 0 else set()
    keep_indices = sorted(all_indices - evict_set)

    return torch.tensor(keep_indices, dtype=torch.long, device=device)


def get_next_token_prediction(model, input_ids):
    """Get the model's next token prediction."""
    with torch.no_grad():
        outputs = model(input_ids)
    last_logits = outputs.logits[:, -1, :]  # [batch, vocab]
    predicted_token = torch.argmax(last_logits, dim=-1)  # [batch]
    return last_logits, predicted_token


def run_eviction_comparison(model, tokenizer, prompt, keep_sizes, policies, device):
    """Compare next-token predictions under different eviction policies."""
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = tokens.shape[1]

    # Reference: full input prediction
    ref_logits, ref_token = get_next_token_prediction(model, tokens)
    ref_word = tokenizer.decode(ref_token[0])

    results = {}

    for policy_name in policies:
        if policy_name == "full":
            continue

        if policy_name not in POLICY_REGISTRY:
            continue

        policy = POLICY_REGISTRY[policy_name]()

        # Get attention scores from reference run
        with torch.no_grad():
            ref_output = model(tokens, output_attentions=True)
        attn_scores = ref_output.attentions[-1]

        for keep_size in keep_sizes:
            if keep_size >= seq_len:
                continue

            # Select tokens to keep
            keep_indices = select_tokens_to_keep(
                policy, attn_scores, seq_len, keep_size, device
            )

            # Truncate input to kept tokens only
            truncated_input = tokens[:, keep_indices]

            # Get prediction on truncated input
            trunc_logits, trunc_token = get_next_token_prediction(model, truncated_input)
            trunc_word = tokenizer.decode(trunc_token[0])

            # Compare
            token_match = (ref_token == trunc_token).item()

            # Logit divergence (on the truncated sequence's last position)
            # We compare the full vocab distribution
            ref_probs = torch.softmax(ref_logits, dim=-1)
            trunc_probs = torch.softmax(trunc_logits, dim=-1)

            # KL divergence
            kl_div = torch.nn.functional.kl_div(
                trunc_probs.log(), ref_probs, reduction="batchmean"
            ).item()

            # Cosine similarity
            ref_norm = ref_logits.float() / (ref_logits.float().norm(dim=-1, keepdim=True) + 1e-8)
            trunc_norm = trunc_logits.float() / (trunc_logits.float().norm(dim=-1, keepdim=True) + 1e-8)
            cosine_sim = (ref_norm * trunc_norm).sum(dim=-1).item()

            # Top-k agreement
            top5_ref = set(torch.topk(ref_logits[0], 5).indices.cpu().tolist())
            top5_trunc = set(torch.topk(trunc_logits[0], 5).indices.cpu().tolist())
            top5_overlap = len(top5_ref & top5_trunc) / 5

            key = f"{policy_name}_k{keep_size}"
            results[key] = {
                "policy": policy_name,
                "keep_size": keep_size,
                "seq_len": seq_len,
                "compression_ratio": round(keep_size / seq_len, 3),
                "ref_prediction": ref_word,
                "evicted_prediction": trunc_word,
                "token_match": token_match,
                "cosine_sim": round(cosine_sim, 6),
                "kl_divergence": round(kl_div, 6),
                "top5_overlap": round(top5_overlap, 4),
            }

    return results, ref_word


def main():
    parser = argparse.ArgumentParser(description="Benchmark eviction effects")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument("--keep-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--policies", nargs="+",
                        default=["lru", "h2o_0.1", "streaming_4_256", "snap", "aege"])
    parser.add_argument("--output", default="results/eviction_effect.csv")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    device = next(model.parameters()).device

    prompts = [
        "The quick brown fox jumps over the lazy dog and then runs through the forest where many animals live together in harmony and peace under the bright sun shining above the tall green trees that sway gently in the warm summer breeze blowing from the distant mountains covered with snow and ice melting slowly into crystal clear streams flowing down to the valley below where flowers bloom in every color imaginable attracting butterflies and bees that dance from petal to petal collecting nectar for their hives built high in the ancient oak trees standing strong for hundreds of years providing shelter and shade to all creatures great and small who call this beautiful place their home",
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data and improve their performance over time without being explicitly programmed for every possible scenario they might encounter in the real world which is complex and constantly changing requiring adaptive approaches that can handle uncertainty and noise in the input signals while still producing reliable and accurate outputs that help humans make better decisions in various domains including healthcare finance education transportation and many other fields where data-driven insights can lead to significant improvements in efficiency and effectiveness",
        "In this essay I will discuss the importance of education in modern society and how it shapes the future of individuals and communities around the world by providing knowledge skills and critical thinking abilities that enable people to participate meaningfully in democratic processes contribute to economic development and cultural enrichment while also fostering understanding and tolerance among diverse groups of people who share this planet and must work together to address the pressing challenges of our time including climate change inequality poverty and conflict that threaten the well-being of current and future generations",
        "The history of computational linguistics traces back to the early days of computer science when researchers first attempted to process natural language using machines leading to decades of progress in parsing translation and understanding of human language culminating in modern large language models that can generate coherent text answer questions and perform complex reasoning tasks across multiple languages and domains demonstrating remarkable capabilities that were once thought to be exclusive to human intelligence",
        "Quantum computing represents a fundamental shift in how we process information leveraging the principles of quantum mechanics to perform calculations that would be impossible for classical computers to complete in any reasonable amount of time opening new possibilities for cryptography drug discovery materials science and optimization problems that have long been considered intractable and requiring entirely new algorithms and programming paradigms to fully exploit the unique properties of quantum bits which can exist in superposition and entanglement states simultaneously",
    ]

    print("=" * 80)
    print("  KV Cache Eviction Effect — Next Token Prediction Comparison")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Keep sizes: {args.keep_sizes}")
    print(f"  Policies: {args.policies}")
    print("=" * 80)

    all_results = []

    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
        seq_len = tokens.shape[1]
        print(f"\n  Prompt {i+1}: \"{prompt}\"")
        print(f"  Sequence length: {seq_len} tokens")

        results, ref_word = run_eviction_comparison(
            model, tokenizer, prompt,
            args.keep_sizes, args.policies, device,
        )

        print(f"  Reference prediction: \"{ref_word}\"")
        print(f"  {'Policy':<20} {'Keep':<6} {'Ratio':<7} {'Match':<7} {'Cosine':<10} {'Top5':<7} {'Prediction'}")
        print(f"  {'-'*75}")

        for key, result in results.items():
            all_results.append(result)
            match_str = "✓" if result["token_match"] else "✗"
            print(f"  {result['policy']:<20} {result['keep_size']:<6} "
                  f"{result['compression_ratio']:<7.3f} {match_str:<7} "
                  f"{result['cosine_sim']:<10.4f} {result['top5_overlap']:<7.0%} "
                  f"\"{result['evicted_prediction']}\"")

    # === Summary ===
    print("\n" + "=" * 80)
    print("  Summary: Token Match Rate by Policy and Compression Ratio")
    print("=" * 80)

    # Group by policy
    from collections import defaultdict
    by_policy = defaultdict(list)
    for r in all_results:
        by_policy[r["policy"]].append(r)

    print(f"  {'Policy':<20} {'Avg Match':<12} {'Avg Cosine':<12} {'Avg Top5':<12} {'Num Tests'}")
    print(f"  {'-'*70}")

    for policy_name, results_list in by_policy.items():
        avg_match = np.mean([r["token_match"] for r in results_list])
        avg_cosine = np.mean([r["cosine_sim"] for r in results_list])
        avg_top5 = np.mean([r["top5_overlap"] for r in results_list])
        print(f"  {policy_name:<20} {avg_match:<12.2%} {avg_cosine:<12.4f} {avg_top5:<12.0%} {len(results_list)}")

    # === Export ===
    import csv
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Exported to {args.output}")


if __name__ == "__main__":
    main()

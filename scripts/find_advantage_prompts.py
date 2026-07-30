"""Find prompts where AEGE outperforms StreamingLLM.

Hypothesis: AEGE should win when:
1. Important tokens are in the MIDDLE (not just first/last)
2. Attention is focused on specific informative tokens (low entropy)
3. Complex reasoning requires remembering middle context
4. Entity references in middle positions need to be preserved
"""

import sys
import torch
import csv
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
    "streaming": lambda: StreamingPolicy(sink_size=4, window_size=64),
    "snap": lambda: SnapPolicy(observation_size=32),
    "aege": lambda: AEGEPolicy(sink_size=4, window_size=64, entropy_weight=0.3),
}


def select_tokens_to_keep(policy, attention_scores, seq_len, keep_size, device):
    num_to_evict = max(0, seq_len - keep_size)
    if num_to_evict == 0:
        return torch.arange(seq_len, device=device)

    dummy_k = torch.randn(1, 1, seq_len, 1, device=device)
    dummy_v = torch.randn(1, 1, seq_len, 1, device=device)
    evict_indices = policy.select_evict(dummy_k, dummy_v, attention_scores, num_to_evict)

    all_indices = set(range(seq_len))
    evict_set = set(evict_indices.cpu().tolist()) if evict_indices.numel() > 0 else set()
    keep_indices = sorted(all_indices - evict_set)
    return torch.tensor(keep_indices, dtype=torch.long, device=device)


def test_prompt(model, tokenizer, prompt, keep_sizes, device):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = tokens.shape[1]

    with torch.no_grad():
        ref_output = model(tokens, output_attentions=True)
    ref_logits = ref_output.logits[:, -1, :]
    ref_token = torch.argmax(ref_logits, dim=-1)
    ref_word = tokenizer.decode(ref_token[0])
    attn_scores = ref_output.attentions[-1]

    results = {}
    for policy_name, policy_fn in POLICY_REGISTRY.items():
        policy = policy_fn()
        for keep_size in keep_sizes:
            if keep_size >= seq_len:
                continue

            keep_indices = select_tokens_to_keep(policy, attn_scores, seq_len, keep_size, device)
            truncated_input = tokens[:, keep_indices]

            with torch.no_grad():
                trunc_output = model(truncated_input)
            trunc_logits = trunc_output.logits[:, -1, :]
            trunc_token = torch.argmax(trunc_logits, dim=-1)
            trunc_word = tokenizer.decode(trunc_token[0])

            token_match = (ref_token == trunc_token).item()

            ref_norm = ref_logits.float() / (ref_logits.float().norm(dim=-1, keepdim=True) + 1e-8)
            trunc_norm = trunc_logits.float() / (trunc_logits.float().norm(dim=-1, keepdim=True) + 1e-8)
            cosine_sim = (ref_norm * trunc_norm).sum(dim=-1).item()

            top5_ref = set(torch.topk(ref_logits[0], 5).indices.cpu().tolist())
            top5_trunc = set(torch.topk(trunc_logits[0], 5).indices.cpu().tolist())
            top5_overlap = len(top5_ref & top5_trunc) / 5

            key = f"{policy_name}_k{keep_size}"
            results[key] = {
                "policy": policy_name,
                "keep_size": keep_size,
                "seq_len": seq_len,
                "compression_ratio": round(keep_size / seq_len, 3),
                "token_match": token_match,
                "cosine_sim": round(cosine_sim, 6),
                "top5_overlap": round(top5_overlap, 4),
                "ref_word": ref_word,
                "pred_word": trunc_word,
            }

    return results, ref_word


def main():
    model, tokenizer = load_model("gpt2", device=None)
    device = next(model.parameters()).device

    # Diverse prompts designed to test different attention patterns
    prompts = [
        # === Category 1: Entity tracking (middle tokens important) ===
        "Albert Einstein who was born in Germany in 1879 developed the theory of relativity which changed our understanding of space and time forever",
        "Marie Curie the famous physicist and chemist discovered radium and polonium through years of painstaking research in her Paris laboratory",
        "The Amazon river which flows through Brazil Peru and Colombia is the largest river by volume and supports an incredibly diverse ecosystem",
        "Shakespeare wrote Romeo and Juliet a tragic love story about two young lovers from feuding families in the city of Verona Italy",
        "Nikola Tesla the Serbian American inventor contributed to the design of the modern alternating current electrical supply system",

        # === Category 2: Reasoning chains (middle context critical) ===
        "If all roses are flowers and all flowers need water then it follows logically that all roses need water to survive and grow properly",
        "The premise states that every student who studies hard will pass the exam and since Maria studies hard every day she will certainly pass",
        "When the temperature drops below zero degrees Celsius water freezes into ice which expands and can cause pipes to burst in winter",
        "Because the stock market crashed in 1929 many banks failed which led to widespread unemployment and eventually the great depression",
        "Since renewable energy sources like solar and wind produce no emissions they are essential for combating climate change and global warming",

        # === Category 3: Information-dense (key facts in middle) ===
        "The speed of light is approximately 299792458 meters per second which Einstein showed is the maximum speed at which information can travel",
        "DNA which stands for deoxyribonucleic acid contains the genetic instructions for the development and function of all known living organisms",
        "The human brain contains approximately 86 billion neurons which communicate through electrical and chemical signals across trillions of synapses",
        "Photosynthesis is the process by which green plants convert sunlight water and carbon dioxide into glucose and oxygen using chlorophyll",
        "The periodic table organizes chemical elements by atomic number and Dmitri Mendeleev is credited with creating its first widely recognized version",

        # === Category 4: Long narratives (middle gets compressed away) ===
        "In the ancient kingdom of Eldoria there lived a wise king who ruled with justice and compassion for his people for many prosperous years until a terrible dragon appeared from the mountains and threatened to destroy everything the kingdom had built over centuries of peace and prosperity",
        "The detective carefully examined every clue at the crime scene including fingerprints footprints and a mysterious letter left on the desk which ultimately led to the identification of the suspect who had been hiding in plain sight for months",
        "Scientists at the research facility spent years developing a new vaccine that could protect against multiple strains of the virus and after extensive clinical trials it was approved for public use saving millions of lives worldwide",
        "The expedition team climbed through treacherous terrain enduring freezing temperatures and altitude sickness before finally reaching the summit of the mountain where they planted a flag and took photographs to document their achievement",
        "The computer program was designed to analyze vast amounts of data and identify patterns that humans might miss which made it invaluable for medical research financial analysis and weather prediction among many other applications",

        # === Category 5: Contrasting patterns (testing entropy) ===
        "The cat sat on the mat while the dog lay on the rug and the bird perched on the branch outside the window of the old house",
        "Quickly running jumping climbing swimming the athlete demonstrated incredible versatility in the competition impressing all the judges and spectators alike",
        "First we mix the flour and sugar then we add the eggs and milk after that we stir everything together and finally we bake at 350 degrees",
        "The president the secretary and the treasurer each gave their own unique perspective on the budget proposal during the heated board meeting",
        "Despite the heavy rain and strong winds the construction workers continued building the bridge connecting the two cities across the wide river",
    ]

    keep_sizes = [16, 32, 48]

    print("=" * 100)
    print("  Prompt Advantage Analysis: AEGE vs StreamingLLM")
    print("=" * 100)

    all_results = []
    aege_wins = []
    streaming_wins = []
    ties = []

    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt[:80]}...\"")
        results, ref_word = test_prompt(model, tokenizer, prompt, keep_sizes, device)

        # Compare AEGE vs Streaming at each keep_size
        for ks in keep_sizes:
            aege_key = f"aege_k{ks}"
            stream_key = f"streaming_k{ks}"

            if aege_key not in results or stream_key not in results:
                continue

            aege = results[aege_key]
            stream = results[stream_key]

            all_results.append({
                "prompt_idx": i + 1,
                "prompt": prompt[:100],
                "keep_size": ks,
                "seq_len": aege["seq_len"],
                "ref_word": ref_word,
                "aege_match": aege["token_match"],
                "aege_cosine": aege["cosine_sim"],
                "aege_pred": aege["pred_word"],
                "stream_match": stream["token_match"],
                "stream_cosine": stream["cosine_sim"],
                "stream_pred": stream["pred_word"],
            })

            entry = {
                "prompt": prompt[:80],
                "keep_size": ks,
                "seq_len": aege["seq_len"],
                "ref": ref_word,
                "aege_pred": aege["pred_word"],
                "stream_pred": stream["pred_word"],
            }

            if aege["token_match"] and not stream["token_match"]:
                aege_wins.append(entry)
                print(f"    k{ks}: AEGE WINS ✓ vs ✗  (aege=\"{aege['pred_word']}\" stream=\"{stream['pred_word']}\")")
            elif stream["token_match"] and not aege["token_match"]:
                streaming_wins.append(entry)
                print(f"    k{ks}: STREAMING WINS ✗ vs ✓  (aege=\"{aege['pred_word']}\" stream=\"{stream['pred_word']}\")")
            elif aege["token_match"] and stream["token_match"]:
                # Both match - check cosine
                if aege["cosine_sim"] > stream["cosine_sim"] + 0.0001:
                    print(f"    k{ks}: TIE (match) but AEGE cosine higher: {aege['cosine_sim']:.6f} vs {stream['cosine_sim']:.6f}")
                elif stream["cosine_sim"] > aege["cosine_sim"] + 0.0001:
                    print(f"    k{ks}: TIE (match) but Streaming cosine higher: {stream['cosine_sim']:.6f} vs {aege['cosine_sim']:.6f}")
                else:
                    print(f"    k{ks}: TIE (match) cosine ≈ {aege['cosine_sim']:.6f}")
                ties.append(entry)
            else:
                # Both miss - check who's closer
                if aege["cosine_sim"] > stream["cosine_sim"] + 0.0001:
                    print(f"    k{ks}: BOTH MISS but AEGE closer (cos {aege['cosine_sim']:.6f} vs {stream['cosine_sim']:.6f})")
                elif stream["cosine_sim"] > aege["cosine_sim"] + 0.0001:
                    print(f"    k{ks}: BOTH MISS but Streaming closer (cos {stream['cosine_sim']:.6f} vs {aege['cosine_sim']:.6f})")
                ties.append(entry)

    # === Summary ===
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"\n  Total comparisons: {len(all_results)}")
    print(f"  AEGE wins (match vs no-match): {len(aege_wins)}")
    print(f"  Streaming wins (match vs no-match): {len(streaming_wins)}")
    print(f"  Ties / Both miss: {len(ties)}")

    if aege_wins:
        print(f"\n  === AEGE WINS ({len(aege_wins)} cases) ===")
        for w in aege_wins:
            print(f"    k{w['keep_size']} seq={w['seq_len']}: \"{w['prompt']}...\"")
            print(f"      ref=\"{w['ref']}\" aege=\"{w['aege_pred']}\" stream=\"{w['stream_pred']}\"")

    if streaming_wins:
        print(f"\n  === STREAMING WINS ({len(streaming_wins)} cases) ===")
        for w in streaming_wins:
            print(f"    k{w['keep_size']} seq={w['seq_len']}: \"{w['prompt']}...\"")
            print(f"      ref=\"{w['ref']}\" aege=\"{w['aege_pred']}\" stream=\"{w['stream_pred']}\"")

    # Export
    output_path = "results/aege_vs_streaming_prompts.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Exported to {output_path}")


if __name__ == "__main__":
    main()

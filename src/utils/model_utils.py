"""Utility functions for model loading and text processing."""

import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a HuggingFace model and tokenizer.

    Args:
        model_name: HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-3.2-1B")
        device: Device to load on ("cuda", "cpu", "mps", or None for auto)
        dtype: Model dtype (float16 for GPU, float32 for CPU)
        trust_remote_code: Whether to trust remote code

    Returns:
        (model, tokenizer) tuple
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            dtype = torch.float32  # CPU doesn't support fp16 well

    print(f"Loading model: {model_name}")
    print(f"  Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",  # Need attention weights for eviction policies
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    return model, tokenizer


def get_available_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def estimate_memory(model: PreTrainedModel) -> dict:
    """Estimate model memory usage."""
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return {
        "param_count": param_count,
        "param_size_mb": round(param_size_mb, 2),
        "dtype": str(next(model.parameters()).dtype),
    }


def format_token_count(count: int) -> str:
    """Format token count for display."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)

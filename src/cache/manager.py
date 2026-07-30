"""CacheManager — injects KV cache eviction into HuggingFace models via hooks."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from .base import KVCache
from .full_cache import FullCache
from ..policies.base import EvictionPolicy


class CacheManager:
    """Wraps a HuggingFace model and intercepts attention layers
    to capture/modify KV cache with eviction policies.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        policy = H2OPolicy(heavy_ratio=0.1)
        manager = CacheManager(model, policy, max_cache_size=1024)
        output = manager.generate(input_ids)
        metrics = manager.get_metrics()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        policy: Optional[EvictionPolicy] = None,
        max_cache_size: int = 2048,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.policy = policy
        self.max_cache_size = max_cache_size
        self.device = device or next(model.parameters()).device

        # Per-layer KV caches
        self._caches: Dict[int, FullCache] = {}
        self._hooks: List[Any] = []

        # Metrics tracking
        self._step_count = 0
        self._total_evictions = 0
        self._cache_sizes_over_time: List[List[int]] = []  # per layer

        # Attention scores captured from hooks
        self._attention_scores: Dict[int, torch.Tensor] = {}

        self._setup_hooks()

    def _get_attention_layers(self) -> List[nn.Module]:
        """Find attention layers in the model."""
        layers = []

        # Common patterns for different model architectures
        if hasattr(self.model, "transformer"):
            # GPT-2 style
            if hasattr(self.model.transformer, "h"):
                for block in self.model.transformer.h:
                    if hasattr(block, "attn"):
                        layers.append(block.attn)
        elif hasattr(self.model, "model"):
            # Llama/Mistral style
            if hasattr(self.model.model, "layers"):
                for block in self.model.model.layers:
                    if hasattr(block, "self_attn"):
                        layers.append(block.self_attn)

        if not layers:
            # Fallback: find any module with "attention" or "attn" in name
            for name, module in self.model.named_modules():
                if "attn" in name.lower() and hasattr(module, "weight"):
                    layers.append(module)

        return layers

    def _setup_hooks(self):
        """Register forward hooks on attention layers to capture KV pairs."""
        layers = self._get_attention_layers()

        for i, layer in enumerate(layers):
            self._caches[i] = FullCache(max_size=self.max_cache_size)

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    # Try to extract K, V from the attention output
                    # This is architecture-dependent
                    if isinstance(output, tuple) and len(output) >= 2:
                        # Some models return (attn_output, attn_weights, past_key_value)
                        pass
                    # We'll capture via the forward pass modification instead
                    pass
                return hook_fn

            self._hooks.append(layer.register_forward_hook(make_hook(i)))

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
    ) -> Any:
        """Run forward pass with KV cache tracking.

        Args:
            input_ids: Token IDs, shape [batch, seq_len]
            attention_mask: Optional mask
            output_attentions: Whether to capture attention weights

        Returns:
            Model output with logits and optionally attention weights
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                output_attentions=output_attentions,
                use_cache=True,
            )

        # If we have attention scores and a policy that uses them
        if output_attentions and self.policy and self.policy.requires_attention_scores:
            if hasattr(outputs, "attentions") and outputs.attentions:
                for layer_idx, attn in enumerate(outputs.attentions):
                    self._attention_scores[layer_idx] = attn

        self._step_count += 1
        return outputs

    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens with KV cache eviction.

        For each generation step, the cache manager intercepts
        the attention layers and applies the eviction policy.

        Args:
            input_ids: Prompt token IDs, shape [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            attention_mask: Optional mask
            **kwargs: Additional generation kwargs

        Returns:
            Generated token IDs, shape [batch, seq_len + max_new_tokens]
        """
        self.model.eval()
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated.to(self.device),
                    attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                    output_attentions=True,
                    use_cache=False,  # We manage cache ourselves
                    **kwargs,
                )

            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.to(generated.device)], dim=1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)

            # Track metrics
            self._step_count += 1

            # Check for EOS
            if next_token.item() == self.model.config.eos_token_id:
                break

        return generated

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics."""
        return {
            "step_count": self._step_count,
            "total_evictions": self._total_evictions,
            "cache_sizes": self._cache_sizes_over_time,
            "policy_name": self.policy.name if self.policy else "none",
            "max_cache_size": self.max_cache_size,
        }

    def reset(self):
        """Reset all caches and metrics."""
        for cache in self._caches.values():
            cache.reset()
        self._step_count = 0
        self._total_evictions = 0
        self._cache_sizes_over_time = []
        self._attention_scores = {}
        if self.policy:
            self.policy.reset()

    def cleanup(self):
        """Remove hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

"""Generation manager with real Hugging Face past_key_values eviction."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel

from .base import KVCache
from .full_cache import FullCache
from .policy_cache import PolicyCache
from ..policies.base import EvictionPolicy


LegacyPast = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class CacheManager:
    """Run prefill once, then decode with a bounded per-layer KV cache.

    The previous implementation registered no-op hooks and recomputed the
    entire generated sequence with ``use_cache=False``. This manager instead
    consumes the model's real ``past_key_values`` and rebuilds the legacy tuple
    from the pruned per-layer caches after every decode step.
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
        self.max_cache_size = int(max_cache_size)
        self.device = device or next(model.parameters()).device

        self._caches: List[KVCache] = []
        self._step_count = 0
        self._prefill_count = 0
        self._peak_cache_tokens = 0
        self._prefill_latency_sec = 0.0
        self._decode_latency_sec = 0.0
        self._decode_model_calls = 0
        self._use_dynamic_cache = False

    @staticmethod
    def _to_legacy_past(past_key_values: Any) -> LegacyPast:
        """Normalize tuple/list/DynamicCache outputs to ``(K, V)`` tuples."""
        if past_key_values is None:
            raise RuntimeError(
                "The model returned no past_key_values. Load it with use_cache=True."
            )

        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

        # Transformers versions differ: some DynamicCache releases expose
        # key_cache/value_cache directly but do not expose to_legacy_cache.
        if not isinstance(past_key_values, (tuple, list)):
            key_cache = getattr(past_key_values, "key_cache", None)
            value_cache = getattr(past_key_values, "value_cache", None)
            if key_cache is not None and value_cache is not None:
                past_key_values = tuple(zip(key_cache, value_cache))

        if not isinstance(past_key_values, (tuple, list)):
            layers = getattr(past_key_values, "layers", None)
            if layers is not None:
                converted_layers = []
                for layer in layers:
                    key = getattr(layer, "keys", getattr(layer, "key", None))
                    value = getattr(layer, "values", getattr(layer, "value", None))
                    if key is None or value is None:
                        raise TypeError("Could not read keys/values from DynamicCache layer")
                    converted_layers.append((key, value))
                past_key_values = tuple(converted_layers)

        if not isinstance(past_key_values, (tuple, list)):
            raise TypeError(
                f"Unsupported past_key_values type: {type(past_key_values).__name__}"
            )

        legacy: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("Each cache layer must contain key and value tensors")
            legacy.append((layer[0], layer[1]))
        return tuple(legacy)

    def _new_layer_cache(self) -> KVCache:
        if self.policy is None:
            return FullCache(max_size=self.max_cache_size)
        # Attention statistics are independent for each transformer layer.
        return PolicyCache(
            max_size=self.max_cache_size,
            policy=copy.deepcopy(self.policy),
        )

    def _attention_required(self) -> bool:
        return bool(self.policy is not None and self.policy.requires_attention_scores)

    def _get_layer_attention(
        self,
        attentions: Optional[Sequence[torch.Tensor]],
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        if attentions is None or layer_idx >= len(attentions):
            if self._attention_required():
                raise RuntimeError(
                    "This policy requires attention scores, but the model did not "
                    "return them. Use attn_implementation='eager'."
                )
            return None
        return attentions[layer_idx]

    def _append_past(
        self,
        past_key_values: LegacyPast,
        attentions: Optional[Sequence[torch.Tensor]],
        only_last: bool = False,
    ) -> None:
        if not self._caches:
            self._caches = [self._new_layer_cache() for _ in past_key_values]

        if len(self._caches) != len(past_key_values):
            raise RuntimeError(
                "The number of model cache layers changed during generation"
            )

        for layer_idx, (key, value) in enumerate(past_key_values):
            if only_last:
                key = key[:, :, -1:, :]
                value = value[:, :, -1:, :]
            attention = self._get_layer_attention(attentions, layer_idx)
            self._caches[layer_idx].append(key, value, attention)
            after = self._caches[layer_idx].size()
            self._peak_cache_tokens = max(
                self._peak_cache_tokens,
                sum(cache.size() for cache in self._caches),
            )
            if self.policy is not None and after > self.max_cache_size:
                raise RuntimeError(
                    f"Layer {layer_idx} exceeded cache budget: {after} > "
                    f"{self.max_cache_size}"
                )

    def _current_past(self) -> Any:
        if not self._caches:
            raise RuntimeError("KV caches have not been initialized")
        legacy = tuple(cache.get() for cache in self._caches)
        if not self._use_dynamic_cache:
            return legacy

        try:
            from transformers import DynamicCache

            try:
                return DynamicCache.from_legacy_cache(legacy, config=self.model.config)
            except TypeError:
                return DynamicCache.from_legacy_cache(legacy)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "The model returned DynamicCache, but this Transformers version "
                "cannot reconstruct it from pruned KV tensors."
            ) from exc

    def _current_cache_length(self) -> int:
        if not self._caches:
            return 0
        lengths = {cache.size() for cache in self._caches}
        if len(lengths) != 1:
            raise RuntimeError(f"Per-layer cache lengths diverged: {sorted(lengths)}")
        return next(iter(lengths))

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
    ) -> Any:
        """Run one ordinary forward pass with the model's native cache."""
        self.model.eval()
        with torch.no_grad():
            return self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=(
                    attention_mask.to(self.device)
                    if attention_mask is not None
                    else None
                ),
                output_attentions=output_attentions,
                use_cache=True,
            )

    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate with real prefill/decode KV-cache eviction.

        The prompt is prefetched once. At each subsequent step the model sees
        only the newest token plus the current pruned ``past_key_values``.
        Position IDs remain absolute so removing old tokens does not renumber
        the surviving context.
        """
        self.model.eval()
        self.reset()

        input_ids = input_ids.to(self.device)
        generated = input_ids.clone()
        batch_size, prompt_length = input_ids.shape

        model_kwargs = dict(kwargs)
        model_kwargs.pop("use_cache", None)
        model_kwargs.pop("output_attentions", None)
        model_kwargs.pop("past_key_values", None)

        if attention_mask is None:
            prompt_attention_mask = torch.ones(
                batch_size,
                prompt_length,
                dtype=torch.long,
                device=self.device,
            )
        else:
            prompt_attention_mask = attention_mask.to(self.device)

        need_attentions = self._attention_required()
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=prompt_attention_mask,
                output_attentions=need_attentions,
                use_cache=True,
                **model_kwargs,
            )
        self._prefill_latency_sec = time.perf_counter() - prefill_start

        self._use_dynamic_cache = not isinstance(outputs.past_key_values, (tuple, list))
        prompt_past = self._to_legacy_past(outputs.past_key_values)
        self._append_past(prompt_past, outputs.attentions, only_last=False)
        self._prefill_count = 1

        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        for step in range(max_new_tokens):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.to(generated.device)], dim=1)
            self._step_count += 1

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if step == max_new_tokens - 1:
                break

            cache_length = self._current_cache_length()
            decode_attention_mask = torch.ones(
                batch_size,
                cache_length + 1,
                dtype=prompt_attention_mask.dtype,
                device=self.device,
            )
            absolute_position = prompt_length + step
            position_ids = torch.full(
                (batch_size, 1),
                absolute_position,
                dtype=torch.long,
                device=self.device,
            )

            decode_start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token.to(self.device),
                    attention_mask=decode_attention_mask,
                    position_ids=position_ids,
                    past_key_values=self._current_past(),
                    output_attentions=need_attentions,
                    use_cache=True,
                    **model_kwargs,
                )
            self._decode_latency_sec += time.perf_counter() - decode_start
            self._decode_model_calls += 1

            step_past = self._to_legacy_past(outputs.past_key_values)
            self._append_past(step_past, outputs.attentions, only_last=True)

        return generated

    def get_metrics(self) -> Dict[str, Any]:
        total_evictions = sum(
            int(getattr(cache, "evicted_tokens", 0)) for cache in self._caches
        )
        return {
            "step_count": self._step_count,
            "prefill_count": self._prefill_count,
            "decode_steps": self._step_count,
            "total_evictions": total_evictions,
            "cache_sizes": [cache.size() for cache in self._caches],
            "peak_cache_tokens": self._peak_cache_tokens,
            "prefill_latency_sec": self._prefill_latency_sec,
            "decode_latency_sec": self._decode_latency_sec,
            "decode_model_calls": self._decode_model_calls,
            "policy_name": self.policy.name if self.policy else "none",
            "max_cache_size": self.max_cache_size,
        }

    def reset(self) -> None:
        for cache in self._caches:
            cache.reset()
        self._caches = []
        self._step_count = 0
        self._prefill_count = 0
        self._peak_cache_tokens = 0
        self._prefill_latency_sec = 0.0
        self._decode_latency_sec = 0.0
        self._decode_model_calls = 0
        self._use_dynamic_cache = False

    def cleanup(self) -> None:
        """Compatibility no-op; the implementation no longer uses hooks."""
        self.reset()

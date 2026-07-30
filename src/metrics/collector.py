"""Metrics collector for KV cache experiments."""

import time
import torch
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StepMetrics:
    """Metrics for a single forward/generation step."""
    step: int
    latency_ms: float
    cache_size: int
    tokens_generated: int
    peak_memory_mb: float
    evictions: int


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for a full experiment."""
    policy_name: str
    model_name: str
    max_cache_size: int
    input_length: int
    steps: List[StepMetrics] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.latency_ms for s in self.steps) / len(self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def peak_memory_mb(self) -> float:
        if not self.steps:
            return 0.0
        return max(s.peak_memory_mb for s in self.steps)

    @property
    def total_evictions(self) -> int:
        return sum(s.evictions for s in self.steps)

    @property
    def tokens_per_second(self) -> float:
        total_time = self.total_latency_ms / 1000.0
        if total_time == 0:
            return 0.0
        total_tokens = sum(s.tokens_generated for s in self.steps)
        return total_tokens / total_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "model": self.model_name,
            "max_cache_size": self.max_cache_size,
            "input_length": self.input_length,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "total_evictions": self.total_evictions,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "num_steps": len(self.steps),
        }


class MetricsCollector:
    """Collects and aggregates metrics during KV cache experiments.

    Usage:
        collector = MetricsCollector()
        collector.start_step()
        # ... run forward pass ...
        collector.end_step(cache_size=512, tokens_generated=1, evictions=10)
        metrics = collector.finalize("h2o", "gpt2", 1024, 128)
    """

    def __init__(self):
        self._steps: List[StepMetrics] = []
        self._step_start: Optional[float] = None
        self._step_idx: int = 0

    def start_step(self):
        """Mark the start of a forward/generation step."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._step_start = time.perf_counter()

    def end_step(
        self,
        cache_size: int = 0,
        tokens_generated: int = 1,
        evictions: int = 0,
    ):
        """Mark the end of a step and record metrics.

        Args:
            cache_size: Current KV cache size in tokens
            tokens_generated: Number of tokens generated in this step
            evictions: Number of evictions performed
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latency_ms = (time.perf_counter() - self._step_start) * 1000.0

        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

        self._steps.append(StepMetrics(
            step=self._step_idx,
            latency_ms=latency_ms,
            cache_size=cache_size,
            tokens_generated=tokens_generated,
            peak_memory_mb=peak_memory,
            evictions=evictions,
        ))
        self._step_idx += 1

    def finalize(
        self,
        policy_name: str,
        model_name: str,
        max_cache_size: int,
        input_length: int,
    ) -> ExperimentMetrics:
        """Finalize and return aggregated metrics.

        Args:
            policy_name: Name of the eviction policy
            model_name: Name of the model
            max_cache_size: Maximum cache size used
            input_length: Input sequence length

        Returns:
            ExperimentMetrics with all collected data
        """
        return ExperimentMetrics(
            policy_name=policy_name,
            model_name=model_name,
            max_cache_size=max_cache_size,
            input_length=input_length,
            steps=self._steps.copy(),
        )

    def reset(self):
        """Reset collector for a new experiment."""
        self._steps = []
        self._step_idx = 0
        self._step_start = None

"""Deterministic model stubs with explicit cost and latency. No network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Capability and economics card for one routable model."""

    name: str
    quality: float  # [0, 1] expected task success on a reference mix
    cost_per_1k_in: float  # USD
    cost_per_1k_out: float  # USD
    latency_p50_ms: float
    latency_p95_ms: float
    context_limit: int = 128_000
    modalities: frozenset[str] = frozenset({"text"})
    capabilities: frozenset[str] = frozenset({"chat", "code", "math"})


@dataclass
class ScriptedModel:
    """Returns pre-written completions; records every prompt."""

    name: str
    replies: list[str]
    logprobs: list[list[float]] | None = None
    prompts: list[str] = field(default_factory=list)
    _i: int = 0

    def complete(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        if self._i >= len(self.replies):
            raise RuntimeError(f"{self.name} exhausted scripted replies")
        text = self.replies[self._i]
        lp = None
        if self.logprobs is not None and self._i < len(self.logprobs):
            lp = list(self.logprobs[self._i])
        self._i += 1
        out: dict[str, Any] = {"model": self.name, "text": text}
        if lp is not None:
            out["token_logprobs"] = lp
        return out


@dataclass
class CostLatencyModel:
    """Toy model that returns a fixed quality score with priced tokens."""

    spec: ModelSpec
    success_on: frozenset[str] = frozenset()
    fail_rate: float = 0.0
    calls: int = 0

    def complete(self, prompt: str, *, n_in: int = 200, n_out: int = 80) -> dict[str, Any]:
        self.calls += 1
        ok = prompt in self.success_on or (
            not self.success_on and (hash(prompt) % 100) / 100 >= self.fail_rate
        )
        cost = (n_in / 1000) * self.spec.cost_per_1k_in + (n_out / 1000) * self.spec.cost_per_1k_out
        return {
            "model": self.spec.name,
            "text": "ok" if ok else "wrong",
            "ok": ok,
            "cost_usd": cost,
            "latency_ms": self.spec.latency_p50_ms,
            "n_in": n_in,
            "n_out": n_out,
        }


# Reference tier cards used throughout the volume demos.
TINY = ModelSpec(
    name="tiny-local",
    quality=0.62,
    cost_per_1k_in=0.0,
    cost_per_1k_out=0.0,
    latency_p50_ms=40,
    latency_p95_ms=80,
    context_limit=8_192,
    capabilities=frozenset({"chat"}),
)
MID = ModelSpec(
    name="mid-open",
    quality=0.78,
    cost_per_1k_in=0.15,
    cost_per_1k_out=0.60,
    latency_p50_ms=350,
    latency_p95_ms=900,
    context_limit=32_768,
    capabilities=frozenset({"chat", "code", "math"}),
)
FRONTIER = ModelSpec(
    name="frontier-api",
    quality=0.92,
    cost_per_1k_in=3.0,
    cost_per_1k_out=15.0,
    latency_p50_ms=800,
    latency_p95_ms=2200,
    context_limit=200_000,
    modalities=frozenset({"text", "image"}),
    capabilities=frozenset({"chat", "code", "math", "vision", "tools"}),
)

DEFAULT_CATALOGUE: tuple[ModelSpec, ...] = (TINY, MID, FRONTIER)

"""Cascades: sequential escalation with deferral threshold optimisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from forge.router.models import CostLatencyModel


@dataclass(frozen=True)
class CascadeResult:
    model: str
    text: str
    ok: bool
    cost_usd: float
    latency_ms: float
    steps: int
    deferred: bool
    confidence: float


@dataclass
class Cascade:
    """Try models in order; escalate when confidence < threshold."""

    models: list[CostLatencyModel]
    confidence_fn: Callable[[dict], float]
    threshold: float = 0.7

    def run(self, prompt: str, *, n_in: int = 200, n_out: int = 80) -> CascadeResult:
        total_cost = 0.0
        total_lat = 0.0
        last: dict | None = None
        conf = 0.0
        for i, m in enumerate(self.models):
            out = m.complete(prompt, n_in=n_in, n_out=n_out)
            total_cost += float(out["cost_usd"])
            total_lat += float(out["latency_ms"])
            conf = float(self.confidence_fn(out))
            last = out
            deferred = conf < self.threshold and i < len(self.models) - 1
            if not deferred:
                return CascadeResult(
                    model=out["model"],
                    text=str(out["text"]),
                    ok=bool(out["ok"]),
                    cost_usd=total_cost,
                    latency_ms=total_lat,
                    steps=i + 1,
                    deferred=False,
                    confidence=conf,
                )
        assert last is not None
        return CascadeResult(
            model=str(last["model"]),
            text=str(last["text"]),
            ok=bool(last["ok"]),
            cost_usd=total_cost,
            latency_ms=total_lat,
            steps=len(self.models),
            deferred=False,
            confidence=conf,
        )


def _fresh(m: CostLatencyModel) -> CostLatencyModel:
    return CostLatencyModel(spec=m.spec, success_on=set(m.success_on), fail_rate=m.fail_rate)


def optimise_threshold(
    prompts: list[str],
    models: list[CostLatencyModel],
    confidence_fn: Callable[[dict], float],
    *,
    min_quality: float | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Grid-search τ to minimise expected cost subject to mean(ok) ≥ min_quality.

    If ``min_quality`` is None, use the mid-tier model's solo quality on ``prompts``.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    mid = models[min(1, len(models) - 1)]
    mid_fresh = _fresh(mid)
    solo = float(np.mean([mid_fresh.complete(p)["ok"] for p in prompts]))
    target = solo if min_quality is None else min_quality

    best_tau = 1.0
    best_cost = float("inf")
    best: dict[str, float] = {
        "quality": 0.0,
        "expected_cost": best_cost,
        "solo_mid_quality": solo,
        "threshold": best_tau,
    }
    for tau in thresholds:
        cascade = Cascade(
            models=[_fresh(m) for m in models],
            confidence_fn=confidence_fn,
            threshold=float(tau),
        )
        costs, oks = [], []
        for p in prompts:
            r = cascade.run(p)
            costs.append(r.cost_usd)
            oks.append(1.0 if r.ok else 0.0)
        quality = float(np.mean(oks))
        exp_cost = float(np.mean(costs))
        if quality + 1e-9 >= target - 0.02 and exp_cost < best_cost:
            best_cost = exp_cost
            best_tau = float(tau)
            best = {
                "quality": quality,
                "expected_cost": exp_cost,
                "solo_mid_quality": solo,
                "threshold": best_tau,
            }
    return best_tau, best


def expected_cascade_cost(probs_defer: list[float], costs: list[float]) -> float:
    """E[cost] = c0 + p0·c1 + p0·p1·c2 + … for sequential deferral probabilities."""
    total = 0.0
    survive = 1.0
    for i, c in enumerate(costs):
        total += survive * c
        if i < len(probs_defer):
            survive *= probs_defer[i]
    return total

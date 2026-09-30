"""Cost- and latency-aware selection under budgets and SLOs."""
from __future__ import annotations

from dataclasses import dataclass

from forge.router.features import RequestFeatures
from forge.router.models import ModelSpec
from forge.router.rules import RouteDecision


@dataclass(frozen=True)
class Budget:
    max_cost_usd: float | None = None
    max_latency_p95_ms: float | None = None
    prefer: str = "cost"  # cost | quality | latency


@dataclass
class QueueState:
    """Per-model queue depth for queue-aware latency prediction."""

    depth: dict[str, int]
    service_ms: dict[str, float]  # mean service time when busy

    def predicted_latency_ms(self, name: str, base_p95: float) -> float:
        q = self.depth.get(name, 0)
        svc = self.service_ms.get(name, base_p95)
        return base_p95 + q * svc


def estimate_request_cost(spec: ModelSpec, n_in: int, n_out: int = 200) -> float:
    return (n_in / 1000) * spec.cost_per_1k_in + (n_out / 1000) * spec.cost_per_1k_out


@dataclass
class CostLatencyRouter:
    catalogue: list[ModelSpec]
    budget: Budget
    queue: QueueState | None = None

    def feasible(self, feat: RequestFeatures) -> list[tuple[ModelSpec, float, float]]:
        """Return (spec, est_cost, est_latency_p95) for models under budget."""
        rows: list[tuple[ModelSpec, float, float]] = []
        n_in = feat.n_tokens_est
        max_cost = feat.max_cost_usd if feat.max_cost_usd is not None else self.budget.max_cost_usd
        max_lat = (
            feat.max_latency_ms if feat.max_latency_ms is not None else self.budget.max_latency_p95_ms
        )
        for spec in self.catalogue:
            cost = estimate_request_cost(spec, n_in)
            lat = spec.latency_p95_ms
            if self.queue is not None:
                lat = self.queue.predicted_latency_ms(spec.name, spec.latency_p95_ms)
            if max_cost is not None and cost > max_cost:
                continue
            if max_lat is not None and lat > max_lat:
                continue
            rows.append((spec, cost, lat))
        return rows

    def route(self, feat: RequestFeatures) -> RouteDecision:
        rows = self.feasible(feat)
        if not rows:
            # degrade: pick cheapest regardless of SLO
            cheapest = min(self.catalogue, key=lambda s: estimate_request_cost(s, feat.n_tokens_est))
            return RouteDecision(
                model=cheapest.name,
                reason="budget_infeasible_degraded",
                candidates=tuple(s.name for s in self.catalogue),
            )
        prefer = self.budget.prefer
        if prefer == "quality":
            rows.sort(key=lambda r: (-r[0].quality, r[1], r[2]))
        elif prefer == "latency":
            rows.sort(key=lambda r: (r[2], -r[0].quality, r[1]))
        else:
            rows.sort(key=lambda r: (r[1], -r[0].quality, r[2]))
        best = rows[0]
        return RouteDecision(
            model=best[0].name,
            reason=f"slo_{prefer}",
            candidates=tuple(r[0].name for r in rows),
        )

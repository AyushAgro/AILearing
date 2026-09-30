"""Rule-based and policy-table routers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from forge.router.features import RequestFeatures
from forge.router.models import ModelSpec


@dataclass(frozen=True)
class RouteDecision:
    model: str
    reason: str
    candidates: tuple[str, ...] = ()


Rule = Callable[[RequestFeatures], str | None]


@dataclass
class RuleRouter:
    """First matching rule wins; otherwise ``default``."""

    rules: list[tuple[str, Rule]]
    default: str
    catalogue: dict[str, ModelSpec] | None = None

    def route(self, feat: RequestFeatures) -> RouteDecision:
        for name, pred in self.rules:
            hit = pred(feat)
            if hit is not None:
                return RouteDecision(model=hit, reason=name, candidates=(hit,))
        return RouteDecision(model=self.default, reason="default", candidates=(self.default,))


def three_line_router(feat: RequestFeatures) -> str:
    """The pedagogical three-line router from Chapter 2."""
    if feat.task == "code" or feat.n_tokens_est > 4000:
        return "mid-open"
    if feat.task in {"math", "tools", "vision"}:
        return "frontier-api"
    return "tiny-local"


def policy_table_router(
    table: dict[str, str],
    default: str = "mid-open",
) -> RuleRouter:
    """Map task labels to models via an explicit policy table."""

    def by_task(feat: RequestFeatures) -> str | None:
        return table.get(feat.task)

    return RuleRouter(rules=[("policy_table", by_task)], default=default)

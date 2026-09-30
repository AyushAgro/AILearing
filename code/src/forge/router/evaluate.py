"""Offline replay evaluation: cost–quality curves and regret."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forge.router.cost import estimate_request_cost
from forge.router.features import RequestFeatures, extract_features
from forge.router.models import ModelSpec


@dataclass(frozen=True)
class LoggedRequest:
    text: str
    chosen: str
    reward: float  # 1 = success
    cost_usd: float
    features: RequestFeatures | None = None


@dataclass
class OfflineReplay:
    catalogue: dict[str, ModelSpec]
    # optional oracle quality per (text, model) for counterfactual eval
    oracle: dict[tuple[str, str], float] | None = None

    def summarise(
        self,
        log: list[LoggedRequest],
        router_choices: list[str],
    ) -> dict[str, float]:
        assert len(log) == len(router_choices)
        costs, rewards = [], []
        for row, choice in zip(log, router_choices):
            spec = self.catalogue[choice]
            feat = row.features or extract_features(row.text)
            costs.append(estimate_request_cost(spec, feat.n_tokens_est))
            if self.oracle is not None:
                rewards.append(float(self.oracle.get((row.text, choice), 0.0)))
            else:
                rewards.append(row.reward if choice == row.chosen else 0.0)
        return {
            "mean_cost": float(np.mean(costs)),
            "mean_quality": float(np.mean(rewards)),
            "n": float(len(log)),
        }


def cost_quality_curve(
    models: list[str],
    costs: dict[str, float],
    qualities: dict[str, float],
) -> list[tuple[float, float, str]]:
    """Pareto-sorted (cost, quality, name) for always-route-to-X baselines."""
    pts = [(costs[m], qualities[m], m) for m in models]
    pts.sort(key=lambda t: t[0])
    frontier: list[tuple[float, float, str]] = []
    best_q = -1.0
    for c, q, m in pts:
        if q > best_q:
            frontier.append((c, q, m))
            best_q = q
    return frontier


def regret(rewards: list[float], best_possible: list[float]) -> float:
    """Cumulative regret Σ (r* − r)."""
    r = np.asarray(rewards, dtype=float)
    b = np.asarray(best_possible, dtype=float)
    return float(np.sum(b - r))

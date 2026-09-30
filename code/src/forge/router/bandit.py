"""LinUCB contextual bandit router with exploration safety."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forge.router.features import RequestFeatures
from forge.router.rules import RouteDecision


def context_vector(feat: RequestFeatures, dim: int = 6) -> np.ndarray:
    x = np.array(
        [
            1.0 if feat.task == "code" else 0.0,
            1.0 if feat.task == "math" else 0.0,
            1.0 if feat.task == "vision" else 0.0,
            1.0 if feat.task == "tools" else 0.0,
            min(feat.n_tokens_est / 8000.0, 1.0),
            1.0,
        ],
        dtype=float,
    )
    assert len(x) == dim
    return x


@dataclass
class LinUCBRouter:
    """Disjoint LinUCB (Li et al., 2010) over model arms."""

    arms: list[str]
    alpha: float = 0.5
    dim: int = 6
    A: dict[str, np.ndarray] = field(default_factory=dict)
    b: dict[str, np.ndarray] = field(default_factory=dict)
    # exploration safety: never pick arms in denylist; force allowlist if set
    denylist: set[str] = field(default_factory=set)
    allowlist: set[str] | None = None
    min_pulls_before_explore: int = 0
    pulls: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.arms:
            self.A.setdefault(a, np.eye(self.dim))
            self.b.setdefault(a, np.zeros(self.dim))
            self.pulls.setdefault(a, 0)

    def _eligible(self) -> list[str]:
        arms = [a for a in self.arms if a not in self.denylist]
        if self.allowlist is not None:
            arms = [a for a in arms if a in self.allowlist]
        return arms or list(self.arms)

    def route(self, feat: RequestFeatures) -> RouteDecision:
        x = context_vector(feat, self.dim)
        scores: dict[str, float] = {}
        for a in self._eligible():
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            exploit = float(theta @ x)
            explore = float(self.alpha * np.sqrt(x @ A_inv @ x))
            # cold-start: pure explore until min pulls
            if self.pulls[a] < self.min_pulls_before_explore:
                scores[a] = explore + 1e3
            else:
                scores[a] = exploit + explore
        order = tuple(sorted(scores, key=scores.get, reverse=True))  # type: ignore[arg-type]
        return RouteDecision(model=order[0], reason=f"linucb:{scores[order[0]]:.3f}", candidates=order)

    def update(self, feat: RequestFeatures, arm: str, reward: float) -> None:
        x = context_vector(feat, self.dim)
        self.A[arm] = self.A[arm] + np.outer(x, x)
        self.b[arm] = self.b[arm] + reward * x
        self.pulls[arm] = self.pulls.get(arm, 0) + 1

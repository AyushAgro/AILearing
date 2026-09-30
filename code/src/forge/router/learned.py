"""Preference matrices and a tiny logistic classifier router."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forge.router.features import RequestFeatures
from forge.router.rules import RouteDecision


@dataclass
class PreferenceMatrix:
    """Matrix-factorisation style scores: P[prompt_cluster, model]."""

    models: list[str]
    clusters: list[str]
    scores: np.ndarray  # shape (C, M), higher is better

    def best(self, cluster: str) -> RouteDecision:
        if cluster not in self.clusters:
            # unknown cluster -> average across clusters
            row = self.scores.mean(axis=0)
            reason = "mf_mean_unknown_cluster"
        else:
            i = self.clusters.index(cluster)
            row = self.scores[i]
            reason = f"mf_cluster:{cluster}"
        j = int(np.argmax(row))
        order = tuple(self.models[k] for k in np.argsort(-row))
        return RouteDecision(model=self.models[j], reason=reason, candidates=order)


def cluster_for(feat: RequestFeatures) -> str:
    if feat.task in {"code", "math", "vision", "tools"}:
        return feat.task
    if feat.n_tokens_est > 2000:
        return "long"
    return "short"


@dataclass
class ClassifierRouter:
    """Softmax over linear scores w·φ(x) for each model."""

    models: list[str]
    weights: np.ndarray  # shape (M, F)
    bias: np.ndarray  # shape (M,)

    def features(self, feat: RequestFeatures) -> np.ndarray:
        return np.array(
            [
                1.0 if feat.task == "code" else 0.0,
                1.0 if feat.task == "math" else 0.0,
                1.0 if feat.task == "vision" else 0.0,
                1.0 if feat.task == "tools" else 0.0,
                min(feat.n_tokens_est / 8000.0, 1.0),
                1.0 if feat.needs_tools else 0.0,
            ],
            dtype=float,
        )

    def scores(self, feat: RequestFeatures) -> np.ndarray:
        x = self.features(feat)
        logits = self.weights @ x + self.bias
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def route(self, feat: RequestFeatures) -> RouteDecision:
        p = self.scores(feat)
        j = int(np.argmax(p))
        order = tuple(self.models[k] for k in np.argsort(-p))
        return RouteDecision(
            model=self.models[j],
            reason=f"clf_p={p[j]:.3f}",
            candidates=order,
        )

    @staticmethod
    def fit_demo() -> "ClassifierRouter":
        """Hand-tuned weights that prefer frontier for hard tasks, tiny for chat."""
        models = ["tiny-local", "mid-open", "frontier-api"]
        # features: code, math, vision, tools, length, needs_tools
        W = np.array(
            [
                [-1.0, -1.2, -2.0, -2.0, -0.5, -2.0],  # tiny
                [1.2, 0.4, -1.0, 0.2, 0.8, 0.3],  # mid
                [0.6, 1.5, 2.0, 1.8, 0.4, 1.5],  # frontier
            ]
        )
        b = np.array([0.8, 0.2, -0.3])
        return ClassifierRouter(models=models, weights=W, bias=b)

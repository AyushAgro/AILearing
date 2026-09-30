"""Ensemble combiners: voting, RRF, generate-then-rank, MoA-style aggregation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable


def majority_vote(answers: list[str]) -> tuple[str, float]:
    if not answers:
        raise ValueError("empty answers")
    c = Counter(answers)
    winner, n = c.most_common(1)[0]
    return winner, n / len(answers)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF score for candidates appearing in multiple ranked lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


@dataclass
class EnsembleVoter:
    """Generate from several models, then fuse."""

    generators: list[Callable[[str], str]]
    ranker: Callable[[str, list[str]], list[str]] | None = None

    def majority(self, prompt: str) -> tuple[str, float, list[str]]:
        answers = [g(prompt) for g in self.generators]
        winner, conf = majority_vote(answers)
        return winner, conf, answers

    def generate_then_rank(self, prompt: str) -> tuple[str, list[str]]:
        answers = [g(prompt) for g in self.generators]
        if self.ranker is None:
            winner, _ = majority_vote(answers)
            return winner, answers
        ranked = self.ranker(prompt, answers)
        return ranked[0], ranked

    def mixture_of_agents(self, prompt: str, aggregator: Callable[[str, list[str]], str]) -> str:
        """MoA: proposers generate, aggregator synthesises (Wang et al., 2024)."""
        proposals = [g(prompt) for g in self.generators]
        return aggregator(prompt, proposals)

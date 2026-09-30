"""Selective prediction, abstention and human escalation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class AbstainPolicy:
    """Utility framing: answer if u_correct·p - u_wrong·(1-p) - cost > u_abstain."""

    u_correct: float = 1.0
    u_wrong: float = 2.0
    u_abstain: float = 0.0
    escalate_cost: float = 0.3
    escalate_utility: float = 0.85  # expected utility if human resolves
    min_confidence: float = 0.55


@dataclass(frozen=True)
class DecisionOutcome:
    action: Action
    utility: float
    reason: str


def expected_answer_utility(p: float, policy: AbstainPolicy, answer_cost: float = 0.0) -> float:
    return policy.u_correct * p - policy.u_wrong * (1.0 - p) - answer_cost


def decide(
    confidence: float,
    policy: AbstainPolicy,
    *,
    answer_cost: float = 0.0,
    allow_escalate: bool = True,
) -> DecisionOutcome:
    """Choose answer / abstain / escalate by comparing expected utilities."""
    u_ans = expected_answer_utility(confidence, policy, answer_cost)
    u_abs = policy.u_abstain
    u_esc = policy.escalate_utility - policy.escalate_cost if allow_escalate else float("-inf")

    # hard floor: never answer below min_confidence
    if confidence < policy.min_confidence:
        u_ans = float("-inf")

    best = max(
        (u_ans, Action.ANSWER, "answer"),
        (u_abs, Action.ABSTAIN, "abstain"),
        (u_esc, Action.ESCALATE, "escalate"),
        key=lambda t: t[0],
    )
    return DecisionOutcome(action=best[1], utility=best[0], reason=best[2])

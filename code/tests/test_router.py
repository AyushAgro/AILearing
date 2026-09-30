"""Smoke tests for forge.router."""
from forge.router.features import extract_features
from forge.router.rules import three_line_router, policy_table_router
from forge.router.capability import CapabilityProfile, CapabilityRouter
from forge.router.models import DEFAULT_CATALOGUE, TINY
from forge.router.confidence import self_consistency, semantic_entropy, conformal_threshold
from forge.router.decision import AbstainPolicy, decide, Action
from forge.router.fallback import CircuitBreaker, BreakerState
from forge.router.cascade import expected_cascade_cost
from forge.router.ensemble import majority_vote
import numpy as np


def test_three_line_tasks():
    assert three_line_router(extract_features("hello")) == "tiny-local"
    assert three_line_router(extract_features("def f():\n pass")) == "mid-open"


def test_capability_image_only_frontier():
    cap = CapabilityRouter([CapabilityProfile(s) for s in DEFAULT_CATALOGUE])
    d = cap.route(extract_features("OCR this image", modality="image"))
    assert d.model == "frontier-api"
    assert d.candidates == ("frontier-api",)


def test_confidence_and_conformal():
    assert self_consistency(["a", "a", "b"]).value == 2 / 3
    assert semantic_entropy(["Paris", "Paris", "Paris"]).value > 0.99
    q = conformal_threshold(np.linspace(0.4, 0.9, 50), alpha=0.1)
    assert 0.4 <= q <= 0.9


def test_decide_abstain_without_escalate():
    o = decide(0.3, AbstainPolicy(), allow_escalate=False)
    assert o.action is Action.ABSTAIN


def test_breaker_opens():
    br = CircuitBreaker(failure_threshold=2, recovery_timeout_s=10)
    br.record_failure(now=0.0)
    br.record_failure(now=0.0)
    assert br.state is BreakerState.OPEN
    assert br.allow(now=1.0) is False


def test_expected_cascade_cost():
    assert abs(expected_cascade_cost([0.4, 0.25], [0.0, 0.12, 1.5]) - 0.198) < 1e-9


def test_majority_vote():
    w, c = majority_vote(["x", "y", "x"])
    assert w == "x" and abs(c - 2 / 3) < 1e-9


def test_policy_table():
    r = policy_table_router({"chat": "tiny-local", "code": "mid-open"})
    assert r.route(extract_features("hi")).model == "tiny-local"

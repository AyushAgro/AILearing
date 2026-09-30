"""forge.router: capability, cost, confidence, cascade and bandit routers."""

from forge.router.models import CostLatencyModel, ModelSpec, ScriptedModel
from forge.router.features import RequestFeatures, extract_features
from forge.router.rules import RuleRouter, RouteDecision
from forge.router.capability import CapabilityProfile, CapabilityRouter
from forge.router.cost import Budget, CostLatencyRouter
from forge.router.learned import ClassifierRouter, PreferenceMatrix
from forge.router.confidence import (
    CalibrationReport,
    ConfidenceEstimate,
    conformal_threshold,
    estimate_confidence,
    expected_calibration_error,
)
from forge.router.cascade import Cascade, CascadeResult, optimise_threshold
from forge.router.fallback import CircuitBreaker, FallbackChain
from forge.router.ensemble import EnsembleVoter, majority_vote, reciprocal_rank_fusion
from forge.router.bandit import LinUCBRouter
from forge.router.decision import AbstainPolicy, DecisionOutcome, decide
from forge.router.evaluate import OfflineReplay, cost_quality_curve, regret

__all__ = [
    "AbstainPolicy",
    "Budget",
    "CalibrationReport",
    "CapabilityProfile",
    "CapabilityRouter",
    "Cascade",
    "CascadeResult",
    "CircuitBreaker",
    "ClassifierRouter",
    "ConfidenceEstimate",
    "CostLatencyModel",
    "CostLatencyRouter",
    "DecisionOutcome",
    "EnsembleVoter",
    "FallbackChain",
    "LinUCBRouter",
    "ModelSpec",
    "OfflineReplay",
    "PreferenceMatrix",
    "RequestFeatures",
    "RouteDecision",
    "RuleRouter",
    "ScriptedModel",
    "conformal_threshold",
    "cost_quality_curve",
    "decide",
    "estimate_confidence",
    "expected_calibration_error",
    "extract_features",
    "majority_vote",
    "optimise_threshold",
    "reciprocal_rank_fusion",
    "regret",
]

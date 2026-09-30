"""Capability profiles and hard-constraint routing."""
from __future__ import annotations

from dataclasses import dataclass

from forge.router.features import RequestFeatures
from forge.router.models import ModelSpec
from forge.router.rules import RouteDecision


@dataclass(frozen=True)
class CapabilityProfile:
    """What a model can accept; quality is separate."""

    spec: ModelSpec

    def accepts(self, feat: RequestFeatures) -> tuple[bool, str]:
        if feat.modality not in self.spec.modalities:
            return False, f"missing modality {feat.modality}"
        if feat.n_tokens_est > self.spec.context_limit:
            return False, f"context {feat.n_tokens_est} > {self.spec.context_limit}"
        if feat.task not in self.spec.capabilities and feat.task != "chat":
            # chat is always allowed if text modality is present
            if feat.task not in self.spec.capabilities:
                return False, f"missing capability {feat.task}"
        if feat.needs_tools and "tools" not in self.spec.capabilities:
            return False, "missing tools capability"
        return True, "ok"


@dataclass
class CapabilityRouter:
    """Filter by capability, then pick highest quality among survivors."""

    profiles: list[CapabilityProfile]
    fallback: str | None = None

    def eligible(self, feat: RequestFeatures) -> list[tuple[CapabilityProfile, str]]:
        out: list[tuple[CapabilityProfile, str]] = []
        for p in self.profiles:
            ok, why = p.accepts(feat)
            if ok:
                out.append((p, why))
        return out

    def route(self, feat: RequestFeatures) -> RouteDecision:
        elig = self.eligible(feat)
        if not elig:
            if self.fallback:
                return RouteDecision(
                    model=self.fallback,
                    reason="no_eligible_fallback",
                    candidates=(),
                )
            raise ValueError(f"no model accepts request task={feat.task}")
        elig.sort(key=lambda t: t[0].spec.quality, reverse=True)
        best = elig[0][0]
        names = tuple(p.spec.name for p, _ in elig)
        return RouteDecision(model=best.spec.name, reason="max_quality_eligible", candidates=names)

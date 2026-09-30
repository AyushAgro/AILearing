"""Fallbacks, degraded modes and multi-provider circuit breakers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout_s: float = 30.0
    failures: int = 0
    state: BreakerState = BreakerState.CLOSED
    opened_at: float | None = None

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.state is BreakerState.CLOSED:
            return True
        if self.state is BreakerState.OPEN:
            assert self.opened_at is not None
            if now - self.opened_at >= self.recovery_timeout_s:
                self.state = BreakerState.HALF_OPEN
                return True
            return False
        return True  # half-open: one probe

    def record_success(self) -> None:
        self.failures = 0
        self.state = BreakerState.CLOSED
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = now


@dataclass
class ProviderCall:
    provider: str
    ok: bool
    text: str
    error: str | None = None
    degraded: bool = False


@dataclass
class FallbackChain:
    """Try providers in order; skip open breakers; optionally degrade."""

    providers: list[str]
    call_fn: Callable[[str, str], ProviderCall]
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    degraded_text: str = "Service temporarily limited. Here is a cached short answer."

    def __post_init__(self) -> None:
        for p in self.providers:
            self.breakers.setdefault(p, CircuitBreaker())

    def execute(self, prompt: str, *, now: float | None = None) -> ProviderCall:
        errors: list[str] = []
        for name in self.providers:
            br = self.breakers[name]
            if not br.allow(now):
                errors.append(f"{name}:breaker_open")
                continue
            try:
                result = self.call_fn(name, prompt)
            except Exception as exc:  # noqa: BLE001 — teaching boundary
                br.record_failure(now)
                errors.append(f"{name}:{exc}")
                continue
            if result.ok:
                br.record_success()
                return result
            br.record_failure(now)
            errors.append(f"{name}:{result.error or 'fail'}")
        return ProviderCall(
            provider="degraded",
            ok=True,
            text=self.degraded_text,
            error=";".join(errors),
            degraded=True,
        )

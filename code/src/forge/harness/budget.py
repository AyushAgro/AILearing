"""Token, dollar, time and step budgets. Exhaustion is a status, not a crash."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# Illustrative prices for the accounting demo. Not a vendor rate card.
PRICE_INPUT_PER_TOKEN = 0.50 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 1.50 / 1_000_000


@dataclass
class Budget:
    max_tokens: int = 20_000
    max_usd: float = 1.0
    max_seconds: float = 30.0
    max_steps: int = 8


@dataclass
class Ledger:
    budget: Budget
    tokens: int = 0
    usd: float = 0.0
    steps: int = 0
    started: float = field(default_factory=time.perf_counter)
    stop_reason: str | None = None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _would_exceed(self, tokens: int, usd: float, steps: int) -> str | None:
        if self.tokens + tokens > self.budget.max_tokens:
            return "token budget exhausted"
        if self.usd + usd > self.budget.max_usd + 1e-12:
            return "cost budget exhausted"
        if self.steps + steps > self.budget.max_steps:
            return "step budget exhausted"
        if self.elapsed() > self.budget.max_seconds:
            return "time budget exhausted"
        return None

    def charge_model(self, input_tokens: int, output_tokens: int) -> str | None:
        usd = (input_tokens * PRICE_INPUT_PER_TOKEN
               + output_tokens * PRICE_OUTPUT_PER_TOKEN)
        reason = self._would_exceed(input_tokens + output_tokens, usd, 1)
        if reason:
            self.stop_reason = reason
            return reason
        self.tokens += input_tokens + output_tokens
        self.usd += usd
        self.steps += 1
        return None

    def snapshot(self) -> dict:
        return {
            "tokens": self.tokens,
            "usd": round(self.usd, 8),
            "steps": self.steps,
            "stop_reason": self.stop_reason,
        }


class TokenBucket:
    """Steady-state rate limit. Capacity is the burst; refill is tokens per second."""

    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.perf_counter()

    def allow(self, cost: float = 1.0, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else now
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True

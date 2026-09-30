"""Retries, timeouts and idempotency keys for tool side effects."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Attempt:
    ok: bool
    value: object
    error: str
    attempts: int
    replayed: bool


@dataclass
class SideEffectStore:
    """Remember results by idempotency key so a retry does not repeat the effect."""

    done: dict[str, object] = field(default_factory=dict)

    def run(self, key: str, fn: Callable[[], object]) -> tuple[object, bool]:
        if key in self.done:
            return self.done[key], True
        value = fn()
        self.done[key] = value
        return value, False


def call_with_retry(fn: Callable[[], object], *, key: str, store: SideEffectStore,
                    attempts: int = 3, idempotent: bool = False) -> Attempt:
    """Retry `fn` until it stops raising, or attempts run out.

    Non-idempotent calls get a single attempt. A second call with the same key
    returns the stored result and does not invoke `fn` again. Wall-clock
    timeouts belong at the tool or sandbox boundary, where the effect can be
    killed rather than observed after it has already happened.
    """
    if not idempotent and key in store.done:
        return Attempt(True, store.done[key], "", 0, True)
    if not idempotent:
        attempts = 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            value, replayed = store.run(key, fn)
            return Attempt(True, value, "", attempt, replayed)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return Attempt(False, None, last_error, attempts, False)

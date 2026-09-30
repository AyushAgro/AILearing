"""Allow / ask / deny policies. Default is deny. First matching rule wins."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatch


class Effect(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class Rule:
    tool: str                                # shell-style pattern
    effect: Effect
    when: Callable[[dict], bool] | None = None
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    effect: Effect
    rule: str
    reason: str


class PolicyEngine:
    def __init__(self, rules: list[Rule], default: Effect = Effect.DENY):
        self.rules = list(rules)
        self.default = default

    def decide(self, tool: str, args: dict) -> Decision:
        for index, rule in enumerate(self.rules):
            if not fnmatch(tool, rule.tool):
                continue
            if rule.when is not None and not rule.when(args):
                continue
            return Decision(rule.effect, f"rule[{index}]:{rule.tool}", rule.reason)
        return Decision(self.default, "default", "no rule matched")


class CredentialStore:
    """Secrets live here. The model prompt must never receive them."""

    def __init__(self, secrets: dict[str, str]):
        self._secrets = dict(secrets)

    def issue(self, scope: str) -> str | None:
        return self._secrets.get(scope)


class PlanGate:
    """Once frozen, tool names outside the plan are denied.

    This is the plan-then-execute control: untrusted tool output can change
    arguments the plan already allowed, but it cannot introduce a new tool.
    A full capability system (CaMeL) also tracks which values may flow where.
    """

    def __init__(self, plan: list[str]):
        self.plan = list(plan)
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def check(self, tool: str) -> Decision | None:
        if self.frozen and tool not in self.plan:
            return Decision(
                Effect.DENY, "plan-gate",
                f"{tool} is not in the frozen plan {self.plan}",
            )
        return None

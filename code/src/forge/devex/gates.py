"""Ship-with-gates runner: checklist metrics for agent-authored changes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str = ""
    category: str = "check"  # spec | test | review | static | security | pr


@dataclass
class GateRunner:
    """Run ordered gates and compute catch-rate style metrics."""

    results: list[GateResult] = field(default_factory=list)

    def run(self, name: str, category: str, fn: Callable[[], tuple[bool, str]]) -> GateResult:
        ok, detail = fn()
        result = GateResult(name=name, passed=ok, detail=detail, category=category)
        self.results.append(result)
        return result

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> dict[str, object]:
        total = len(self.results)
        failed = [r for r in self.results if not r.passed]
        by_cat: dict[str, list[GateResult]] = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(r)
        catch = {
            cat: sum(1 for r in items if not r.passed) / max(1, len(items))
            for cat, items in by_cat.items()
        }
        return {
            "total": total,
            "passed": total - len(failed),
            "failed": len(failed),
            "all_passed": self.all_passed,
            "fail_rate_by_category": catch,
            "failures": [r.name for r in failed],
        }

    def format(self) -> str:
        lines = []
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            extra = f" — {r.detail}" if r.detail else ""
            lines.append(f"[{mark}] ({r.category}) {r.name}{extra}")
        s = self.summary()
        lines.append(
            f"summary: {s['passed']}/{s['total']} passed; "
            f"fail_rate_by_category={s['fail_rate_by_category']}"
        )
        return "\n".join(lines)


def default_gates(
    *,
    has_spec: bool,
    tests_green: bool,
    human_reviewed: bool,
    lint_clean: bool,
    no_secret_diff: bool,
    deps_allowed: bool,
    pr_scoped: bool,
) -> GateRunner:
    """Build the Volume 26 ship gate from boolean evidence (toy / pedagogical)."""
    runner = GateRunner()
    runner.run("spec written", "spec", lambda: (has_spec, "acceptance criteria present" if has_spec else "missing"))
    runner.run("tests green", "test", lambda: (tests_green, "pytest -q" if tests_green else "failing"))
    runner.run("human review", "review", lambda: (human_reviewed, "checklist done" if human_reviewed else "pending"))
    runner.run("static analysis", "static", lambda: (lint_clean, "ruff+mypy" if lint_clean else "issues"))
    runner.run("no secrets in diff", "security", lambda: (no_secret_diff, "clean" if no_secret_diff else "secret-like token"))
    runner.run("deps allow-listed", "security", lambda: (deps_allowed, "ok" if deps_allowed else "unknown package"))
    runner.run("PR scoped", "pr", lambda: (pr_scoped, "one intent" if pr_scoped else "drive-by edits"))
    return runner

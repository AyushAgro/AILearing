"""Toy metrics over an agent session transcript (files, tests, gate failures)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptMetrics:
    files_touched: set[str] = field(default_factory=set)
    tests_run: int = 0
    tests_failed: int = 0
    gate_failures: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    secret_warnings: int = 0

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "edit":
            path = event.get("path")
            if path:
                self.files_touched.add(str(path))
        elif kind == "test":
            self.tests_run += int(event.get("run", 1))
            self.tests_failed += int(event.get("failed", 0))
        elif kind == "gate_fail":
            name = event.get("name", "unknown")
            self.gate_failures.append(str(name))
        elif kind == "command":
            cmd = str(event.get("cmd", ""))
            self.commands.append(cmd)
            lower = cmd.lower()
            if any(tok in lower for tok in ("cat .env", "printenv", "echo $api")):
                self.secret_warnings += 1

    def summary(self) -> dict[str, object]:
        return {
            "files_touched": sorted(self.files_touched),
            "n_files": len(self.files_touched),
            "tests_run": self.tests_run,
            "tests_failed": self.tests_failed,
            "gate_failures": list(self.gate_failures),
            "n_gate_failures": len(self.gate_failures),
            "commands": len(self.commands),
            "secret_warnings": self.secret_warnings,
        }

    def format(self) -> str:
        s = self.summary()
        return (
            f"files={s['n_files']} {s['files_touched']}; "
            f"tests_run={s['tests_run']} failed={s['tests_failed']}; "
            f"gate_failures={s['n_gate_failures']} {s['gate_failures']}; "
            f"secret_warnings={s['secret_warnings']}"
        )


def score_transcript(events: list[dict[str, Any]]) -> TranscriptMetrics:
    metrics = TranscriptMetrics()
    for event in events:
        metrics.observe(event)
    return metrics

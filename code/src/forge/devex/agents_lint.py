"""Lint AGENTS.md for required sections and dangerous content."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_HEADINGS = (
    "## Commands",
    "## Layout",
    "## Rules",
)
SECRET_PATTERNS = (
    (re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+"), "possible secret assignment"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai-like key shape"),
    (re.compile(r"(?i)BEGIN (RSA |OPENSSH )?PRIVATE KEY"), "private key material"),
)
# Imperative encouragement only — "do not disable CI" is allowed and expected.
def _encourages_bad_practice(line: str) -> str | None:
    lower = line.strip().lower()
    if lower.startswith(("#", "-", "*")):
        lower = lower.lstrip("#-* ").strip()
    for prefix in ("do not ", "don't ", "never ", "must not ", "do not ever "):
        if lower.startswith(prefix):
            return None
    if "disable ci" in lower:
        return "must not tell agents to disable CI"
    if "skip tests" in lower:
        return "must not tell agents to skip tests"
    if "paste" in lower and ".env" in lower:
        return "must not encourage pasting .env into prompts"
    return None


@dataclass(frozen=True)
class LintFinding:
    severity: str  # error | warning
    message: str
    line: int | None = None


def lint_agents_md(text: str | Path) -> list[LintFinding]:
    """Return findings for an AGENTS.md body or path."""
    if isinstance(text, Path):
        body = text.read_text(encoding="utf-8")
    else:
        body = text
    findings: list[LintFinding] = []
    lower = body.lower()

    for heading in REQUIRED_HEADINGS:
        if heading.lower() not in lower:
            findings.append(LintFinding("error", f"missing required section: {heading}"))

    if "pytest" not in lower and "test" not in lower:
        findings.append(LintFinding("warning", "Commands section should mention how to run tests"))

    lines = body.splitlines()
    for i, line in enumerate(lines, start=1):
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(LintFinding("error", f"{label} on line {i}", i))
        label = _encourages_bad_practice(line)
        if label:
            findings.append(LintFinding("error", f"{label} (line {i})", i))

    if len(body) > 12_000:
        findings.append(LintFinding("warning", f"file is large ({len(body)} chars); prefer a short always-on file"))

    return findings


def format_findings(findings: list[LintFinding]) -> str:
    if not findings:
        return "AGENTS.md lint: OK"
    lines = [f"AGENTS.md lint: {len(findings)} finding(s)"]
    for f in findings:
        loc = f" L{f.line}" if f.line else ""
        lines.append(f"  [{f.severity}]{loc} {f.message}")
    return "\n".join(lines)

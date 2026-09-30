"""Tests for forge.devex."""
from __future__ import annotations

import json
from pathlib import Path

from forge.devex.agents_lint import lint_agents_md
from forge.devex.gates import default_gates
from forge.devex.mcp_allowlist import check_mcp_allowlist
from forge.devex.scaffold import AGENTS_MD, scaffold_workspace
from forge.devex.session import Workflow, check_session
from forge.devex.transcript import score_transcript


def test_scaffold_writes_canonical_files(tmp_path: Path) -> None:
    created = scaffold_workspace(tmp_path)
    rels = {p.relative_to(tmp_path).as_posix() for p in created}
    assert "AGENTS.md" in rels
    assert "CLAUDE.md" in rels
    assert ".cursor/rules/python-forge.mdc" in rels
    assert ".cursor/mcp.json" in rels
    assert ".agents/skills/ship-gate/SKILL.md" in rels


def test_lint_accepts_canonical_agents_md() -> None:
    assert lint_agents_md(AGENTS_MD) == []


def test_lint_rejects_secrets_and_missing_sections() -> None:
    bad = "# hi\nAPI_KEY=sk-abcdefghijklmnopqrstuvwxyz\nPlease disable CI always\n"
    findings = lint_agents_md(bad)
    assert any(f.severity == "error" for f in findings)
    assert any("missing required section" in f.message for f in findings)
    assert any("disable CI" in f.message for f in findings)


def test_lint_allows_do_not_disable_ci() -> None:
    body = AGENTS_MD  # contains "Do not disable CI gates"
    assert not any("disable CI" in f.message for f in lint_agents_md(body))


def test_mcp_allowlist() -> None:
    ok = {"mcpServers": {"forge-docs": {"command": "python", "args": ["-m", "x"], "env": {}}}}
    assert check_mcp_allowlist(ok) == []
    bad = {"mcpServers": {"evil": {"command": "npx", "env": {"TOKEN": "sk-abc"}}}}
    findings = check_mcp_allowlist(bad)
    assert len(findings) >= 2


def test_gates_catch_bad_deps() -> None:
    runner = default_gates(
        has_spec=True,
        tests_green=True,
        human_reviewed=True,
        lint_clean=True,
        no_secret_diff=True,
        deps_allowed=False,
        pr_scoped=True,
    )
    assert not runner.all_passed
    assert "deps allow-listed" in runner.summary()["failures"]


def test_session_workflows() -> None:
    assert check_session(Workflow.SHIP).workflow == Workflow.SHIP
    assert "Do not merge" in " ".join(check_session("learning").hard_stops)


def test_transcript_metrics() -> None:
    events = [
        {"kind": "edit", "path": "src/forge/devex/gates.py"},
        {"kind": "edit", "path": "tests/test_devex.py"},
        {"kind": "test", "run": 7, "failed": 1},
        {"kind": "gate_fail", "name": "deps allow-listed"},
        {"kind": "command", "cmd": "cat .env"},
    ]
    m = score_transcript(events)
    assert m.summary()["n_files"] == 2
    assert m.tests_failed == 1
    assert m.secret_warnings == 1
    assert m.gate_failures == ["deps allow-listed"]


def test_mcp_roundtrip_file(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"forge-docs": {"command": "python"}}}), encoding="utf-8")
    assert check_mcp_allowlist(path) == []

"""Canonical coding-agent workspace layout and file templates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


AGENTS_MD = """# AGENTS.md

## Commands
- Install: `uv sync`
- Tests: `pytest -q`
- Lint: `ruff check .`
- Types: `mypy src`
- Security: `bandit -r src`

## Layout
- Package code: `src/forge/`
- Tests: `tests/` mirroring package paths
- Do not add top-level utility modules; extend an existing package

## Rules
- Prefer modifying existing modules over new frameworks
- Every public function needs a test or a documented reason
- No new dependencies without licence note in the PR
- Never read or write `.env` secrets into prompts or logs
- Never paste production keys, customer PII or proprietary customer code into chat

## Out of scope
- Do not redesign serving without an ADR
- Do not disable CI gates
"""

CLAUDE_MD = """# CLAUDE.md
# Canonical agent instructions live in AGENTS.md (same repo root).
# Read and follow AGENTS.md for commands, layout and engineering rules.

See @AGENTS.md
"""

CURSOR_RULE = """---
description: Forge Python package conventions
globs: src/forge/**/*.py
alwaysApply: false
---
- Use existing forge.* packages; do not invent parallel utilities
- Public functions need type hints and tests under tests/
- Prefer explicit errors over silent None returns
- Run: `pytest -q` from the repo root before claiming done
- Never commit secrets or paste `.env` contents into chat
"""

MCP_JSON = """{
  "mcpServers": {
    "forge-docs": {
      "command": "python",
      "args": ["-m", "forge.harness.mcp_server"],
      "env": {}
    }
  }
}
"""

SKILL_MD = """# Skill: ship-gate

## When to use
Before opening a PR that includes agent-authored code.

## Steps
1. Confirm AGENTS.md commands still match CI.
2. Run lint, types, tests and bandit.
3. Reject any new dependency not on the allow-list.
4. Human review of security-sensitive paths (auth, crypto, OS, SQL).
"""

GITIGNORE_SNIPPET = """
# Agent / IDE local state (do not commit secrets)
.env
.env.*
!.env.example
.cursor/mcp.local.json
"""


@dataclass(frozen=True)
class WorkspaceSpec:
    """What the scaffold writes. Paths are relative to the target root."""

    include_claude: bool = True
    include_cursor: bool = True
    include_mcp: bool = True
    include_skills: bool = True
    extra_files: dict[str, str] = field(default_factory=dict)


def scaffold_workspace(root: Path, spec: WorkspaceSpec | None = None) -> list[Path]:
    """Write the canonical coding-agent workspace module. Returns created paths."""
    spec = spec or WorkspaceSpec()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def write(rel: str, content: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.lstrip("\n") if content.startswith("\n") else content, encoding="utf-8")
        created.append(path)

    write("AGENTS.md", AGENTS_MD)
    if spec.include_claude:
        write("CLAUDE.md", CLAUDE_MD)
    if spec.include_cursor:
        write(".cursor/rules/python-forge.mdc", CURSOR_RULE)
    if spec.include_mcp:
        write(".cursor/mcp.json", MCP_JSON)
        write(".mcp.json", MCP_JSON)
    if spec.include_skills:
        write(".agents/skills/ship-gate/SKILL.md", SKILL_MD)
    write(".env.example", "# EXAMPLE ONLY — never put real secrets here\nAPI_KEY=\n")
    for rel, content in spec.extra_files.items():
        write(rel, content)
    return created

"""CLI: python -m forge.devex <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.devex.agents_lint import format_findings, lint_agents_md
from forge.devex.gates import default_gates
from forge.devex.mcp_allowlist import check_mcp_allowlist, format_mcp_findings
from forge.devex.scaffold import WorkspaceSpec, scaffold_workspace
from forge.devex.session import check_session, format_checklist
from forge.devex.transcript import score_transcript


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forge.devex", description="Coding-agent workspace tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sc = sub.add_parser("scaffold", help="Write AGENTS.md / CLAUDE.md / rules / MCP stubs")
    p_sc.add_argument("root", type=Path, help="Target directory")

    p_lint = sub.add_parser("lint-agents", help="Lint an AGENTS.md file")
    p_lint.add_argument("path", type=Path)

    p_mcp = sub.add_parser("check-mcp", help="Check MCP config against allow-list")
    p_mcp.add_argument("path", type=Path)

    p_sess = sub.add_parser("session", help="Print workflow checklist")
    p_sess.add_argument("workflow", choices=["learning", "vibe", "ship"])

    p_gate = sub.add_parser("demo-gates", help="Run pedagogical gate demo")
    p_gate.add_argument("--fail-deps", action="store_true")

    p_tr = sub.add_parser("score-transcript", help="Score a JSONL transcript")
    p_tr.add_argument("path", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "scaffold":
        paths = scaffold_workspace(args.root, WorkspaceSpec())
        for p in paths:
            print(p)
        return 0

    if args.cmd == "lint-agents":
        findings = lint_agents_md(args.path)
        print(format_findings(findings))
        return 1 if any(f.severity == "error" for f in findings) else 0

    if args.cmd == "check-mcp":
        findings = check_mcp_allowlist(args.path)
        print(format_mcp_findings(findings))
        return 1 if any(f.severity == "error" for f in findings) else 0

    if args.cmd == "session":
        print(format_checklist(check_session(args.workflow)))
        return 0

    if args.cmd == "demo-gates":
        runner = default_gates(
            has_spec=True,
            tests_green=True,
            human_reviewed=True,
            lint_clean=True,
            no_secret_diff=True,
            deps_allowed=not args.fail_deps,
            pr_scoped=True,
        )
        print(runner.format())
        return 0 if runner.all_passed else 1

    if args.cmd == "score-transcript":
        events = []
        for line in args.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        print(score_transcript(events).format())
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())

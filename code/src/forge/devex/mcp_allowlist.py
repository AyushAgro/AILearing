"""Check MCP config against an allow-list of server names and commands."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class McpFinding:
    severity: str
    message: str
    server: str | None = None


DEFAULT_ALLOWLIST = frozenset({
    "forge-docs",
    "filesystem",
    "git",
    "sqlite",
})


def _load(path_or_obj: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    if isinstance(path_or_obj, Path):
        return json.loads(path_or_obj.read_text(encoding="utf-8"))
    return json.loads(path_or_obj)


def check_mcp_allowlist(
    config: Path | str | dict[str, Any],
    allowlist: frozenset[str] | set[str] | None = None,
    *,
    forbid_env_secrets: bool = True,
) -> list[McpFinding]:
    """Validate mcpServers entries. Unknown names are errors; secret-like env values too."""
    allow = frozenset(allowlist) if allowlist is not None else DEFAULT_ALLOWLIST
    data = _load(config)
    servers = data.get("mcpServers") or data.get("servers") or {}
    findings: list[McpFinding] = []

    if not isinstance(servers, dict):
        return [McpFinding("error", "mcpServers must be an object")]

    for name, spec in servers.items():
        if name not in allow:
            findings.append(McpFinding(
                "error",
                f"server '{name}' not on allow-list {sorted(allow)}",
                name,
            ))
        if not isinstance(spec, dict):
            findings.append(McpFinding("error", "server spec must be an object", name))
            continue
        cmd = spec.get("command")
        if not cmd:
            findings.append(McpFinding("warning", "missing command", name))
        env = spec.get("env") or {}
        if forbid_env_secrets and isinstance(env, dict):
            for key, value in env.items():
                blob = f"{key}={value}"
                if any(tok in blob.lower() for tok in ("sk-", "password", "secret", "token=")):
                    findings.append(McpFinding(
                        "error",
                        f"refusing secret-like env for '{name}': {key}",
                        name,
                    ))
    return findings


def format_mcp_findings(findings: list[McpFinding]) -> str:
    if not findings:
        return "MCP allow-list: OK"
    lines = [f"MCP allow-list: {len(findings)} finding(s)"]
    for f in findings:
        who = f" ({f.server})" if f.server else ""
        lines.append(f"  [{f.severity}]{who} {f.message}")
    return "\n".join(lines)

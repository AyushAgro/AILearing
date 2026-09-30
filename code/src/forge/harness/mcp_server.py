"""Minimal MCP-shaped JSON-RPC server over stdin/stdout.

Speaks initialize, tools/list and tools/call. This is a teaching subset of the
Model Context Protocol, not a complete implementation of the specification.
"""
from __future__ import annotations

import json
import sys

NOTES = {
    "refund": "Refunds are accepted within 30 days of delivery.",
    "shipping": "Standard shipping takes 3 to 5 business days.",
}


def lookup_note(name: str) -> str:
    if name not in NOTES:
        raise KeyError(name)
    return NOTES[name]


def handle(request: dict) -> dict:
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "forge-notes"}}
    elif method == "tools/list":
        result = {"tools": [{
            "name": "lookup_note",
            "description": "Read a local policy note by name.",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }]}
    elif method == "tools/call":
        if params.get("name") != "lookup_note":
            raise ValueError(f"unknown tool {params.get('name')}")
        text = lookup_note(str(params.get("arguments", {}).get("name", "")))
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    else:
        return {"jsonrpc": "2.0", "id": request.get("id"),
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = handle(json.loads(line))
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32000, "message": str(exc)}}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

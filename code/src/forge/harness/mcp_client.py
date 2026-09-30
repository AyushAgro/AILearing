"""Client for the teaching MCP server. The server process is untrusted."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


class McpClient:
    def __init__(self, server_path: Path):
        self.proc = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self._next_id = 1

    def request(self, method: str, params: dict | None = None) -> dict:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("mcp pipes are closed")
        message = {"jsonrpc": "2.0", "id": self._next_id, "method": method,
                   "params": params or {}}
        self._next_id += 1
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("mcp server closed stdout")
        return json.loads(line)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=2)

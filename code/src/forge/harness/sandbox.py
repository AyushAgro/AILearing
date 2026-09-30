"""Two isolation levels: a subprocess with limits, and a container specification.

The subprocess is a real process boundary with CPU, file-size and wall-clock
limits. It is not a security sandbox: the child keeps the parent's user,
filesystem and network. The container specification is what you run when the
code is untrusted. gVisor and Firecracker are stronger levels again; this
module does not pretend to implement them.
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    except (ValueError, OSError):
        pass


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    seconds: float
    returncode: int | None
    level: str


def run_subprocess(code: str, *, timeout: float = 3.0, max_out: int = 500) -> SandboxResult:
    """Run `code` in a child interpreter. cwd is an empty temporary directory."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as cwd:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
                capture_output=True, text=True,
                timeout=timeout, preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                False, "", f"TimeoutExpired after {timeout}s",
                time.perf_counter() - started, None, "subprocess",
            )
    stdout, stderr = _trim(completed.stdout, max_out), _trim(completed.stderr, max_out)
    return SandboxResult(
        completed.returncode == 0, stdout, stderr,
        time.perf_counter() - started, completed.returncode, "subprocess",
    )


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated {len(text)} chars]"


def container_argv(image: str, workdir: str, code: str) -> list[str]:
    """The production invocation. Flags are the policy; the image is the rootfs."""
    return [
        "docker", "run", "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--cpus=0.5",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        "-v", f"{workdir}:/work:rw",
        "-w", "/work",
        image,
        "python", "-I", "-c", code,
    ]


def container_spec_ok(argv: list[str]) -> list[str]:
    """Return the required flags that are missing. Empty means the spec holds."""
    required = [
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=256m",
    ]
    return [flag for flag in required if flag not in argv]


def probe_filesystem(home: str) -> dict:
    """Show what a subprocess child can still reach. Prints no directory entries."""
    code = (
        "import json, os, socket\n"
        f"home = {home!r}\n"
        "s = socket.socket(); s.close()\n"
        "print(json.dumps({\n"
        "  'outside_workspace_readable': os.path.isdir(home),\n"
        "  'socket_created': True,\n"
        "}))\n"
    )
    result = run_subprocess(code)
    if not result.ok:
        return {"ok": False, "stderr": result.stderr}
    return json.loads(result.stdout)

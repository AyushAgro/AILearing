"""Tool registry: schema check, discovery, serial and parallel invocation."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


def validate(schema: dict, args: dict) -> str | None:
    """Minimal JSON-Schema subset: required keys, no extras, string/integer types."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    props: dict = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in props:
            return f"unexpected argument '{key}'"
        expected = props[key].get("type")
        if expected == "string" and not isinstance(value, str):
            return f"'{key}' must be a string"
        if expected == "integer" and not isinstance(value, int):
            return f"'{key}' must be an integer"
    return None


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., object]
    description: str
    schema: dict
    mutating: bool = False
    idempotent: bool = True
    credential_scope: str | None = None
    streams: bool = False


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""
    seconds: float = 0.0
    streamed_chunks: int = 0


@dataclass
class ToolRuntime:
    specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"tool {spec.name} already registered")
        self.specs[spec.name] = spec

    def discover(self) -> list[dict]:
        found = []
        for spec in self.specs.values():
            found.append({
                "name": spec.name,
                "description": spec.description,
                "schema": spec.schema,
                "mutating": spec.mutating,
            })
        return found

    def invoke(self, name: str, args: dict, *, credential: str | None = None,
               timeout: float = 5.0) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return ToolResult(False, "", f"unknown tool '{name}'")
        problem = validate(spec.schema, args)
        if problem:
            return ToolResult(False, "", problem)
        started = time.perf_counter()
        try:
            if spec.streams:
                chunks = list(self._stream(spec, args, credential))
                output = "".join(chunks)
                return ToolResult(True, output, "", time.perf_counter() - started,
                                  streamed_chunks=len(chunks))
            kwargs = dict(args)
            if spec.credential_scope:
                kwargs["_credential"] = credential
            output = spec.fn(**kwargs)
            return ToolResult(True, str(output), "", time.perf_counter() - started)
        except TimeoutError as exc:
            return ToolResult(False, "", str(exc), time.perf_counter() - started)
        except Exception as exc:  # tool bugs become observations, not harness crashes
            return ToolResult(False, "", f"{type(exc).__name__}: {exc}",
                              time.perf_counter() - started)

    def _stream(self, spec: ToolSpec, args: dict,
                credential: str | None) -> Iterator[str]:
        kwargs = dict(args)
        if spec.credential_scope:
            kwargs["_credential"] = credential
        yield from spec.fn(**kwargs)

    def invoke_parallel(self, calls: list[tuple[str, dict]]) -> list[ToolResult]:
        for name, _args in calls:
            spec = self.specs.get(name)
            if spec is None:
                return [ToolResult(False, "", f"unknown tool '{name}'")]
            if spec.mutating:
                return [ToolResult(
                    False, "",
                    f"refusing parallel batch: '{name}' is mutating",
                )]
        with ThreadPoolExecutor(max_workers=len(calls) or 1) as pool:
            futures = [pool.submit(self.invoke, name, args) for name, args in calls]
            return [future.result() for future in futures]

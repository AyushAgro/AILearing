"""Spans shaped like the OpenTelemetry GenAI semantic conventions.

Attribute names follow the development conventions (operation, provider, usage,
tool). Production exports these through the OpenTelemetry SDK; this tracer
keeps them in memory so tests need no collector. Names are not stable forever:
the GenAI conventions are still marked Development.
"""
from __future__ import annotations

import itertools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterator


def _hex(counter: int, width: int) -> str:
    return f"{counter:0{width}x}"


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    attributes: dict
    start: float
    end: float | None = None

    def finish(self) -> None:
        self.end = time.perf_counter()

    def as_dict(self) -> dict:
        duration_ms = None if self.end is None else round((self.end - self.start) * 1000, 3)
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "attributes": self.attributes,
            "duration_ms": duration_ms,
        }


@dataclass
class Tracer:
    agent_name: str
    trace_id: str = "4bf92f3577b34da6a3ce929d0e0e4736"
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))
    spans: list[Span] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)

    def _span_id(self) -> str:
        return _hex(next(self._ids), 16)

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        current = Span(name, self.trace_id, self._span_id(), parent, dict(attributes),
                       time.perf_counter())
        self.spans.append(current)
        self._stack.append(current.span_id)
        try:
            yield current
        finally:
            current.finish()
            self._stack.pop()

    def export(self) -> list[dict]:
        return [span.as_dict() for span in self.spans]

    def usage_tokens(self) -> int:
        total = 0
        for span in self.spans:
            attrs = span.attributes
            total += int(attrs.get("gen_ai.usage.input_tokens", 0))
            total += int(attrs.get("gen_ai.usage.output_tokens", 0))
        return total


@dataclass
class Cassette:
    """Recorded tool results for replay. Divergence is a failed assertion."""

    entries: dict[tuple, str] = field(default_factory=dict)

    def record(self, tool: str, args: dict, output: str) -> None:
        self.entries[(tool, _freeze(args))] = output

    def replay(self, tool: str, args: dict) -> str:
        key = (tool, _freeze(args))
        if key not in self.entries:
            raise KeyError(f"no cassette entry for {tool} {args}")
        return self.entries[key]


def _freeze(args: dict) -> tuple:
    return tuple(sorted((key, repr(value)) for key, value in args.items()))

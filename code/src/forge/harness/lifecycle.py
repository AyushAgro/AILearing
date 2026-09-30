"""Session, turn and step lifecycle with an append-only event log."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Phase(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


# Legal edges only. Anything else is a programming error, not a model choice.
TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.CREATED: {Phase.RUNNING, Phase.CANCELLED},
    Phase.RUNNING: {
        Phase.PAUSED, Phase.AWAITING_APPROVAL, Phase.COMPLETED,
        Phase.FAILED, Phase.CANCELLED,
    },
    Phase.PAUSED: {Phase.RUNNING, Phase.CANCELLED, Phase.FAILED},
    Phase.AWAITING_APPROVAL: {Phase.RUNNING, Phase.CANCELLED, Phase.FAILED},
    Phase.CANCELLED: set(),
    Phase.COMPLETED: set(),
    Phase.FAILED: set(),
}

TERMINAL = {Phase.CANCELLED, Phase.COMPLETED, Phase.FAILED}


@dataclass
class Session:
    session_id: str
    goal: str
    phase: Phase = Phase.CREATED
    turn: int = 1
    step: int = 0
    events: list[dict] = field(default_factory=list)

    def transition(self, to: Phase, **detail: object) -> None:
        allowed = TRANSITIONS[self.phase]
        if to not in allowed:
            raise ValueError(f"illegal transition {self.phase} -> {to}")
        self.phase = to
        self.event("phase", to=to.value, **detail)

    def event(self, kind: str, **detail: object) -> dict:
        record = {
            "session_id": self.session_id,
            "turn": self.turn,
            "step": self.step,
            "kind": kind,
            **detail,
        }
        self.events.append(record)
        return record

    def next_step(self) -> int:
        self.step += 1
        return self.step


class SessionStore:
    """One JSON object per line. Replay is 'read the file', not a side query."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

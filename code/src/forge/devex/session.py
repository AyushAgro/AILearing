"""Pre-session checklist for learning / vibe / ship workflows."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Workflow(StrEnum):
    LEARNING = "learning"
    VIBE = "vibe"
    SHIP = "ship"


@dataclass(frozen=True)
class SessionChecklist:
    workflow: Workflow
    items: tuple[str, ...]
    hard_stops: tuple[str, ...]


LEARNING = SessionChecklist(
    workflow=Workflow.LEARNING,
    items=(
        "State the concept you want to understand in one sentence",
        "Ask for a Socratic explanation, not a paste-ready solution",
        "Run a tiny experiment yourself before accepting an answer",
        "Write one note and one test that encode what you learned",
        "Mark Documented vs Inferred when quoting vendor or paper claims",
    ),
    hard_stops=(
        "Do not merge learning-session code to main",
        "Do not paste production secrets or customer code",
    ),
)

VIBE = SessionChecklist(
    workflow=Workflow.VIBE,
    items=(
        "Create a throwaway branch with a time-box (e.g. 60–90 minutes)",
        "Write a one-line goal; refuse scope expansion mid-session",
        "Prefer disposable prototypes over production paths",
        "Capture what worked in a note; delete or quarantine the branch",
        "Stop when the timer ends — decide ship-gate or discard",
    ),
    hard_stops=(
        "Do not open a production PR from a vibe session without Workflow C",
        "Do not disable CI to keep the vibe going",
        "Do not paste production secrets or customer code",
    ),
)

SHIP = SessionChecklist(
    workflow=Workflow.SHIP,
    items=(
        "Write acceptance criteria and failing tests first",
        "Bound the agent to specific files / packages",
        "Run the ship gate (lint, types, tests, security, deps)",
        "Human review: scope, security paths, hallucinated APIs",
        "Open a small PR; record gate failures caught before merge",
    ),
    hard_stops=(
        "Do not merge without human review",
        "Do not add dependencies without allow-list check",
        "Do not paste production secrets or customer code",
    ),
)

BY_WORKFLOW = {
    Workflow.LEARNING: LEARNING,
    Workflow.VIBE: VIBE,
    Workflow.SHIP: SHIP,
}


def check_session(workflow: Workflow | str) -> SessionChecklist:
    wf = Workflow(workflow) if not isinstance(workflow, Workflow) else workflow
    return BY_WORKFLOW[wf]


def format_checklist(checklist: SessionChecklist) -> str:
    lines = [f"workflow={checklist.workflow.value}", "items:"]
    for i, item in enumerate(checklist.items, 1):
        lines.append(f"  {i}. {item}")
    lines.append("hard_stops:")
    for stop in checklist.hard_stops:
        lines.append(f"  - {stop}")
    return "\n".join(lines)

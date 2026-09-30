"""Deterministic stand-in for a chat model. No network, no weights."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ScriptedModel:
    """Returns pre-written actions, recording every prompt it was shown."""

    actions: list[dict]
    prompts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.actions = list(self.actions)

    def complete(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        if not self.actions:
            return {"final": "stopped: model has no further scripted actions"}
        return self.actions.pop(0)

    @staticmethod
    def dumps(action: dict) -> str:
        return json.dumps(action, sort_keys=True)

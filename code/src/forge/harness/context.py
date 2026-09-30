"""Token accounting, tool-result truncation and context compaction."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Approximate token count: 4 characters per token, minimum 1.

    Production code must call the model tokeniser. This estimate is only good
    enough to force compaction in tests and to show the accounting shape.
    """
    return max(1, (len(text) + 3) // 4)


@dataclass
class Message:
    role: str
    content: str
    kind: str = "text"          # text | tool | summary
    tokens: int = 0

    def __post_init__(self) -> None:
        self.tokens = estimate_tokens(self.content)


@dataclass
class ContextManager:
    max_tokens: int
    tool_cap_tokens: int = 80
    messages: list[Message] = field(default_factory=list)
    truncations: int = 0
    compactions: int = 0

    def add(self, role: str, content: str, kind: str = "text") -> Message:
        if kind == "tool":
            content, truncated = self._cap_tool(content)
            if truncated:
                self.truncations += 1
        message = Message(role, content, kind)
        self.messages.append(message)
        return message

    def _cap_tool(self, content: str) -> tuple[str, bool]:
        if estimate_tokens(content) <= self.tool_cap_tokens:
            return content, False
        # Keep a prefix the model can quote, and say the rest was removed.
        keep_chars = self.tool_cap_tokens * 4
        note = f"\n[truncated to {self.tool_cap_tokens} tokens]"
        return content[:keep_chars] + note, True

    def tokens(self) -> int:
        return sum(message.tokens for message in self.messages)

    def render(self) -> str:
        lines = [f"{message.role}: {message.content}" for message in self.messages]
        return "\n".join(lines)

    def needs_compaction(self) -> bool:
        return self.tokens() > self.max_tokens

    def compact(self, summarise: Callable[[str], str]) -> bool:
        """Replace middle tool results with one summary. Keep the edges.

        The first message (system) and the last two messages stay verbatim.
        If that is the whole transcript, compaction refuses rather than
        dropping the system prompt or the latest observation.
        """
        if len(self.messages) < 4:
            return False
        head, middle, tail = self.messages[:1], self.messages[1:-2], self.messages[-2:]
        toolish = [m for m in middle if m.kind in {"tool", "summary"}]
        if not toolish:
            return False
        body = "\n".join(f"{m.role}: {m.content}" for m in toolish)
        summary = summarise(body)
        kept = [m for m in middle if m.kind not in {"tool", "summary"}]
        self.messages = head + kept + [Message("system", summary, "summary")] + tail
        self.compactions += 1
        return True

    def assert_system_intact(self, system_prefix: str) -> None:
        if not self.messages or not self.messages[0].content.startswith(system_prefix):
            raise RuntimeError("compaction dropped or rewrote the system prompt")

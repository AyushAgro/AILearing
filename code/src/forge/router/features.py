"""Request features used by every router in this package."""
from __future__ import annotations

import re
from dataclasses import dataclass


_CODE_RE = re.compile(r"```|def |class |import |SELECT |function ", re.I)
_MATH_RE = re.compile(r"\b(integral|derivative|prove|equation|\\\\frac)\b|[=≈≤≥]", re.I)
_VISION_RE = re.compile(r"\b(image|screenshot|photo|diagram|ocr)\b", re.I)


@dataclass(frozen=True)
class RequestFeatures:
    text: str
    n_tokens_est: int
    task: str  # chat | code | math | vision | tools
    needs_tools: bool
    modality: str  # text | image
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    tenant: str = "default"


def estimate_tokens(text: str) -> int:
    """Rough English/code token estimate: ~4 chars per token."""
    return max(1, (len(text) + 3) // 4)


def classify_task(text: str) -> str:
    if _VISION_RE.search(text):
        return "vision"
    if _CODE_RE.search(text):
        return "code"
    if _MATH_RE.search(text):
        return "math"
    if "tool:" in text.lower() or "call the" in text.lower():
        return "tools"
    return "chat"


def extract_features(
    text: str,
    *,
    max_latency_ms: float | None = None,
    max_cost_usd: float | None = None,
    tenant: str = "default",
    modality: str | None = None,
) -> RequestFeatures:
    task = classify_task(text)
    mod = modality or ("image" if task == "vision" else "text")
    return RequestFeatures(
        text=text,
        n_tokens_est=estimate_tokens(text),
        task=task,
        needs_tools=task == "tools",
        modality=mod,
        max_latency_ms=max_latency_ms,
        max_cost_usd=max_cost_usd,
        tenant=tenant,
    )

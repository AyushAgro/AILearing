"""Human approval, diff review and an append-only audit log.

Approval state lives outside the model transcript. A tool call in the ASK
state does not run until a person decides. A deadline with no decision denies.
"""
from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field


def unified_diff(path: str, before: str, after: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(before_lines, after_lines, fromfile=path, tofile=path)
    return "".join(diff)


@dataclass
class Approval:
    approval_id: str
    session_id: str
    tool: str
    args: dict
    diff: str
    status: str = "pending"          # pending | allowed | denied | escalated
    reason: str = ""
    created: float = field(default_factory=time.time)


@dataclass
class AuditLog:
    records: list[dict] = field(default_factory=list)

    def append(self, kind: str, **detail: object) -> None:
        self.records.append({"kind": kind, **detail})


class ApprovalGate:
    def __init__(self, audit: AuditLog, timeout_s: float = 3600.0):
        self.audit = audit
        self.timeout_s = timeout_s
        self._items: dict[str, Approval] = {}
        self._seq = 0

    def request(self, session_id: str, tool: str, args: dict, diff: str = "") -> Approval:
        self._seq += 1
        approval = Approval(f"apr-{self._seq}", session_id, tool, args, diff)
        self._items[approval.approval_id] = approval
        self.audit.append("approval_requested", approval_id=approval.approval_id,
                          tool=tool, session_id=session_id)
        return approval

    def decide(self, approval_id: str, allow: bool, reason: str, now: float | None = None) -> Approval:
        approval = self._items[approval_id]
        if approval.status != "pending":
            raise ValueError(f"{approval_id} is already {approval.status}")
        now = time.time() if now is None else now
        if now - approval.created > self.timeout_s:
            approval.status = "denied"
            approval.reason = "approval deadline passed; default deny"
        else:
            approval.status = "allowed" if allow else "denied"
            approval.reason = reason
        self.audit.append("approval_decided", approval_id=approval_id,
                          status=approval.status, reason=approval.reason)
        return approval

    def escalate(self, approval_id: str, reason: str) -> Approval:
        approval = self._items[approval_id]
        approval.status = "escalated"
        approval.reason = reason
        self.audit.append("approval_escalated", approval_id=approval_id, reason=reason)
        return approval

    def pending(self) -> list[Approval]:
        return [item for item in self._items.values() if item.status == "pending"]

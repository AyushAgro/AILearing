"""AgentRuntime: one step loop that every other harness module plugs into."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from forge.harness.approval import ApprovalGate, AuditLog
from forge.harness.budget import Ledger
from forge.harness.context import ContextManager, estimate_tokens
from forge.harness.lifecycle import Phase, Session, SessionStore
from forge.harness.model import ScriptedModel
from forge.harness.policy import Decision, Effect, PlanGate, PolicyEngine
from forge.harness.sandbox import run_subprocess
from forge.harness.tools import ToolRuntime
from forge.harness.tracing import Cassette, Tracer


Approver = Callable[[str, str, dict, str], bool]


@dataclass
class RunResult:
    status: str
    answer: str
    session: Session
    observations: list[str] = field(default_factory=list)


class AgentRuntime:
    def __init__(self, *, model: ScriptedModel, tools: ToolRuntime, policy: PolicyEngine,
                 ledger: Ledger, context: ContextManager, session: Session,
                 store: SessionStore, tracer: Tracer, gate: ApprovalGate,
                 audit: AuditLog, plan: PlanGate | None = None,
                 approver: Approver | None = None,
                 credentials: dict[str, str] | None = None,
                 cassette: Cassette | None = None):
        self.model = model
        self.tools = tools
        self.policy = policy
        self.ledger = ledger
        self.context = context
        self.session = session
        self.store = store
        self.tracer = tracer
        self.gate = gate
        self.audit = audit
        self.plan = plan
        self.approver = approver
        self.credentials = credentials or {}
        self.cassette = cassette
        self.pending: tuple[str, dict] | None = None

    def run(self, goal: str) -> RunResult:
        session = self.session
        if session.phase == Phase.CREATED:
            session.transition(Phase.RUNNING)
            self.context.add("system", "You are a harnessed agent. Follow policy.")
            self.context.add("user", goal)
        observations: list[str] = []
        answer = ""
        if self.pending and session.phase == Phase.RUNNING:
            tool, args = self.pending
            self.pending = None
            output = self._execute(tool, args)
            observations.append(output)
            self.context.add("tool", f"{tool}: {output}", kind="tool")
        while session.phase == Phase.RUNNING:
            session.next_step()
            prompt = self.context.render()
            with self.tracer.span(
                "chat scripted-model",
                **{
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "forge",
                    "gen_ai.request.model": "scripted-model",
                    "gen_ai.agent.name": self.tracer.agent_name,
                },
            ) as span:
                action = self.model.complete(prompt)
                in_tokens = estimate_tokens(prompt)
                out_tokens = estimate_tokens(json.dumps(action, sort_keys=True))
                span.attributes["gen_ai.usage.input_tokens"] = in_tokens
                span.attributes["gen_ai.usage.output_tokens"] = out_tokens
            reason = self.ledger.charge_model(in_tokens, out_tokens)
            self._persist("model", action=action)
            if reason:
                if "final" in action:
                    answer = str(action["final"])
                session.transition(Phase.FAILED, reason=reason)
                break
            if "final" in action:
                answer = str(action["final"])
                session.transition(Phase.COMPLETED)
                self.context.add("assistant", answer)
                break
            tool = str(action.get("tool", ""))
            args = dict(action.get("args") or {})
            observation, stop = self._act(tool, args)
            observations.append(observation)
            self.context.add("tool", f"{tool}: {observation}", kind="tool")
            if self.context.needs_compaction():
                self.context.compact(lambda body: f"summary of earlier tools: {body[:80]}")
            if stop:
                break
        self._flush()
        return RunResult(self._status(), answer, session, observations)

    def resume(self, allow: bool, reason: str = "") -> RunResult:
        """Continue a session that stopped in AWAITING_APPROVAL."""
        approval = self.gate.pending()
        if not approval:
            raise ValueError("no pending approval")
        self.gate.decide(approval[0].approval_id, allow, reason or "resume")
        if not allow:
            tool, _args = self.pending or ("tool", {})
            self.pending = None
            self.session.transition(Phase.RUNNING)
            self.context.add("tool", f"{tool}: denied by approver", kind="tool")
        else:
            self.session.transition(Phase.RUNNING)
        result = self.run(self.session.goal)
        result.observations = result.observations
        return result

    def _status(self) -> str:
        if self.session.phase == Phase.FAILED and self.ledger.stop_reason:
            return "exhausted"
        return self.session.phase.value

    def _act(self, tool: str, args: dict) -> tuple[str, bool]:
        decision = self._decide(tool, args)
        self.audit.append("policy", tool=tool, effect=decision.effect.value,
                          rule=decision.rule, reason=decision.reason)
        if decision.effect == Effect.DENY:
            return f"denied ({decision.reason})", False
        diff = str(args.get("diff", ""))
        if decision.effect == Effect.ASK:
            approval = self.gate.request(self.session.session_id, tool, args, diff)
            if self.approver is None:
                self.pending = (tool, args)
                self.session.transition(Phase.AWAITING_APPROVAL,
                                        approval_id=approval.approval_id)
                return f"paused for {approval.approval_id}", True
            allowed = self.approver(approval.approval_id, tool, args, diff)
            decided = self.gate.decide(approval.approval_id, allowed,
                                       "approver" if allowed else "approver rejected")
            if decided.status != "allowed":
                return f"denied by approver ({decided.reason})", False
        return self._execute(tool, args), False

    def _decide(self, tool: str, args: dict) -> Decision:
        if self.plan is not None:
            blocked = self.plan.check(tool)
            if blocked is not None:
                return blocked
        return self.policy.decide(tool, args)

    def _execute(self, tool: str, args: dict) -> str:
        spec = self.tools.specs.get(tool)
        credential = None
        if spec and spec.credential_scope:
            credential = self.credentials.get(spec.credential_scope)
        with self.tracer.span(
            f"execute_tool {tool}",
            **{
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool,
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.id": f"call-{self.session.step}",
            },
        ):
            if tool == "run_python":
                result = run_subprocess(str(args.get("code", "")))
                output = result.stdout.strip() if result.ok else result.stderr.strip()
                ok = result.ok
            else:
                result = self.tools.invoke(tool, args, credential=credential)
                output = result.output if result.ok else result.error
                ok = result.ok
        if self.cassette is not None and ok:
            self.cassette.record(tool, args, output)
        self._persist("tool_finished", tool=tool, ok=ok)
        return output

    def _persist(self, kind: str, **detail: object) -> None:
        record = self.session.event(kind, **detail)
        self.store.append(record)

    def _flush(self) -> None:
        record = self.session.event("phase_snapshot", phase=self.session.phase.value)
        self.store.append(record)

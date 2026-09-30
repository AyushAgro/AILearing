"""Tests for forge.harness. They do not call a model or the network."""
from __future__ import annotations

from pathlib import Path

import pytest

from forge.harness.approval import ApprovalGate, AuditLog
from forge.harness.bench import run_bench
from forge.harness.budget import Budget, Ledger
from forge.harness.context import ContextManager
from forge.harness.lifecycle import Phase, Session, SessionStore
from forge.harness.policy import Effect, PolicyEngine, Rule
from forge.harness.reliability import SideEffectStore, call_with_retry
from forge.harness.sandbox import container_argv, container_spec_ok, run_subprocess
from forge.harness.tools import ToolRuntime, ToolSpec


def test_illegal_transition_and_reload(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.jsonl")
    session = Session("s", "goal")
    session.transition(Phase.RUNNING)
    store.append(session.events[-1])
    store.append(session.event("marker"))
    session.transition(Phase.COMPLETED)
    store.append(session.events[-1])
    with pytest.raises(ValueError, match="illegal"):
        session.transition(Phase.RUNNING)
    assert [row["kind"] for row in store.load()] == ["phase", "marker", "phase"]


def test_compaction_keeps_system_prompt() -> None:
    ctx = ContextManager(max_tokens=80, tool_cap_tokens=20)
    ctx.add("system", "SYSTEM: keep me")
    ctx.add("user", "question")
    for index in range(5):
        ctx.add("tool", "x " * 100 + f"FACT-{index}", kind="tool")
    assert ctx.needs_compaction()
    assert ctx.compact(lambda _body: "summary only")
    ctx.assert_system_intact("SYSTEM:")
    assert "FACT-0" not in ctx.render()
    assert ctx.tokens() < ctx.max_tokens or ctx.compactions == 1


def test_default_deny_and_parallel_mutation() -> None:
    engine = PolicyEngine([Rule("search_*", Effect.ALLOW)])
    assert engine.decide("wire_money", {}).effect == Effect.DENY
    runtime = ToolRuntime()
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"]}
    runtime.register(ToolSpec("search_notes", lambda name: name, "read", schema))
    runtime.register(ToolSpec(
        "send_email", lambda to, body: to, "mail",
        {"type": "object", "properties": {"to": {"type": "string"},
                                          "body": {"type": "string"}},
         "required": ["to", "body"]},
        mutating=True,
    ))
    refused = runtime.invoke_parallel([
        ("search_notes", {"name": "a"}),
        ("send_email", {"to": "a@b.c", "body": "no"}),
    ])
    assert refused[0].ok is False
    assert "mutating" in refused[0].error


def test_idempotent_retry_does_not_double_charge() -> None:
    store = SideEffectStore()
    calls = {"n": 0}

    def charge() -> int:
        calls["n"] += 1
        return calls["n"]

    first, _ = store.run("pay", charge)
    second, replayed = store.run("pay", charge)
    assert first == second == 1
    assert replayed is True
    failed = call_with_retry(
        lambda: (_ for _ in ()).throw(RuntimeError("no")),
        key="once", store=store, attempts=5, idempotent=False,
    )
    assert failed.ok is False
    assert failed.attempts == 1


def test_approval_deadline_defaults_to_deny() -> None:
    gate = ApprovalGate(AuditLog(), timeout_s=5)
    approval = gate.request("s", "write_file", {}, "diff")
    decided = gate.decide(approval.approval_id, True, "please", now=approval.created + 30)
    assert decided.status == "denied"
    assert "default deny" in decided.reason


def test_budget_stops_before_the_next_step() -> None:
    ledger = Ledger(Budget(max_steps=2, max_tokens=10_000, max_usd=1))
    assert ledger.charge_model(10, 10) is None
    assert ledger.charge_model(10, 10) is None
    assert ledger.charge_model(10, 10) == "step budget exhausted"
    assert ledger.steps == 2


def test_subprocess_is_not_a_filesystem_sandbox() -> None:
    result = run_subprocess("print(6 * 7)")
    assert result.ok and result.stdout.strip() == "42"
    argv = container_argv("python:3.12-slim", "/work", "print(1)")
    assert container_spec_ok(argv) == []
    assert "--network=none" in argv


def test_bench_blocks_injection_and_hides_secrets() -> None:
    rows = {case.name: case for case in run_bench()}
    assert rows["benign"].utility == 1 and rows["benign"].attack == 0
    assert rows["injection_open_policy"].attack == 1
    assert rows["injection_least_privilege"].utility == 1
    assert rows["injection_least_privilege"].attack == 0
    assert rows["injection_plan_gate"].attack == 0
    assert rows["injection_plan_gate"].utility == 1
    secret = rows["credential_scope"]
    assert secret.secret_in_prompt == 0
    assert secret.secret_received_by_tool == 1

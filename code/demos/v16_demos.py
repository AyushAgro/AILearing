"""Chapter demos for Volume 16. Output in this file's run is pasted into the book."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from forge.harness.approval import ApprovalGate, AuditLog, unified_diff
from forge.harness.bench import run_bench
from forge.harness.budget import PRICE_INPUT_PER_TOKEN, Budget, Ledger, TokenBucket
from forge.harness.context import ContextManager, estimate_tokens
from forge.harness.lifecycle import Phase, Session, SessionStore
from forge.harness.mcp_client import McpClient
from forge.harness.model import ScriptedModel
from forge.harness.policy import CredentialStore, Effect, PlanGate, PolicyEngine, Rule
from forge.harness.reliability import SideEffectStore, call_with_retry
from forge.harness.runtime import AgentRuntime
from forge.harness.sandbox import container_argv, container_spec_ok, probe_filesystem, run_subprocess
from forge.harness.tools import ToolRuntime, ToolSpec
from forge.harness.tracing import Cassette, Tracer


def chapter(title: str) -> None:
    print(f"\n===== {title} =====")


def demo_lifecycle() -> None:
    chapter("lifecycle")
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root / "session.jsonl")
    session = Session("ses-1", "Summarise the refund policy")

    def persist() -> None:
        store.append(session.events[-1])

    session.transition(Phase.RUNNING)
    persist()
    session.next_step()
    store.append(session.event("tool_started", tool="search_notes", key="k1"))
    session.transition(Phase.PAUSED)
    persist()
    session.transition(Phase.RUNNING)
    persist()
    session.transition(Phase.COMPLETED)
    persist()
    print("phase", session.phase.value, "events", len(session.events))
    print("reloaded", [row["kind"] for row in store.load()])
    try:
        session.transition(Phase.RUNNING)
    except ValueError as exc:
        print("rejected:", exc)


def demo_context() -> None:
    chapter("context")
    ctx = ContextManager(max_tokens=150, tool_cap_tokens=24)
    ctx.add("system", "SYSTEM: never drop this rule.")
    ctx.add("user", "Summarise the corpus.")
    for index in range(6):
        blob = f"FACT-{index} " + ("token " * 80)
        ctx.add("tool", blob, kind="tool")
    print(f"before tokens={ctx.tokens()} truncations={ctx.truncations} "
          f"over_budget={ctx.needs_compaction()}")
    ctx.compact(lambda body: "summary: earlier searches ran; details dropped")
    ctx.assert_system_intact("SYSTEM:")
    print(f"after  tokens={ctx.tokens()} compactions={ctx.compactions}")
    print("system kept:", ctx.messages[0].content)
    rendered = ctx.render()
    print("FACT-0 still present:", "FACT-0" in rendered)
    print("FACT-5 still present:", "FACT-5" in rendered)
    print("summary present:", "summary:" in rendered)


def demo_tools_and_mcp() -> None:
    chapter("tools")
    runtime = ToolRuntime()

    def search_notes(name: str) -> str:
        return f"note {name}: 30 days"

    def stream_log(name: str):
        yield f"line {name}-a\n"
        yield f"line {name}-b\n"

    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"]}
    runtime.register(ToolSpec("search_notes", search_notes, "Read a note", schema))
    runtime.register(ToolSpec("stream_log", stream_log, "Stream a log", schema, streams=True))
    runtime.register(ToolSpec(
        "send_email", lambda to, body: f"sent {to}", "Send mail",
        {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "body"]},
        mutating=True, idempotent=False,
    ))
    bad = runtime.invoke("search_notes", {})
    print("missing arg:", bad.error)
    print("unknown:", runtime.invoke("drop_table", {}).error)
    streamed = runtime.invoke("stream_log", {"name": "refund"})
    print(f"stream chunks={streamed.streamed_chunks} output={streamed.output!r}")
    started = time.perf_counter()

    def slow(name: str) -> str:
        time.sleep(0.15)
        return name

    runtime.register(ToolSpec("slow", slow, "Slow read", schema))
    runtime.invoke_parallel([("slow", {"name": "a"}), ("slow", {"name": "b"})])
    parallel_s = time.perf_counter() - started
    started = time.perf_counter()
    runtime.invoke("slow", {"name": "a"})
    runtime.invoke("slow", {"name": "b"})
    serial_s = time.perf_counter() - started
    print(f"parallel={parallel_s:.3f}s serial={serial_s:.3f}s")
    refused = runtime.invoke_parallel([
        ("search_notes", {"name": "refund"}),
        ("send_email", {"to": "a@b.c", "body": "hi"}),
    ])
    print("mutating batch:", refused[0].error)

    chapter("mcp")
    server = Path(__file__).resolve().parents[1] / "src" / "forge" / "harness" / "mcp_server.py"
    client = McpClient(server)
    try:
        init = client.request("initialize")
        listed = client.request("tools/list")
        called = client.request("tools/call", {"name": "lookup_note",
                                               "arguments": {"name": "refund"}})
    finally:
        client.close()
    print("server:", init["result"]["serverInfo"]["name"],
          "protocol:", init["result"]["protocolVersion"])
    print("tools:", [tool["name"] for tool in listed["result"]["tools"]])
    print("call:", called["result"]["content"][0]["text"])


def demo_policy() -> None:
    chapter("policy")
    engine = PolicyEngine([
        Rule("search_*", Effect.ALLOW, reason="read-only"),
        Rule("send_email", Effect.ALLOW,
             when=lambda args: str(args.get("to", "")).endswith("@example.com"),
             reason="allowlisted recipient"),
        Rule("send_email", Effect.DENY, reason="recipient not allowlisted"),
        Rule("run_python", Effect.ASK, reason="code execution needs a person"),
    ])
    samples = [
        ("search_notes", {"name": "refund"}),
        ("send_email", {"to": "ada@example.com"}),
        ("send_email", {"to": "attacker@evil.test"}),
        ("run_python", {"code": "print(1)"}),
        ("wire_money", {"amount": 10}),
    ]
    for tool, args in samples:
        decision = engine.decide(tool, args)
        print(f"{tool:12} {decision.effect.value:5}  {decision.reason}")
    store = CredentialStore({"mail": "sk-live-SHOULD-NOT-LEAK"})
    print("issued to tool runtime:", store.issue("mail") is not None)
    print("unknown scope:", store.issue("payments"))
    gate = PlanGate(["search_notes"])
    gate.freeze()
    print("plan blocks send_email:", gate.check("send_email").reason)


def demo_budget() -> None:
    chapter("budget")
    ledger = Ledger(Budget(max_tokens=500, max_usd=1.0, max_seconds=30, max_steps=4))
    for step in range(1, 7):
        reason = ledger.charge_model(80, 20)
        if reason:
            print(f"step {step}: {reason}")
            break
        print(f"step {step}: charged tokens={ledger.tokens} usd={ledger.usd:.8f}")
    print("price input $/token", PRICE_INPUT_PER_TOKEN)
    bucket = TokenBucket(rate_per_sec=1.0, capacity=2.0)
    bucket.updated = 0.0
    bucket.tokens = 2.0
    burst = [bucket.allow(now=0.0) for _ in range(3)]
    print("burst", burst, "after 1s", bucket.allow(now=1.0))


def demo_reliability() -> None:
    chapter("reliability")
    store = SideEffectStore()
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("upstream reset")
        return "ok"

    result = call_with_retry(flaky, key="read-1", store=store, attempts=4, idempotent=True)
    print(f"read retries={result.attempts} value={result.value} calls={calls['n']}")
    again = call_with_retry(flaky, key="read-1", store=store, attempts=4, idempotent=True)
    print(f"replayed={again.replayed} calls_still={calls['n']}")

    charges = {"n": 0}

    def charge() -> dict:
        charges["n"] += 1
        return {"charge_id": charges["n"], "amount": 10}

    first, replayed = store.run("pay-9", charge)
    print("crashed after effect, before tool_finished event")
    second, replayed = store.run("pay-9", charge)
    print(f"resume charge_id={second['charge_id']} replayed={replayed} "
          f"charges={charges['n']}")
    unsafe = call_with_retry(charge, key="pay-new", store=store, attempts=3, idempotent=False)
    print(f"non-idempotent attempts={unsafe.attempts} value={unsafe.value}")


def demo_sandbox() -> None:
    chapter("sandbox")
    ok = run_subprocess("print(2 + 2)")
    print(f"ok={ok.ok} stdout={ok.stdout.strip()!r} level={ok.level}")
    hung = run_subprocess("import time\ntime.sleep(10)", timeout=0.4)
    print(f"sleep: ok={hung.ok} stderr={hung.stderr.strip()!r}")
    home = os.path.expanduser("~")
    probe = probe_filesystem(home)
    print("subprocess probe:", probe)
    argv = container_argv("python:3.12-slim", "/tmp/work", "print(2 + 2)")
    print("container spec missing flags:", container_spec_ok(argv) or "none")
    print("container argv head:", " ".join(argv[:8]))


def demo_trace_and_approval() -> None:
    chapter("trace-approval")
    before = "def add(a, b):\n    return a - b\n"
    after = "def add(a, b):\n    return a + b\n"
    print(unified_diff("maths.py", before, after).rstrip())
    audit = AuditLog()
    gate = ApprovalGate(audit, timeout_s=10)
    approval = gate.request("ses-9", "write_file", {"path": "maths.py"}, "diff")
    decided = gate.decide(approval.approval_id, False, "diff changes behaviour",
                          now=approval.created + 1)
    print("decision", decided.status, decided.reason)
    late = gate.request("ses-9", "write_file", {"path": "maths.py"}, "diff")
    expired = gate.decide(late.approval_id, True, "too late", now=late.created + 99)
    print("expired", expired.status, expired.reason)
    print("audit", [row["kind"] + ":" + row.get("status", "requested") for row in audit.records])


def _runtime(root: Path, actions: list[dict], *, policy: PolicyEngine,
             budget: Budget, tools: ToolRuntime, name: str,
             approver=None, plan: list[str] | None = None) -> AgentRuntime:
    plan_gate = PlanGate(plan) if plan else None
    if plan_gate:
        plan_gate.freeze()
    return AgentRuntime(
        model=ScriptedModel(actions), tools=tools, policy=policy,
        ledger=Ledger(budget), context=ContextManager(max_tokens=4_000),
        session=Session(name, "goal"), store=SessionStore(root / f"{name}.jsonl"),
        tracer=Tracer(agent_name=name), gate=ApprovalGate(AuditLog()),
        audit=AuditLog(), plan=plan_gate, approver=approver,
        cassette=Cassette(),
    )


def demo_runtime() -> None:
    chapter("runtime")
    root = Path(tempfile.mkdtemp())
    tools = ToolRuntime()
    tools.register(ToolSpec(
        "search_notes", lambda name: "Refunds within 30 days.",
        "Read a note",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    ))
    tools.register(ToolSpec(
        "write_file", lambda path, content: f"wrote {path} ({len(content)} chars)",
        "Write a file",
        {"type": "object", "properties": {"path": {"type": "string"},
                                          "content": {"type": "string"}},
         "required": ["path", "content"]},
        mutating=True, idempotent=False,
    ))
    policy = PolicyEngine([
        Rule("search_*", Effect.ALLOW, reason="read"),
        Rule("write_file", Effect.ASK, reason="mutation"),
        Rule("run_python", Effect.ALLOW, reason="sandboxed code"),
    ])
    runtime = _runtime(
        root,
        [
            {"tool": "search_notes", "args": {"name": "refund"}},
            {"tool": "write_file", "args": {"path": "report.md", "content": "30 days"}},
            {"tool": "run_python", "args": {"code": "print(2 + 2)"}},
            {"final": "Report written. 2 + 2 = 4."},
        ],
        policy=policy, budget=Budget(max_steps=8, max_tokens=50_000),
        tools=tools, name="research",
    )
    paused = runtime.run("Write a short refund note and check arithmetic.")
    print("paused", paused.status, paused.observations)
    finished = runtime.resume(True, reason="diff is a new report, no overwrite of secrets")
    print("resumed", finished.status, finished.answer)
    print("observations", finished.observations)
    names = [span["name"] for span in runtime.tracer.export()]
    print("spans", names)
    replay = runtime.cassette.replay("search_notes", {"name": "refund"})
    print("replay", replay)

    runaway = _runtime(
        root,
        [{"tool": "search_notes", "args": {"name": "refund"}}] * 6,
        policy=policy, budget=Budget(max_steps=3, max_tokens=50_000),
        tools=tools, name="runaway",
    )
    stopped = runaway.run("Keep searching.")
    print("runaway", stopped.status, "steps", runaway.ledger.steps,
          "stop", runaway.ledger.stop_reason, "tool_runs", len(stopped.observations))


def demo_bench() -> None:
    chapter("bench")
    print(f"{'case':28} utility attack status secret_in_prompt tool_got_secret")
    for case in run_bench():
        print(f"{case.name:28} {case.utility:7} {case.attack:6} {case.status:12} "
              f"{case.secret_in_prompt:16} {case.secret_received_by_tool:14}")


def main() -> None:
    demo_lifecycle()
    demo_context()
    demo_tools_and_mcp()
    demo_policy()
    demo_budget()
    demo_reliability()
    demo_sandbox()
    demo_trace_and_approval()
    demo_runtime()
    demo_bench()


if __name__ == "__main__":
    main()

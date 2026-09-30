"""Graph nodes: thin, idempotent wrappers around the typed agents.

Each node:
  1. honours STEP_DELAY_SEC (test hook for the kill-and-resume chaos test)
  2. checks the idempotency cache (completed node + same inputs => cached
     result, zero side effects)
  3. validates inputs/outputs with Pydantic on every handoff
  4. appends a TraceStep and updates token/search usage
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..agents.citations import render_markdown, validate_citations
from ..agents.supervisor import SupervisorAgent, SupervisorView
from ..agents.workers import PlannerAgent, ResearcherAgent, WriterAgent
from ..config import Settings
from ..providers.base import LLMProvider, SearchProvider
from ..providers.llm_openai import LLMProviderError
from ..providers.tokens import TokenCounter
from ..state.schemas import (
    ErrorCode,
    Finding,
    PlanRequest,
    ResearcherResult,
    ResearchPlan,
    ResearchReport,
    ResearchRequest,
    RoutingDecision,
    RoutingTarget,
    RunBudgets,
    RunError,
    RunStatus,
    SubQuestion,
    SubQuestionStatus,
    TokenUsage,
    TraceStep,
    WriteRequest,
)
from ..storage.registry import RunRegistry
from .state import GraphState


@dataclass
class NodeDeps:
    settings: Settings
    llm: LLMProvider
    search: SearchProvider
    registry: RunRegistry
    counter: TokenCounter
    planner: PlannerAgent
    researcher: ResearcherAgent
    writer: WriterAgent
    supervisor: SupervisorAgent


def make_deps(settings: Settings, registry: RunRegistry,
              llm: LLMProvider | None = None,
              search: SearchProvider | None = None) -> NodeDeps:
    from ..providers.factory import build_llm, build_search
    llm = llm or build_llm(settings)
    search = search or build_search(settings)
    counter = TokenCounter()
    return NodeDeps(
        settings=settings, llm=llm, search=search, registry=registry,
        counter=counter,
        planner=PlannerAgent(llm),
        researcher=ResearcherAgent(llm, search, counter),
        writer=WriterAgent(llm),
        supervisor=SupervisorAgent(),
    )


# ------------------------------------------------------------------ helpers
def _fingerprint(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _elapsed_sec(state: GraphState) -> float:
    started = state.get("started_at")
    if not started:
        return 0.0
    dt = datetime.fromisoformat(started)
    return (datetime.now(timezone.utc) - dt).total_seconds()


def _usage(state: GraphState) -> TokenUsage:
    return TokenUsage.model_validate(state.get("usage") or {})


def _budgets(state: GraphState) -> RunBudgets:
    return RunBudgets.model_validate(state["budgets"])


def _trace(state: GraphState, deps: NodeDeps, *, node: str, agent: str,
           input_summary: str, output_summary: str,
           tools: list[str] | None = None, latency_ms: int = 0,
           tokens_in: int = 0, tokens_out: int = 0, cached: bool = False,
           error: RunError | None = None,
           routing: RoutingDecision | None = None) -> dict:
    step = TraceStep(
        seq=len(state.get("trace") or []) + 1,
        run_id=state["run_id"], node=node, agent=agent,
        input_summary=input_summary[:500], output_summary=output_summary[:800],
        tools_used=tools or [], latency_ms=latency_ms,
        tokens_in=tokens_in, tokens_out=tokens_out, cached=cached,
        error=error, routing_decision=routing,
    )
    return {"trace": [step.model_dump(mode="json")]}


def _fail(state: GraphState, deps: NodeDeps, node: str, code: ErrorCode,
          message: str, routing: RoutingDecision | None = None,
          **detail) -> dict:
    err = RunError(code=code, message=message, node=node, detail=detail)
    upd = {"status": RunStatus.FAILED.value, "error": err.model_dump(mode="json")}
    upd.update(_trace(state, deps, node=node, agent="system",
                      input_summary="", output_summary=f"FAILED: {message}",
                      error=err, routing=routing))
    return upd


def _maybe_delay(deps: NodeDeps) -> None:
    if deps.settings.step_delay_sec > 0:
        time.sleep(deps.settings.step_delay_sec)


# ---------------------------------------------------------------- plan node
def plan_node(state: GraphState, deps: NodeDeps) -> dict:
    _maybe_delay(deps)
    run_id = state["run_id"]
    fp = _fingerprint(["plan", state["question"]])
    if (cached := deps.registry.idem_get(f"{run_id}:plan:{fp}")) is not None:
        plan = ResearchPlan.model_validate_json(cached)
        upd = {"plan": plan.model_dump(mode="json"),
               "status": RunStatus.RUNNING.value}
        upd.update(_trace(state, deps, node="plan", agent="planner",
                          input_summary=state["question"],
                          output_summary=f"{len(plan.subquestions)} sub-questions (cached)",
                          tools=["llm.complete"], cached=True))
        return upd
    start = time.monotonic()
    try:
        plan, resp = deps.planner.plan(PlanRequest(
            question=state["question"],
            max_subquestions=_budgets(state).max_subquestions))
    except (LLMProviderError, ValueError) as e:
        code = ErrorCode.PROVIDER_ERROR if isinstance(e, LLMProviderError) else ErrorCode.PLAN_FAILED
        return _fail(state, deps, "plan", code, str(e))
    if not plan.subquestions:
        return _fail(state, deps, "plan", ErrorCode.PLAN_FAILED,
                      "planner returned zero sub-questions")
    deps.registry.idem_set(f"{run_id}:plan:{fp}", plan.model_dump_json())
    usage = _usage(state)
    usage.input_tokens += resp.tokens_in
    usage.output_tokens += resp.tokens_out
    upd = {"plan": plan.model_dump(mode="json"),
           "status": RunStatus.RUNNING.value,
           "usage": usage.model_dump()}
    upd.update(_trace(
        state, deps, node="plan", agent="planner",
        input_summary=state["question"],
        output_summary=f"{len(plan.subquestions)} sub-questions: " +
                       "; ".join(s.question[:60] for s in plan.subquestions),
        tools=["llm.complete"], latency_ms=int((time.monotonic() - start) * 1000),
        tokens_in=resp.tokens_in, tokens_out=resp.tokens_out))
    return upd


# ----------------------------------------------------------- supervise node
def supervise_node(state: GraphState, deps: NodeDeps) -> dict:
    _maybe_delay(deps)
    start = time.monotonic()
    plan = ResearchPlan.model_validate(state["plan"])
    results = [ResearcherResult.model_validate(r)
               for r in (state.get("researcher_results") or [])]
    # NOTE: supervisor mutates sub-question statuses on its local copy of the
    # plan; the updated plan is written back so the change is durable.
    view = SupervisorView(
        plan=plan,
        researcher_results=results,
        findings_count=len(state.get("findings") or []),
        usage=_usage(state),
        budgets=_budgets(state),
        revision_count=state.get("revision_count", 0),
        elapsed_sec=_elapsed_sec(state),
        failed=state.get("status") == RunStatus.FAILED.value,
    )
    decision = deps.supervisor.decide(view)
    upd: dict = {"routing": decision.model_dump(mode="json"),
                 "plan": plan.model_dump(mode="json")}
    agent = "supervisor"
    if decision.next == RoutingTarget.FAIL:
        upd.update(_fail(state, deps, "supervise",
                         decision.error_code or ErrorCode.INTERNAL_ERROR,
                         decision.reason, routing=decision))
        upd["routing"] = decision.model_dump(mode="json")
        return upd
    if decision.next == RoutingTarget.RESEARCH:
        sq = next(s for s in plan.subquestions if s.id == decision.subquestion_id)
        sq.status = SubQuestionStatus.IN_PROGRESS
        sq.attempts += 1
        if any(r.subquestion_id == sq.id and r.status.value == "insufficient"
               for r in results):
            upd["revision_count"] = view.revision_count + 1
        upd["plan"] = plan.model_dump(mode="json")
        upd["active_subquestion"] = sq.model_dump(mode="json")
    trace_upd = _trace(state, deps, node="supervise", agent=agent,
                       input_summary=(f"{len(results)} results, "
                                      f"{view.findings_count} findings, "
                                      f"{view.usage.total_tokens} tokens"),
                       output_summary=f"-> {decision.next.value}: {decision.reason}",
                       latency_ms=int((time.monotonic() - start) * 1000),
                       routing=decision)
    upd.update(trace_upd)
    return upd


# ------------------------------------------------------------ research node
def research_node(state: GraphState, deps: NodeDeps) -> dict:
    _maybe_delay(deps)
    run_id = state["run_id"]
    sq = SubQuestion.model_validate(state["active_subquestion"])
    results = [ResearcherResult.model_validate(r)
               for r in (state.get("researcher_results") or [])]
    refinement = ""
    if sq.attempts > 1:
        refinement = "alternative angles and sources"
    fp = _fingerprint(["research", sq.id, str(sq.attempts), refinement])
    if (cached := deps.registry.idem_get(f"{run_id}:research:{fp}")) is not None:
        res = ResearcherResult.model_validate_json(cached)
        # Idempotent replay: never append a result or findings already present
        # (e.g. supervisor re-dispatch after a validate retry, or a resumed
        # stream that merged a snapshot into the checkpoint state).
        have_result = any(r.subquestion_id == sq.id for r in results)
        have_fids = {f.get("finding_id") for f in (state.get("findings") or [])}
        fresh = [f for f in res.findings if f.finding_id not in have_fids]
        upd = {"researcher_results": ([] if have_result
                                      else [res.model_dump(mode="json")]),
               "findings": [f.model_dump(mode="json") for f in fresh],
               "active_subquestion": None}
        upd.update(_trace(state, deps, node="research", agent="researcher",
                          input_summary=sq.question,
                          output_summary=f"{len(fresh)} new findings (cached)",
                          tools=["search", "llm.complete"], cached=True))
        return upd

    budgets = _budgets(state)
    usage = _usage(state)
    searches_left = budgets.max_searches - usage.searches
    start = time.monotonic()
    try:
        res = deps.researcher.research(
            ResearchRequest(subquestion=sq, attempt=sq.attempts,
                            refinement_hint=refinement),
            searches_left=searches_left,
            finding_id_offset=len(state.get("findings") or []),
        )
    except LLMProviderError as e:
        return _fail(state, deps, "research", ErrorCode.PROVIDER_ERROR, str(e),
                     subquestion_id=sq.id)
    deps.registry.idem_set(f"{run_id}:research:{fp}", res.model_dump_json())
    usage.input_tokens += res.tokens_in
    usage.output_tokens += res.tokens_out
    usage.searches += res.searches_used
    upd = {"researcher_results": [res.model_dump(mode="json")],
           "findings": [f.model_dump(mode="json") for f in res.findings],
           "active_subquestion": None,
           "usage": usage.model_dump()}
    upd.update(_trace(
        state, deps, node="research", agent="researcher",
        input_summary=f"{sq.id} (attempt {sq.attempts}): {sq.question}",
        output_summary=f"{res.status.value}: {len(res.findings)} findings, "
                       f"{res.searches_used} searches",
        tools=["search", "llm.complete"],
        latency_ms=int((time.monotonic() - start) * 1000),
        tokens_in=res.tokens_in, tokens_out=res.tokens_out,
        error=res.error))
    return upd


# --------------------------------------------------------------- write node
def write_node(state: GraphState, deps: NodeDeps) -> dict:
    _maybe_delay(deps)
    run_id = state["run_id"]
    findings = [Finding.model_validate(f) for f in (state.get("findings") or [])]
    plan = ResearchPlan.model_validate(state["plan"])
    feedback = state.get("validation_feedback") or ""
    fp = _fingerprint(["write"] + sorted(f.finding_id for f in findings) + [feedback])
    if (cached := deps.registry.idem_get(f"{run_id}:write:{fp}")) is not None:
        report = ResearchReport.model_validate_json(cached)
        upd = {"report": report.model_dump(mode="json")}
        upd.update(_trace(state, deps, node="write", agent="writer",
                          input_summary=f"{len(findings)} findings",
                          output_summary=f"{len(report.sections)} sections (cached)",
                          tools=["llm.complete"], cached=True))
        return upd
    start = time.monotonic()
    try:
        report, resp = deps.writer.write(WriteRequest(
            question=state["question"], plan=plan, findings=findings,
            validation_feedback=feedback))
    except (LLMProviderError, ValueError) as e:
        code = ErrorCode.PROVIDER_ERROR if isinstance(e, LLMProviderError) else ErrorCode.INTERNAL_ERROR
        return _fail(state, deps, "write", code, str(e))
    deps.registry.idem_set(f"{run_id}:write:{fp}", report.model_dump_json())
    usage = _usage(state)
    usage.input_tokens += resp.tokens_in
    usage.output_tokens += resp.tokens_out
    upd = {"report": report.model_dump(mode="json"),
           "writer_attempts": state.get("writer_attempts", 0) + 1,
           "usage": usage.model_dump()}
    upd.update(_trace(
        state, deps, node="write", agent="writer",
        input_summary=f"{len(findings)} findings" + (" + validation feedback" if feedback else ""),
        output_summary=f"{len(report.sections)} sections, {len(report.citations)} citations",
        tools=["llm.complete"], latency_ms=int((time.monotonic() - start) * 1000),
        tokens_in=resp.tokens_in, tokens_out=resp.tokens_out))
    return upd


# ------------------------------------------------------------ validate node
def validate_node(state: GraphState, deps: NodeDeps) -> dict:
    _maybe_delay(deps)
    start = time.monotonic()
    report = ResearchReport.model_validate(state["report"])
    findings = [Finding.model_validate(f) for f in (state.get("findings") or [])]
    problems = validate_citations(report, findings)
    attempts = state.get("writer_attempts", 0)
    if not problems:
        md = render_markdown(report)
        upd = {"status": RunStatus.COMPLETED.value, "report_markdown": md}
        upd.update(_trace(state, deps, node="validate", agent="system",
                          input_summary=f"{len(report.citations)} citations",
                          output_summary="all citations resolve; report complete",
                          latency_ms=int((time.monotonic() - start) * 1000)))
        return upd
    if attempts < 2:
        feedback = "Citation validation failed: " + "; ".join(problems)
        upd = {"report": None, "validation_feedback": feedback}
        upd.update(_trace(state, deps, node="validate", agent="system",
                          input_summary=f"{len(report.citations)} citations",
                          output_summary=f"invalid, retrying writer: {feedback[:200]}",
                          latency_ms=int((time.monotonic() - start) * 1000)))
        return upd
    return _fail(state, deps, "validate", ErrorCode.CITATION_INVALID,
                 "; ".join(problems))

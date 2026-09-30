"""Orchestration helpers shared by the API and the worker."""

from __future__ import annotations

import json

from ..config import Settings, get_settings
from ..graph.builder import build_graph
from ..graph.nodes import make_deps
from ..graph.state import GraphState
from ..state.schemas import (
    Progress,
    ResearchStatusResponse,
    RunBudgets,
    RunStatus,
    TokenUsage,
    TraceResponse,
    TraceStep,
    utcnow,
)
from ..storage.checkpoint import RedisCheckpointer
from ..storage.registry import (
    FileRunRegistry,
    RedisRunRegistry,
    RunRegistry,
)


def build_initial_state(run_id: str, question: str,
                        budgets: RunBudgets) -> dict:
    state: GraphState = {
        "run_id": run_id,
        "question": question,
        "status": RunStatus.QUEUED.value,
        "plan": None,
        "active_subquestion": None,
        "researcher_results": [],
        "findings": [],
        "report": None,
        "report_markdown": "",
        "validation_feedback": "",
        "writer_attempts": 0,
        "trace": [],
        "usage": TokenUsage().model_dump(),
        "budgets": budgets.model_dump(),
        "revision_count": 0,
        "error": None,
        "started_at": utcnow(),
        "routing": None,
    }
    return dict(state)


def snapshot_to_status(raw: str, settings: Settings) -> ResearchStatusResponse:
    snap = json.loads(raw)
    usage = TokenUsage.model_validate(snap.get("usage") or {})
    plan = snap.get("plan") or {}
    subqs = plan.get("subquestions", [])
    answered = sum(1 for s in subqs if s.get("status") == "answered")
    error = snap.get("error")
    return ResearchStatusResponse(
        run_id=snap["run_id"],
        status=RunStatus(snap.get("status") or "QUEUED"),
        question=snap.get("question", ""),
        progress=Progress(
            subquestions_total=len(subqs),
            subquestions_answered=answered,
            searches_used=usage.searches,
            tokens_used=usage.total_tokens,
        ),
        report_markdown=snap.get("report_markdown") or None,
        error=error,
        usage=usage,
        estimated_cost_usd=round(usage.cost_usd(
            settings.price_input_per_1m_usd,
            settings.price_output_per_1m_usd), 6),
        created_at=snap.get("started_at", ""),
        updated_at=snap.get("started_at", ""),
    )


def trace_to_response(run_id: str, raw_steps: list[str]) -> TraceResponse:
    return TraceResponse(
        run_id=run_id,
        steps=[TraceStep.model_validate_json(s) for s in raw_steps],
    )


def make_registry(settings: Settings, tmp_dir: str | None = None) -> RunRegistry:
    """Production: Redis. tmp_dir set: file-backed (offline tests only)."""
    if tmp_dir:
        return FileRunRegistry(tmp_dir)
    import redis
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    return RedisRunRegistry(client)


def drive_run(graph, registry: RunRegistry, run_id: str,
              initial: dict | None) -> dict:
    """Drive the graph to completion; persist a snapshot after every node.

    initial=None resumes from the Redis checkpoint (thread_id == run_id).
    Returns the final graph state.
    """
    config = {"configurable": {"thread_id": run_id}}
    stream_input = initial  # None => resume
    for _chunk in graph.stream(stream_input, config):
        snap = graph.get_state(config).values
        registry.save_snapshot(run_id, json.dumps(dict(snap)))
    final = dict(graph.get_state(config).values)
    registry.save_snapshot(run_id, json.dumps(final))
    return final


def build_worker_graph(settings: Settings | None = None,
                       registry: RunRegistry | None = None,
                       llm=None, search=None,
                       checkpointer=None):
    """Assemble graph + deps for the worker (injectable for tests)."""
    settings = settings or get_settings()
    registry = registry or make_registry(settings)
    deps = make_deps(settings, registry, llm=llm, search=search)
    if checkpointer is None:
        import redis
        client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        checkpointer = RedisCheckpointer(client)
    graph = build_graph(deps, checkpointer)
    return graph, deps, registry
